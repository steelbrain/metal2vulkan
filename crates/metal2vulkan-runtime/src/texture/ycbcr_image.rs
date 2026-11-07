//! The image of a two-plane YCbCr texture over a surface's two planes (`ycbcr.rs`): an image of its
//! sampled and storage usage, a shadow whose copy in converts the planes
//! (`ShadowSource::Converted`). A texture read converted is an `R16G16B16A16_SFLOAT` image of the
//! converted texels, which the conversion writes; an `_sRGB` twin's an `R32G32B32A32_SFLOAT` one
//! (its steps of 1/8192 are no halves above 1/4); one read as codes an `A2B10G10R10_UNORM_PACK32`
//! image of 10-bit codes or an `R8G8B8A8_UNORM` one of 8-bit codes, which the conversion writes
//! packed through an `R32_UINT` view (a store of its bits, where a store of `code / (2^b - 1)`
//! through the image's own format would be the device's conversion to the code).
//!
//! **Planes read in place** (`PlanesInPlace`). On a device that reads host memory with no bus
//! between (`ProfileFields::ycbcr_in_place`: an integrated GPU, a CPU device), a texture of
//! `YCBCR8_420_2P` or `YCBCR10_420_2P` over plane 0 of a two-plane surface, without a swizzle,
//! also states its planes as a function reads them in place (`bind::YCbCrInPlace`): a slot whose
//! function reads the texture only so binds a uniform texel buffer over the planes' bytes, and the
//! draw or dispatch reads those bytes alone, so the image is not converted for it. Where a plane
//! cannot be viewed so, the texture keeps its image alone, counted by the reason
//! (`WorkCounts::ycbcr_images_kept_*`): the device, the layout, a swizzle, the planes' first byte
//! off `minTexelBufferOffsetAlignment`, more elements than `maxTexelBufferElements`, planes past
//! their buffer, or a format the device views no uniform texel buffer of. Where the device
//! converts the planes' 4:2:0 format as a linear image with Metal's reconstruction, the planes
//! are also an image of it over their bytes (`ycbcr_planes.rs`), which a slot whose every sample
//! the device's sampler takes samples through the device's conversion instead of the texel
//! buffer.

use super::ycbcr_planes::PlanesImage;
use super::{image_error, IdUse, Image, Storage, Texture};
use crate::address_map::AddressModeMap;
use crate::binding_view::VkView;
use crate::buffer_texture::{view_usage, Texels};
use crate::device::lock;
use crate::error::{internal, Error};
use crate::image_plan::plan_image;
use crate::resource::ResourceKey;
use crate::shadow::{ShadowBuffer, ShadowCopies, ShadowSource};
use crate::shadow_merge::ScratchSpares;
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::{Access, Use, UseRange};
use crate::values::ResourceOptions;
use crate::view_plan::ViewRecord;
use crate::ycbcr::{YCbCrConversion, YCbCrWriteBack};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{YCbCrCodes, YCbCrInPlace, YCbCrOutput, YCbCrPlanes};
use metal2vulkan::device::{FormatUsage, VkFeature};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{
    vk_format, ycbcr_format, ycbcr_renders, ChromaSubsampling, YCbCrReads,
};
use metal2vulkan::metal::TextureSwizzle as S;
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, TextureSwizzle, TextureSwizzleChannels, TextureType,
    TextureUsage,
};
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags,
    ImageUsageFlags, ImageViewType,
};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The formats of a texture's image and of the view the conversion writes it through, by what
/// its texels read: the converted texels as halves, every channel a multiple of 1/2048 up to 1,
/// which a half holds exactly; the sRGB twins' as floats, every channel a multiple of 1/8192,
/// which a half does not hold above 1/4; 10-bit codes as `RGB10A2Unorm` texels, alpha 3, and
/// 8-bit codes as `RGBA8Unorm` ones, alpha 255, written as their packed words.
fn image_formats(codes: YCbCrCodes) -> [Format; 2] {
    match (codes.output, codes.bits) {
        (YCbCrOutput::Codes, 8) => [Format::R8G8B8A8_UNORM, Format::R32_UINT],
        (YCbCrOutput::Codes, _) => [Format::A2B10G10R10_UNORM_PACK32, Format::R32_UINT],
        (YCbCrOutput::Srgb, _) => [Format::R32G32B32A32_SFLOAT; 2],
        _ => [Format::R16G16B16A16_SFLOAT; 2],
    }
}

/// The modes Metal samples a two-plane YCbCr texture with, by its codes: a 4:2:0 or 4:2:2 texture
/// samples repeat, mirrored repeat and mirror-clamp-to-edge as clamp-to-edge, a 4:4:4 one as
/// themselves; one read as codes samples clamp-to-zero and clamp-to-border-color as
/// clamp-to-edge too, under every border color, where the others read the border (its alpha 1,
/// as `alpha_reads_one` reads every texel's).
fn ycbcr_modes(codes: YCbCrCodes) -> AddressModeMap {
    let edge = SamplerAddressMode::ClampToEdge;
    let wrapped = match codes.subsampling {
        ChromaSubsampling::Chroma444 => AddressModeMap::IDENTITY,
        _ => AddressModeMap::IDENTITY
            .with(SamplerAddressMode::Repeat, edge)
            .with(SamplerAddressMode::MirrorRepeat, edge)
            .with(SamplerAddressMode::MirrorClampToEdge, edge),
    };
    match codes.output {
        YCbCrOutput::Codes => wrapped
            .with(SamplerAddressMode::ClampToZero, edge)
            .with(SamplerAddressMode::ClampToBorderColor, edge),
        _ => wrapped,
    }
}

/// Whether `planes` read each code alone where `pf` reads its codes converted or through the
/// sRGB curve: a texture over a surface of a P3 matrix (`ycbcr.rs`), whose image holds the codes.
fn codes_alone(pf: PixelFormat, planes: &YCbCrPlanes) -> bool {
    planes.codes.output == YCbCrOutput::Codes
        && ycbcr_format(pf).is_some_and(|f| f.reads != YCbCrReads::Codes)
}

/// The code-reading format of `codes`' bits and subsampling, unpacked (`RGB8_420_2P` to
/// `RGB10_444_2P`), whose image holds the codes.
fn code_format(codes: YCbCrCodes) -> Option<PixelFormat> {
    PixelFormat::ALL.iter().copied().find(|&pf| {
        ycbcr_format(pf).is_some_and(|f| {
            f.reads == YCbCrReads::Codes
                && !f.packed
                && f.bits == codes.bits
                && f.subsampling == codes.subsampling
        })
    })
}

/// Whether each row of `planes` ends before the next starts, so that a write of one row's codes
/// changes no other's (a format's rows of wider codes than the surface's may overlap).
fn rows_apart(planes: &YCbCrPlanes) -> bool {
    let codes = planes.codes;
    let [w, _] = planes.size;
    let [cw, _] = planes.chroma_size();
    let row = |n: u32| codes.units(u64::from(n)) * u64::from(codes.bytes);
    row(w) <= u64::from(planes.luma[1]) && row(2 * cw) <= u64::from(planes.chroma[1])
}

/// `swizzle` over a read whose red, green and blue are `read`'s channels, alpha one.
fn over_read(read: [TextureSwizzle; 3], swizzle: TextureSwizzleChannels) -> TextureSwizzleChannels {
    let of = |channel: TextureSwizzle| match channel {
        S::Red => read[0],
        S::Green => read[1],
        S::Blue => read[2],
        S::Alpha => S::One,
        other => other,
    };
    TextureSwizzleChannels {
        red: of(swizzle.red),
        green: of(swizzle.green),
        blue: of(swizzle.blue),
        alpha: of(swizzle.alpha),
    }
}

/// A two-plane YCbCr texture's planes as a function reads them in place (the module docs): their
/// words, their first byte in the texels' buffer, and their image the device's conversion samples
/// where it has one (`ycbcr_planes.rs`).
pub(crate) struct PlanesInPlace {
    pub(super) words: YCbCrInPlace,
    start: u64,
    pub(super) image: Option<Arc<PlanesImage>>,
}

/// Why a two-plane YCbCr texture keeps its image alone, its planes read in place by no function
/// (the module docs), in `WorkCounts::ycbcr_images_kept_*`'s order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeptImage {
    /// The device's type: a discrete, virtual or other GPU, or none stated.
    Device,
    /// A layout the form does not read: another format, plane 1, a P3 surface's codes.
    Layout,
    /// The texture's swizzle, which the form does not apply.
    Swizzle,
    /// The planes' first byte is off `minTexelBufferOffsetAlignment` in the buffer.
    Alignment,
    /// The planes hold more elements than `maxTexelBufferElements`.
    Elements,
    /// The planes reach past their buffer, whose bytes past its end Metal reads as zeros.
    Reach,
    /// The device views no uniform texel buffer of the planes' format in a buffer of the runtime's
    /// usage (`buffer_texture::view_usage`).
    View,
}

impl KeptImage {
    /// The reasons, in their counters' order.
    pub(crate) const ALL: [KeptImage; 7] = [
        KeptImage::Device,
        KeptImage::Layout,
        KeptImage::Swizzle,
        KeptImage::Alignment,
        KeptImage::Elements,
        KeptImage::Reach,
        KeptImage::View,
    ];
}

impl Device {
    /// The planes in place of a converted texture of `descriptor` over `texels`, whose planes
    /// `planes` lie in `span` of the texels' buffer (the module docs), or why it keeps its image
    /// alone.
    fn planes_in_place(
        &self,
        texels: &Texels,
        descriptor: &TextureDescriptor,
        (planes, span): (&YCbCrPlanes, &Range<u64>),
    ) -> Result<PlanesInPlace, KeptImage> {
        let profile = self.compile_device().profile();
        if !profile.reads_ycbcr_planes_in_place() {
            return Err(KeptImage::Device);
        }
        let words = YCbCrInPlace::of(planes).ok_or(KeptImage::Layout)?;
        if texels.staged().is_some() {
            return Err(KeptImage::Layout);
        }
        if descriptor.swizzle != TextureSwizzleChannels::DEFAULT {
            return Err(KeptImage::Swizzle);
        }
        if span.start.saturating_add(words.view_bytes()) > span.end {
            return Err(KeptImage::Reach);
        }
        let limits = profile.limits();
        let (_, at) = texels.buffer.at(span.start);
        if !at.is_multiple_of(limits.min_texel_buffer_offset_alignment.max(1)) {
            return Err(KeptImage::Alignment);
        }
        if words.elements() > u64::from(limits.max_texel_buffer_elements) {
            return Err(KeptImage::Elements);
        }
        let maintenance5 = self.enabled().features.contains(&VkFeature::Maintenance5);
        view_usage(
            |usage| profile.supports_format(words.format(), usage),
            false,
            maintenance5,
        )
        .map_err(|_| KeptImage::View)?;
        Ok(PlanesInPlace {
            words,
            start: span.start,
            image: None,
        })
    }

    /// The two-plane YCbCr texture of `descriptor` (checked as `layout`) over `texels`, the luma
    /// plane, for Metal's `call`, whose planes `planes` lie in `span` of the texels' buffer: its
    /// image of the texels Metal reads and the conversion (`ycbcr.rs`). It reads alpha 1 and maps
    /// the address modes Metal samples otherwise (`ycbcr_modes`). Refused `NotImplemented` where
    /// the device holds no such image, and as `new_texture` refuses the image or the view the
    /// conversion writes it through.
    pub(crate) fn new_converted_texture(
        &self,
        call: &'static str,
        texels: Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
        (planes, span): (&YCbCrPlanes, Range<u64>),
    ) -> Result<Texture, Error> {
        let descriptor = TextureDescriptor {
            alpha_reads_one: true,
            address_mode_map: ycbcr_modes(planes.codes),
            ..*descriptor
        };
        let texels = Arc::new(texels);
        let mut record = ViewRecord::of_descriptor(layout, &descriptor);
        if !codes_alone(layout.pixel_format, planes) {
            let mut in_place = self.planes_in_place(&texels, &descriptor, (planes, &span));
            let counts = &self.shared().counts;
            match &in_place {
                Ok(_) => counts.ycbcr_texture_in_place(),
                Err(kept) => counts.ycbcr_image_kept(*kept),
            }
            // Its planes as an image the device's conversion samples, where it has one.
            if let Ok(p) = &mut in_place {
                match self.planes_image(&texels, (planes, &p.words), p.start)? {
                    Ok(image) => p.image = Some(image),
                    Err(kept) => counts.ycbcr_planes_image_kept(kept),
                }
            }
            let storage =
                self.converted_storage(call, texels, &descriptor, layout, (planes, span))?;
            let made = (storage, in_place.ok());
            return Ok(Texture::made_with_planes(self, made, record, None, id_use));
        }
        // Over a surface of a P3 matrix: an image of the codes of the code-reading format, its
        // luma, Cb and Cr (green, blue and red there) read as red, green and blue.
        let code_format = code_format(planes.codes)
            .ok_or_else(|| internal("a P3 texture of codes no format reads"))?;
        let coded = TextureDescriptor {
            pixel_format: code_format,
            swizzle: TextureSwizzleChannels::DEFAULT,
            ..descriptor
        };
        let coded_layout = coded
            .check_over_surface(self.properties().texture_rules())
            .map_err(|_| internal("a P3 texture's descriptor past its own"))?;
        let storage =
            self.converted_storage(call, texels, &coded, &coded_layout, (planes, span))?;
        record.read_as = Some((
            code_format,
            over_read([S::Green, S::Blue, S::Red], descriptor.swizzle),
        ));
        Ok(Texture::made(self, storage, record, None, id_use))
    }

    /// The storage of a two-plane YCbCr texture of `descriptor` (checked as `layout`) over
    /// `texels`, for Metal's `call`, whose planes `planes` lie in `span` of the texels' buffer:
    /// its image and the conversion's shadow (`new_converted_texture`, and a view as the
    /// texture's `_sRGB` twin, `Texture::converted_view`).
    fn converted_storage(
        &self,
        call: &'static str,
        texels: Arc<Texels>,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        (planes, span): (&YCbCrPlanes, Range<u64>),
    ) -> Result<Storage, Error> {
        let pf = layout.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let plan = plan_image(
            layout,
            descriptor.usage | TextureUsage::SHADER_READ,
            false,
            &facts,
        )
        .map_err(|refusal| image_error(call, refusal))?;
        let [format, written] = image_formats(planes.codes);
        let sampled = facts
            .optimal(format)
            .contains(FormatFeatureFlags2::SAMPLED_IMAGE);
        let storage = |f| {
            facts
                .optimal(f)
                .contains(FormatFeatureFlags2::STORAGE_IMAGE)
        };
        if plan.format != format || !sampled || !storage(written) {
            return Err(gap());
        }
        let mut plan = crate::image_plan::ImagePlan {
            usage: plan.usage | ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
            ..plan
        };
        // Written through a view of another format: mutable, viewed as that format too (an image
        // already viewed as any format keeps its empty list), and of extended usage where the
        // image's own format has no storage.
        if written != format {
            let listed = !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
            plan.flags |= ImageCreateFlags::MUTABLE_FORMAT;
            if !storage(format) {
                plan.flags |= ImageCreateFlags::EXTENDED_USAGE;
            }
            if listed || !plan.view_formats.is_empty() {
                if plan.view_formats.is_empty() {
                    plan.view_formats.push(format);
                }
                plan.view_formats.push(written);
            }
        }
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = self.new_image(plan, layout, descriptor, hazard)?;
        // The conversion's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: written,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let conversion = YCbCrConversion::new(self, (planes, span.clone()), (raw, at), view)?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::Converted {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                conversion: Box::new(conversion),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            // Metal drops a function's writes to the texture.
            drops_writes: true,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            stood: AtomicU64::new(0),
            standing: Default::default(),
        };
        Ok(Storage::Shadow {
            image: image.shared(),
            texels,
            copies: Arc::new(copies),
        })
    }
}

impl Texture {
    /// The uniform texel buffer view of the texture's planes read in place (the module docs), made
    /// at its first bind and kept with the texels. Internal for a texture whose planes are not.
    pub(crate) fn planes_view(&self) -> Result<vk::BufferView, Error> {
        let (Some(p), Some(texels)) = (&self.0.planes_in_place, self.buffer_texels()) else {
            return Err(internal(
                "a planes view of a texture without planes in place",
            ));
        };
        let (_, at) = texels.buffer.at(p.start);
        texels.view_of(p.words.format(), false, (at, p.words.view_bytes()))
    }

    /// The view of the texture's planes as an image the device's conversion samples
    /// (`ycbcr_planes.rs`), which a slot that samples them through it binds. Internal for a
    /// texture whose planes take no such image.
    pub(crate) fn planes_image_view(&self) -> Result<vk::ImageView, Error> {
        match self
            .0
            .planes_in_place
            .as_ref()
            .and_then(|p| p.image.as_ref())
        {
            Some(image) => Ok(image.view),
            None => Err(internal(
                "a planes image view of a texture whose planes take no image",
            )),
        }
    }

    /// A read by `read` of the bytes of the texture's planes read in place, which a binding of
    /// them makes, through their texel buffer or their image (the module docs). Internal for a
    /// texture whose planes are not.
    pub(crate) fn planes_use(&self, read: Access) -> Result<Use<ResourceKey>, Error> {
        let (Some(p), Some(texels)) = (&self.0.planes_in_place, self.buffer_texels()) else {
            return Err(internal(
                "a planes use of a texture without planes in place",
            ));
        };
        let end = p.start.saturating_add(p.words.view_bytes());
        let (identity, range) = texels.buffer.tracked(p.start..end);
        Ok(Use {
            key: identity.key,
            tracked: texels.tracked(),
            range: UseRange::Bytes(range),
            read,
            write: Access::NONE,
        })
    }

    /// The storage of a view presenting `record` of this texture in its `_sRGB` twin's format
    /// (`view_plan::check_format` admits `YCBCR8_420_2P` as `YCBCR8_420_2P_sRGB` and
    /// `YCBCR10_420_2P` as `YCBCR10_420_2P_sRGB`, as Metal makes them, its usage `ShaderRead`
    /// alone): a converted image of the view's format over the same planes, which reads what
    /// Metal's view reads, a texture of that format over the surface's reads bit for bit. `None`
    /// for a texture whose texels are no conversion of planes. Refused as
    /// a texture of the view's format over the planes is.
    pub(super) fn converted_view(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Option<Result<Storage, Error>> {
        let Storage::Shadow { texels, copies, .. } = self.storage() else {
            return None;
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return None;
        };
        let pf = record.pixel_format;
        let (planes, span) = conversion.source();
        // A twin's view of a texture over a surface of a P3 matrix reads no rule found.
        if codes_alone(self.0.record.pixel_format, &planes) {
            return Some(Err(Error::Unsupported(Unsupported::not_implemented(
                Feature::TextureView,
            ))));
        }
        let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) else {
            return Some(Err(internal(
                "a converted view of a format no conversion reads",
            )));
        };
        let planes = YCbCrPlanes::new(
            planes.size,
            planes.luma,
            planes.chroma,
            (planes.range, planes.matrix),
            codes,
        );
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = record.usage;
        descriptor.swizzle = record.swizzle;
        descriptor.resource_options = texels.options();
        descriptor.alpha_reads_one = true;
        descriptor.address_mode_map = ycbcr_modes(codes);
        let device = self.device();
        Some(
            descriptor
                .check_over_surface(device.properties().texture_rules())
                .map_err(|_| internal("a converted view's descriptor past its texture's"))
                .and_then(|layout| {
                    device.converted_storage(
                        call,
                        Arc::clone(texels),
                        &descriptor,
                        &layout,
                        (&planes, span),
                    )
                }),
        )
    }
}

impl Texture {
    /// The storage of a view presenting `record` of this two-plane YCbCr texture in its own
    /// format as another type (a 2D array, which `view_plan::check_view` admits), and the format
    /// and swizzle its bindings read that storage through (`ViewRecord::read_as`). `None` for a
    /// texture whose texels are no conversion of planes, and a view in another format or of the
    /// texture's own type.
    ///
    /// Metal's view reads the luma plane's codes alone (our probe `ycv`, the M3 Max under macOS
    /// 27.0.1, `read()` and a linear `sample()` of every texel, bit for bit): a texture read
    /// converted (and a packed one) reads `(Y, 0, 0, 1)`, one read as codes `(0, Y, 0, 1)`, `Y` the
    /// code over `2^b - 1`, and its linear sample is the luma channel of the code-reading
    /// format's sample over the same planes; a swizzle applies to that read. So the view is an
    /// image of the codes over the same planes (the code-reading format of the texture's bits and
    /// subsampling, `RGB8_420_2P` to `RGB10_444_2P`; a packed texture's codes held unpacked) read
    /// through a swizzle that keeps its luma. An `_sRGB` twin's view reads no defined value on
    /// Metal (its texels change from run to run) and reads as its converted format's here.
    /// Refused (`NotImplemented(TextureView)`) for a texture over plane 1 of its surface, which
    /// is not measured.
    pub(super) fn luma_view(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Option<Result<(Storage, (PixelFormat, TextureSwizzleChannels)), Error>> {
        let Storage::Shadow { texels, copies, .. } = self.storage() else {
            return None;
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return None;
        };
        let own = &self.0.record;
        if record.pixel_format != own.pixel_format || record.texture_type == own.texture_type {
            return None;
        }
        let gap = || Error::Unsupported(Unsupported::not_implemented(Feature::TextureView));
        let (planes, span) = conversion.source();
        if conversion.reads_zeros() || codes_alone(own.pixel_format, &planes) {
            return Some(Err(gap()));
        }
        let read = planes.codes;
        // The code-reading format of the texture's bits and subsampling, unpacked: its image
        // holds the codes as the view reads them.
        let Some(code_format) = code_format(read) else {
            return Some(Err(gap()));
        };
        let mut codes = read;
        codes.output = YCbCrOutput::Codes;
        let planes = YCbCrPlanes::new(
            planes.size,
            planes.luma,
            planes.chroma,
            (planes.range, planes.matrix),
            codes,
        );
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(code_format, w, h, false);
        descriptor.usage = record.usage;
        descriptor.resource_options = texels.options();
        descriptor.alpha_reads_one = true;
        descriptor.address_mode_map = ycbcr_modes(codes);
        // The luma, green of the codes, where the texture reads it: red read converted, green
        // read as codes; then the view's swizzle over that read.
        let luma = match read.output {
            YCbCrOutput::Codes => [S::Zero, S::Green, S::Zero],
            _ => [S::Green, S::Zero, S::Zero],
        };
        let swizzle = over_read(luma, record.swizzle);
        let device = self.device();
        Some(
            descriptor
                .check_over_surface(device.properties().texture_rules())
                .map_err(|_| internal("a luma view's descriptor past its texture's"))
                .and_then(|layout| {
                    device.converted_storage(
                        call,
                        Arc::clone(texels),
                        &descriptor,
                        &layout,
                        (&planes, span),
                    )
                })
                .map(|storage| (storage, (code_format, swizzle))),
        )
    }
}

impl Texture {
    /// Whether this two-plane YCbCr texture reads each code alone, over a surface of a P3 matrix
    /// (`ycbcr.rs`): a pass that loads it loads nothing measured.
    pub(crate) fn ycbcr_codes_alone(&self) -> bool {
        let Storage::Shadow { copies, .. } = self.storage() else {
            return false;
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return false;
        };
        codes_alone(self.0.record.pixel_format, &conversion.source().0)
    }

    /// The write shadow of a 4:4:4 two-plane YCbCr texture whose writes Metal stores into its
    /// planes (`metal::format::ycbcr_writes_stored`), for Metal's `call`: an `RGBA32Uint` image of
    /// the texture's size, which a function writes each texel to as its red, green and blue bits
    /// and the mark 1 (`StorageImageClass::YCbCrWrite`), and its copies, a clear in and the
    /// encoding of the marked texels' codes back into the planes (`ShadowSource::YCbCrWrite`,
    /// `ycbcr.rs`). Refused (`Unsupported`, `NotImplemented`, `Feature::Format`, naming the
    /// format) for a texture over plane 1 of its surface, whose writes are not measured, and one
    /// whose texels are no conversion of planes; as `new_texture` refuses the image.
    pub(super) fn ycbcr_write_shadow(
        &self,
        call: &'static str,
    ) -> Result<(Image, ShadowCopies), Error> {
        let pf = self.0.record.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let Storage::Shadow { texels, copies, .. } = self.storage() else {
            return Err(gap());
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return Err(gap());
        };
        let (planes, span) = conversion.source();
        // Over a surface of a P3 matrix the writes are not measured.
        if conversion.reads_zeros() || !rows_apart(&planes) || codes_alone(pf, &planes) {
            return Err(gap());
        }
        let device = self.device();
        let [width, height] = planes.size;
        let layout = TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA32Uint,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let facts = device
            .shared()
            .textures
            .facts(device.compile_device().profile());
        let plan = plan_image(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // Written by the function, read by the encoding, and cleared.
        let needed = ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST;
        if plan.format != Format::R32G32B32A32_UINT || !plan.usage.contains(needed) {
            return Err(gap());
        }
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, w, h, false);
        descriptor.usage = usage;
        // The device's alone: cleared at every copy in, never read by the CPU.
        descriptor.resource_options = ResourceOptions::private();
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = device.new_image(plan, &layout, &descriptor, hazard)?;
        // The write-back's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: Format::R32G32B32A32_UINT,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let write_back =
            YCbCrWriteBack::new(device, (&planes, span.clone()), at, (view, false), None)?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::YCbCrWrite {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                image: image.raw.get(),
                write_back: Box::new(write_back),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            stood: AtomicU64::new(0),
            standing: Default::default(),
        };
        Ok((image, copies))
    }
}

impl Texture {
    /// The render shadow of a two-plane YCbCr texture a pass stores as Metal does
    /// (`metal::format::ycbcr_renders`), for Metal's `call`, made at its first need and kept
    /// with the texture: a texture of the texture's format and size over its planes whose image
    /// is `R32G32B32A32_SFLOAT`, the color attachment a pass renders the values it leaves in
    /// each texel into (`FormatEmulation::YCbCrRendered`), and its copies: in, a clear, or for a
    /// pass that loads the attachment the planes converted into floats; back, the store of each
    /// stored texel's codes into the planes (`ShadowSource::YCbCrWrite` of a store, `ycbcr.rs`),
    /// which leaves the texture's converted texels stale. Refused
    /// (`Unsupported`, `NotImplemented`, `Feature::Format`, naming the format) for a texture
    /// over plane 1 of its surface, one of a size its chroma blocks do not tile, and one whose
    /// texels are no conversion of planes, none of whose stores is measured; as `new_texture`
    /// refuses the image.
    pub(crate) fn ycbcr_render_shadow(&self, call: &'static str) -> Result<Texture, Error> {
        let mut held = lock(&self.0.render_shadow);
        if let Some(shadow) = held.as_ref() {
            return Ok(shadow.clone());
        }
        let shadow = self.new_render_shadow(call)?;
        *held = Some(shadow.clone());
        Ok(shadow)
    }

    /// A new render shadow ([`ycbcr_render_shadow`](Self::ycbcr_render_shadow)).
    fn new_render_shadow(&self, call: &'static str) -> Result<Texture, Error> {
        let pf = self.0.record.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let Storage::Shadow { texels, copies, .. } = self.storage() else {
            return Err(gap());
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return Err(gap());
        };
        let (planes, span) = conversion.source();
        if !ycbcr_renders(pf) || conversion.reads_zeros() || !rows_apart(&planes) {
            return Err(gap());
        }
        // Over a surface of a P3 matrix the pass stores by BT.601 as the format's codes (the
        // planes' matrix), where the texture reads the codes alone; it loads nothing measured.
        let loads = !codes_alone(pf, &planes);
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).ok_or_else(gap)?;
        let planes = YCbCrPlanes::new(
            planes.size,
            planes.luma,
            planes.chroma,
            (planes.range, planes.matrix),
            codes,
        );
        let [width, height] = planes.size;
        let [across, down] = planes.codes.subsampling.halved();
        if (across && width % 2 != 0) || (down && height % 2 != 0) {
            return Err(gap());
        }
        let device = self.device();
        let profile = device.compile_device().profile();
        let float = TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA32Float,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE;
        let facts = device.shared().textures.facts(profile);
        let plan = plan_image(&float, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // Rendered, read by the store through a view of its bits, cleared, and written by the
        // load's conversion.
        let needed = ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::STORAGE
            | ImageUsageFlags::TRANSFER_DST;
        let mapping =
            vk_format(pf, &[FormatUsage::ColorAttachment].into(), profile).map_err(|_| gap())?;
        let bits = Format::R32G32B32A32_UINT;
        let storage = |f| {
            facts
                .optimal(f)
                .contains(FormatFeatureFlags2::STORAGE_IMAGE)
        };
        if plan.format != Format::R32G32B32A32_SFLOAT
            || mapping.format != plan.format
            || !plan.usage.contains(needed)
            || !storage(bits)
            || !storage(plan.format)
        {
            return Err(gap());
        }
        // The texture's format, held in floats; viewed as their bits too.
        let plan = crate::image_plan::ImagePlan {
            pixel_format: pf,
            flags: plan.flags | ImageCreateFlags::MUTABLE_FORMAT,
            view_formats: vec![plan.format, bits],
            mapping,
            ..plan
        };
        let layout = TextureLayout {
            pixel_format: pf,
            ..float
        };
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        // The device's alone: cleared at every copy in, never read by the CPU.
        descriptor.resource_options = ResourceOptions::private();
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = device.new_image(plan, &layout, &descriptor, hazard)?;
        // The write-back's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: bits,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        // The load's (`ycbcr.rs`), which converts the planes into the floats.
        let floats = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: Format::R32G32B32A32_SFLOAT,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let write_back = YCbCrWriteBack::new(
            device,
            (&planes, span.clone()),
            at,
            (view, true),
            loads.then_some(floats),
        )?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::YCbCrWrite {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                image: image.raw.get(),
                write_back: Box::new(write_back),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            stood: AtomicU64::new(0),
            standing: Default::default(),
        };
        let storage = Storage::Shadow {
            image: image.shared(),
            texels: Arc::clone(texels),
            copies: Arc::new(copies),
        };
        let record = ViewRecord::of_descriptor(&layout, &descriptor);
        Ok(Texture::made(device, storage, record, None, IdUse::Runtime))
    }
}
