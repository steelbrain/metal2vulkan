//! Metal texture views (`newTextureViewWithPixelFormat:` and its longer forms) and the Vulkan image
//! views the runtime makes of a texture.
//!
//! A Metal texture, or a view of one, presents part of an image: a [`ViewRecord`] of its pixel
//! format, texture type, levels, slices, swizzle and usage. A view costs Metal nothing, and the
//! runtime makes no Vulkan object for one: it makes the `VkImageView`s a binding needs when the
//! binding is written (`binding_view.rs`), from the record and the binding plan's view.
//!
//! Metal's rules for a view ([`check_view`]), measured on an Apple M3 Max and M4:
//!
//! - **Formats.** A view takes the texture's own format, its sRGB twin, and any format of the same
//!   class: an uncompressed color format of the same bytes per pixel (`RGBA8Unorm` as `R32Uint`,
//!   `BGRA8Unorm`, `RGB10A2Unorm`, `RG11B10Float`, `RGB9E5Float` or `RG16Snorm`, `B5G6R5Unorm` as
//!   `R16Uint`, `R8Unorm` as `A8Unorm`), another member of its compressed family (`BC4_RUnorm` as
//!   `BC4_RSnorm`), or the stencil of a combined depth/stencil format (`Depth32Float_Stencil8` as
//!   `X32_Stencil8`). Metal's API validation stops a view of the class other than the sRGB twin
//!   unless the texture has `PixelFormatView` usage; plain Metal makes it and renders through it
//!   the same (Metal probe cases `view.*.plain.*`, the M3 Max and the M4), so the runtime makes it
//!   too: where the texture's image was not made for the view's format, the view holds its texels
//!   apart (`texture/view_shadow.rs`). Metal stops every other pair whatever the usage: another
//!   size, compressed and uncompressed, two compressed families (`BC1_RGBA` as `BC4_RUnorm`), depth
//!   or stencil and color (`Depth16Unorm` and `R16Unorm`), a combined format as its depth. An
//!   extended-range format of one plane (`BGRA10_XR`, `BGR10_XR` and their sRGB twins) views only
//!   as itself and its twin, and no other format views as one, though their bytes per pixel match
//!   (the consumer's report, an M3 Max under macOS 26.7: plain Metal aborts each of the 116 other
//!   pairs among the 32- and 64-bit color formats, "source texture pixelFormat not compatible with
//!   texture view pixelFormat"), as a 4:2:2 format does; and so does one of two planes
//!   (`RGB10A8_2P_XR10` and its sRGB twin: an M4 under macOS 27.0.1, plain Metal
//!   aborts each of its 72 pairs with 18 color formats of 1 to 16 bytes, the one-plane ones among
//!   them, in both orders). Metal views a two-plane texture as its sRGB twin, which the runtime
//!   refuses `Unsupported(Format)`: those reads are not measured. A converted two-plane YCbCr
//!   texture (`YCBCR8_420_2P`, `YCBCR10_420_2P`) views as its `_sRGB` twin, which converts the same
//!   planes (`texture/ycbcr_image.rs`); the other order is refused `Unsupported(Format)`,
//!   unmeasured.
//! - **Types.** One-dimensional types view as each other, and so do the 2D types with the cube
//!   types, the multisample types, and 3D alone. A view of a type that is not an array has one
//!   slice, a cube six, a cube array a multiple of six. Metal makes a cube view of an image that
//!   is not square, which Vulkan cannot (`NotRepresentable(TextureView)`). A two-plane YCbCr
//!   texture's view of another type (a 2D array of its slice) in its own format reads its luma
//!   codes alone, as Metal's does (`Texture::luma_view`); one in its `_sRGB` twin's format is
//!   refused `NotImplemented(TextureView)`, unmeasured.
//! - **Ranges.** Levels and slices count from the viewed texture's own (a view of a view is
//!   relative to that view), are not empty, and stay in it.
//! - **Swizzles.** A view made without a swizzle has the viewed texture's swizzle; one made with a
//!   swizzle has that swizzle, which replaces the texture's rather than composing with it. Metal
//!   applies a swizzle to the channels the view's format gives, border texels included.
//! - **Usage.** A view has the viewed texture's usage, except that a view made with a swizzle other
//!   than the default keeps only `ShaderRead` and `PixelFormatView` of it (a texture of `Unknown`
//!   usage has Metal's `effective_usage`). Metal's validation stops a write binding of such a view
//!   of a texture with `ShaderWrite`; plain Metal reads it swizzled and writes through it with the
//!   swizzle undone (Metal probe cases `rw.view.read` and `rw.view.write`, the M3 Max and the M4).
//!   A Vulkan storage view takes no swizzle, so at a storage slot the function applies it to its
//!   reads and undoes it on its writes, and the view takes none (`binding_view`), as it takes none
//!   at a storage slot the function only queries, whose answers no swizzle reaches; a texture made
//!   resident for a function reading handles stays the runtime's gap
//!   (`NotImplemented(TextureSwizzle)`). A texture's usage keeps the bits Metal's texture drops
//!   (`TextureUsage::DROPPED`), and so do its views; Metal asserts on a swizzled view of a texture
//!   whose usage is none (one made with 0x8 alone), plain too
//!   (`ViewProblem::SwizzledViewOfNoUsage`).
//!
//! **Reinterpretation.** A device whose `VK_KHR_portability_subset` withholds
//! `imageViewFormatReinterpretation` makes no Vulkan view whose format has other components, or
//! components of other widths, than its image's
//! (`VUID-VkImageViewCreateInfo-imageViewFormatReinterpretation-04466`), though Metal views
//! `RGBA8Unorm` as `R32Uint`. The runtime enables the feature where the subset has it (MoltenVK),
//! and an adopted device has it where its consumer says it enabled it
//! (`RawDevice::view_format_reinterpretation`). Elsewhere such a view holds its texels apart
//! (`texture/view_shadow.rs`): an image of its own, a shadow of the viewed image's texels the
//! encoders copy, or a texture over the same bytes for a texture over a buffer. A view the shadow
//! cannot serve (of a sparse texture or a heap's texture, or one whose copies would not keep every
//! bit) is refused, typed (`Unsupported`, `Feature::ViewFormatReinterpretation`), and any Vulkan
//! view of an image is checked again ([`keeps_components`]).
//!
//! The Vulkan image views of a record are `binding_view.rs`'s.

use super::address_map::AddressModeMap;
use super::chroma_pairs::is_four_two_two;
use super::copy_plan::block_extent;
use super::image_plan::{block_family, srgb_twin};
use super::texture_descriptor::{
    texture_usage, TextureDescriptor, TextureLayout, TextureSwizzleChannels,
};
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{Error as CompileError, Feature, Unsupported};
use metal2vulkan::metal::format::{byte_format, vk_format, ycbcr, ycbcr_format, ExtendedRange};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureType, TextureUsage};
use metal2vulkan::vk::Format;
use std::collections::BTreeSet;
use std::ops::Range;

/// What a Metal texture, or a view of one, presents of its image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ViewRecord {
    pub(super) pixel_format: PixelFormat,
    pub(super) texture_type: TextureType,
    /// The image's level-0 extent.
    pub(super) image_extent: [u32; 3],
    /// The image's levels this presents, from `base_level`.
    pub(super) base_level: u32,
    pub(super) level_count: u32,
    /// The image's layers (Metal's slices) this presents, from `base_layer`.
    pub(super) base_layer: u32,
    pub(super) layer_count: u32,
    pub(super) samples: u32,
    pub(super) swizzle: TextureSwizzleChannels,
    pub(super) usage: TextureUsage,
    /// The texture's alpha reads 1 in its shaders' reads (`TextureDescriptor::alpha_reads_one`):
    /// its views' too.
    pub(super) alpha_reads_one: bool,
    /// The texture's address-mode map (`TextureDescriptor::address_mode_map`): its views' too.
    pub(super) address_mode_map: AddressModeMap,
    /// The texture's `framebufferOnly`, `rotation` and `isDrawable`
    /// (`TextureDescriptor::framebuffer_only`, `rotation`, `is_drawable`): its views' too, as
    /// plain Metal's views keep them.
    pub(super) presented: Presented,
    /// A view that reads its image in the image's own format (`texture/view_shadow.rs`): that
    /// format, and the swizzle over its channels through which it reads what the view reads
    /// (`metal::format::swizzle_over`). Its bindings, and the work that reads its texels' meaning
    /// (`generate_mipmaps`), take [`bound`](Self::bound). `None` for every other texture.
    pub(super) read_as: Option<(PixelFormat, TextureSwizzleChannels)>,
}

/// A texture's SPI presentation properties (`ViewRecord::presented`): none for every texture
/// but one over an IOSurface plane made with them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Presented {
    pub(super) framebuffer_only: bool,
    /// 0 to 4 (`texture_descriptor::MAX_ROTATION`).
    pub(super) rotation: u8,
    pub(super) is_drawable: bool,
}

impl ViewRecord {
    /// A texture of `layout` made with `usage` and `swizzle`: all of its image, with the usage
    /// Metal gives the texture, its alpha its own and its address modes the identity's.
    pub(super) fn of_texture(
        layout: &TextureLayout,
        usage: TextureUsage,
        swizzle: TextureSwizzleChannels,
    ) -> ViewRecord {
        ViewRecord {
            pixel_format: layout.pixel_format,
            texture_type: layout.texture_type,
            image_extent: layout.extent,
            base_level: 0,
            level_count: layout.levels,
            base_layer: 0,
            layer_count: layout.layers,
            samples: layout.samples,
            swizzle,
            usage: texture_usage(usage, swizzle),
            alpha_reads_one: false,
            address_mode_map: AddressModeMap::IDENTITY,
            presented: Presented::default(),
            read_as: None,
        }
    }

    /// The texture `descriptor` makes, of `layout`: [`of_texture`](Self::of_texture) of its usage
    /// and swizzle, with its alpha, its address-mode map and its presentation properties (a
    /// rotation past `u8`, which `TextureDescriptor::check` refuses, is none).
    pub(super) fn of_descriptor(
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
    ) -> ViewRecord {
        ViewRecord {
            alpha_reads_one: descriptor.alpha_reads_one,
            address_mode_map: descriptor.address_mode_map,
            presented: Presented {
                framebuffer_only: descriptor.framebuffer_only,
                rotation: u8::try_from(descriptor.rotation).unwrap_or(0),
                is_drawable: descriptor.is_drawable,
            },
            ..ViewRecord::of_texture(layout, descriptor.usage, descriptor.swizzle)
        }
    }

    /// What the texture's bindings present: the record itself, or, for a view that reads its
    /// image in the image's own format (`read_as`), that format through its swizzle, as a texture
    /// of the format made with the swizzle reads.
    pub(super) fn bound(&self) -> ViewRecord {
        match self.read_as {
            Some((pixel_format, swizzle)) => ViewRecord {
                pixel_format,
                swizzle,
                read_as: None,
                ..*self
            },
            None => *self,
        }
    }

    /// The swizzle the texture's reads, samples and gathers apply: its swizzle, each channel whose
    /// source is alpha reading 1 where its alpha reads 1. Writes, render targets and copies keep
    /// the plain image.
    pub(super) fn sampled_swizzle(&self) -> TextureSwizzleChannels {
        if !self.alpha_reads_one {
            return self.swizzle;
        }
        let one = |source: TextureSwizzle| match source {
            TextureSwizzle::Alpha => TextureSwizzle::One,
            other => other,
        };
        let s = self.swizzle;
        TextureSwizzleChannels::new(one(s.red), one(s.green), one(s.blue), one(s.alpha))
    }

    /// The extent of the first level this presents (Metal's `width`, `height` and `depth`).
    pub(super) fn extent(&self) -> [u32; 3] {
        self.image_extent
            .map(|axis| (axis >> self.base_level.min(31)).max(1))
    }

    /// Metal's `arrayLength`: the slices, or the cubes of a cube type.
    pub(super) fn array_length(&self) -> u32 {
        match self.texture_type {
            TextureType::Cube | TextureType::CubeArray => self.layer_count / 6,
            _ => self.layer_count,
        }
    }
}

/// A view Metal is asked for: `newTextureViewWithPixelFormat:` and its longer forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ViewRequest {
    pub(super) pixel_format: PixelFormat,
    pub(super) texture_type: TextureType,
    /// Levels and slices of the viewed texture, counted from its own first.
    pub(super) levels: Range<u32>,
    pub(super) slices: Range<u32>,
    /// `None` keeps the viewed texture's swizzle.
    pub(super) swizzle: Option<TextureSwizzleChannels>,
}

impl ViewRequest {
    /// `newTextureViewWithPixelFormat:`: the viewed texture's type, levels and slices.
    pub(super) fn of_format(source: &ViewRecord, pixel_format: PixelFormat) -> ViewRequest {
        ViewRequest {
            pixel_format,
            texture_type: source.texture_type,
            levels: 0..source.level_count,
            slices: 0..source.layer_count,
            swizzle: None,
        }
    }
}

/// A texture view, or a texture's use, that Metal's API validation stops, or plain Metal answers
/// nil for ([`NotAFormat`](Self::NotAFormat), an [`Error::Nil`](crate::Error::Nil)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewProblem {
    /// [`PixelFormat::Invalid`] or [`PixelFormat::Unspecialized`]: plain Metal makes no such view
    /// and answers nil (an M3 Max under macOS 26.7, the consumer's report), an
    /// [`Error::Nil`](crate::Error::Nil).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 26.7: plain Metal makes no such view and answers
    /// nil, as the consumer reported).
    NotAFormat,
    /// A format Metal never views the texture's as: another size or class (compressed,
    /// uncompressed, depth or stencil), another compressed family, or a combined depth/stencil
    /// format as its depth; a one-plane extended-range format (`BGRA10_XR`, `BGR10_XR`) as any
    /// format but its sRGB twin, or another format as one (plain Metal aborts: "source texture
    /// pixelFormat not compatible with texture view pixelFormat"). And any format but its own for a texture made in `X32_Stencil8`,
    /// whose `Depth32Float_Stencil8` view plain Metal makes over the stencil texture's bytes in
    /// its GPU's own layout, no defined answer.
    ///
    /// Basis: metal-documented (Metal aborts "source texture pixelFormat not compatible with
    /// texture view pixelFormat").
    FormatIncompatible,
    /// A texture type Metal does not view the texture's type as.
    ///
    /// Basis: unknown (plain Metal's answer for a view type the texture's type is not viewed as is
    /// unmeasured: the cube view of a 2D texture on record is stopped over its slice range, not its
    /// type).
    TypeIncompatible,
    /// An empty level range, or one past the texture's levels.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "(newLevelRange.location + newLevelRange.length)(7) must be <= `[texture
    /// mipmapLevelCount](4)`", and "newLevelRange.length must not be 0" for an empty one).
    LevelRange,
    /// An empty slice range, or one past the texture's slices.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "(newSliceRange.location + newSliceRange.length)(3) must be <= (2)").
    SliceRange,
    /// A slice count the view's type does not take: one for a type that is not an array, six for
    /// a cube, a multiple of six for a cube array.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "newSliceRange.length(2) must equal (1) for textureType (MTLTextureType2D)").
    SliceCount,
    /// More than one slice of a 2D array over a buffer: Metal asserts on such a view, plain and
    /// validated (`sliceRange.length == 1`; a Metal probe on the M3 Max and an M4).
    ///
    /// Basis: metal-measured (an M3 Max and an M4: Metal asserts `sliceRange.length == 1` on such a
    /// view, plain and validated).
    BufferSlices,
    /// A view with a swizzle of a texture whose usage is none (a texture made with Apple's private
    /// usage 0x8 alone): Metal asserts on such a view, plain and validated (`_usage !=
    /// MTLTextureUsageUnknown`, an Apple M4). Its views without a swizzle are
    /// made, and so is a swizzled view of a texture of `ShaderWrite` or `RenderTarget` usage alone,
    /// which keeps no usage.
    ///
    /// Basis: metal-measured (an Apple M4: Metal asserts `_usage != MTLTextureUsageUnknown` on such
    /// a view, plain and validated).
    SwizzledViewOfNoUsage,
}

/// Why a view or a texture's use cannot be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ViewRefusal {
    /// Metal's API validation stops it.
    Invalid(ViewProblem),
    /// Plain Metal answers nil for it.
    Nil(ViewProblem),
    /// The device or the runtime lacks what it needs.
    Unsupported(Unsupported),
}

impl From<ViewProblem> for ViewRefusal {
    fn from(problem: ViewProblem) -> ViewRefusal {
        ViewRefusal::Invalid(problem)
    }
}

/// The class Metal views a format within: uncompressed color formats by bytes per pixel,
/// block-compressed formats by their family ([`block_family`]), and depth/stencil formats by their
/// Vulkan format (the stencil formats that view a combined format share its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewClass {
    Color { size: u32 },
    Block(PixelFormat),
    DepthStencil(Format),
}

/// Whether `pf` is a two-plane extended-range format, whose image does not hold Metal's bytes.
fn two_planes(pf: PixelFormat) -> bool {
    ExtendedRange::of(pf).is_some_and(|f| !f.holds_metal_bytes())
}

/// Whether `pf` is an extended-range format (`BGRA10_XR`, `BGR10_XR`, `RGB10A8_2P_XR10` and
/// their sRGB twins), which Metal views as its twin alone.
fn extended_range(pf: PixelFormat) -> bool {
    ExtendedRange::of(pf).is_some()
}

fn view_class(pf: PixelFormat) -> Option<ViewClass> {
    let format = byte_format(pf)?;
    let block = format.block()?;
    Some(if block.depth.is_some() || block.stencil.is_some() {
        ViewClass::DepthStencil(format)
    } else if block_extent(pf).is_some() {
        ViewClass::Block(block_family(pf))
    } else {
        ViewClass::Color { size: block.size }
    })
}

/// Whether `source` and `view` are two held two-plane YCbCr formats of one layout, one read
/// converted and the other its `_sRGB` twin: `Some(true)` for a view of the converted one as
/// its twin, `Some(false)` for the other order, `None` for any other pair.
fn ycbcr_twins(source: PixelFormat, view: PixelFormat) -> Option<bool> {
    use metal2vulkan::metal::format::YCbCrReads as R;
    let (Some(s), Some(v)) = (ycbcr_format(source), ycbcr_format(view)) else {
        return None;
    };
    let layout = s.bits == v.bits && s.subsampling == v.subsampling && s.packed == v.packed;
    if !(layout && ycbcr(source) && ycbcr(view)) {
        return None;
    }
    match (s.reads, v.reads) {
        (R::Converted, R::Srgb) => Some(true),
        (R::Srgb, R::Converted) => Some(false),
        _ => None,
    }
}

/// Plain Metal's verdict on viewing a texture of `source` as `view`, whatever the texture's
/// usage (see the module documentation); a format with no Vulkan representation is refused as
/// `metal::format` refuses it on `profile`.
fn check_format(
    source: PixelFormat,
    view: PixelFormat,
    profile: &DeviceProfile,
) -> Result<(), ViewRefusal> {
    if matches!(view, PixelFormat::Invalid | PixelFormat::Unspecialized) {
        return Err(ViewRefusal::Nil(ViewProblem::NotAFormat));
    }
    if view == source || srgb_twin(source) == Some(view) {
        return Ok(());
    }
    // Metal views a 4:2:2 texture in no other format, the other order included ("not castable",
    // plain too).
    if [source, view].into_iter().any(is_four_two_two) {
        return Err(ViewProblem::FormatIncompatible.into());
    }
    // Metal views a converted two-plane YCbCr texture as its `_sRGB` twin, whatever its usage
    // (`YCBCR8_420_2P` as `YCBCR8_420_2P_sRGB` and `YCBCR10_420_2P` as
    // `YCBCR10_420_2P_sRGB`, `ShaderRead` alone and with `PixelFormatView`), which reads what a
    // texture of the twin over the same planes reads (`texture/ycbcr_image.rs`). The other order
    // is not measured, the runtime's gap.
    match ycbcr_twins(source, view) {
        Some(true) => return Ok(()),
        Some(false) => {
            return Err(ViewRefusal::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(view),
            ))
        }
        None => {}
    }
    // Metal views a `YCBCR8_420_2P` texture in its own format only ("Texture Creation"
    // validation stops `R16Unorm`, plain too: the consumer's report); a view as one of another
    // texture is not measured, the runtime's gap.
    if ycbcr(source) {
        return Err(ViewProblem::FormatIncompatible.into());
    }
    if ycbcr(view) {
        return Err(ViewRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(view),
        ));
    }
    // Plain Metal views a two-plane extended-range texture as its sRGB twin,
    // whose reads it does not measure: the runtime's gap.
    if two_planes(source) && two_planes(view) {
        return Err(ViewRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(view),
        ));
    }
    // A padded 10-bit texture's view in another format, or another texture's in one, is not
    // measured: the runtime's gap. Nor is an `RGB10A2Unorm_sRGB` texture's view as
    // `RGB10A2Unorm`, or the other order, which plain Metal makes (an Apple M3 Max under macOS
    // 27.0.1), its reads unmeasured.
    if [source, view].into_iter().any(|pf| {
        metal2vulkan::metal::format::padded_words(pf)
            || ExtendedRange::of(pf) == Some(ExtendedRange::Rgb10A2Srgb)
    }) {
        return Err(ViewRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(view),
        ));
    }
    // A bfloat texture's view in another format, or another texture's in one, is not measured:
    // the runtime's gap.
    if [source, view]
        .into_iter()
        .any(|pf| ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat))
    {
        return Err(ViewRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(view),
        ));
    }
    // Nor an extended-range texture in any format but its twin (above), the other order
    // included, though a view class of the same size holds a one-plane one (a two-plane one too).
    if [source, view].into_iter().any(extended_range) {
        return Err(ViewProblem::FormatIncompatible.into());
    }
    let (Some(a), Some(b)) = (view_class(source), view_class(view)) else {
        return Err(match vk_format(view, &BTreeSet::new(), profile).err() {
            Some(CompileError::Unsupported(u)) => ViewRefusal::Unsupported(u.naming_format(view)),
            _ => ViewRefusal::Unsupported(Unsupported::not_implemented(Feature::TextureView)),
        });
    };
    if a != b {
        return Err(ViewProblem::FormatIncompatible.into());
    }
    Ok(())
}

/// The usage a view with a swizzle other than the default keeps of its source's: `ShaderRead`,
/// `PixelFormatView` and Apple's private `NoConcurrentAccess` (0x4000) and `BlockWritesOnly`
/// (0x10000), where `Video` (0x8000) goes with the write and render target bits (the consumer's
/// M3 Max: a swizzled view of usage 0x1c01d answers 0x14011, one of 0x10005 answers 0x10001).
const SWIZZLED_VIEW_USAGE: TextureUsage = TextureUsage::SHADER_READ
    .union(TextureUsage::PIXEL_FORMAT_VIEW)
    .union(TextureUsage::NO_CONCURRENT_ACCESS)
    .union(TextureUsage::BLOCK_WRITES_ONLY);

/// The family of texture types Metal views as each other.
fn type_family(ty: TextureType) -> u8 {
    use TextureType as T;
    match ty {
        T::Type1D | T::Type1DArray => 0,
        T::Type2D | T::Type2DArray | T::Cube | T::CubeArray => 1,
        T::Type2DMultisample | T::Type2DMultisampleArray => 2,
        T::Type3D => 3,
        T::TextureBuffer => 4,
    }
}

/// The view Metal makes of `source` for `request`, or why it stops it, on `profile` (for the
/// refusal of a format the device has no representation for).
pub(super) fn check_view(
    source: &ViewRecord,
    request: &ViewRequest,
    profile: &DeviceProfile,
) -> Result<ViewRecord, ViewRefusal> {
    use TextureType as T;
    check_format(source.pixel_format, request.pixel_format, profile)?;
    let ty = request.texture_type;
    if type_family(ty) != type_family(source.texture_type) {
        return Err(ViewProblem::TypeIncompatible.into());
    }
    // Metal views a two-plane YCbCr texture in its own format (or its `_sRGB` twin's) retyped,
    // as a 2D array of its one slice, and reads that view otherwise than the texture: in its own
    // format the luma codes alone (`Texture::luma_view`); in the twin's, unmeasured, the
    // runtime's gap. A view of its own type reads as the texture.
    if ycbcr(source.pixel_format)
        && ty != source.texture_type
        && request.pixel_format != source.pixel_format
    {
        return Err(ViewRefusal::Unsupported(
            Unsupported::not_implemented(Feature::TextureView).naming_format(request.pixel_format),
        ));
    }
    let (levels, slices) = (&request.levels, &request.slices);
    if levels.is_empty() || levels.end > source.level_count {
        return Err(ViewProblem::LevelRange.into());
    }
    if slices.is_empty() || slices.end > source.layer_count {
        return Err(ViewProblem::SliceRange.into());
    }
    let count = slices.end - slices.start;
    let fits = match ty {
        T::Type1D | T::Type2D | T::Type2DMultisample | T::Type3D => count == 1,
        T::Cube => count == 6,
        T::CubeArray => count.is_multiple_of(6),
        _ => true,
    };
    if !fits {
        return Err(ViewProblem::SliceCount.into());
    }
    if matches!(ty, T::Cube | T::CubeArray) && source.image_extent[0] != source.image_extent[1] {
        // Metal makes it; a Vulkan cube view needs a square, cube-compatible image.
        return Err(ViewRefusal::Unsupported(Unsupported::not_representable(
            Feature::TextureView,
        )));
    }
    let swizzle = request.swizzle.unwrap_or(source.swizzle);
    // The source's usage holds the bits Metal's texture drops (`effective_usage`), which a view
    // keeps; a source of no other usage is one Metal's texture answers none for.
    let usage = match request.swizzle {
        Some(s) if !s.is_default() => {
            let dropped = source.usage.difference(source.usage.without_dropped());
            if source.usage.without_dropped().is_empty() {
                return Err(ViewProblem::SwizzledViewOfNoUsage.into());
            }
            source
                .usage
                .intersection(SWIZZLED_VIEW_USAGE.union(dropped))
        }
        _ => source.usage,
    };
    Ok(ViewRecord {
        pixel_format: request.pixel_format,
        texture_type: ty,
        image_extent: source.image_extent,
        base_level: source.base_level + levels.start,
        level_count: levels.end - levels.start,
        base_layer: source.base_layer + slices.start,
        layer_count: count,
        samples: source.samples,
        swizzle,
        usage,
        alpha_reads_one: source.alpha_reads_one,
        address_mode_map: source.address_mode_map,
        presented: source.presented,
        read_as: None,
    })
}

/// The number of components of `format` and the width they share; `None` for a format whose
/// components differ in width (`A2B10G10R10_UNORM_PACK32`) and for a block-compressed one, whose
/// components the format table does not record.
fn component_layout(format: Format) -> Option<(u32, u32)> {
    let block = format.block()?;
    let bits = block
        .component_bits
        .filter(|&bits| bits > 0 && block.texels == 1)?;
    Some((block.size * 8 / bits, bits))
}

/// Whether a Vulkan view of format `view` keeps the components of its image of format `image`:
/// the same format, or as many components of the same width. A device whose portability subset
/// withholds `imageViewFormatReinterpretation` makes only such views (the module's
/// reinterpretation note). A format of components of mixed widths, or block-compressed, keeps only
/// its own: the format table records no more of it, so such a pair is refused there even where its
/// components match (`RGB10A2Unorm` as `RGB10A2Uint`, an sRGB twin of a compressed format).
pub(super) fn keeps_components(image: Format, view: Format) -> bool {
    image == view
        || matches!(
            (component_layout(image), component_layout(view)),
            (Some(a), Some(b)) if a == b
        )
}

/// The refusal of a view that would reinterpret its image's components where the device does not
/// (the module's reinterpretation note).
pub(super) fn reinterpretation_refused() -> Unsupported {
    Unsupported::not_representable(Feature::ViewFormatReinterpretation)
}
