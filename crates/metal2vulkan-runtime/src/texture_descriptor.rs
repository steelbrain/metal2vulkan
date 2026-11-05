//! `MTLTextureDescriptor` as a value, and the rules Metal's API validation applies to it.
//!
//! [`TextureDescriptor::check`] gives the texture's [`TextureLayout`]: its type, extent, levels,
//! Vulkan array layers and samples. A descriptor Metal stops, or answers nil for, is a
//! [`TextureProblem`] ([`Refused`] says which); what the Vulkan device cannot do is decided later,
//! against the device ([`super::image_plan`]).
//!
//! The rules are the ones `MTLTexture.h` states for each property: a 1D texture has height 1, only
//! a 3D texture has a depth other than 1, only an array type has an array length other than 1 (at
//! most 2048), multisample, 1D and 1D array textures have one level (the 1D types
//! measured: plain Metal asserts "the texture type MTLTextureType1D requires that
//! mipmapLevelCount is 1"), only multisample types have more than one sample, a cube is square, and
//! a texture has no more levels than its full chain. Every extent is at least 1. More are measured
//! (an Apple M3 Max on macOS 26.7 and an M4 on 27.0.1, the same on both):
//!
//! - A swizzled texture is not written by shaders or rendered to (Metal stops `ShaderWrite` or
//!   `RenderTarget` usage with a swizzle other than the default at creation, validation on or
//!   off), but for one of Apple's private `writeSwizzleEnabled`
//!   ([`TextureDescriptor::write_swizzle_enabled`]), which Metal makes from the device's, a
//!   heap's and a buffer's creators, and whose writes go through the swizzle; the shared creator
//!   still stops it. The runtime does not build one yet (`NotImplemented(WriteSwizzle)`); the heap
//!   size query answers for it.
//! - The texture types a pixel format takes (probe 5, every format in each 1D, 1D array, 2D, 2D
//!   array, cube, cube array and 3D texture): a depth or stencil format no 1D or 3D type; a
//!   compressed format no 1D type, and an ETC2, EAC or PVRTC one no 3D type either; a 4:2:2
//!   format (`GBGR422`, `BGRG422`) only 2D, of one level (plain too); and no texture is made in an
//!   `X24_Stencil8` format, which only views a texture (API validation). A depth or stencil format
//!   takes no texture buffer either (plain Metal stops it, `Depth32Float`); a `Depth32Float` 2D
//!   multisample texture is made, and the other multisample textures of these formats are not
//!   measured, and not refused here.
//! - A side past the type's largest is stopped, whatever the format (an M4): 16384 for every type
//!   but a 3D texture (2048 on each axis) and a texture buffer, 2^28 texels wide.
//! - A multisample texture has more than one sample, at a count the device takes
//!   (`supportsTextureSampleCount:`) or an Apple GPU's does (2 and 4); Metal stops another (3,
//!   8 and 16 on an M3 Max and an M4; the same probes). A count only an Apple GPU takes is the
//!   device's gap, refused as its image is planned (`Unsupported`, `TextureSampleCount`).
//! - A texture buffer takes as many levels as its width's full chain: Metal's device, heap and
//!   shared creators make one of separate levels, with no buffer behind it, and a buffer's
//!   creator stops it ("Linear texture: cannot be mipmapped"; the same probes). The runtime does
//!   not build one yet (`NotImplemented(TextureBufferLevels)`); the heap size query answers
//!   Metal's, the size of its first level.
//! - A texture made in `X32_Stencil8` is a stencil texture as a `Stencil8` one is, in the same
//!   types: plain Metal makes it, its allocation and heap size those of `Stencil8`, in every usage
//!   and storage mode (an M4 under macOS 27.0.1; an M3 Max under 26.7, the consumer's report),
//!   though Metal's API validation stops it ("Cannot use an Xn format when creating a new texture",
//!   but for a heap's texture). The runtime holds it as it holds the `X32_Stencil8` view of a
//!   `Depth32Float_Stencil8` texture, through the stencil of a `Depth32Float_Stencil8` image, which
//!   a pipeline of that stencil format renders to.
//! - The extents a pixel format takes (probe 6, every format in 2D textures from 1x1 to 16x8,
//!   in 2D array, cube and cube array textures of 8x8, 6x6 and 7x5, and in 3D textures of 4x4x4,
//!   7x5x3 and 8x8x2): a PVRTC texture is square with a power-of-two side, and a 4:2:2 one has
//!   an even width. Metal stops either at creation, validation on or off. No other format has
//!   an extent rule there: the other compressed formats take 1x1, 3x3 and 7x5 textures.
//! - A format Metal does not describe ([`Device::pixel_format_info`](crate::Device::pixel_format_info)
//!   answers [`PixelFormatInfo::INVALID`](crate::PixelFormatInfo::INVALID)) takes no texture:
//!   Metal's validation stops it from the device's
//!   creator as it stops `Invalid` ("has invalid pixelFormat"), and the runtime stops it from every
//!   creator and the heap size query alike.

use super::address_map::AddressModeMap;
use super::capabilities::sample_count_supported;
use super::copy_plan::block_extent;
use super::error::{invalid, nil, Error, Problem, RuntimeGap};
use super::image_plan::block_family;
use super::pixel_format_info::describes;
use super::resource_ids::IdPlacement;
use super::values::{ResourceOptions, StorageMode};
use metal2vulkan::metal::format::{
    footprint, is_depth_stencil, padded_words, ycbcr, ycbcr_format, ExtendedRange, Footprint,
};
pub use metal2vulkan::metal::TextureSwizzleChannels;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::SampleCountFlags;

/// Metal's largest texture rotation: 4, which stores as 1 does.
pub(super) const MAX_ROTATION: u64 = 4;

/// Metal's largest array length (`MTLTextureDescriptor.arrayLength`: "less than 2048", which Metal
/// accepts as at most 2048).
pub(super) const MAX_ARRAY_LENGTH: u64 = 2048;

/// Metal's largest cube array length: 341 cubes, whose 2046 faces are the most within
/// [`MAX_ARRAY_LENGTH`] layers. Metal stops 342 and more, plain too, for every format (an M3 Max
/// under macOS 26.7, and an M4 under 27.0.1: `RGBA8Unorm`, `BGRA10_XR`, both two-plane
/// extended-range formats and a two-plane YCbCr one).
pub(super) const MAX_CUBE_ARRAY_LENGTH: u64 = MAX_ARRAY_LENGTH / 6;

/// The creator a descriptor is checked for ([`TextureDescriptor::check_made`]): the rules are the
/// same, but for the order of a few and what each makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Creator {
    /// The device's, an automatic heap's, a sparse heap's, or a wrapped image's: the creators
    /// that lay the image out themselves.
    Device,
    /// A placement heap's `newTextureWithDescriptor:offset:`, which lays the texture at the
    /// caller's offset. It takes every rule [`Creator::Device`] takes; the one difference
    /// measured is a format Metal lays out no image storage for, which it makes here and answers
    /// nil for from the device's own creator and from an automatic heap
    /// (`metal::format::packed_codes`).
    Placement,
    /// A buffer's, or an IOSurface plane's without its YCbCr facts: the linear creators, whose
    /// own rules stop a texture of more than one level (`linear_plan`).
    Buffer,
    /// Over an IOSurface plane with its YCbCr facts: a YCbCr format is made.
    OverSurface,
    /// `newSharedTextureWithDescriptor:`.
    Shared,
    /// The heap size query, which answers for what Metal makes and the runtime does not build
    /// yet.
    SizeQuery,
}

/// Why a descriptor makes no texture: Metal's validation ends the process, or plain Metal answers
/// nil, each for a [`TextureProblem`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Refused {
    /// Metal's validation ends the process ([`Error::Invalid`]).
    Stops(TextureProblem),
    /// Plain Metal answers nil ([`Error::Nil`]).
    Nil(TextureProblem),
    /// Metal makes the texture and the runtime does not build it yet
    /// ([`Error::NotImplemented`]).
    NotYet(RuntimeGap),
}

impl Refused {
    /// What `call` answers for it.
    pub(super) fn error(self, call: &'static str) -> Error {
        match self {
            Refused::Stops(problem) => invalid(call, Problem::Texture(problem)),
            Refused::Nil(problem) => nil(call, Problem::Texture(problem)),
            Refused::NotYet(gap) => Error::NotImplemented(gap),
        }
    }
}

impl From<TextureProblem> for Refused {
    /// A problem Metal's validation stops.
    fn from(problem: TextureProblem) -> Refused {
        Refused::Stops(problem)
    }
}

/// `MTLTextureDescriptor`. `Default` is Metal's: a 2D `RGBA8Unorm` texture of 1×1, one level,
/// one sample, one slice, read by shaders, with every channel from itself, resource options of
/// raw value 0 (shared storage, the default CPU cache and hazard tracking modes), and a resource
/// ID the runtime picks; its alpha its own and its address modes the identity's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    /// `textureType`.
    pub texture_type: TextureType,
    /// `pixelFormat`. A format the device cannot make is refused at creation.
    pub pixel_format: PixelFormat,
    /// `width`, in texels.
    pub width: u64,
    /// `height`, in texels; 1 for a 1D or buffer texture.
    pub height: u64,
    /// `depth`, in texels; 1 for a texture that is not 3D.
    pub depth: u64,
    /// `mipmapLevelCount`.
    pub mipmap_level_count: u64,
    /// `sampleCount`.
    pub sample_count: u64,
    /// `arrayLength`: the slices of an array type (a cube array's cubes); 1 otherwise.
    pub array_length: u64,
    /// `usage`. `Unknown` (no bits) is taken as Metal takes it: as `ShaderRead | ShaderWrite |
    /// RenderTarget | PixelFormatView`.
    pub usage: TextureUsage,
    /// `resourceOptions`: the storage mode, CPU cache mode and hazard tracking mode.
    pub resource_options: ResourceOptions,
    /// `allowGPUOptimizedContents`: whether the GPU may keep the texture in a layout of its own
    /// (on Apple's GPUs, lossless compression with its metadata). Off, the runtime sizes and makes
    /// a heap's texture's image as one of `PixelFormatView` usage (`MUTABLE_FORMAT` with no list
    /// of formats): Metal lays out a color texture with the property off as one of
    /// `PixelFormatView` usage, and MoltenVK sizes that image as Metal does, so
    /// [`Device::heap_texture_size_and_align`](crate::Device::heap_texture_size_and_align)
    /// answers Metal's size there. A depth or stencil texture's answer on MoltenVK stays the
    /// optimized texture's, announced once per device (`optimized-depth-stencil-texture`). A
    /// device's own texture keeps the image of its usage, with the driver's compression, and
    /// announces nothing: nothing shows its layout. The texture's usage stays the descriptor's.
    pub allow_gpu_optimized_contents: bool,
    /// `swizzle`: each channel's source as shaders read it; one other than the default is refused
    /// with `ShaderWrite` or `RenderTarget` usage, but with `write_swizzle_enabled`.
    pub swizzle: TextureSwizzleChannels,
    /// `writeSwizzleEnabled` (Apple's private property): a swizzle other than the default takes
    /// `ShaderWrite` and `RenderTarget` usage, and the texture's writes go through it as its
    /// reads do (a `BGRA` swizzle's write of `(0.25, 0.5, 0.75, 1)` to an `RGBA8Unorm` texture
    /// stores `(0.75, 0.5, 0.25, 1)`). Plain Metal makes such a texture from the device's, a
    /// heap's and a buffer's creators, and its validation stops it from
    /// `newSharedTextureWithDescriptor:` (an M4 under macOS 27.0.1 and an M3 Max under 26.7). The
    /// runtime does not build it yet: those creators refuse it `NotImplemented(WriteSwizzle)`, and
    /// [`Device::heap_texture_size_and_align`](crate::Device::heap_texture_size_and_align)
    /// answers its size. Without a swizzle, or without writes, it changes nothing.
    pub write_swizzle_enabled: bool,
    /// Whether shaders read the texture's alpha as 1 (a runtime extension; Metal has no such
    /// property): its reads, samples and gathers, border texels included, read each channel whose
    /// source is alpha (after `swizzle`) as 1, as Metal reads a format without alpha, while its
    /// writes, render targets and copies keep the alpha it holds. It is a component mapping of the
    /// texture's sampled views, so unlike `swizzle` it takes `ShaderWrite` and `RenderTarget`
    /// usage; a function writing it at a storage slot reads it so too. The texture's views keep
    /// it.
    pub alpha_reads_one: bool,
    /// The address modes the texture is sampled with (a runtime extension; Metal has no such
    /// property): each mode a sampler states, read as the map's ([`AddressModeMap`],
    /// `address_map.rs`). Applied at bind to a runtime sampler at a slot; a constexpr sampler, a
    /// slot whose textures map it otherwise in one stage, and a function sampling handles that
    /// can reach the texture are refused (`RuntimeGap::MappedTextureSampling`). The texture's views
    /// keep it. The identity, the default, changes nothing.
    pub address_mode_map: AddressModeMap,
    /// `resourceIndex` with `forceResourceIndex` (Apple's private properties, a runtime
    /// extension): where the texture goes in the space of texture resource IDs
    /// (`resource_ids.rs`). [`IdPlacement::Outside`], no ID at all, is
    /// [`Device::new_texture`](crate::Device::new_texture)'s and
    /// [`Device::new_texture_with_vk_image`](crate::Device::new_texture_with_vk_image)'s alone.
    pub placement: IdPlacement,
    /// `framebufferOnly` (`setFramebufferOnly:`, SPI): the texture is meant for render passes
    /// alone. Plain Metal makes such a texture over an IOSurface plane only
    /// ([`Buffer::new_texture_with_iosurface_plane`](crate::Buffer::new_texture_with_iosurface_plane));
    /// the device's, a buffer's and a heap's creators assert on it (refused
    /// [`TextureProblem::FramebufferOnly`]), and `newSharedTextureWithDescriptor:` makes the
    /// texture without it. Its other uses (blits, CPU access, shader reads and writes), which
    /// Metal's API validation stops, read and write it as plain Metal does.
    pub framebuffer_only: bool,
    /// `rotation` (`setRotation:`, SPI): 0, or 1 to 4 for a framebuffer-only texture over an
    /// IOSurface plane, which a render pass draws unturned and stores turned: rotation 2 by 180
    /// degrees, 1 and 4 a quarter turn (its store `height` texels wide and `width` tall, a
    /// pass's texel `(x, y)` at `(height - 1 - y, x)`), 3 the other quarter turn (at
    /// `(y, width - 1 - x)`). Its other uses are plain Metal's: CPU access and blits address
    /// the plane as a texture of its own extent at the plane's rows, and
    /// functions see the stored frame unturned (its extent their `get_width()` and
    /// `get_height()`), their writes dropped. Refused past 4
    /// ([`TextureProblem::Rotation`]) and without `framebuffer_only`
    /// ([`TextureProblem::RotationWithoutFramebufferOnly`]) at creation, as Metal's descriptor
    /// validation asserts, plain too.
    pub rotation: u64,
    /// `isDrawable` (`setIsDrawable:`, SPI): kept and reported by the texture
    /// ([`Texture::is_drawable`](crate::Texture::is_drawable)); plain Metal draws, stores and
    /// rotates such a texture as one without it.
    pub is_drawable: bool,
}

impl Default for TextureDescriptor {
    fn default() -> TextureDescriptor {
        TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA8Unorm,
            width: 1,
            height: 1,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TextureUsage::SHADER_READ,
            resource_options: ResourceOptions::default(),
            allow_gpu_optimized_contents: true,
            swizzle: TextureSwizzleChannels::DEFAULT,
            write_swizzle_enabled: false,
            alpha_reads_one: false,
            address_mode_map: AddressModeMap::IDENTITY,
            placement: IdPlacement::Any,
            framebuffer_only: false,
            rotation: 0,
            is_drawable: false,
        }
    }
}

impl TextureDescriptor {
    /// `texture2DDescriptorWithPixelFormat:width:height:mipmapped:`: a 2D texture with the full
    /// chain of levels when `mipmapped`.
    pub fn texture_2d(
        pixel_format: PixelFormat,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> TextureDescriptor {
        TextureDescriptor {
            pixel_format,
            width,
            height,
            mipmap_level_count: if mipmapped {
                full_chain(&[width, height])
            } else {
                1
            },
            ..TextureDescriptor::default()
        }
    }

    /// `textureCubeDescriptorWithPixelFormat:size:mipmapped:`.
    pub fn texture_cube(
        pixel_format: PixelFormat,
        size: u64,
        mipmapped: bool,
    ) -> TextureDescriptor {
        TextureDescriptor {
            texture_type: TextureType::Cube,
            ..TextureDescriptor::texture_2d(pixel_format, size, size, mipmapped)
        }
    }

    /// `textureBufferDescriptorWithPixelFormat:width:resourceOptions:usage:`.
    pub fn texture_buffer(
        pixel_format: PixelFormat,
        width: u64,
        resource_options: ResourceOptions,
        usage: TextureUsage,
    ) -> TextureDescriptor {
        TextureDescriptor {
            texture_type: TextureType::TextureBuffer,
            pixel_format,
            width,
            usage,
            resource_options,
            ..TextureDescriptor::default()
        }
    }

    /// The usage the runtime makes a heap's texture's image of this descriptor for, and sizes it by
    /// (`image_plan`, `texture/heap.rs`): its usage ([`own_image_usage`](Self::own_image_usage)),
    /// with `PixelFormatView` where it does not allow GPU-optimized contents. Metal lays out a
    /// color texture with `allowGPUOptimizedContents` off as it lays out the texture with
    /// `PixelFormatView` usage added and the property on: the same size and alignment, without the
    /// compression metadata (each of 14 uncompressed color formats in 2D, mipmapped, array, cube,
    /// 3D, 1D and multisample shapes, `Shared` and `Private`; an M3 Max on macOS 26.7 and an M4 on
    /// 27.0.1 alike). MoltenVK makes a `PixelFormatView` Metal texture of such an image
    /// (`MUTABLE_FORMAT` with no list of formats), sized as Metal sizes one. The texture's own
    /// usage, which its views are checked against, stays the descriptor's.
    ///
    /// A descriptor of `write_swizzle_enabled` keeps its usage, swizzled or not, as Metal's
    /// texture reports it (`ShaderRead | ShaderWrite` of a `BGRA` swizzle): the
    /// heap size query sizes it so.
    pub(super) fn image_usage(&self) -> TextureUsage {
        let usage = self.own_image_usage();
        if self.allow_gpu_optimized_contents {
            usage
        } else {
            usage | TextureUsage::PIXEL_FORMAT_VIEW
        }
    }

    /// The usage the runtime makes the image of a device's own texture of this descriptor for
    /// (`Device::new_texture`): [`image_usage`](Self::image_usage) without the `PixelFormatView`
    /// that `allowGPUOptimizedContents` off adds. The property changes the texture's layout alone,
    /// which a device's own texture never shows: nothing answers its size, no other resource
    /// aliases its bytes, and its views, blits and fresh contents follow the descriptor's own
    /// usage and the property (`view_plan.rs`, `compressed_blit.rs`, `heap_tiled.rs`'s pattern).
    /// So its image keeps the lossless compression a driver gives an image of the texture's usage,
    /// which a driver that compresses drops from a `MUTABLE_FORMAT` image of no list of formats
    /// (ANV). A heap's texture takes `image_usage`: the heap's size answer and its aliasing show
    /// the layout. So does a device's own texture where the device's copies of the compressed
    /// image would rewrite texels Metal keeps (`Device::copy_exact_usage`, MoltenVK's
    /// `RG11B10Float` and `RGB9E5Float` images): there the compression shows in its blits.
    pub(super) fn own_image_usage(&self) -> TextureUsage {
        if self.write_swizzle_enabled {
            effective_usage(self.usage)
        } else {
            texture_usage(self.usage, self.swizzle)
        }
    }

    /// The texture this descriptor makes from the device's or a heap's creator on a device of
    /// `rules`, or the rule it breaks.
    pub(super) fn check(&self, rules: TextureRules) -> Result<TextureLayout, Refused> {
        self.check_made(Creator::Device, rules)
    }

    /// The texture this descriptor makes over an IOSurface plane, or the rule it breaks: as
    /// [`check`](Self::check), but a YCbCr format, which Metal makes only over a surface's
    /// planes, is made (`metal::format::ycbcr`).
    pub(super) fn check_over_surface(&self, rules: TextureRules) -> Result<TextureLayout, Refused> {
        self.check_made(Creator::OverSurface, rules)
    }

    /// The shareable texture this descriptor makes (`newSharedTextureWithDescriptor:`), or the
    /// rule it breaks: as [`check`](Self::check), but a format of which Metal makes no texture
    /// alone answers nil before the sample count's rule, where the device's creator stops a
    /// count it does not take (`RGBA16Float_XR` at 8 samples: nil from the shared creator, a stop
    /// from the device's).
    pub(super) fn check_shared(&self, rules: TextureRules) -> Result<TextureLayout, Refused> {
        self.check_made(Creator::Shared, rules)
    }

    /// [`check`](Self::check) for `creator`: a YCbCr format made over a surface, and what Metal
    /// makes and the runtime does not build yet refused ([`Refused::NotYet`]) once every rule
    /// Metal stops or answers nil for passed, but for the heap size query.
    ///
    /// Every rule Metal's validation stops comes before the formats of which plain Metal makes no
    /// texture and answers nil (`NoTextureOfFormat`, `Unspecialized`): Metal ends the process at
    /// each of them for those formats too, from every creator and the heap size query (an M4
    /// under macOS 27.0.1 and an M3 Max under 26.7 alike: an extent of 0 or past the largest side,
    /// a 1D height or a non-3D depth other than 1, a cube not square, an array length the type
    /// lacks, a level count past the chain or other than 1 on 1D, a sample count on a type that is
    /// not multisample, a swizzle with writes, a rotation past 4 or without `framebufferOnly`, and
    /// `ShaderAtomic` usage).
    pub(super) fn check_made(
        &self,
        creator: Creator,
        rules: TextureRules,
    ) -> Result<TextureLayout, Refused> {
        use TextureType as T;
        let over_surface = creator == Creator::OverSurface;
        let ty = self.texture_type;
        let multisample = matches!(ty, T::Type2DMultisample | T::Type2DMultisampleArray);
        let pf = self.pixel_format;
        // Plain Metal ends its process over a texture of a format it describes none of, as it
        // does over `Invalid`, before any other rule (`TextureProblem::NotAFormat`).
        if !describes(pf) {
            return Err(TextureProblem::NotAFormat.into());
        }
        // Metal stops a two-plane YCbCr multisample texture, plain too ("does not support
        // multisample textures"; every one of the 69, an Apple M4 under macOS 27.0.1; a 2D
        // multisample array too).
        if ycbcr_multisample(pf, ty) {
            return Err(TextureProblem::TypeForFormat.into());
        }
        // Plain Metal stops a multisample `Unspecialized` texture ("does not support multisample
        // textures") and, plain too, one of `ShaderAtomic` usage ("not a valid format to be used
        // with texture atomics"; an M4 under macOS 27.0.1).
        if pf == PixelFormat::Unspecialized && multisample {
            return Err(TextureProblem::TypeForFormat.into());
        }
        // No format of which Metal makes no texture alone takes `ShaderAtomic` usage: plain
        // Metal stops it.
        if (no_texture_alone(pf) || pf == PixelFormat::Unspecialized)
            && self.usage.contains(TextureUsage::SHADER_ATOMIC)
        {
            return Err(TextureProblem::UsageForFormat.into());
        }
        // A type plain Metal answers nil for comes after every rule it stops.
        let type_nil = match type_for_format(pf, ty) {
            Ok(()) => None,
            Err(Refused::Nil(problem)) => Some(problem),
            Err(stops) => return Err(stops),
        };
        // Plain Metal makes a memoryless two-plane extended-range cube and cube array, where it
        // answers nil for one of another storage mode (its validation stops a memoryless cube of
        // every format). The device's creation refuses it (`Device::new_texture`).
        let memoryless = self.resource_options.storage_mode().ok() == Some(StorageMode::Memoryless);
        let type_nil = type_nil.filter(|_| !(memoryless_two_plane_cube(pf, ty) && memoryless));
        if two_planes(pf) && self.usage.contains(TextureUsage::SHADER_ATOMIC) {
            return Err(TextureProblem::UsageForFormat.into());
        }
        // Plain Metal stops a depth or stencil texture of `ShaderAtomic` usage ("is not a valid
        // format to be used with texture atomics": the size query of `Depth16Unorm`,
        // `Depth32Float`, `Stencil8`, `Depth32Float_Stencil8` and `X32_Stencil8` at every usage
        // set with it; `newTextureWithDescriptor:` of a `Private` one, the
        // consumer's M3 Max under 26.7), on every device alike.
        if is_depth_stencil(pf) && self.usage.contains(TextureUsage::SHADER_ATOMIC) {
            return Err(TextureProblem::UsageForFormat.into());
        }
        let (w, h, d) = (self.width, self.height, self.depth);
        if w == 0 || h == 0 || d == 0 {
            return Err(TextureProblem::ZeroExtent.into());
        }
        let one_dimensional = matches!(ty, T::Type1D | T::Type1DArray | T::TextureBuffer);
        if (one_dimensional && h != 1) || (ty != T::Type3D && d != 1) {
            return Err(TextureProblem::ExtentForType.into());
        }
        if let Some(max) = largest_side(ty) {
            if w > max || h > max || d > max {
                return Err(TextureProblem::PastLargestSide { max }.into());
            }
        }
        if matches!(ty, T::Cube | T::CubeArray) && w != h {
            return Err(TextureProblem::CubeNotSquare.into());
        }
        extent_for_format(pf, w, h)?;
        let array = matches!(
            ty,
            T::Type1DArray | T::Type2DArray | T::CubeArray | T::Type2DMultisampleArray
        );
        let length = self.array_length;
        let max_length = match ty {
            T::CubeArray => MAX_CUBE_ARRAY_LENGTH,
            _ => MAX_ARRAY_LENGTH,
        };
        if (!array && length != 1) || length == 0 || length > max_length {
            return Err(TextureProblem::ArrayLength.into());
        }
        // Metal refuses a mipmapped 4:2:2 texture, plain too ("the MTLPixelFormat422 pixel
        // format does not support mipmaps"). Plain Metal makes a texture buffer of as many levels
        // as its width's full chain (7 of a 64-texel one) from the device's, a heap's and the
        // shared creator, and stops one more; a buffer's creator stops more than one
        // (`linear_plan`).
        let single_level = multisample
            || matches!(ty, T::Type1D | T::Type1DArray)
            || matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422);
        let chain = match ty {
            T::Type3D => full_chain(&[w, h, d]),
            _ => full_chain(&[w, h]),
        };
        let levels = self.mipmap_level_count;
        if levels == 0 || levels > chain || (single_level && levels != 1) {
            return Err(TextureProblem::MipmapLevelCount { max: chain }.into());
        }
        let samples = self.sample_count;
        if samples == 0 || (!multisample && samples != 1) {
            return Err(TextureProblem::SampleCount.into());
        }
        // Plain Metal stops a multisample texture of one sample ("sampleCount must be > 1 for
        // multisample textures") and of a count the device does not take ("sampleCount (8) is
        // not supported by device": 3, 8 and 16 on an Apple M3 Max under macOS 26.7 and an M4
        // under 27.0.1, `RGBA8Unorm`, `RGBA16Float` and `RGBA16Float_XR` alike, from every
        // creator and the heap size query). The shared creator answers nil first for a format of
        // which Metal makes no texture alone. A count an Apple GPU takes and the device does not is
        // the device's gap, which its image's plan refuses (`Unsupported`, `TextureSampleCount`),
        // not a stop.
        let counts = rules.sample_counts | APPLE_SAMPLE_COUNTS;
        if multisample && (samples == 1 || !sample_count_supported(counts, samples)) {
            if creator == Creator::Shared && no_texture_alone(pf) {
                return Err(Refused::Nil(TextureProblem::NoTextureOfFormat));
            }
            return Err(TextureProblem::SampleCount.into());
        }
        // Metal makes a swizzle with writes under `writeSwizzleEnabled`, but from the shared
        // creator, which stops it as without.
        let writes = TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
        let swizzled = !self.swizzle.is_default();
        let write_swizzle_taken = self.write_swizzle_enabled && creator != Creator::Shared;
        if swizzled && self.usage.intersects(writes) && !write_swizzle_taken {
            return Err(TextureProblem::SwizzleWithWrites.into());
        }
        // Metal's descriptor validation, plain too: "rotation is not a valid MTLTextureRotation"
        // past 4, and "a non-zero rotation set but is not marked as framebufferOnly".
        if self.rotation > MAX_ROTATION {
            return Err(TextureProblem::Rotation.into());
        }
        if self.rotation != 0 && !self.framebuffer_only {
            return Err(TextureProblem::RotationWithoutFramebufferOnly.into());
        }
        if let Some(problem) = type_nil {
            return Err(Refused::Nil(problem));
        }
        if odd_sided_chroma_pairs(pf, [w, h], ty, self.usage) {
            return Err(Refused::Nil(TextureProblem::ExtentForFormat));
        }
        // Metal lays out no image storage for a packed 10-bit format
        // (`metal::format::packed_codes`), so it makes a texture of one from exactly the creators
        // that need no image laid out and answers nil from the two that would lay it out
        // themselves ([`no_storage_laid_out`]). Where Metal makes one, the answer past this carve
        // is this crate's own `Unsupported(Format, NotImplemented)` out of the creator's image or
        // texel planning, which is true of the format in every shape; Metal's nil would claim
        // Metal refuses the descriptor, which the measurement shows it does not.
        let storage = self.resource_options.storage_mode().ok();
        let no_storage = no_storage_laid_out(pf, creator, storage);
        if no_texture_alone(pf) && !(over_surface && ycbcr(pf)) && !no_storage {
            return Err(Refused::Nil(TextureProblem::NoTextureOfFormat));
        }
        // Plain Metal makes no `Unspecialized` texture and answers nil, whatever the storage
        // and usage.
        if pf == PixelFormat::Unspecialized {
            return Err(Refused::Nil(TextureProblem::Unspecialized));
        }
        // What Metal makes and the runtime does not build yet, once every rule Metal stops or
        // answers nil for passed: a texture buffer of more than one level from the creators
        // that make one (a buffer's stops it, `linear_plan`), and a swizzle that writes. The
        // heap size query answers for both, as Metal's does.
        let made_here = matches!(
            creator,
            Creator::Device | Creator::Placement | Creator::Shared
        );
        if ty == T::TextureBuffer && levels > 1 && made_here {
            return Err(Refused::NotYet(RuntimeGap::TextureBufferLevels));
        }
        if swizzled
            && write_swizzle_taken
            && effective_usage(self.usage).intersects(writes)
            && creator != Creator::SizeQuery
        {
            return Err(Refused::NotYet(RuntimeGap::WriteSwizzle));
        }
        let narrow = |v: u64| u32::try_from(v).map_err(|_| TextureProblem::ExtentOverflow);
        let faces = if matches!(ty, T::Cube | T::CubeArray) {
            6
        } else {
            1
        };
        Ok(TextureLayout {
            texture_type: ty,
            pixel_format: self.pixel_format,
            extent: [narrow(w)?, narrow(h)?, narrow(d)?],
            levels: narrow(levels)?,
            // The array length is at most 2048, so six faces of it fit.
            layers: (length * faces) as u32,
            samples: narrow(samples)?,
        })
    }
}

/// The usage a texture made with Metal `usage` has. Metal reports `Unknown` (no bits) as
/// `ShaderRead | ShaderWrite | RenderTarget | PixelFormatView` (0x17) and treats it so: a format
/// view of it needs no further usage, and a swizzled view of it keeps `ShaderRead |
/// PixelFormatView` (measured on an Apple M3 Max).
///
/// A usage holding only bits Metal's texture drops (`TextureUsage::DROPPED`: 0x8) is not `Unknown`:
/// Metal's texture of 0x8 alone answers no usage (0), and Metal keeps it compressed as one of a
/// usage. The runtime keeps the bit in a texture's usage, so its compression,
/// its image and its views follow the descriptor's bits, and
/// [`Texture::usage`](crate::Texture::usage) answers without it.
pub(super) fn effective_usage(usage: TextureUsage) -> TextureUsage {
    if usage.is_empty() {
        TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW
    } else {
        usage
    }
}

/// The usage of a texture made with Metal `usage` and `swizzle`: [`effective_usage`], of which a
/// texture with a swizzle other than the default keeps only `ShaderRead` and `PixelFormatView`,
/// as Metal reports it (a swizzled descriptor of `Unknown` usage reports `0x11`; measured on an
/// Apple M3 Max and M4). [`TextureDescriptor::check`] refuses a swizzled descriptor asking for
/// more. The bits Metal's texture drops stay ([`effective_usage`]).
pub(super) fn texture_usage(usage: TextureUsage, swizzle: TextureSwizzleChannels) -> TextureUsage {
    let usage = effective_usage(usage);
    if swizzle.is_default() {
        usage
    } else {
        let dropped = usage.difference(usage.without_dropped());
        usage.intersection(TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW | dropped)
    }
}

/// The number of levels of a full chain for an extent whose largest axis is the largest of `axes`:
/// `floor(log2(max)) + 1`.
pub(super) fn full_chain(axes: &[u64]) -> u64 {
    let max = axes.iter().copied().max().unwrap_or(1).max(1);
    u64::from(u64::BITS - max.leading_zeros())
}

/// What a checked descriptor makes: the texture's type, format, level-0 extent, levels, Vulkan
/// array layers (six per cube), and samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TextureLayout {
    pub(super) texture_type: TextureType,
    pub(super) pixel_format: PixelFormat,
    pub(super) extent: [u32; 3],
    pub(super) levels: u32,
    pub(super) layers: u32,
    pub(super) samples: u32,
}

/// A texture descriptor plain Metal itself refuses, ending its process ([`Error::Invalid`]), or
/// answers nil for ([`Error::Nil`]: [`NoTextureOfFormat`](Self::NoTextureOfFormat),
/// [`Unspecialized`](Self::Unspecialized), and a two-plane extended-range cube's
/// [`TypeForFormat`](Self::TypeForFormat)).
///
/// [`Error::Invalid`]: crate::Error::Invalid
/// [`Error::Nil`]: crate::Error::Nil
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureProblem {
    /// `Invalid`, or a format Metal describes none of
    /// ([`Device::pixel_format_info`](crate::Device::pixel_format_info) answers
    /// [`PixelFormatInfo::INVALID`](crate::PixelFormatInfo::INVALID)): plain Metal's own descriptor
    /// validation ends its process over either, the API validation layer off.
    /// `newTextureWithDescriptor:` and `heapTextureSizeAndAlignWithDescriptor:` both assert in
    /// `-[MTLTextureDescriptorInternal validateWithDevice:]` ("MTLTextureDescriptor has invalid
    /// pixelFormat (255)") for `Invalid`, for `Depth24Unorm_Stencil8` and for `X24_Stencil8`, the
    /// two an Apple GPU's Metal describes none of, while `X32_Stencil8` beside them is described
    /// and made; `setPixelFormat:` takes each of those three values and reads it back, and ends
    /// the process itself on a raw value no release names ("pixelFormat (999) is not a valid
    /// MTLPixelFormat"). That one assertion lists every rule the descriptor broke at once, so
    /// which of them the runtime names is its own choice: the process ends either way (an M3 Max
    /// under macOS 27.0.1).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal's descriptor validation ends the
    /// process, "MTLTextureDescriptor has invalid pixelFormat (0)", the API validation layer off).
    NotAFormat,
    /// A width, height or depth of 0.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor has width of zero", and the same for a height of zero).
    ZeroExtent,
    /// A height other than 1 for a 1D or buffer texture, or a depth other than 1 for a texture
    /// that is not 3D.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "the texture type
    /// MTLTextureType1D requires that height is 1", and the same for a 2D texture of depth 2).
    ExtentForType,
    /// A cube whose width and height differ.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor width must equal height for cube textures").
    CubeNotSquare,
    /// An array length other than 1 for a type that is not an array, or 0, or more than 2048 (341
    /// for a cube array: Metal stops 342 cubes and more, plain too).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor has arrayLength of zero", where an array length of 2 is made).
    ArrayLength,
    /// A level count of 0, more than the full chain `max`, or more than 1 for a multisample,
    /// 1D, 1D array or 4:2:2 texture.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "requests 5 mipmap
    /// levels, but the dimensions (8, 8, 1) can only support a maxiumum of 4 levels", and the same
    /// for 0).
    MipmapLevelCount {
        /// The full chain's levels.
        max: u64,
    },
    /// A sample count of 0, or other than 1 for a type that is not multisample; for a
    /// multisample type, 1 or a count neither the device nor an Apple GPU takes.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "has sampleCount
    /// set but is using a type that does not allow sampleCount" for 0 and for 2 of a 2D texture,
    /// and "sampleCount (3) is not supported by device").
    SampleCount,
    /// A swizzle other than the default with `ShaderWrite` or `RenderTarget` usage, without
    /// `write_swizzle_enabled` (with it too from `newSharedTextureWithDescriptor:`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Texture swizzling
    /// is not compatable with MTLTextureUsageShaderWrite", where a swizzle of read usage alone is
    /// made).
    SwizzleWithWrites,
    /// An extent or count past what any device can hold (`u32`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "has width
    /// (4294967296) greater than the maximum allowed size of 16384").
    ExtentOverflow,
    /// A texture type the pixel format does not take: Metal stops it (`Invalid`), but a
    /// two-plane extended-range cube or cube array, which plain Metal answers nil for (`Nil`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "depth/stencil
    /// pixel formats require texture type of `MTLTextureType2D[Array]` or
    /// `MTLTextureTypeCube[Array]`"
    /// for a depth texture buffer, and the same for a compressed one).
    TypeForFormat,
    /// `X24_Stencil8`: a format of texture views only. (A texture made in `X32_Stencil8` is a
    /// stencil texture.)
    ///
    /// Basis: unknown (whether plain Metal makes a texture of `X24_Stencil8` from a descriptor is
    /// unmeasured: an Apple GPU's Metal describes none of that format, so `NotAFormat` answers
    /// there first).
    ViewOnlyFormat,
    /// An extent the pixel format does not take: a PVRTC texture that is not square with a
    /// power-of-two side, or a 4:2:2 texture of odd width; and, an [`Error::Nil`]
    /// of plain Metal, a render target of a 4:2:0 or 4:2:2 two-plane YCbCr format over a surface
    /// whose width or height is odd.
    ///
    /// Basis: unknown (plain Metal's answer for a PVRTC texture that is not square and for a 4:2:2
    /// texture of odd width is unmeasured; the odd two-plane render target's nil is the one
    /// measured part).
    ExtentForFormat,
    /// A usage the pixel format does not take: `ShaderAtomic` of `RGB10A8_2P_XR10`, its sRGB
    /// twin or `Unspecialized` (measured on an Apple M4, macOS 27.0.1: Metal's descriptor
    /// validation asserts, plain too for `Unspecialized`), or of a format of which Metal makes
    /// no texture alone (plain Metal asserts).
    ///
    /// Basis: metal-measured (an Apple M4, macOS 27.0.1: plain Metal's descriptor validation
    /// asserts over `ShaderAtomic` of an extended-range format).
    UsageForFormat,
    /// `Unspecialized`, of which Metal makes no texture, an [`Error::Nil`]:
    /// plain Metal answers nil for the texture, a heap's and a buffer's, in every storage mode
    /// and usage (an M3 Max under macOS 26.7, the consumer's report), once every rule Metal stops
    /// passed, where [`Device::heap_texture_size_and_align`] answers 0 bytes at the alignment
    /// Metal's query answers for the texture type. A multisample one is
    /// [`TypeForFormat`](Self::TypeForFormat) and one of `ShaderAtomic` usage
    /// [`UsageForFormat`](Self::UsageForFormat): Metal stops both, plain too.
    ///
    /// [`Device::heap_texture_size_and_align`]: crate::Device::heap_texture_size_and_align
    ///
    /// Basis: metal-measured (an M3 Max, macOS 26.7: plain Metal answers nil for a texture of
    /// `Unspecialized` in every storage mode and usage, as the consumer reported).
    Unspecialized,
    /// A rotation past 4 (Metal's descriptor validation asserts, plain too).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's descriptor validation
    /// ends the process, SIGABRT, "rotation is not a valid MTLTextureRotation", at
    /// `heapTextureSizeAndAlignWithDescriptor:` and at `newTextureWithDescriptor:iosurface:plane:`
    /// of a rotation of 5, where a rotation of 4 beside it answers a size and makes the texture,
    /// the API validation layer off).
    Rotation,
    /// A rotation other than 0 without `framebuffer_only` (Metal's descriptor validation
    /// asserts, plain too).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's descriptor validation
    /// ends the process, SIGABRT, "MTLTextureDescriptor has a non-zero rotation set but is not
    /// marked as framebufferOnly", for a rotation of 1 without it, where a rotation of 0 without it
    /// answers a size and a rotation of 1 with it makes the texture, the API validation layer off).
    RotationWithoutFramebufferOnly,
    /// `framebuffer_only` for a creator that makes no such texture: the device's, a buffer's or
    /// a heap's (plain Metal asserts "framebufferOnly == NO", or on the rotation).
    ///
    /// Basis: metal-documented (Metal asserts "framebufferOnly == NO").
    FramebufferOnly,
    /// A format of which Metal makes no texture from a descriptor alone: the 69 two-plane YCbCr
    /// formats (`metal::format::ycbcr_format`, which it makes over an IOSurface's planes,
    /// [`Buffer::new_texture_with_iosurface_plane`]), `RGBA16Float_XR`, `AnyDepth` and
    /// `AnyStencil`, an [`Error::Nil`]. Plain Metal answers nil (an Apple M4
    /// under macOS 27.0.1; with an M3 Max under 26.7): from the device, a heap and a buffer (the
    /// linear SPI's too), for every texture type the format takes (a two-plane YCbCr format no
    /// multisample one, `AnyDepth` and `AnyStencil` a 2D, 2D array, cube and cube array one, which
    /// [`TypeForFormat`](TextureProblem::TypeForFormat) stops; `RGBA16Float_XR` every type, 2
    /// and 4 samples too), once every rule Metal stops passed, and before a heap's or a buffer's
    /// own rules (storage mode, offset, row pitch, bytes past the buffer).
    /// [`Device::heap_texture_size_and_align`] answers 0 bytes for those, at the alignment its
    /// docs state.
    ///
    /// [`Buffer::new_texture_with_iosurface_plane`]: crate::Buffer::new_texture_with_iosurface_plane
    /// [`Device::heap_texture_size_and_align`]: crate::Device::heap_texture_size_and_align
    ///
    /// Basis: metal-measured (an Apple M4, macOS 27.0.1, and an M3 Max under 26.7: plain Metal
    /// answers nil from the device, a heap and a buffer, for every texture type the format takes).
    NoTextureOfFormat,
    /// A width, height or depth past the largest side Metal makes the type with, `max`: 16384
    /// for every type but a 3D texture and a texture buffer, 2048 for a 3D one, 2^28 texels for a
    /// texture buffer. Plain Metal's own descriptor validation ends its process over one, the API
    /// validation layer off: one texel past the side asserts in `-[MTLTextureDescriptorInternal
    /// validateWithDevice:]` ("MTLTextureDescriptor has width (16385) greater than the maximum
    /// allowed size of 16384"), at `newTextureWithDescriptor:` and at
    /// `heapTextureSizeAndAlignWithDescriptor:` alike, for every texture type (1D, 1D array, 2D, 2D
    /// array, 2D multisample, 2D multisample array, cube, cube array, 3D and texture buffer), in
    /// `RGBA8Unorm`, `R8Unorm`, `Depth32Float` and `BC1_RGBA`, in `Private`, `Shared` and
    /// `Managed` storage, and far past it (32768 on a 2D texture); the side itself is made for
    /// every type (an M3 Max under macOS 27.0.1).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal asserts "MTLTextureDescriptor
    /// has width (16385) greater than the maximum allowed size of 16384", the layer off, where the
    /// side itself is made).
    PastLargestSide {
        /// The largest side of the type.
        max: u64,
    },
}

/// Whether Metal makes no texture of `pf` from a descriptor alone
/// ([`TextureProblem::NoTextureOfFormat`]).
///
/// `R10Unorm_PACKED` and `RG10Unorm_PACKED` are here on the measurement, not on a lack of ours:
/// Metal takes a descriptor naming either in every texture type and `newTextureWithDescriptor:`
/// answers **nil** for every one, while the heap size query answers 0 bytes
/// (`metal::format::Footprint::NoTexture`; an Apple M3 Max under macOS 27.0.1, plain Metal, every
/// type in `Shared` and `Private` storage, the `format-footprints-packed-without-textures`
/// recording). `Managed` storage answers nil with them. The creators that lay out no image
/// storage of their own do **not**: a placement heap, the shared creator, a buffer, an IOSurface
/// plane and `Memoryless` storage each make a texture of both, which [`no_storage_laid_out`]
/// carries and `check_made` carves out of this rule.
pub(super) fn no_texture_alone(pf: PixelFormat) -> bool {
    ycbcr_format(pf).is_some()
        || matches!(footprint(pf), Some(Footprint::NoTexture))
        || matches!(
            pf,
            PixelFormat::RGBA16FloatXR | PixelFormat::AnyDepth | PixelFormat::AnyStencil
        )
}

/// Whether `pf` is a two-plane YCbCr format and `ty` a multisample type, which Metal's
/// validation stops (`check`).
fn ycbcr_multisample(pf: PixelFormat, ty: TextureType) -> bool {
    ycbcr_format(pf).is_some()
        && matches!(
            ty,
            TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
        )
}

/// Whether Metal makes no texture of type `ty` in `pf` by the type rules `check` applies (a
/// two-plane format's multisample or cube texture, a flat compressed format's 3D one, ...): a
/// format's capability probe of that type answers no (`families::formats`), as the texture the
/// probe stands for is not made; Metal's stop and its nil alike.
pub(super) fn type_refused(pf: PixelFormat, ty: TextureType) -> bool {
    ycbcr_multisample(pf, ty)
        || matches!(
            type_for_format(pf, ty),
            Err(Refused::Stops(TextureProblem::TypeForFormat)
                | Refused::Nil(TextureProblem::TypeForFormat))
        )
}

/// Whether plain Metal makes no texture of `pf` at `[width, height]` under `usage` because the
/// sides are not both even: a **one-plane 4:2:2** format (`metal::format::Footprint::OnePlane`
/// with `pairs`, Apple's three `YCBCR8_422_1P*` formats), where the usage has `ShaderWrite` or
/// `RenderTarget`, or is `Unknown`, which is every usage. Metal answers nil and the heap size
/// query 0 bytes; under `ShaderRead` or `PixelFormatView` usage it makes a texture of any sides.
///
/// Measured on an Apple M3 Max under macOS 27.0.1, plain Metal with no API validation layer: for
/// each of the three formats, every pair of 27 sides from 1 to 100 as a 2D texture under nine
/// usages, 6561 descriptors a format, the size query and `newTextureWithDescriptor:` together —
/// the query answers 0 bytes **exactly** where the creator answers nil, and that is exactly the
/// odd-sided descriptors of those usages. A 2D array's and 3D texture's answer alike (8955 more
/// answers over the same formats). Unlike `GBGR422`'s even-width rule, which Metal's validation
/// stops ("MTLPixelFormat422 requires even width"), this one is a nil: the process survives.
fn odd_sided_chroma_pairs(
    pf: PixelFormat,
    [width, height]: [u64; 2],
    ty: TextureType,
    usage: TextureUsage,
) -> bool {
    use metal2vulkan::metal::format::Footprint;
    if !matches!(footprint(pf), Some(Footprint::OnePlane { pairs: true, .. })) {
        return false;
    }
    let writes = usage.is_empty()
        || usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET);
    let rows = match ty {
        TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer => 1,
        _ => height,
    };
    writes && !(width.is_multiple_of(2) && rows.is_multiple_of(2))
}

/// Whether Metal lays a texture of `pf` out in planes of rows rather than tiles, where this
/// crate holds no image of the format (`metal::format::Footprint`): Apple's nine one-plane YCbCr
/// formats and the four `RGB10A8_2P` ones. They take the type rules a two-plane extended-range
/// format takes, which is what the measurement shows ([`type_for_format`]).
fn laid_in_planes(pf: PixelFormat) -> bool {
    matches!(
        footprint(pf),
        Some(Footprint::OnePlane { .. } | Footprint::TwoPlanes)
    )
}

/// Whether `pf` is a two-plane extended-range format (`RGB10A8_2P_XR10` or its sRGB twin).
fn two_planes(pf: PixelFormat) -> bool {
    ExtendedRange::of(pf).is_some_and(|f| !f.holds_metal_bytes())
}

/// Whether plain Metal makes a texture of `pf` from `creator` in `storage` because it lays out no
/// image storage for the format and that creator needs none laid out
/// ([`Footprint::NoTexture`], whose whole membership they are: `R10Unorm_PACKED` and
/// `RG10Unorm_PACKED`, whose packed 10-bit codes cross byte boundaries, so the format has no
/// element size to lay out and Metal lays out none; `metal::format::packed_codes` states the
/// measurement).
///
/// `false` for every other format: this is the one family measured to divide the creators this
/// way, and the one whose heap footprint counts no bytes.
///
/// Which creators, measured on an Apple M3 Max under macOS 27.0.1 (26A434), plain Metal with no
/// API validation layer, one cell per fresh process and four identical runs, with an
/// `RGBA8Unorm` control beside every cell so that a cell's nil could not be read as its storage
/// mode's (`metal::format::packed_codes` carries the figures):
///
/// - A **placement** heap's `newTextureWithDescriptor:offset:` at offset 0 makes one in `Shared`
///   and in `Private` storage, `allocatedSize` 0; an **automatic** heap answers nil in both, as
///   the device's own creator does in `Shared`, `Managed` and `Private`.
/// - `newSharedTextureWithDescriptor:` makes one in `Private` storage, `allocatedSize` 16 where
///   the control's is 16528. It answers nil in `Shared` and `Managed` for the control too, which
///   is the storage mode's rule and not the format's, and `Device::new_shared_texture` answers
///   that nil before this descriptor is checked at all.
/// - A buffer's `newTextureWithDescriptor:offset:bytesPerRow:` makes one over the buffer's own
///   bytes in `Shared` and in `Private` storage, and an IOSurface plane's does too; the texture's
///   `allocatedSize` is the buffer's. A plane's YCbCr creator takes no such format
///   ([`Creator::OverSurface`]).
/// - `Memoryless` storage from the device's own creator makes one, where every mode that holds
///   memory answers nil: a 2D 8x8 texture of either format, 4 of 4 runs under each of
///   `ShaderRead`, `ShaderRead | ShaderWrite`, `ShaderRead | RenderTarget` and `RenderTarget`
///   usage (`conformance/probes/packed10_texture_alone.m`). A memoryless texture holds no memory
///   either, which is why it is the same rule and not a mode's exception.
/// - The heap size query answers 0 bytes rather than a texture, so it keeps the nil
///   (`Device::heap_texture_size_and_align` answers the footprint for it).
///
/// Only a 2D texture is measured per creator, and only `Shared` and `Private` for the buffer's;
/// `check_made` carves every type and mode of a creator that makes one in together, because the
/// answer past the carve is this crate's own `Unsupported(Format, NotImplemented)`, which is true
/// of the format in every shape, where the nil would claim Metal refuses the descriptor.
fn no_storage_laid_out(pf: PixelFormat, creator: Creator, storage: Option<StorageMode>) -> bool {
    matches!(footprint(pf), Some(Footprint::NoTexture))
        && match creator {
            Creator::Device => storage == Some(StorageMode::Memoryless),
            Creator::Shared => storage == Some(StorageMode::Private),
            Creator::Placement | Creator::Buffer => true,
            Creator::OverSurface | Creator::SizeQuery => false,
        }
}

/// Whether `ty` in `pf` is a two-plane extended-range cube or cube array, which plain Metal makes
/// memoryless alone (`check_made`).
pub(super) fn memoryless_two_plane_cube(pf: PixelFormat, ty: TextureType) -> bool {
    two_planes(pf) && matches!(ty, TextureType::Cube | TextureType::CubeArray)
}

/// Whether Metal makes a texture `width` by `height` in `pf` (the module's measured rules).
fn extent_for_format(pf: PixelFormat, width: u64, height: u64) -> Result<(), TextureProblem> {
    use PixelFormat as P;
    let refused = match block_family(pf) {
        P::PVRTCRGB2BPP | P::PVRTCRGB4BPP | P::PVRTCRGBA2BPP | P::PVRTCRGBA4BPP => {
            width != height || !width.is_power_of_two()
        }
        P::GBGR422 | P::BGRG422 => !width.is_multiple_of(2),
        _ => false,
    };
    if refused {
        Err(TextureProblem::ExtentForFormat)
    } else {
        Ok(())
    }
}

/// Whether Metal makes a texture of type `ty` in `pf` (the module's measured rules), and why not.
fn type_for_format(pf: PixelFormat, ty: TextureType) -> Result<(), Refused> {
    use PixelFormat as P;
    use TextureType as T;
    if pf == P::X24Stencil8 {
        return Err(TextureProblem::ViewOnlyFormat.into());
    }
    let one_dimensional = matches!(ty, T::Type1D | T::Type1DArray);
    let multisample = matches!(ty, T::Type2DMultisample | T::Type2DMultisampleArray);
    // A depth or stencil format takes no texture buffer either, and `AnyDepth` and `AnyStencil`
    // no multisample type: plain Metal stops each ("depth/stencil pixel formats require texture
    // type of MTLTextureType2D[Array] or MTLTextureTypeCube[Array]", "does not support
    // multisample textures"; `Depth32Float`, `AnyDepth` and `AnyStencil`).
    let any_depth = matches!(pf, P::AnyDepth | P::AnyStencil);
    let refused = if is_depth_stencil(pf) || any_depth {
        one_dimensional || ty == T::Type3D || ty == T::TextureBuffer || (any_depth && multisample)
    } else if two_planes(pf) || laid_in_planes(pf) {
        // Measured on an Apple M4 (macOS 27.0.1): Metal answers nil for a cube and a cube array
        // (plain and validated alike) and its validation asserts on a multisample
        // texture; it makes 1D, 2D, 2D array and 3D ones, mipmapped too. Apple's nine one-plane
        // YCbCr formats and the four `RGB10A8_2P` ones answer alike on an M3 Max under 27.0.1
        // (the `format-footprints-one-plane-ycbcr` and `-two-plane-ten-bit` recordings: a size
        // and no texture for a cube and a cube array, a texture for the other six types).
        if matches!(ty, T::Cube | T::CubeArray) {
            return Err(Refused::Nil(TextureProblem::TypeForFormat));
        }
        multisample
    } else if matches!(pf, P::GBGR422 | P::BGRG422) {
        ty != T::Type2D
    // The padded 10-bit formats take no multisample texture: Metal's validation asserts
    // "MTLTextureDescriptor pixelFormat (MTLPixelFormatR10Unorm_X6) does not support multisample
    // textures" on one of 1 and of 4 samples alike, re-read on an Apple M3 Max under macOS
    // 27.0.1 (`R10Unorm_X6` and `RG10Unorm_X12`, a `Private` 8x8 render target), and their flags
    // list neither MSAA nor resolve.
    // Metal's texture descriptor validation asserts on one of 4 samples (`R16Bfloat`,
    // `RG16Bfloat`, `RGBA16Bfloat`; an Apple M4 under macOS 27.0.1).
    } else if padded_words(pf) || ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat) {
        multisample
    } else if block_extent(pf).is_some() {
        let flat = matches!(
            block_family(pf),
            P::ETC2RGB8
                | P::ETC2RGB8A1
                | P::EACRGBA8
                | P::EACR11Unorm
                | P::EACRG11Unorm
                | P::PVRTCRGB2BPP
                | P::PVRTCRGB4BPP
                | P::PVRTCRGBA2BPP
                | P::PVRTCRGBA4BPP
        );
        one_dimensional || (flat && ty == T::Type3D)
    } else {
        false
    };
    if refused {
        Err(TextureProblem::TypeForFormat.into())
    } else {
        Ok(())
    }
}

/// What a descriptor's check takes of the device ([`TextureDescriptor::check_made`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TextureRules {
    /// The sample counts its textures take (`supportsTextureSampleCount:`).
    pub(super) sample_counts: SampleCountFlags,
}

/// The sample counts an Apple GPU's textures take: 1, 2 and 4 (an M3 Max under macOS 26.7 and an
/// M4 under 27.0.1 make 2 and 4 samples and stop 3, 8 and 16).
pub(super) const APPLE_SAMPLE_COUNTS: SampleCountFlags = SampleCountFlags::from_raw(0b111);

/// The widest texture buffer Metal makes: 2^28 texels ([`largest_side`]).
const MAX_TEXTURE_BUFFER_WIDTH: u64 = 1 << 28;

/// The largest width, height and depth Metal makes a texture of type `ty` with: 16384 for every
/// type but a 3D texture and a texture buffer, 2048 for a 3D one (Metal's validation stops a
/// larger one, "greater than the maximum allowed size", whatever the format; an M4 under macOS
/// 27.0.1), and 2^28 for a texture buffer's width (268, 435, 456 made, one more stopped, `R8Unorm`
/// and `RGBA8Unorm`, from every creator and the heap size query; an M3 Max under 26.7 and an M4
/// under 27.0.1).
pub(super) fn largest_side(ty: TextureType) -> Option<u64> {
    use TextureType as T;
    match ty {
        T::Type1D
        | T::Type1DArray
        | T::Type2D
        | T::Type2DArray
        | T::Type2DMultisample
        | T::Type2DMultisampleArray
        | T::Cube
        | T::CubeArray => Some(16384),
        T::Type3D => Some(2048),
        T::TextureBuffer => Some(MAX_TEXTURE_BUFFER_WIDTH),
    }
}
