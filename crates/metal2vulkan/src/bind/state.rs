//! What is bound, as Metal states it: a texture's facts ([`TextureFacts`]) and their
//! [`TextureShape`], a sampler's descriptor ([`SamplerDescriptor`]) and its [`SamplerShape`], and
//! a compute encoder's stage-in region ([`StageInRegion`]).

use std::hash::Hash;

use crate::metal;

/// A compute encoder's stage-in region (`setStageInRegion:`, `MTLRegion`), in threads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInRegion {
    /// The region's origin (`MTLRegion.origin`): `[[grid_origin]]`.
    pub origin: [u32; 3],
    /// The region's size (`MTLRegion.size`): `[[grid_size]]`.
    pub size: [u32; 3],
}

impl StageInRegion {
    /// The region of `size` threads from `origin`.
    pub fn new(origin: [u32; 3], size: [u32; 3]) -> StageInRegion {
        StageInRegion { origin, size }
    }
}

impl TextureFacts {
    /// The facts of an image texture that can change the variant, specialization or views a slot
    /// binds it through (its view facts): the type, the pixel format, the sample count, the
    /// usage, whether it is sparse, its level count, which decides a storage slot's level views
    /// and pushed level count, its swizzle, its address-mode map and its
    /// [`offset_base_axes`](Self::offset_base_axes). Its array length only changes the view's
    /// subresource range, which is every layer.
    ///
    /// The level count is stated whatever the usage: plain Metal writes a texture made without
    /// `ShaderWrite` at every level, through every write form, as one made with it.
    pub fn shape(&self) -> TextureShape {
        TextureShape::Texture {
            texture_type: self.texture_type,
            pixel_format: self.pixel_format,
            sample_count: self.sample_count,
            usage: self.usage,
            mipmap_level_count: self.mipmap_level_count,
            sparse: self.sparse,
            swizzle: self.swizzle,
            address_modes: self.address_modes,
            offset_base_axes: self.offset_base_axes(),
        }
    }
}

impl SamplerDescriptor {
    /// The facts of this descriptor that decide whether a sampler use takes it: the reduction
    /// mode and comparison, whether each value the Vulkan sampler cannot take as is is present,
    /// whether its border is opaque black, and whether it filters linearly
    /// (`metal::sampler::check_sampler` reads them).
    pub fn shape(&self) -> SamplerShape {
        use metal::{SamplerAddressMode as A, SamplerBorderColor as B};
        let modes = [
            self.s_address_mode,
            self.t_address_mode,
            self.r_address_mode,
        ];
        // `ClampToZero` reads the border color as `ClampToBorderColor` does.
        let bordered = modes
            .iter()
            .any(|m| matches!(m, A::ClampToZero | A::ClampToBorderColor));
        let (lo, hi) = (self.lod_min_clamp, self.lod_max_clamp);
        SamplerShape {
            reduction: self.reduction_mode,
            compare: self.compare_function != metal::CompareFunction::Never,
            anisotropic: self.max_anisotropy > 1,
            mirror_clamp_to_edge: modes.contains(&A::MirrorClampToEdge),
            valid_lod_clamps: !(lo.is_nan() || hi.is_nan() || lo > hi),
            opaque_black_border: bordered && self.border_color == B::OpaqueBlack,
            linear: self.min_filter == metal::SamplerMinMagFilter::Linear
                || self.mag_filter == metal::SamplerMinMagFilter::Linear
                || self.mip_filter == metal::SamplerMipFilter::Linear,
            cubic: self.min_filter == metal::SamplerMinMagFilter::Cubic
                || self.mag_filter == metal::SamplerMinMagFilter::Cubic,
        }
    }
}

/// The view facts of an image texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureFacts {
    pub texture_type: metal::TextureType,
    pub pixel_format: metal::PixelFormat,
    pub sample_count: u32,
    pub usage: metal::TextureUsage,
    pub array_length: u32,
    pub mipmap_level_count: u32,
    /// `MTLTexture.isSparse`: the texture was made from a sparse heap, so regions of it may be
    /// unmapped. A function that observes the residency of its accesses of the texture
    /// (`sparse_color::resident()`, `RuntimeFact::SparseTexture`) answers a texture that is not
    /// sparse on every device, and a sparse one only where the module has its sparse form
    /// (`shaderResourceResidency`); the binder refuses it elsewhere (`Feature::SparseTexture`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub sparse: bool,
    /// `MTLTexture.swizzle`: the source of each channel the texture's reads and samples return,
    /// border texels included. A sample emulated by fetches reads its border taps through it, and
    /// an opaque-black border the device replaces is checked against it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzle: metal::TextureSwizzleChannels,
    /// The texture's address-mode map (`metal::sampler::AddressModeMap`, the runtime's
    /// extension): a slot a constexpr sampler samples takes it through the pipeline variant
    /// (`VariantKey::address_maps`) where it is not the identity.
    #[cfg_attr(feature = "serde", serde(default))]
    pub address_modes: metal::sampler::AddressModeMap,
    /// The level of the texture whose levels a view presents at which the view's first level
    /// lies (`MTLTexture.parentRelativeLevel`): 0 for a texture that is not a view, and for a
    /// view that starts at its parent's base level.
    #[cfg_attr(feature = "serde", serde(default))]
    pub parent_relative_level: u32,
    /// The base-level size, `[width, height, depth]`, of the texture whose levels a view presents
    /// (`MTLTexture.parentTexture`'s, the texture whose image holds the view's texels), where
    /// [`parent_relative_level`](Self::parent_relative_level) is not 0; unread where it is.
    /// Metal's texel-offset rule reads it, not the view's own: a slot whose function places an
    /// offset by that rule specializes on the axes where the two differ in being powers of two
    /// (`SpecKey::OffsetBaseAxes`, [`TextureFacts::offset_base_axes`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub parent_base_size: [u32; 3],
}

impl TextureFacts {
    /// A single-sample, single-level, single-layer texture.
    pub fn new(texture_type: metal::TextureType, pixel_format: metal::PixelFormat) -> Self {
        TextureFacts {
            texture_type,
            pixel_format,
            sample_count: 1,
            usage: metal::TextureUsage::SHADER_READ,
            array_length: 1,
            mipmap_level_count: 1,
            sparse: false,
            swizzle: metal::TextureSwizzleChannels::DEFAULT,
            address_modes: metal::sampler::AddressModeMap::IDENTITY,
            parent_relative_level: 0,
            parent_base_size: [0; 3],
        }
    }

    /// The `SpecKey::OffsetBaseAxes` of a slot bound to this texture: bit `a` for each plane axis
    /// `a` of its type (a cube's and a texture buffer's none, which take no texel offset) on which
    /// it is a view whose first level lies above its parent's base level
    /// ([`parent_relative_level`](Self::parent_relative_level)), whose parent's base-level size
    /// ([`parent_base_size`](Self::parent_base_size)) is not a power of two where its own,
    /// `max(1, parent >> parent_relative_level)`, is, and some level of which has size 2 or 4.
    /// Metal drops a texel offset on a repeating or mirrored axis whose level size lies in
    /// `[2, 7]` and whose base-level size, the parent's, is not a power of two: such an axis's
    /// levels, powers of two, drop it at sizes 2 and 4 under that parent and keep it under the
    /// view's own base. On every other axis the view's own base-level size decides as the parent's
    /// does, so the bit is 0 and the texture shares its pipelines.
    pub fn offset_base_axes(&self) -> u8 {
        use metal::TextureType as T;
        let level = self.parent_relative_level;
        let axes = match self.texture_type {
            _ if level == 0 => 0,
            T::Type1D | T::Type1DArray => 1,
            T::Type2D | T::Type2DArray | T::Type2DMultisample | T::Type2DMultisampleArray => 2,
            T::Type3D => 3,
            _ => 0,
        };
        let last = self.mipmap_level_count.max(1) - 1;
        let mut bits = 0;
        for (a, &parent) in self.parent_base_size.iter().enumerate().take(axes) {
            let own = parent.checked_shr(level).unwrap_or(0).max(1);
            let smallest = own.checked_shr(last).unwrap_or(0).max(1);
            let small = own >= 2 && smallest <= 4;
            if parent != 0 && !parent.is_power_of_two() && own.is_power_of_two() && small {
                bits |= 1 << a;
            }
        }
        bits
    }
}

/// The facts of an `MTLSamplerDescriptor`; `metal::sampler::vk_sampler` maps one to a
/// `vk::SamplerDesc`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerDescriptor {
    pub min_filter: metal::SamplerMinMagFilter,
    pub mag_filter: metal::SamplerMinMagFilter,
    pub mip_filter: metal::SamplerMipFilter,
    pub s_address_mode: metal::SamplerAddressMode,
    pub t_address_mode: metal::SamplerAddressMode,
    pub r_address_mode: metal::SamplerAddressMode,
    pub border_color: metal::SamplerBorderColor,
    pub normalized_coordinates: bool,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub lod_average: bool,
    pub max_anisotropy: u32,
    /// `Never` (the Metal default) on a sampler used without comparison.
    pub compare_function: metal::CompareFunction,
    pub reduction_mode: metal::SamplerReductionMode,
    /// `lodBias`: any value, NaN and the infinities included, as Metal takes it, its validation
    /// layer too. What it changes is the GPU's (`metal::sampler::SamplerUse::lod_bias`). An Apple
    /// GPU of the families the runtime claims, which lists the bias under Apple10 alone, samples
    /// every level as with a bias of 0 (an Apple M3 Max on macOS 27.0.1, Apple9: implicit,
    /// biased, explicit-level, gradient and clamped samples in fragment functions and kernels,
    /// float and integer textures, every mip filter, biases from -1000 to 1000): there the Vulkan
    /// sampler's `mipLodBias` is 0 whatever it holds. A Mac with an Intel or AMD GPU adds it to
    /// every sample's level and clamps the sum into the texture's levels (an AMD Radeon Pro 5300M
    /// and an Intel UHD 630: explicit-level, gradient and implicit samples): there it is the
    /// device's `mipLodBias`, within the device's limit
    /// ([`device_lod_bias`](crate::metal::sampler::device_lod_bias)). Metal makes another sampler
    /// state for another bias.
    #[cfg_attr(feature = "serde", serde(default))]
    pub lod_bias: f32,
}

impl SamplerDescriptor {
    /// `MTLSamplerDescriptor`'s defaults: nearest filtering without mipmaps, clamp to edge,
    /// transparent black borders, normalized coordinates, LOD clamps `[0, FLT_MAX]`, no
    /// anisotropy, no comparison, weighted-average reduction, and no LOD bias. The fields are
    /// public; set the ones that differ.
    pub fn new() -> SamplerDescriptor {
        SamplerDescriptor {
            min_filter: metal::SamplerMinMagFilter::Nearest,
            mag_filter: metal::SamplerMinMagFilter::Nearest,
            mip_filter: metal::SamplerMipFilter::NotMipmapped,
            s_address_mode: metal::SamplerAddressMode::ClampToEdge,
            t_address_mode: metal::SamplerAddressMode::ClampToEdge,
            r_address_mode: metal::SamplerAddressMode::ClampToEdge,
            border_color: metal::SamplerBorderColor::TransparentBlack,
            normalized_coordinates: true,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            lod_average: false,
            max_anisotropy: 1,
            compare_function: metal::CompareFunction::Never,
            reduction_mode: metal::SamplerReductionMode::WeightedAverage,
            lod_bias: 0.0,
        }
    }
}

impl Default for SamplerDescriptor {
    fn default() -> SamplerDescriptor {
        SamplerDescriptor::new()
    }
}

/// The level count of a texture shape stated without one.
#[cfg(feature = "serde")]
fn one_level() -> u32 {
    1
}

/// The shape of a bound texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureShape {
    Texture {
        texture_type: metal::TextureType,
        pixel_format: metal::PixelFormat,
        sample_count: u32,
        /// Which views the texture admits: a heap writes a sampled view only of a
        /// `ShaderRead` texture and a storage view only of a `ShaderWrite` or `BlockWritesOnly`
        /// one.
        usage: metal::TextureUsage,
        /// The texture's `mipmapLevelCount` ([`TextureFacts::shape`]): a storage slot's level
        /// views view each level, and a neutral past the last.
        #[cfg_attr(feature = "serde", serde(default = "one_level"))]
        mipmap_level_count: u32,
        /// The texture is sparse (`MTLTexture.isSparse`, [`TextureFacts::sparse`]): it selects
        /// the sparse form of a slot whose residency the function observes, or the binder's refusal
        /// where the module has none.
        #[cfg_attr(feature = "serde", serde(default))]
        sparse: bool,
        /// The texture's swizzle ([`TextureFacts::swizzle`]): the border taps of a sample emulated
        /// by fetches read through it, and an opaque-black border the device replaces is checked
        /// against it.
        #[cfg_attr(feature = "serde", serde(default))]
        swizzle: metal::TextureSwizzleChannels,
        /// The texture's address-mode map ([`TextureFacts::address_modes`]): a slot a constexpr
        /// sampler samples compiles under it where it is not the identity
        /// (`VariantKey::address_maps`).
        #[cfg_attr(feature = "serde", serde(default))]
        address_modes: metal::sampler::AddressModeMap,
        /// The plane axes on which the base-level size Metal's texel-offset rule reads, a view's
        /// parent's, is not a power of two where the texture's own is
        /// ([`TextureFacts::offset_base_axes`]): a slot whose function places an offset by that
        /// rule specializes on them (`SpecKey::OffsetBaseAxes`).
        #[cfg_attr(feature = "serde", serde(default))]
        offset_base_axes: u8,
    },
}

impl TextureShape {
    /// The texture's swizzle.
    pub(crate) fn swizzle(&self) -> metal::TextureSwizzleChannels {
        let TextureShape::Texture { swizzle, .. } = *self;
        swizzle
    }
}

/// The shape of a sampler: the modes that are runtime facts, and the descriptor values that decide
/// whether Vulkan can express it (checked by `metal::sampler::check_sampler` when
/// `metal::sampler::vk_sampler` makes it).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerShape {
    pub reduction: metal::SamplerReductionMode,
    pub compare: bool,
    /// `max_anisotropy > 1` (`samplerAnisotropy`).
    pub anisotropic: bool,
    /// Some axis uses `MirrorClampToEdge` (`samplerMirrorClampToEdge`).
    pub mirror_clamp_to_edge: bool,
    /// The LOD clamps are numbers and `lod_min_clamp <= lod_max_clamp`.
    pub valid_lod_clamps: bool,
    /// The border the sampler replaces texels with is opaque black: whether it reads exactly
    /// through a texture's view depends on the view (`metal::sampler::check_border_view`).
    pub opaque_black_border: bool,
    /// Some filter (minification, magnification, or mip) is linear: whether the sampler needs
    /// the view format's filter features depends on its use and the view
    /// (`metal::sampler::vk_filtering`), which the binder checks where the sampler meets textures
    /// through separate descriptors.
    pub linear: bool,
    /// The min or the mag filter is `Cubic`: Metal samples through the sampler with its bicubic
    /// filter whatever the level, which a function emulates under its pipeline variant's
    /// cubic samplers (`VariantKey::cubic_samplers`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub cubic: bool,
}
