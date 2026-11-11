//! Apple's private (SPI) `MTLDevice` properties.
//!
//! An `MTLDevice` answers far more properties than Metal's headers name: limits such as
//! `maxComputeBuffers` and `gpuCoreCount`, and flags such as `supportsTileShaders` (our probe
//! lists 366 integer and `BOOL` getters on an Apple M4). Private Metal is Metal: a program that
//! asks one gets an answer. [`Device::private_properties`] answers the ones [`PrivateCount`] and
//! [`PrivateFlag`] name, each typed by its source ([`PrivateAnswer`]):
//!
//! - **The device's** ([`PrivateAnswer::Device`]), where the runtime knows what the property
//!   promises. The answer is read from what the runtime acts on: a binding table's entries from
//!   the encoders' tables, a texture's sides and layers from the descriptor's rules, the device's
//!   Vulkan limits through the GPU-family rows that judge them (`Device::family_rows`), and a
//!   capability from the family row or the runtime piece that provides it. Metal's measured
//!   answer rides beside it, so an answer below Metal's shows what the device lacks.
//! - **None of the device's** ([`PrivateAnswer::Metal`]), where no document says what the
//!   property promises and nothing the device reports maps to it: [`PrivateAnswer::value`] gives
//!   none, and Metal's measured answer, on Apple's GPU, rides alone
//!   ([`PrivateAnswer::metal`]), never as this device's (It was the value given).
//!   [`Device::private_properties`] announces those properties once
//!   (`metal2vulkan-degrade-private-properties-unknown`).
//!
//! Metal's measured answers ([`PrivateAnswer::metal`]) are an Apple M3 Max's at the release every
//! device answers as (`metal_release.rs`; each getter in a process of its own, none faulted). An
//! Apple M4 at that release answers every one alike but `gpuCoreCount` (10) and
//! `supportsComputeCompressedTextureWrite` (yes). An answer conflates the chip and the release:
//! Metal's `gpuCoreCount` is the measured chip's, and the device's own is its driver's
//! ([`DeviceProperties::gpu_core_count`]), where it reports one.
//!
//! Metal's public properties are [`DeviceProperties`]'s, not repeated here.

use crate::argument_table::{BUFFER_SLOTS, SAMPLER_SLOTS, TEXTURE_SLOTS};
use crate::capabilities::{DeviceProperties, RATE_MAP_LAYERS};
use crate::compute::THREADGROUP_SLOTS;
use crate::families::{
    FamilyRow, FeatureRow, FormatCapabilities, FormatCapability, GpuFamily, LimitRow, Refusal,
    RowAudit, RowState,
};
use crate::linear_plan::LINEAR_ALIGNMENT;
use crate::render_state::MAX_VIEWPORTS;
use crate::texture_descriptor::{largest_side, MAX_ARRAY_LENGTH};
use crate::{Degradation, Device};
use metal2vulkan::device::{MetalDeviceProperties, VkFeature};
use metal2vulkan::error::{Capability, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType};

/// The family whose rows judge the device for the measured answers: Apple9, the newest family
/// both measured chips are (an M3 Max and an M4 answer `supportsFamily:` yes for 1009 and no for
/// 1010).
pub(crate) const MEASURED_FAMILY: GpuFamily = GpuFamily::Apple9;

/// How the runtime answers a private property on a device (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrivateAnswer<T> {
    /// The runtime's answer on the device, read from what the runtime acts on, beside Metal's
    /// measured answer. An answer below Metal's (or no where Metal
    /// answers yes) is what the device lacks.
    Device {
        /// The runtime's answer.
        answer: T,
        /// Metal's measured answer.
        metal: T,
    },
    /// No answer of the device's: no document describes the property and nothing the device
    /// reports maps to it. It holds Metal's measured answer, on
    /// Apple's GPU, which is not this device's.
    Metal(T),
}

impl<T: Copy> PrivateAnswer<T> {
    /// The device's answer, where it has one: `None` for [`PrivateAnswer::Metal`], whose value
    /// is Apple's GPU's, never this device's (the module docs). Previously it gave Metal's.
    pub fn value(self) -> Option<T> {
        match self {
            PrivateAnswer::Device { answer, .. } => Some(answer),
            PrivateAnswer::Metal(_) => None,
        }
    }

    /// Metal's measured answer.
    pub fn metal(self) -> T {
        match self {
            PrivateAnswer::Device { metal, .. } | PrivateAnswer::Metal(metal) => metal,
        }
    }
}

/// Apple's private `NSUInteger` device properties the runtime answers, one per selector
/// ([`selector`](Self::selector)). Each says where its answer comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PrivateCount {
    /// `gpuCoreCount`: the device's where its driver reports its cores
    /// ([`DeviceProperties::gpu_core_count`]: NVIDIA's streaming multiprocessors, AMD's compute
    /// units, Arm's shader cores), else none (Intel's drivers, MoltenVK, lavapipe).
    GpuCoreCount,
    /// `maxVertexAmplificationCount`: Metal's (2 where the public
    /// `supportsVertexAmplificationCount:` takes 1 to 8, so not the views a pipeline may draw). No
    /// Vulkan limit means it: on Apple's own GPUs MoltenVK reports `maxMultiviewViewCount` 32.
    MaxVertexAmplificationCount,
    /// `deviceLinearTextureAlignmentBytes`: the device's: the largest of its
    /// `minimumLinearTextureAlignmentForPixelFormat:` answers over the formats a buffer can back
    /// ([`Device::minimum_linear_texture_alignment_for_pixel_format`]), at least the 16 bytes
    /// Metal answers on an Apple GPU: 256 on Mesa's RADV, 64 on lavapipe.
    DeviceLinearTextureAlignmentBytes,
    /// `linearTextureAlignmentBytes`: as
    /// [`DeviceLinearTextureAlignmentBytes`](Self::DeviceLinearTextureAlignmentBytes).
    LinearTextureAlignmentBytes,
    /// `iosurfaceTextureAlignmentBytes`: the device's: 16, the alignment the runtime takes an
    /// IOSurface plane's offset and rows at on every device.
    IosurfaceTextureAlignmentBytes,
    /// `iosurfaceReadOnlyTextureAlignmentBytes`: as
    /// [`IosurfaceTextureAlignmentBytes`](Self::IosurfaceTextureAlignmentBytes): a plane read
    /// only is placed by the same rule.
    IosurfaceReadOnlyTextureAlignmentBytes,
    /// `maxVertexBuffers`: the device's (the buffer table's entries, the `BufferArgumentEntries`
    /// row's bound where it refuses).
    MaxVertexBuffers,
    /// `maxFragmentBuffers`: as [`MaxVertexBuffers`](Self::MaxVertexBuffers).
    MaxFragmentBuffers,
    /// `maxComputeBuffers`: as [`MaxVertexBuffers`](Self::MaxVertexBuffers).
    MaxComputeBuffers,
    /// `maxTileBuffers`: as [`MaxVertexBuffers`](Self::MaxVertexBuffers): a tile function's
    /// table, wherever the render encoder runs one ([`SupportsTileShaders`](PrivateFlag::SupportsTileShaders)
    /// says whether it runs all Metal does).
    MaxTileBuffers,
    /// `maxVertexTextures`: the device's (the texture table's entries, the
    /// `TextureArgumentEntries` row's bound where it refuses).
    MaxVertexTextures,
    /// `maxFragmentTextures`: as [`MaxVertexTextures`](Self::MaxVertexTextures).
    MaxFragmentTextures,
    /// `maxComputeTextures`: as [`MaxVertexTextures`](Self::MaxVertexTextures).
    MaxComputeTextures,
    /// `maxTileTextures`: as [`MaxVertexTextures`](Self::MaxVertexTextures), a tile function's
    /// table.
    MaxTileTextures,
    /// `maxVertexSamplers`: the device's (the sampler table's entries, the
    /// `SamplerArgumentEntries` row's bound where it refuses).
    MaxVertexSamplers,
    /// `maxFragmentSamplers`: as [`MaxVertexSamplers`](Self::MaxVertexSamplers).
    MaxFragmentSamplers,
    /// `maxComputeSamplers`: as [`MaxVertexSamplers`](Self::MaxVertexSamplers).
    MaxComputeSamplers,
    /// `maxTileSamplers`: as [`MaxVertexSamplers`](Self::MaxVertexSamplers), a tile function's
    /// table.
    MaxTileSamplers,
    /// `maxComputeLocalMemorySizes`: the device's (the compute encoder's threadgroup memory
    /// table's entries, every one an arena of the threadgroup's memory).
    MaxComputeLocalMemorySizes,
    /// `maxComputeThreadgroupMemory`: Metal's (not the public `maxThreadgroupMemoryLength`,
    /// which is half of it). No Vulkan limit means it: on Apple's own GPUs MoltenVK reports
    /// `maxComputeSharedMemorySize` 32768, the public length.
    MaxComputeThreadgroupMemory,
    /// `maxComputeThreadgroupMemoryAlignmentBytes`: the device's (Metal's alignment where the
    /// `ThreadgroupMemoryAlignment` row is exact: lengths are taken in its multiples).
    MaxComputeThreadgroupMemoryAlignmentBytes,
    /// `maxPredicatedNestingDepth`: the device's, Metal's `2^32 - 1`, as command predication
    /// runs on every device (ifs nest without limit, loops three deep).
    MaxPredicatedNestingDepth,
    /// `maxRasterizationRateLayerCount`: the device's (the layers a rasterization rate map
    /// takes, `supportsRasterizationRateMapWithLayerCount:`).
    MaxRasterizationRateLayerCount,
    /// `maxTextureWidth2D`: the device's (the descriptor's largest side of a 2D texture, the
    /// `TextureSize2D` row's bound where it refuses).
    MaxTextureWidth2D,
    /// `maxTextureHeight2D`: as [`MaxTextureWidth2D`](Self::MaxTextureWidth2D).
    MaxTextureHeight2D,
    /// `maxTextureLayers`: the device's (the descriptor's largest array length, the
    /// `TextureArrayLayers` row's bound where it refuses).
    MaxTextureLayers,
    /// `maxTotalComputeThreadsPerThreadgroup`: the device's (the compile layer's
    /// `maxTotalThreadsPerThreadgroup`, the `ThreadsPerThreadgroup` row's bound where it
    /// refuses).
    MaxTotalComputeThreadsPerThreadgroup,
    /// `maxViewportCount`: the device's (the viewports a render encoder takes, one without
    /// `multiViewport`, at most the device's `maxViewports`).
    MaxViewportCount,
}

impl PrivateCount {
    /// Every count, in declaration order.
    pub const ALL: [PrivateCount; 28] = {
        use PrivateCount as C;
        [
            C::GpuCoreCount,
            C::MaxVertexAmplificationCount,
            C::DeviceLinearTextureAlignmentBytes,
            C::LinearTextureAlignmentBytes,
            C::IosurfaceTextureAlignmentBytes,
            C::IosurfaceReadOnlyTextureAlignmentBytes,
            C::MaxVertexBuffers,
            C::MaxFragmentBuffers,
            C::MaxComputeBuffers,
            C::MaxTileBuffers,
            C::MaxVertexTextures,
            C::MaxFragmentTextures,
            C::MaxComputeTextures,
            C::MaxTileTextures,
            C::MaxVertexSamplers,
            C::MaxFragmentSamplers,
            C::MaxComputeSamplers,
            C::MaxTileSamplers,
            C::MaxComputeLocalMemorySizes,
            C::MaxComputeThreadgroupMemory,
            C::MaxComputeThreadgroupMemoryAlignmentBytes,
            C::MaxPredicatedNestingDepth,
            C::MaxRasterizationRateLayerCount,
            C::MaxTextureWidth2D,
            C::MaxTextureHeight2D,
            C::MaxTextureLayers,
            C::MaxTotalComputeThreadsPerThreadgroup,
            C::MaxViewportCount,
        ]
    };

    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        self.measured().0
    }

    /// Metal's measured answer for the property on its Apple GPU (the module docs).
    pub(crate) fn metal(self) -> u64 {
        self.measured().1
    }

    /// The selector and Metal's measured answers on an Apple M3 Max: under macOS 26.7, and under
    /// 27.0.1.
    fn measured(self) -> (&'static str, u64) {
        use PrivateCount as C;
        match self {
            C::GpuCoreCount => ("gpuCoreCount", 40),
            C::MaxVertexAmplificationCount => ("maxVertexAmplificationCount", 2),
            C::DeviceLinearTextureAlignmentBytes => ("deviceLinearTextureAlignmentBytes", 16),
            C::LinearTextureAlignmentBytes => ("linearTextureAlignmentBytes", 16),
            C::IosurfaceTextureAlignmentBytes => ("iosurfaceTextureAlignmentBytes", 16),
            C::IosurfaceReadOnlyTextureAlignmentBytes => {
                ("iosurfaceReadOnlyTextureAlignmentBytes", 16)
            }
            C::MaxVertexBuffers => ("maxVertexBuffers", 31),
            C::MaxFragmentBuffers => ("maxFragmentBuffers", 31),
            C::MaxComputeBuffers => ("maxComputeBuffers", 31),
            C::MaxTileBuffers => ("maxTileBuffers", 31),
            C::MaxVertexTextures => ("maxVertexTextures", 128),
            C::MaxFragmentTextures => ("maxFragmentTextures", 128),
            C::MaxComputeTextures => ("maxComputeTextures", 128),
            C::MaxTileTextures => ("maxTileTextures", 128),
            C::MaxVertexSamplers => ("maxVertexSamplers", 16),
            C::MaxFragmentSamplers => ("maxFragmentSamplers", 16),
            C::MaxComputeSamplers => ("maxComputeSamplers", 16),
            C::MaxTileSamplers => ("maxTileSamplers", 16),
            C::MaxComputeLocalMemorySizes => ("maxComputeLocalMemorySizes", 31),
            C::MaxComputeThreadgroupMemory => ("maxComputeThreadgroupMemory", 65536),
            C::MaxComputeThreadgroupMemoryAlignmentBytes => {
                ("maxComputeThreadgroupMemoryAlignmentBytes", 16)
            }
            C::MaxPredicatedNestingDepth => ("maxPredicatedNestingDepth", 0xffff_ffff),
            C::MaxRasterizationRateLayerCount => ("maxRasterizationRateLayerCount", 2),
            C::MaxTextureWidth2D => ("maxTextureWidth2D", 16384),
            C::MaxTextureHeight2D => ("maxTextureHeight2D", 16384),
            C::MaxTextureLayers => ("maxTextureLayers", 2048),
            C::MaxTotalComputeThreadsPerThreadgroup => {
                ("maxTotalComputeThreadsPerThreadgroup", 1024)
            }
            C::MaxViewportCount => ("maxViewportCount", 16),
        }
    }
}

/// Apple's private `BOOL` device properties the runtime answers, one per selector
/// ([`selector`](Self::selector)). Each says where its answer comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PrivateFlag {
    /// `isFramebufferReadSupported`: the device's (the `ProgrammableBlending` row: a fragment
    /// function reads its color attachments): yes where the device orders color input
    /// attachments in rasterization order, or orders a pixel interlock and renders every color
    /// format in its own form (MoltenVK and RADV); no where the device renders a format only
    /// through the view of its word (`R8Unorm_sRGB` and `RG8Unorm_sRGB` on NVIDIA and ANV), whose
    /// fetch is not built, and where it has neither form.
    IsFramebufferReadSupported,
    /// `isLargeMRTSupported`: Metal's.
    IsLargeMrtSupported,
    /// `isRGB10A2GammaSupported`: the device's: whether the runtime filters a texture of
    /// `RGB10A2Unorm_sRGB` on it ([`Device::pixel_format_capabilities`]): yes on every device,
    /// whose function filters the texture's held words as Metal does (Metal's answer on an Apple
    /// M3 Max under macOS 27.0.1: yes).
    IsRgb10a2GammaSupported,
    /// `supportsBufferWithAddressRanges`: the device's, no (the runtime makes no buffer over
    /// address ranges).
    SupportsBufferWithAddressRanges,
    /// `supportsBufferWithIOSurface`: the device's, no (the runtime makes no buffer over an
    /// IOSurface).
    SupportsBufferWithIosurface,
    /// `supportsCommandBufferJump`: the device's, yes, as command predication runs on every
    /// device.
    SupportsCommandBufferJump,
    /// `supportsComputeCompressedTextureWrite`: Metal's (whether a kernel's writes keep a
    /// texture's lossless compression, which Vulkan leaves to the driver, is not documented).
    SupportsComputeCompressedTextureWrite,
    /// `supportsConditionalLoadStore`: Metal's (no document or AIR form names it).
    SupportsConditionalLoadStore,
    /// `supportsDynamicAttributeStride`: the device's (the `DynamicAttributeStride` row).
    SupportsDynamicAttributeStride,
    /// `supportsExtendedXR10Formats`: the device's (the `ExtendedRangePixelFormats` row: the
    /// runtime filters a texture of each XR10 format).
    SupportsExtendedXr10Formats,
    /// `supportsFloat16BCubicFiltering`: the device's, yes: the runtime filters a half-float
    /// texture through a bicubic sampler on every device, by fetches, its sums Metal's
    /// accumulators. The forms it refuses (a bicubic sampler with a mip filter or in an argument
    /// buffer) are refused for every format alike.
    SupportsFloat16BcubicFiltering,
    /// `supportsHeapWithAddressRanges`: the device's, no (the runtime makes no heap over address
    /// ranges).
    SupportsHeapWithAddressRanges,
    /// `supportsImageBlocks`: the device's (the `Imageblocks` row).
    SupportsImageBlocks,
    /// `supportsMemoryOrderAtomics`: the device's: yes where it is created with
    /// `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope`. An atomic of any order runs on
    /// every device (acquire and release semantics), but the coherent device memory such an atomic
    /// orders is reached only under the Vulkan memory model at device scope, which a device
    /// without both refuses (`lower::sync::memory_model`).
    SupportsMemoryOrderAtomics,
    /// `supportsNativeHardwareFP16`: the device's (`shaderFloat16`, enabled: its shaders run
    /// 16-bit float arithmetic).
    SupportsNativeHardwareFp16,
    /// `supportsNonSquareTileShaders`: the device's (the `NonsquareTileDispatch` row).
    SupportsNonSquareTileShaders,
    /// `supportsProgrammableSamplePositions`: the device's (the public
    /// `areProgrammableSamplePositionsSupported`, which Metal answers alike).
    SupportsProgrammableSamplePositions,
    /// `supportsResourceDetachBacking`: the device's (a no-copy buffer or heap is detached from
    /// its pages, `detach_backing`, wherever one is made: the device imports host memory, by its
    /// pointer or as dma-bufs of its pages).
    SupportsResourceDetachBacking,
    /// `supportsSharedTextureHandles`: the device's, yes (a shareable texture's handle, within
    /// the process, `new_shared_texture_handle`).
    SupportsSharedTextureHandles,
    /// `supportsSIMDReduction`: the device's (the `SimdReduction` and `QuadReduction` rows).
    SupportsSimdReduction,
    /// `supportsSIMDShuffleAndFill`: the device's (the `SimdShiftAndFill` row).
    SupportsSimdShuffleAndFill,
    /// `supportsTexture2DMultisampleArray`: the device's (a 2D multisample array texture is
    /// made wherever a multisample texture is: the device takes a sample count above 1).
    SupportsTexture2dMultisampleArray,
    /// `supportsTileShaders`: the device's (the `TileShaders` row: no on every device while the
    /// runtime refuses part of Metal's tile shading, though its render encoders run tile
    /// functions).
    SupportsTileShaders,
}

impl PrivateFlag {
    /// Every flag, in declaration order.
    pub const ALL: [PrivateFlag; 23] = {
        use PrivateFlag as F;
        [
            F::IsFramebufferReadSupported,
            F::IsLargeMrtSupported,
            F::IsRgb10a2GammaSupported,
            F::SupportsBufferWithAddressRanges,
            F::SupportsBufferWithIosurface,
            F::SupportsCommandBufferJump,
            F::SupportsComputeCompressedTextureWrite,
            F::SupportsConditionalLoadStore,
            F::SupportsDynamicAttributeStride,
            F::SupportsExtendedXr10Formats,
            F::SupportsFloat16BcubicFiltering,
            F::SupportsHeapWithAddressRanges,
            F::SupportsImageBlocks,
            F::SupportsMemoryOrderAtomics,
            F::SupportsNativeHardwareFp16,
            F::SupportsNonSquareTileShaders,
            F::SupportsProgrammableSamplePositions,
            F::SupportsResourceDetachBacking,
            F::SupportsSharedTextureHandles,
            F::SupportsSimdReduction,
            F::SupportsSimdShuffleAndFill,
            F::SupportsTexture2dMultisampleArray,
            F::SupportsTileShaders,
        ]
    };

    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        self.measured().0
    }

    /// Metal's measured answer for the property on its Apple GPU (the module docs).
    pub(crate) fn metal(self) -> bool {
        self.measured().1
    }

    /// The selector and Metal's measured answer on an Apple M3 Max (the module docs).
    fn measured(self) -> (&'static str, bool) {
        use PrivateFlag as F;
        match self {
            F::IsFramebufferReadSupported => ("isFramebufferReadSupported", true),
            F::IsLargeMrtSupported => ("isLargeMRTSupported", true),
            F::IsRgb10a2GammaSupported => ("isRGB10A2GammaSupported", true),
            F::SupportsBufferWithAddressRanges => ("supportsBufferWithAddressRanges", true),
            F::SupportsBufferWithIosurface => ("supportsBufferWithIOSurface", true),
            F::SupportsCommandBufferJump => ("supportsCommandBufferJump", true),
            F::SupportsComputeCompressedTextureWrite => {
                ("supportsComputeCompressedTextureWrite", false)
            }
            F::SupportsConditionalLoadStore => ("supportsConditionalLoadStore", true),
            F::SupportsDynamicAttributeStride => ("supportsDynamicAttributeStride", true),
            F::SupportsExtendedXr10Formats => ("supportsExtendedXR10Formats", true),
            F::SupportsFloat16BcubicFiltering => ("supportsFloat16BCubicFiltering", true),
            F::SupportsHeapWithAddressRanges => ("supportsHeapWithAddressRanges", true),
            F::SupportsImageBlocks => ("supportsImageBlocks", true),
            F::SupportsMemoryOrderAtomics => ("supportsMemoryOrderAtomics", true),
            F::SupportsNativeHardwareFp16 => ("supportsNativeHardwareFP16", true),
            F::SupportsNonSquareTileShaders => ("supportsNonSquareTileShaders", true),
            F::SupportsProgrammableSamplePositions => ("supportsProgrammableSamplePositions", true),
            F::SupportsResourceDetachBacking => ("supportsResourceDetachBacking", true),
            F::SupportsSharedTextureHandles => ("supportsSharedTextureHandles", true),
            F::SupportsSimdReduction => ("supportsSIMDReduction", true),
            F::SupportsSimdShuffleAndFill => ("supportsSIMDShuffleAndFill", true),
            F::SupportsTexture2dMultisampleArray => ("supportsTexture2DMultisampleArray", true),
            F::SupportsTileShaders => ("supportsTileShaders", true),
        }
    }
}

impl DeviceProperties {
    /// `supportsPrimitiveType:` (Apple SPI) as the runtime answers it: yes for Metal's five
    /// public primitive types (raw 0 to 4), which render encoders draw; no for the four private
    /// ones Metal draws too (Metal answers yes for raw 0 to 8 on an Apple M3 Max and an M4), which
    /// no encoder takes.
    pub fn supports_primitive_type(&self, raw: u64) -> bool {
        u32::try_from(raw).is_ok_and(|raw| PrimitiveType::try_from(raw).is_ok())
    }
}

/// The runtime's answers of Apple's private device properties on one device
/// ([`Device::private_properties`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivateProperties {
    counts: [PrivateAnswer<u64>; PrivateCount::ALL.len()],
    flags: [PrivateAnswer<bool>; PrivateFlag::ALL.len()],
}

impl PrivateProperties {
    /// The answer for `property` (the module docs).
    pub fn count(&self, property: PrivateCount) -> PrivateAnswer<u64> {
        self.counts[property as usize]
    }

    /// The answer for `property` (the module docs).
    pub fn flag(&self, property: PrivateFlag) -> PrivateAnswer<bool> {
        self.flags[property as usize]
    }

    /// The selectors of the properties the device has no answer for ([`PrivateAnswer::Metal`]),
    /// in declaration order, the counts first.
    pub fn unknown(&self) -> Vec<&'static str> {
        let counts = PrivateCount::ALL
            .into_iter()
            .filter(|&p| matches!(self.count(p), PrivateAnswer::Metal(_)))
            .map(PrivateCount::selector);
        let flags = PrivateFlag::ALL
            .into_iter()
            .filter(|&p| matches!(self.flag(p), PrivateAnswer::Metal(_)))
            .map(PrivateFlag::selector);
        counts.chain(flags).collect()
    }

    /// The answers on a device of `properties`, whose compile layer answers
    /// `metal`, and whose rows of [`MEASURED_FAMILY`] are `rows`, and the runtime gives a pixel
    /// format the capabilities `formats` answers and a texture over a buffer the linear
    /// alignment `linear_alignment` (`Device::device_linear_texture_alignment`), and makes no-copy
    /// buffers where `no_copy` (it imports host memory: `host_memory.rs`'s `Importer`).
    pub(crate) fn of(
        properties: &DeviceProperties,
        metal: &MetalDeviceProperties,
        rows: &[RowAudit],
        (formats, linear_alignment): (&dyn Fn(PixelFormat) -> FormatCapabilities, u64),
        no_copy: bool,
    ) -> PrivateProperties {
        let answers = Answers {
            properties,
            metal,
            rows,
            formats,
            linear_alignment,
            no_copy,
        };
        PrivateProperties {
            counts: PrivateCount::ALL.map(|p| {
                let metal = p.metal();
                answers
                    .count(p)
                    .map_or(PrivateAnswer::Metal(metal), |answer| {
                        PrivateAnswer::Device { answer, metal }
                    })
            }),
            flags: PrivateFlag::ALL.map(|p| {
                let metal = p.metal();
                answers
                    .flag(p)
                    .map_or(PrivateAnswer::Metal(metal), |answer| {
                        PrivateAnswer::Device { answer, metal }
                    })
            }),
        }
    }
}

impl Device {
    /// Apple's private (SPI) device properties [`PrivateCount`] and [`PrivateFlag`] name, as the
    /// runtime answers them on this device (the module docs). It judges the
    /// device by the GPU-family rows, so read it once.
    pub fn private_properties(&self) -> PrivateProperties {
        self.current().private_properties_now()
    }

    /// [`Device::private_properties`] on the generation that serves (`device/line.rs`).
    pub(crate) fn private_properties_now(&self) -> PrivateProperties {
        let compile = self.compile_device();
        let answers = PrivateProperties::of(
            self.properties(),
            compile.metal_properties(),
            &self.family_rows_now(MEASURED_FAMILY),
            (
                &|pf| self.pixel_format_capabilities_now(pf),
                self.device_linear_texture_alignment(),
            ),
            self.shared().host_import.is_ok(),
        );
        // The properties the device has no answer for, announced once.
        let unknown = answers.unknown();
        if !unknown.is_empty() {
            self.degrade(Degradation::PrivatePropertiesUnknown, || {
                (
                    format!("the device {}", self.name()),
                    format!(
                        "it has no answer for {}: nothing it reports maps to them, so \
                         `PrivateAnswer::value` gives none, and Metal's measured answers (Apple's \
                         GPU's) are only `PrivateAnswer::metal`",
                        unknown.join(", ")
                    ),
                )
            });
        }
        answers
    }
}

/// What the device's answers are read from.
struct Answers<'a> {
    properties: &'a DeviceProperties,
    metal: &'a MetalDeviceProperties,
    /// The device's rows of [`MEASURED_FAMILY`].
    rows: &'a [RowAudit],
    /// The texture capabilities the runtime gives a pixel format on the device
    /// ([`Device::pixel_format_capabilities`]).
    formats: &'a dyn Fn(PixelFormat) -> FormatCapabilities,
    /// The device's linear texture alignment (`Device::device_linear_texture_alignment`).
    linear_alignment: u64,
    /// The device makes no-copy buffers: it imports host memory, by its pointer or as dma-bufs
    /// of its pages (`host_memory.rs`'s `Importer`).
    no_copy: bool,
}

impl Answers<'_> {
    /// The state of `row` on the device; `None` where the family lists no such row.
    fn state(&self, row: FamilyRow) -> Option<&RowState> {
        self.rows.iter().find(|a| a.row == row).map(|a| &a.state)
    }

    /// Whether the feature row is exact on the device.
    fn exact(&self, row: FeatureRow) -> bool {
        self.state(FamilyRow::Feature(row)) == Some(&RowState::Exact)
    }

    /// The most the device takes of a limit row whose cell the runtime meets with `full`: all of
    /// it where the row is exact, else the device's limit the row is refused for, else none.
    fn limit(&self, row: LimitRow, full: u64) -> u64 {
        match self.state(FamilyRow::Limit(row)) {
            Some(RowState::Exact) => full,
            Some(RowState::Refused(Refusal::Unsupported(unsupported))) => {
                match &unsupported.needs {
                    Needs::AnyOf(alternatives) => alternatives
                        .iter()
                        .flatten()
                        .find_map(|capability| match capability {
                            Capability::Limit { available, .. } => Some((*available).min(full)),
                            _ => None,
                        })
                        .unwrap_or(0),
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

    /// The device's answer for `property`, or `None` for Metal's.
    fn count(&self, property: PrivateCount) -> Option<u64> {
        use LimitRow as L;
        use PrivateCount as C;
        let buffers = self.limit(L::BufferArgumentEntries, u64::from(BUFFER_SLOTS));
        let textures = self.limit(L::TextureArgumentEntries, u64::from(TEXTURE_SLOTS));
        let samplers = self.limit(L::SamplerArgumentEntries, u64::from(SAMPLER_SLOTS));
        let side_2d = largest_side(TextureType::Type2D).unwrap_or(0);
        Some(match property {
            C::GpuCoreCount => u64::from(self.properties.gpu_core_count?),
            C::MaxVertexAmplificationCount | C::MaxComputeThreadgroupMemory => return None,
            C::DeviceLinearTextureAlignmentBytes | C::LinearTextureAlignmentBytes => {
                self.linear_alignment
            }
            C::IosurfaceTextureAlignmentBytes | C::IosurfaceReadOnlyTextureAlignmentBytes => {
                LINEAR_ALIGNMENT
            }
            C::MaxVertexBuffers
            | C::MaxFragmentBuffers
            | C::MaxComputeBuffers
            | C::MaxTileBuffers => buffers,
            C::MaxVertexTextures
            | C::MaxFragmentTextures
            | C::MaxComputeTextures
            | C::MaxTileTextures => textures,
            C::MaxVertexSamplers
            | C::MaxFragmentSamplers
            | C::MaxComputeSamplers
            | C::MaxTileSamplers => samplers,
            C::MaxComputeLocalMemorySizes => u64::from(THREADGROUP_SLOTS),
            // The row's cell is Metal's alignment, which the encoders take every multiple of.
            C::MaxComputeThreadgroupMemoryAlignmentBytes => {
                let metal = property.metal();
                self.limit(L::ThreadgroupMemoryAlignment, metal)
            }
            C::MaxPredicatedNestingDepth => property.metal(),
            C::MaxRasterizationRateLayerCount => RATE_MAP_LAYERS,
            C::MaxTextureWidth2D | C::MaxTextureHeight2D => self.limit(L::TextureSize2D, side_2d),
            C::MaxTextureLayers => self.limit(L::TextureArrayLayers, MAX_ARRAY_LENGTH),
            C::MaxTotalComputeThreadsPerThreadgroup => self.limit(
                L::ThreadsPerThreadgroup,
                u64::from(self.metal.max_total_threads_per_threadgroup),
            ),
            C::MaxViewportCount => {
                if self.properties.description.has(VkFeature::MultiViewport) {
                    let most = u64::try_from(MAX_VIEWPORTS).unwrap_or(u64::MAX);
                    most.min(u64::from(self.properties.max_viewports))
                } else {
                    1
                }
            }
        })
    }

    /// The device's answer for `property`, or `None` for Metal's.
    fn flag(&self, property: PrivateFlag) -> Option<bool> {
        use FeatureRow as R;
        use PrivateFlag as F;
        Some(match property {
            F::IsLargeMrtSupported
            | F::SupportsComputeCompressedTextureWrite
            | F::SupportsConditionalLoadStore => return None,
            F::IsRgb10a2GammaSupported => {
                (self.formats)(PixelFormat::RGB10A2UnormSRGB).contains(FormatCapability::Filter)
            }
            F::SupportsExtendedXr10Formats => self.exact(R::ExtendedRangePixelFormats),
            F::SupportsFloat16BcubicFiltering => true,
            F::SupportsMemoryOrderAtomics => {
                let d = &self.properties.description;
                d.has(VkFeature::VulkanMemoryModel)
                    && d.has(VkFeature::VulkanMemoryModelDeviceScope)
            }
            F::SupportsNativeHardwareFp16 => {
                self.properties.description.has(VkFeature::ShaderFloat16)
            }
            F::IsFramebufferReadSupported => self.exact(R::ProgrammableBlending),
            F::SupportsBufferWithAddressRanges
            | F::SupportsBufferWithIosurface
            | F::SupportsHeapWithAddressRanges => false,
            F::SupportsCommandBufferJump => true,
            F::SupportsDynamicAttributeStride => self.exact(R::DynamicAttributeStride),
            F::SupportsImageBlocks => self.exact(R::Imageblocks),
            F::SupportsNonSquareTileShaders => self.exact(R::NonsquareTileDispatch),
            F::SupportsProgrammableSamplePositions => {
                self.properties
                    .are_programmable_sample_positions_supported()
                    == Ok(true)
            }
            F::SupportsResourceDetachBacking => self.no_copy,
            F::SupportsSharedTextureHandles => true,
            F::SupportsSimdReduction => {
                self.exact(R::SimdReduction) && self.exact(R::QuadReduction)
            }
            F::SupportsSimdShuffleAndFill => self.exact(R::SimdShiftAndFill),
            F::SupportsTexture2dMultisampleArray => [2, 4, 8, 16, 32, 64]
                .into_iter()
                .any(|count| self.properties.supports_texture_sample_count(count) == Ok(true)),
            F::SupportsTileShaders => self.exact(R::TileShaders),
        })
    }
}
