//! Textures: optimal images in memory of their own, each written into set 1's texture arrays at
//! the index the consumer names, in one home layout for its life.
//!
//! A texture's image has the descriptor's usage plus transfer source and destination, since Metal
//! lets the CPU and blits reach any texture. Its home layout is the one its descriptors name, and
//! the image leaves it only inside a copy: general where a shader may write it, shader-read-only
//! where shaders only read it, the attachment layout where only render passes use it. A new image
//! moves from undefined to home in the upload of the next submission, ahead of every command
//! buffer committed after [`Device::new_texture`] returns, or ahead of its first copy; nothing
//! holds it until then, so a texture released first is freed at once.
//!
//! Shaders under the translator's identity domains read a texture through its view alone, so a
//! texture is made only where its view gives Metal's values: its format directly, or through the
//! view's component mapping. A storage element is written only where the device stores the format
//! through a storage image declared without a format.

use crate::arrays::{IndexClass, ResourceId};
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit, Refusal};
use crate::memory::{self, Mapping, MemoryClass};
use crate::reaper::{Free, Job};
use crate::submit::Held;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{self as formats, FormatEmulation};
use metal2vulkan::metal::resource_word::TextureWord;
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// `MTLStorageMode` of a texture. Shared, Managed and Private textures are all device images;
/// Shared and Managed differ from Private only in that [`Texture::replace_region`] and
/// [`Texture::get_bytes`] take them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageMode {
    /// `MTLStorageModeShared`.
    Shared,
    /// `MTLStorageModeManaged`.
    Managed,
    /// `MTLStorageModePrivate`.
    Private,
    /// `MTLStorageModeMemoryless`: refused, [`Gap::MemorylessTexture`].
    Memoryless,
}

/// `MTLTextureDescriptor`: the fields the runtime reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    /// `textureType`. `TextureBuffer` is refused, [`Gap::TextureBuffer`].
    pub texture_type: TextureType,
    /// `pixelFormat`.
    pub pixel_format: PixelFormat,
    /// `width`, in texels.
    pub width: u32,
    /// `height`, in texels; 1 for a 1D texture.
    pub height: u32,
    /// `depth`, in texels; 1 for a texture that is not 3D.
    pub depth: u32,
    /// `mipmapLevelCount`.
    pub mipmap_level_count: u32,
    /// `sampleCount`.
    pub sample_count: u32,
    /// `arrayLength`: the slices of an array type (a cube array's cubes); 1 otherwise.
    pub array_length: u32,
    /// `usage`. `Unknown` (no bits) is taken as Metal takes it: shader read, shader write,
    /// render target and pixel-format view.
    pub usage: TextureUsage,
    /// `storageMode`.
    pub storage_mode: StorageMode,
    /// `swizzle`: each channel's source as shaders read it.
    pub swizzle: TextureSwizzleChannels,
}

impl TextureDescriptor {
    /// `textureDescriptorWithPixelFormat:width:height:mipmapped:NO` of `texture_type`: depth,
    /// levels, samples and array length 1, usage `ShaderRead`, storage Shared, and each channel
    /// from itself, as Metal's defaults are.
    pub fn new(
        texture_type: TextureType,
        pixel_format: PixelFormat,
        width: u32,
        height: u32,
    ) -> TextureDescriptor {
        TextureDescriptor {
            texture_type,
            pixel_format,
            width,
            height,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TextureUsage::SHADER_READ,
            storage_mode: StorageMode::Shared,
            swizzle: TextureSwizzleChannels::DEFAULT,
        }
    }
}

/// `MTLRegion`: an origin and a size in texels, `[x, y, z]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    /// The first texel.
    pub origin: [u32; 3],
    /// The texels on each axis.
    pub size: [u32; 3],
}

/// `MTLTexture`: an optimal-tiling `VkImage` in device-local memory of its own, its views, and the
/// elements of set 1 at its index.
///
/// A cheap handle: clones share one texture. The texture holds the device. Once its last handle
/// is gone, and with it the last command buffer that used it, its elements of set 1 read null
/// again and its index is free; the reaper frees its image once the last upload that moves or
/// copies it has retired too.
#[derive(Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
}

struct TextureInner {
    device: Device,
    descriptor: TextureDescriptor,
    image: Arc<Image>,
    /// How bytes move between the image and the CPU; `None` where no copy is built.
    unit: Option<CopyUnit>,
    /// Its class and index; `None` for a texture no shader reaches.
    index: Option<(IndexClass, u32)>,
    word: u32,
    /// Its resource id in the encoders' hazard sets.
    id: u32,
}

/// A texture's Vulkan objects: its image, its memory and its views. The texture and each upload
/// that moves or copies it hold them; the reaper frees them once the last is gone.
pub(crate) struct Image {
    core: Arc<Core>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The allocation's bytes.
    bytes: u64,
    /// The view of the sampled-texture array's element, where shaders read the texture.
    sampled: Option<vk::ImageView>,
    /// The view of the storage-texture array's element, where shaders write it and the device
    /// stores its format through a storage image without a format.
    storage: Option<vk::ImageView>,
    /// The layout the image is in outside a copy, which its elements name.
    home: vk::ImageLayout,
    /// Every aspect of the image's format, which a change of layout names.
    aspects: vk::ImageAspectFlags,
    /// An upload recorded its move from undefined to `home`. Read and set under the work lock.
    homed: AtomicBool,
}

impl Image {
    /// The image's move from undefined to its home layout, where no upload has recorded it yet:
    /// recorded once, by the upload that is to hold the image. Under the work lock.
    pub(crate) fn homing(&self) -> Option<vk::ImageMemoryBarrier2<'static>> {
        if self.homed.swap(true, Ordering::Relaxed) {
            return None;
        }
        Some(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(self.home)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(whole(self.aspects)),
        )
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        self.core.reaper.give(Job::Free(Free::Texture {
            views: [self.sampled, self.storage],
            image: self.image,
            memory: self.memory,
        }));
    }
}

/// The unit of a copy between an image and bytes: the aspect, an element's bytes and the texels
/// an element covers.
#[derive(Clone, Copy)]
struct CopyUnit {
    aspect: vk::ImageAspectFlags,
    bytes: u32,
    block: [u32; 2],
}

impl CopyUnit {
    /// Whether the unit is a 32-bit float depth, which no upload takes
    /// ([`Gap::FloatDepthUpload`]).
    fn float_depth(&self) -> bool {
        self.aspect == vk::ImageAspectFlags::DEPTH && self.bytes == 4
    }
}

/// The texture's shape, from its descriptor.
pub(crate) struct Shape {
    image_type: vk::ImageType,
    flags: vk::ImageCreateFlags,
    view_type: vk::ImageViewType,
    extent: [u32; 3],
    pub(crate) layers: u32,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        let core = self.device.core();
        core.ids.give_back(self.id);
        // An upload holds the image past the texture: the pending one's bound counts it.
        if Arc::strong_count(&self.image) > 1 {
            core.work.released(self.image.bytes);
        }
        if let Some((class, index)) = self.index {
            // SAFETY: the texture owns its elements until it vacates the index below, and no
            // work that reads them is pending: each command buffer that used the texture held it
            // until it retired.
            unsafe {
                core.arrays
                    .write_texture(&core.device, class, index, None, None)
            };
            lock(&core.arrays.grants).vacate(class, index);
            core.counters.fall(Gauge::DescriptorSlots);
        }
    }
}

impl Device {
    /// `newTextureWithDescriptor:`, at `index`: an optimal image of the descriptor in device-local
    /// memory of its own, its contents undefined, as Metal's are. `index` is given exactly where
    /// shaders reach the texture (shader-read or shader-write usage), and is then an index of a
    /// live grant of the class the usage picks: [`IndexClass::StorageTexture`] with shader-write
    /// usage, else [`IndexClass::Texture`]. The texture's elements of set 1 are written before
    /// the call returns, and its image moved to its home layout ahead of every command buffer
    /// committed after it. Until the next submission nothing else holds the texture: released
    /// first, it is freed at once.
    ///
    /// A swizzle composes with the view's own component mapping. Shader-write usage writes the
    /// storage element where the device stores the format through a storage image without a
    /// format; elsewhere the texture has none, and a write through its index is dropped as
    /// through a nil texture.
    ///
    /// Err: what Metal refuses ([`Refusal::TextureZeroExtent`], [`Refusal::TextureExtentForType`],
    /// [`Refusal::TextureCubeNotSquare`], [`Refusal::TextureArrayLength`],
    /// [`Refusal::TextureLevelCount`], [`Refusal::TextureSampleCount`],
    /// [`Refusal::TextureSwizzleWithWrites`]); [`Refusal::TextureIndex`];
    /// [`Refusal::IndicesNotGranted`] and [`Refusal::IndexInUse`] for an index not granted or
    /// taken; [`Limit::TextureSize`] and [`Limit::TextureSamples`] past the device's;
    /// [`Gap::TextureFormat`], [`Gap::TextureBuffer`] and [`Gap::MemorylessTexture`];
    /// [`Limit::MemoryAllocations`] where its memory would take the device past its count;
    /// [`Error::Vulkan`] where the image or its memory cannot be made; [`Error::Quiesced`].
    pub fn new_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        let d = *descriptor;
        let shape = shape(&d)?;
        if d.storage_mode == StorageMode::Memoryless {
            return Err(Error::Unsupported(Gap::MemorylessTexture));
        }
        let usage = match d.usage.is_empty() {
            true => {
                TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE
                    | TextureUsage::RENDER_TARGET
                    | TextureUsage::PIXEL_FORMAT_VIEW
            }
            false => d.usage,
        };
        if d.swizzle != TextureSwizzleChannels::DEFAULT
            && usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET)
        {
            return Err(Error::Refused(Refusal::TextureSwizzleWithWrites));
        }
        let reached = usage.intersects(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE);
        if reached != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        let class = match usage.contains(TextureUsage::SHADER_WRITE) {
            true => IndexClass::StorageTexture,
            false => IndexClass::Texture,
        };
        let core = self.core();
        let plan = plan(core, &d, usage, &shape)?;
        let index = match index {
            Some(index) => {
                lock(&core.arrays.grants).occupy(class, index)?;
                core.counters.rise(Gauge::DescriptorSlots);
                Some((class, index))
            }
            None => None,
        };
        // SAFETY: the device is live, and `plan` was checked against its image format query.
        let made = unsafe { make(core, &d, &shape, &plan) };
        let (image, memory, bytes, sampled, storage) = match made {
            Ok(made) => made,
            Err(error) => {
                if let Some((class, index)) = index {
                    lock(&core.arrays.grants).vacate(class, index);
                    core.counters.fall(Gauge::DescriptorSlots);
                }
                return Err(error);
            }
        };
        core.counters.rise(Gauge::Textures);
        let image = Arc::new(Image {
            core: Arc::clone(core),
            image,
            memory,
            bytes,
            sampled,
            storage,
            home: plan.home,
            aspects: plan.aspects,
            homed: AtomicBool::new(false),
        });
        let inner = TextureInner {
            device: self.clone(),
            descriptor: d,
            image,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
        };
        if let Some((class, index)) = index {
            // SAFETY: the index is occupied for the texture, whose views are live; its image is
            // in its home layout whenever work uses it, the upload of the next submission going
            // first.
            unsafe {
                core.arrays.write_texture(
                    &core.device,
                    class,
                    index,
                    sampled.map(|view| (view, plan.home)),
                    storage,
                )
            };
        }
        let texture = Texture {
            inner: Arc::new(inner),
        };
        core.work.home(core, &texture.inner.image)?;
        Ok(texture)
    }
}

/// What a texture is made of, decided before anything is made.
struct Plan {
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    /// The sampled view's component mapping and aspect, where shaders read the texture.
    sampled: Option<(vk::ComponentMapping, vk::ImageAspectFlags)>,
    /// Whether the texture has a storage view.
    storage: bool,
    home: vk::ImageLayout,
    aspects: vk::ImageAspectFlags,
    unit: Option<CopyUnit>,
    word: u32,
}

/// Checks a descriptor's shape as Metal does.
pub(crate) fn shape(d: &TextureDescriptor) -> Result<Shape, Error> {
    use TextureType as T;
    let (image_type, flags) =
        formats::image_type(d.texture_type).ok_or(Error::Unsupported(Gap::TextureBuffer))?;
    let view_type =
        formats::image_view_type(d.texture_type).ok_or(Error::Unsupported(Gap::TextureBuffer))?;
    let refused = |refusal| Err(Error::Refused(refusal));
    if d.width == 0 || d.height == 0 || d.depth == 0 {
        return refused(Refusal::TextureZeroExtent);
    }
    let (one_d, three_d) = (
        matches!(d.texture_type, T::Type1D | T::Type1DArray),
        d.texture_type == T::Type3D,
    );
    if (one_d && d.height != 1) || (!three_d && d.depth != 1) {
        return refused(Refusal::TextureExtentForType);
    }
    let cube = matches!(d.texture_type, T::Cube | T::CubeArray);
    if cube && d.width != d.height {
        return refused(Refusal::TextureCubeNotSquare);
    }
    let array = matches!(
        d.texture_type,
        T::Type1DArray | T::Type2DArray | T::CubeArray | T::Type2DMultisampleArray
    );
    if d.array_length == 0 || (!array && d.array_length != 1) {
        return refused(Refusal::TextureArrayLength {
            array_length: d.array_length,
        });
    }
    let multisample = matches!(
        d.texture_type,
        T::Type2DMultisample | T::Type2DMultisampleArray
    );
    if d.sample_count == 0 || (!multisample && d.sample_count != 1) {
        return refused(Refusal::TextureSampleCount {
            sample_count: d.sample_count,
        });
    }
    let max = match multisample || one_d {
        true => 1,
        false => 32 - d.width.max(d.height).max(d.depth).leading_zeros(),
    };
    if d.mipmap_level_count == 0 || d.mipmap_level_count > max {
        return refused(Refusal::TextureLevelCount {
            levels: d.mipmap_level_count,
            max,
        });
    }
    let layers =
        match cube {
            true => d.array_length.checked_mul(6).ok_or(Error::Refused(
                Refusal::TextureArrayLength {
                    array_length: d.array_length,
                },
            ))?,
            false => d.array_length,
        };
    Ok(Shape {
        image_type: image_type.into(),
        flags: flags.into(),
        view_type: view_type.into(),
        extent: [d.width, d.height, d.depth],
        layers,
    })
}

/// Decides the image, its views and its home layout, and checks them against the device's image
/// format query.
fn plan(
    core: &Core,
    d: &TextureDescriptor,
    usage: TextureUsage,
    shape: &Shape,
) -> Result<Plan, Error> {
    let pf = d.pixel_format;
    let gap = || Error::Unsupported(Gap::TextureFormat { pixel_format: pf });
    let word = TextureWord::of(pf, d.swizzle).ok_or_else(gap)?.0;
    let profile = core.translator.profile();
    let mapping =
        formats::vk_format(pf, &formats::texture_usages(pf, usage), profile).map_err(|_| gap())?;
    // The forms whose view gives Metal's values with no conversion in the shader.
    let viewed = matches!(
        mapping.emulation,
        FormatEmulation::Direct
            | FormatEmulation::SwizzledView
            | FormatEmulation::AlphaInRed
            | FormatEmulation::RotatedView
            | FormatEmulation::XChannel
            | FormatEmulation::DepthInCombined
            | FormatEmulation::StencilInCombined
    );
    if !viewed {
        return Err(gap());
    }
    let block = mapping.format.block().ok_or_else(gap)?;
    let format: vk::Format = mapping.format.into();
    let metal_aspects: vk::ImageAspectFlags = mapping.aspects.into();
    let mut aspects = vk::ImageAspectFlags::empty();
    if block.depth.is_some() {
        aspects |= vk::ImageAspectFlags::DEPTH;
    }
    if block.stencil.is_some() {
        aspects |= vk::ImageAspectFlags::STENCIL;
    }
    if aspects.is_empty() {
        aspects = vk::ImageAspectFlags::COLOR;
    }
    let read = usage.contains(TextureUsage::SHADER_READ);
    let storage = usage.contains(TextureUsage::SHADER_WRITE)
        && mapping.emulation == FormatEmulation::Direct
        && profile.supports_format(mapping.format, FormatUsage::StorageImage)
        && profile.supports_format(mapping.format, FormatUsage::StorageImageWriteWithoutFormat);
    let render = usage.contains(TextureUsage::RENDER_TARGET);
    let mut image_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if read {
        image_usage |= vk::ImageUsageFlags::SAMPLED;
    }
    if storage {
        image_usage |= vk::ImageUsageFlags::STORAGE;
    }
    if render {
        image_usage |= match metal_aspects
            .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        {
            true => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            false => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        };
    }
    let home = match (storage, read, render) {
        (true, _, _) => vk::ImageLayout::GENERAL,
        (false, true, _) => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        (false, false, true) => vk::ImageLayout::ATTACHMENT_OPTIMAL,
        (false, false, false) => vk::ImageLayout::GENERAL,
    };
    let samples = vk::SampleCountFlags::from_raw(d.sample_count);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(shape.image_type)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(image_usage)
        .flags(shape.flags);
    let properties = core.image_format_properties(&info).ok_or_else(gap)?;
    let max = properties.max_extent;
    let max_extent = [max.width, max.height, max.depth];
    if shape.extent.iter().zip(max_extent).any(|(&e, m)| e > m)
        || shape.layers > properties.max_array_layers
    {
        return Err(Error::Limit(Limit::TextureSize {
            extent: shape.extent,
            layers: shape.layers,
            max_extent,
            max_layers: properties.max_array_layers,
        }));
    }
    if !d.sample_count.is_power_of_two() || !properties.sample_counts.contains(samples) {
        return Err(Error::Limit(Limit::TextureSamples {
            sample_count: d.sample_count,
            counts: properties.sample_counts.as_raw(),
        }));
    }
    // A sampled view selects one aspect: depth where the format has it, else stencil.
    let view_aspect = if metal_aspects.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else if metal_aspects.contains(vk::ImageAspectFlags::STENCIL) {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let unit = copy_unit(pf, format, metal_aspects, block.size, d.sample_count);
    let sampled = match read {
        true => Some((
            components(mapping.components.into(), d.swizzle).ok_or_else(gap)?,
            view_aspect,
        )),
        false => None,
    };
    Ok(Plan {
        format,
        usage: image_usage,
        samples,
        sampled,
        storage,
        home,
        aspects,
        unit,
        word,
    })
}

/// The unit of a copy of `pf`'s bytes, held in `format`: a color format's element (its texel, or
/// a compressed format's block), a depth format's depth, a stencil format's byte. `None` for a
/// format of depth and stencil both, a depth the device holds in other bits, and a multisample
/// texture.
fn copy_unit(
    pf: PixelFormat,
    format: vk::Format,
    metal_aspects: vk::ImageAspectFlags,
    element: u32,
    samples: u32,
) -> Option<CopyUnit> {
    use vk::ImageAspectFlags as A;
    if samples != 1 {
        return None;
    }
    let (aspect, bytes) = match metal_aspects {
        A::COLOR => (A::COLOR, element),
        A::DEPTH => match format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => (A::DEPTH, 4),
            vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => (A::DEPTH, 2),
            _ => return None,
        },
        A::STENCIL => (A::STENCIL, 1),
        _ => return None,
    };
    Some(CopyUnit {
        aspect,
        bytes,
        block: formats::block_extent(pf).unwrap_or([1, 1]),
    })
}

/// The view's component mapping: Metal's `swizzle` over the format's own `mapping`; `None` for a
/// swizzle source this crate does not know.
pub(crate) fn components(
    mapping: vk::ComponentMapping,
    swizzle: TextureSwizzleChannels,
) -> Option<vk::ComponentMapping> {
    use vk::ComponentSwizzle as C;
    let own = [mapping.r, mapping.g, mapping.b, mapping.a];
    let base = |channel: usize| match own[channel] {
        C::IDENTITY => [C::R, C::G, C::B, C::A][channel],
        other => other,
    };
    let pick = |source: TextureSwizzle| match source {
        TextureSwizzle::Zero => Some(C::ZERO),
        TextureSwizzle::One => Some(C::ONE),
        TextureSwizzle::Red => Some(base(0)),
        TextureSwizzle::Green => Some(base(1)),
        TextureSwizzle::Blue => Some(base(2)),
        TextureSwizzle::Alpha => Some(base(3)),
        _ => None,
    };
    Some(vk::ComponentMapping {
        r: pick(swizzle.red)?,
        g: pick(swizzle.green)?,
        b: pick(swizzle.blue)?,
        a: pick(swizzle.alpha)?,
    })
}

/// Every level and layer of an image of `aspects`.
fn whole(aspects: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspects,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

/// The image, its memory and the memory's bytes, and its views: the sampled view and the storage
/// view.
type Objects = (
    vk::Image,
    vk::DeviceMemory,
    u64,
    Option<vk::ImageView>,
    Option<vk::ImageView>,
);

/// Makes the image, binds it to memory of its own, and makes its views.
///
/// # Safety
///
/// `core`'s device is live, and `plan` was checked against its image format query for `shape`.
unsafe fn make(
    core: &Core,
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
) -> Result<Objects, Error> {
    let mut made = Made::default();
    // SAFETY: the caller's contract; what was made is destroyed where a later step fails.
    let objects = unsafe { made.make(core, d, shape, plan) };
    if objects.is_err() {
        // SAFETY: nothing else holds what was made: no descriptor and no command names it.
        unsafe { made.destroy(core) };
    }
    objects
}

/// What [`make`] has made so far.
#[derive(Default)]
struct Made {
    image: Option<vk::Image>,
    memory: Option<vk::DeviceMemory>,
    views: Vec<vk::ImageView>,
}

impl Made {
    /// # Safety
    ///
    /// As [`make`].
    unsafe fn make(
        &mut self,
        core: &Core,
        d: &TextureDescriptor,
        shape: &Shape,
        plan: &Plan,
    ) -> Result<Objects, Error> {
        let device = &core.device;
        let [width, height, depth] = shape.extent;
        let info = vk::ImageCreateInfo::default()
            .flags(shape.flags)
            .image_type(shape.image_type)
            .format(plan.format)
            .extent(vk::Extent3D {
                width,
                height,
                depth,
            })
            .mip_levels(d.mipmap_level_count)
            .array_layers(shape.layers)
            .samples(plan.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(plan.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: the caller's contract: the device makes such an image.
        let image =
            unsafe { device.create_image(&info, None) }.map_err(Error::vulkan("vkCreateImage"))?;
        self.image = Some(image);
        // SAFETY: the image is the device's.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let bits = requirements.memory_type_bits;
        // Vulkan gives every image of no external handle type a device-local type.
        let local = bits & core.device_local;
        let memory_type = match local {
            0 => bits.trailing_zeros(),
            _ => local.trailing_zeros(),
        };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut dedicated);
        // SAFETY: the type is one the image's requirements allow, and the allocation its own.
        let memory =
            unsafe { memory::allocate(device, &core.counters, core.max_allocations, &allocate) }?;
        self.memory = Some(memory);
        // SAFETY: the allocation is the image's size, of a type its requirements allow.
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(Error::vulkan("vkBindImageMemory"))?;
        let mut view = |components, aspect| {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(shape.view_type)
                .format(plan.format)
                .components(components)
                .subresource_range(whole(aspect));
            // SAFETY: the image is bound, of the view's format, and has the view's usage.
            let view = unsafe { device.create_image_view(&info, None) }
                .map_err(Error::vulkan("vkCreateImageView"))?;
            self.views.push(view);
            Ok::<_, Error>(view)
        };
        let sampled = match plan.sampled {
            Some((components, aspect)) => Some(view(components, aspect)?),
            None => None,
        };
        let storage = match plan.storage {
            true => Some(view(
                vk::ComponentMapping::default(),
                vk::ImageAspectFlags::COLOR,
            )?),
            false => None,
        };
        Ok((image, memory, requirements.size, sampled, storage))
    }

    /// Destroys what was made, the allocation's gauge falling with it.
    ///
    /// # Safety
    ///
    /// Nothing else holds what was made.
    unsafe fn destroy(&self, core: &Core) {
        let device = &core.device;
        // SAFETY: the caller's contract.
        unsafe {
            for &view in &self.views {
                device.destroy_image_view(view, None);
            }
            if let Some(image) = self.image {
                device.destroy_image(image, None);
            }
            if let Some(memory) = self.memory {
                device.free_memory(memory, None);
                core.counters.fall(Gauge::Blocks);
            }
        }
    }
}

/// A copy between the texture and bytes, checked: the subresource, the region and how its rows
/// lie in the bytes.
struct Transfer {
    unit: CopyUnit,
    level: u32,
    /// The array layer; 0 for a 3D texture.
    layer: u32,
    offset: vk::Offset3D,
    extent: vk::Extent3D,
    /// One row of elements' bytes.
    row: u64,
    /// Rows of elements in an image.
    rows: u64,
    /// Images (a 3D region's depth).
    images: u64,
    bytes_per_row: u64,
    bytes_per_image: u64,
}

impl Transfer {
    /// The staging buffer's bytes: the region's rows, packed.
    fn len(&self) -> u64 {
        self.row * self.rows * self.images
    }

    /// Each row's offset in the caller's bytes and in the staging buffer.
    fn rows(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.images).flat_map(move |image| {
            (0..self.rows).map(move |row| {
                (
                    (image * self.bytes_per_image + row * self.bytes_per_row) as usize,
                    ((image * self.rows + row) * self.row) as usize,
                )
            })
        })
    }

    fn buffer_image_copy(&self) -> vk::BufferImageCopy {
        vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: self.unit.aspect,
                mip_level: self.level,
                base_array_layer: self.layer,
                layer_count: 1,
            },
            image_offset: self.offset,
            image_extent: self.extent,
        }
    }
}

impl TextureInner {
    /// Checks a copy of `region` of `level` and `slice` with `len` bytes, rows `bytes_per_row`
    /// and images `bytes_per_image` apart. `None` for an empty region, which copies nothing.
    fn copy(
        &self,
        region: Region,
        level: u32,
        slice: u32,
        len: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<Option<Transfer>, Error> {
        let d = &self.descriptor;
        if matches!(
            d.storage_mode,
            StorageMode::Private | StorageMode::Memoryless
        ) {
            return Err(Error::Refused(Refusal::TextureBytesPrivate));
        }
        let unit = self.unit.ok_or(Error::Unsupported(Gap::TextureBytes {
            pixel_format: d.pixel_format,
        }))?;
        let outside = || Error::Refused(Refusal::TextureRegion { level, slice });
        let three_d = d.texture_type == TextureType::Type3D;
        let slices = match d.texture_type {
            TextureType::Cube | TextureType::CubeArray => d.array_length * 6,
            _ => d.array_length,
        };
        if level >= d.mipmap_level_count || slice >= slices {
            return Err(outside());
        }
        let extent = [d.width, d.height, d.depth].map(|e| (e >> level).max(1));
        let [bw, bh] = unit.block;
        let block = [bw, bh, 1];
        for axis in 0..3 {
            let (origin, size) = (region.origin[axis], region.size[axis]);
            let end = origin.checked_add(size).ok_or_else(outside)?;
            let on_grid =
                origin % block[axis] == 0 && (size % block[axis] == 0 || end == extent[axis]);
            if end > extent[axis] || !on_grid {
                return Err(outside());
            }
        }
        if region.size.contains(&0) {
            return Ok(None);
        }
        let [width, height, depth] = region.size;
        let row = u64::from(width.div_ceil(bw)) * u64::from(unit.bytes);
        let rows = u64::from(height.div_ceil(bh));
        let images = u64::from(depth);
        let image_span = (rows - 1)
            .checked_mul(bytes_per_row)
            .and_then(|span| span.checked_add(row));
        let span = image_span.and_then(|image_span| {
            (images - 1)
                .checked_mul(bytes_per_image)
                .and_then(|span| span.checked_add(image_span))
        });
        let fits = (rows == 1 || bytes_per_row >= row)
            && (images == 1 || image_span.is_some_and(|image_span| bytes_per_image >= image_span))
            && span.is_some_and(|span| span <= len);
        if !fits {
            return Err(Error::Refused(Refusal::TextureBytesLayout {
                bytes_per_row,
                bytes_per_image,
                len,
            }));
        }
        let [x, y, z] = region.origin;
        Ok(Some(Transfer {
            unit,
            level,
            layer: match three_d {
                true => 0,
                false => slice,
            },
            offset: vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: z as i32,
            },
            extent: vk::Extent3D {
                width,
                height,
                depth,
            },
            row,
            rows,
            images,
            bytes_per_row,
            bytes_per_image,
        }))
    }

    /// Records `copy` between the staging buffer and the image, `to_image` for a replace: the
    /// subresource leaves home for the transfer layout and comes back, and a read's bytes are
    /// made visible to the host.
    ///
    /// # Safety
    ///
    /// `command` is recording under the work lock, and the upload holds the image and `staging`.
    unsafe fn record(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        copy: &Transfer,
        staging: vk::Buffer,
        to_image: bool,
    ) {
        use vk::AccessFlags2 as Access;
        use vk::PipelineStageFlags2 as Stage;
        let (layout, access) = match to_image {
            true => (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                Access::TRANSFER_WRITE,
            ),
            false => (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, Access::TRANSFER_READ),
        };
        let range = vk::ImageSubresourceRange {
            aspect_mask: self.image.aspects,
            base_mip_level: copy.level,
            level_count: 1,
            base_array_layer: copy.layer,
            layer_count: 1,
        };
        let barrier = |old, new, src: (Stage, Access), dst: (Stage, Access)| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image.image)
                .subresource_range(range)
        };
        let anything = (
            Stage::ALL_COMMANDS,
            Access::MEMORY_READ | Access::MEMORY_WRITE,
        );
        let away = [barrier(
            self.image.home,
            layout,
            (Stage::ALL_COMMANDS, Access::MEMORY_WRITE),
            (Stage::COPY, access),
        )];
        let written = match to_image {
            true => Access::TRANSFER_WRITE,
            false => Access::NONE,
        };
        let home = [barrier(
            layout,
            self.image.home,
            (Stage::COPY, written),
            anything,
        )];
        let host = [vk::MemoryBarrier2::default()
            .src_stage_mask(Stage::COPY)
            .src_access_mask(Access::TRANSFER_WRITE)
            .dst_stage_mask(Stage::HOST)
            .dst_access_mask(Access::HOST_READ)];
        let back = vk::DependencyInfo::default().image_memory_barriers(&home);
        let back = match to_image {
            true => back,
            false => back.memory_barriers(&host),
        };
        let regions = [copy.buffer_image_copy()];
        // SAFETY: the caller's contract; the subresource is in `layout` between the barriers.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&away),
            );
            match to_image {
                true => device.cmd_copy_buffer_to_image(
                    command,
                    staging,
                    self.image.image,
                    layout,
                    &regions,
                ),
                false => device.cmd_copy_image_to_buffer(
                    command,
                    self.image.image,
                    layout,
                    staging,
                    &regions,
                ),
            }
            device.cmd_pipeline_barrier2(command, &back);
        }
    }
}

impl Texture {
    /// Its resource id in the encoders' hazard sets.
    pub(crate) fn id(&self) -> u32 {
        self.inner.id
    }

    /// Whether `core` is the device this texture was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.inner.device.core(), core)
    }

    /// `gpuResourceID`: its index (0 for a texture no shader reaches) and the translator's
    /// `TextureWord` of its pixel format and swizzle.
    pub fn gpu_resource_id(&self) -> ResourceId {
        ResourceId {
            index: self.inner.index.map_or(0, |(_, index)| index),
            word: self.inner.word,
        }
    }

    /// `textureType`, `pixelFormat`, `width` and the rest, as made.
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.inner.descriptor
    }

    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`: the bytes of
    /// `region` of `level` and `slice` (a 3D texture's slice is 0), rows `bytes_per_row` and
    /// images `bytes_per_image` apart in `bytes`, a compressed format's rows being rows of
    /// blocks. The bytes go to a staging buffer before the call returns, and the upload copies
    /// them into the image ahead of every command buffer committed after it, so the call is
    /// ordered as Metal's CPU write is. Counted in [`Metrics::bytes_copied`]. An empty region
    /// copies nothing.
    ///
    /// Err: [`Refusal::TextureBytesPrivate`]; [`Refusal::TextureRegion`] for a level, slice or
    /// region outside the texture or off its format's block grid;
    /// [`Refusal::TextureBytesLayout`]; [`Gap::TextureBytes`]; [`Gap::FloatDepthUpload`] for a
    /// 32-bit float depth; [`Limit::MemoryAllocations`] where the staging buffer's memory would
    /// take the device past its count; [`Error::Vulkan`] where the staging buffer cannot be made;
    /// [`Error::Quiesced`].
    ///
    /// [`Metrics::bytes_copied`]: crate::Metrics::bytes_copied
    pub fn replace_region(
        &self,
        region: Region,
        level: u32,
        slice: u32,
        bytes: &[u8],
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<(), Error> {
        let inner = &self.inner;
        let len = bytes.len() as u64;
        let Some(copy) = inner.copy(region, level, slice, len, bytes_per_row, bytes_per_image)?
        else {
            return Ok(());
        };
        if copy.unit.float_depth() {
            return Err(Error::Unsupported(Gap::FloatDepthUpload {
                pixel_format: inner.descriptor.pixel_format,
            }));
        }
        let core = inner.device.core();
        let staging = Arc::new(Staging::new(core, copy.len())?);
        let row = copy.row as usize;
        for (from, to) in copy.rows() {
            // SAFETY: `copy` checked that each row lies in `bytes`, and the staging buffer holds
            // every row packed; no one else has the staging buffer yet.
            unsafe {
                bytes
                    .as_ptr()
                    .add(from)
                    .copy_to_nonoverlapping(staging.mapping.get().as_ptr().add(to), row)
            };
        }
        core.counters.copied(copy.len());
        let buffer = staging.buffer;
        core.work.upload(
            core,
            copy.len(),
            [
                Held::Texture(Arc::clone(&inner.image)),
                Held::Staging(staging),
            ],
            // SAFETY: recording under the work lock, the upload holding the image and the
            // staging buffer.
            |device, command| unsafe { inner.record(device, command, &copy, buffer, true) },
        )
    }

    /// `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:`: the image's bytes of
    /// `region` of `level` and `slice`, laid out in `bytes` as [`Texture::replace_region`] takes
    /// them. The upload copies them to a staging buffer, the call submits it at once and waits
    /// for it, and copies them out. Metal's contract holds: the program has waited for the work
    /// that writes the texture. Counted in [`Metrics::bytes_copied`].
    ///
    /// Err: as [`Texture::replace_region`]; [`Error::Vulkan`] with `VK_ERROR_DEVICE_LOST` where
    /// the device is lost first.
    ///
    /// [`Metrics::bytes_copied`]: crate::Metrics::bytes_copied
    pub fn get_bytes(
        &self,
        bytes: &mut [u8],
        bytes_per_row: u64,
        bytes_per_image: u64,
        region: Region,
        level: u32,
        slice: u32,
    ) -> Result<(), Error> {
        let inner = &self.inner;
        let len = bytes.len() as u64;
        let Some(copy) = inner.copy(region, level, slice, len, bytes_per_row, bytes_per_image)?
        else {
            return Ok(());
        };
        let core = inner.device.core();
        let staging = Arc::new(Staging::new(core, copy.len())?);
        let buffer = staging.buffer;
        core.work.upload(
            core,
            copy.len(),
            [
                Held::Texture(Arc::clone(&inner.image)),
                Held::Staging(Arc::clone(&staging)),
            ],
            // SAFETY: as in `replace_region`.
            |device, command| unsafe { inner.record(device, command, &copy, buffer, false) },
        )?;
        let value = core.work.flush();
        core.done.wait(core, value)?;
        // SAFETY: the staging buffer holds every row packed, the copy into it completed and its
        // writes are visible to the host; `copy` checked that each row lies in `bytes`.
        unsafe {
            crate::readback::texels(
                staging.mapping.get().as_ptr(),
                bytes,
                copy.rows(),
                copy.row as usize,
            )
        };
        core.counters.copied(copy.len());
        Ok(())
    }
}

/// A host-visible buffer of a copy's bytes, in memory of its own of the Shared type, mapped for
/// its life. The reaper frees it once its last holder, the upload that copies through it among
/// them, is gone.
pub(crate) struct Staging {
    core: Arc<Core>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapping: Mapping,
}

impl Staging {
    /// A staging buffer of `len` bytes.
    fn new(core: &Arc<Core>, len: u64) -> Result<Staging, Error> {
        let device = &core.device;
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live.
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(Error::vulkan("vkCreateBuffer"))?;
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(core.memory_type(MemoryClass::Shared));
        // SAFETY: the Shared type serves a buffer of every usage, so one of fewer too.
        let allocated =
            unsafe { memory::allocate(device, &core.counters, core.max_allocations, &allocate) };
        let memory = match allocated {
            Ok(memory) => memory,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        // SAFETY: the allocation is the buffer's size, of a type it allows; the memory is
        // host-visible and not mapped.
        let mapped = unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(Error::vulkan("vkBindBufferMemory"))
                .and_then(|()| {
                    device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                        .map_err(Error::vulkan("vkMapMemory"))
                })
        };
        let pointer = match mapped {
            Ok(pointer) => pointer,
            Err(error) => {
                // SAFETY: nothing uses them.
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                core.counters.fall(Gauge::Blocks);
                return Err(error);
            }
        };
        // A successful map gives a non-null address.
        let Some(pointer) = std::ptr::NonNull::new(pointer.cast::<u8>()) else {
            // SAFETY: nothing uses them.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            core.counters.fall(Gauge::Blocks);
            return Err(Error::Vulkan {
                call: "vkMapMemory",
                result: vk::Result::ERROR_MEMORY_MAP_FAILED,
            });
        };
        Ok(Staging {
            core: Arc::clone(core),
            buffer,
            memory,
            mapping: Mapping::new(pointer),
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.core.reaper.give(Job::Free(Free::Staging {
            buffer: self.buffer,
            memory: self.memory,
        }));
    }
}
