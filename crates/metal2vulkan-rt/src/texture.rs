//! Textures: optimal images placed in the device's texture blocks ([`crate::places`]), each
//! written into set 1's texture arrays at the index the consumer names, in one home layout for its
//! life. An image the device requires an allocation of its own for, or one larger than a block, or
//! one the blocks' memory type does not serve, has memory of its own.
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
//!
//! A Managed texture's bytes, where the consumer names them ([`Device::new_managed_texture`]), lie
//! in its Shared buffer as it lays them out; [`Texture::upload_region`] copies a region of them
//! into the image by the device, and [`Texture::replace_region`] and [`Texture::get_bytes`] write
//! and read them there.

use crate::arrays::{Owner, ResourceId};
use crate::buffer::Buffer;
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit, Refusal, RenderPassKind, TextureViewKind};
use crate::memory::{self, Block, BlockId, MemoryClass};
use crate::places::Place;
use crate::reaper::{Free, Job};
use crate::shadow::{self, Area};
use crate::staging::Stage;
use crate::submit::Held;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{self as formats, FormatEmulation};
use metal2vulkan::metal::resource_word::TextureWord;
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// `MTLStorageMode` of a texture. Shared, Managed and Private textures are all device images;
/// Shared and Managed differ from Private only in that [`Texture::replace_region`] and
/// [`Texture::get_bytes`] take them. A Managed texture made by [`Device::new_managed_texture`]
/// also has the consumer's bytes of it, which those two calls reach.
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

/// No Metal counterpart: where one level and slice of a Managed texture's bytes lie in the
/// consumer's buffer ([`Device::new_managed_texture`]): the offset of its first texel, and its rows
/// and images apart, a compressed format's rows being rows of blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BytesPlace {
    /// The level and slice's first byte in the buffer.
    pub offset: u64,
    /// The bytes from one row's start to the next's.
    pub bytes_per_row: u64,
    /// The bytes from one image's start to the next's: a 3D level's slices. Read only where the
    /// level has more than one image.
    pub bytes_per_image: u64,
}

/// No Metal counterpart: a heap's memory, `len` bytes at `offset` in a Private block the consumer
/// holds, taken at the heap's creation; [`Device::new_heap_texture`] places textures in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeapRange {
    /// The block.
    pub block: BlockId,
    /// The heap's first byte in the block.
    pub offset: u64,
    /// The heap's bytes.
    pub len: u64,
}

/// `MTLSizeAndAlign`: the bytes a resource takes in a heap, and what its offset there is a
/// multiple of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SizeAndAlign {
    /// `size`.
    pub size: u64,
    /// `align`, a power of two.
    pub align: u64,
}

/// `MTLRegion`: an origin and a size in texels, `[x, y, z]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    /// The first texel.
    pub origin: [u32; 3],
    /// The texels on each axis.
    pub size: [u32; 3],
}

/// `MTLTexture`: an optimal-tiling `VkImage` in device-local memory, its views, and the elements of
/// set 1 at its index. The image is placed in a Private block the device takes from its pool; it
/// has an allocation of its own only where the device requires one for it
/// (`requiresDedicatedAllocation`), where it is larger than a block, or where the blocks' memory
/// type does not serve it.
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
    /// The image's format.
    format: vk::Format,
    /// How bytes move between the image and the CPU or another image; `None` where no copy is
    /// built.
    unit: Option<CopyUnit>,
    /// Its index of the [`IndexClass::Texture`](crate::IndexClass::Texture) class and its owner
    /// word there; `None` for a texture no shader reaches.
    index: Option<(u32, Owner)>,
    word: u32,
    /// Its resource id in the encoders' hazard sets.
    id: u32,
    /// The bytes the host reaches in a buffer: a Managed texture's, or a texture over a buffer's.
    named: Option<Named>,
    /// A view's parent and place in the image ([`Texture::new_texture_view`]); `None` for a
    /// texture with an image of its own.
    view: Option<View>,
    /// How the image holds the pixel format, which a pass's clear value follows.
    emulation: FormatEmulation,
    /// The image views render passes rendered into, one for each level and layer of the image
    /// (`Texture::attachment`), kept until the texture is gone: at most its levels times its
    /// slices.
    attachments: Mutex<Vec<AttachmentView>>,
    /// Its view of its image's shadow, where a draw may read an attachment through it on a device
    /// without the attachment feedback loop (`crate::shadow`): the sampled view's own type,
    /// mapping and range, and the view once a draw read through it, which its second element
    /// names.
    shadow: Option<ShadowView>,
}

/// A texture's view of its image's shadow (`TextureInner::shadow`).
struct ShadowView {
    view_type: vk::ImageViewType,
    components: vk::ComponentMapping,
    range: vk::ImageSubresourceRange,
    /// Made at the first draw that reads through it, and freed with the texture.
    view: Mutex<Option<vk::ImageView>>,
}

impl ShadowView {
    /// The view of `image`'s shadow a texture of `plan` and `index` takes, of `view_type` and
    /// `range` (the sampled view's): where its image has a shadow, the texture is read and shaders
    /// reach it.
    fn of(
        image: &Image,
        plan: &Plan,
        index: Option<u32>,
        view_type: vk::ImageViewType,
        range: impl FnOnce(vk::ImageAspectFlags) -> vk::ImageSubresourceRange,
    ) -> Option<ShadowView> {
        let (components, aspect) = plan.sampled?;
        (image.shadowing.is_some() && index.is_some()).then(|| ShadowView {
            view_type,
            components,
            range: range(aspect),
            view: Mutex::new(None),
        })
    }
}

/// An image view of one level and layer of a texture's image, for render passes.
struct AttachmentView {
    level: u32,
    layer: u32,
    view: vk::ImageView,
}

/// A level and slice of a texture as a render pass attachment ([`Texture::attachment`]).
pub(crate) struct Attachment {
    /// The image view of the level and layer, every aspect, with the identity swizzle.
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,
    /// The level and layer in the image.
    pub(crate) level: u32,
    pub(crate) layer: u32,
    /// Every aspect of the image's format.
    pub(crate) aspects: vk::ImageAspectFlags,
    /// The image's home layout.
    pub(crate) home: vk::ImageLayout,
    /// The level's width and height.
    pub(crate) extent: [u32; 2],
    /// The image's format, the view's.
    pub(crate) format: vk::Format,
}

/// A texture's bytes in a buffer, which [`Texture::replace_region`] and [`Texture::get_bytes`]
/// write and read there.
struct Named {
    buffer: Buffer,
    /// A place for each level and slice, in order.
    places: Vec<BytesPlace>,
    form: Form,
}

/// How a texture's image holds its named bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// A Managed texture's ([`Device::new_managed_texture`]): [`Texture::upload_region`] copies
    /// them into the image.
    Managed,
    /// A texture over a buffer in place ([`Buffer::new_texture`]): the image lies in them.
    InPlace,
    /// A texture over a buffer the device's linear image does not hold: an image of its own,
    /// which the submissions copy them into and back (`crate::over`).
    Copied,
}

/// A texture view's parent, its place in the image, and its own image views.
struct View {
    /// The texture whose image the view reads, which the view keeps: the first in its chain of
    /// parents with an image of its own.
    parent: Texture,
    /// The view's first level and array layer in the image.
    base: [u32; 2],
    _views: Views,
}

/// A texture view's image views, which the reaper frees once the view is gone.
struct Views {
    core: Arc<Core>,
    sampled: Option<vk::ImageView>,
    storage: Option<vk::ImageView>,
}

impl Drop for Views {
    fn drop(&mut self) {
        (self.core.reaper).give(Job::Free(Free::Views([self.sampled, self.storage])));
    }
}

/// A texture's Vulkan objects: its image, its memory and its views. The texture and each upload
/// that moves or copies it hold them; the reaper frees them once the last is gone.
pub(crate) struct Image {
    core: Arc<Core>,
    image: vk::Image,
    backing: Backing,
    /// The memory's bytes.
    bytes: u64,
    /// The view of the sampled-texture array's element, where shaders read the texture.
    sampled: Option<vk::ImageView>,
    /// The view of the storage-texture array's element, where shaders write it and the device
    /// stores its format through a storage image without a format.
    storage: Option<vk::ImageView>,
    /// The layout the image is in outside a copy, which its elements name.
    home: vk::ImageLayout,
    /// A draw may read an attachment of its own pass through a texture over the image (a
    /// shader-read colour render target of no storage view, its home shader-read-only, on a
    /// device that reads attachments): each such texture has a second element of the
    /// sampled-texture array, at its index plus [`TEXTURES`](crate::floor::TEXTURES). Through the
    /// attachment feedback loop the image has the feedback usage and the element names the
    /// feedback layout, which a pass moves the whole image to for a draw that reads it; on a
    /// device without the loop the element names the texture's view of the image's shadow
    /// (`shadowing`).
    attachment_reads: bool,
    /// Where a draw reads the image as an attachment on a device without the feedback loop: what
    /// its shadow is made of, and the shadow once made (`crate::shadow`).
    shadowing: Option<Box<Shadowing>>,
    /// The image is another's shadow: its drop lowers the shadow gauges, not the textures gauge.
    shadow: bool,
    /// Every aspect of the image's format, which a change of layout names.
    aspects: vk::ImageAspectFlags,
    /// The layout the image is made in: undefined, or preinitialized for a texture over a buffer
    /// in place, whose move home keeps the bytes.
    first: vk::ImageLayout,
    /// An upload recorded its move from `first` to `home`. Read and set under the work lock.
    homed: AtomicBool,
    /// What holds the memory the image lies in, where it is not the image's own, while anything
    /// holds the image.
    _under: Option<Under>,
}

/// The usage of an image's shadow: sampled through its textures' views of it, and copied into.
const SHADOW_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// What an image's shadow is made of, and the shadow once a draw made it (`crate::shadow`).
struct Shadowing {
    /// The texture's descriptor and shape, and its plan with the shadow's usage and no view.
    made: (TextureDescriptor, Shape, Plan),
    shadow: Mutex<Option<Arc<Image>>>,
}

/// What holds the memory an image lies in that is not its own.
enum Under {
    /// A heap texture's block.
    Heap { _block: Arc<Block> },
    /// The buffer a texture over a buffer lies over: its block, or its device copy.
    Buffer { _buffer: Buffer },
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
                .old_layout(self.first)
                .new_layout(self.home)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(whole(self.aspects)),
        )
    }

    /// Whether its textures read it as an attachment through the attachment feedback loop: it has
    /// the feedback usage, and each texture's second element names the feedback layout.
    fn looped(&self) -> bool {
        self.attachment_reads && self.shadowing.is_none()
    }

    /// Its shadow, made where it has none yet: an image of its format, extent, levels and layers,
    /// bound as a texture's image is, whose move from undefined to its home layout,
    /// shader-read-only, the upload of the next submission records, ahead of every command buffer
    /// committed after this returns. The shadow gauges rise with it. `None` for an image of no
    /// shadow.
    ///
    /// Err: [`Error::Vulkan`] where the image cannot be made or bound; [`Error::Quiesced`].
    fn shadow(&self) -> Result<Option<vk::Image>, Error> {
        let Some(shadowing) = &self.shadowing else {
            return Ok(None);
        };
        let mut shadow = lock(&shadowing.shadow);
        if let Some(made) = &*shadow {
            return Ok(Some(made.image));
        }
        let core = &self.core;
        let (d, shape, plan) = &shadowing.made;
        // SAFETY: the device is live, and the texture's plan checked the shadow's usage against
        // its image format query for the shape (`plan`).
        let (image, backing, bytes, _, _) = unsafe { make(core, d, shape, plan, Bind::Pooled) }?;
        core.counters.rise(Gauge::Shadows);
        core.counters.rise_by(Gauge::ShadowBytes, bytes);
        let made = Arc::new(Image {
            core: Arc::clone(core),
            image,
            backing,
            bytes,
            sampled: None,
            storage: None,
            home: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            attachment_reads: false,
            shadowing: None,
            shadow: true,
            aspects: self.aspects,
            first: vk::ImageLayout::UNDEFINED,
            homed: AtomicBool::new(false),
            _under: None,
        });
        core.work.home(core, &made)?;
        *shadow = Some(made);
        Ok(Some(image))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let free = match self.shadow {
            true => Free::Shadow {
                image: self.image,
                backing: self.backing,
                bytes: self.bytes,
            },
            false => Free::Texture {
                views: [self.sampled, self.storage],
                image: self.image,
                backing: self.backing,
            },
        };
        self.core.reaper.give(Job::Free(free));
    }
}

/// The memory an image is bound to.
#[derive(Clone, Copy)]
pub(crate) enum Backing {
    /// A place in a texture block.
    Placed(Place),
    /// An allocation of its own, and its bytes where it spilled out of device-local memory (else
    /// 0).
    Own(vk::DeviceMemory, u64),
    /// A place in a block the image holds: a heap's range ([`Device::new_heap_texture`]), or a
    /// buffer's bytes ([`Buffer::new_texture`]).
    InBlock,
}

/// The unit of a copy between an image and bytes or another image: the aspect, an element's
/// bytes and the texels an element covers.
#[derive(Clone, Copy)]
pub(crate) struct CopyUnit {
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) bytes: u32,
    pub(crate) block: [u32; 2],
}

impl CopyUnit {
    /// Whether the unit is a 32-bit float depth, which no upload takes
    /// ([`Gap::FloatDepthUpload`]).
    pub(crate) fn float_depth(&self) -> bool {
        self.aspect == vk::ImageAspectFlags::DEPTH && self.bytes == 4
    }

    /// What a copy's offset in a buffer is a multiple of: the element's bytes, and 4 for a depth
    /// or stencil aspect.
    fn offset_align(&self) -> u64 {
        match self.aspect == vk::ImageAspectFlags::COLOR {
            true => u64::from(self.bytes),
            false => u64::from(self.bytes).max(4),
        }
    }
}

/// The texture's shape, from its descriptor.
#[derive(Clone, Copy)]
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
        let attachments =
            (self.attachments.get_mut()).unwrap_or_else(std::sync::PoisonError::into_inner);
        if !attachments.is_empty() {
            let views = attachments.drain(..).map(|a| a.view).collect();
            core.reaper.give(Job::Free(Free::AttachmentViews(views)));
        }
        // A view's id and image are its parent's, which it keeps.
        if self.view.is_none() {
            if let Some(named) = (self.named.as_ref()).filter(|named| named.form == Form::Copied) {
                match named.buffer.held().1 {
                    // Its buffer keeps its image where a copy back is owed
                    // (`crate::over::ManagedOvers`).
                    Some(_) => named.buffer.over_entries().release(self.id),
                    None => {
                        core.copied_overs.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
            core.ids.give_back(self.id);
            // An upload holds the image past the texture: the pending one's bound counts it.
            if Arc::strong_count(&self.image) > 1 {
                core.work.released(self.image.bytes);
            }
        }
        if let Some((index, owner)) = self.index {
            // SAFETY: no work that reads its elements is pending: each command buffer that used
            // the texture held it until it retired.
            unsafe {
                (core.arrays).vacate_texture(
                    &core.device,
                    &core.counters,
                    (index, owner),
                    self.image.attachment_reads,
                )
            };
        }
        // Its element is null, or another texture's: its view of the shadow goes before the
        // shadow, which its image holds.
        let shadow = (self.shadow.as_mut()).and_then(|shadow| {
            (shadow.view.get_mut())
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
        });
        if let Some(view) = shadow {
            core.reaper.give(Job::Free(Free::ShadowView(view)));
        }
    }
}

impl Device {
    /// `newTextureWithDescriptor:`, at `index`: an optimal image of the descriptor in device-local
    /// memory, placed in a texture block ([`Texture`]), its contents undefined, as Metal's are.
    /// `index` is given exactly where shaders reach the texture (shader-read or shader-write
    /// usage), and is then an index of a live [`IndexClass::Texture`](crate::IndexClass::Texture)
    /// grant, whatever the usage. The texture's elements of set 1 are written before the call
    /// returns, and its image moved to its home layout ahead of every command buffer committed
    /// after it. Until the next submission nothing else holds the texture: released first, it is
    /// freed at once.
    ///
    /// An index another texture holds, a live one or one the program released that a command
    /// buffer not yet retired still holds, is taken at once, as Metal's later texture at a
    /// resource ID takes the ID: the new texture owns the index and its elements from then, and
    /// the earlier one's going, at its release or once its command buffers retire, touches
    /// neither. No work in flight may read a displaced or released texture through the index
    /// (the program's duty, as it is Metal's; nothing checks it).
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
    /// [`Refusal::IndicesNotGranted`] for an index not granted; [`Limit::TextureSize`] and [`Limit::TextureSamples`] past the device's;
    /// [`Gap::TextureFormat`], [`Gap::TextureBuffer`] and [`Gap::MemorylessTexture`];
    /// [`Limit::MemoryAllocations`] where its memory would take the device past its count;
    /// [`Error::Vulkan`] where the image or its memory cannot be made; [`Error::Quiesced`].
    pub fn new_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        self.texture(descriptor, index, Source::Plain)
    }

    /// `heapTextureSizeAndAlignWithDescriptor:`: the bytes and alignment a texture of
    /// `descriptor` takes in a heap ([`Device::new_heap_texture`]) on this device. They are the
    /// device's image's own memory requirements, not Metal's, so they differ by device: the size
    /// rounded up to, and the alignment raised to, the device's buffer-image granularity, so that
    /// a buffer placed by the same answers never shares a granularity page with the texture.
    ///
    /// Err: as [`Device::new_texture`], but for the index and allocation errors;
    /// [`Gap::HeapTexture`] for an image the device requires an allocation of its own for, or
    /// that does not take the Private memory type.
    pub fn heap_texture_size_and_align(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<SizeAndAlign, Error> {
        let (shape, plan, _) = self.prepare(descriptor)?;
        let core = self.core();
        let info = image_info(descriptor, &shape, &plan);
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut needs = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::DeviceImageMemoryRequirements::default().create_info(&info);
        // SAFETY: the device is live, and `plan` was checked against its image format query.
        unsafe { (core.device).get_device_image_memory_requirements(&query, &mut needs) };
        let requirements = needs.memory_requirements;
        heap_fit(core, descriptor, &requirements, &dedicated)
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:`: the device's linear row rounding, the
    /// smallest power of two from the format's texel bytes to 4,096 at which the device lays the
    /// rows of a linear image of the format, of every width tried, at the row rounded up to it;
    /// the largest over reading, reading and rendering, and reading and a shader's writes, where
    /// the device makes such an image; Metal's 16 where no candidate holds, so that a texture of
    /// the format over a buffer is never in place. The device answers: each image is made, read
    /// with `vkGetImageSubresourceLayout` and destroyed, never bound, once per format for the
    /// device's life. The answer differs by device, as Metal's does by GPU, and is below Metal's 16
    /// on a device that lays a linear row at the row itself.
    ///
    /// A texture over a buffer ([`Buffer::new_texture`]) whose pitch is its row rounded up to the
    /// answer is in place where its place in the buffer's memory is also a multiple of the linear
    /// image's memory alignment (`VkMemoryRequirements::alignment`), which the answer does not
    /// include; at any other place it is copied.
    ///
    /// Err: [`Gap::TextureFormat`] for a format no texture is made of.
    pub fn minimum_linear_texture_alignment(
        &self,
        pixel_format: PixelFormat,
    ) -> Result<u64, Error> {
        const WIDTHS: [u32; 9] = [1, 3, 17, 64, 65, 100, 129, 1000, 1920];
        let core = self.core();
        if let Some(&answer) = lock(&core.linear_alignments).get(&pixel_format) {
            return Ok(answer);
        }
        let read = TextureUsage::SHADER_READ;
        let usages = [
            read,
            read | TextureUsage::RENDER_TARGET,
            read | TextureUsage::SHADER_WRITE,
        ];
        let mut answer = None;
        for usage in usages {
            let mut rows = Vec::with_capacity(WIDTHS.len());
            let mut texel = 1;
            for width in WIDTHS {
                let mut d = TextureDescriptor::new(TextureType::Type2D, pixel_format, width, 3);
                d.usage = usage;
                let (shape, plan) = match self.prepare(&d) {
                    Ok((shape, plan, _)) => (shape, plan),
                    Err(error) if usage == read => return Err(error),
                    Err(_) => break,
                };
                let Some(unit) = plan.unit.filter(|unit| unit.block == [1, 1]) else {
                    break;
                };
                let Some((pitch, _)) = linear_layout(core, &d, &shape, &plan) else {
                    break;
                };
                texel = u64::from(unit.bytes);
                rows.push((u64::from(width * unit.bytes), pitch));
            }
            if rows.len() < WIDTHS.len() {
                continue;
            }
            answer = answer.max(linear_rounding(&rows, texel));
        }
        let answer = answer.unwrap_or(16);
        lock(&core.linear_alignments).insert(pixel_format, answer);
        Ok(answer)
    }

    /// For the device tests: the memory alignment of the device's linear image of `d`, `None`
    /// where it makes none.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn linear_memory_alignment(&self, d: &TextureDescriptor) -> Option<u64> {
        let (shape, plan, _) = self.prepare(d).ok()?;
        linear_layout(self.core(), d, &shape, &plan).map(|(_, align)| align)
    }

    /// `MTLHeap`'s `newTextureWithDescriptor:`, at `index`: a texture as [`Device::new_texture`]
    /// makes it, its image bound at `offset` in `heap`, where the consumer's own placement put it
    /// by [`Device::heap_texture_size_and_align`]. Nothing is allocated, and the bytes are those
    /// the heap's memory holds there, as Metal's heap textures are not zeroed. Textures may alias
    /// one another and the heap's buffers, as Metal's aliasable heap resources do; what one reads
    /// after another wrote is undefined, as in Metal. The texture holds the block.
    ///
    /// Err: as [`Device::new_texture`]; [`Error::NoSuchBlock`] for a block the consumer does not
    /// hold; [`Gap::HeapTexture`] for a Shared block, or as
    /// [`Device::heap_texture_size_and_align`]; [`Refusal::HeapTexturePlace`] for a heap range
    /// that leaves its block, or an image that does not fit in the heap at `offset` or is not
    /// aligned there as the device requires.
    pub fn new_heap_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        heap: HeapRange,
        offset: u64,
    ) -> Result<Texture, Error> {
        let block = self.block(heap.block)?;
        if block.memory.class != MemoryClass::Private {
            return Err(Error::Unsupported(Gap::HeapTexture {
                pixel_format: descriptor.pixel_format,
            }));
        }
        self.texture(descriptor, index, Source::Heap(block, heap, offset))
    }

    /// `newTextureWithDescriptor:` with `MTLStorageModeManaged`, at `index`: as
    /// [`Device::new_texture`] (the descriptor's storage mode is taken as Managed), with the
    /// consumer's bytes of the texture named in `bytes`, a buffer the host reaches, at `places`:
    /// one for each level and, within it, each slice in order (a cube's six faces each a slice, a
    /// 3D level one slice of several images). The consumer writes them; [`Texture::upload_region`]
    /// copies a region of them into the image, and a blit encoder's
    /// [`synchronize_resource`](crate::BlitCommandEncoder::synchronize_resource) copies the image
    /// back into them. The texture keeps the buffer.
    ///
    /// Metal keeps a Managed texture's bytes beside its image itself; here the consumer lays them
    /// out, and the device copies straight from them, so each place's rows are a whole number of
    /// texel blocks apart, its images a whole number of rows apart, and its offset a multiple of
    /// the texel block's bytes, of 4 for a depth or stencil format.
    ///
    /// Err: as [`Device::new_texture`]; [`Refusal::OtherDevice`] for another device's buffer;
    /// [`Refusal::ManagedTextureBytes`] for places that do not lie so, or a buffer the host does
    /// not reach; [`Gap::TextureBytes`] for a format the runtime copies no bytes of.
    pub fn new_managed_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        bytes: &Buffer,
        places: &[BytesPlace],
    ) -> Result<Texture, Error> {
        if !bytes.is_of(self.core()) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let mut d = *descriptor;
        d.storage_mode = StorageMode::Managed;
        self.texture(&d, index, Source::Managed(bytes, places))
    }

    /// A descriptor's checks, as Metal makes them, and what its texture is made of: the shape,
    /// the plan, and the usage taken.
    fn prepare(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<(Shape, Plan, TextureUsage), Error> {
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
        let plan = plan(self.core(), &d, usage, &shape)?;
        Ok((shape, plan, usage))
    }

    /// A texture of `descriptor` at `index`, its image made as `source` says.
    fn texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        source: Source<'_>,
    ) -> Result<Texture, Error> {
        let d = *descriptor;
        let (shape, plan, usage) = self.prepare(&d)?;
        let reached = usage.intersects(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE);
        if reached != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        let core = self.core();
        let (named, first, home) = match &source {
            Source::Plain | Source::Heap(..) => (None, vk::ImageLayout::UNDEFINED, plan.home),
            Source::Managed(bytes, places) => {
                check_places(&d, plan.unit, bytes, places)?;
                let named = Named {
                    buffer: (*bytes).clone(),
                    places: places.to_vec(),
                    form: Form::Managed,
                };
                (Some(named), vk::ImageLayout::UNDEFINED, plan.home)
            }
            Source::Over(buffer, offset, bytes_per_row) => {
                let in_place =
                    check_over(core, &d, &shape, &plan, buffer, *offset, *bytes_per_row)?;
                let named = Named {
                    buffer: (*buffer).clone(),
                    places: vec![BytesPlace {
                        offset: *offset,
                        bytes_per_row: *bytes_per_row,
                        bytes_per_image: 0,
                    }],
                    form: match in_place {
                        true => Form::InPlace,
                        false => Form::Copied,
                    },
                };
                (Some(named), vk::ImageLayout::UNDEFINED, plan.home)
            }
        };
        let (mut named, mut first, mut home) = (named, first, home);
        let (bind, mut block) = match source {
            Source::Plain | Source::Managed(..) => (Bind::Pooled, None),
            Source::Heap(block, heap, offset) => {
                let bind = Bind::Heap {
                    memory: block.memory.device_memory(),
                    base: block.memory.base(),
                    block_len: block.memory.len,
                    heap,
                    offset,
                };
                (bind, Some(Under::Heap { _block: block }))
            }
            Source::Over(..) if named.as_ref().is_some_and(|n| n.form == Form::Copied) => {
                (Bind::Pooled, None)
            }
            Source::Over(buffer, offset, bytes_per_row) => {
                // The host reaches the image's memory too: general, from preinitialized.
                first = vk::ImageLayout::PREINITIALIZED;
                home = vk::ImageLayout::GENERAL;
                // The bytes shaders reach: a Managed buffer's device copy, where it has one.
                let (memory, class, at) = match buffer.held() {
                    (_, Some(copy)) => (copy.memory(), MemoryClass::Private, copy.offset()),
                    (block, None) => (&block.memory, block.memory.class, buffer.offset()),
                };
                let bind = Bind::InPlace {
                    memory: memory.device_memory(),
                    base: memory.base(),
                    memory_type: core.memory_type(class),
                    block_len: memory.len,
                    at: at + offset,
                    bytes_per_row,
                };
                let under = Under::Buffer {
                    _buffer: buffer.clone(),
                };
                (bind, Some(under))
            }
        };
        // SAFETY: the device is live, and `plan` was checked against its image format query; a
        // heap's or a buffer's block is the device's, and the texture holds it.
        let mut made = unsafe { make(core, &d, &shape, &plan, bind) };
        // Where the device's linear image does not take the texture's layout, an image of its own
        // that the submissions copy the bytes into and back.
        if let (Err(Error::Unsupported(Gap::TextureOverBuffer { .. })), Some(named)) =
            (&made, &mut named)
        {
            named.form = Form::Copied;
            (first, home) = (vk::ImageLayout::UNDEFINED, plan.home);
            block = None;
            // SAFETY: as above.
            made = unsafe { make(core, &d, &shape, &plan, Bind::Pooled) };
        }
        let (image, backing, bytes, sampled, storage) = made?;
        core.counters.rise(Gauge::Textures);
        let attachment_reads =
            plan.attachment_reads && home == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        // On a device without the feedback loop, a draw reads such an image through its shadow.
        let shadowing = (attachment_reads && core.feedback.is_none()).then(|| {
            Box::new(Shadowing {
                made: (d, shape, plan.shadow()),
                shadow: Mutex::new(None),
            })
        });
        let image = Arc::new(Image {
            core: Arc::clone(core),
            image,
            backing,
            bytes,
            sampled,
            storage,
            home,
            attachment_reads,
            shadowing,
            shadow: false,
            aspects: plan.aspects,
            first,
            homed: AtomicBool::new(false),
            _under: block,
        });
        let shadow = ShadowView::of(&image, &plan, index, shape.view_type, whole);
        // The texture's elements of set 1, written as it takes its index; a refusal frees the
        // image as `image` drops.
        let index = match index {
            Some(index) => {
                let sampled = sampled.map(|view| (view, home));
                // SAFETY: the texture's views are live; its image is in its home layout whenever
                // work uses it, the upload of the next submission going first.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (sampled, storage),
                        image.looped(),
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let inner = TextureInner {
            device: self.clone(),
            descriptor: d,
            image,
            format: plan.format,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
            named,
            view: None,
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow,
        };
        // Over a buffer of no device copy, the commands track what they write while it lives
        // (`crate::over`).
        let copied = (inner.named.as_ref()).filter(|named| named.form == Form::Copied);
        if copied.is_some_and(|named| named.buffer.held().1.is_none()) {
            core.copied_overs.fetch_add(1, Ordering::Relaxed);
        }
        let texture = Texture {
            inner: Arc::new(inner),
        };
        core.work.home(core, &texture.inner.image)?;
        // Over a Managed buffer's device copy, its buffer keeps what Metal's Managed events owe it
        // (`crate::over::ManagedOvers`), once the move to its home layout goes ahead of every
        // later commit, a capture's copy in among them.
        if let (Some(buffer), Some((_, copy)), Some(unit)) = (
            texture.managed_over(),
            texture.copied_over(),
            texture.inner.unit,
        ) {
            let place =
                (texture.inner.named.as_ref()).map_or(BytesPlace::default(), |n| n.places[0]);
            let end = place.offset
                + u64::from(d.height - 1) * place.bytes_per_row
                + u64::from(d.width) * u64::from(unit.bytes);
            let image = Arc::clone(&texture.inner.image);
            (buffer.over_entries()).add(texture.inner.id, image, copy, (place.offset, end));
        }
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
    /// A draw may read an attachment of its own pass through the texture, and each sampled view
    /// has a second element ([`Image`]'s `attachment_reads`): the image has the feedback usage on
    /// a device with the attachment feedback loop, and a shadow on one without it.
    attachment_reads: bool,
    aspects: vk::ImageAspectFlags,
    unit: Option<CopyUnit>,
    word: u32,
    emulation: FormatEmulation,
}

impl Plan {
    /// The usage of a linear image of the plan, over a buffer in place: without the feedback
    /// usage, as its home is general and no pass moves it to the feedback layout.
    fn linear_usage(&self) -> vk::ImageUsageFlags {
        self.usage & !vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT
    }

    /// The plan of the image's shadow (`crate::shadow`): its format, samples and aspects, with
    /// the shadow's usage and no view of its own.
    fn shadow(&self) -> Plan {
        Plan {
            usage: SHADOW_USAGE,
            sampled: None,
            storage: false,
            attachment_reads: false,
            ..*self
        }
    }
}

/// The slices of each level of a texture of descriptor `d`: a cube's six faces each a slice, and
/// one for a 3D texture.
fn slices(d: &TextureDescriptor) -> u32 {
    match d.texture_type {
        TextureType::Cube | TextureType::CubeArray => d.array_length * 6,
        TextureType::Type3D => 1,
        _ => d.array_length,
    }
}

/// Checks a Managed texture's places: one for each level and slice, each laid out as a
/// buffer-to-image copy reads it and holding its level's bytes in `bytes`.
fn check_places(
    d: &TextureDescriptor,
    unit: Option<CopyUnit>,
    bytes: &Buffer,
    places: &[BytesPlace],
) -> Result<(), Error> {
    let unit = unit.ok_or(Error::Unsupported(Gap::TextureBytes {
        pixel_format: d.pixel_format,
    }))?;
    let slices = slices(d);
    let refused = |level, slice| Error::Refused(Refusal::ManagedTextureBytes { level, slice });
    if places.len() as u64 != u64::from(d.mipmap_level_count) * u64::from(slices) {
        return Err(refused(d.mipmap_level_count, 0));
    }
    if !bytes.is_shared() {
        return Err(refused(0, 0));
    }
    let block_bytes = u64::from(unit.bytes);
    let align = unit.offset_align();
    let base = bytes.device_range().1;
    let [bw, bh] = unit.block;
    for (at, place) in places.iter().enumerate() {
        let (level, slice) = (at as u32 / slices, at as u32 % slices);
        let [width, height, depth] = [d.width, d.height, d.depth].map(|e| (e >> level).max(1));
        let row = u64::from(width.div_ceil(bw)) * block_bytes;
        let rows = u64::from(height.div_ceil(bh));
        let images = u64::from(depth);
        let bpr = place.bytes_per_row;
        let bpi = place.bytes_per_image;
        let span = (images - 1)
            .checked_mul(bpi)
            .and_then(|span| span.checked_add((rows - 1).checked_mul(bpr)?))
            .and_then(|span| span.checked_add(row))
            .and_then(|span| span.checked_add(place.offset));
        let laid = bpr.is_multiple_of(align)
            && bpr >= row
            && (base + place.offset).is_multiple_of(align)
            && (images == 1 || (bpr > 0 && bpi.is_multiple_of(bpr) && bpi >= rows * bpr))
            && span.is_some_and(|end| end <= bytes.length());
        if !laid {
            return Err(refused(level, slice));
        }
    }
    Ok(())
}

/// Checks a texture over `buffer` at `offset`, rows `bytes_per_row` apart: rows a whole number of
/// texels apart and at least a row, an offset a whole number of texels in, and every texel
/// inside the buffer; and that the runtime makes it: 2D of one level, sample and slice, of a color
/// format of one texel an element, its first texel a whole number of texels into the bytes
/// shaders reach. Whether the device's linear image may hold it in place is the answer: in memory
/// that did not spill, of a format and usage the device makes linear images of, with every format
/// feature its optimal image has that the usage uses.
fn check_over(
    core: &Core,
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
    buffer: &Buffer,
    offset: u64,
    bytes_per_row: u64,
) -> Result<bool, Error> {
    let gap = || {
        Error::Unsupported(Gap::TextureOverBuffer {
            pixel_format: d.pixel_format,
        })
    };
    let one = d.mipmap_level_count == 1 && d.sample_count == 1 && d.array_length == 1;
    if d.texture_type != TextureType::Type2D || !one {
        return Err(gap());
    }
    let unit = (plan.unit)
        .filter(|unit| unit.aspect == vk::ImageAspectFlags::COLOR && unit.block == [1, 1])
        .ok_or_else(gap)?;
    let texel = u64::from(unit.bytes);
    let row = u64::from(d.width) * texel;
    let end = u64::from(d.height - 1)
        .checked_mul(bytes_per_row)
        .and_then(|span| span.checked_add(row))
        .and_then(|span| span.checked_add(offset));
    let laid = bytes_per_row.is_multiple_of(texel)
        && bytes_per_row >= row
        && offset.is_multiple_of(texel)
        && end.is_some_and(|end| end <= buffer.length());
    if !laid {
        return Err(Error::Refused(Refusal::BufferTextureLayout {
            offset,
            bytes_per_row,
            len: buffer.length(),
        }));
    }
    // The copy's offset in the bytes shaders reach is a whole number of texels.
    if !(buffer.device_range().1 + offset).is_multiple_of(texel) {
        return Err(gap());
    }
    let (block, copy) = buffer.held();
    let spilled = copy.is_none() && block.memory.spilled();
    let extent = linear_extent(core, shape, plan)
        .is_some_and(|max| d.width <= max.width && d.height <= max.height);
    let (linear, optimal) = core.format_features(plan.format);
    let features = linear.contains(optimal & kept_features(plan.usage));
    Ok(!spilled && extent && features)
}

/// The format features an image of `usage` is used by, which a linear image is to keep where the
/// device's optimal image of its format has them: copies (a blit encoder's copies and fills are
/// transfers; the runtime records no image blit); sampling and its linear filter; a shader's
/// reads, writes and atomics without a format; rendering and blending.
fn kept_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags2 {
    type F = vk::FormatFeatureFlags2;
    let mut kept = F::TRANSFER_SRC | F::TRANSFER_DST;
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        kept |= F::SAMPLED_IMAGE | F::SAMPLED_IMAGE_FILTER_LINEAR;
    }
    if usage.contains(vk::ImageUsageFlags::STORAGE) {
        kept |= F::STORAGE_IMAGE
            | F::STORAGE_IMAGE_ATOMIC
            | F::STORAGE_READ_WITHOUT_FORMAT
            | F::STORAGE_WRITE_WITHOUT_FORMAT;
    }
    if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        kept |= F::COLOR_ATTACHMENT | F::COLOR_ATTACHMENT_BLEND;
    }
    kept
}

/// The largest extent of a linear image of `plan`'s format and usage of `shape`'s type, where the
/// device makes one.
fn linear_extent(core: &Core, shape: &Shape, plan: &Plan) -> Option<vk::Extent3D> {
    let linear = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(plan.format)
        .ty(shape.image_type)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(plan.linear_usage())
        .flags(shape.flags);
    core.image_format_properties(&linear)
        .map(|properties| properties.max_extent)
}

/// The row pitch and memory alignment of a linear image of `d`, `shape` and `plan`, where the
/// device makes one: the image is made, read and destroyed, never bound.
/// The smallest power of two from `texel` (a texel's bytes, rounded up to one) to 4,096 at which
/// every `(row, pitch)`, a linear image's row bytes and the pitch the device lays it at, has its
/// pitch at the row rounded up to it; `None` where none has. A device that lays a row at the row
/// itself answers the texel's bytes: a texture over a buffer at any pitch of whole texels is then
/// its linear image.
pub(crate) fn linear_rounding(rows: &[(u64, u64)], texel: u64) -> Option<u64> {
    let first = texel.next_power_of_two().trailing_zeros();
    (first..=12).map(|power| 1u64 << power).find(|&candidate| {
        (rows.iter()).all(|&(row, pitch)| pitch == row.next_multiple_of(candidate))
    })
}

fn linear_layout(
    core: &Core,
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
) -> Option<(u64, u64)> {
    let max = linear_extent(core, shape, plan)?;
    if d.width > max.width || d.height > max.height {
        return None;
    }
    let info = image_info(d, shape, plan)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(plan.linear_usage())
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    let device = &core.device;
    // SAFETY: the device is live, and makes a linear image of the format, usage and extent.
    let image = unsafe { device.create_image(&info, None) }.ok()?;
    let color = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: the image is the device's, linear, of one level and layer of a color format; it is
    // destroyed once read, never bound.
    let (layout, requirements) = unsafe {
        let layout = device.get_image_subresource_layout(image, color);
        let requirements = device.get_image_memory_requirements(image);
        device.destroy_image(image, None);
        (layout, requirements)
    };
    Some((layout.row_pitch, requirements.alignment))
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
    // A draw may read an attachment of its own pass through such a texture
    // (`crate::render_encoder`), where the device reads attachments: in the feedback layout where
    // it has the attachment feedback loop, else through a copy into the image's shadow.
    let mut attachment_reads = core.attachment_reads
        && read
        && render
        && !storage
        && aspects == vk::ImageAspectFlags::COLOR;
    let looped = attachment_reads && core.feedback.is_some();
    if looped {
        image_usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
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
    let mut properties = core.image_format_properties(&info);
    // A format the device makes without the feedback usage alone is made so, and its draws read
    // no attachment of their own pass through it.
    if properties.is_none() && looped {
        attachment_reads = false;
        image_usage &= !vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        properties = core.image_format_properties(&info.usage(image_usage));
    }
    let properties = properties.ok_or_else(gap)?;
    // Likewise a format the device makes no shadow of.
    if attachment_reads && !looped {
        attachment_reads = core
            .image_format_properties(&info.usage(SHADOW_USAGE))
            .is_some();
    }
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
        attachment_reads,
        aspects,
        unit,
        word,
        emulation: mapping.emulation,
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

/// An image view of `range` of `image`, of `view_type` and `format`, with `components`.
///
/// # Safety
///
/// `image` is the device's and bound, and takes such a view: its format, usage and range.
unsafe fn image_view(
    device: &ash::Device,
    image: vk::Image,
    view_type: vk::ImageViewType,
    format: vk::Format,
    components: vk::ComponentMapping,
    range: vk::ImageSubresourceRange,
) -> Result<vk::ImageView, Error> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .components(components)
        .subresource_range(range);
    // SAFETY: the caller's contract.
    unsafe { device.create_image_view(&info, None) }.map_err(Error::vulkan("vkCreateImageView"))
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
    Backing,
    u64,
    Option<vk::ImageView>,
    Option<vk::ImageView>,
);

/// Makes the image, binds it to a place in a texture block or to memory of its own, and makes its
/// views.
///
/// # Safety
///
/// `core`'s device is live, and `plan` was checked against its image format query for `shape`; a
/// heap's memory is the device's, and lives while the image does.
unsafe fn make(
    core: &Core,
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
    bind: Bind,
) -> Result<Objects, Error> {
    let mut made = Made::default();
    // SAFETY: the caller's contract; what was made is destroyed where a later step fails.
    let objects = unsafe { made.make(core, d, shape, plan, bind) };
    if objects.is_err() {
        // SAFETY: nothing else holds what was made: no descriptor and no command names it.
        unsafe { made.destroy(core) };
    }
    objects
}

/// What a texture's image is made from.
enum Source<'a> {
    /// A place in a texture block, or memory of its own.
    Plain,
    /// As `Plain`, with a Managed texture's named bytes.
    Managed(&'a Buffer, &'a [BytesPlace]),
    /// A block, a heap's range of it, and the texture's offset in the heap.
    Heap(Arc<Block>, HeapRange, u64),
    /// A buffer, the texture's offset in it and its rows' pitch.
    Over(&'a Buffer, u64, u64),
}

/// Where [`make`] binds the image.
enum Bind {
    /// A place in a texture block, or memory of its own.
    Pooled,
    /// A linear image, made preinitialized, at `at` in a block at `base` in the allocation
    /// `memory`, of type `memory_type` and of `block_len` bytes, where the device lays its rows
    /// `bytes_per_row` apart from there.
    InPlace {
        memory: vk::DeviceMemory,
        base: u64,
        memory_type: u32,
        block_len: u64,
        at: u64,
        bytes_per_row: u64,
    },
    /// `offset` in `heap`, whose block of `block_len` bytes lies at `base` in the allocation
    /// `memory`.
    Heap {
        memory: vk::DeviceMemory,
        base: u64,
        block_len: u64,
        heap: HeapRange,
        offset: u64,
    },
}

/// The image of a texture of `d`, of `shape`, as `plan` makes it.
fn image_info(d: &TextureDescriptor, shape: &Shape, plan: &Plan) -> vk::ImageCreateInfo<'static> {
    let [width, height, depth] = shape.extent;
    vk::ImageCreateInfo::default()
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
        .initial_layout(vk::ImageLayout::UNDEFINED)
}

/// A heap texture's size and alignment from its image's `requirements`: rounded to the device's
/// buffer-image granularity, so that no buffer shares a granularity page with the image.
///
/// Err: [`Gap::HeapTexture`] where the image requires an allocation of its own or does not take
/// the Private memory type.
fn heap_fit(
    core: &Core,
    d: &TextureDescriptor,
    requirements: &vk::MemoryRequirements,
    dedicated: &vk::MemoryDedicatedRequirements,
) -> Result<SizeAndAlign, Error> {
    let private = 1 << core.memory_type(MemoryClass::Private);
    if dedicated.requires_dedicated_allocation != vk::FALSE
        || requirements.memory_type_bits & private == 0
    {
        return Err(Error::Unsupported(Gap::HeapTexture {
            pixel_format: d.pixel_format,
        }));
    }
    let granularity = core.granularity.max(1);
    Ok(SizeAndAlign {
        size: requirements.size.next_multiple_of(granularity),
        align: requirements.alignment.max(granularity),
    })
}

/// What [`make`] has made so far.
#[derive(Default)]
struct Made {
    image: Option<vk::Image>,
    backing: Option<Backing>,
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
        bind: Bind,
    ) -> Result<Objects, Error> {
        let device = &core.device;
        let info = match bind {
            Bind::InPlace { .. } => image_info(d, shape, plan)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(plan.linear_usage())
                .initial_layout(vk::ImageLayout::PREINITIALIZED),
            Bind::Pooled | Bind::Heap { .. } => image_info(d, shape, plan),
        };
        // SAFETY: the caller's contract: the device makes such an image.
        let image =
            unsafe { device.create_image(&info, None) }.map_err(Error::vulkan("vkCreateImage"))?;
        self.image = Some(image);
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut needs = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let of = vk::ImageMemoryRequirementsInfo2::default().image(image);
        // SAFETY: the image is the device's.
        unsafe { device.get_image_memory_requirements2(&of, &mut needs) };
        let requirements = needs.memory_requirements;
        let bits = requirements.memory_type_bits;
        // A place in a texture block where the image takes the blocks' memory type and the device
        // does not require an allocation of its own for it (preferring one is no reason): one of
        // its own length where it is longer than a block.
        let placed = dedicated.requires_dedicated_allocation == vk::FALSE
            && bits & (1 << core.memory_type(MemoryClass::Private)) != 0;
        // Where the device-local memory has no texture block left, memory of its own.
        let pooled = match placed && matches!(bind, Bind::Pooled) {
            true => match (core.places).place(core, requirements.size, requirements.alignment) {
                Ok(found) => Some(found),
                Err(error) if error.is_out_of_device_memory() => None,
                Err(error) => return Err(error),
            },
            false => None,
        };
        let (backing, memory, offset) = if let Bind::InPlace {
            memory,
            base,
            memory_type,
            block_len,
            at,
            bytes_per_row,
        } = bind
        {
            let color = vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            };
            // SAFETY: the image is the device's, linear, of one level and layer of a color format.
            let layout = unsafe { device.get_image_subresource_layout(image, color) };
            let fits = layout.offset == 0
                && layout.row_pitch == bytes_per_row
                && dedicated.requires_dedicated_allocation == vk::FALSE
                && bits & (1 << memory_type) != 0
                && (base + at).is_multiple_of(requirements.alignment)
                && at
                    .checked_add(requirements.size)
                    .is_some_and(|end| end <= block_len);
            if !fits {
                return Err(Error::Unsupported(Gap::TextureOverBuffer {
                    pixel_format: d.pixel_format,
                }));
            }
            (Backing::InBlock, memory, base + at)
        } else if let Bind::Heap {
            memory,
            base,
            block_len,
            heap,
            offset,
        } = bind
        {
            let fit = heap_fit(core, d, &requirements, &dedicated)?;
            let refused = Error::Refused(Refusal::HeapTexturePlace {
                offset,
                size: fit.size,
                align: fit.align,
                heap_len: heap.len,
            });
            let heap_end = heap.offset.checked_add(heap.len);
            let at = heap.offset.checked_add(offset);
            let end = offset.checked_add(fit.size);
            let fits = heap_end.is_some_and(|heap_end| heap_end <= block_len)
                && end.is_some_and(|end| end <= heap.len)
                && at.is_some_and(|at| (base + at).is_multiple_of(fit.align));
            match (fits, at) {
                (true, Some(at)) => (Backing::InBlock, memory, base + at),
                _ => return Err(refused),
            }
        } else if let Some((place, block)) = pooled {
            (
                Backing::Placed(place),
                block.device_memory(),
                block.base() + place.offset,
            )
        } else {
            // Vulkan gives every image of no external handle type a device-local type.
            let local = bits & core.device_local;
            let memory_type = match local {
                0 => bits.trailing_zeros(),
                _ => local.trailing_zeros(),
            };
            let mut own = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type)
                .push_next(&mut own);
            // SAFETY: the type, and each a spill takes, is one the image's requirements allow, and
            // the allocation its own.
            let (memory, spilled) =
                unsafe { memory::allocate_spilling(core, MemoryClass::Private, bits, &allocate) }?;
            let spilled = match spilled {
                true => requirements.size,
                false => 0,
            };
            core.counters.rise_by(Gauge::SpilledBytes, spilled);
            (Backing::Own(memory, spilled), memory, 0)
        };
        self.backing = Some(backing);
        // SAFETY: the place or allocation holds the image's size at its alignment, in memory of a
        // type its requirements allow.
        unsafe { device.bind_image_memory(image, memory, offset) }
            .map_err(Error::vulkan("vkBindImageMemory"))?;
        let mut view = |components, aspect| {
            let (view_type, format) = (shape.view_type, plan.format);
            // SAFETY: the image is bound, of the view's format, and has the view's usage.
            let view =
                unsafe { image_view(device, image, view_type, format, components, whole(aspect)) }?;
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
        Ok((image, backing, requirements.size, sampled, storage))
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
            match self.backing {
                Some(Backing::Own(memory, spilled)) => {
                    device.free_memory(memory, None);
                    core.counters.fall(Gauge::Blocks);
                    core.counters.fall_by(Gauge::SpilledBytes, spilled);
                }
                Some(Backing::Placed(place)) => core.places.release(core, place),
                Some(Backing::InBlock) | None => {}
            }
        }
    }
}

/// A region of one level and slice of a texture, checked against the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    pub(crate) level: u32,
    /// The image's array layer: the slice, 0 for a 3D texture.
    pub(crate) layer: u32,
    /// The region's first texel and its size in texels, inside the level.
    pub(crate) origin: [u32; 3],
    pub(crate) size: [u32; 3],
}

impl Placed {
    /// Its first texel, for a copy command.
    pub(crate) fn offset(&self) -> vk::Offset3D {
        // Inside a level, whose extent is a `u32` the image format query bounds far below 2^31.
        let [x, y, z] = self.origin.map(|axis| axis as i32);
        vk::Offset3D { x, y, z }
    }

    /// Its size, for a copy command.
    pub(crate) fn extent(&self) -> vk::Extent3D {
        let [width, height, depth] = self.size;
        vk::Extent3D {
            width,
            height,
            depth,
        }
    }

    /// Its level and layer in `aspect`, for a copy command.
    pub(crate) fn layers(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: self.level,
            base_array_layer: self.layer,
            layer_count: 1,
        }
    }
}

/// Checks `region` of `level` and `slice` of a texture of `d` whose copy unit covers `block`
/// texels: the level and slice are the texture's (a cube's slices are six a cube, a 3D texture
/// has slice 0 alone), and the region lies inside the level, its origin on the block grid and its
/// size whole blocks or reaching the level's edge. `None` for an empty region. The place is in
/// the image, which the texture starts at `base` of: a view's first level and array layer.
///
/// Err: [`Refusal::TextureRegion`].
pub(crate) fn place(
    d: &TextureDescriptor,
    base: [u32; 2],
    block: [u32; 2],
    region: Region,
    level: u32,
    slice: u32,
) -> Result<Option<Placed>, Error> {
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
    let [bw, bh] = block;
    let block = [bw, bh, 1];
    for axis in 0..3 {
        let (origin, size) = (region.origin[axis], region.size[axis]);
        let end = origin.checked_add(size).ok_or_else(outside)?;
        let on_grid = origin % block[axis] == 0 && (size % block[axis] == 0 || end == extent[axis]);
        if end > extent[axis] || !on_grid {
            return Err(outside());
        }
    }
    if region.size.contains(&0) {
        return Ok(None);
    }
    Ok(Some(Placed {
        level: base[0] + level,
        layer: match three_d {
            true => 0,
            false => base[1] + slice,
        },
        origin: region.origin,
        size: region.size,
    }))
}

/// A copy between the texture and bytes, checked: the subresource, the region and how its rows
/// lie in the bytes.
struct Transfer {
    unit: CopyUnit,
    placed: Placed,
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

    /// The runs of bytes between the caller's bytes and the staging buffer, which holds the
    /// region's rows packed: each run's offset in the caller's bytes, its offset in the staging
    /// buffer and its length. One run where the caller's rows lie packed too, else one a row.
    fn runs(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.runs_to(self.row, self.row * self.rows)
    }

    /// The runs of bytes between the caller's bytes and bytes that hold the region's rows
    /// `to_row` and its images `to_image` apart: each run's offset in the caller's bytes, its
    /// offset in the others and its length. One run where both lie packed, else one a row.
    fn runs_to(&self, to_row: u64, to_image: u64) -> impl Iterator<Item = (usize, usize, usize)> {
        let image = self.row * self.rows;
        let packed = self.bytes_per_row == self.row
            && to_row == self.row
            && (self.images == 1 || (self.bytes_per_image == image && to_image == image));
        let (images, rows, len) = match packed {
            true => (1, 1, self.len()),
            false => (self.images, self.rows, self.row),
        };
        let (from_row, from_image) = (self.bytes_per_row, self.bytes_per_image);
        (0..images).flat_map(move |image| {
            (0..rows).map(move |row| {
                (
                    (image * from_image + row * from_row) as usize,
                    (image * to_image + row * to_row) as usize,
                    len as usize,
                )
            })
        })
    }

    /// The copy of the region at `offset` in a buffer, its rows `row_length` and its images
    /// `image_height` texels apart there (0 for packed).
    fn buffer_image_copy(
        &self,
        offset: u64,
        row_length: u32,
        image_height: u32,
    ) -> vk::BufferImageCopy {
        vk::BufferImageCopy {
            buffer_offset: offset,
            buffer_row_length: row_length,
            buffer_image_height: image_height,
            image_subresource: self.placed.layers(self.unit.aspect),
            image_offset: self.placed.offset(),
            image_extent: self.placed.extent(),
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
        let Some(placed) = place(d, self.base(), unit.block, region, level, slice)? else {
            return Ok(None);
        };
        let [bw, bh] = unit.block;
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
        Ok(Some(Transfer {
            unit,
            placed,
            row,
            rows,
            images,
            bytes_per_row,
            bytes_per_image,
        }))
    }

    /// The barrier that moves `placed`'s level and layer, every aspect, from home to `layout`
    /// (a transfer layout) for a copy's `access` (its reads or its writes), after every earlier
    /// command's writes.
    fn away(
        &self,
        placed: &Placed,
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        use vk::AccessFlags2 as Access;
        use vk::PipelineStageFlags2 as Stage;
        self.barrier(placed, self.image.home, layout)
            .src_stage_mask(Stage::ALL_COMMANDS)
            .src_access_mask(Access::MEMORY_WRITE)
            .dst_stage_mask(Stage::COPY)
            .dst_access_mask(access)
    }

    /// The barrier that moves `placed`'s level and layer back home from `layout` once a copy has
    /// run, its writes (into `TRANSFER_DST_OPTIMAL`) visible to every later command.
    fn back(&self, placed: &Placed, layout: vk::ImageLayout) -> vk::ImageMemoryBarrier2<'static> {
        use vk::AccessFlags2 as Access;
        use vk::PipelineStageFlags2 as Stage;
        let written = match layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => Access::TRANSFER_WRITE,
            _ => Access::NONE,
        };
        self.barrier(placed, layout, self.image.home)
            .src_stage_mask(Stage::COPY)
            .src_access_mask(written)
            .dst_stage_mask(Stage::ALL_COMMANDS)
            .dst_access_mask(Access::MEMORY_READ | Access::MEMORY_WRITE)
    }

    /// A layout change of `placed`'s level and layer, every aspect, with no scopes yet.
    fn barrier(
        &self,
        placed: &Placed,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        vk::ImageMemoryBarrier2::default()
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.image.aspects,
                base_mip_level: placed.level,
                level_count: 1,
                base_array_layer: placed.layer,
                layer_count: 1,
            })
    }

    /// A Managed texture's named bytes of `region` of `level` and `slice`: the copy as they lie,
    /// the buffer, the place, and the region's first element's offset in the buffer. `None` for
    /// an empty region.
    fn named(
        &self,
        region: Region,
        level: u32,
        slice: u32,
    ) -> Result<Option<(Transfer, &Buffer, BytesPlace, u64)>, Error> {
        let refused = || Error::Refused(Refusal::ManagedTextureBytes { level, slice });
        let named = self.named.as_ref().ok_or_else(refused)?;
        let (bytes, places) = (&named.buffer, &named.places);
        let d = &self.descriptor;
        let slices = slices(d);
        let place = match level < d.mipmap_level_count && slice < slices {
            true => places[(level * slices + slice) as usize],
            false => BytesPlace::default(),
        };
        let len = bytes.length().saturating_sub(place.offset);
        let pitches = (place.bytes_per_row, place.bytes_per_image);
        let Some(copy) = self.copy(region, level, slice, len, pitches.0, pitches.1)? else {
            return Ok(None);
        };
        let [x, y, z] = region.origin.map(u64::from);
        let [bw, bh] = copy.unit.block.map(u64::from);
        let start = place.offset
            + z * place.bytes_per_image
            + y / bh * place.bytes_per_row
            + x / bw * u64::from(copy.unit.bytes);
        if !(bytes.device_range().1 + start).is_multiple_of(copy.unit.offset_align()) {
            return Err(refused());
        }
        Ok(Some((copy, bytes, place, start)))
    }

    /// The buffer a texture over a buffer lies over ([`Buffer::new_texture`]).
    fn over(&self) -> Option<&Buffer> {
        (self.named.as_ref())
            .filter(|named| named.form != Form::Managed)
            .map(|named| &named.buffer)
    }

    /// Where the host reaches a Managed texture's named bytes of a non-empty `region` of `level`
    /// and `slice`: the region's first element and its place. `None` for a texture without named
    /// bytes.
    fn named_host(
        &self,
        region: Region,
        level: u32,
        slice: u32,
    ) -> Result<Option<(*mut u8, BytesPlace)>, Error> {
        if self.named.is_none() {
            return Ok(None);
        }
        let refused = || Error::Refused(Refusal::ManagedTextureBytes { level, slice });
        let (_, bytes, place, start) = self.named(region, level, slice)?.ok_or_else(refused)?;
        let host = bytes.contents().ok_or_else(refused)?;
        // SAFETY: the texture's creation checked that its places lie in the buffer.
        Ok(Some((unsafe { host.as_ptr().add(start as usize) }, place)))
    }

    /// Records `copy` between `region` of `buffer` and the image, `to_image` for a replace: the
    /// subresource leaves home for the transfer layout and comes back, and a read's bytes are
    /// made visible to the host.
    ///
    /// # Safety
    ///
    /// `command` is recording under the work lock, and the upload holds the image and the
    /// buffer's memory.
    unsafe fn record(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        copy: &Transfer,
        (buffer, region): (vk::Buffer, vk::BufferImageCopy),
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
        let away = [self.away(&copy.placed, layout, access)];
        let home = [self.back(&copy.placed, layout)];
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
        let regions = [region];
        // SAFETY: the caller's contract; the subresource is in `layout` between the barriers.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&away),
            );
            match to_image {
                true => device.cmd_copy_buffer_to_image(
                    command,
                    buffer,
                    self.image.image,
                    layout,
                    &regions,
                ),
                false => device.cmd_copy_image_to_buffer(
                    command,
                    self.image.image,
                    layout,
                    buffer,
                    &regions,
                ),
            }
            device.cmd_pipeline_barrier2(command, &back);
        }
    }
}

impl TextureInner {
    /// Where the texture starts in its image: a view's first level and array layer, else the
    /// image's first.
    fn base(&self) -> [u32; 2] {
        self.view.as_ref().map_or([0, 0], |view| view.base)
    }
}

impl Texture {
    /// Its resource id in the encoders' hazard sets: a view's is its parent's, so the hazards
    /// between them are one texture's.
    pub(crate) fn id(&self) -> u32 {
        self.inner.id
    }

    /// Where the texture starts in its image: a view's first level and array layer.
    pub(crate) fn base(&self) -> [u32; 2] {
        self.inner.base()
    }

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:`: a texture over `levels` and
    /// `slices` of this one's image, of `texture_type`, at `index`, with elements of set 1 of its
    /// own and no copy; `newTextureViewWithPixelFormat:` is the call with the texture's own type,
    /// every level and every slice. The view keeps this texture, as Metal's `parentTexture`, and
    /// its descriptor is a view's in Metal: the extent of its first level, `levels.len()` levels
    /// and `slices.len()` slices (a cube's six faces each one). Copies and encoders that name the
    /// view reach the levels and slices under it; its hazards are its parent's own. A view of a
    /// Managed texture names the texture's bytes of those levels and slices, so the CPU's reads
    /// and writes through either reach the same bytes, as Metal's do.
    ///
    /// Built: the texture's own pixel format, over every slice in the texture's own type, a slice
    /// subrange of a 1D or 2D array, or one slice of a 2D array as a 2D texture, each over any
    /// level subrange. `index` is given exactly where shaders reach the view, as
    /// [`Device::new_texture`] takes it; its usage, storage mode and sample count are the
    /// texture's.
    ///
    /// Err: [`Refusal::TextureViewRange`] for an empty range of levels or slices or one past the
    /// texture's, and [`Refusal::TextureViewFormat`] for another pixel format of a texture made
    /// without `PixelFormatView` usage, as Metal's validation stops them; [`Gap::TextureView`]
    /// for another kind Metal makes, naming it; [`Refusal::TextureIndex`],
    /// [`Refusal::IndicesNotGranted`] as [`Device::new_texture`];
    /// [`Error::Vulkan`] where a view cannot be made.
    pub fn new_texture_view(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u32>,
        slices: Range<u32>,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        use TextureType as T;
        let inner = &self.inner;
        let p = &inner.descriptor;
        let level_count = levels.end.saturating_sub(levels.start);
        let slice_count = slices.end.saturating_sub(slices.start);
        if level_count == 0
            || levels.end > p.mipmap_level_count
            || slice_count == 0
            || slices.end > self::slices(p)
        {
            return Err(Error::Refused(Refusal::TextureViewRange {
                level: levels.start,
                level_count,
                slice: slices.start,
                slice_count,
            }));
        }
        let gap = |kind| Err(Error::Unsupported(Gap::TextureView { kind }));
        if pixel_format != p.pixel_format {
            if !p.usage.is_empty() && !p.usage.contains(TextureUsage::PIXEL_FORMAT_VIEW) {
                return Err(Error::Refused(Refusal::TextureViewFormat {
                    texture: p.pixel_format,
                    view: pixel_format,
                }));
            }
            return gap(TextureViewKind::PixelFormat);
        }
        if p.swizzle != TextureSwizzleChannels::DEFAULT {
            return gap(TextureViewKind::Swizzle);
        }
        let every_slice = slice_count == self::slices(p);
        match (p.texture_type, texture_type) {
            (from, to) if from == to => {
                if !every_slice && !matches!(to, T::Type1DArray | T::Type2DArray) {
                    return gap(TextureViewKind::Slices);
                }
            }
            (T::Type2DArray, T::Type2D) if slice_count == 1 => {}
            _ => return gap(TextureViewKind::TextureType),
        }
        let mut d = *p;
        d.texture_type = texture_type;
        [d.width, d.height, d.depth] =
            [p.width, p.height, p.depth].map(|extent| (extent >> levels.start).max(1));
        d.mipmap_level_count = level_count;
        d.array_length = match texture_type {
            T::Cube | T::CubeArray => slice_count / 6,
            _ => slice_count,
        };
        let (shape, plan, usage) = inner.device.prepare(&d)?;
        let reached = usage.intersects(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE);
        if reached != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        let core = inner.device.core();
        let [level, layer] = inner.base();
        let base = [level + levels.start, layer + slices.start];
        let range = |aspect_mask| vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: base[0],
            level_count,
            base_array_layer: base[1],
            layer_count: shape.layers,
        };
        let image = inner.image.image;
        let make = |(components, aspect)| {
            let (view_type, format) = (shape.view_type, plan.format);
            // SAFETY: the image is bound, of the view's format, has the view's usage, and holds
            // the range: the texture's levels and layers hold the view's.
            unsafe {
                image_view(
                    &core.device,
                    image,
                    view_type,
                    format,
                    components,
                    range(aspect),
                )
            }
        };
        let storage = (plan.storage)
            .then_some((vk::ComponentMapping::default(), vk::ImageAspectFlags::COLOR));
        let mut views = Views {
            core: Arc::clone(core),
            sampled: None,
            storage: None,
        };
        let mut made = || {
            views.sampled = plan.sampled.map(make).transpose()?;
            views.storage = storage.map(make).transpose()?;
            Ok::<_, Error>(())
        };
        let made = made();
        // The textures gauge counts the view from here; the reaper's free of its views lowers it.
        core.counters.rise(Gauge::Textures);
        let (sampled, storage) = (views.sampled, views.storage);
        // A failed view's image views go to the reaper as `views` drops, and so do they where the
        // view is refused its index.
        made?;
        let shadow = ShadowView::of(&inner.image, &plan, index, shape.view_type, range);
        let index = match index {
            Some(index) => {
                let sampled = sampled.map(|view| (view, inner.image.home));
                // SAFETY: the view's image views are live; the image is in its home layout
                // whenever work uses it.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (sampled, storage),
                        inner.image.looped(),
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let parent = match &inner.view {
            Some(view) => view.parent.clone(),
            None => self.clone(),
        };
        // A view names the parent's bytes of its own levels and slices: a Managed texture's, or a
        // texture over a buffer's.
        let per_level = self::slices(p);
        let named = (inner.named.as_ref()).map(|named| {
            let at = |level, slice| named.places[(level * per_level + slice) as usize];
            let level_places = |level| slices.clone().map(move |slice| at(level, slice));
            Named {
                buffer: named.buffer.clone(),
                places: levels.clone().flat_map(level_places).collect(),
                form: named.form,
            }
        });
        let view = TextureInner {
            device: inner.device.clone(),
            descriptor: d,
            image: Arc::clone(&inner.image),
            format: plan.format,
            unit: plan.unit,
            index,
            word: plan.word,
            id: inner.id,
            named,
            view: Some(View {
                parent,
                base,
                _views: views,
            }),
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow,
        };
        Ok(Texture {
            inner: Arc::new(view),
        })
    }

    /// Whether `core` is the device this texture was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.inner.device.core(), core)
    }

    /// Its image.
    pub(crate) fn image(&self) -> vk::Image {
        self.inner.image.image
    }

    /// Whether it reaches `level` and `layer` of its image: a view reaches its own levels and
    /// slices alone.
    pub(crate) fn reaches(&self, level: u32, layer: u32) -> bool {
        let d = &self.inner.descriptor;
        let [first_level, first_layer] = self.inner.base();
        (first_level..first_level + d.mipmap_level_count).contains(&level)
            && (first_layer..first_layer + slices(d)).contains(&layer)
    }

    /// Whether a draw may read an attachment of its own pass through it ([`Image`]'s
    /// `attachment_reads`): its image in the feedback layout, or, on a device without the
    /// attachment feedback loop, its view of its image's shadow.
    pub(crate) fn attachment_reads(&self) -> bool {
        let image = &self.inner.image;
        image.attachment_reads && (image.looped() || self.inner.shadow.is_some())
    }

    /// Its resource id while a draw reads an attachment through it: its second element's, at its
    /// index plus [`TEXTURES`](crate::floor::TEXTURES), with its own word. Nil for a
    /// texture no shader reaches.
    pub(crate) fn second_resource_id(&self) -> ResourceId {
        match self.inner.index {
            Some((index, _)) => ResourceId {
                index: index + crate::floor::TEXTURES,
                word: self.inner.word,
            },
            None => ResourceId::default(),
        }
    }

    /// Its image's shadow, and its own view of it at its second element, each made where it has
    /// none yet (`crate::shadow`): what a draw reads an attachment through on a device without
    /// the attachment feedback loop. The element is written where the texture still owns its
    /// index. `None` for a texture with no view of a shadow.
    ///
    /// Err: [`Error::Vulkan`] where the shadow or the view cannot be made; [`Error::Quiesced`].
    pub(crate) fn shadow(&self) -> Result<Option<vk::Image>, Error> {
        let (Some(at), Some(index)) = (&self.inner.shadow, self.inner.index) else {
            return Ok(None);
        };
        let Some(shadow) = self.inner.image.shadow()? else {
            return Ok(None);
        };
        let mut view = lock(&at.view);
        if view.is_none() {
            let core = self.inner.device.core();
            // SAFETY: the shadow is bound, of the image's format, levels and layers, which hold
            // the sampled view's range, and of the sampled usage.
            let made = unsafe {
                image_view(
                    &core.device,
                    shadow,
                    at.view_type,
                    self.inner.format,
                    at.components,
                    at.range,
                )
            }?;
            *view = Some(made);
            // SAFETY: the set has the second elements (the device reads attachments); the view
            // lives until the texture's elements are vacated, and the shadow is at home,
            // shader-read-only, whenever work reads it.
            unsafe { (core.arrays).place_shadow(&core.device, index, made) };
        }
        Ok(Some(shadow))
    }

    /// The format its image's shadow is made in, the image's; `None` for an image of no shadow.
    pub(crate) fn shadow_format(&self) -> Option<vk::Format> {
        (self.inner.image.shadowing.as_ref()).map(|shadowing| shadowing.made.2.format)
    }

    /// What a draw reads of its image through it (`crate::shadow`): with `own_pixel`, the draw's
    /// scissor (per axis `[first, end)`), the rectangle a fragment function that reads only at its
    /// own pixel reads of its first level and layer; else every level and layer it reaches.
    pub(crate) fn read_areas(&self, own_pixel: Option<[[u32; 2]; 2]>) -> Vec<Area> {
        let d = &self.inner.descriptor;
        let [level, layer] = self.inner.base();
        match own_pixel {
            Some(scissor) => vec![shadow::own_pixel(
                level,
                layer,
                [d.width, d.height],
                scissor,
            )],
            None => {
                let root = match &self.inner.view {
                    Some(view) => &view.parent.inner.descriptor,
                    None => d,
                };
                let size = [root.width, root.height];
                shadow::whole(level, d.mipmap_level_count, layer, slices(d), size)
            }
        }
    }

    /// The bytes of a texel of its image, which its copies move; 0 where no copy is built.
    pub(crate) fn texel_bytes(&self) -> u64 {
        self.inner.unit.map_or(0, |unit| u64::from(unit.bytes))
    }

    /// Every level and layer of its image, every aspect.
    pub(crate) fn whole(&self) -> vk::ImageSubresourceRange {
        let root = match &self.inner.view {
            Some(view) => &view.parent.inner.descriptor,
            None => &self.inner.descriptor,
        };
        vk::ImageSubresourceRange {
            aspect_mask: self.inner.image.aspects,
            base_mip_level: 0,
            level_count: root.mipmap_level_count,
            base_array_layer: 0,
            layer_count: slices(root),
        }
    }

    /// Its image's format.
    pub(crate) fn format(&self) -> vk::Format {
        self.inner.format
    }

    /// How its image holds its pixel format.
    pub(crate) fn emulation(&self) -> FormatEmulation {
        self.inner.emulation
    }

    /// `level` and `slice` of the texture as a render pass attachment: the image view of that
    /// one level and layer of its image, every aspect, with the identity swizzle, made at the
    /// first pass that renders into it and kept with the texture.
    ///
    /// Err: [`Refusal::AttachmentUsage`] for a texture made without `RenderTarget` usage;
    /// [`Gap::RenderPass`] of [`RenderPassKind::Multisample`] for one of more than one sample,
    /// of [`RenderPassKind::TextureType`] for one but a 2D texture, a 2D array, a cube or a cube
    /// array; [`Refusal::AttachmentPlace`] for a level or slice past the texture's;
    /// [`Error::Vulkan`] where the view cannot be made.
    pub(crate) fn attachment(&self, level: u32, slice: u32) -> Result<Attachment, Error> {
        let d = &self.inner.descriptor;
        if !d.usage.is_empty() && !d.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(Error::Refused(Refusal::AttachmentUsage {
                pixel_format: d.pixel_format,
            }));
        }
        let gap = |kind| Err(Error::Unsupported(Gap::RenderPass { kind }));
        match d.texture_type {
            TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray => {
                return gap(RenderPassKind::Multisample)
            }
            _ if d.sample_count > 1 => return gap(RenderPassKind::Multisample),
            TextureType::Type2D
            | TextureType::Type2DArray
            | TextureType::Cube
            | TextureType::CubeArray => {}
            _ => return gap(RenderPassKind::TextureType),
        }
        if level >= d.mipmap_level_count || slice >= slices(d) {
            return Err(Error::Refused(Refusal::AttachmentPlace { level, slice }));
        }
        let [first_level, first_layer] = self.inner.base();
        let (level_in_image, layer) = (first_level + level, first_layer + slice);
        let image = &self.inner.image;
        let mut views = lock(&self.inner.attachments);
        let kept = (views.iter())
            .find(|a| a.level == level_in_image && a.layer == layer)
            .map(|a| a.view);
        let view = match kept {
            Some(view) => view,
            None => {
                let core = self.inner.device.core();
                let range = vk::ImageSubresourceRange {
                    aspect_mask: image.aspects,
                    base_mip_level: level_in_image,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                };
                // SAFETY: the image is bound, of the texture's format, holds the level and layer
                // (the texture's own, which lie in its image), and has the attachment usage its
                // render target usage gave it.
                let view = unsafe {
                    image_view(
                        &core.device,
                        image.image,
                        vk::ImageViewType::TYPE_2D,
                        self.inner.format,
                        vk::ComponentMapping::default(),
                        range,
                    )
                }?;
                core.counters.rise(Gauge::AttachmentViews);
                views.push(AttachmentView {
                    level: level_in_image,
                    layer,
                    view,
                });
                view
            }
        };
        Ok(Attachment {
            view,
            image: image.image,
            level: level_in_image,
            layer,
            aspects: image.aspects,
            home: image.home,
            extent: [(d.width >> level).max(1), (d.height >> level).max(1)],
            format: self.inner.format,
        })
    }

    /// The unit its copies move; `None` where no copy is built.
    pub(crate) fn unit(&self) -> Option<CopyUnit> {
        self.inner.unit
    }

    /// A Managed texture's named bytes ([`Device::new_managed_texture`]): the buffer and its
    /// places, one for each level and, within it, each slice. `None` for any other texture.
    pub(crate) fn named_bytes(&self) -> Option<(&Buffer, &[BytesPlace])> {
        let named = (self.inner.named.as_ref()).filter(|named| named.form == Form::Managed)?;
        Some((&named.buffer, &named.places))
    }

    /// The barrier that moves `placed` from home to the transfer `layout` for a copy's
    /// `access`, after every earlier command's writes.
    pub(crate) fn away(
        &self,
        placed: &Placed,
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        self.inner.away(placed, layout, access)
    }

    /// The barrier that moves `placed` back home from the transfer `layout` once a copy has run,
    /// its writes visible to every later command.
    pub(crate) fn back(
        &self,
        placed: &Placed,
        layout: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        self.inner.back(placed, layout)
    }

    /// `gpuResourceID`: its index (0 for a texture no shader reaches) and the translator's
    /// `TextureWord` of its pixel format and swizzle.
    pub fn gpu_resource_id(&self) -> ResourceId {
        ResourceId {
            index: self.inner.index.map_or(0, |(index, _)| index),
            word: self.inner.word,
        }
    }

    /// Its id in the encoders' hazard sets: a texture over a buffer in place's is the buffer's,
    /// as the two are the same bytes and a use of either is a use of both.
    pub(crate) fn hazard_id(&self) -> u32 {
        match &self.inner.named {
            Some(named) if named.form == Form::InPlace => named.buffer.id(),
            _ => self.inner.id,
        }
    }

    /// Whether the host reads the GPU's writes to it with no copy: a texture in place over a
    /// Shared buffer.
    pub(crate) fn host_reads(&self) -> bool {
        match &self.inner.named {
            Some(named) => named.form == Form::InPlace && named.buffer.is_shared(),
            None => false,
        }
    }

    /// Of a texture over a buffer of no device copy in an image of its own, which the submissions
    /// copy into and back by the Shared rules (`crate::over`): the buffer and the copy.
    pub(crate) fn copied_shared(&self) -> Option<(&Buffer, OverCopy)> {
        self.copied_over()
            .filter(|(buffer, _)| buffer.held().1.is_none())
    }

    /// Of a texture over a Managed buffer's device copy in an image of its own, which Metal's
    /// Managed events owe copies (`crate::over::ManagedOvers`): the buffer.
    pub(crate) fn managed_over(&self) -> Option<&Buffer> {
        let named = (self.inner.named.as_ref()).filter(|named| named.form == Form::Copied)?;
        named.buffer.held().1.map(|_| &named.buffer)
    }

    /// Of a texture over a buffer in an image of its own (`crate::over`): the buffer, and the
    /// copy between the bytes shaders reach of it and the image.
    pub(crate) fn copied_over(&self) -> Option<(&Buffer, OverCopy)> {
        let inner = &self.inner;
        let named = (inner.named.as_ref()).filter(|named| named.form == Form::Copied)?;
        let unit = inner.unit?;
        let place = named.places[0];
        let d = &inner.descriptor;
        let (buffer, at) = named.buffer.device_range();
        let region = vk::BufferImageCopy {
            buffer_offset: at + place.offset,
            buffer_row_length: u32::try_from(place.bytes_per_row / u64::from(unit.bytes)).ok()?,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: unit.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: d.width,
                height: d.height,
                depth: 1,
            },
        };
        let copy = OverCopy {
            image: inner.image.image,
            home: inner.image.home,
            aspects: inner.image.aspects,
            buffer,
            region,
            bytes: u64::from(d.width) * u64::from(d.height) * u64::from(unit.bytes),
            shared: named.buffer.is_shared(),
        };
        Some((&named.buffer, copy))
    }

    /// For the device tests: the bytes its image's memory requirements name.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn image_size(&self) -> u64 {
        self.inner.image.bytes
    }

    /// `textureType`, `pixelFormat`, `width` and the rest, as made.
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.inner.descriptor
    }

    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`: the bytes of
    /// `region` of `level` and `slice` (a 3D texture's slice is 0), rows `bytes_per_row` and
    /// images `bytes_per_image` apart in `bytes`, a compressed format's rows being rows of
    /// blocks. The bytes go to the device's staging region before the call returns (a staging
    /// buffer of their own past the region's 16 MiB), and the upload copies them into the image
    /// ahead of every command buffer committed after it, so the call is ordered as Metal's CPU
    /// write is. Where the region has no room, the call first waits for the oldest copy in it to
    /// complete. Counted in [`Metrics::bytes_copied`]. An empty region copies nothing.
    ///
    /// A Managed texture with named bytes ([`Device::new_managed_texture`]) takes them there
    /// instead, as Metal writes a Managed texture's CPU copy, and then
    /// [`Texture::upload_region`] copies them into the image; nothing is staged or counted. A
    /// texture over a buffer ([`Buffer::new_texture`]) takes them in the buffer, which is its
    /// image's memory; nothing is staged, copied or counted.
    ///
    /// Err: [`Refusal::TextureBytesPrivate`]; [`Refusal::TextureRegion`] for a level, slice or
    /// region outside the texture or off its format's block grid;
    /// [`Refusal::TextureBytesLayout`]; [`Gap::TextureBytes`]; [`Gap::FloatDepthUpload`] for a
    /// 32-bit float depth; [`Limit::MemoryAllocations`] where the region's block or a staging
    /// buffer's memory would take the device past its count; [`Error::Vulkan`] where either
    /// cannot be made, and with `VK_ERROR_DEVICE_LOST` where the device is lost while the call
    /// waits; [`Error::Quiesced`]; over a Managed buffer, what [`Buffer::did_modify_range`]
    /// returns.
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
        if let Some((host, place)) = inner.named_host(region, level, slice)? {
            // SAFETY: `copy` checked that each row lies in `bytes`, and the texture's creation
            // that each row of its place lies in the named buffer.
            unsafe {
                for (from, to, len) in copy.runs_to(place.bytes_per_row, place.bytes_per_image) {
                    bytes
                        .as_ptr()
                        .add(from)
                        .copy_to_nonoverlapping(host.add(to), len)
                }
            }
            let Some(buffer) = inner.over() else {
                return self.upload_region(region, level, slice);
            };
            // The image lies in the bytes shaders reach: a Managed buffer's device copy takes the
            // rows' span, as `did_modify_range` copies it; any other buffer is the bytes.
            let [x, y, _] = region.origin.map(u64::from);
            let [width, height, _] = region.size.map(u64::from);
            let texel = u64::from(copy.unit.bytes);
            let start = place.offset + y * place.bytes_per_row + x * texel;
            let span = (height - 1) * place.bytes_per_row + width * texel;
            return buffer.did_modify_range(start, span);
        }
        let core = inner.device.core();
        let stage = core.work.stage(core, copy.len(), false)?;
        let host = stage.host().as_ptr();
        for (from, to, len) in copy.runs() {
            // SAFETY: `copy` checked that each row lies in `bytes`, and the stage holds every row
            // packed; it is this call's alone until its copy is recorded.
            unsafe {
                bytes
                    .as_ptr()
                    .add(from)
                    .copy_to_nonoverlapping(host.add(to), len)
            };
        }
        core.counters.copied(copy.len());
        core.work.upload(
            core,
            Some(&stage),
            [Held::Texture(Arc::clone(&inner.image))],
            // SAFETY: recording under the work lock, the upload holding the image and the stage.
            |device, command| unsafe {
                let region = copy.buffer_image_copy(stage.offset(), 0, 0);
                inner.record(device, command, &copy, (stage.buffer(), region), true)
            },
        )
    }

    /// No Metal counterpart: the copy into a Managed texture's image of its named bytes
    /// ([`Device::new_managed_texture`]) of `region` of `level` and `slice`, a 3D texture's slice
    /// being 0, which the consumer has written: what Metal does for a Managed texture's CPU write.
    /// The upload copies them from the named buffer straight into the image, with no CPU copy,
    /// ahead of every command buffer committed after this call on every queue; the copy reads the
    /// bytes when the GPU reaches it. [`Texture::replace_region`] on such a texture writes the named bytes,
    /// then calls this. An empty region copies nothing.
    ///
    /// Err: [`Refusal::ManagedTextureBytes`] for a texture without named bytes, or a depth or
    /// stencil region whose first byte is not a multiple of 4 in the buffer;
    /// [`Refusal::TextureRegion`] for a level, slice or region outside the texture or off its
    /// format's block grid; [`Gap::FloatDepthUpload`] for a 32-bit float depth;
    /// [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's command buffer cannot be
    /// made.
    pub fn upload_region(&self, region: Region, level: u32, slice: u32) -> Result<(), Error> {
        let inner = &self.inner;
        if inner
            .named
            .as_ref()
            .is_none_or(|named| named.form != Form::Managed)
        {
            return Err(Error::Refused(Refusal::ManagedTextureBytes {
                level,
                slice,
            }));
        }
        let Some((copy, bytes, place, start)) = inner.named(region, level, slice)? else {
            return Ok(());
        };
        if copy.unit.float_depth() {
            return Err(Error::Unsupported(Gap::FloatDepthUpload {
                pixel_format: inner.descriptor.pixel_format,
            }));
        }
        let [bw, bh] = copy.unit.block.map(u64::from);
        let row_length = place.bytes_per_row / u64::from(copy.unit.bytes) * bw;
        let image_height = match copy.images > 1 {
            true => place.bytes_per_image / place.bytes_per_row * bh,
            false => 0,
        };
        let refused = || Error::Refused(Refusal::ManagedTextureBytes { level, slice });
        let row_length = u32::try_from(row_length).map_err(|_| refused())?;
        let image_height = u32::try_from(image_height).map_err(|_| refused())?;
        let (buffer, at) = bytes.device_range();
        let region = copy.buffer_image_copy(at + start, row_length, image_height);
        let core = inner.device.core();
        core.work.upload(
            core,
            None,
            [
                Held::Texture(Arc::clone(&inner.image)),
                Held::Block(Arc::clone(bytes.held().0)),
            ],
            // SAFETY: recording under the work lock, the upload holding the image and the
            // named buffer's block.
            |device, command| unsafe {
                inner.record(device, command, &copy, (buffer, region), true)
            },
        )
    }

    /// `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:`: the image's bytes of
    /// `region` of `level` and `slice`, laid out in `bytes` as [`Texture::replace_region`] takes
    /// them. The upload copies them to the device's staging region, the call submits it at once
    /// and waits for it, and copies them out. Metal's contract holds: the program has waited for
    /// the work that writes the texture. Counted in [`Metrics::bytes_copied`].
    ///
    /// A Managed texture with named bytes ([`Device::new_managed_texture`]) reads them there, as
    /// Metal reads a Managed texture's CPU copy, with no submission or wait; the GPU's writes reach
    /// them only by a synchronize. A texture over a buffer ([`Buffer::new_texture`]) reads them in
    /// the buffer, with no submission or wait: the GPU's writes are there once the work that
    /// wrote them completed.
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
        if let Some((host, place)) = inner.named_host(region, level, slice)? {
            let runs = copy.runs_to(place.bytes_per_row, place.bytes_per_image);
            // SAFETY: `copy` checked that each row lies in `bytes`, and the texture's creation
            // that each row of its place lies in the named buffer.
            unsafe { crate::readback::texels(host, bytes, runs) };
            return Ok(());
        }
        self.read_image(bytes, &copy)
    }

    /// The image's bytes of `region` of `level` and `slice`, as [`Texture::get_bytes`] reads them
    /// from a texture without named bytes, once a commit has taken the Managed buffers' named
    /// ranges (`crate::capture`) and the pending upload has run: for the tests of one with them.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn image_bytes(
        &self,
        bytes: &mut [u8],
        bytes_per_row: u64,
        region: Region,
        level: u32,
        slice: u32,
    ) -> Result<(), Error> {
        let len = bytes.len() as u64;
        let core = self.inner.device.core();
        let queue = self.inner.device.new_command_queue(1)?;
        let command_buffer = queue.command_buffer()?;
        command_buffer.record_for_test(&[], |_, _| {})?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed();
        drop((command_buffer, queue));
        let value = core.work.flush();
        core.done.wait(core, value)?;
        match (self.inner).copy(region, level, slice, len, bytes_per_row, len)? {
            Some(copy) => self.read_image(bytes, &copy),
            None => Ok(()),
        }
    }

    /// Reads `copy` of the image into `bytes` through the staging region, submitting the upload
    /// and waiting for it.
    fn read_image(&self, bytes: &mut [u8], copy: &Transfer) -> Result<(), Error> {
        let inner = &self.inner;
        let core = inner.device.core();
        let stage = core.work.stage(core, copy.len(), true)?;
        core.work.upload(
            core,
            Some(&stage),
            [Held::Texture(Arc::clone(&inner.image))],
            // SAFETY: as in `replace_region`.
            |device, command| unsafe {
                let region = copy.buffer_image_copy(stage.offset(), 0, 0);
                inner.record(device, command, copy, (stage.buffer(), region), false)
            },
        )?;
        let value = core.work.flush();
        let waited = core.done.wait(core, value);
        if waited.is_ok() {
            // SAFETY: the stage holds every row packed, the copy into it completed and its writes
            // are visible to the host, and no other call takes its span while this one reads it;
            // `copy` checked that each row lies in `bytes`.
            unsafe { crate::readback::texels(stage.host().as_ptr(), bytes, copy.runs()) };
            core.counters.copied(copy.len());
        }
        if let Stage::Region { number, .. } = stage {
            core.work.unread(number);
        }
        waited
    }
}

impl Buffer {
    /// `newTextureWithDescriptor:offset:bytesPerRow:`, at `index` as for
    /// [`Device::new_texture`]: a texture whose texels are the buffer's bytes from `offset`,
    /// rows `bytes_per_row` apart, with the buffer's storage mode whatever the descriptor's. Its
    /// image is a linear image in the bytes shaders reach (a Managed buffer's device copy, where
    /// it has one), made preinitialized so that its move home keeps the bytes, in the general
    /// layout, where the device lays such an image's rows at `bytes_per_row` and the place meets
    /// its memory alignment: the GPU reaches the buffer's bytes, and nothing is copied that the
    /// buffer does not copy. [`Device::minimum_linear_texture_alignment`] answers the offset and
    /// pitch at which a texture is so. [`Texture::replace_region`] writes the buffer's bytes, and
    /// of a Managed buffer's device copy copies the rows' span into it as
    /// [`Buffer::did_modify_range`] does; [`Texture::get_bytes`] reads the buffer's bytes, of a
    /// Managed buffer the consumer's, as Metal reads a Managed texture's CPU copy. The texture
    /// holds the buffer.
    ///
    /// Where the device's linear image does not take the texture (another pitch, an offset off
    /// its memory alignment, a format or usage its linear tiling lacks, memory that spilled), the
    /// texture is an optimal image of its own, which the device copies the bytes shaders reach
    /// into and back (`crate::over`): over a Shared buffer once a submission, over a Managed
    /// buffer's device copy only at Metal's Managed events. Each copy counts in
    /// [`Metrics::over_copies`](crate::Metrics::over_copies).
    ///
    /// Err: as [`Device::new_texture`]; [`Refusal::BufferTextureLayout`] for rows that are not a
    /// whole number of texels apart or are closer than a row, an offset that is not a whole
    /// number of texels, or texels past the buffer; [`Gap::TextureOverBuffer`] for a texture
    /// other than 2D of one level, sample and slice, or of a format of other than one color texel
    /// an element, or whose first texel in the bytes shaders reach is not a whole number of
    /// texels in.
    pub fn new_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<Texture, Error> {
        let mut d = *descriptor;
        d.storage_mode = match self.held() {
            (_, Some(_)) => StorageMode::Managed,
            (block, None) => match block.memory.class {
                MemoryClass::Shared => StorageMode::Shared,
                MemoryClass::Private => StorageMode::Private,
            },
        };
        let over = Source::Over(self, offset, bytes_per_row);
        self.device().texture(&d, index, over)
    }
}

/// A texture over a buffer in an image of its own: the copy between the bytes shaders reach of
/// the buffer and its image (`crate::over`).
#[derive(Clone, Copy)]
pub(crate) struct OverCopy {
    pub(crate) image: vk::Image,
    /// The image's layout outside a copy.
    pub(crate) home: vk::ImageLayout,
    pub(crate) aspects: vk::ImageAspectFlags,
    /// The `VkBuffer` of the bytes shaders reach: the block's, or a Managed buffer's device copy's.
    pub(crate) buffer: vk::Buffer,
    pub(crate) region: vk::BufferImageCopy,
    /// The texels' bytes.
    pub(crate) bytes: u64,
    /// The buffer is Shared: the host reads what a copy back writes.
    pub(crate) shared: bool,
}

// SAFETY: the handle is its one `Arc` and nothing else, and its address is that `Arc`'s pointer.
unsafe impl crate::kept::Handle for Texture {
    fn address(&self) -> usize {
        Arc::as_ptr(&self.inner) as *const () as usize
    }
}
