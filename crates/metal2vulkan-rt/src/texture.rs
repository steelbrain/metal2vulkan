//! Textures: optimal images placed in the device's texture blocks ([`crate::places`]), each
//! written into set 1's texture arrays at the index the consumer names, in one home layout for its
//! life. An image the device requires an allocation of its own for, or one larger than a block, or
//! one the blocks' memory type does not serve, has memory of its own.
//!
//! A texture's image has the descriptor's usage plus transfer source and destination, since Metal
//! lets the CPU and blits reach any texture. Its home layout is the one its descriptors name, and
//! the image leaves it only inside a copy: general where a shader may write it, shader-read-only
//! where shaders only read it, the attachment layout where only render passes use it. A new image
//! moves from undefined to home in the upload of the next commit, ahead of every command buffer
//! committed after [`Device::new_texture`] returns, or ahead of its first copy; nothing
//! holds it until then, so a texture released first is freed at once.
//!
//! A texture the runtime places (a new texture of [`Device::new_texture`],
//! [`Device::new_texture_for_views`] or [`Device::new_managed_texture`]) reads zero until written:
//! that upload clears every subresource of its image between the move from undefined and the
//! move home ([`Zero`]), on every device, its first texture in memory no resource used included.
//! Metal does not say what a new texture holds. The
//! runtime's choice is zero, as an Apple M3 Max answered for new Shared and Managed textures,
//! because programs read textures they never wrote, and the memory's last bytes would follow
//! process history.
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
//!
//! A drawable's texture ([`Device::new_drawable_texture`]) is over an image of the consumer's
//! swapchain, which the runtime never destroys. Its home layout is a texture's of its usage; it
//! moves there only once the consumer acquired it ([`Texture::drawable_acquired`]), in the upload
//! of the next commit as a new texture's image does, from undefined the first time and from
//! the present layout after a present, which keeps its texels. The command buffer that presents
//! it moves it from home to the present layout at its end
//! ([`CommandBuffer::present_drawable`](crate::CommandBuffer::present_drawable)).

use crate::arrays::{Owner, ResourceId};
use crate::buffer::{Buffer, BufferInner, BufferObject};
use crate::copy_out;
use crate::counters::{Counters, Gauge, Metric, Site};
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit, Refusal, RenderPassKind, TextureViewKind};
use crate::hazard::HeapId;
use crate::host_bytes::HostBytes;
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
use metal2vulkan::pipeline::PrecisionNote;
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
    /// `textureType`. `TextureBuffer` is refused, [`Gap::TextureBuffer`], and `CubeArray`,
    /// [`Gap::CubeArrayTexture`].
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
    /// render target and pixel-format view. The runtime keeps the field as given and answers no
    /// `usage` getter; Metal's texture reports its usage by three rules (measured on an Apple M3
    /// Max under macOS 27.0.1): no bits report those four usages (`0x17`), bits that name no
    /// usage are dropped (`0x8` reports `0`), and `ShaderAtomic` is dropped (an `R32Uint`
    /// texture of `0x21` reports `0x1`).
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

/// No Metal counterpart: a heap's memory, `len` bytes at `offset` in a block the consumer holds,
/// taken at the heap's creation; [`Device::new_heap_texture`] and
/// [`Device::new_heap_texture_for_views`] place textures in it (in a Private block) and
/// [`Device::new_heap_buffer`] buffers. The heap is the one at `offset` in `block`:
/// every resource placed in it takes one id in the compute encoders' hazard sets.
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
/// A cheap handle: clones share one texture. A handle holds the texture and the device; what a
/// command buffer keeps of it is the texture alone, which holds no handle of the device. Once its
/// last handle is gone, and with it the last command buffer that used it, its elements of set 1
/// read null again and its index is free; the reaper frees its image once the last upload that
/// moves or copies it has retired too.
#[derive(Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
    device: Device,
}

/// The texture itself: what a handle, a command buffer that uses it and a view of it hold.
pub(crate) struct TextureInner {
    core: Arc<Core>,
    descriptor: TextureDescriptor,
    image: Arc<Image>,
    /// The format of its views: the image's, or a view's own.
    format: vk::Format,
    /// Its views' format reads as storage without a format ([`Plan`]'s `storage_reads`).
    storage_reads: bool,
    /// The usage its views take where their format is not the image's (`VkImageViewUsageCreateInfo`):
    /// the image's that the format's features hold. `None` where its views take the image's.
    views_usage: Option<vk::ImageUsageFlags>,
    /// How bytes move between the image and the CPU or another image; `None` where no copy is
    /// built.
    unit: Option<CopyUnit>,
    /// Its index of the [`IndexClass::Texture`](crate::IndexClass::Texture) class and its owner
    /// word there; `None` for a texture no shader reaches.
    index: Option<(u32, Owner)>,
    word: u32,
    /// Its resource id in the encoders' hazard sets.
    id: u32,
    /// The heap it was placed in ([`Device::new_heap_texture`]), whose id its uses take in the
    /// compute encoders' hazard sets; a view's is its parent's.
    heap: Option<Arc<HeapId>>,
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
    /// The image's samples, one bit (`VkSampleCountFlags`).
    pub(crate) samples: vk::SampleCountFlags,
    /// The image renders as a linear colour attachment alone (`Image::linear_attachment`), so a
    /// pass renders it only beside other such images ([`Gap::LinearPlanePass`]).
    pub(crate) linear: bool,
}

/// A texture's bytes in a buffer, which [`Texture::replace_region`] and [`Texture::get_bytes`]
/// write and read there.
struct Named {
    buffer: Arc<BufferInner>,
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
    parent: Arc<TextureInner>,
    /// The view's first level and array layer in the image.
    base: [u32; 2],
    /// The swizzle its texels are sampled and read through: the one it was made with over its
    /// parent's, as Metal combines them, or its parent's where it was made with none.
    reads: TextureSwizzleChannels,
    views: Views,
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
    /// The sampled view: the sampled-texture array's element, where shaders read the texture, and
    /// the view a draw reads an attachment through (`attachment_reads`), which a colour render
    /// target not read has for that alone.
    sampled: Option<vk::ImageView>,
    /// The view of the storage-texture array's element, where shaders write it and the device
    /// stores its format through a storage image without a format.
    storage: Option<vk::ImageView>,
    /// The layout the image is in outside a copy, which its elements name.
    home: vk::ImageLayout,
    /// A draw may read an attachment of its own pass through a texture over the image (a colour
    /// render target not in place over a buffer, on a device that reads attachments; its home is
    /// general where it has a storage view, shader-read-only where shaders read it, else the
    /// attachment layout): each such texture has a second element of the sampled-texture array, at its index plus
    /// [`TEXTURES`](crate::floor::TEXTURES). Through the attachment feedback loop the image has
    /// the feedback usage and the element names the feedback layout, which a pass moves the whole
    /// image to from its home, and back, for a draw that reads it; on a device without the loop
    /// the element names the texture's view of the image's shadow (`shadowing`).
    attachment_reads: bool,
    /// Where a draw reads the image as an attachment on a device without the feedback loop: what
    /// its shadow is made of, and the shadow once made (`crate::shadow`).
    shadowing: Option<Box<Shadowing>>,
    /// The image is another's shadow: its drop lowers the shadow gauges, not the textures gauge.
    shadow: bool,
    /// Every aspect of the image's format, which a change of layout names.
    aspects: vk::ImageAspectFlags,
    /// The image's format and usage, and the formats it lists for views of another format (empty
    /// where it is not mutable).
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    view_formats: Vec<vk::Format>,
    /// The image is cube-compatible ([`Plan`]'s `cube`), so a cube view takes six of its layers.
    cube: bool,
    /// A surface plane's `LINEAR` image whose format renders only through
    /// `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV` (`crate::surface`): Vulkan renders it
    /// as a colour attachment only where every colour attachment of the pass is such an image,
    /// which a pass's begin holds ([`Gap::LinearPlanePass`]). False for every other image.
    linear_attachment: bool,
    /// The layout its move home starts from: undefined; preinitialized for a texture over a buffer
    /// in place, whose move home keeps the bytes.
    first: vk::ImageLayout,
    /// An upload recorded its move from `first` to `home`. Read and set under the work lock.
    homed: AtomicBool,
    /// How the move home zeroes it, for an image the runtime places for a texture of its own
    /// (`Source::Plain` and `Source::Managed`); `None` for every other image.
    zero: Option<Zero>,
    /// A drawable's image ([`Device::new_drawable_texture`]): whose turn it is between the
    /// consumer's acquires and presents. `None` for every other image.
    drawable: Option<Drawable>,
    /// What holds the memory the image lies in, where it is not the image's own, while anything
    /// holds the image.
    _under: Option<Under>,
}

/// A drawable's image's turn, between the consumer's acquires and presents.
struct Drawable {
    turn: Mutex<Turn>,
    /// The move home the last acquire owes starts in the present layout, which keeps the texels:
    /// the acquire followed a committed present. Else it starts undefined, as a swapchain's new
    /// image is. Decided at the acquire, as the submit thread records the move after the commit
    /// of the command buffer that presents the image again.
    from_present: AtomicBool,
}

/// Whose a drawable's image is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Turn {
    /// The presentation engine's, never acquired by the runtime.
    New,
    /// The presentation engine's, presented since its last acquire.
    Engine,
    /// The runtime's, acquired: in its home layout from the next upload on.
    Acquired,
    /// A command buffer not yet committed presents it.
    Presenting,
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
    /// A surface plane's image in place ([`Device::new_exported_surface`]): the opaque
    /// handle's allocation it is bound in, or none for a dma-buf's plane on its own import.
    Surface { _block: Option<Arc<Block>> },
}

/// How a new image the runtime places reads zero until written: the commands its move home
/// records between the move from undefined and the move to its home layout, which write every
/// texel of every subresource.
#[derive(Clone, Copy)]
pub(crate) enum Zero {
    /// A colour clear (`vkCmdClearColorImage`) of every level and layer, to zero.
    Color,
    /// A depth-stencil clear (`vkCmdClearDepthStencilImage`) of every aspect, level and layer,
    /// to depth 0 and stencil 0.
    DepthStencil,
    /// Copies from the zero source ([`crate::fill::zero_regions`]) for a single-sampled format no
    /// clear writes: a block-compressed one, or one of texel blocks wider than a texel
    /// (`VUID-vkCmdClearColorImage-image-00007`, `-01545`).
    Copy {
        unit: CopyUnit,
        extent: [u32; 3],
        levels: u32,
        layers: u32,
        volume: bool,
    },
}

impl Zero {
    /// The zeroing of an image of `plan`, of `levels` and of `shape`.
    fn of(plan: &Plan, levels: u32, shape: &Shape) -> Zero {
        if plan
            .aspects
            .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        {
            return Zero::DepthStencil;
        }
        match plan.unit {
            Some(unit) if unit.block != [1, 1] => Zero::Copy {
                unit,
                extent: shape.extent,
                levels,
                layers: shape.layers,
                volume: shape.image_type == vk::ImageType::TYPE_3D,
            },
            _ => Zero::Color,
        }
    }
}

/// A new image's move home, as an upload records it ([`Image::homing`]).
pub(crate) struct Homing {
    /// The move from the layout the image is made in: to its home layout, or to the transfer
    /// destination layout where the image is zeroed first.
    pub(crate) first: vk::ImageMemoryBarrier2<'static>,
    /// Where the image is zeroed: its zeroing, its memory's bytes, and its move home after it.
    pub(crate) zero: Option<Zeroing>,
}

/// A new image's zeroing in its move home ([`Homing`]).
pub(crate) struct Zeroing {
    pub(crate) image: vk::Image,
    pub(crate) aspects: vk::ImageAspectFlags,
    pub(crate) zero: Zero,
    pub(crate) bytes: u64,
    /// The move from the transfer destination layout to the image's home layout, after the
    /// zeroing's writes.
    pub(crate) home: vk::ImageMemoryBarrier2<'static>,
}

impl Zeroing {
    /// Records the zeroing into `command`: the clear of every subresource, or the copies from the
    /// zero source `zeros` that write every texel.
    ///
    /// # Safety
    ///
    /// `command` is recording, outside a render pass; the image is the device's, live until the
    /// command completes, and in the transfer destination layout (its move [`Homing::first`]
    /// recorded); `zeros` is the device's zero source.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        zeros: vk::Buffer,
    ) {
        let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        let range = [whole(self.aspects)];
        // SAFETY: the caller's contract. Every image the runtime makes has transfer-destination
        // usage, which its format query checked.
        unsafe {
            match self.zero {
                Zero::Color => device.cmd_clear_color_image(
                    command,
                    self.image,
                    layout,
                    &vk::ClearColorValue::default(),
                    &range,
                ),
                Zero::DepthStencil => device.cmd_clear_depth_stencil_image(
                    command,
                    self.image,
                    layout,
                    &vk::ClearDepthStencilValue::default(),
                    &range,
                ),
                Zero::Copy {
                    unit,
                    extent,
                    levels,
                    layers,
                    volume,
                } => device.cmd_copy_buffer_to_image(
                    command,
                    zeros,
                    self.image,
                    layout,
                    &crate::fill::zero_regions(unit, extent, levels, layers, volume),
                ),
            }
        }
    }
}

impl Image {
    /// The image's move from the layout it is made in to its home layout, where no upload has
    /// recorded it yet: recorded once, by the upload that is to hold the image (a drawable's once
    /// for each acquire, [`Work::home`](crate::submit::Work::home)), with its zeroing between
    /// two moves where it has one ([`Zero`]). Under the work lock.
    pub(crate) fn homing(&self) -> Option<Homing> {
        if self.homed.swap(true, Ordering::Relaxed) {
            return None;
        }
        // A drawable's image moves home at each acquire: from the present layout after a present.
        // The consumer waited for the acquire's fence, so the presentation engine's reads are
        // done and nothing earlier needs waiting for.
        let from_present = (self.drawable.as_ref())
            .is_some_and(|drawable| drawable.from_present.load(Ordering::Relaxed));
        let first = match from_present {
            true => vk::ImageLayout::PRESENT_SRC_KHR,
            false => self.first,
        };
        let every = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(whole(self.aspects));
        let Some(zero) = self.zero else {
            return Some(Homing {
                first: barrier
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(every)
                    .old_layout(first)
                    .new_layout(self.home),
                zero: None,
            });
        };
        // Cleared or copied over in the transfer destination layout, then moved home: the
        // clears and copies are transfers (`VK_PIPELINE_STAGE_2_CLEAR_BIT` is in
        // `ALL_TRANSFER`).
        let transfer = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        Some(Homing {
            first: barrier
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(first)
                .new_layout(transfer),
            zero: Some(Zeroing {
                image: self.image,
                aspects: self.aspects,
                zero,
                bytes: self.bytes,
                home: barrier
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(every)
                    .old_layout(transfer)
                    .new_layout(self.home),
            }),
        })
    }

    /// Whether its move home copies from the zero source ([`Zero::Copy`]).
    pub(crate) fn copies_zeros(&self) -> bool {
        matches!(self.zero, Some(Zero::Copy { .. }))
    }

    /// Its move home is owed, recorded by the next upload that holds it ([`homing`](Self::homing)):
    /// a new image's, or a drawable's at each acquire. Under the work lock.
    pub(crate) fn owe_home(&self) {
        self.homed.store(false, Ordering::Relaxed);
    }

    /// The layout a copy names the image in, for `transfer`, its role's transfer layout: home,
    /// general, for an image made preinitialized (a linear image over a buffer in place), which
    /// stays in a host-accessible layout through every use, copies included, so that it and the
    /// buffer are aliases that interpret their bytes alike (Vulkan's memory aliasing); `transfer`
    /// for any other image.
    pub(crate) fn copy_layout(&self, transfer: vk::ImageLayout) -> vk::ImageLayout {
        match self.first == vk::ImageLayout::PREINITIALIZED || self.surface() {
            true => self.home,
            false => transfer,
        }
    }

    /// Whether the image is a surface plane's in place ([`Device::new_exported_surface`]), which
    /// stays in the general layout through every copy, as the other API reads its bytes linearly.
    pub(crate) fn surface(&self) -> bool {
        matches!(self._under, Some(Under::Surface { .. }))
    }

    /// Whether its textures read it as an attachment through the attachment feedback loop: it has
    /// the feedback usage, and each texture's second element names the feedback layout.
    fn looped(&self) -> bool {
        self.attachment_reads && self.shadowing.is_none()
    }

    /// Its shadow, made where it has none yet: an image of its format, extent, levels and layers,
    /// bound as a texture's image is, whose move from undefined to its home layout,
    /// shader-read-only, the upload of the next commit records, ahead of every command buffer
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
            format: plan.format,
            usage: plan.usage,
            view_formats: plan.view_formats.clone(),
            cube: plan.cube,
            linear_attachment: false,
            first: vk::ImageLayout::UNDEFINED,
            homed: AtomicBool::new(false),
            zero: None,
            drawable: None,
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
    /// An allocation of its own, its bytes, and its bytes where it spilled out of device-local
    /// memory (else 0), counted in the textures' own bytes gauge.
    Own(vk::DeviceMemory, u64, u64),
    /// A place in a block the image holds: a heap's range ([`Device::new_heap_texture`]), or a
    /// buffer's bytes ([`Buffer::new_texture`]).
    InBlock,
    /// The consumer's swapchain's ([`Device::new_drawable_texture`]): neither the image nor its
    /// memory is the runtime's to free.
    Drawable,
    /// A surface plane's own dedicated import: of the surface's dma-buf
    /// ([`Device::new_exported_surface`]), or of a Direct3D 11 texture's handle
    /// (`Device::import_d3d11_surface`, on Windows), of its bytes, counted in the surface imports
    /// gauges.
    Import(vk::DeviceMemory, u64),
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

/// A Managed texture's named bytes of one region ([`TextureInner::named`]): the copy as they lie,
/// the buffer, the place, and the region's first element's offset in the buffer.
type NamedPart<'a> = (Transfer, &'a Arc<BufferInner>, BytesPlace, u64);

impl Drop for TextureInner {
    fn drop(&mut self) {
        let core = &self.core;
        let attachments =
            (self.attachments.get_mut()).unwrap_or_else(std::sync::PoisonError::into_inner);
        if !attachments.is_empty() {
            let views = attachments.drain(..).map(|a| a.view).collect();
            core.reaper.give(Job::Free(Free::AttachmentViews(views)));
        }
        // A view's id and image are its parent's, which it keeps.
        if self.view.is_none() {
            if let Some(named) = (self.named.as_ref()).filter(|named| named.form == Form::Copied) {
                core.counters.fall(Gauge::TexturesCopiedOver);
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
    /// memory, placed in a texture block ([`Texture`]), which reads zero until written: the
    /// runtime's choice, where Metal does not say what a new texture holds (the module docs).
    /// `index` is given exactly where shaders reach the texture (shader-read or shader-write
    /// usage, or render-target usage of a colour pixel format, which a fragment function reads
    /// through its attachment; [`Refusal::TextureIndex`]), and is then an index of a live
    /// [`IndexClass::Texture`](crate::IndexClass::Texture) grant, whatever the usage. A colour
    /// render target shaders neither read nor write writes only its second element there, the
    /// one a draw reads it through as an attachment, and a slot that binds it reads Metal's nil
    /// texture. The texture's elements of set 1 are written before the call
    /// returns, and its image moved to its home layout ahead of every command buffer committed
    /// after it. Until the submission of the next commit nothing else holds the texture: released
    /// first, it is freed at once.
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
    /// [`Gap::TextureFormat`], [`Gap::TextureBuffer`], [`Gap::CubeArrayTexture`] and
    /// [`Gap::MemorylessTexture`];
    /// [`Limit::MemoryAllocations`] where its memory would take the device past its count;
    /// [`Error::Vulkan`] where the image or its memory cannot be made; [`Error::Quiesced`].
    pub fn new_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        self.texture(descriptor, Reach::Index(index), Source::Plain)
    }

    /// No Metal counterpart of its own: a texture as [`Device::new_texture`] makes it, of any
    /// usage, with no index of its own. Shaders reach it only through its views
    /// ([`Texture::new_texture_view`]), each with an index and elements of set 1 of its own, as
    /// each Metal texture has a `gpuResourceID` of its own. It is the one image under textures
    /// the consumer makes as its views: the textures Metal makes over one IOSurface plane
    /// (`newTextureWithDescriptor:iosurface:plane:`), for one. Copies, uploads and readbacks reach
    /// it as they reach any texture. Where Metal gives every texture an index, this one has none:
    /// its [`Texture::gpu_resource_id`] answers index 0, and a slot that binds it reads Metal's
    /// nil texture.
    ///
    /// Err: as [`Device::new_texture`], but for [`Refusal::TextureIndex`] and
    /// [`Refusal::IndicesNotGranted`].
    pub fn new_texture_for_views(&self, descriptor: &TextureDescriptor) -> Result<Texture, Error> {
        self.texture(descriptor, Reach::Views, Source::Plain)
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
        let mut list = plan.format_list();
        let info = image_info(descriptor, &shape, &plan, &mut list);
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
                let Some((pitch, _, _)) = linear_layout(core, &d, &shape, &plan) else {
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
        linear_layout(self.core(), d, &shape, &plan).map(|(_, _, align)| align)
    }

    /// For the device tests: the bytes from its first texel that the device's linear image of `d`
    /// advertises for its subresource (`VkSubresourceLayout::size`), `None` where it makes none.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn linear_subresource_size(&self, d: &TextureDescriptor) -> Option<u64> {
        let (shape, plan, _) = self.prepare(d).ok()?;
        linear_layout(self.core(), d, &shape, &plan).map(|(_, size, _)| size)
    }

    /// `MTLHeap`'s `newTextureWithDescriptor:`, at `index`: a texture as [`Device::new_texture`]
    /// makes it, its image bound at `offset` in `heap`, where the consumer's own placement put it
    /// by [`Device::heap_texture_size_and_align`]. Nothing is allocated, and the bytes are those
    /// the heap's memory holds there, as Metal's heap textures are not zeroed. Textures may alias
    /// one another and the heap's buffers, as Metal's aliasable heap resources do; what one reads
    /// after another wrote is undefined, as in Metal. Its uses take the heap's id in the compute
    /// encoders' hazard sets, as a heap buffer's do ([`Device::new_heap_buffer`]). The texture
    /// holds the block.
    ///
    /// Err: as [`Device::new_texture`]; [`Error::NoSuchBlock`] for a block the consumer does not
    /// hold; [`Gap::HeapTexture`] for a Shared block or an exported one
    /// ([`Device::new_exported_block`](crate::Device::new_exported_block)), or as
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
        let source = self.heap_source(descriptor, heap, offset)?;
        self.texture(descriptor, Reach::Index(index), source)
    }

    /// `MTLHeap`'s `newTextureWithDescriptor:offset:`, for a texture reached through its views
    /// alone: a texture as [`Device::new_texture_for_views`] makes it, its image bound at
    /// `offset` in `heap` as [`Device::new_heap_texture`] binds one, by the same rule. Nothing is
    /// allocated, and the bytes are those the heap's memory holds there; its uses, and its
    /// views', take the heap's id in the compute encoders' hazard sets, and the texture holds the
    /// block. Where Metal's heap texture has a `gpuResourceID` of its own, this one has none, as
    /// [`Device::new_texture_for_views`] says: shaders reach it through its views
    /// ([`Texture::new_texture_view`]) alone.
    ///
    /// Err: as [`Device::new_texture_for_views`]; [`Error::NoSuchBlock`], [`Gap::HeapTexture`]
    /// and [`Refusal::HeapTexturePlace`] as [`Device::new_heap_texture`].
    pub fn new_heap_texture_for_views(
        &self,
        descriptor: &TextureDescriptor,
        heap: HeapRange,
        offset: u64,
    ) -> Result<Texture, Error> {
        let source = self.heap_source(descriptor, heap, offset)?;
        self.texture(descriptor, Reach::Views, source)
    }

    /// The source of a texture placed at `offset` in `heap`: the heap's block, which the consumer
    /// holds and which is a Private block of the pool's memory.
    ///
    /// Err: [`Error::NoSuchBlock`]; [`Gap::HeapTexture`] for a Shared or exported block.
    fn heap_source(
        &self,
        descriptor: &TextureDescriptor,
        heap: HeapRange,
        offset: u64,
    ) -> Result<Source<'static>, Error> {
        let block = self.block(heap.block)?;
        if block.memory.class != MemoryClass::Private || block.memory.own.is_some() {
            return Err(Error::Unsupported(Gap::HeapTexture {
                pixel_format: descriptor.pixel_format,
            }));
        }
        Ok(Source::Heap(block, heap, offset))
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
        self.texture(&d, Reach::Index(index), Source::Managed(bytes, places))
    }

    /// A descriptor's checks, as Metal makes them, and what its texture is made of: the shape,
    /// the plan, and the usage taken.
    fn prepare(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<(Shape, Plan, TextureUsage), Error> {
        let d = *descriptor;
        let (shape, usage) = own_checks(&d)?;
        let plan = plan(self.core(), &d, usage, &shape)?;
        Ok((shape, plan, usage))
    }

    /// No Metal counterpart of its making: the texture a `CAMetalDrawable` answers (`texture`),
    /// at `index`, over `image`, an image of the consumer's swapchain, made with `swapchain` on
    /// this device. The runtime never destroys the image or frees its memory: the consumer does,
    /// with its swapchain, once every command buffer that used the image has completed, and the
    /// move home of its last acquire with it, as the crate's guide says (Presenting, Destroying a
    /// swapchain); the texture's views may outlive the image. The texture's descriptor is a 2D texture of one level of the swapchain's image extent, its
    /// image's format the one the runtime holds the pixel format in, and its home layout, views,
    /// elements of set 1 and hazards a texture's of its usage, as [`Device::new_texture`] makes
    /// them. Its other-format views are [`Gap::TextureView`], and a draw that reads it as an
    /// attachment of its own pass is a [`Gap::Draw`], as the swapchain's image has no format list
    /// and no feedback usage.
    ///
    /// The image is the presentation engine's until the consumer acquires it: it is used from
    /// [`Texture::drawable_acquired`] to the commit of the command buffer that presents it
    /// ([`CommandBuffer::present_drawable`](crate::CommandBuffer::present_drawable)), and then
    /// again after the next acquire. Metal hands a drawable out from `nextDrawable` and takes it
    /// back at its present; here the consumer owns the swapchain, the acquires and the presents.
    ///
    /// Err: what [`Device::new_texture`] refuses of the descriptor but the device's image format
    /// query, which the swapchain's creation answered; [`Refusal::DrawableDescriptor`] for a
    /// descriptor that is not a 2D texture of one level of the swapchain's image extent, or a
    /// swapchain of more than one array layer; [`Refusal::DrawableFormat`] for a swapchain of
    /// another image format; [`Refusal::DrawableUsage`] for a swapchain whose image usage lacks
    /// some of the texture's; [`Refusal::TextureIndex`], [`Refusal::IndicesNotGranted`] as
    /// [`Device::new_texture`]; [`Error::Vulkan`] where a view cannot be made.
    ///
    /// # Safety
    ///
    /// Vulkan's rules on a swapchain's images, which the runtime cannot check:
    ///
    /// - `image` is one of the images `vkGetSwapchainImagesKHR` gives of the swapchain made on
    ///   this device with `swapchain` (its `pNext` chain is not read), whose sharing takes the
    ///   device's queue family
    ///   ([`RawDevice::queue_family_index`](crate::RawDevice::queue_family_index)), and no
    ///   drawable texture is over it while this one lives. It is not acquired, or acquired and in
    ///   the undefined layout, as `vkGetSwapchainImagesKHR` gives it.
    /// - The swapchain is destroyed only once every command buffer that used the texture has
    ///   completed, and a command buffer with commands committed after its last acquire has too;
    ///   the texture is not used after.
    pub unsafe fn new_drawable_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        image: vk::Image,
        swapchain: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<Texture, Error> {
        let d = *descriptor;
        let (shape, usage) = own_checks(&d)?;
        let extent = swapchain.image_extent;
        if d.texture_type != TextureType::Type2D
            || d.mipmap_level_count != 1
            || [d.width, d.height] != [extent.width, extent.height]
            || swapchain.image_array_layers != 1
        {
            return Err(Error::Refused(Refusal::DrawableDescriptor));
        }
        let core = self.core();
        let plan = form(core, &d, usage)?;
        if plan.format != swapchain.image_format {
            return Err(Error::Refused(Refusal::DrawableFormat {
                pixel_format: d.pixel_format,
                held: plan.format.as_raw(),
                swapchain: swapchain.image_format.as_raw(),
            }));
        }
        let missing = plan.usage & !swapchain.image_usage;
        if !missing.is_empty() {
            return Err(Error::Refused(Refusal::DrawableUsage {
                missing: missing.as_raw(),
            }));
        }
        if reached(d.pixel_format, usage) != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        // Views of the image's own format take its whole usage, which the swapchain's creation
        // checked against the format (`VUID-VkSwapchainCreateInfoKHR-imageFormat-01778`).
        let mut made = Made::default();
        // SAFETY: the image is the device's and bound (the caller's contract), of the plan's
        // format, and has the plan's usage.
        let views = unsafe { made.views(&core.device, image, &shape, &plan, plan.usage) };
        if views.is_err() {
            // SAFETY: nothing else holds the views; `made` holds no image.
            unsafe { made.destroy(core) };
        }
        let (sampled, storage) = views?;
        core.counters.rise(Gauge::Textures);
        let image = Arc::new(Image {
            core: Arc::clone(core),
            image,
            backing: Backing::Drawable,
            bytes: 0,
            sampled,
            storage,
            home: plan.home,
            attachment_reads: false,
            shadowing: None,
            shadow: false,
            aspects: plan.aspects,
            format: plan.format,
            usage: swapchain.image_usage,
            view_formats: Vec::new(),
            cube: false,
            linear_attachment: false,
            first: vk::ImageLayout::UNDEFINED,
            // No move home until the consumer acquires it.
            homed: AtomicBool::new(true),
            zero: None,
            drawable: Some(Drawable {
                turn: Mutex::new(Turn::New),
                from_present: AtomicBool::new(false),
            }),
            _under: None,
        });
        // The texture's elements of set 1, written as it takes its index; a refusal frees the
        // views as `image` drops.
        let index = match index {
            Some(index) => {
                let sampled = sampled.map(|view| (view, plan.home));
                // SAFETY: the texture's views are live; its image is in its home layout whenever
                // work uses it, from the upload after each acquire to its present.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (sampled, storage),
                        None,
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let inner = TextureInner {
            core: Arc::clone(core),
            descriptor: d,
            image,
            format: plan.format,
            storage_reads: plan.storage_reads,
            views_usage: None,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
            heap: None,
            named: None,
            view: None,
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow: None,
        };
        Ok(Texture {
            inner: Arc::new(inner),
            device: self.clone(),
        })
    }

    /// A texture of `descriptor`, reached as `reach` says, its image made as `source` says.
    fn texture(
        &self,
        descriptor: &TextureDescriptor,
        reach: Reach,
        source: Source<'_>,
    ) -> Result<Texture, Error> {
        let d = *descriptor;
        let (shape, plan, usage) = self.prepare(&d)?;
        let index = match reach {
            Reach::Index(index) if reached(d.pixel_format, usage) != index.is_some() => {
                return Err(Error::Refused(Refusal::TextureIndex { index }));
            }
            Reach::Index(index) => index,
            Reach::Views => None,
        };
        let core = self.core();
        // A heap's texture takes the heap's id in the hazard sets.
        let heap_id = match &source {
            Source::Heap(_, heap, _) => Some(core.heap_ids.of(core, heap.block, heap.offset)),
            _ => None,
        };
        let (named, first, home) = match &source {
            Source::Plain | Source::Heap(..) => (None, vk::ImageLayout::UNDEFINED, plan.home),
            Source::Managed(bytes, places) => {
                check_places(&d, plan.unit, bytes, places)?;
                let named = Named {
                    buffer: Arc::clone(bytes.object()),
                    places: places.to_vec(),
                    form: Form::Managed,
                };
                (Some(named), vk::ImageLayout::UNDEFINED, plan.home)
            }
            Source::Over(buffer, offset, bytes_per_row) => {
                let in_place =
                    check_over(core, &d, &shape, &plan, buffer, *offset, *bytes_per_row)?;
                let named = Named {
                    buffer: Arc::clone(buffer.object()),
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
        // A texture the runtime places reads zero until written ([`Zero`]): not one in a heap's
        // memory or over a buffer's bytes.
        let placed = matches!(source, Source::Plain | Source::Managed(..));
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
                let rows = u64::from(d.height) * bytes_per_row;
                let reach = (usage.contains(TextureUsage::RENDER_TARGET))
                    .then(|| render_reach(rows, buffer.length() - offset));
                let bind = Bind::InPlace {
                    memory: memory.device_memory(),
                    base: memory.base(),
                    memory_type: core.memory_type(class),
                    block_len: memory.len,
                    at: at + offset,
                    bytes_per_row,
                    reach,
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
        // A clear zeroes it in its move home.
        let zero = placed.then(|| Zero::of(&plan, d.mipmap_level_count, &shape));
        // An image in place over a buffer is the buffer's linear image, of no feedback usage
        // (`Plan::linear_usage`); any other keeps its home, shader-read-only or general, which a
        // pass moves it from to the feedback layout and back.
        let laid_over = named.as_ref().is_some_and(|n| n.form == Form::InPlace);
        let attachment_reads = plan.attachment_reads && !laid_over;
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
            format: plan.format,
            usage: plan.usage,
            view_formats: plan.view_formats.clone(),
            cube: plan.cube,
            linear_attachment: false,
            first,
            homed: AtomicBool::new(false),
            zero,
            drawable: None,
            _under: block,
        });
        let shadow = ShadowView::of(&image, &plan, index, shape.view_type, whole);
        // The texture's elements of set 1, written as it takes its index; a refusal frees the
        // image as `image` drops.
        let index = match index {
            Some(index) => {
                let element = sampled.filter(|_| plan.read).map(|view| (view, home));
                let looped = sampled.filter(|_| image.looped());
                // SAFETY: the texture's views are live; its image is in its home layout whenever
                // work uses it, the upload of the next commit going first.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (element, storage),
                        looped,
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let inner = TextureInner {
            core: Arc::clone(core),
            descriptor: d,
            image,
            format: plan.format,
            storage_reads: plan.storage_reads,
            views_usage: None,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
            heap: heap_id,
            named,
            view: None,
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow,
        };
        // Over a buffer of no device copy, the commands track what they write while it lives
        // (`crate::over`).
        let copied = (inner.named.as_ref()).filter(|named| named.form == Form::Copied);
        if copied.is_some() {
            core.counters.rise(Gauge::TexturesCopiedOver);
        }
        if copied.is_some_and(|named| named.buffer.held().1.is_none()) {
            core.copied_overs.fetch_add(1, Ordering::Relaxed);
        }
        let texture = Texture {
            inner: Arc::new(inner),
            device: self.clone(),
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
    /// The sampled view's component mapping and aspect, where shaders read the texture or, for
    /// a colour render target, where a draw may read it as an attachment of its own pass
    /// ([`Plan::reads_attachments`]); `plan` drops it from a colour render target whose draws
    /// read no attachment through it.
    sampled: Option<(vk::ComponentMapping, vk::ImageAspectFlags)>,
    /// Whether shaders read the texture: its sampled view is its sampled-texture element. A
    /// colour render target not read has the view for its second element alone, and its element
    /// stays the null one, which a slot that binds it reads as Metal's nil texture.
    read: bool,
    /// Whether the texture has a storage view.
    storage: bool,
    /// Whether a storage view of the plan's format reads without a format: the format's
    /// `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`, the format query a translator's
    /// storage read by the formats' own bits takes, which a device without
    /// `shaderStorageImageReadWithoutFormat` answers per format. A shader's read of a storage view
    /// whose format lacks it is outside Vulkan's valid usage
    /// (`VUID-vkCmdDispatch-OpTypeImage-07028`).
    storage_reads: bool,
    home: vk::ImageLayout,
    /// A draw may read an attachment of its own pass through the texture, and each sampled view
    /// has a second element ([`Image`]'s `attachment_reads`): the image has the feedback usage on
    /// a device with the attachment feedback loop, and a shadow on one without it.
    attachment_reads: bool,
    aspects: vk::ImageAspectFlags,
    unit: Option<CopyUnit>,
    word: u32,
    emulation: FormatEmulation,
    /// The format's compatibility class, which every format the image lists shares.
    class: &'static str,
    /// The usage a view in the plan's format takes: of sampling, storage and attachments, those
    /// the format's features hold.
    supported: vk::ImageUsageFlags,
    /// The formats the image lists for its views, its own first, where Metal views it as another
    /// format the device holds in the image format's class (`view_formats`); empty where it has no
    /// such view, and the image is not mutable.
    view_formats: Vec<vk::Format>,
    /// The image is made cube-compatible, so Metal's cube view of six of a 2D array's slices
    /// views it ([`Texture::new_texture_view`]): a square 2D array of six slices or more whose
    /// format the device's image format query makes so ([`plan`]). Vulkan takes the flag on such
    /// an image alone (`VUID-VkImageCreateInfo-flags-08865`, `-08866`); a 2D array is never
    /// multisample, which `VUID-VkImageCreateInfo-samples-02257` bars. A cube's own image is
    /// cube-compatible through its shape's flags.
    cube: bool,
}

impl Plan {
    /// The usage of a linear image of the plan, over a buffer in place: without the feedback
    /// usage, as no pass moves it to the feedback layout, and without the sampled usage where
    /// shaders do not read the texture, as a colour render target takes it for attachment reads
    /// alone.
    fn linear_usage(&self) -> vk::ImageUsageFlags {
        let mut usage = self.usage & !vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        if !self.read {
            usage &= !vk::ImageUsageFlags::SAMPLED;
        }
        usage
    }

    /// Whether a draw may read an attachment of its own pass through a texture of the plan, where
    /// its image takes the feedback usage or a shadow: a colour render target, read by shaders or
    /// not, with a storage view or not, on a device that reads attachments, as Metal reads any
    /// render target. Vulkan's feedback layout asks the attachment and sampled usages beside the
    /// feedback usage (`VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006`) and bars none, the
    /// storage usage included; in it the image is only rendered and sampled. A target not read
    /// takes the sampled usage for the reads alone (`plan`).
    fn reads_attachments(&self, core: &Core) -> bool {
        core.attachment_reads
            && self.sampled.is_some()
            && self.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            && self.aspects == vk::ImageAspectFlags::COLOR
    }

    /// Whether the device makes the shadow of an image of the plan over `shape`: its image format
    /// query of an optimal image of the plan's format, flags and view formats, with the shadow's
    /// usage.
    fn shadow_offered(&self, core: &Core, shape: &Shape) -> bool {
        let mut list = self.format_list();
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(self.format)
            .ty(shape.image_type)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(SHADOW_USAGE)
            .flags(self.flags(shape));
        if !self.view_formats.is_empty() {
            info = info.push_next(&mut list);
        }
        core.image_format_properties(&info).is_some()
    }

    /// The plan of the image's shadow (`crate::shadow`): its format, samples and aspects, with
    /// the shadow's usage and no view of its own.
    fn shadow(&self) -> Plan {
        Plan {
            usage: SHADOW_USAGE,
            sampled: None,
            storage: false,
            attachment_reads: false,
            view_formats: self.view_formats.clone(),
            ..*self
        }
    }

    /// The image's create flags over `shape`'s: mutable where it lists formats for its views, and
    /// cube-compatible where the plan makes it so.
    fn flags(&self, shape: &Shape) -> vk::ImageCreateFlags {
        let mut flags = shape.flags;
        if !self.view_formats.is_empty() {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        if self.cube {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        flags
    }

    /// The image's format list, which its create info and its format queries chain where it is
    /// mutable.
    fn format_list(&self) -> vk::ImageFormatListCreateInfo<'_> {
        vk::ImageFormatListCreateInfo::default().view_formats(&self.view_formats)
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
    let texel = over_texel(d, plan)?;
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
    // Over an imported block, the copy: an image bound there would be made for the import's
    // handle, so in the undefined layout (VUID-VkImageCreateInfo-pNext-01443), and the runtime
    // makes no image in imported memory. Over an exported block, the copy too: its memory is
    // dedicated to its buffer, and no image is bound in it (VUID-vkBindImageMemory-memory-10926).
    let own = copy.is_none() && block.memory.own.is_some();
    let spilled = copy.is_none() && block.memory.spilled();
    let extent = linear_extent(core, shape, plan)
        .is_some_and(|max| d.width <= max.width && d.height <= max.height);
    let (linear, optimal) = core.format_features(plan.format);
    let features = linear.contains(optimal & kept_features(plan.usage));
    Ok(!own && !spilled && extent && features)
}

/// The bytes of a texel of a texture over a buffer that the runtime makes: 2D of one level,
/// sample and slice, of a color format of one texel an element.
///
/// Err: [`Gap::TextureOverBuffer`] for any other.
fn over_texel(d: &TextureDescriptor, plan: &Plan) -> Result<u64, Error> {
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
    Ok(u64::from(unit.bytes))
}

/// The format features an image of `usage` is used by, which a linear image is to keep where the
/// device's optimal image of its format has them: copies (a blit encoder's copies and fills are
/// transfers; the runtime records no image blit); sampling and its linear filter; a shader's
/// reads, writes and atomics without a format; rendering and blending. A texture over a buffer
/// takes this test, as a surface plane in place does (`crate::surface::held_usage`).
pub(crate) fn kept_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags2 {
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
    let mut list = plan.format_list();
    let mut linear = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(plan.format)
        .ty(shape.image_type)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(plan.linear_usage())
        .flags(plan.flags(shape));
    if !plan.view_formats.is_empty() {
        linear = linear.push_next(&mut list);
    }
    core.image_format_properties(&linear)
        .map(|properties| properties.max_extent)
}

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

/// The bytes from its first texel that a render target over a buffer may reach in place: its own
/// `rows` (its height times its pitch) or the buffer's `rest` from it, the fewer. A store, a clear
/// or a resolve of an attachment may read and write back any byte of the subresource bound for it
/// (Vulkan's load and store operations), the bytes of its rows that are not texels among them, so
/// that subresource is to lie within the texture's own rows and the buffer.
pub(crate) fn render_reach(rows: u64, rest: u64) -> u64 {
    rows.min(rest)
}

/// Whether a linear image whose subresource the device advertises as `size` bytes from its place
/// lies in place within `reach`, a render target's ([`render_reach`]); a texture of any other
/// usage, `None`, writes its texels alone, and its subresource may end past them.
pub(crate) fn in_reach(size: u64, reach: Option<u64>) -> bool {
    reach.is_none_or(|reach| size <= reach)
}

/// The row pitch, the subresource's size and the memory alignment of a linear image of `d`,
/// `shape` and `plan`, where the device makes one: the image is made, read and destroyed, never
/// bound.
fn linear_layout(
    core: &Core,
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
) -> Option<(u64, u64, u64)> {
    let max = linear_extent(core, shape, plan)?;
    if d.width > max.width || d.height > max.height {
        return None;
    }
    let mut list = plan.format_list();
    let info = image_info(d, shape, plan, &mut list)
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
    Some((layout.row_pitch, layout.size, requirements.alignment))
}

/// The checks a texture's descriptor and a view's both take, with the shape and the usage taken:
/// an unknown usage is every usage, as Metal takes it.
fn checked(d: &TextureDescriptor) -> Result<(Shape, TextureUsage), Error> {
    let shape = shape(d)?;
    if d.storage_mode == StorageMode::Memoryless {
        return Err(Error::Unsupported(Gap::MemorylessTexture));
    }
    if d.texture_type == TextureType::CubeArray {
        return Err(Error::Unsupported(Gap::CubeArrayTexture));
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
    Ok((shape, usage))
}

/// The checks a texture's own descriptor takes, Metal's: [`checked`]'s, and a swizzle only of
/// read usage. With the shape and the usage taken.
fn own_checks(d: &TextureDescriptor) -> Result<(Shape, TextureUsage), Error> {
    let (shape, usage) = checked(d)?;
    if d.swizzle != TextureSwizzleChannels::DEFAULT
        && usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET)
    {
        return Err(Error::Refused(Refusal::TextureSwizzleWithWrites));
    }
    Ok((shape, usage))
}

/// `swizzle` over `parent`'s, as Metal combines a view's swizzle with its parent texture's: each
/// channel the view takes from red, green, blue or alpha is that channel of the parent as the
/// parent's swizzle reads it, and zero and one stay (Apple's example: `[R, G, A, B]` over
/// `[R, 1, 1, G]` is `[R, 1, G, 1]`).
pub(crate) fn combined(
    parent: TextureSwizzleChannels,
    swizzle: TextureSwizzleChannels,
) -> TextureSwizzleChannels {
    let of = |source| match source {
        TextureSwizzle::Red => parent.red,
        TextureSwizzle::Green => parent.green,
        TextureSwizzle::Blue => parent.blue,
        TextureSwizzle::Alpha => parent.alpha,
        other => other,
    };
    TextureSwizzleChannels {
        red: of(swizzle.red),
        green: of(swizzle.green),
        blue: of(swizzle.blue),
        alpha: of(swizzle.alpha),
    }
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
    let mut plan = form(core, d, usage)?;
    let pf = d.pixel_format;
    let gap = || Error::Unsupported(Gap::TextureFormat { pixel_format: pf });
    let profile = core.translator.profile();
    plan.view_formats = view_formats(profile, pf, usage, &plan);
    // A square 2D array of six slices or more is made cube-compatible where the device's image
    // format query takes the flag (`Plan::cube`).
    let mut cube =
        d.texture_type == TextureType::Type2DArray && d.width == d.height && d.array_length >= 6;
    let cube_flag = |cube: bool| match cube {
        true => vk::ImageCreateFlags::CUBE_COMPATIBLE,
        false => vk::ImageCreateFlags::empty(),
    };
    // A draw may read an attachment of its own pass through such a texture
    // (`crate::render_encoder`), where the device reads attachments: in the feedback layout where
    // it has the attachment feedback loop, else through a copy into the image's shadow. The image
    // takes the sampled usage for it where shaders do not read the texture.
    let mut attachment_reads = plan.reads_attachments(core);
    let looped = attachment_reads && core.feedback.is_some();
    let mut image_usage = plan.usage;
    if attachment_reads {
        image_usage |= vk::ImageUsageFlags::SAMPLED;
    }
    if looped {
        image_usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    let mut list = plan.format_list();
    let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(plan.format)
        .ty(shape.image_type)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(image_usage)
        .flags(plan.flags(shape) | cube_flag(cube));
    if !plan.view_formats.is_empty() {
        info = info.push_next(&mut list);
    }
    let mut properties = core.image_format_properties(&info);
    // A square 2D array whose format the device does not make cube-compatible is made without
    // the flag, and a cube view of it is `TextureViewKind::CubeOfArray`.
    if properties.is_none() && cube {
        cube = false;
        info = info.flags(plan.flags(shape));
        properties = core.image_format_properties(&info);
    }
    // A format the device makes without the usages the attachment reads add alone is made so,
    // and its draws read no attachment of their own pass through it.
    if properties.is_none() && attachment_reads {
        attachment_reads = false;
        image_usage = plan.usage;
        properties = core.image_format_properties(&info.usage(image_usage));
    }
    // Likewise a format the device makes no shadow of: a target shaders do not read is then made
    // without the sampled usage.
    if properties.is_some() && attachment_reads && !looped {
        attachment_reads = core
            .image_format_properties(&info.usage(SHADOW_USAGE))
            .is_some();
        if !attachment_reads && image_usage != plan.usage {
            image_usage = plan.usage;
            properties = core.image_format_properties(&info.usage(image_usage));
        }
    }
    let properties = properties.ok_or_else(gap)?;
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
    if !d.sample_count.is_power_of_two() || !properties.sample_counts.contains(plan.samples) {
        return Err(Error::Limit(Limit::TextureSamples {
            sample_count: d.sample_count,
            counts: properties.sample_counts.as_raw(),
        }));
    }
    plan.usage = image_usage;
    plan.attachment_reads = attachment_reads;
    plan.cube = cube;
    if !plan.read && !attachment_reads {
        plan.sampled = None;
    }
    Ok(plan)
}

/// Whether a format held so gives Metal's values through its view, with no conversion in the
/// shader.
fn viewed(emulation: FormatEmulation) -> bool {
    matches!(
        emulation,
        FormatEmulation::Direct
            | FormatEmulation::SwizzledView
            | FormatEmulation::AlphaInRed
            | FormatEmulation::RotatedView
            | FormatEmulation::XChannel
            | FormatEmulation::DepthInCombined
            | FormatEmulation::StencilInCombined
    )
}

/// The formats an image of `plan`, of a texture of `pf` and `usage`, lists for its views, its own
/// first: those Metal views such a texture as (its sRGB sibling, and with `PixelFormatView` usage
/// every format `metal::format::views_with_usage` takes) that the device holds for the usage in a
/// viewed form of the image format's class. Empty where there is none but its own: the image is
/// then not mutable.
fn view_formats(
    profile: &metal2vulkan::device::DeviceProfile,
    pf: PixelFormat,
    usage: TextureUsage,
    plan: &Plan,
) -> Vec<vk::Format> {
    let others: Vec<PixelFormat> = match usage.contains(TextureUsage::PIXEL_FORMAT_VIEW) {
        true => (PixelFormat::ALL.iter().copied())
            .filter(|&view| view != pf && formats::views_with_usage(pf, view))
            .collect(),
        false => formats::srgb_sibling(pf).into_iter().collect(),
    };
    let mut list = vec![plan.format];
    for view in others {
        let usages = formats::texture_usages(view, usage);
        let Ok(mapping) = formats::vk_format(view, &usages, profile) else {
            continue;
        };
        let class = mapping.format.block().map(|block| block.class);
        let format: vk::Format = mapping.format.into();
        if viewed(mapping.emulation) && class == Some(plan.class) && !list.contains(&format) {
            list.push(format);
        }
    }
    if list.len() == 1 {
        list.clear();
    }
    list
}

/// Whether shaders reach a texture of `pf` and `usage` (Metal's unknown usage already its four
/// bits), which takes an index exactly where they do ([`Refusal::TextureIndex`]): it is read or
/// written, or it is a colour render target (a pixel format with no depth or stencil), which a
/// fragment function reads through its attachment, as Metal gives every texture a
/// `gpuResourceID`. A depth or stencil target shaders neither read nor write is not reached.
fn reached(pf: PixelFormat, usage: TextureUsage) -> bool {
    usage.intersects(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE)
        || (usage.contains(TextureUsage::RENDER_TARGET) && !formats::is_depth_stencil(pf))
}

/// What a texture of `d` and `usage` holds in its views, as its own image or as a view of
/// another's: the format and how it holds Metal's, the aspects, the sampled view's components and
/// aspect, whether it has a storage view, the image usage its uses ask, its home layout, the copy
/// unit and the word. An image's own parts, its format list, the feedback usage and the format
/// query's checks, are `plan`'s.
fn form(core: &Core, d: &TextureDescriptor, usage: TextureUsage) -> Result<Plan, Error> {
    let pf = d.pixel_format;
    let gap = || Error::Unsupported(Gap::TextureFormat { pixel_format: pf });
    let word = TextureWord::of(pf, d.swizzle).ok_or_else(gap)?.0;
    let profile = core.translator.profile();
    let mapping =
        formats::vk_format(pf, &formats::texture_usages(pf, usage), profile).map_err(|_| gap())?;
    if !viewed(mapping.emulation) {
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
    let storage_reads =
        profile.supports_format(mapping.format, FormatUsage::StorageImageReadWithoutFormat);
    let render = usage.contains(TextureUsage::RENDER_TARGET);
    let supports = |use_| profile.supports_format(mapping.format, use_);
    let mut supported = vk::ImageUsageFlags::empty();
    for (use_, flag) in [
        (FormatUsage::SampledImage, vk::ImageUsageFlags::SAMPLED),
        (FormatUsage::StorageImage, vk::ImageUsageFlags::STORAGE),
        (
            FormatUsage::ColorAttachment,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ),
        (
            FormatUsage::DepthStencilAttachment,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
    ] {
        if supports(use_) {
            supported |= flag;
        }
    }
    if supported.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        supported |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
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
    // A sampled view selects one aspect: depth where the format has it, else stencil.
    let view_aspect = if metal_aspects.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else if metal_aspects.contains(vk::ImageAspectFlags::STENCIL) {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let unit = copy_unit(pf, format, metal_aspects, block.size, d.sample_count);
    // A colour render target has a sampled view where shaders read it or a draw may read it as an
    // attachment of its own pass; `plan` gives an image the sampled usage for the second.
    let color_target = image_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    let sampled = match read || color_target {
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
        read,
        storage,
        storage_reads,
        home,
        attachment_reads: false,
        aspects,
        unit,
        word,
        emulation: mapping.emulation,
        class: block.class,
        supported,
        view_formats: Vec::new(),
        cube: false,
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

/// An image view of `range` of `image`, of `view_type` and `format`, with `components`, and with
/// `usage` where its format is another than the image's (`VkImageViewUsageCreateInfo`: the image's
/// usage that the format's features hold), else the image's.
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
    usage: Option<vk::ImageUsageFlags>,
) -> Result<vk::ImageView, Error> {
    let mut usage_info = vk::ImageViewUsageCreateInfo::default();
    let mut info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .components(components)
        .subresource_range(range);
    if let Some(usage) = usage {
        usage_info = usage_info.usage(usage);
        info = info.push_next(&mut usage_info);
    }
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

/// The image, its memory and the memory's bytes, and its views (the sampled view and the storage
/// view).
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

/// Where shaders reach a texture the device makes.
enum Reach {
    /// At its own index, given exactly where its usage reaches shaders.
    Index(Option<u32>),
    /// Through its views alone ([`Device::new_texture_for_views`]): no index of its own.
    Views,
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
    /// `bytes_per_row` apart from there; of a render target, where its subresource ends within
    /// `reach` bytes of there ([`in_reach`]).
    InPlace {
        memory: vk::DeviceMemory,
        base: u64,
        memory_type: u32,
        block_len: u64,
        at: u64,
        bytes_per_row: u64,
        reach: Option<u64>,
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
fn image_info<'a, 'b: 'a>(
    d: &TextureDescriptor,
    shape: &Shape,
    plan: &Plan,
    list: &'a mut vk::ImageFormatListCreateInfo<'b>,
) -> vk::ImageCreateInfo<'a> {
    let [width, height, depth] = shape.extent;
    let info = vk::ImageCreateInfo::default()
        .flags(plan.flags(shape))
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
    match plan.view_formats.is_empty() {
        true => info,
        false => info.push_next(list),
    }
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
        let mut list = plan.format_list();
        let info = image_info(d, shape, plan, &mut list);
        let info = match bind {
            Bind::InPlace { .. } => info
                .tiling(vk::ImageTiling::LINEAR)
                .usage(plan.linear_usage())
                .initial_layout(vk::ImageLayout::PREINITIALIZED),
            Bind::Pooled | Bind::Heap { .. } => info,
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
            reach,
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
                && in_reach(layout.size, reach)
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
            let (memory, spilled) = unsafe {
                memory::allocate_spilling(
                    core,
                    MemoryClass::Private,
                    bits,
                    &allocate,
                    Some(Site::Texture),
                )
            }?;
            let spilled = match spilled {
                true => requirements.size,
                false => 0,
            };
            core.counters.rise_by(Gauge::SpilledBytes, spilled);
            core.counters
                .rise_by(Gauge::TextureOwnBytes, requirements.size);
            (Backing::Own(memory, requirements.size, spilled), memory, 0)
        };
        self.backing = Some(backing);
        // SAFETY: the place or allocation holds the image's size at its alignment, in memory of a
        // type its requirements allow.
        unsafe { device.bind_image_memory(image, memory, offset) }
            .map_err(Error::vulkan("vkBindImageMemory"))?;
        // SAFETY: the image is bound, of the plan's format, and has `info`'s usage.
        let usage = info.usage;
        let (sampled, storage) = unsafe { self.views(device, image, shape, plan, usage) }?;
        Ok((image, backing, requirements.size, sampled, storage))
    }

    /// Makes `image`'s views of `plan`: its sampled view, where it has one and the image's
    /// `usage` the sampled usage, and its storage view, where it has one.
    ///
    /// # Safety
    ///
    /// `image` is the device's, bound, of the plan's format, and has `usage`, the storage usage
    /// among it where the plan has a storage view.
    unsafe fn views(
        &mut self,
        device: &ash::Device,
        image: vk::Image,
        shape: &Shape,
        plan: &Plan,
        usage: vk::ImageUsageFlags,
    ) -> Result<(Option<vk::ImageView>, Option<vk::ImageView>), Error> {
        let mut view = |components, aspect| {
            let (view_type, format) = (shape.view_type, plan.format);
            // SAFETY: the caller's contract.
            let range = whole(aspect);
            let view =
                unsafe { image_view(device, image, view_type, format, components, range, None) }?;
            self.views.push(view);
            Ok::<_, Error>(view)
        };
        let sampled = match plan.sampled {
            Some((components, aspect)) if usage.contains(vk::ImageUsageFlags::SAMPLED) => {
                Some(view(components, aspect)?)
            }
            _ => None,
        };
        let storage = match plan.storage {
            true => Some(view(
                vk::ComponentMapping::default(),
                vk::ImageAspectFlags::COLOR,
            )?),
            false => None,
        };
        Ok((sampled, storage))
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
                Some(Backing::Own(memory, bytes, spilled)) => {
                    device.free_memory(memory, None);
                    core.counters.fall(Gauge::Blocks);
                    core.counters.fall_by(Gauge::TextureOwnBytes, bytes);
                    core.counters.fall_by(Gauge::SpilledBytes, spilled);
                }
                Some(Backing::Placed(place)) => core.places.release(core, place),
                // A surface's plane images are made by `crate::surface` and `crate::d3d11`,
                // never here.
                Some(Backing::Import(..)) => {}
                Some(Backing::InBlock | Backing::Drawable) | None => {}
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

    /// The barrier that moves `placed`'s level and layer, every aspect, from home to the copy
    /// layout of `layout`, a transfer layout ([`Image::copy_layout`]: home itself for an image
    /// over a buffer in place), for a copy's `access` (its reads or its writes), after every
    /// earlier command's writes.
    fn away(
        &self,
        placed: &Placed,
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        use vk::AccessFlags2 as Access;
        use vk::PipelineStageFlags2 as Stage;
        self.barrier(placed, self.image.home, self.image.copy_layout(layout))
            .src_stage_mask(Stage::ALL_COMMANDS)
            .src_access_mask(Access::MEMORY_WRITE)
            .dst_stage_mask(Stage::COPY)
            .dst_access_mask(access)
    }

    /// The barrier that moves `placed`'s level and layer back home from the copy layout of
    /// `layout` once a copy has run, its writes (a copy into the image, whose transfer layout is
    /// `TRANSFER_DST_OPTIMAL`) visible to every later command.
    fn back(&self, placed: &Placed, layout: vk::ImageLayout) -> vk::ImageMemoryBarrier2<'static> {
        use vk::AccessFlags2 as Access;
        use vk::PipelineStageFlags2 as Stage;
        let written = match layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => Access::TRANSFER_WRITE,
            _ => Access::NONE,
        };
        self.barrier(placed, self.image.copy_layout(layout), self.image.home)
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
    ) -> Result<Option<NamedPart<'_>>, Error> {
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
    fn over(&self) -> Option<&Arc<BufferInner>> {
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
    /// made visible to the host. Its two barriers are counted in `counters`.
    ///
    /// # Safety
    ///
    /// `command` is recording under the work lock, and the upload holds the image and the
    /// buffer's memory.
    unsafe fn record(
        &self,
        device: &ash::Device,
        counters: &Counters,
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
        let named = self.image.copy_layout(layout);
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
        // SAFETY: the caller's contract; the subresource is in `named` between the barriers.
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
                    named,
                    &[region],
                ),
                false => device.cmd_copy_image_to_buffer(
                    command,
                    self.image.image,
                    named,
                    buffer,
                    copy_out::regions(&copy.unit, region).as_slice(),
                ),
            }
            device.cmd_pipeline_barrier2(command, &back);
        }
        counters.count(Metric::Barriers);
        counters.count(Metric::Barriers);
    }
}

impl TextureInner {
    /// For the device tests: whether its image has a storage view, its home general.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn has_storage_view(&self) -> bool {
        self.image.storage.is_some()
    }

    /// Where the texture starts in its image: a view's first level and array layer, else the
    /// image's first.
    fn base(&self) -> [u32; 2] {
        self.view.as_ref().map_or([0, 0], |view| view.base)
    }

    /// The swizzle its texels are sampled and read through: a view's over its parent's, else its
    /// descriptor's.
    fn reads(&self) -> TextureSwizzleChannels {
        (self.view.as_ref()).map_or(self.descriptor.swizzle, |view| view.reads)
    }

    /// A layout change of the texture's levels `levels`, counted from its first, over each of its
    /// slices and every aspect, from `old` to `new`, `None` naming the image's home layout, with
    /// no scopes yet.
    fn levels_barrier(
        &self,
        levels: Range<u32>,
        old: Option<vk::ImageLayout>,
        new: Option<vk::ImageLayout>,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let [level, layer] = self.base();
        let home = self.image.home;
        vk::ImageMemoryBarrier2::default()
            .old_layout(old.unwrap_or(home))
            .new_layout(new.unwrap_or(home))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.image.aspects,
                base_mip_level: level + levels.start,
                level_count: levels.end - levels.start,
                base_array_layer: layer,
                layer_count: slices(&self.descriptor),
            })
    }
}

impl Texture {
    /// A Managed texture's named bytes ([`Device::new_managed_texture`]), as a handle of the
    /// buffer, and their places, one for each level and, within it, each slice. `None` for any
    /// other texture.
    pub(crate) fn named_bytes(&self) -> Option<(Buffer, &[BytesPlace])> {
        let named = (self.inner.named.as_ref()).filter(|named| named.form == Form::Managed)?;
        let bytes = Buffer::of(Arc::clone(&named.buffer), self.device.clone());
        Some((bytes, &named.places))
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
    /// Built: the texture's own pixel format or another its image lists, over every slice in the
    /// texture's own type, a slice subrange of a 1D or 2D array, a 2D texture as a 2D array of its
    /// one slice, one slice of a 2D array as a 2D texture, a cube's faces as a 2D texture (one
    /// face) or a 2D array (any faces), and six slices of a square 2D array as a cube, its image
    /// made cube-compatible where the device's format query takes the flag, each over any level
    /// subrange. A view as a cube array is
    /// [`Gap::CubeArrayTexture`], as a cube array texture is. A texture's image lists the formats Metal views
    /// it as that the device holds in a viewed form of the image format's compatibility class: its
    /// sRGB sibling whatever its usage, and with `PixelFormatView` usage, or an empty usage
    /// (Metal's `MTLTextureUsageUnknown`, which the runtime takes as every usage), every
    /// uncompressed colour format of the same bytes a texel; such an image is mutable, with that
    /// format list. A view
    /// of another format takes the image's usage that its format's features hold: it has no
    /// storage element where the image has no storage usage, and renders where its format holds
    /// an attachment, else [`Gap::TextureFormat`] at the pass. `index` is given exactly where shaders reach the view, as
    /// [`Device::new_texture`] takes it; its usage, storage mode, sample count and swizzle are the
    /// texture's.
    ///
    /// Err: [`Refusal::TextureViewRange`] for an empty range of levels or slices or one past the
    /// texture's, [`Refusal::TextureViewSlices`] for a slice count the view's type does not take,
    /// and [`Refusal::TextureViewFormat`] for another pixel format of a texture made with a usage
    /// that is not empty and lacks `PixelFormatView`, as Metal's validation stops them;
    /// [`Gap::TextureView`] for another kind Metal makes, naming it;
    /// [`Refusal::TextureIndex`], [`Refusal::IndicesNotGranted`] as [`Device::new_texture`];
    /// [`Error::Vulkan`] where a view cannot be made.
    pub fn new_texture_view(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u32>,
        slices: Range<u32>,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        self.view(pixel_format, texture_type, levels, slices, None, index)
    }

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`: the view
    /// [`Texture::new_texture_view`] makes, its texels sampled and read through `swizzle`
    /// combined with this texture's own, as Metal combines them: each channel the view takes from
    /// red, green, blue or alpha is that channel of this texture as its own swizzle reads it. The
    /// view's descriptor holds `swizzle` as given, which is what Metal's `swizzle` answers of the
    /// view, and its resource ID the `TextureWord` of the combined one; a view made without a
    /// swizzle ([`Texture::new_texture_view`]) holds this texture's, and reads as it does. Metal
    /// takes the call whatever the texture's usage (its own descriptor refuses a swizzle
    /// with shader-write or render-target usage), and so does this.
    ///
    /// Apple defines the swizzle for sampling and reading. The view's sampled element (`sample`,
    /// and `read` of an `access::read` texture) takes the combined swizzle. Its storage element
    /// and its attachment views map each channel to itself, as Vulkan requires of a storage image
    /// and of an attachment: a render pass and a shader's write store the texels unswizzled. A
    /// slot that reads through the storage element, `access::read_write`, would read them
    /// unswizzled: a draw or dispatch that binds the view there is [`Gap::SwizzledReadWrite`].
    ///
    /// Err: as [`Texture::new_texture_view`]; [`Gap::TextureFormat`] for a swizzle source this
    /// crate does not know.
    pub fn new_texture_view_with_swizzle(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u32>,
        slices: Range<u32>,
        swizzle: TextureSwizzleChannels,
        index: Option<u32>,
    ) -> Result<Texture, Error> {
        let swizzle = Some(swizzle);
        self.view(pixel_format, texture_type, levels, slices, swizzle, index)
    }

    /// The view [`Texture::new_texture_view_with_swizzle`] makes, of `swizzle` where the call
    /// states one, else reading as this texture does ([`Texture::new_texture_view`]).
    fn view(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u32>,
        slices: Range<u32>,
        swizzle: Option<TextureSwizzleChannels>,
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
        let viewable = match p.usage.is_empty() || p.usage.contains(TextureUsage::PIXEL_FORMAT_VIEW)
        {
            true => formats::views_with_usage(p.pixel_format, pixel_format),
            false => formats::views_without_usage(p.pixel_format, pixel_format),
        };
        if !viewable {
            return Err(Error::Refused(Refusal::TextureViewFormat {
                texture: p.pixel_format,
                view: pixel_format,
            }));
        }
        let every_slice = slice_count == self::slices(p);
        let fits = match texture_type {
            T::Type2D => slice_count == 1,
            T::Cube => slice_count == 6,
            T::CubeArray => slice_count.is_multiple_of(6),
            _ => true,
        };
        if !fits {
            return Err(Error::Refused(Refusal::TextureViewSlices {
                texture_type,
                slice_count,
            }));
        }
        // Metal's casts (Apple's `makeTextureView(pixelFormat:textureType:levels:slices:)`): 1D to
        // 1D, 2D to 2D or a 2D array, a 2D array, a cube or a cube array to any of those four, and
        // 3D to 3D. Vulkan views a cube's image, which is cube-compatible, as a 2D texture or a 2D
        // array of its layers, and any image as a 2D array of one layer, with no query, so each is
        // one form on every device. A cube array is not built, nor a view as one.
        match (p.texture_type, texture_type) {
            (_, T::CubeArray) => return Err(Error::Unsupported(Gap::CubeArrayTexture)),
            (from, to) if from == to => {
                if !every_slice && !matches!(to, T::Type1DArray | T::Type2DArray) {
                    return gap(TextureViewKind::Slices);
                }
            }
            (T::Type2D, T::Type2DArray) | (T::Type2DArray, T::Type2D) => {}
            (T::Cube, T::Type2D | T::Type2DArray) => {}
            (T::Type2DArray, T::Cube) if inner.image.cube => {}
            (T::Type2DArray, T::Cube) => return gap(TextureViewKind::CubeOfArray),
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
        d.pixel_format = pixel_format;
        // The view reads through its swizzle over this texture's, and its descriptor answers the
        // swizzle it was made with.
        let reads = swizzle.map_or(inner.reads(), |swizzle| combined(inner.reads(), swizzle));
        d.swizzle = reads;
        // A view takes its descriptor's checks but the swizzle's with its usage, which Metal
        // makes of a texture's own descriptor alone.
        let (shape, usage) = checked(&d)?;
        let mut plan = form(&inner.core, &d, usage)?;
        // Another format is built where the image lists it; its views take the image's usage
        // that the format's features hold.
        let image = &inner.image;
        let views_usage = match plan.format == image.format {
            true => None,
            false if image.view_formats.contains(&plan.format) => {
                Some(image.usage & plan.supported)
            }
            false if image.surface() => return gap(TextureViewKind::SurfacePixelFormat),
            false => return gap(TextureViewKind::PixelFormat),
        };
        let usable = views_usage.unwrap_or(image.usage);
        plan.storage &= usable.contains(vk::ImageUsageFlags::STORAGE);
        if !usable.contains(vk::ImageUsageFlags::SAMPLED) {
            plan.sampled = None;
        }
        if reached(d.pixel_format, usage) != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        let core = &inner.core;
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
        let make = |(components, aspect, role): (_, _, vk::ImageUsageFlags)| {
            let (view_type, format) = (shape.view_type, plan.format);
            let usage = views_usage.map(|usage| usage & role);
            // SAFETY: the image is bound, lists the view's format, has the view's usage, and holds
            // the range: the texture's levels and layers hold the view's.
            unsafe {
                image_view(
                    &core.device,
                    image,
                    view_type,
                    format,
                    components,
                    range(aspect),
                    usage,
                )
            }
        };
        let sampled_role =
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        let sampled = (plan.sampled).map(|(components, aspect)| (components, aspect, sampled_role));
        let storage = (plan.storage).then_some((
            vk::ComponentMapping::default(),
            vk::ImageAspectFlags::COLOR,
            vk::ImageUsageFlags::STORAGE,
        ));
        let mut views = Views {
            core: Arc::clone(core),
            sampled: None,
            storage: None,
        };
        let mut made = || {
            views.sampled = sampled.map(make).transpose()?;
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
                let element = (sampled.filter(|_| plan.read)).map(|view| (view, inner.image.home));
                let looped = sampled.filter(|_| inner.image.looped());
                // SAFETY: the view's image views are live; the image is in its home layout
                // whenever work uses it.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (element, storage),
                        looped,
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let parent = match &inner.view {
            Some(view) => Arc::clone(&view.parent),
            None => Arc::clone(&self.inner),
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
            core: Arc::clone(&inner.core),
            descriptor: TextureDescriptor {
                swizzle: swizzle.unwrap_or(p.swizzle),
                ..d
            },
            image: Arc::clone(&inner.image),
            format: plan.format,
            storage_reads: plan.storage_reads,
            views_usage,
            unit: plan.unit,
            index,
            word: plan.word,
            id: inner.id,
            heap: inner.heap.clone(),
            named,
            view: Some(View {
                parent,
                base,
                reads,
                views,
            }),
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow,
        };
        Ok(Texture {
            inner: Arc::new(view),
            device: self.device.clone(),
        })
    }

    /// No Metal counterpart: the runtime's part of `CAMetalLayer`'s `nextDrawable`, for a
    /// drawable's texture ([`Device::new_drawable_texture`]) whose swapchain image the consumer has
    /// acquired. The image moves to its home layout in the upload of the next commit, ahead of
    /// every command buffer committed after the call: from the present layout where a command
    /// buffer presented it, which keeps its texels as Vulkan keeps a presented image's, else from
    /// undefined. The texture is used from here until the commit of the command buffer that
    /// presents it ([`CommandBuffer::present_drawable`](crate::CommandBuffer::present_drawable)).
    ///
    /// The consumer acquires with a fence and waits for it before the call: the move waits on no
    /// semaphore, so the presentation engine's reads of the image are to be done.
    ///
    /// Err: [`Refusal::NotDrawable`] for a texture not over a swapchain's image;
    /// [`Refusal::DrawableAcquired`] for a drawable acquired and not presented since;
    /// [`Error::Quiesced`]; [`Error::Vulkan`] where the upload cannot be begun.
    ///
    /// # Safety
    ///
    /// The consumer acquired the image (`vkAcquireNextImageKHR`) since its texture was made or
    /// last presented, and waited for the acquire's fence; its swapchain is not destroyed. These
    /// are Vulkan's rules on a swapchain's image, which the runtime cannot check.
    pub unsafe fn drawable_acquired(&self) -> Result<(), Error> {
        let image = &self.inner.image;
        let drawable = (image.drawable.as_ref()).ok_or(Error::Refused(Refusal::NotDrawable))?;
        let before = {
            let mut turn = lock(&drawable.turn);
            let before = *turn;
            let from_present = match before {
                Turn::New => false,
                Turn::Engine => true,
                Turn::Acquired | Turn::Presenting => {
                    return Err(Error::Refused(Refusal::DrawableAcquired))
                }
            };
            // Before the move is owed: the submit thread reads it as it records the move.
            drawable.from_present.store(from_present, Ordering::Relaxed);
            *turn = Turn::Acquired;
            before
        };
        let core = &self.inner.core;
        let homed = core.work.home(core, image);
        if homed.is_err() {
            *lock(&drawable.turn) = before;
        }
        homed
    }

    /// `gpuResourceID`: its index (0 for a texture no shader reaches) and the translator's
    /// `TextureWord` of its pixel format and swizzle.
    pub fn gpu_resource_id(&self) -> ResourceId {
        TextureObject::gpu_resource_id(self)
    }

    /// `textureType`, `pixelFormat`, `width` and the rest, as made.
    pub fn descriptor(&self) -> &TextureDescriptor {
        TextureObject::descriptor(self)
    }

    /// No Metal counterpart: the static note of a read, sample or gather of its texels, where its
    /// own pixel format is block-compressed ([`PrecisionNote::CompressedTexelDecode`]: a texel is
    /// the device's decode of its blocks, where Metal interpolates with 6-bit weights); `None` for
    /// any other format. Decided by the pixel format alone, on every device: a view answers by
    /// its own format, and its blocks' bytes, copies and sizes are Metal's either way.
    pub fn precision_note(&self) -> Option<PrecisionNote> {
        let compressed = formats::block_extent(self.inner.descriptor.pixel_format).is_some();
        compressed.then_some(PrecisionNote::CompressedTexelDecode)
    }

    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`: the bytes of
    /// `region` of `level` and `slice` (a 3D texture's slice is 0), rows `bytes_per_row` and
    /// images `bytes_per_image` apart in `bytes`, a compressed format's rows being rows of
    /// blocks. The call reads `bytes` in place, each byte of the region's rows once, by raw
    /// copies ([`HostBytes`]): the CPU copies them once, at the call, as Metal's does. The bytes go
    /// to the device's staging region before the call returns (a staging buffer of their own past
    /// the region's 16 MiB), and the upload copies them into the image ahead of every command
    /// buffer committed after it, so the call is ordered as Metal's CPU write is. Where the region has no room, the call first waits for the oldest copy in it to
    /// complete. Counted in [`Metrics::bytes_copied`]. An empty region copies nothing.
    ///
    /// A Managed texture with named bytes ([`Device::new_managed_texture`]) takes them there
    /// instead, as Metal writes a Managed texture's CPU copy, and then
    /// [`Texture::upload_region`] copies them into the image; nothing is staged or counted. A
    /// texture over a buffer ([`Buffer::new_texture`]) takes them in the buffer, which is its
    /// image's memory; nothing is staged, copied or counted.
    ///
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler;
    /// [`Refusal::TextureBytesPrivate`]; [`Refusal::TextureRegion`] for a level, slice or
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
        bytes: HostBytes<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<(), Error> {
        crate::command::not_in_scheduled_handler()?;
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
                    bytes.copy_to(from, host.add(to), len)
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
        let core = &inner.core;
        let stage = core.work.stage(core, copy.len(), false)?;
        let host = stage.host().as_ptr();
        for (from, to, len) in copy.runs() {
            // SAFETY: `copy` checked that each row lies in `bytes`, and the stage holds every row
            // packed; it is this call's alone until its copy is recorded.
            unsafe { bytes.copy_to(from, host.add(to), len) };
        }
        core.counters.copied(copy.len());
        core.work.upload(
            core,
            Some(&stage),
            [Held::Texture(Arc::clone(&inner.image))],
            // SAFETY: recording under the work lock, the upload holding the image and the stage.
            |device, command| unsafe {
                let region = copy.buffer_image_copy(stage.offset(), 0, 0);
                inner.record(
                    device,
                    &core.counters,
                    command,
                    &copy,
                    (stage.buffer(), region),
                    true,
                )
            },
        )
    }

    /// No Metal counterpart: the copy into a Managed texture's image of its named bytes
    /// ([`Device::new_managed_texture`]) of `region` of `level` and `slice`, a 3D texture's slice
    /// being 0, which the consumer has written: what Metal does for a Managed texture's CPU write.
    /// The upload copies them from the named buffer straight into the image, with no CPU copy,
    /// ahead of every command buffer committed after this call on every queue; the copy reads the
    /// bytes when the GPU reaches it, and the upload keeps the image and the named buffer's block
    /// until it completes, so the consumer may release the texture and the buffer once the call
    /// returns.
    /// [`Texture::replace_region`] on such a texture writes the named bytes, then calls this. An
    /// empty region copies nothing.
    ///
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler;
    /// [`Refusal::ManagedTextureBytes`] for a texture without named bytes, or a depth or
    /// stencil region whose first byte is not a multiple of 4 in the buffer;
    /// [`Refusal::TextureRegion`] for a level, slice or region outside the texture or off its
    /// format's block grid; [`Gap::FloatDepthUpload`] for a 32-bit float depth;
    /// [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's command buffer cannot be
    /// made.
    pub fn upload_region(&self, region: Region, level: u32, slice: u32) -> Result<(), Error> {
        crate::command::not_in_scheduled_handler()?;
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
        let core = &inner.core;
        core.work.upload(
            core,
            None,
            [
                Held::Texture(Arc::clone(&inner.image)),
                Held::Block(Arc::clone(bytes.held().0)),
            ],
            // SAFETY: recording under the work lock, the upload holding the image and the named
            // buffer's block, the bytes the copy reads: named bytes lie in Shared memory with no
            // device copy (`check_places`).
            |device, command| unsafe {
                inner.record(
                    device,
                    &core.counters,
                    command,
                    &copy,
                    (buffer, region),
                    true,
                )
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
        crate::command::not_in_scheduled_handler()?;
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
    /// ranges (`crate::capture`) and the uploads not yet submitted have run: for the tests of one
    /// with them.
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
        let core = &self.inner.core;
        let queue = self.device.new_command_queue(1)?;
        let command_buffer = queue.command_buffer()?;
        command_buffer.record_for_test(&[], |_, _| {})?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()?;
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
        let core = &inner.core;
        let stage = core.work.stage(core, copy.len(), true)?;
        core.work.upload(
            core,
            Some(&stage),
            [Held::Texture(Arc::clone(&inner.image))],
            // SAFETY: as in `replace_region`.
            |device, command| unsafe {
                let region = copy.buffer_image_copy(stage.offset(), 0, 0);
                inner.record(
                    device,
                    &core.counters,
                    command,
                    copy,
                    (stage.buffer(), region),
                    false,
                )
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

/// What the runtime reads of a texture, through the consumer's handle ([`Texture`]) or through
/// the texture as a command buffer, a view of it or the submit thread keeps it
/// ([`TextureInner`]), which holds no handle of the device.
pub(crate) trait TextureObject {
    /// The texture itself.
    fn object(&self) -> &Arc<TextureInner>;

    /// Its resource id in the encoders' hazard sets: a view's is its parent's, so the hazards
    /// between them are one texture's.
    fn id(&self) -> u32 {
        self.object().id
    }

    /// Where the texture starts in its image: a view's first level and array layer.
    fn base(&self) -> [u32; 2] {
        TextureInner::base(self.object())
    }

    /// Whether `core` is the device this texture was made on.
    fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.object().core, core)
    }

    /// Its image.
    fn image(&self) -> vk::Image {
        self.object().image.image
    }

    /// Whether it reaches `level` and `layer` of its image: a view reaches its own levels and
    /// slices alone.
    fn reaches(&self, level: u32, layer: u32) -> bool {
        let d = &self.object().descriptor;
        let [first_level, first_layer] = TextureInner::base(self.object());
        (first_level..first_level + d.mipmap_level_count).contains(&level)
            && (first_layer..first_layer + slices(d)).contains(&layer)
    }

    /// Whether a draw may read an attachment of its own pass through it ([`Image`]'s
    /// `attachment_reads`): its image in the feedback layout, or, on a device without the
    /// attachment feedback loop, its view of its image's shadow.
    fn attachment_reads(&self) -> bool {
        let image = &self.object().image;
        image.attachment_reads && (image.looped() || self.object().shadow.is_some())
    }

    /// Whether a draw reads an attachment of its own pass through it in the feedback layout
    /// ([`Image`]'s `looped`), where its image has the feedback usage; else through its view of
    /// the image's shadow.
    fn looped(&self) -> bool {
        self.object().image.looped()
    }

    /// Its resource id while a draw reads an attachment through it: its second element's, at its
    /// index plus [`TEXTURES`](crate::floor::TEXTURES), with its own word. Nil for a
    /// texture no shader reaches.
    fn second_resource_id(&self) -> ResourceId {
        match self.object().index {
            Some((index, _)) => ResourceId {
                index: index + crate::floor::TEXTURES,
                word: self.object().word,
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
    fn shadow(&self) -> Result<Option<vk::Image>, Error> {
        let (Some(at), Some(index)) = (&self.object().shadow, self.object().index) else {
            return Ok(None);
        };
        let Some(shadow) = self.object().image.shadow()? else {
            return Ok(None);
        };
        let mut view = lock(&at.view);
        if view.is_none() {
            let core = &self.object().core;
            // SAFETY: the shadow is bound, of the image's format, levels and layers, which hold
            // the sampled view's range, and of the sampled usage.
            let made = unsafe {
                image_view(
                    &core.device,
                    shadow,
                    at.view_type,
                    self.object().format,
                    at.components,
                    at.range,
                    (self.object().views_usage).map(|_| vk::ImageUsageFlags::SAMPLED),
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
    fn shadow_format(&self) -> Option<vk::Format> {
        (self.object().image.shadowing.as_ref()).map(|shadowing| shadowing.made.2.format)
    }

    /// What a draw reads of its image through it (`crate::shadow`): with `own_pixel`, the draw's
    /// scissor (per axis `[first, end)`), the rectangle a fragment function that reads only at its
    /// own pixel reads of its first level and layer; else every level and layer it reaches.
    fn read_areas(&self, own_pixel: Option<[[u32; 2]; 2]>) -> Vec<Area> {
        let d = &self.object().descriptor;
        let [level, layer] = TextureInner::base(self.object());
        match own_pixel {
            Some(scissor) => vec![shadow::own_pixel(
                level,
                layer,
                [d.width, d.height],
                scissor,
            )],
            None => {
                let root = match &self.object().view {
                    Some(view) => &view.parent.descriptor,
                    None => d,
                };
                let size = [root.width, root.height];
                shadow::whole(level, d.mipmap_level_count, layer, slices(d), size)
            }
        }
    }

    /// The bytes of a texel of its image, which its copies move; 0 where no copy is built.
    fn texel_bytes(&self) -> u64 {
        self.object().unit.map_or(0, |unit| u64::from(unit.bytes))
    }

    /// Every level and layer of its image, every aspect.
    fn whole(&self) -> vk::ImageSubresourceRange {
        let root = match &self.object().view {
            Some(view) => &view.parent.descriptor,
            None => &self.object().descriptor,
        };
        vk::ImageSubresourceRange {
            aspect_mask: self.object().image.aspects,
            base_mip_level: 0,
            level_count: root.mipmap_level_count,
            base_array_layer: 0,
            layer_count: slices(root),
        }
    }

    /// Its image's format.
    fn format(&self) -> vk::Format {
        self.object().format
    }

    /// The format its image is made in, which a blit of the image filters by: a view in another
    /// format's differs from [`format`](Self::format).
    fn image_format(&self) -> vk::Format {
        self.object().image.format
    }

    /// A layout change of its levels `levels`, counted from its first, over each of its slices,
    /// from `old` to `new`, `None` naming its home layout, with no scopes yet
    /// ([`BlitCommandEncoder::generate_mipmaps`](crate::BlitCommandEncoder::generate_mipmaps)).
    fn levels_barrier(
        &self,
        levels: Range<u32>,
        old: Option<vk::ImageLayout>,
        new: Option<vk::ImageLayout>,
    ) -> vk::ImageMemoryBarrier2<'static> {
        TextureInner::levels_barrier(self.object(), levels, old, new)
    }

    /// How its image holds its pixel format.
    fn emulation(&self) -> FormatEmulation {
        self.object().emulation
    }

    /// `level` and `slice` of the texture as a render pass attachment: the image view of that
    /// one level and layer of its image, every aspect, with the identity swizzle, made at the
    /// first pass that renders into it and kept with the texture.
    ///
    /// Err: [`Refusal::AttachmentUsage`] for a texture made without `RenderTarget` usage;
    /// [`Gap::RenderPass`] of [`RenderPassKind::TextureType`] for one but a 2D texture, a 2D
    /// array, a cube, a cube array or a 2D multisample texture or array; [`Refusal::AttachmentPlace`] for a level or slice past the texture's;
    /// [`Error::Vulkan`] where the view cannot be made.
    fn attachment(&self, level: u32, slice: u32) -> Result<Attachment, Error> {
        let d = &self.object().descriptor;
        if !d.usage.is_empty() && !d.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(Error::Refused(Refusal::AttachmentUsage {
                pixel_format: d.pixel_format,
            }));
        }
        let gap = |kind| Err(Error::Unsupported(Gap::RenderPass { kind }));
        match d.texture_type {
            TextureType::Type2D
            | TextureType::Type2DArray
            | TextureType::Cube
            | TextureType::CubeArray
            | TextureType::Type2DMultisample
            | TextureType::Type2DMultisampleArray => {}
            _ => return gap(RenderPassKind::TextureType),
        }
        if level >= d.mipmap_level_count || slice >= slices(d) {
            return Err(Error::Refused(Refusal::AttachmentPlace { level, slice }));
        }
        let [first_level, first_layer] = TextureInner::base(self.object());
        let (level_in_image, layer) = (first_level + level, first_layer + slice);
        let image = &self.object().image;
        // A view of another format than the image's renders where the format's features hold an
        // attachment.
        let renders =
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        let looped = vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        let usage = match self.object().views_usage {
            None => None,
            Some(usage) if usage.intersects(renders) => Some(usage & (renders | looped)),
            Some(_) => {
                return Err(Error::Unsupported(Gap::TextureFormat {
                    pixel_format: d.pixel_format,
                }))
            }
        };
        let mut views = lock(&self.object().attachments);
        let kept = (views.iter())
            .find(|a| a.level == level_in_image && a.layer == layer)
            .map(|a| a.view);
        let view = match kept {
            Some(view) => view,
            None => {
                let core = &self.object().core;
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
                        self.object().format,
                        vk::ComponentMapping::default(),
                        range,
                        usage,
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
            format: self.object().format,
            samples: vk::SampleCountFlags::from_raw(d.sample_count),
            linear: image.linear_attachment,
        })
    }

    /// The unit its copies move; `None` where no copy is built.
    fn unit(&self) -> Option<CopyUnit> {
        self.object().unit
    }

    /// The layout a copy names the texture's image in, for its role's transfer `layout`
    /// ([`Image::copy_layout`]): general for a texture over a buffer in place, `layout` for any
    /// other.
    fn copy_layout(&self, layout: vk::ImageLayout) -> vk::ImageLayout {
        self.object().image.copy_layout(layout)
    }

    /// The barrier that moves `placed` from home to the copy layout of the transfer `layout` for
    /// a copy's `access`, after every earlier command's writes.
    fn away(
        &self,
        placed: &Placed,
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        TextureInner::away(self.object(), placed, layout, access)
    }

    /// The barrier that moves `placed` back home from the copy layout of the transfer `layout`
    /// once a copy has run, its writes visible to every later command.
    fn back(&self, placed: &Placed, layout: vk::ImageLayout) -> vk::ImageMemoryBarrier2<'static> {
        TextureInner::back(self.object(), placed, layout)
    }

    /// A command buffer presents the drawable: it is the runtime's, acquired, from now presented
    /// by that command buffer.
    ///
    /// Err: [`Refusal::NotDrawable`]; [`Refusal::DrawableNotAcquired`].
    fn present_turn(&self) -> Result<(), Error> {
        let drawable =
            (self.object().image.drawable.as_ref()).ok_or(Error::Refused(Refusal::NotDrawable))?;
        let mut turn = lock(&drawable.turn);
        if *turn != Turn::Acquired {
            return Err(Error::Refused(Refusal::DrawableNotAcquired));
        }
        *turn = Turn::Presenting;
        Ok(())
    }

    /// The command buffer that presents the drawable was dropped uncommitted: it is acquired
    /// again, for another present.
    fn unpresent(&self) {
        if let Some(drawable) = &self.object().image.drawable {
            *lock(&drawable.turn) = Turn::Acquired;
        }
    }

    /// The command buffer that presents the drawable is committed: the image is the presentation
    /// engine's once that command buffer has run, in the present layout.
    fn presented(&self) {
        if let Some(drawable) = &self.object().image.drawable {
            *lock(&drawable.turn) = Turn::Engine;
        }
    }

    /// The move of its whole image from home to the present layout, after every earlier command's
    /// writes. It delays nothing after it: the present's semaphore wait and its own visibility
    /// operations order the presentation engine's reads (Vulkan's note on `VkPresentInfoKHR`).
    fn present_barrier(&self) -> vk::ImageMemoryBarrier2<'static> {
        let image = &self.object().image;
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(image.home)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(whole(image.aspects))
    }

    /// Its id as bytes, which a pass's attachments and the copies over buffers name it by: a
    /// texture over a buffer in place's is the buffer's, as the two are the same bytes and a use
    /// of either is a use of both.
    fn hazard_id(&self) -> u32 {
        match &self.object().named {
            Some(named) if named.form == Form::InPlace => named.buffer.id(),
            _ => self.object().id,
        }
    }

    /// Its id in the compute encoders' hazard sets: its heap's where the consumer placed it in a
    /// heap ([`Device::new_heap_texture`]), its surface's for an exported surface's plane in place
    /// and its views ([`crate::Surface::buffer`]), its buffer's there for one in place over a
    /// buffer, and [`hazard_id`](Self::hazard_id) elsewhere.
    fn tracked_id(&self) -> u32 {
        match (&self.object().heap, &self.object().named) {
            (Some(heap), _) => heap.get(),
            (None, Some(named)) if named.form == Form::InPlace => named.buffer.tracked_id(),
            _ => self.object().id,
        }
    }

    /// Where it is made for shader writes (`ShaderWrite`, or Metal's unknown usage) but has no
    /// storage element, which shaders write it and read it as storage through
    /// (`Arrays::place_texture`): its pixel format, which [`Gap::StorageElement`] names.
    fn unstored(&self) -> Option<PixelFormat> {
        let d = &self.object().descriptor;
        let written = d.usage.is_empty() || d.usage.contains(TextureUsage::SHADER_WRITE);
        let storage = match &self.object().view {
            Some(view) => view.views.storage,
            None => self.object().image.storage,
        };
        (written && storage.is_none()).then_some(d.pixel_format)
    }

    /// Where it has a storage element the device cannot read without a format, which shaders read
    /// it as storage through (`access::read_write`): its views' format lacks
    /// `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT` ([`Plan`]'s `storage_reads`, taken at
    /// its make). Its pixel format, which [`Gap::StorageElement`] names.
    fn unread(&self) -> Option<PixelFormat> {
        let inner = self.object();
        let storage = match &inner.view {
            Some(view) => view.views.storage,
            None => inner.image.storage,
        };
        (storage.is_some() && !inner.storage_reads).then_some(inner.descriptor.pixel_format)
    }

    /// Whether it reads through a swizzle other than each channel's own.
    fn swizzled(&self) -> bool {
        TextureInner::reads(self.object()) != TextureSwizzleChannels::DEFAULT
    }

    /// Whether the host reads the GPU's writes to it with no copy: a texture in place over a
    /// Shared buffer.
    fn host_reads(&self) -> bool {
        match &self.object().named {
            Some(named) => named.form == Form::InPlace && named.buffer.is_shared(),
            None => false,
        }
    }

    /// Of a texture over a buffer of no device copy in an image of its own, which the submissions
    /// copy into and back by the Shared rules (`crate::over`): the buffer and the copy.
    fn copied_shared(&self) -> Option<(&Arc<BufferInner>, OverCopy)> {
        self.copied_over()
            .filter(|(buffer, _)| buffer.held().1.is_none())
    }

    /// Of a texture over a Managed buffer's device copy in an image of its own, which Metal's
    /// Managed events owe copies (`crate::over::ManagedOvers`): the buffer.
    fn managed_over(&self) -> Option<&Arc<BufferInner>> {
        let named = (self.object().named.as_ref()).filter(|named| named.form == Form::Copied)?;
        named.buffer.held().1.map(|_| &named.buffer)
    }

    /// Of a texture over a buffer in an image of its own (`crate::over`): the buffer, and the
    /// copy between the bytes shaders reach of it and the image.
    fn copied_over(&self) -> Option<(&Arc<BufferInner>, OverCopy)> {
        let inner = &self.object();
        let named = (inner.named.as_ref()).filter(|named| named.form == Form::Copied)?;
        let unit = inner.unit?;
        let place = named.places[0];
        let d = &inner.descriptor;
        let (buffer, at) = named.buffer.device_range();
        let (block, device_copy) = named.buffer.held();
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
            unit,
            bytes: u64::from(d.width) * u64::from(d.height) * u64::from(unit.bytes),
            shared: named.buffer.is_shared(),
            imported: device_copy.is_none() && block.memory.imported(),
        };
        Some((&named.buffer, copy))
    }

    /// For the device tests: the bytes its image's memory requirements name.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fn image_size(&self) -> u64 {
        self.object().image.bytes
    }

    /// For the device tests: the texture block its image is placed in and the offset there;
    /// `None` for an image of no place.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fn place(&self) -> Option<(BlockId, u64)> {
        match &self.object().image.backing {
            Backing::Placed(place) => Some((place.block(), place.offset)),
            _ => None,
        }
    }

    /// `gpuResourceID`: its index (0 for a texture no shader reaches) and the translator's
    /// `TextureWord` of its pixel format and swizzle.
    fn gpu_resource_id(&self) -> ResourceId {
        ResourceId {
            index: self.object().index.map_or(0, |(index, _)| index),
            word: self.object().word,
        }
    }

    /// `textureType`, `pixelFormat`, `width` and the rest, as made.
    fn descriptor(&self) -> &TextureDescriptor {
        &self.object().descriptor
    }
}

impl TextureObject for Texture {
    fn object(&self) -> &Arc<TextureInner> {
        &self.inner
    }
}

impl TextureObject for Arc<TextureInner> {
    fn object(&self) -> &Arc<TextureInner> {
        self
    }
}

impl Buffer {
    /// `newTextureWithDescriptor:offset:bytesPerRow:`, at `index` as for
    /// [`Device::new_texture`]: a texture whose texels are the buffer's bytes from `offset`,
    /// rows `bytes_per_row` apart, with the buffer's storage mode whatever the descriptor's. Its
    /// image is a linear image in the bytes shaders reach (a Managed buffer's device copy, where
    /// it has one), made preinitialized so that its move home keeps the bytes, in the general
    /// layout, where the device lays such an image's rows at `bytes_per_row` and the place meets
    /// its memory alignment, and, of a texture with render-target usage, where the subresource
    /// the device advertises ends within the texture's own rows (`height` rows of
    /// `bytes_per_row`) and the buffer: the GPU reaches the buffer's bytes, and nothing is copied
    /// that the buffer does not copy. What differs: the bytes of a render-target texture's rows
    /// that are not texels may be rewritten with their earlier values while a pass renders it, as
    /// an attachment's store, clear or resolve may write back any byte of its subresource.
    /// [`Device::minimum_linear_texture_alignment`] answers the offset and
    /// pitch at which a texture is so. [`Texture::replace_region`] writes the buffer's bytes, and
    /// of a Managed buffer's device copy copies the rows' span into it as
    /// [`Buffer::did_modify_range`] does; [`Texture::get_bytes`] reads the buffer's bytes, of a
    /// Managed buffer the consumer's, as Metal reads a Managed texture's CPU copy. The texture
    /// holds the buffer.
    ///
    /// Where the device's linear image does not take the texture (another pitch, an offset off
    /// its memory alignment, a format or usage its linear tiling lacks, memory that spilled, a
    /// render target whose advertised subresource passes its rows or the buffer), over a buffer
    /// in an imported block (a no-copy buffer's, [`Device::import_host_memory`],
    /// `Device::import_dma_buf` on Unix), and over a buffer in an exported block
    /// ([`Device::new_exported_block`], whose memory is dedicated to its buffer), the
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
        self.texture_over(descriptor, Reach::Index(index), offset, bytes_per_row)
    }

    /// A surface plane's texture over this buffer, as [`Buffer::new_texture`] makes it, but that
    /// with no index a plane shaders read or write is reached through its views alone
    /// ([`Device::new_texture_for_views`]).
    pub(crate) fn new_plane_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<Texture, Error> {
        let reach = match index {
            Some(_) => Reach::Index(index),
            None => Reach::Views,
        };
        self.texture_over(descriptor, reach, offset, bytes_per_row)
    }

    fn texture_over(
        &self,
        descriptor: &TextureDescriptor,
        reach: Reach,
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
        self.device().texture(&d, reach, over)
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
    /// The texture's copy unit, which a copy back's regions take ([`copy_out::regions`]).
    pub(crate) unit: CopyUnit,
    /// The texels' bytes.
    pub(crate) bytes: u64,
    /// The buffer is Shared: the host reads what a copy back writes.
    pub(crate) shared: bool,
    /// The bytes shaders reach are in an imported block (`Metrics::over_copies_imported`).
    pub(crate) imported: bool,
}

impl crate::kept::Handle for Texture {
    type Kept = Arc<TextureInner>;

    fn object(&self) -> &Arc<TextureInner> {
        &self.inner
    }

    fn address_of(object: &Arc<TextureInner>) -> usize {
        Arc::as_ptr(object) as *const () as usize
    }
}

/// A surface plane's image in place, made and bound by `crate::surface`, for
/// [`Device::surface_texture`]: its backing, its memory's bytes, its usage and view-format list,
/// and the opaque handle's allocation it is bound in.
pub(crate) struct SurfaceImage {
    pub(crate) image: vk::Image,
    pub(crate) backing: Backing,
    pub(crate) bytes: u64,
    pub(crate) usage: vk::ImageUsageFlags,
    /// The image's own format: the texture's, or one its texture views as the texture's, which
    /// `view_formats` then lists with it.
    pub(crate) format: vk::Format,
    pub(crate) view_formats: Vec<vk::Format>,
    pub(crate) block: Option<Arc<Block>>,
    /// The image is in the general layout from its first use, with no move from undefined: a
    /// Direct3D 11 texture's, which Vulkan takes as general after Direct3D 11's access (section
    /// 12.5.2).
    pub(crate) general: bool,
    /// The image is `LINEAR` and its format renders only through
    /// `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV` (`crate::surface::Query::plane`):
    /// the image is marked ([`Image`]'s `linear_attachment`).
    pub(crate) linear_attachment: bool,
}

/// Destroys a surface plane's image not wrapped in a texture, and frees its own import.
///
/// # Safety
///
/// Nothing uses the image or its memory.
pub(crate) unsafe fn discard_surface_image(core: &Core, made: SurfaceImage) {
    // SAFETY: the caller's contract.
    unsafe { core.device.destroy_image(made.image, None) };
    if let Backing::Import(memory, bytes) = made.backing {
        // SAFETY: the image bound to it is destroyed.
        unsafe { core.device.free_memory(memory, None) };
        core.counters.fall(Gauge::Blocks);
        core.counters.surface_import_freed(bytes);
    }
}

impl Device {
    /// A surface plane's texture of `d`, checked as [`Buffer::new_texture`] checks a texture
    /// over a buffer: its image format, the image usage its uses ask, its view formats and its
    /// texel's bytes.
    pub(crate) fn plane_plan(
        &self,
        d: &TextureDescriptor,
    ) -> Result<crate::surface::PlanePlan, Error> {
        let (_, plan, _) = self.prepare(d)?;
        let texel = over_texel(d, &plan)?;
        Ok(crate::surface::PlanePlan {
            format: plan.format,
            usage: plan.linear_usage(),
            looped: plan.reads_attachments(self.core()) && self.core().feedback.is_some(),
            view_formats: plan.view_formats.clone(),
            texel,
        })
    }

    /// The texture of a surface plane's image, at `index` as for [`Device::new_texture`]: its
    /// home layout general, from undefined in the next upload, which goes ahead of every command
    /// buffer committed after the make, or already general where the image is (a Direct3D 11
    /// texture's). Its format is the descriptor's, a view of the image's own where they differ.
    /// The texture holds the image, which its drop frees with its memory. It takes `unit`'s id in
    /// the hazard sets, the surface's ([`HeapId::unit`]).
    ///
    /// Err: [`Refusal::TextureIndex`]; [`Gap::SurfaceUsage`] for a usage the texture's uses ask
    /// that the image lacks; [`Error::Vulkan`] where a view cannot be made; what the texture's
    /// index takes.
    pub(crate) fn surface_texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        made: SurfaceImage,
        unit: Arc<HeapId>,
    ) -> Result<Texture, Error> {
        let core = self.core();
        let d = *descriptor;
        let checked = self.prepare(&d).and_then(|(shape, plan, usage)| {
            // With no index, a plane shaders reach is reached through its views alone.
            if index.is_some() && !reached(d.pixel_format, usage) {
                return Err(Error::Refused(Refusal::TextureIndex { index }));
            }
            let missing = plan.linear_usage() & !made.usage;
            if !missing.is_empty() {
                return Err(Error::Unsupported(Gap::SurfaceUsage {
                    pixel_format: d.pixel_format,
                    missing: missing.as_raw(),
                }));
            }
            Ok((shape, plan))
        });
        let (shape, plan) = match checked {
            Ok(checked) => checked,
            Err(error) => {
                // SAFETY: nothing uses the image or its memory.
                unsafe { discard_surface_image(core, made) };
                return Err(error);
            }
        };
        let mut views = Made::default();
        // SAFETY: the image is the device's and bound, of the plan's format or mutable with it
        // listed, with the plan's usage and more, which the plan's format's features hold.
        let made_views =
            unsafe { views.views(&core.device, made.image, &shape, &plan, made.usage) };
        let (sampled, storage) = match made_views {
            Ok(made_views) => made_views,
            Err(error) => {
                // SAFETY: nothing else holds the views; `views` holds no image. Nothing uses the
                // image or its memory.
                unsafe {
                    views.destroy(core);
                    discard_surface_image(core, made);
                }
                return Err(error);
            }
        };
        core.counters.rise(Gauge::Textures);
        let home = vk::ImageLayout::GENERAL;
        let general = made.general;
        // A view of the image's format as the texture's takes the image's usage that its own
        // format's features hold, as a texture view of another format does.
        let views_usage = (made.format != plan.format).then_some(made.usage & plan.supported);
        // A draw may read an attachment of its own pass through it as through a texture of its
        // own image: in the feedback layout where the plane's image took the feedback usage
        // (`crate::surface`), else through a copy into its shadow.
        let looped = made
            .usage
            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT);
        let shadowing = (!looped
            && plan.reads_attachments(core)
            && plan.shadow_offered(core, &shape))
        .then(|| {
            Box::new(Shadowing {
                made: (d, shape, plan.shadow()),
                shadow: Mutex::new(None),
            })
        });
        let image = Arc::new(Image {
            core: Arc::clone(core),
            image: made.image,
            backing: made.backing,
            bytes: made.bytes,
            sampled,
            storage,
            home,
            attachment_reads: looped || shadowing.is_some(),
            shadowing,
            shadow: false,
            aspects: plan.aspects,
            format: made.format,
            usage: made.usage,
            view_formats: made.view_formats,
            cube: false,
            linear_attachment: made.linear_attachment,
            first: match general {
                true => home,
                false => vk::ImageLayout::UNDEFINED,
            },
            // An image already general owes no move.
            homed: AtomicBool::new(general),
            zero: None,
            drawable: None,
            _under: Some(Under::Surface { _block: made.block }),
        });
        let shadow = ShadowView::of(&image, &plan, index, shape.view_type, whole);
        // The texture's elements of set 1, written as it takes its index; a refusal frees the
        // image as `image` drops.
        let index = match index {
            Some(index) => {
                let element = sampled.filter(|_| plan.read).map(|view| (view, home));
                let looped = sampled.filter(|_| image.looped());
                // SAFETY: the texture's views are live; its image is in its home layout whenever
                // work uses it: the upload that moves it goes ahead of that work, or it is general
                // after the other API's access.
                let owner = unsafe {
                    (core.arrays).place_texture(
                        &core.device,
                        &core.counters,
                        index,
                        (element, storage),
                        looped,
                    )
                }?;
                Some((index, owner))
            }
            None => None,
        };
        let inner = TextureInner {
            core: Arc::clone(core),
            descriptor: d,
            image,
            format: plan.format,
            storage_reads: plan.storage_reads,
            views_usage,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
            heap: Some(unit),
            named: None,
            view: None,
            emulation: plan.emulation,
            attachments: Mutex::new(Vec::new()),
            shadow,
        };
        let texture = Texture {
            inner: Arc::new(inner),
            device: self.clone(),
        };
        // For the device tests: the plane's texture apart from its move home, which takes the work
        // lock ([`crate::surface::steps`]).
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::surface::steps::mark("plane texture");
        if !general {
            core.work.home(core, &texture.inner.image)?;
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::surface::steps::mark("plane home");
        Ok(texture)
    }
}

impl Texture {
    /// The texture's image, for an upload to hold.
    pub(crate) fn held_image(&self) -> Arc<Image> {
        Arc::clone(&self.inner.image)
    }
}
