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

use crate::arrays::{IndexClass, ResourceId};
use crate::buffer::Buffer;
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit, Refusal, TextureViewKind};
use crate::memory::{self, Block, BlockId, MemoryClass, BLOCK_LEN};
use crate::places::Place;
use crate::reaper::{Free, Job};
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
use std::sync::Arc;

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
    /// Its class and index; `None` for a texture no shader reaches.
    index: Option<(IndexClass, u32)>,
    word: u32,
    /// Its resource id in the encoders' hazard sets.
    id: u32,
    /// A Managed texture's bytes the consumer names: the buffer, and a place for each level and
    /// slice in order.
    managed: Option<(Buffer, Vec<BytesPlace>)>,
    /// A view's parent and place in the image ([`Texture::new_texture_view`]); `None` for a
    /// texture with an image of its own.
    view: Option<View>,
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
    /// Every aspect of the image's format, which a change of layout names.
    aspects: vk::ImageAspectFlags,
    /// An upload recorded its move from undefined to `home`. Read and set under the work lock.
    homed: AtomicBool,
    /// A heap texture's block, which the image's memory is, held while anything holds the image.
    _heap: Option<Arc<Block>>,
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
            backing: self.backing,
        }));
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
    /// A place in a heap's range of a block the image holds ([`Device::new_heap_texture`]).
    Heap,
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
        // A view's id and image are its parent's, which it keeps.
        if self.view.is_none() {
            core.ids.give_back(self.id);
            // An upload holds the image past the texture: the pending one's bound counts it.
            if Arc::strong_count(&self.image) > 1 {
                core.work.released(self.image.bytes);
            }
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
    /// memory, placed in a texture block ([`Texture`]), its contents undefined, as Metal's are.
    /// `index` is given exactly where shaders reach the texture (shader-read or shader-write
    /// usage), and is then an index of a live grant of the class the usage picks:
    /// [`IndexClass::StorageTexture`] with shader-write usage, else [`IndexClass::Texture`]. The
    /// texture's elements of set 1 are written before the call returns, and its image moved to its
    /// home layout ahead of every command buffer committed after it. Until the next submission
    /// nothing else holds the texture: released first, it is freed at once.
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
        self.texture(descriptor, index, None, None)
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
        self.texture(descriptor, index, None, Some((block, heap, offset)))
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
        self.texture(&d, index, Some((bytes, places)), None)
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

    /// [`Device::new_texture`]; [`Device::new_managed_texture`] with its bytes; and
    /// [`Device::new_heap_texture`] with its block, heap and offset.
    fn texture(
        &self,
        descriptor: &TextureDescriptor,
        index: Option<u32>,
        managed: Option<(&Buffer, &[BytesPlace])>,
        heap: Option<(Arc<Block>, HeapRange, u64)>,
    ) -> Result<Texture, Error> {
        let d = *descriptor;
        let (shape, plan, usage) = self.prepare(&d)?;
        let reached = usage.intersects(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE);
        if reached != index.is_some() {
            return Err(Error::Refused(Refusal::TextureIndex { index }));
        }
        let class = match usage.contains(TextureUsage::SHADER_WRITE) {
            true => IndexClass::StorageTexture,
            false => IndexClass::Texture,
        };
        let core = self.core();
        if let Some((bytes, places)) = managed {
            check_places(&d, plan.unit, bytes, places)?;
        }
        let index = match index {
            Some(index) => {
                lock(&core.arrays.grants).occupy(class, index)?;
                core.counters.rise(Gauge::DescriptorSlots);
                Some((class, index))
            }
            None => None,
        };
        let bind = match &heap {
            Some((block, heap, offset)) => Bind::Heap {
                memory: block.memory.device_memory(),
                block_len: block.memory.len,
                heap: *heap,
                offset: *offset,
            },
            None => Bind::Pooled,
        };
        // SAFETY: the device is live, and `plan` was checked against its image format query; a
        // heap's block is the device's, and the texture holds it.
        let made = unsafe { make(core, &d, &shape, &plan, bind) };
        let (image, backing, bytes, sampled, storage) = match made {
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
            backing,
            bytes,
            sampled,
            storage,
            home: plan.home,
            aspects: plan.aspects,
            homed: AtomicBool::new(false),
            _heap: heap.map(|(block, ..)| block),
        });
        let inner = TextureInner {
            device: self.clone(),
            descriptor: d,
            image,
            format: plan.format,
            unit: plan.unit,
            index,
            word: plan.word,
            id: core.ids.take(),
            managed: managed.map(|(bytes, places)| (bytes.clone(), places.to_vec())),
            view: None,
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

/// Where [`make`] binds the image.
enum Bind {
    /// A place in a texture block, or memory of its own.
    Pooled,
    /// `offset` in `heap`, whose block's memory is `memory`, of `block_len` bytes.
    Heap {
        memory: vk::DeviceMemory,
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
        let info = image_info(d, shape, plan);
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
        // does not require an allocation of its own for it (preferring one is no reason).
        let placed = dedicated.requires_dedicated_allocation == vk::FALSE
            && requirements.size <= BLOCK_LEN
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
        let (backing, memory, offset) = if let Bind::Heap {
            memory,
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
                && at.is_some_and(|at| at.is_multiple_of(fit.align));
            match (fits, at) {
                (true, Some(at)) => (Backing::Heap, memory, at),
                _ => return Err(refused),
            }
        } else if let Some((place, block)) = pooled {
            (Backing::Placed(place), block.device_memory(), place.offset)
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
                Some(Backing::Heap) | None => {}
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
        let (bytes, places) = self.managed.as_ref().ok_or_else(refused)?;
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

    /// Where the host reaches a Managed texture's named bytes of a non-empty `region` of `level`
    /// and `slice`: the region's first element and its place. `None` for a texture without named
    /// bytes.
    fn named_host(
        &self,
        region: Region,
        level: u32,
        slice: u32,
    ) -> Result<Option<(*mut u8, BytesPlace)>, Error> {
        if self.managed.is_none() {
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
    /// view reach the levels and slices under it; its hazards are its parent's own.
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
    /// [`Refusal::IndicesNotGranted`] and [`Refusal::IndexInUse`] as [`Device::new_texture`];
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
        let class = match usage.contains(TextureUsage::SHADER_WRITE) {
            true => IndexClass::StorageTexture,
            false => IndexClass::Texture,
        };
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
        if let Some(index) = index {
            lock(&core.arrays.grants).occupy(class, index)?;
            core.counters.rise(Gauge::DescriptorSlots);
        }
        let index = index.map(|index| (class, index));
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
        let parent = match &inner.view {
            Some(view) => view.parent.clone(),
            None => self.clone(),
        };
        let view = TextureInner {
            device: inner.device.clone(),
            descriptor: d,
            image: Arc::clone(&inner.image),
            format: plan.format,
            unit: plan.unit,
            index,
            word: plan.word,
            id: inner.id,
            managed: None,
            view: Some(View {
                parent,
                base,
                _views: views,
            }),
        };
        // A failed view drops here: its index is vacated and its views freed.
        made?;
        if let Some((class, index)) = index {
            // SAFETY: the index is occupied for the view, whose image views are live; the image
            // is in its home layout whenever work uses it.
            unsafe {
                core.arrays.write_texture(
                    &core.device,
                    class,
                    index,
                    sampled.map(|view| (view, inner.image.home)),
                    storage,
                )
            };
        }
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

    /// Its image's format.
    pub(crate) fn format(&self) -> vk::Format {
        self.inner.format
    }

    /// The unit its copies move; `None` where no copy is built.
    pub(crate) fn unit(&self) -> Option<CopyUnit> {
        self.inner.unit
    }

    /// A Managed texture's named bytes ([`Device::new_managed_texture`]): the buffer and its
    /// places, one for each level and, within it, each slice. `None` for any other texture.
    pub(crate) fn named_bytes(&self) -> Option<(&Buffer, &[BytesPlace])> {
        let (bytes, places) = self.inner.managed.as_ref()?;
        Some((bytes, places))
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
            index: self.inner.index.map_or(0, |(_, index)| index),
            word: self.inner.word,
        }
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
    /// [`Texture::upload_region`] copies them into the image; nothing is staged or counted.
    ///
    /// Err: [`Refusal::TextureBytesPrivate`]; [`Refusal::TextureRegion`] for a level, slice or
    /// region outside the texture or off its format's block grid;
    /// [`Refusal::TextureBytesLayout`]; [`Gap::TextureBytes`]; [`Gap::FloatDepthUpload`] for a
    /// 32-bit float depth; [`Limit::MemoryAllocations`] where the region's block or a staging
    /// buffer's memory would take the device past its count; [`Error::Vulkan`] where either
    /// cannot be made, and with `VK_ERROR_DEVICE_LOST` where the device is lost while the call
    /// waits; [`Error::Quiesced`].
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
            return self.upload_region(region, level, slice);
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
    /// ahead of every command buffer committed after this call on every queue, as
    /// [`Buffer::did_modify_range`] copies a Managed buffer's bytes; the copy reads the bytes when
    /// the GPU reaches it. [`Texture::replace_region`] on such a texture writes the named bytes,
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
    /// them only by a synchronize.
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
    /// from a texture without named bytes: for the tests of one with them.
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
