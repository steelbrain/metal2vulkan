//! `Texture` (`MTLTexture`) and `Device::new_texture`, texture views, and `replace_region` and
//! `get_bytes`.
//!
//! - **Storage.** A texture is an optimal-tiled `VkImage` ([`image_plan`](super::image_plan)
//!   plans it) in the device's memory, placed on whole `bufferImageGranularity` pages so it never
//!   shares one with a buffer. Every image the runtime makes rests in the `GENERAL` layout from
//!   the device's first submission after its making, which moves it there before any command
//!   (`texture/first_move.rs`), so ordering its accesses needs memory barriers only.
//! - **Contents.** A new `Shared` or `Managed` texture reads zero, as Metal's do; a `Private` one
//!   holds what its memory held, as Metal's do (measured on an Apple M3 Max and M4).
//! - **Views.** A view (`new_texture_view*`) shares its texture's image, and so its resource
//!   identity: hazard tracking sees one resource. It is a [`ViewRecord`] of the part of the image
//!   it presents; the runtime makes the `VkImageView`s a binding needs when it binds one. A view
//!   of a texture over a buffer shares all of its storage: the buffer's bytes, a texture
//!   buffer's texel buffer views of them, and a 2D texture's aliasing image or shadow and its
//!   copies. Metal stops a view of more than one slice of an array over a buffer.
//! - **CPU access.** `replace_region` and `get_bytes` copy through a staging buffer on the
//!   device's internal submission, synchronously, as Metal's are synchronous: they wait for the
//!   device's earlier submissions too. Metal refuses them on a private texture, on a combined
//!   depth/stencil format and for an empty region; a depth-only or stencil-only format is copied.
//!   `Managed` storage follows Metal's contract for the device's memory model (`managed.rs`). A
//!   managed texture with an image of its own is copied so on every device, its GPU writes
//!   visible without `synchronizeTexture`, as on Apple silicon, which the contract allows; one
//!   over a buffer is the buffer's bytes, which on a device without unified memory have a host
//!   copy: `replace_region` updates both, `get_bytes` reads the host copy, which
//!   `synchronize_resource` updates.
//! - **Texture buffers** hold no image: their texels are a buffer's bytes, viewed as a texel
//!   buffer where a function binds them (`buffer_texture.rs`). A 2D texture or array over a
//!   buffer holds both: the bytes, which CPU access reads and writes, and an image: a linear
//!   image bound to the buffer's memory where the device lays it out as Metal does and the memory
//!   is the runtime's own (its uses are then uses of the buffer's bytes), else one the encoders
//!   copy them into and back (`shadow.rs`), as over a no-copy buffer's host pages always.
//! - **Memoryless** textures are transient attachments (`VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`)
//!   in lazily allocated memory where the device has it, else in device-local memory, and start
//!   undefined: Metal keeps no contents for them (a Metal probe on an M4). Metal makes them of
//!   every type, with any usage (its validation stops all but the 2D, 2D array and multisample
//!   types; plain Metal renders through a 1D, 3D or cube one); anything but a render pass's
//!   attachment is refused (`Invalid`, `StorageMode`), as Metal's validation refuses CPU access,
//!   blits, mipmaps, views and bindings of one; a pass may neither load nor store one
//!   (`render_pass.rs`).
//! - **The caller's images** (`texture/wrapped.rs`): a texture over an image the caller made and
//!   keeps. The runtime never destroys or frees it ([`Owner::Caller`]); the device's batches move
//!   it from the caller's layout to `GENERAL` and back.
//! - **Shared textures** (`texture/shared.rs`): a `Private` texture made shareable, whose
//!   in-process handle opens new textures over the same image.
//! - **Depth twins**. A `Depth32Float` texture the program only reads or samples, on a device whose
//!   depth images its driver is not measured to keep every `float` in (without
//!   `VK_EXT_depth_range_unrestricted`, or keeping a denormal), holds its texels in an
//!   `R32_SFLOAT` image, which keeps any `float`, and the same texels in a `D32_SFLOAT` twin
//!   ([`image_plan`](super::image_plan)), which the views a comparison the device makes reads
//!   present. Every copy
//!   into the texture writes both (`replace_region`, the blit encoder's copies), and reads come
//!   from its image; the twin shares the image's resource identity.
//! - **Not yet:** blits of textures over buffers are refused `NotImplemented`.

use super::address::{ObjectsToken, RangeToken};
use super::address_map::AddressModeMap;
use super::attachment_stand_in::KeptBacking;
use super::attributes::PurgeableState;
use super::binding_view::{binding_view, texture_facts, views_as, VkView};
use super::buffer::{Buffer, CallerMemory, HostPages, MemoryHold, NotedTexture};
use super::buffer_texture::{Texels, TexelsHome};
use super::churn_counts::{Owned, Recycled};
use super::clear_value;
use super::color_resolve::KeptResolves;
use super::command_buffer::CommandBuffer;
use super::copy_plan::{
    check_region, copy_unit, level_extent, plan_host_copy, surface_depth_unit, BlitOption,
    CopyProblem, CopyRefusal, HostLayout, ImageRegion, TextureRegion,
};
use super::degradation::Degradation;
use super::device::lock;
use super::error::{internal, invalid, nil, vulkan, Error, MemoryKind, Problem, RuntimeGap};
use super::exit_guard;
use super::families::FormatAnswers;
use super::framebuffer_fetch::UsageAnswers;
use super::fresh_unwritten::{FreshKind, Notice};
use super::heap::{Heap, HeapProblem, HeapType};
use super::heap_sets;
use super::host_bytes::HostBytes;
use super::image_plan::{
    check_image_limits, holds_depth_in_colour, plan_image, plan_images, DepthImageFacts,
    ImageFacts, ImagePlan, ImageRefusal,
};
use super::key_hash::KeyMap;
use super::linear_plan::{
    aliases, check_linear_limits, holding, linear_lack, LinearImage, LinearLack,
};
use super::memory::{MemoryClass, Placement};
use super::mipmap::MipmapChecks;
use super::nan_clear;
use super::neutral::Neutrals;
use super::pvrtc::{
    cpu_order_of_rows, holds_blocks, image_block_extent, image_region, is_pvrtc, rows_of_cpu_order,
};
use super::recycle::{self, ImageMaking, Recycler};
use super::resource::{Identity, ObjectId, ResourceKey};
use super::resource_ids::{Holder, IdHold, IdPlacement, IdSpace, ResourceId};
use super::retained::HeldBy;
use super::shadow::ShadowCopies;
use super::shadow_copy_back::{self, with_word_view, WordView};
use super::shadow_merge::{with_unorm_view, UnormView};
use super::sharing::Sharing;
use super::sparse::emulated::TileMapBuffer;
use super::sparse::texture::{sparse_refusal, SparseExtras, SparseImage, SparseShape, Zeroing};
use super::sparse::SparsePageSize;
use super::sparse::SparseProblem;
use super::storage_placeholder::Placeholders;
use super::texture_descriptor::{
    memoryless_two_plane_cube, texture_usage, TextureDescriptor, TextureLayout, TextureProblem,
    TextureSwizzleChannels,
};
use super::tracker::{Access, Use, UseRange};
use super::two_planes;
use super::values::{CpuCacheMode, HazardTrackingMode, Region, ResourceOptions, StorageMode};
use super::view_plan::{
    check_view, keeps_components, reinterpretation_refused, ViewRecord, ViewRefusal, ViewRequest,
};
use super::Device;
use crate::device::functions::Vk;
use crate::release;
use ash::vk;
use ash::vk::Handle;
use decoded_twin::{Decoded, PassScratch};
use first_move::{FirstMove, Fresh, Pattern, Patterned, TexelCopy, PATTERN_BYTES};
use heap_tiled::{FreshDepth, FreshPattern};
use heap_two_planes::TwoPlaneHeapLayout;
use metal2vulkan::bind::{AstcDecode, EtcDecode, ImageViewDesc, PvrtcDecode, TextureFacts};
use metal2vulkan::device::{
    DeviceDescription, DeviceProfile, DriverFact, FormatSupport, VkFeature,
};
use metal2vulkan::error::{
    Capability, Feature, LinearNeed, RuntimeDriverDefect, SparseNeed, Unsupported,
};
use metal2vulkan::metal::format::{
    aspects, is_depth_stencil, storage_view_formats, ycbcr_format, ClearRoute, ExtendedRange,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageUsageFlags,
    ImageViewType, SampleCountFlags,
};
use stand_in::StandIn;
use std::any::Any;
use std::borrow::Cow;
use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;
use wrapped::WrappedImage;

mod attached_reads;
mod attributes;
pub(super) mod copy_images;
pub(crate) mod decoded_twin;
pub(crate) mod first_move;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod first_move_lifetimes;
pub(super) mod heap;
pub(super) mod heap_tiled;
pub(super) mod heap_two_planes;
pub(crate) mod held_depth;
pub(crate) mod host_copy;
pub(super) mod kept_image;
pub(super) mod requirements;
#[cfg(unix)]
mod rotation;
mod shared;
pub(super) mod signed_depth;
mod staging_copy;
pub(super) mod stand_in;
pub(super) mod storage_form;
pub(super) mod surface_depth;
pub(super) mod surface_two_planes;
mod twin_decode;
mod usage_shadow;
mod view_shadow;
pub(super) mod wrapped;
mod ycbcr_image;

pub(crate) use attached_reads::feedback_candidate;
pub use shared::SharedTextureHandle;
pub use wrapped::{VkImageLayouts, WrappedImageProblem};

/// The kind of multisample image [`TextureState::sample_moves`] answers for.
pub(super) type SampleMovesKey = (Format, PixelFormat, ImageUsageFlags, SampleCountFlags);

/// What a device's textures share: the format facts their images are planned from, and how an
/// image is placed in a memory block, apart from linear resources (and, on a driver that needs
/// it, from the compression data's chunk).
pub(super) struct TextureState {
    formats: FormatSupport,
    /// `shaderStorageImageMultisample` is enabled.
    storage_multisample: bool,
    /// What the device's driver is measured to keep in a `D32_SFLOAT` image.
    depth_images: DepthImageFacts,
    /// `ImageFacts::linear_attachments`.
    linear_attachments: bool,
    /// `ImageFacts::multisample_depth_twin`.
    multisample_depth_twin: bool,
    /// How the runtime places an image in a memory block of its own.
    placement: ImagePlacement,
    /// The verdicts of `generate_mipmaps`' per-format checks (`mipmap.rs`).
    pub(super) mipmaps: Mutex<MipmapChecks>,
    /// The neutral images and samplers bindings take where nothing is bound (`neutral.rs`).
    pub(super) neutrals: Mutex<Neutrals>,
    /// Parked images, and the makings of the live recyclable ones (`recycle.rs`).
    pub(super) recycled: Mutex<Recycler<vk::Image, ImageMaking>>,
    /// The placeholder images of the colors a render pass instance leaves out of its subpass for
    /// the pixel interlock (`storage_placeholder.rs`).
    pub(super) storage_placeholders: Mutex<Placeholders>,
    /// Whether the usage a framebuffer fetch's form needs costs a color render target's image
    /// nothing, by its description (`framebuffer_fetch.rs`).
    pub(super) usage_answers: Mutex<UsageAnswers>,
    /// Whether the device moves a multisample image's samples to and from a buffer, by the
    /// image's format, pixel format, usage and sample count, and what the moves into the image
    /// change (`drawn-depth-copy`): a heap texture's tiled shadow's moves (`texture/heap.rs`,
    /// `Device::sample_moves_run`), one entry a kind of image, `None` where it does not.
    pub(super) sample_moves: Mutex<Vec<(SampleMovesKey, Option<Option<&'static str>>)>>,
    /// Each pixel format's texture capabilities, worked out at its first query
    /// (`families/formats.rs`).
    pub(super) format_answers: Mutex<HashMap<PixelFormat, FormatAnswers>>,
    /// Each Vulkan format's `DRM_FORMAT_MOD_LINEAR` tiling features, asked of the driver at the
    /// first image of that tiling planned in it (`dma_buf.rs`'s `modifier_features`): one entry
    /// per format at most.
    #[cfg(unix)]
    pub(super) modifier_features: Mutex<HashMap<Format, FormatFeatureFlags2>>,
    /// The memory types a heap's buffer and its textures' images take, worked out at the first
    /// heap's (`texture/heap.rs`).
    pub(super) heap_types: std::sync::OnceLock<heap::HeapTypes>,
    /// The memory requirements of the images heap textures' plans asked about, by their creation
    /// parameters (`texture/requirements.rs`).
    pub(super) requirements: Mutex<requirements::RequirementAnswers>,
    /// The fetched render targets waiting for their storage form (`texture/storage_form.rs`).
    storage_forms: storage_form::StorageForms,
    /// The new resources' first GPU work, which the device's next submission does
    /// (`texture/first_move.rs`).
    pub(super) first_moves: first_move::FirstMoves,
    /// The decoded twins' CPU decodes and the thread that runs them (`texture/twin_decode.rs`).
    pub(super) twin_decodes: twin_decode::TwinDecodes,
    /// The decodes queued for ASTC carriers' second decoded twins, which the device's next batch
    /// with work records (`texture/decoded_twin.rs`).
    pub(super) second_decodes: decoded_twin::SecondDecodes,
    /// The multisampled depth textures held in the attachment layout between passes
    /// (`texture/held_depth.rs`).
    pub(crate) held: held_depth::HeldImages,
}

impl TextureState {
    /// The texture state of a device enabling `description`, whose `bufferImageGranularity` is
    /// `buffer_image_granularity`, `linearColorAttachment` where `linear_attachments`, and a
    /// multisampled `Depth32Float` texture held in colour where `multisample_depth_twin`
    /// (`MultisampleDepthTwinPolicy::Exact`).
    pub(super) fn new(
        description: &DeviceDescription,
        buffer_image_granularity: u64,
        linear_attachments: bool,
        multisample_depth_twin: bool,
    ) -> TextureState {
        TextureState {
            formats: description.formats.clone(),
            linear_attachments,
            multisample_depth_twin,
            storage_multisample: description.has(VkFeature::ShaderStorageImageMultisample),
            depth_images: DepthImageFacts::of(&description.driver_facts()),
            placement: ImagePlacement::new(
                buffer_image_granularity,
                &crate::driver_defects::of(description),
            ),
            mipmaps: Mutex::new(MipmapChecks::new()),
            neutrals: Mutex::new(Neutrals::default()),
            recycled: Mutex::new(Recycler::new(recycle::IMAGES)),
            storage_placeholders: Mutex::new(Placeholders::default()),
            usage_answers: Mutex::new(UsageAnswers::default()),
            sample_moves: Mutex::new(Vec::new()),
            format_answers: Mutex::new(HashMap::new()),
            #[cfg(unix)]
            modifier_features: Mutex::new(HashMap::new()),
            heap_types: std::sync::OnceLock::new(),
            requirements: Mutex::new(requirements::RequirementAnswers::new(
                description.has(VkFeature::Maintenance4),
            )),
            storage_forms: storage_form::StorageForms::default(),
            // A batch holds the texture slots' writes in place of each registration waiting for
            // its place where the heap sets take writes while pending, the
            // `UPDATE_UNUSED_WHILE_PENDING` the device's profile derives from the same feature
            // (`heap_sets.rs`).
            first_moves: first_move::FirstMoves::new(
                !description.has(VkFeature::DescriptorBindingUpdateUnusedWhilePending),
            ),
            twin_decodes: twin_decode::TwinDecodes::default(),
            second_decodes: Mutex::new(Vec::new()),
            held: held_depth::HeldImages::default(),
        }
    }

    /// The facts images are planned from on a device of `profile`.
    pub(super) fn facts<'a>(&'a self, profile: &'a DeviceProfile) -> ImageFacts<'a> {
        ImageFacts {
            profile,
            formats: &self.formats,
            storage_multisample: self.storage_multisample,
            depth_images: self.depth_images,
            linear_attachments: self.linear_attachments,
            multisample_depth_twin: self.multisample_depth_twin,
        }
    }

    /// The granularity optimal images are placed and sized in.
    pub(super) fn granularity(&self) -> u64 {
        self.placement.granularity
    }

    /// How the runtime places an image in a memory block of its own on the device.
    pub(super) fn placement(&self) -> ImagePlacement {
        self.placement
    }
}

/// The bytes of compression data the aux map of Intel's Xe-LP GPUs gives each 64-KiB block of an
/// image's texels: one chunk per block (Bspec 44930, as Mesa's `intel_aux_map.h` cites it).
const AUX_MAP_CHUNK: u64 = 256;

/// How the runtime places an image in a memory block of its own: at a multiple of the larger of
/// the image's alignment and `bufferImageGranularity`, its size rounded up to whole granularity
/// pages, so no linear resource shares a page with it.
///
/// A heap's image takes neither ([`ImagePlacement::in_heap`]): the heap's own buffer covers its
/// memory from end to end (`texture/heap.rs`), so no padding of the image's footprint can put it
/// in a granularity page no buffer lies in, and the padding only widens the number a program
/// places heap resources by, which is `heapTextureSizeAndAlignWithDescriptor:`'s answer. The aux
/// map's chunks stay: those bytes are the image's own compression data, written past its reported
/// size.
///
/// Where the device's driver answers a compressed image's size short of the memory its
/// compression data takes ([`RuntimeDriverDefect::ImageSizeShortOfCompressionData`]: ANV on
/// Intel's Xe-LP GPUs), the size is also rounded up to whole 256-byte chunks of the aux map's. ANV
/// puts the compression data last in the image's memory, at a multiple of 4096 bytes from the
/// image's start and `size / 256` bytes long (16 for a 4096-byte image), so the rounding ends the
/// image at the end of its last chunk, and no other resource lies in a chunk the GPU writes the
/// image's compression data through: a resource bound right after the 16 bytes lost the rest of
/// their 64-byte line to a stale write-back, and one bound at the next 256 bytes kept its contents.
/// The cost there is at most 240 bytes an image; other devices keep the image's own size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImagePlacement {
    /// `bufferImageGranularity`, a power of two.
    granularity: u64,
    /// The multiple an image's size is rounded up to: the granularity, or, where the driver needs
    /// it, at least the aux map's chunk; a power of two.
    size_unit: u64,
    /// The bytes of compression data the driver writes past an image's reported size, rounded up
    /// to: the aux map's chunk where the driver needs it, else 1; a power of two.
    chunk: u64,
}

impl ImagePlacement {
    /// The placement on a device whose `bufferImageGranularity` is `granularity` and whose
    /// driver has `defects` (`driver_defects.rs`).
    pub(crate) fn new(granularity: u64, defects: &BTreeSet<RuntimeDriverDefect>) -> ImagePlacement {
        let granularity = granularity.max(1).next_power_of_two();
        let chunk = if defects.contains(&RuntimeDriverDefect::ImageSizeShortOfCompressionData) {
            AUX_MAP_CHUNK
        } else {
            1
        };
        ImagePlacement {
            granularity,
            size_unit: granularity.max(chunk),
            chunk,
        }
    }

    /// This placement for an image bound into a heap's memory: the image's own alignment and
    /// size, in whole chunks of compression data where the driver writes them past the size (the
    /// type docs).
    pub(crate) fn in_heap(self) -> ImagePlacement {
        ImagePlacement {
            granularity: 1,
            size_unit: self.chunk,
            chunk: self.chunk,
        }
    }

    /// The placement an image of the driver's `requirements` takes.
    pub(crate) fn place(self, requirements: vk::MemoryRequirements) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            alignment: requirements.alignment.max(self.granularity),
            size: requirements.size.next_multiple_of(self.size_unit),
            ..requirements
        }
    }
}

/// A Metal texture (`MTLTexture`), or a view of one. Cheap to clone (a retain); `Send + Sync`. Two handles are
/// equal when they are the same texture object (a view is another object than its texture).
#[derive(Clone)]
pub struct Texture(Arc<TextureShared>);

pub(super) struct TextureShared {
    storage: Storage,
    /// What this texture presents of its image.
    record: ViewRecord,
    /// The texture a view was made from (`parentTexture`).
    parent: Option<Texture>,
    /// The texture's `gpuResourceID`, its own also for a view (`resource_ids.rs`).
    id: IdHold,
    /// The texture object's `label`, its own also for a view (`texture/attributes.rs`).
    label: Mutex<Option<String>>,
    /// Its identity (`object_id`), its own also for a view.
    object_id: ObjectId,
    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`), its
    /// own also for a view.
    held_by: HeldBy,
    /// The image a slot binding takes where the texture's image and usage both lack the usage
    /// the binding needs (`texture/usage_shadow.rs`), made at the first such binding.
    usage_shadow: Mutex<Option<Arc<usage_shadow::UsageShadow>>>,
    /// The storage images a draw that reads the texture where its pass renders it takes the
    /// texture's texels into before it reads them (`texture/attached_reads.rs`), one a form, each
    /// made at the first such draw: on the texture a view was made from, never a view's own.
    snapshot: Mutex<Vec<Texture>>,
    /// A memoryless texture's backings, one per queue and subresource, made at the first pass
    /// that needs one and kept with the texture (`attachment_stand_in::kept_backing`).
    kept_backings: Mutex<Vec<KeptBacking>>,
    /// A two-plane YCbCr texture's render shadow, which its passes render into
    /// (`texture/ycbcr_image.rs`), or that of a view its bindings present in its image's own
    /// format (`Texture::view_render_shadow`), made at the first pass that renders it.
    render_shadow: Mutex<Option<Texture>>,
    /// A heap's texture's heap (a sparse, automatic or placement heap's), which the texture
    /// object keeps, as Metal's does; its image holds it weakly (`SparseImage::heap`,
    /// `HeapPlace::heap`), so a command buffer that keeps the image keeps nothing of the heap but
    /// its memory, and the heap goes at the program's release, as Metal frees it.
    heap: Option<Heap>,
    /// The facts a binding plan reads of the texture ([`facts`]), worked out at the first draw or
    /// plan that reads them and kept: they follow from what the texture is made with alone (its
    /// record, its image's plan and twins, its storage) and the device's profile, none of which
    /// changes while it lives. Working them out again at every draw that binds it was each
    /// bound texture's walk of its storage view formats and image plan.
    ///
    /// [`facts`]: Texture::facts
    facts: OnceLock<TextureFacts>,
}

impl Drop for TextureShared {
    /// A program's texture going: the device's texture table no longer names it, and its image
    /// waits for the command buffers that may still read it through its ID (`heap_sets::gone`).
    /// Then the texture object is gone, as Metal frees it: the caller's range token drops with the
    /// last object over its image or texels (`ObjectsToken`).
    fn drop(&mut self) {
        let (owned, device) = match (&self.storage, &self.heap) {
            (Storage::Image(image), Some(_)) => (Owned::TextureInHeap, &image.device),
            (Storage::Image(image), None) => (Owned::TextureImage, &image.device),
            (Storage::Shadow { image, .. }, _) => (Owned::TextureShadow, &image.device),
            (Storage::Alias { image, .. }, _) => (Owned::TextureAlias, &image.device),
            (Storage::Texels(texels), _) => (Owned::TextureTexels, texels.buffer.device()),
        };
        device.shared().counts.churn.released(owned);
        let id = self.id.id().0;
        if id != 0 {
            let (device, keeper): (&Device, Option<Box<dyn Any + Send + Sync>>) =
                match &self.storage {
                    Storage::Image(image)
                    | Storage::Alias { image, .. }
                    | Storage::Shadow { image, .. } => {
                        (&image.device, Some(Box::new(Arc::clone(image))))
                    }
                    Storage::Texels(texels) => (texels.buffer.device(), None),
                };
            heap_sets::gone(device, id, self.id.serial(), keeper);
        }
        // No lock is held: the token goes with no runtime lock held.
        let last = self
            .storage
            .objects()
            .is_some_and(ObjectsToken::object_gone);
        // The last texture object over a heap's texture's image gives its range back to an
        // automatic heap, as Metal's texture does at its release, whatever keeps the image. First
        // its waiting move goes unmade: the range may be another resource's from here, which a
        // later submission's move of this image would write over.
        if let (true, Some(heap)) = (last, &self.heap) {
            if let Some(place) = self.storage.heap_place() {
                if let Storage::Image(image) = &self.storage {
                    drop(image.take());
                }
                place.give_back(heap);
            }
        }
    }
}

/// What holds a texture's texels; a view's are its texture's (the `Arc`s cloned).
#[derive(Clone)]
enum Storage {
    /// An image, shared by the texture and every view of it.
    Image(Arc<Image>),
    /// A buffer's bytes: a texture buffer.
    Texels(Arc<Texels>),
    /// A buffer's bytes, and a linear image aliasing them in the buffer's memory.
    Alias {
        image: Arc<Image>,
        texels: Arc<Texels>,
    },
    /// A buffer's bytes, and an image the encoders copy them into and back (`shadow.rs`).
    Shadow {
        image: Arc<Image>,
        texels: Arc<Texels>,
        copies: Arc<ShadowCopies>,
    },
}

impl Storage {
    /// Where an automatic or placement heap's texture's image (a view's too) lies in its heap.
    fn heap_place(&self) -> Option<&heap::HeapPlace> {
        match self {
            Storage::Image(image) => image.owner.heap_place(),
            Storage::Texels(_) | Storage::Alias { .. } | Storage::Shadow { .. } => None,
        }
    }

    /// The texture objects over an image or texels of the texture's own and the caller's range
    /// token for them (`ObjectsToken`); `None` over another resource's bytes.
    fn objects(&self) -> Option<&ObjectsToken> {
        match self {
            Storage::Image(image) => Some(&image.objects),
            Storage::Texels(texels) => Some(&texels.objects),
            Storage::Alias { .. } | Storage::Shadow { .. } => None,
        }
    }
}

/// One `VkImage` and its memory, shared by a texture and every view of it.
struct Image {
    device: Device,
    raw: Raw,
    /// Who owns the image and its memory, and so what its drop releases.
    owner: Owner,
    plan: ImagePlan,
    options: ResourceOptions,
    allow_gpu_optimized_contents: bool,
    identity: Identity,
    /// The `VkImageView`s bindings of the image's textures needed (`binding_view.rs`), made at
    /// their first bind. A finite set per image: its view types, formats and level ranges. Views
    /// of the `VkImage` it presents: they go as it presents another (a stand-in, its own again, a
    /// rebind's or a storage form's), each binding making its view again.
    views: Mutex<KeyMap<VkView, vk::ImageView>>,
    /// The `VkImageView`s of its own `VkImage` that the runtime's copies of its texels hold for
    /// the image's life ([`own_view`](Self::own_view)): a shadow's word view
    /// (`shadow_copy_back.rs`), a YCbCr conversion's (`texture/ycbcr_image.rs`). Never of a
    /// stand-in, and never let go as the image presents one or its own again: its drop destroys
    /// them, before its `VkImage`.
    own_views: Mutex<KeyMap<VkView, vk::ImageView>>,
    /// The objects of exact color resolves into the image that its passes made, kept for its next
    /// pass of the same views (`color_resolve.rs`).
    resolves: KeptResolves,
    /// The image's depth twin: a `D32_SFLOAT` image of the same texels, which the views a
    /// comparison reads present, for an `R32_SFLOAT` image holding a `Depth32Float` texture.
    twin: Option<Box<Image>>,
    /// The image's decoded twin (`texture/decoded_twin.rs`): a BC1 to BC5 image's texels as
    /// Metal decodes them, which its sampled views present, on a device whose driver is not
    /// measured to decode them so; an ASTC or PVRTC carrier's.
    decoded: Option<Box<Decoded>>,
    /// An ASTC carrier's second decoded twin (`texture/decoded_twin.rs`): its blocks' decode in
    /// the texture's other colour space (its LDR form's for an sRGB texture, its sRGB form's for
    /// an LDR one), made at the first view of the texture in that form.
    second: OnceLock<Box<Decoded>>,
    /// The storage form a fetched render target's image may take once (`texture/storage_form.rs`).
    form: storage_form::Form,
    /// Whether a blit wrote bytes outside the image's texels, which Metal's blit of a two-plane
    /// extended-range texture keeps and the runtime does not hold (`two_planes.rs`).
    kept_outside: AtomicBool,
    /// A two-plane extended-range texture's bytes between the rows of plane 0, which Metal's
    /// fills and blits with a buffer write and its blits read back, and the image does not hold
    /// (`two_planes/buffer_rows.rs`): plane 0 of each slice laid out as Metal lays it, made at
    /// the first such write.
    between: OnceLock<Buffer>,
    /// Its 32-bit float depth may hold a value whose magnitude stand-in is not itself (a negative
    /// value, a NaN, or, where the device's depth test compares a denormal as zero, a denormal),
    /// which a write that may store one set: a pass that loads it renders over its magnitude
    /// stand-in, or its words are stripped of their signs after a pass with depth writes whose
    /// stores the device's pass leaves signed (`signed_depth.rs`). Set once, never cleared; only
    /// on an image with a form ([`Texture::magnitude_form`]).
    signed_depth: AtomicBool,
    /// While the bytes of its texture are detached (`detach.rs`): the stand-in it presents
    /// (`stand_in.rs`), whose read image is its `VkImage` meanwhile, and, for an image in memory of
    /// its own, its own image and views, which a reattach swaps back. Locked under `views`.
    detached: Mutex<Option<DetachedImage>>,
    /// Its `VkImage`'s move to `GENERAL`, until the device's next submission takes it
    /// (`texture/first_move.rs`); `None` once taken, and for a caller's image or a sparse one.
    first_move: Mutex<Option<FirstMove>>,
    /// It was made with host-transfer usage, moved to `GENERAL` and zeroed on the host, so its
    /// CPU copies may run there (`texture/host_copy.rs`).
    host_copies: bool,
    /// Its `VkImage` has linear tiling, or `DRM_FORMAT_MOD_LINEAR`'s, whose rows lie as a linear
    /// image's: a texture over a buffer's memory (`alias_image`), or a heap's linear image
    /// (`texture/heap.rs`).
    linear: bool,
    /// It was moved to `GENERAL` and zeroed on the device's staging queue as it was made, so its
    /// CPU copies may run there (`texture/staging_copy.rs`).
    staging_copies: bool,
    /// The texture objects over the image (the texture, its views, the textures opened from its
    /// handles), and the caller's token for its texture's range
    /// (`Device::new_texture_with_range_token`), set once as the texture is made: dropped as the
    /// last of them goes (`TextureShared`'s drop), whatever keeps the image after that.
    objects: ObjectsToken,
}

/// An image presenting its stand-in while its texture's bytes are detached (`Image::detached`).
struct DetachedImage {
    stand_in: Arc<StandIn>,
    /// The image's own `VkImage`, for an image in memory of its own: a reattach swaps it back,
    /// and the image's drop destroys it. `None` for one bound to the detached pages, which went
    /// with them; a reattach makes it again over the new pages (`Device::rebound_heap_images`).
    own: Option<vk::Image>,
}

/// What becomes of an `ImageObjects`'s `VkImage` as it goes.
enum RawGoes {
    /// It is destroyed.
    Destroyed,
    /// It is a stand-in's read image: the stand-in is let go of (`stand_in.rs`).
    WithStandIn(Arc<StandIn>),
    /// It lives on as the image of a texture whose bytes are detached, in memory of its own
    /// (`DetachedImage::own`): only the views go.
    Kept,
}

/// An image's `VkImage`: its own for good, but for a texture bound to the bytes of a heap over
/// host pages, whose image a rebind re-makes over the new pages ([`Image::swap_in`]). With it,
/// what keeps that `VkImage` beside the image ([`RawHold`]).
struct Raw {
    image: AtomicU64,
    import: Mutex<Option<RawHold>>,
}

/// What keeps an image's `VkImage` beside the image: the dedicated import of dma-buf pages it is
/// bound to (`dma_buf.rs`), freed after it is destroyed.
pub(super) enum RawHold {
    Import { _pages: ImageImport },
}

/// The dedicated import of the pages a texture's image over dma-buf pages is bound to
/// (`dma_buf.rs`), freed after the image is destroyed.
#[cfg(unix)]
pub(super) type ImageImport = Arc<crate::dma_buf::DmaBufPages>;

/// No image is made over dma-buf pages on this system.
#[cfg(not(unix))]
pub(super) type ImageImport = std::convert::Infallible;

impl Raw {
    fn new(raw: vk::Image) -> Raw {
        Raw {
            image: AtomicU64::new(raw.as_raw()),
            import: Mutex::new(None),
        }
    }

    /// `raw`, bound to `import`, its dedicated import.
    #[cfg(unix)]
    fn imported(raw: vk::Image, import: ImageImport) -> Raw {
        Raw {
            image: AtomicU64::new(raw.as_raw()),
            import: Mutex::new(Some(RawHold::Import { _pages: import })),
        }
    }

    /// Whether the `VkImage` now is bound to a dedicated import of its own pages.
    fn dedicated(&self) -> bool {
        match &*lock(&self.import) {
            Some(RawHold::Import { .. }) => true,
            None => false,
        }
    }

    /// The `VkImage` now.
    fn get(&self) -> vk::Image {
        vk::Image::from_raw(self.image.load(Ordering::Acquire))
    }

    /// Hold `raw`, bound to `import` where it has a dedicated one, from now on: the `VkImage` it
    /// held and that one's import.
    fn swap(&self, raw: vk::Image, import: Option<RawHold>) -> (vk::Image, Option<RawHold>) {
        let mut held = lock(&self.import);
        let old = self.image.swap(raw.as_raw(), Ordering::AcqRel);
        let old_import = std::mem::replace(&mut *held, import);
        (vk::Image::from_raw(old), old_import)
    }
}

/// A `VkImage` and views of it that go when this does: a heap texture's image re-made over a
/// rebind's new pages until it is swapped in, then the image and views it replaced, which the
/// old pages keep until the last command buffer that used them completed (`rebind.rs`).
pub(super) struct ImageObjects {
    device: Device,
    raw: vk::Image,
    views: Vec<vk::ImageView>,
    /// What becomes of `raw` as this goes.
    raw_goes: RawGoes,
    /// What keeps `raw` beside these objects ([`RawHold`]): a dedicated import, freed as this goes,
    /// after `raw` is destroyed.
    import: Option<RawHold>,
}

impl Drop for ImageObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command buffer uses them: one that used the image before it was
        // swapped out holds the pages that keep them until it completed, and a new image no
        // command has used yet.
        unsafe {
            for &view in &self.views {
                release::object(raw, view);
            }
            // A stand-in's read image goes with the stand-in, once its last holder lets go (this
            // one's after the views above).
            match &self.raw_goes {
                RawGoes::Destroyed => {
                    release::object(raw, self.raw);
                }
                RawGoes::WithStandIn(stand_in) => {
                    debug_assert_eq!(stand_in.read(), self.raw, "the stand-in's read image")
                }
                RawGoes::Kept => {}
            }
        }
    }
}

/// A heap texture's image re-made over a rebind's new pages ([`Device::rebound_heap_images`]),
/// swapped in once the rebind is made; dropped unapplied, the new image goes.
pub(super) struct ImageSwap {
    image: Arc<Image>,
    objects: ImageObjects,
    /// The new `VkImage`'s move to `GENERAL`, which the image notes as the swap is made.
    first: FirstMove,
    /// The new pages' import, which the image holds from the swap on.
    memory: MemoryHold,
}

/// A linear image for a texture over a buffer's bytes, made and not yet bound
/// ([`Device::linear_image`]): its memory requirements, the layout of its first subresource, and
/// the layout it starts in.
struct LinearMade {
    raw: vk::Image,
    requirements: vk::MemoryRequirements,
    report: vk::SubresourceLayout,
    initial: vk::ImageLayout,
}

/// Why a texture over a buffer takes no linear image: the need that failed, which the
/// shadowed-texture notice names.
pub(super) enum LinearRefusal {
    /// What the device's linear images lack ([`LinearNeed`]): a feature of the format's
    /// tiling, any such image (the format query refuses it), or a limit.
    Need(LinearNeed),
    /// For host allocations: the device imports none for such an image, or only a dedicated
    /// one (its external memory features).
    NotImportable(vk::ExternalMemoryFeatureFlags),
}

impl LinearRefusal {
    /// The refusal of a linear image of `plan`, in words.
    pub(super) fn describe(&self, plan: &ImagePlan) -> String {
        let (format, usage) = (plan.format, plan.usage);
        match self {
            LinearRefusal::Need(LinearNeed::Features { format, features }) => format!(
                "the linear tiling of {format:?} lacks {features:?}, which its optimal tiling \
                 has for the texture's usage"
            ),
            LinearRefusal::Need(LinearNeed::Image) => format!(
                "the device's image format query reports no linear image of {format:?}, usage \
                 {usage:?}, flags {:?}",
                plan.flags
            ),
            LinearRefusal::Need(LinearNeed::Extent {
                required,
                available,
            }) => format!(
                "an extent of {required} texels, where its linear images of {format:?} \
                 hold {available}"
            ),
            LinearRefusal::Need(LinearNeed::MipLevels {
                required,
                available,
            }) => format!(
                "{required} mip levels, where its linear images of {format:?} hold \
                 {available}"
            ),
            LinearRefusal::Need(LinearNeed::ArrayLayers {
                required,
                available,
            }) => format!(
                "{required} array layers, where its linear images of {format:?} hold \
                 {available}"
            ),
            LinearRefusal::Need(need) => format!("{need:?} of its linear images"),
            LinearRefusal::NotImportable(features) => format!(
                "it imports no host allocation for a linear image of {format:?}, usage \
                 {usage:?}, but with a dedicated allocation (importable {}, dedicated only {})",
                features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
                features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
            ),
        }
    }
}

/// The first subresource of a colour image: what a linear image's layout is read of.
fn first_color_subresource() -> vk::ImageSubresource {
    vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    }
}

/// An image aliasing a texture's buffer bytes, or why none does: what the shadow's notice
/// (`shadowed-texture`) names.
type Aliased = Result<Arc<Image>, String>;

/// The images a rebind made for the textures over the bytes it gives pages it imported as
/// dma-bufs ([`Device::dma_buf_images_again`]), before it settled: each texture's image with its
/// new `VkImage`, bound to a dedicated import of its own pages (`dma_buf.rs`). Dropped unused,
/// they go.
#[derive(Default)]
pub(super) struct Remade(Vec<(Arc<Image>, ImageObjects)>);

impl Remade {
    /// The objects made for `image`, where some were.
    fn take(&mut self, image: &Arc<Image>) -> Option<ImageObjects> {
        let at = self
            .0
            .iter()
            .position(|(made, _)| Arc::ptr_eq(made, image))?;
        Some(self.0.swap_remove(at).1)
    }
}

impl ImageObjects {
    /// `raw`, a new image of `device`'s, with no views yet.
    fn new(device: &Device, raw: vk::Image) -> ImageObjects {
        ImageObjects {
            device: device.clone(),
            raw,
            views: Vec::new(),
            raw_goes: RawGoes::Destroyed,
            import: None,
        }
    }

    /// `raw`, a new image of `device`'s bound to `import`, its dedicated import of dma-buf pages
    /// (`dma_buf.rs`), with no views yet.
    #[cfg(unix)]
    fn imported(device: &Device, raw: vk::Image, import: ImageImport) -> ImageObjects {
        ImageObjects {
            device: device.clone(),
            raw,
            views: Vec::new(),
            raw_goes: RawGoes::Destroyed,
            import: Some(RawHold::Import { _pages: import }),
        }
    }

    /// Whether they are a stand-in's image and the views made of it (`stand_in.rs`), which
    /// command buffers reach only through the heap sets: what a reattach retires
    /// (`heap_sets::retire`) rather than leave to pages.
    pub(super) fn of_stand_in(&self) -> bool {
        matches!(self.raw_goes, RawGoes::WithStandIn(_))
    }
}

impl ImageSwap {
    /// `objects`, a new image for `image` bound to `pages`, host pages, whose move is `first`
    /// (holding the pages), to swap in.
    fn new(
        image: Arc<Image>,
        objects: ImageObjects,
        first: FirstMove,
        pages: &HostPages,
    ) -> ImageSwap {
        let memory = MemoryHold::Host {
            _import: Arc::clone(pages.import()),
        };
        ImageSwap {
            image,
            objects,
            first,
            memory,
        }
    }

    /// `objects`, a new image for `image` over `pages`, dma-buf pages, bound to a dedicated import
    /// of them (`dma_buf.rs`) and made `UNDEFINED`, to swap in: its first move acquires it from
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` for `family` (`Device::foreign_acquirer`).
    fn acquired(
        image: Arc<Image>,
        objects: ImageObjects,
        pages: &Arc<HostPages>,
        family: u32,
    ) -> ImageSwap {
        let undefined = vk::ImageLayout::UNDEFINED;
        let first = FirstMove::new(
            objects.raw,
            image.plan().format,
            undefined,
            pages_hold(pages),
        );
        ImageSwap::new(image, objects, first.acquired(family), pages)
    }

    /// Whether the swap is `image`'s.
    fn is_of(&self, image: &Arc<Image>) -> bool {
        Arc::ptr_eq(&self.image, image)
    }

    /// Swap the new image in: the image and views it replaced, for the old pages to keep. The
    /// image notes the new one's move, which replaces the old one's where no submission took that:
    /// nothing used the old `VkImage` then.
    pub(super) fn apply(self) -> ImageObjects {
        let ImageSwap {
            image,
            mut objects,
            first,
            memory,
        } = self;
        image.swap_in(&mut objects);
        *lock(&image.first_move) = Some(first);
        image.register_first_move();
        // The image holds the memory it is bound to now; the old pages hold their own.
        if let Some(held) = image.owner.memory() {
            let old = lock(held).replace(memory);
            drop(old);
        }
        objects
    }
}

impl Image {
    /// The image's plan: the one it was made of, or its storage form's once it took that
    /// (`texture/storage_form.rs`), the same but for `STORAGE` usage.
    fn plan(&self) -> &ImagePlan {
        self.form.plan().unwrap_or(&self.plan)
    }

    /// Count, under the views lock, that the image lets its views go: the views a bind kept by
    /// what they were made of are then stale (`DeviceShared::views_epoch`).
    pub(super) fn views_go(&self) {
        self.device
            .shared()
            .views_epoch
            .fetch_add(1, Ordering::AcqRel);
    }

    /// The `VkImageView` `view`, from the image's views or made now. `view` is checked against the
    /// image's plan by the caller.
    fn view(&self, view: &VkView) -> Result<vk::ImageView, Error> {
        let mut views = lock(&self.views);
        if let Some(&raw) = views.get(view) {
            return Ok(raw);
        }
        self.may_view(view)?;
        let image = match &*lock(&self.detached) {
            Some(detached) => detached.stand_in.for_usage(view.usage),
            None => self.raw.get(),
        };
        // SAFETY: the image is live; the caller checked the view against the image's plan
        // (format, usage, levels and layers), which a stand-in's images share, the sink holding
        // every usage of the plan and the read image every one but `STORAGE`.
        let raw = unsafe { self.make_view(image, view) }?;
        views.insert(*view, raw);
        Ok(raw)
    }

    /// The `VkImageView` `view` of the image's own `VkImage`, from its own views or made now,
    /// which stays until the image drops ([`Image::own_views`]): for a copy of the image's texels
    /// that holds the view as long as it holds the image. Only for an image in memory of its own
    /// that took no storage form, whose `VkImage` is its own for its life: a detach keeps it while
    /// the image presents a stand-in (`present_stand_in`), and no rebind swaps another in; a
    /// texture's own image, which alone may take a storage form (`texture/storage_form.rs`), holds
    /// no such copy. `view` is checked against the image's plan by the caller.
    fn own_view(&self, view: &VkView) -> Result<vk::ImageView, Error> {
        // Under the views lock, as every change of the `VkImage` the image presents.
        let _presented = lock(&self.views);
        let mut own = lock(&self.own_views);
        if let Some(&raw) = own.get(view) {
            return Ok(raw);
        }
        if !self.own_memory() || self.form.plan().is_some() {
            return Err(internal(
                "a held view of an image whose VkImage is not its own for its life",
            ));
        }
        self.may_view(view)?;
        // SAFETY: the image's own `VkImage` is live while the image is; the caller checked the
        // view against the image's plan.
        let raw = unsafe { self.make_view(self.own_raw(), view) }?;
        own.insert(*view, raw);
        Ok(raw)
    }

    /// Whether the device makes `view` of the image: `Ok` where it does.
    fn may_view(&self, view: &VkView) -> Result<(), Error> {
        // A device that reinterprets no image's components makes no view that would
        // (`view_plan.rs`).
        if !self.device.shared().view_reinterpretation
            && !keeps_components(self.plan().format, view.format)
        {
            return Err(Error::Unsupported(reinterpretation_refused()));
        }
        // Nor one with a component mapping where its portability subset withholds
        // `imageViewFormatSwizzle`: the format's own mapping is refused where the format is
        // chosen, so this is a texture's swizzle. A view's mapping is canonical
        // (`binding_view::compose`): a channel from itself is `IDENTITY`.
        let swizzle = VkFeature::ImageViewFormatSwizzle;
        if view.components != ComponentMapping::IDENTITY
            && self
                .device
                .compile_device()
                .profile()
                .enabled()
                .portability_withholds(swizzle)
        {
            return Err(Error::Unsupported(Unsupported::needs(
                Feature::TextureSwizzle,
                Capability::Feature(swizzle),
            )));
        }
        Ok(())
    }

    /// Make the `VkImageView` `view` of `image`.
    ///
    /// # Safety
    ///
    /// `image` is live, and `view` fits its plan (format, usage, levels and layers).
    unsafe fn make_view(&self, image: vk::Image, view: &VkView) -> Result<vk::ImageView, Error> {
        let mut usage = vk::ImageViewUsageCreateInfo::default();
        let info = view.create_info(image, &mut usage);
        // SAFETY: the caller's.
        unsafe { crate::creation::image_view(self.device.shared().vk(), &info) }.map_err(vulkan)
    }

    /// Exchange the image's `VkImage` and views with `objects`'s: the image is `objects`'s new
    /// one from now on, with no views yet, and `objects` holds what it replaced. Under the views'
    /// lock, so no view of the old image is made after.
    fn swap_in(&self, objects: &mut ImageObjects) {
        let mut views = lock(&self.views);
        self.views_go();
        let (old, import) = self.raw.swap(objects.raw, objects.import.take());
        objects.raw = old;
        objects.import = import;
        objects.views.extend(views.drain().map(|(_, view)| view));
        // A stand-in's read image swapped out goes as the stand-in's (an image in memory of its
        // own swaps its own back instead, `present_own`).
        if let Some(detached) = lock(&self.detached).take() {
            objects.raw_goes = RawGoes::WithStandIn(detached.stand_in);
        }
    }

    /// The image, shared (by its texture, or the runtime's holder of it): registered with the
    /// device's first moves where its move, or its depth twin's, waits (`texture/first_move.rs`).
    fn shared(self) -> Arc<Image> {
        let image = Arc::new(self);
        if image.move_waits() {
            image.register_first_move();
        }
        image
    }

    /// Register the image with the device's first moves (`texture/first_move.rs`).
    fn register_first_move(self: &Arc<Self>) {
        let weak: Weak<Image> = Arc::downgrade(self);
        let fresh: Weak<dyn Fresh> = weak;
        self.device.shared().textures.first_moves.register(fresh);
    }

    /// Whether its move, or its depth twin's or one of its decoded twins', waits for a
    /// submission.
    fn move_waits(&self) -> bool {
        lock(&self.first_move).is_some()
            || (self.twin.as_ref()).is_some_and(|twin| lock(&twin.first_move).is_some())
            || (self.decoded.as_ref()).is_some_and(|d| lock(&d.image.first_move).is_some())
            || (self.second.get()).is_some_and(|d| lock(&d.image.first_move).is_some())
    }

    /// Make its move now where it waits, by an internal submission waited for: for a `VkImage`
    /// handed out, which rests in `GENERAL` (`Texture::vk_image`), and holds what host calls copied
    /// in.
    fn settle(&self) -> Result<(), Error> {
        let firsts = &self.device.shared().textures.first_moves;
        let at: *const Image = self;
        if !self.move_waits() && !firsts.texels_wait(at.cast()) {
            return Ok(());
        }
        self.submit_copy(|_, _| {})
    }

    /// Record a host call's copy of its texels with `record`, on an internal submission the call
    /// waits for, which makes its own waiting move first, one of an image over host pages
    /// included: the call holds its texture, and so the pages, until the submission completed.
    /// So the copy finds it in `GENERAL`.
    fn submit_copy(&self, record: impl FnOnce(&Vk, vk::CommandBuffer)) -> Result<(), Error> {
        let key = self.identity.key;
        self.device
            .submit_internal_naming(|named| named == key, record)
    }

    /// Whether `raw` is its `VkImage`: the one it presents, or its own while it presents a
    /// stand-in.
    fn holds(&self, raw: vk::Image) -> bool {
        self.raw.get() == raw || lock(&self.detached).as_ref().and_then(|d| d.own) == Some(raw)
    }

    /// Whether a host call's copy into it may wait for the device's next submission: its
    /// `VkImage` is its own for good, in memory of its own or in a heap's own pages, which no
    /// rebind swaps (`FirstMoves::register_texels`).
    fn defers_copies(&self) -> bool {
        match &self.owner {
            Owner::Runtime { .. } => true,
            Owner::Heap(place) => matches!(*lock(&place.memory), Some(MemoryHold::Fixed { .. })),
            _ => false,
        }
    }

    /// Whether it is in memory of its own (its drop frees its placement), rather than bound to
    /// a buffer's, a heap's or the caller's.
    fn own_memory(&self) -> bool {
        self.owner.releases().1.is_some()
    }

    /// Present `stand_in` from now on, its texture's bytes detached (`detach.rs`): its `VkImage`
    /// is the stand-in's read image, and its views are made anew of the stand-in. The views it
    /// had are answered, to go once no command buffer may read them; with its own `VkImage` too
    /// where that is bound to the detached pages, while an image in memory of its own keeps its
    /// `VkImage` for a reattach to swap back, and the views its copies hold of it
    /// ([`own_view`](Self::own_view)). Nothing where it presents a stand-in already.
    fn present_stand_in(&self, stand_in: Arc<StandIn>) -> Option<ImageObjects> {
        // Declared first, the hold on the detached pages' memory goes after the locks below.
        let mut _let_go = None;
        let mut views = lock(&self.views);
        let mut detached = lock(&self.detached);
        if detached.is_some() {
            return None;
        }
        self.views_go();
        let (old, import) = self.raw.swap(stand_in.read(), None);
        // An image in memory of its own keeps its `VkImage`, which no dedicated import is bound
        // to (only an image over dma-buf pages has one).
        let own = self.own_memory().then_some(old);
        // An image bound to the detached pages lets go of their memory: its `VkImage` goes with
        // the pages.
        if let (None, Some(held)) = (own, self.owner.memory()) {
            _let_go = lock(held).take();
        }
        let raw_goes = match own {
            Some(_) => RawGoes::Kept,
            None => RawGoes::Destroyed,
        };
        *detached = Some(DetachedImage { stand_in, own });
        // A dedicated import of dma-buf pages goes with the `VkImage` bound to it.
        Some(ImageObjects {
            device: self.device.clone(),
            raw: old,
            views: views.drain().map(|(_, view)| view).collect(),
            raw_goes,
            import,
        })
    }

    /// Present its own `VkImage` again, its texture's bytes given pages again, for an image in
    /// memory of its own: the stand-in's read image and the views made of the stand-in are
    /// answered, to go once no command buffer may read them. Nothing for an image presenting no
    /// stand-in, or one bound to the detached pages, which a rebind makes again over the new
    /// pages and swaps in (`swap_in`).
    fn present_own(&self) -> Option<ImageObjects> {
        let mut views = lock(&self.views);
        let mut detached = lock(&self.detached);
        let own = detached.as_ref()?.own?;
        let stand_in = detached.take()?.stand_in;
        self.views_go();
        let (old, import) = self.raw.swap(own, None);
        Some(ImageObjects {
            device: self.device.clone(),
            raw: old,
            views: views.drain().map(|(_, view)| view).collect(),
            raw_goes: RawGoes::WithStandIn(stand_in),
            import,
        })
    }

    /// Whether it presents a stand-in.
    fn presents_stand_in(&self) -> bool {
        lock(&self.detached).is_some()
    }

    /// Its own `VkImage`: the one it presents, or, while it presents a stand-in, the one it
    /// keeps in memory of its own (what a copy of its texels takes, `view_shadow.rs`).
    fn own_raw(&self) -> vk::Image {
        match &*lock(&self.detached) {
            Some(DetachedImage { own: Some(own), .. }) => *own,
            _ => self.raw.get(),
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // Once the process is exiting, the image is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        // A held image's scheduler state goes with it (`texture/held_depth.rs`).
        shared.textures.held.remove(self.identity.key);
        let owned = match &self.owner {
            Owner::Runtime { .. } => Owned::ImageOwnMemory,
            Owner::Buffer { .. } => Owned::ImageInBuffer,
            Owner::Heap(_) | Owner::HeapTwin => Owned::ImageInHeap,
            _ => Owned::ImageOther,
        };
        shared.counts.churn.released(owned);
        let (image, memory) = self.owner.releases();
        // An image that took its storage form is in the form's memory: the memory it was made in
        // goes with its first `VkImage`, which the switch's command buffer holds until it
        // completes (`texture/storage_form.rs`).
        let memory = self.form.placement().or(memory);
        // SAFETY: no pending command buffer uses the image or a view of it: a committed command
        // buffer holds every texture it uses until it completes. The views are the runtime's
        // own; the image only where it made it.
        // An image presenting a stand-in destroys its own `VkImage` where it kept one, and
        // never the stand-in's.
        let (raw, detached) = match &*lock(&self.detached) {
            Some(detached) => (detached.own, true),
            None => (Some(self.raw.get()), false),
        };
        // The kept resolve objects go before the views they were made of.
        self.resolves.clear();
        // SAFETY: as above.
        unsafe {
            for &view in lock(&self.views).values() {
                release::object(shared.vk(), view);
            }
            for &view in lock(&self.own_views).values() {
                release::object(shared.vk(), view);
            }
        }
        // An image the recycler noted goes to it, its memory with it, unless it is no longer the
        // image of its making alone (`recycle.rs`).
        let shareable = matches!(
            self.owner,
            Owner::Runtime {
                shareable: true,
                ..
            }
        );
        let own = raw.filter(|_| image);
        let parked = match (own, memory) {
            (Some(raw), Some(placement)) => {
                let mut recycled = lock(&shared.textures.recycled);
                match recycled.holds(raw) {
                    true if !detached && !shareable && self.form.placement().is_none() => {
                        let now = Instant::now();
                        recycled.park(raw, placement.handed_over(), now).ok()
                    }
                    true => {
                        recycled.forget(raw);
                        None
                    }
                    false => None,
                }
            }
            (Some(raw), None) => {
                lock(&shared.textures.recycled).forget(raw);
                None
            }
            (None, _) => None,
        };
        match parked {
            Some(gone) => {
                shared.counts.churn.parked(Recycled::Image);
                self.device.let_go_parked(gone);
            }
            None => {
                if let Some(raw) = own {
                    // SAFETY: as above.
                    unsafe { release::object(shared.vk(), raw) };
                }
                if let Some(placement) = memory {
                    lock(&shared.memory).free(shared.vk(), placement);
                }
            }
        }
        if let Owner::Caller(wrapped) = &self.owner {
            shared.interop.wrapped.forget(wrapped);
        }
        shared.fresh.forget(self.identity.key);
        // A heap's range went back as the last texture object over the image went, or goes
        // back as the `HeapPlace` drops.
    }
}

impl Fresh for Image {
    fn take(&self) -> Vec<FirstMove> {
        // A move of a `VkImage` the image no longer holds goes unmade: one bound to pages a
        // detach let go of (`present_stand_in`), destroyed once nothing may read it, which no
        // command uses again.
        let own = lock(&self.first_move)
            .take()
            .filter(|first| self.holds(first.image));
        let twin = (self.twin.as_ref()).and_then(|twin| {
            lock(&twin.first_move)
                .take()
                .filter(|first| twin.holds(first.image))
        });
        let decoded = (self.decoded.as_deref().into_iter())
            .chain(self.second.get().map(Box::as_ref))
            .filter_map(|decoded| {
                lock(&decoded.image.first_move)
                    .take()
                    .filter(|first| decoded.image.holds(first.image))
            });
        own.into_iter().chain(twin).chain(decoded).collect()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        for first in moves {
            let twin = (self.twin.as_deref()).filter(|twin| twin.raw.get() == first.image);
            let decoded = (self.decoded.as_deref().into_iter())
                .chain(self.second.get().map(Box::as_ref))
                .map(|decoded| &decoded.image)
                .find(|decoded| decoded.raw.get() == first.image);
            let image = twin.or(decoded).unwrap_or(self);
            // A rebind may have swapped another `VkImage` in meanwhile, whose move it noted.
            if image.holds(first.image) {
                let mut waiting = lock(&image.first_move);
                if waiting.is_none() {
                    *waiting = Some(first);
                }
            }
        }
    }

    fn rebindable(&self) -> Option<ResourceKey> {
        let waiting = lock(&self.first_move);
        (waiting.as_ref())
            .is_some_and(|first| first.rebindable())
            .then_some(self.identity.key)
    }
}

/// Who owns an image and its memory.
pub(super) enum Owner {
    /// The runtime made the image and placed it in memory of its own; both go with it. `shareable`
    /// for a texture of `newSharedTextureWithDescriptor:` (`texture/shared.rs`). `purgeable` is
    /// the purgeable state of the image's texture and of every texture opened from its handles,
    /// where Metal keeps one (`texture/attributes.rs`).
    Runtime {
        placement: Placement,
        shareable: bool,
        purgeable: Mutex<PurgeableState>,
    },
    /// The runtime made a linear image in a buffer's memory: the image goes
    /// with it, the memory stays the buffer's. The image holds the memory, so it outlives every
    /// command that reaches the image, a batch that took its first move and holds only the image
    /// included. It holds the memory alone, never the buffer: the texture objects hold the buffer,
    /// as a Metal texture over a buffer retains it, and Metal frees the buffer at their release
    /// even while a command buffer still reaches the texture through its ID: its
    /// range and range token go then, and a no-copy buffer's deallocator runs (the caller's hold on
    /// an IOSurface goes).
    Buffer {
        /// The memory the image is bound to ([`MemoryHold`]); `None` while the buffer's bytes
        /// are detached. A rebind's swap holds the new pages' instead.
        memory: Mutex<Option<MemoryHold>>,
    },
    /// The caller made the image and its memory, and keeps both (`new_texture_with_vk_image`,
    /// `texture/wrapped.rs`): the runtime never destroys or frees either.
    Caller(Arc<WrappedImage>),
    /// The runtime made a sparse image of a sparse heap (`sparse/texture.rs`): the image goes with
    /// it; its memory is bound by unit as Metal maps its tiles.
    Sparse(Box<SparseImage>),
    /// The runtime made an image in an automatic or placement heap's memory (`texture/heap.rs`):
    /// the image goes with it, the memory stays the heap's, and an automatic heap takes the range
    /// back.
    Heap(Box<heap::HeapPlace>),
    /// The runtime made the depth twin of a heap texture in the heap's memory, inside the texture's
    /// range (`texture/heap.rs`): the image goes with it; the memory and the range are the
    /// texture's image's.
    HeapTwin,
    /// The runtime made a heap's depth or stencil image in memory of its own, a shadow of the
    /// heap's bytes in the texture's range (`texture/heap.rs`): the image and its memory go with
    /// it, and an automatic heap takes the range back.
    HeapShadow(Box<heap::HeapShadow>),
    /// The runtime made a heap's two-plane extended-range image in memory of its own, apart from
    /// the heap's bytes in the texture's range (`texture/heap.rs`): the image and its memory go
    /// with it, and an automatic heap takes the range back.
    HeapUnaliased(Box<heap::HeapUnaliased>),
    /// The runtime made the image of a texture view of another component layout in memory of
    /// its own, a shadow of the viewed image's texels on a device whose Vulkan views reinterpret
    /// no image's components (`texture/view_shadow.rs`): the image and its memory go with it.
    ViewShadow(Box<view_shadow::ViewShadow>),
}

impl Owner {
    /// The hold on another resource's memory an image of this owner is bound to: a buffer's, or
    /// an automatic or placement heap's.
    fn memory(&self) -> Option<&Mutex<Option<MemoryHold>>> {
        match self {
            Owner::Buffer { memory } => Some(memory),
            Owner::Heap(place) => Some(&place.memory),
            _ => None,
        }
    }

    /// Where an automatic or placement heap's texture's image lies in its heap.
    fn heap_place(&self) -> Option<&heap::HeapPlace> {
        match self {
            Owner::Heap(place) => Some(place),
            Owner::HeapShadow(shadow) => Some(&shadow.place),
            Owner::HeapUnaliased(unaliased) => Some(&unaliased.place),
            _ => None,
        }
    }

    /// What dropping an image of this owner releases: whether the `VkImage` is destroyed, and
    /// the placement freed.
    pub(super) fn releases(&self) -> (bool, Option<&Placement>) {
        match self {
            Owner::Runtime { placement, .. } => (true, Some(placement)),
            Owner::HeapShadow(shadow) => (true, Some(&shadow.placement)),
            Owner::HeapUnaliased(unaliased) => (true, Some(&unaliased.placement)),
            Owner::ViewShadow(shadow) => (true, Some(&shadow.placement)),
            Owner::Buffer { .. } | Owner::Sparse(_) | Owner::Heap(_) | Owner::HeapTwin => {
                (true, None)
            }
            Owner::Caller(_) => (false, None),
        }
    }
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = &self.0.record;
        f.debug_struct("Texture")
            .field("texture_type", &r.texture_type)
            .field("pixel_format", &r.pixel_format)
            .field("extent", &r.extent())
            .field("mipmap_level_count", &r.level_count)
            .field("array_length", &r.array_length())
            .field("view", &self.0.parent.is_some())
            .finish()
    }
}

impl PartialEq for Texture {
    fn eq(&self, other: &Texture) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Texture {}

impl std::hash::Hash for Texture {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

/// The aspects of an image of `format`: its depth and stencil, or its color.
pub(super) fn image_aspects(format: Format) -> ImageAspectFlags {
    let block = format.block();
    let mut aspects = ImageAspectFlags::empty();
    if block.is_some_and(|b| b.depth.is_some()) {
        aspects |= ImageAspectFlags::DEPTH;
    }
    if block.is_some_and(|b| b.stencil.is_some()) {
        aspects |= ImageAspectFlags::STENCIL;
    }
    if aspects.is_empty() {
        ImageAspectFlags::COLOR
    } else {
        aspects
    }
}

fn image_error(call: &'static str, refusal: ImageRefusal) -> Error {
    match refusal {
        ImageRefusal::NotAFormat => invalid(call, Problem::Texture(TextureProblem::NotAFormat)),
        ImageRefusal::TextureBuffer => Error::NotImplemented(RuntimeGap::Textures),
        ImageRefusal::Unsupported(unsupported) => Error::Unsupported(unsupported),
    }
}

/// `plan`, the image of a texture of `layout` and Metal `usage` (with a depth twin where `twin`),
/// with the storage usage a texture without `ShaderWrite` or `ShaderAtomic` lacks, and the
/// sampled usage one without `ShaderRead` lacks, each where `take` takes it: the candidate is the
/// image the texture with the usage would have, offered (`take(plan, candidate)`, which answers
/// the plan to go on with) only where it keeps the image's format and twin and has every usage
/// `plan` has. Plain Metal writes and reads such a texture (`Device::with_undeclared_usage`).
fn undeclared_usage(
    mut plan: ImagePlan,
    layout: &TextureLayout,
    usage: TextureUsage,
    twin: bool,
    facts: &ImageFacts<'_>,
    mut take: impl FnMut(ImagePlan, ImagePlan) -> Result<ImagePlan, Error>,
) -> Result<ImagePlan, Error> {
    let missing = [
        (
            TextureUsage::SHADER_WRITE,
            TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC,
            ImageUsageFlags::STORAGE,
        ),
        (
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ,
            ImageUsageFlags::SAMPLED,
        ),
    ];
    for (metal, declared, vk) in missing {
        if usage.intersects(declared) || plan.usage.contains(vk) {
            continue;
        }
        // A texture over memory it is given takes the usage too, of a padded format among them.
        let Ok((candidate, candidate_twin)) = plan_images(layout, usage | metal, false, facts)
        else {
            continue;
        };
        if candidate.format != plan.format
            || candidate_twin.is_some() != twin
            || !candidate.usage.contains(vk | plan.usage)
        {
            continue;
        }
        plan = take(plan, candidate)?;
    }
    Ok(plan)
}

fn view_error(call: &'static str, refusal: ViewRefusal) -> Error {
    match refusal {
        ViewRefusal::Invalid(problem) => invalid(call, Problem::TextureView(problem)),
        ViewRefusal::Nil(problem) => nil(call, Problem::TextureView(problem)),
        ViewRefusal::Unsupported(unsupported) => Error::Unsupported(unsupported),
    }
}

fn copy_error(call: &'static str, refusal: CopyRefusal) -> Error {
    match refusal {
        CopyRefusal::Invalid(problem) => invalid(call, Problem::Copy(problem)),
        CopyRefusal::Unsupported(unsupported) => Error::Unsupported(unsupported),
    }
}

/// What the runtime makes its own images for ([`Device::new_planned_texture`]): an encoder's
/// command buffer, which holds each until it completes; the device, whose caller holds each as
/// long as a command may use it; or a maker that lends again images made before, to a later step
/// of a copy or a later copy of a queue (`texture/copy_images.rs`).
pub(super) trait ImageMaker {
    /// The device that makes them.
    fn device(&self) -> &Device;
    /// A texture of `descriptor` and `plan` it made before and lends again, held as
    /// [`hold`](Self::hold) holds one; `None` where it lends none, and one is made.
    fn lend(&self, descriptor: &TextureDescriptor, plan: &ImagePlan) -> Option<Texture> {
        let _ = (descriptor, plan);
        None
    }
    /// Hold `texture`, one of them, made of `descriptor` and `plan`, as the maker holds them.
    fn hold(&self, texture: Texture, descriptor: &TextureDescriptor, plan: &ImagePlan);
}

impl ImageMaker for CommandBuffer {
    fn device(&self) -> &Device {
        CommandBuffer::device(self)
    }

    fn hold(&self, texture: Texture, _: &TextureDescriptor, _: &ImagePlan) {
        self.hold_own_texture(texture);
    }
}

impl ImageMaker for Device {
    fn device(&self) -> &Device {
        self
    }

    /// The caller holds it.
    fn hold(&self, _: Texture, _: &TextureDescriptor, _: &ImagePlan) {}
}

impl Device {
    /// `newTextureWithDescriptor:`: a texture of `descriptor`, its contents undefined.
    ///
    /// A texture buffer's texels are a new buffer's, of the descriptor's options and as long as
    /// its row of texels, zero at first.
    ///
    /// A `Memoryless` texture is a transient render attachment and nothing else (the module
    /// docs).
    ///
    /// Refused: a descriptor Metal stops (`Invalid` with a [`TextureProblem`]); a
    /// format, size or sample count the device lacks (`Unsupported`, naming the format or limit;
    /// for a memoryless texture, a format the device cannot render to); what Metal makes and the
    /// runtime does not build yet, a texture buffer of more than one level
    /// (`NotImplemented(TextureBufferLevels)`) and a swizzle that writes under
    /// `write_swizzle_enabled` (`NotImplemented(WriteSwizzle)`), as a heap's creator refuses them
    /// too.
    ///
    /// The descriptor's `placement` gives the texture's resource ID (`resource_ids.rs`); an ID
    /// of 16381 or more is refused `Invalid` (`Problem::ResourceId`), and one no set-1 array of
    /// the device can hold `Unsupported`. [`IdPlacement::Outside`] gives it none: it answers 0,
    /// holds no slot, and counts in no plain allocation or reservation, otherwise made as here (a
    /// runtime extension, for a caller's helper texture).
    pub fn new_texture(&self, descriptor: &TextureDescriptor) -> Result<Texture, Error> {
        const CALL: &str = "new_texture";
        self.keep_room();
        let descriptor = &self.texture_options(CALL, descriptor)?;
        // An `Outside` texture takes no ID, so no ID's limit applies.
        if descriptor.placement != IdPlacement::Outside {
            self.check_texture_placement(
                CALL,
                descriptor.placement,
                descriptor.usage,
                descriptor.texture_type,
            )?;
        }
        let texture = self.texture_as(CALL, descriptor, IdUse::Program(descriptor.placement))?;
        self.take_device_sampler();
        Ok(texture)
    }

    /// [`Device::new_texture`] with `token` (a runtime extension), dropped once the runtime frees
    /// the texture's image ([`RangeToken`]): after the texture, every view of it, and the command
    /// buffers that hold any of them are gone, as Metal frees a texture of its own memory's
    /// range; a texture buffer's once its texels' buffer is free. A refused call's drops before
    /// it returns.
    pub fn new_texture_with_range_token(
        &self,
        descriptor: &TextureDescriptor,
        token: RangeToken,
    ) -> Result<Texture, Error> {
        let texture = self.new_texture(descriptor)?;
        texture.keep_range_token(token)?;
        Ok(texture)
    }

    /// A sparse texture of `descriptor` from a sparse heap of `page`-byte tiles, planned, with how
    /// the runtime holds it (`SparsePlan::backing`), or `call`'s refusal of it, as
    /// [`Heap::new_texture`](crate::Heap::new_texture) answers: a sparse resident image where the
    /// device makes one of the texture's image (`Device::check_sparse_image`: the device's sparse
    /// residency of its type and sample count, and the format's sparse properties listing its
    /// aspects for its usage); else an ordinary image whose unmapped tiles the runtime keeps at
    /// zero, where no function reads an image's own residency (the device does not enable
    /// `shaderResourceResidency`), with its tile map where its type has one. A cube or
    /// multisampled texture has none: a slot that reads residency from a tile map refuses it as it
    /// is bound (`NotImplemented(SparseTileMaps)`), and every other read is the image's, zero at an
    /// unmapped tile. Either way the runtime zeroes its unmapped texels (`Device::sparse_zeroing`),
    /// which is Metal's answer while a tile is unmapped and not after it is mapped
    /// (`docs/FAST-PATHS.md`'s known gaps).
    /// Refused otherwise: where a function may read the image's residency, as `sparse_refusal`
    /// names what the device lacks (an ordinary image would answer resident everywhere at a slot
    /// of the sparse arm and through a handle); where the runtime cannot zero the texture, as
    /// `Device::sparse_zeroing` refuses it. Each of the descriptor's refusals comes first.
    pub(crate) fn plan_sparse_texture(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        page: SparsePageSize,
    ) -> Result<SparsePlan, Error> {
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        options.cpu_cache_mode()?;
        options.hazard_tracking_mode()?;
        if storage != StorageMode::Private {
            let problem = HeapProblem::StorageMode {
                heap: StorageMode::Private,
                resource: storage,
            };
            return Err(invalid(call, Problem::Heap(problem)));
        }
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error(call))?;
        two_planes_refused(layout.pixel_format)?;
        let texture_type = layout.texture_type;
        if matches!(
            texture_type,
            TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer
        ) {
            let problem = SparseProblem::TextureType { texture_type };
            return Err(nil(call, Problem::Sparse(problem)));
        }
        let pixel_format = layout.pixel_format;
        if aspects(pixel_format).is_some_and(|planes| {
            planes.contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL)
        }) {
            let problem = SparseProblem::DualPlane { pixel_format };
            return Err(invalid(call, Problem::Sparse(problem)));
        }
        self.check_texture_placement(call, descriptor.placement, descriptor.usage, texture_type)?;
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (mut plan, twin) = plan_images(&layout, descriptor.image_usage(), false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        if twin.is_some() {
            // A depth twin would need its own memory bound alike.
            return Err(Error::NotImplemented(RuntimeGap::SparseHeaps));
        }
        // An ASTC, ETC2, EAC or 4:2:2 carrier's samples read a decoded twin, which tiles GPU
        // commands map would leave behind (`astc.rs`, `etc.rs`, `chroma_pairs.rs`).
        if super::astc::decode_of(&plan).is_some()
            || super::etc::decode_of(&plan).is_some()
            || super::chroma_pairs::decode_of(&plan).is_some()
        {
            return Err(Error::NotImplemented(RuntimeGap::DecodedHeapTexture));
        }
        let mask: vk::ImageAspectFlags = image_aspects(plan.format).into();
        let [width, height, depth] = layout.extent;
        let shape = SparseShape {
            texture_type,
            pixel_format,
            samples: u64::from(plan.samples.as_raw()),
            extent: [u64::from(width), u64::from(height), u64::from(depth)],
            levels: u64::from(plan.levels),
        };
        let tile = shape.tile(page);
        let first = super::sparse::first_mipmap_in_tail(shape.extent, shape.levels, tile);
        let zeroing = self.sparse_zeroing(&mut plan)?;
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info =
            sparse_create_info(plan.create_info(self.shared().sharing(), &view_formats, &mut list));
        let backing = match self.check_sparse_image(&info, mask) {
            Ok(()) => SparseBacking::Bound,
            Err(need) if self.reads_image_residency() => return Err(sparse_refusal(need)),
            // A texture whose type has no tile map (a cube, a multisampled one) is made without
            // one: a slot that reads residency from a map refuses it as it is bound
            // (`binding.rs`).
            Err(need) => SparseBacking::Emulated { need },
        };
        Ok(SparsePlan {
            layout,
            plan,
            shape,
            mask,
            tile,
            first,
            zeroing,
            backing,
        })
    }

    /// How the runtime zeroes the unmapped texels of a sparse texture of `plan`, which takes the
    /// usage that needs: a single-sampled image that takes transfers by copies of zeros; a
    /// multisampled one, which no copy writes, by clears in a render pass of its level and layer,
    /// where a clear writes every texel of it (`clears_to_zero`: it takes transfers, which its
    /// first clear and a move's copies need, and its format is an attachment's), made an
    /// attachment. Refused otherwise (`NotImplemented(SparseTileMaps)`: the runtime writes such an
    /// image's texels no other way), and a multisampled texture past the device's framebuffers
    /// (`Unsupported(Feature::RenderTargetSize)`, as a pass is).
    fn sparse_zeroing(&self, plan: &mut ImagePlan) -> Result<Zeroing, Error> {
        if plan.samples.as_raw() == 1 {
            return match plan.usage.contains(ImageUsageFlags::TRANSFER_DST) {
                true => Ok(Zeroing::Copies),
                false => Err(Error::NotImplemented(RuntimeGap::SparseTileMaps)),
            };
        }
        if !self.clears_to_zero(plan) {
            return Err(Error::NotImplemented(RuntimeGap::SparseTileMaps));
        }
        let [width, height, _] = plan.extent;
        super::render_pass::framebuffer_fits([width, height], 1, self.shared().max_framebuffer)
            .map_err(Error::Unsupported)?;
        plan.usage |= match image_aspects(plan.format).contains(ImageAspectFlags::COLOR) {
            true => ImageUsageFlags::COLOR_ATTACHMENT,
            false => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        };
        Ok(Zeroing::Clears)
    }

    /// A sparse texture of `descriptor` in sparse heap `heap`, held as
    /// [`plan_sparse_texture`](Self::plan_sparse_texture) answers: a sparse resident image with no memory
    /// bound, or an ordinary image, with its tile map where its type has one,
    /// its storage `Private`, its CPU cache mode the descriptor's and its hazard tracking mode the
    /// heap's, as Metal's are.
    fn new_sparse_texture(
        &self,
        heap: &Heap,
        descriptor: &TextureDescriptor,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture";
        let sparse_heap = heap
            .sparse()
            .ok_or_else(|| internal("a sparse texture of a heap that is not sparse"))?;
        let SparsePlan {
            layout,
            plan,
            shape,
            mask,
            tile,
            first,
            zeroing,
            backing,
        } = self.plan_sparse_texture(CALL, descriptor, sparse_heap.page)?;
        let cache = descriptor.resource_options.cpu_cache_mode()?;
        let (texture_type, pixel_format) = (shape.texture_type, shape.pixel_format);
        let [width, height, depth] = layout.extent;
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let allow = descriptor.allow_gpu_optimized_contents;
        // Its tile map where its type has one: every function reads its residency there.
        let tile_map = TileMapBuffer::new(self, &shape, plan.layers, tile, first)?;
        let extras = SparseExtras {
            zeros: sparse_heap.zeros()?,
            zeroing,
            tile_map,
        };
        if let SparseBacking::Emulated { need } = backing {
            let image =
                self.new_emulated_texture(heap, (plan, shape), mask, extras, (cache, allow))?;
            let storage = Storage::Image(image.shared());
            let id = IdUse::Program(descriptor.placement);
            let texture = Texture::made(self, storage, record, None, id);
            self.start_tile_map(&texture)?;
            heap.hold_texture(&texture);
            self.degrade(Degradation::EmulatedSparseTexture, || {
                let object = format!(
                    "sparse {pixel_format:?} {texture_type:?} texture of {width}x{height}x{depth}"
                );
                let why = format!(
                    "the device makes no sparse image of it ({need:?}), and its functions read \
                     residency from tile maps (it does not enable shaderResourceResidency)"
                );
                (object, why)
            });
            return Ok(texture);
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info =
            sparse_create_info(plan.create_info(self.shared().sharing(), &view_formats, &mut list));
        let limits = self.image_limits(&info, &view_formats)?;
        check_image_limits(&plan, texture_type, &limits).map_err(Error::Unsupported)?;
        // The device's first sparse image, on a driver whose first sparse resource waits for
        // every queue's work: announced before the call that waits.
        let first_sparse = !shared.sparse_image_made.swap(true, Ordering::Relaxed);
        let facts = &self.properties().driver_facts;
        if first_sparse && facts.contains(&DriverFact::FirstSparseResourceWaitsForQueues) {
            self.degrade(Degradation::FirstSparseResourceWait, || {
                let object = format!(
                    "the device's first sparse texture, {pixel_format:?} {texture_type:?} of \
                     {width}x{height}x{depth}"
                );
                let why = "the driver sets its sparse translation tables up as it makes the \
                           device's first sparse resource, with a batch on every queue it waits \
                           for, after the work submitted there before \
                           (DriverFact::FirstSparseResourceWaitsForQueues)"
                    .to_owned();
                (object, why)
            });
        }
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it, sparse residency included.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` was just made with the sparse flags, and nothing uses it.
        let sparse = unsafe { SparseImage::new(heap, raw, (shape, plan.layers), mask, extras) }?;
        let hazard = heap.hazard_tracking_mode();
        // Moved by the device's next submission; it holds no memory yet, so nothing is zeroed.
        let first = FirstMove::new(raw, plan.format, vk::ImageLayout::UNDEFINED, None);
        let image = Image {
            device: self.clone(),
            raw: Raw::new(raw),
            owner: Owner::Sparse(Box::new(sparse)),
            plan,
            options: ResourceOptions::new(StorageMode::Private, cache, hazard),
            allow_gpu_optimized_contents: allow,
            identity: Identity::new(hazard, false),
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(Some(first)),
            host_copies: false,
            linear: false,
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        let storage = Storage::Image(image.shared());
        let id = IdUse::Program(descriptor.placement);
        let texture = Texture::made(self, storage, record, None, id);
        // Dropping `texture` on failure releases the image.
        self.start_tile_map(&texture)?;
        if !shared.sparse.strict {
            self.degrade(Degradation::NonStrictResidency, || {
                let object = format!(
                    "sparse {pixel_format:?} {texture_type:?} texture of {width}x{height}x{depth}"
                );
                let why = "the device is not residencyNonResidentStrict: a read of a texel no \
                           mapped tile's device unit holds answers the device's value, not zero, \
                           and a write to one may be kept";
                (object, why.to_owned())
            });
        }
        heap.hold_texture(&texture);
        Ok(texture)
    }

    /// The image of an emulated sparse texture of `plan` and `shape` from `heap`, as
    /// [`plan_sparse_texture`](Self::plan_sparse_texture) planned it: an ordinary image in memory
    /// of its own, its unmapped tiles zeroed by the runtime (by the copies of zeros its first move
    /// notes, or a multisampled one's clear, made by the device's next submission), in the
    /// `UNDEFINED` layout until that move. Refused as an ordinary texture is.
    fn new_emulated_texture(
        &self,
        heap: &Heap,
        (plan, shape): (ImagePlan, SparseShape),
        mask: vk::ImageAspectFlags,
        extras: SparseExtras,
        (cache, allow_gpu_optimized_contents): (CpuCacheMode, bool),
    ) -> Result<Image, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan.create_info(self.shared().sharing(), &view_formats, &mut list);
        let limits = self.image_limits(&info, &view_formats)?;
        check_image_limits(&plan, shape.texture_type, &limits).map_err(Error::Unsupported)?;
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        let requirements = shared.textures.placement().place(requirements);
        let placement = match crate::memory::allocate_image(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) = unsafe {
            crate::creation::bind_image(raw_device, raw, placement.memory, placement.offset)
        } {
            lock(&shared.memory).free(raw_device, &placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        // Moved, and zeroed, by the device's next submission.
        let first = FirstMove::new(raw, plan.format, vk::ImageLayout::UNDEFINED, None);
        let first = if extras.zeroing == Zeroing::Clears {
            first.cleared()
        } else {
            let units = copy_units(plan.format, plan.pixel_format);
            let volume = shape.texture_type == TextureType::Type3D;
            let (length, copies) = zero_copies(
                plan.extent,
                plan.levels,
                plan.layers,
                volume,
                &units,
                ZERO_BYTES,
            );
            match shared.textures.first_moves.zeros(self, length) {
                Ok(zeros) => first.zeroed(zeros, &copies),
                Err(error) => {
                    // SAFETY: the image is unused.
                    unsafe { release::object(raw_device, raw) };
                    lock(&shared.memory).free(raw_device, &placement);
                    return Err(error);
                }
            }
        };
        let sparse = SparseImage::over_image(heap, (shape, plan.layers), mask, placement, extras);
        let sparse = match sparse {
            Ok(sparse) => sparse,
            Err(error) => {
                // SAFETY: the image is unused; its memory is freed with it below.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let hazard = heap.hazard_tracking_mode();
        Ok(Image {
            device: self.clone(),
            raw: Raw::new(raw),
            owner: Owner::Sparse(Box::new(sparse)),
            plan,
            options: ResourceOptions::new(StorageMode::Private, cache, hazard),
            allow_gpu_optimized_contents,
            identity: Identity::new(hazard, false),
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(Some(first)),
            host_copies: false,
            linear: false,
            staging_copies: false,
            objects: ObjectsToken::default(),
        })
    }

    /// Give sparse texture `texture`, just made, its tile map's bytes with nothing mapped, where
    /// its type has a tile map: staged for the device's next submission, which copies them in
    /// before any of its commands (`Buffer::fill_first_from`), so the making waits for no GPU
    /// work. Its image's move (and an emulated one's zeros) the image noted as it was made.
    fn start_tile_map(&self, texture: &Texture) -> Result<(), Error> {
        let Some(image) = texture.sparse_image() else {
            return Err(internal("a sparse texture without its sparse image"));
        };
        let Some(tile_map) = image.tile_map() else {
            return Ok(());
        };
        let bytes = image.tile_map_bytes(&std::collections::HashSet::new());
        let length = bytes.len() as u64;
        let staging = self.own_buffer(length, ResourceOptions::shared())?;
        let contents = staging
            .contents()
            .ok_or_else(|| internal("a Shared staging buffer without host bytes"))?;
        // SAFETY: the staging buffer holds `length` host-coherent bytes at `contents`, which no
        // command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        tile_map.buffer.fill_first_from(staging, length)
    }

    /// `plan`, the image of a texture of `layout` and Metal `usage` (with a depth twin where
    /// `twin`), with the storage usage a texture without `ShaderWrite` or `ShaderAtomic` lacks,
    /// and the sampled usage one without `ShaderRead` lacks, each where it costs the image
    /// nothing (`if_free`: the same memory size and alignment): plain Metal writes and reads such
    /// a texture (an M4 under macOS 27.0.1; its validation stops them), and a valid program's
    /// texture keeps its compression. The candidate is the image the texture with the usage would
    /// have, taken only where it keeps the image's format and twin; elsewhere a binding that needs
    /// the usage is the runtime's gap (`Feature::UndeclaredTextureUsage`, `binding_view`).
    fn with_undeclared_usage(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        usage: TextureUsage,
        twin: bool,
        facts: &ImageFacts<'_>,
    ) -> Result<ImagePlan, Error> {
        undeclared_usage(plan, layout, usage, twin, facts, |plan, candidate| {
            self.if_free(plan, candidate, layout.texture_type)
        })
    }

    /// A texture of `descriptor` only the runtime uses (a check image): as
    /// [`new_texture`](Self::new_texture) makes it, with an ID no program reads.
    pub(super) fn new_runtime_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<Texture, Error> {
        self.texture_as("new_texture", descriptor, IdUse::Runtime)
    }

    fn texture_as(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        id_use: IdUse,
    ) -> Result<Texture, Error> {
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        let hazard = options.hazard_tracking_mode()?;
        options.cpu_cache_mode()?;
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error(call))?;
        // The device makes no framebuffer-only texture: plain Metal asserts "framebufferOnly ==
        // NO".
        if descriptor.framebuffer_only {
            return Err(invalid(
                call,
                Problem::Texture(TextureProblem::FramebufferOnly),
            ));
        }
        // A memoryless texture of any type is made, as plain Metal makes and renders through one.
        let memoryless = storage == StorageMode::Memoryless;
        // But for a two-plane extended-range cube or cube array, which plain Metal makes
        // memoryless: the runtime's two-plane attachments are built and measured as 2D ones alone.
        if memoryless && memoryless_two_plane_cube(layout.pixel_format, layout.texture_type) {
            return Err(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(layout.pixel_format),
            ));
        }
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        if layout.texture_type == TextureType::TextureBuffer {
            return self.new_texture_buffer(descriptor, &layout, id_use, TexelsHome::Buffer);
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let mut usage = descriptor.own_image_usage();
        let (mut plan, mut twin) = plan_images(&layout, usage, memoryless, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // An image whose copies would rewrite texels Metal keeps is made uncompressed.
        if let Some(exact) = self.copy_exact_usage(descriptor, &layout, plan.format, usage) {
            usage = exact;
            (plan, twin) = plan_images(&layout, usage, memoryless, &facts)
                .map_err(|refusal| image_error(call, refusal))?;
        }
        let plan = match memoryless {
            true => plan,
            false => self.with_undeclared_usage(plan, &layout, usage, twin.is_some(), &facts)?,
        };
        let plan = self.with_free_usage(plan, layout.texture_type)?;
        let mut image = self.new_texture_image(plan, &layout, descriptor, hazard)?;
        // A multisampled depth the device renders faster out of `GENERAL` rests in the attachment
        // layout between the passes that hold it (`texture/held_depth.rs`).
        let private = storage == StorageMode::Private;
        if private && twin.is_none() && held_depth::eligible(self, &image.plan) {
            let raw = image.raw.get();
            let held = held_depth::HeldImage::of_plan(image.identity.key, raw, &image.plan);
            self.shared().textures.held.insert(held);
        }
        if let Some(twin) = twin {
            let twin = self.new_texture_image(twin, &layout, descriptor, hazard)?;
            image.twin = Some(Box::new(twin));
            self.degrade_depth_twin(&layout, &facts, true);
        } else if image.plan.pixel_format == PixelFormat::Depth32Float
            && image.plan.format == Format::R32_SFLOAT
        {
            // A multisampled `Depth32Float` texture held in colour, with no twin beside it
            // (`MultisampleDepthTwinPolicy::Exact`).
            self.degrade_depth_twin(&layout, &facts, false);
        }
        let image = match memoryless {
            true => {
                self.degrade_block_decode(&layout);
                image
            }
            false => self.with_decoded_twin(image, &layout, descriptor, true)?,
        };
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let storage = Storage::Image(image.shared());
        let texture = Texture::made(self, storage, record, None, id_use);
        texture.start_decoded_twin()?;
        Ok(texture)
    }

    /// A 2D texture or array of `descriptor` (checked as `layout`) over `texels`, for Metal's
    /// `call`: a linear image aliasing the texels in the buffer's memory where the device makes
    /// one that lies on Metal's layout (`alias_image`), else an image of the descriptor's usage,
    /// tracked as the buffer is, which the encoders copy the texels into and back (`shadow.rs`).
    /// Refused as `new_texture` refuses the image, and (`NotImplemented`) where the device
    /// cannot copy a shadowed texture's format. Its resource ID is as `id_use` says (the runtime's
    /// own for a texture buffer's attachment twin, `Texture::attachment_twin`).
    pub(super) fn new_texture_over_buffer(
        &self,
        call: &'static str,
        texels: Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
    ) -> Result<Texture, Error> {
        let record = ViewRecord::of_descriptor(layout, descriptor);
        let storage = self.storage_over_buffer(call, Arc::new(texels), descriptor, layout)?;
        Ok(Texture::made(self, storage, record, None, id_use))
    }

    /// The storage of a texture of `descriptor` (checked as `layout`) over `texels`, as
    /// [`new_texture_over_buffer`](Self::new_texture_over_buffer) makes it: an aliasing image,
    /// else a shadow (announced once). Refused as it is.
    fn storage_over_buffer(
        &self,
        call: &'static str,
        texels: Arc<Texels>,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
    ) -> Result<Storage, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        // The buffer's rows lay the texels out, whatever `allowGPUOptimizedContents` says: the
        // image is planned for the texture's own usage.
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // The usage the texture lacks, as for a texture of its own: plain Metal writes a texture
        // over a buffer made without `ShaderWrite` and reads one made without `ShaderRead`. A
        // linear image aliasing the buffer's memory takes it where one with it
        // still lies on Metal's layout there: a linear image is never compressed, and its memory is
        // the buffer's, so the usage costs it nothing. Else the image without it, and else the
        // shadow, with it where that costs the shadow nothing.
        let wider = undeclared_usage(plan.clone(), layout, usage, false, &facts, |_, wider| {
            Ok(wider)
        })?;
        // Whether an image may hold the texels: none where their rows (or slices) lie with the
        // buffer's bytes between them (`linear_plan::holding`), and else why.
        let holds = holding(&texels.layout, usage);
        // Over bytes detached from their pages (`detach.rs`) no image aliases them: the texture
        // is their shadow, which presents a stand-in until they lie in pages again.
        let detached = texels.buffer.detached();
        // Nor over the caller's host pages (a no-copy buffer's, or a no-copy heap's buffer's,
        // imported by their pointer or as dma-bufs): an image bound to imported memory starts
        // `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), and its first move, a layout
        // transition or an acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT`, "may perform read and write
        // accesses on all memory bound to the image subresource range", whatever the texture's
        // usage. A CPU write to the bytes then could be undone, where Metal keeps every one, and
        // Metal's contract leaves no moment the CPU may not write them. The shadow's copies
        // address its texels alone, through the buffer.
        let host = texels.buffer.caller_memory() == Some(CallerMemory::HostPages);
        // Nor over fresh pages the runtime imported for a long buffer of its own
        // (`fresh_pages.rs`), for the same reason.
        let fresh = texels.buffer.in_fresh_pages();
        // The plans a linear image aliasing the bytes is tried with, the first the device makes:
        // with the feedback loop usage a draw takes that reads the texture its own pass renders
        // through a snapshot (`feedback_candidate`, `texture/attached_reads.rs`), which costs a
        // linear image nothing as the usage the texture lacks does; then the usage the texture
        // lacks; then the texture's own.
        let feedback = feedback_candidate(&wider, self);
        let mut plans: Vec<&ImagePlan> = feedback.iter().collect();
        if wider != plan {
            plans.push(&wider);
        }
        plans.push(&plan);
        // An aliasing image, counted where it took the feedback loop usage.
        let alias = |image: Arc<Image>, texels: Arc<Texels>| {
            let usage = image.plan().usage;
            if usage.contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT) {
                self.shared().counts.feedback_loop_alias_image();
            }
            Ok(Storage::Alias { image, texels })
        };
        // Where no image aliases the bytes, the reason the plan's attempt gives is the notice's.
        let why = if detached {
            "the buffer's bytes were detached from their pages (detach_backing) as the texture \
             was made"
                .to_owned()
        } else if host {
            "the buffer's bytes are the caller's host pages, which an image could alias only from \
             UNDEFINED, whose first move Vulkan lets read and write every byte it is bound to, \
             where Metal keeps every CPU write"
                .to_owned()
        } else if fresh {
            self.shared().counts.shadow_over_fresh_pages();
            "the buffer's bytes lie in fresh host pages imported for it, which an image could alias \
             only from UNDEFINED, whose first move Vulkan lets read and write every byte it is \
             bound to, where Metal keeps every CPU write"
                .to_owned()
        } else {
            let over = (&*texels, &holds);
            for &wider in &plans[..plans.len() - 1] {
                let made = self.alias_image(wider.clone(), descriptor, over, &facts)?;
                if let Ok(image) = made {
                    return alias(image, texels);
                }
            }
            match self.alias_image(plan.clone(), descriptor, over, &facts)? {
                Ok(image) => return alias(image, texels),
                Err(why) => why,
            }
        };
        let plan = self.with_undeclared_usage(plan, layout, usage, false, &facts)?;
        if !plan
            .usage
            .contains(ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST)
        {
            return Err(Error::NotImplemented(RuntimeGap::Textures));
        }
        // Where the device copies a shadow back faster by a dispatch, the image takes the
        // integer view the dispatch reads (`shadow_copy_back.rs`).
        let (plan, word) = match shadow_copy_back::dispatches(self) {
            true => with_word_view(plan, &facts, shadow_copy_back::dispatches_in(self)),
            false => (plan, None),
        };
        // Without it, an 8-bit unsigned normalized image takes the view of its own format its
        // merges read through (`shadow_merge.rs`).
        let (plan, unorm) = match word {
            Some(_) => (plan, None),
            None => with_unorm_view(plan, &facts),
        };
        let hazard = texels.options().hazard_tracking_mode()?;
        // A depth or stencil texture over a surface's plane copies the aspect the plane holds
        // alone (`copy_plan::surface_depth_unit`); its image's other aspect stays the zeros it
        // was made with, as Metal reads the stencil of `Depth32Float_Stencil8` there.
        let every = texture_aspects(&plan);
        let held = surface_depth_unit(layout.pixel_format).map(|unit| unit.aspect.into());
        let aspects = match held {
            Some(held) if every.contains(held) => held,
            _ => every,
        };
        // Every other shadow's copy in writes each of its texels before any command reads it
        // (`shadow.rs`), so it starts as its memory lies: zeros there would never be read, and
        // writing them took most of the making (the host's zeros copied into a 1307x512 image
        // took 0.9 ms of 1 ms on ANV).
        let image = match aspects == every {
            true => {
                let identity = Identity::new(hazard, false);
                let owner = Device::runtime_owner;
                self.new_image_as_it_lies(plan, layout, descriptor, identity, owner)?
            }
            false => self.new_image(plan, layout, descriptor, hazard)?,
        };
        // A view the device does not make leaves the copies back copies. The copies hold it as
        // long as the image, which keeps it through a detach (`Image::own_view`).
        let words = word.and_then(|(format, texel, stores)| {
            let usage = match stores {
                true => ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
                false => ImageUsageFlags::SAMPLED,
            };
            let view = image.own_view(&VkView {
                view_type: ImageViewType::TYPE_2D,
                format,
                components: ComponentMapping::IDENTITY,
                aspect: ImageAspectFlags::COLOR,
                base_level: 0,
                level_count: 1,
                base_layer: 0,
                layer_count: 1,
                usage,
            });
            view.ok().map(|view| WordView {
                view,
                texel,
                stores,
            })
        });
        let unorm = unorm.and_then(|read| {
            let view = image.own_view(&VkView {
                view_type: ImageViewType::TYPE_2D,
                format: image.plan().format,
                components: ComponentMapping::IDENTITY,
                aspect: ImageAspectFlags::COLOR,
                base_level: 0,
                level_count: 1,
                base_layer: 0,
                layer_count: 1,
                usage: ImageUsageFlags::SAMPLED,
            });
            view.ok().map(|view| UnormView { view, read })
        });
        let views = (words, unorm);
        let copies = texels.shadow_copies(image.raw.get(), image.identity, aspects, views)?;
        // Its copies keep its own image, which it presents again once the bytes lie in pages.
        self.stand_in_while_detached(&texels.buffer, &image)?;
        let (offset, bytes_per_row) = (texels.layout.offset, texels.layout.row_pitch);
        self.degrade(Degradation::ShadowedTexture, || {
            let format = descriptor.pixel_format;
            let [width, height, _] = layout.extent;
            let object = format!(
                "a 2D {format:?} texture of {width}x{height} over a buffer at offset {offset} with \
                 rows of {bytes_per_row} bytes"
            );
            (object, why)
        });
        self.shared().counts.shadowed_texture_made(aspects != every);
        Ok(Storage::Shadow {
            image: image.shared(),
            texels,
            copies: Arc::new(copies),
        })
    }

    /// A tracked `Private` texture of `descriptor` whose image `plan` states, planned by the
    /// caller rather than from the descriptor: an image only the runtime's commands use, such as
    /// the combined depth/stencil image a pass of separate depth and stencil textures renders
    /// into (`depth_stencil_scratch.rs`), made for `maker`, which holds it as its
    /// [`ImageMaker`] says, or the one of the same descriptor and plan `maker` lends again. Its
    /// image starts in `UNDEFINED`, which the device's next submission leaves for `GENERAL`
    /// (`texture/first_move.rs`): making it waits for no GPU work. Refused as `new_texture`
    /// refuses the image.
    pub(super) fn new_planned_texture(
        maker: &dyn ImageMaker,
        descriptor: &TextureDescriptor,
        plan: ImagePlan,
    ) -> Result<Texture, Error> {
        if let Some(texture) = maker.lend(descriptor, &plan) {
            return Ok(texture);
        }
        let device = maker.device();
        let layout = descriptor
            .check(device.properties().texture_rules())
            .map_err(|refused| refused.error("new_texture"))?;
        if descriptor.resource_options.storage_mode()? != StorageMode::Private {
            return Err(internal("an encoder's own image that is not `Private`"));
        }
        let key = plan.clone();
        let image = device.new_image(plan, &layout, descriptor, HazardTrackingMode::Tracked)?;
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let storage = Storage::Image(image.shared());
        let texture = Texture::made(device, storage, record, None, IdUse::Runtime);
        maker.hold(texture.clone(), descriptor, &key);
        Ok(texture)
    }

    /// The memory requirements of an image of `plan` (of Metal's `texture_type`), from an image
    /// made for the query and destroyed at once, with no memory bound; `None` where the device
    /// makes no such image (`check_image_limits`).
    pub(super) fn image_requirements(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
    ) -> Result<Option<vk::MemoryRequirements>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan.create_info(self.shared().sharing(), &view_formats, &mut list);
        let limits = self.image_limits(&info, &view_formats)?;
        if check_image_limits(plan, texture_type, &limits).is_err() {
            return Ok(None);
        }
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device, which nothing else uses: it is destroyed
        // right after the query, with no memory bound.
        let requirements = unsafe {
            let requirements = raw_device.get_image_memory_requirements(raw);
            release::object(raw_device, raw);
            requirements
        };
        Ok(Some(requirements))
    }

    /// The device's limits for an image of `info`, viewed as `view_formats`: all 0 where it
    /// makes no such image.
    pub(super) fn image_limits(
        &self,
        info: &vk::ImageCreateInfo<'_>,
        view_formats: &[vk::Format],
    ) -> Result<vk::ImageFormatProperties, Error> {
        let shared = self.shared();
        let mut query_list = vk::ImageFormatListCreateInfo::default().view_formats(view_formats);
        let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(info.format)
            .ty(info.image_type)
            .tiling(info.tiling)
            .usage(info.usage)
            .flags(info.flags);
        if !view_formats.is_empty() {
            query = query.push_next(&mut query_list);
        }
        let mut properties = vk::ImageFormatProperties2::default();
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later; `query` and its
        // chain live for the call.
        let queried = unsafe {
            shared
                .instance
                .raw
                .get_physical_device_image_format_properties2(
                    shared.physical,
                    &query,
                    &mut properties,
                )
        };
        match queried {
            Ok(()) => Ok(properties.image_format_properties),
            // The combination is unsupported: every limit is 0, as the query reports it.
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(vk::ImageFormatProperties::default()),
            Err(error) => Err(vulkan(error)),
        }
    }

    /// The device's limits for an image of `info` (with `view_formats`) made for host
    /// allocations (`VK_EXT_external_memory_host`), where it imports host memory for such an
    /// image without a dedicated allocation; why not where it does not, or makes no such image
    /// (The refusal named).
    pub(super) fn host_image_limits(
        &self,
        info: &vk::ImageCreateInfo<'_>,
        view_formats: &[vk::Format],
    ) -> Result<Result<vk::ImageFormatProperties, LinearRefusal>, Error> {
        let shared = self.shared();
        let mut query_list = vk::ImageFormatListCreateInfo::default().view_formats(view_formats);
        let mut host = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(info.format)
            .ty(info.image_type)
            .tiling(info.tiling)
            .usage(info.usage)
            .flags(info.flags)
            .push_next(&mut host);
        if !view_formats.is_empty() {
            query = query.push_next(&mut query_list);
        }
        let mut external = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external);
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later; `query` and its
        // chain, and `properties` and its chain, live for the call.
        let queried = unsafe {
            shared
                .instance
                .raw
                .get_physical_device_image_format_properties2(
                    shared.physical,
                    &query,
                    &mut properties,
                )
        };
        let limits = properties.image_format_properties;
        let features = external.external_memory_properties.external_memory_features;
        match queried {
            Ok(()) => {
                let importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                    && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
                match importable {
                    true => Ok(Ok(limits)),
                    false => Ok(Err(LinearRefusal::NotImportable(features))),
                }
            }
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
                Ok(Err(LinearRefusal::Need(LinearNeed::Image)))
            }
            Err(error) => Err(vulkan(error)),
        }
    }

    /// The linear image of `plan` aliasing `texels` in their buffer's memory (the runtime's own),
    /// in the `GENERAL` layout and holding the buffer's bytes, where the device makes one that
    /// lies exactly on Metal's layout there (`linear_plan::linear_lack` and `aliases`); why not
    /// where it does not, and for a buffer over the caller's `VkBuffer`, whose memory's handle type
    /// an image bound to it would have to be made for (a texture over the caller's host pages is
    /// always a shadow: `storage_over_buffer`). It starts `PREINITIALIZED`, which its move to
    /// `GENERAL` keeps. The image's hazard tracking is the buffer's. `holds`: whether an image may
    /// hold the texels (`linear_plan::holding`), or why not.
    fn alias_image(
        &self,
        plan: ImagePlan,
        descriptor: &TextureDescriptor,
        (texels, holds): (&Texels, &Result<(), String>),
        facts: &ImageFacts<'_>,
    ) -> Result<Aliased, Error> {
        match texels.buffer.caller_memory() {
            None => {}
            Some(CallerMemory::HostPages) => {
                return Err(internal("an aliasing image over the caller's host pages"));
            }
            Some(CallerMemory::VkBuffer) => {
                return Ok(Err(
                    "the buffer is over the caller's VkBuffer, whose memory backs no \
                               image of the runtime's"
                        .to_owned(),
                ))
            }
        }
        if let Err(why) = holds {
            return Ok(Err(why.clone()));
        }
        let (memory, buffer_offset, type_index) = texels.buffer.memory();
        let memory_offset = buffer_offset + texels.layout.offset;
        let made = if let Some(LinearLack { format, features }) = linear_lack(&plan, facts) {
            Err(LinearRefusal::Need(LinearNeed::Features { format, features }).describe(&plan))
        } else {
            self.linear_image(&plan)?
                .map_err(|refusal| refusal.describe(&plan))
        };
        let LinearMade {
            raw,
            requirements,
            report,
            initial,
        } = match made {
            Ok(made) => made,
            Err(why) => {
                return Ok(Err(format!(
                    "the device makes no linear image of the texture: {why}"
                )));
            }
        };
        let raw_device = self.shared().vk();
        let room = texels.buffer.padded_length() - texels.layout.offset;
        let image = LinearImage {
            alignment: requirements.alignment,
            size: requirements.size,
            offset: report.offset,
            subresource: report.size,
            row_pitch: report.row_pitch,
            array_pitch: report.array_pitch,
        };
        let why = if !aliases(&texels.layout, memory_offset, room, &image) {
            Some(format!(
                "the device's linear image of the texture lays its texels out otherwise than \
                 Metal at the texels' offset and row pitch, or holds other bytes than its texels \
                 (rows {} bytes apart, a subresource of {} bytes, aligned to {})",
                report.row_pitch, report.size, requirements.alignment
            ))
        } else if requirements.memory_type_bits & (1 << type_index) == 0 {
            Some(
                "the device's linear image of the texture takes no memory of the buffer's type"
                    .to_owned(),
            )
        } else {
            None
        };
        if let Some(why) = why {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(Err(why));
        }
        // SAFETY: the image is live and unbound; the memory is the buffer's, which outlives the
        // texture holding it, in a type the image takes, at an offset `aliases` checked against
        // the image's alignment and size.
        if let Err(error) =
            unsafe { crate::creation::bind_image(raw_device, raw, memory, memory_offset) }
        {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        let image = Image {
            device: self.clone(),
            raw: Raw::new(raw),
            owner: Owner::Buffer {
                memory: Mutex::new(texels.buffer.memory_hold()),
            },
            plan,
            options: texels.options(),
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            // The buffer's bytes, tracked as the texture is.
            identity: Identity {
                tracked: texels.tracked(),
                ..texels.buffer.identity()
            },
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: true,
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        // Leaving `PREINITIALIZED` keeps the bytes the image holds: the buffer's.
        let first = FirstMove::new(raw, image.plan().format, initial, None);
        *lock(&image.first_move) = Some(first);
        Ok(Ok(image.shared()))
    }

    /// The image of a texture over dma-buf pages (over a buffer, or placed in a heap), `made` of
    /// `plan` (`dma_buf.rs`), with its `owner`, its options and whether it allows GPU-optimized
    /// contents, and the `identity` of its bytes: a `DRM_FORMAT_MOD_LINEAR` image, whose rows are a
    /// linear image's (a write of its bytes is a write of its texels), its first move acquiring
    /// it from `VK_QUEUE_FAMILY_FOREIGN_EXT` into `GENERAL`, keeping the bytes, and holding
    /// `pages`, the host pages its bytes lie in.
    #[cfg(unix)]
    fn dma_buf_texture_image(
        &self,
        made: crate::dma_buf::DmaBufImage,
        plan: ImagePlan,
        owner: Owner,
        (options, allow_gpu_optimized_contents): (ResourceOptions, bool),
        identity: Identity,
        pages: &Arc<HostPages>,
    ) -> Image {
        let crate::dma_buf::DmaBufImage { raw, import } = made;
        let image = Image {
            device: self.clone(),
            raw: Raw::imported(raw, import),
            owner,
            plan,
            options,
            allow_gpu_optimized_contents,
            identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: true,
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        let undefined = vk::ImageLayout::UNDEFINED;
        let first = FirstMove::new(raw, image.plan().format, undefined, pages_hold(pages))
            .acquired(self.foreign_acquirer());
        *lock(&image.first_move) = Some(first);
        image
    }

    /// A linear image of `plan` (of Metal's `texture_type`), unbound, with its memory
    /// requirements and its first subresource's layout, to start `PREINITIALIZED` (it keeps its
    /// memory's bytes); why not where the device makes no such image (the format query, or
    /// `check_image_limits`; the refusal named).
    fn linear_image(&self, plan: &ImagePlan) -> Result<Result<LinearMade, LinearRefusal>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan
            .create_info(self.shared().sharing(), &view_formats, &mut list)
            .tiling(vk::ImageTiling::LINEAR)
            .initial_layout(vk::ImageLayout::PREINITIALIZED);
        let limits = self.image_limits(&info, &view_formats)?;
        if let Err(need) = check_linear_limits(plan, &limits) {
            return Ok(Err(LinearRefusal::Need(need)));
        }
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live linear image of this device, and its first subresource has
        // the color aspect (every format a buffer backs is a color format).
        let (requirements, report) = unsafe {
            (
                raw_device.get_image_memory_requirements(raw),
                raw_device.get_image_subresource_layout(raw, first_color_subresource()),
            )
        };
        Ok(Ok(LinearMade {
            raw,
            requirements,
            report,
            initial: vk::ImageLayout::PREINITIALIZED,
        }))
    }

    /// The images of `heap`'s textures bound to its bytes (a no-copy heap's), made for a rebind's
    /// new pages, which it imported as dma-bufs (`placement`), the bytes at `pointer`: for each,
    /// one in the driver's layout at its offset (`texture/heap.rs`). Made before the rebind
    /// settles, with no lock of the runtime's held: the source is called once for each. Nothing
    /// for pages imported by their pointer. Refused (`NotImplemented(RebindTextures)`), what was
    /// made released, where the device makes no such image of one (`dma_buf_image`'s reasons).
    #[cfg(unix)]
    pub(super) fn dma_buf_images_again(
        &self,
        heap: &Heap,
        placement: &Placement,
        pointer: NonNull<u8>,
    ) -> Result<Remade, Error> {
        let mut remade = Remade::default();
        let Some(imports) = placement.dma_buf_pages() else {
            return Ok(remade);
        };
        let type_index = placement.type_index();
        self.dma_buf_heap_images_again(heap, imports, type_index, pointer, &mut remade)?;
        Ok(remade)
    }

    /// No pages are imported as dma-bufs on this system: nothing is made.
    #[cfg(not(unix))]
    pub(super) fn dma_buf_images_again(
        &self,
        _heap: &Heap,
        _placement: &Placement,
        _pointer: NonNull<u8>,
    ) -> Result<Remade, Error> {
        Ok(Remade::default())
    }

    /// The image of `plan` for a texture of `layout` made with `descriptor` (and its hazard
    /// tracking mode `hazard`), checked against the device's limits for it, placed, and in the
    /// `GENERAL` layout.
    fn new_image(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        hazard: HazardTrackingMode,
    ) -> Result<Image, Error> {
        let identity = Identity::new(hazard, false);
        self.new_image_owned(plan, layout, descriptor, identity, Device::runtime_owner)
    }

    /// [`new_image`](Self::new_image)'s image for a texture of its own memory the program made
    /// (`newTextureWithDescriptor:`, `newSharedTextureWithDescriptor:`): zero in every texel, as
    /// the texels of Metal's new textures read zero in every storage mode but `Memoryless`, whose
    /// texture holds nothing (an Apple M4 under macOS 27.0.1: `Private` ones
    /// Metal does not compress read zero, each time, also right after a texture of the same
    /// shape was written and released; those it compresses read a fixed pattern, which the
    /// runtime does not give yet). The runtime's own images keep [`new_image`](Self::new_image)'s
    /// start: an image its first command writes in full need not be zeroed first.
    fn new_texture_image(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        hazard: HazardTrackingMode,
    ) -> Result<Image, Error> {
        let identity = Identity::new(hazard, false);
        let held = descriptor.resource_options.storage_mode()? == StorageMode::Memoryless;
        self.new_image_starting(
            plan,
            layout,
            descriptor,
            identity,
            Start::held_or_zero(held),
            Device::runtime_owner,
        )
    }

    /// The owner of an image the runtime placed in memory of its own at `placement`.
    fn runtime_owner(placement: Placement, _: vk::Image) -> Owner {
        Owner::Runtime {
            placement,
            shareable: false,
            purgeable: Mutex::new(PurgeableState::NonVolatile),
        }
    }

    /// A `VkImage` of `plan` for a texture of `texture_type`, checked against the device's limits
    /// for it, placed in memory of `class` of its own and bound there, in the `UNDEFINED` layout,
    /// shared as `sharing` says (the device's, or the staging queue's).
    fn placed_image(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
        class: MemoryClass,
        sharing: Sharing<'_>,
    ) -> Result<(vk::Image, Placement), Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan.create_info(sharing, &view_formats, &mut list);
        let limits = self.image_limits(&info, &view_formats)?;
        check_image_limits(plan, texture_type, &limits).map_err(Error::Unsupported)?;
        // SAFETY: the device is live and `info` (with its format list) valid and within the
        // device's limits for it.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        let mut requirements = shared.textures.placement().place(requirements);
        // Its host copies read and write through the host's mapping of its memory, which the
        // host reads slowly where it does not cache it: such memory of the class where the
        // device has it, else `Shared` memory's with unified memory
        // (`memory::host_cached_placement`). Never memory the host maps without coherence, cached
        // or not: the runtime flushes no mapped range (`memory::host_cached`). Where only such
        // memory is allowed, the placement fails and `host_placed_image` makes the image without
        // host-transfer usage, its copies through submissions, as on a device without host copies.
        let hosted = plan.usage.contains(ImageUsageFlags::HOST_TRANSFER);
        let cached = match hosted {
            true => {
                let memory = lock(&shared.memory);
                requirements.memory_type_bits &= !memory.host_incoherent();
                memory.host_cached_placement(requirements.memory_type_bits, class)
            }
            false => None,
        };
        let class = match cached {
            Some((cached, types)) => {
                requirements.memory_type_bits = types;
                cached
            }
            None => class,
        };
        let placement =
            match crate::memory::allocate_image(&shared.memory, raw_device, &requirements, class) {
                Ok(placement) => placement,
                Err(error) => {
                    // SAFETY: the image is unused.
                    unsafe { release::object(raw_device, raw) };
                    return Err(error);
                }
            };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) = unsafe {
            crate::creation::bind_image(raw_device, raw, placement.memory, placement.offset)
        } {
            lock(&shared.memory).free(raw_device, &placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        if hosted && cached.is_none() {
            shared.counts.uncached_host_image();
        }
        Ok((raw, placement))
    }

    /// [`placed_image`](Self::placed_image)'s image, or a parked image of the same making
    /// (`recycle.rs`), noted as one the recycler may park as it goes.
    fn recycled_image(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
        class: MemoryClass,
        sharing: Sharing<'_>,
    ) -> Result<(vk::Image, Placement), Error> {
        let shared = self.shared();
        let making = ImageMaking {
            plan: plan.clone(),
            texture_type,
            class,
            families: sharing.families().to_vec(),
        };
        let (taken, gone) = lock(&shared.textures.recycled).take(&making, Instant::now());
        self.let_go_parked(gone);
        if let Some(taken) = taken {
            shared.counts.churn.taken(Recycled::Image);
            return Ok(taken);
        }
        let (raw, placement) = self.placed_image(plan, texture_type, class, sharing)?;
        lock(&shared.textures.recycled).made(raw, making);
        Ok((raw, placement))
    }

    /// Destroy the parked images the recycler let go of, `gone`.
    fn let_go_parked(&self, gone: recycle::Gone<vk::Image>) {
        if gone.is_empty() {
            return;
        }
        let shared = self.shared();
        shared.counts.churn.evicted(Recycled::Image, gone.len());
        recycle::destroy(shared.vk(), &mut lock(&shared.memory), gone);
    }

    /// [`placed_image`](Self::placed_image)'s image of `plan` with host-transfer usage, and that
    /// plan, where the device copies on the host (`texture/host_copy.rs`), host copies reach the
    /// plan's format on this driver ([`host_copy::reaches_format`]), the plan's format has
    /// `HOST_IMAGE_TRANSFER` at optimal tiling, and the device makes and places the image so (in
    /// host-cached memory of `class` where the device has it; only there where the image would
    /// otherwise copy on the staging queue, `staged`); `None` elsewhere, the image made as before.
    fn host_placed_image(
        &self,
        plan: &ImagePlan,
        texture_type: TextureType,
        class: MemoryClass,
        staged: bool,
    ) -> Option<(ImagePlan, vk::Image, Placement)> {
        let shared = self.shared();
        shared.host_copies.as_ref()?;
        // An image whose texel block holds more than one texel takes no host copies on a driver
        // that copies such an image out of its host-visible memory as the memory is bound, through
        // a row pitch whose own check it fails (`texture/host_copy.rs`, `driver_defects.rs`).
        if !host_copy::reaches_format(plan.format, &self.properties().driver_defects) {
            return None;
        }
        // An image that would copy on the staging queue takes host copies only from host-cached
        // memory: the host reads uncached memory slowly (`texture/staging_copy.rs`).
        if staged && !lock(&shared.memory).has_host_cached(class) {
            return None;
        }
        let facts = shared.textures.facts(self.compile_device().profile());
        if !(facts.optimal(plan.format)).contains(FormatFeatureFlags2::HOST_IMAGE_TRANSFER) {
            return None;
        }
        let mut hosted = plan.clone();
        hosted.usage |= ImageUsageFlags::HOST_TRANSFER;
        let (raw, placement) =
            (self.placed_image(&hosted, texture_type, class, shared.sharing())).ok()?;
        Some((hosted, raw, placement))
    }

    /// [`new_image`](Self::new_image)'s image, of `identity`, its owner made by `owner` of its
    /// placement and `VkImage` (a heap's unaliased texture and a shadow whose copies leave an
    /// aspect out, `texture/heap.rs`; a view's shadow, `texture/view_shadow.rs`).
    fn new_image_owned(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        identity: Identity,
        owner: impl FnOnce(Placement, vk::Image) -> Owner,
    ) -> Result<Image, Error> {
        let storage = descriptor.resource_options.storage_mode()?;
        // Metal's new `Shared` and `Managed` textures read zero (measured on an Apple M3 Max and
        // M4); `Private` ones hold what the memory held, and `Memoryless` ones hold nothing.
        let held = matches!(storage, StorageMode::Private | StorageMode::Memoryless);
        let start = Start::held_or_zero(held);
        self.new_image_starting(plan, layout, descriptor, identity, start, owner)
    }

    /// [`new_image_owned`](Self::new_image_owned)'s image holding what its memory held, whatever
    /// its storage mode: a heap texture's in memory of its own where its bytes in the heap are no
    /// layout of its texels (`texture/heap.rs`, `AliasRoute::Lossless`), as a heap's textures hold
    /// what their bytes held; and a shadow whose copy in writes every texel of every aspect before
    /// any command reads it: a texture's over a buffer (`storage_over_buffer`), and a heap
    /// texture's over the heap's bytes (`texture/heap.rs`, `new_heap_shadow` and
    /// `heap_planes_image`).
    fn new_image_as_it_lies(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        identity: Identity,
        owner: impl FnOnce(Placement, vk::Image) -> Owner,
    ) -> Result<Image, Error> {
        self.new_image_starting(plan, layout, descriptor, identity, Start::Held, owner)
    }

    /// [`new_image_owned`](Self::new_image_owned)'s image, holding what its memory held where
    /// `start` says so, else zeroed: on the host where it takes host copies, else by a clear
    /// where one writes its format ([`clears_to_zero`](Self::clears_to_zero)), else by copies of
    /// the device's zeros. (or moved alone, [`Start::Moved`].)
    fn new_image_starting(
        &self,
        plan: ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        identity: Identity,
        start: Start,
        owner: impl FnOnce(Placement, vk::Image) -> Owner,
    ) -> Result<Image, Error> {
        let held = start == Start::Held;
        let moved = start == Start::Moved;
        let storage = descriptor.resource_options.storage_mode()?;
        let class = plan.memory_class(storage, &self.properties().driver_defects);
        // A `Shared` or `Managed` texture's image with host-transfer usage where the device
        // copies on the host and makes and places it so (`texture/host_copy.rs`), else as
        // before.
        // A PVRTC texture copies through submissions: MoltenVK's host copies of its blocks take
        // Metal's CPU order, not Vulkan's rows. A `Private` texture takes no CPU
        // copy, so its image takes no host-transfer usage.
        let private = storage == StorageMode::Private;
        // A `Shared` or `Managed` texture's image the staging queue copies, where it takes no
        // host copies from host-cached memory (`texture/staging_copy.rs`).
        // Vulkan copies nothing into or out of a multisampled image, on the host or a queue
        // (`VUID-VkCopyMemoryToImageInfo-dstImage-07973`,
        // `VUID-vkCmdCopyBufferToImage-dstImage-07973`), and Metal makes no CPU copy of a
        // multisample texture: such an image takes neither path, and a clear writes its zeros.
        let one_sample = layout.samples == 1;
        let staged = !held && !private && one_sample && self.stages(&plan, layout.pixel_format);
        let hosted = match held || private || !one_sample || is_pvrtc(descriptor.pixel_format) {
            true => None,
            false => self.host_placed_image(&plan, layout.texture_type, class, staged),
        };
        let (plan, raw, placement, host_copies) = match hosted {
            Some((plan, raw, placement)) => (plan, raw, placement, true),
            None => {
                let sharing = match staged {
                    true => self.staging_sharing(),
                    false => None,
                };
                let sharing = sharing.unwrap_or_else(|| self.shared().sharing());
                let (raw, placement) =
                    self.recycled_image(&plan, layout.texture_type, class, sharing)?;
                (plan, raw, placement, false)
            }
        };
        let staging_copies = staged && !host_copies;
        // It reads no source, and is the device's fast clear where it has one; it alone writes a
        // multisampled image's samples, which no copy writes.
        // A queue of transfers alone clears nothing: it copies the zeros.
        let clear =
            !held && !moved && !host_copies && !staging_copies && self.clears_to_zero(&plan);
        let zero = if held || moved || clear {
            None
        } else if host_copies || plan.usage.contains(ImageUsageFlags::TRANSFER_DST) {
            let units = copy_units(plan.format, plan.pixel_format);
            let volume = layout.texture_type == TextureType::Type3D;
            Some(zero_copies(
                plan.extent,
                plan.levels,
                plan.layers,
                volume,
                &units,
                ZERO_BYTES,
            ))
        } else {
            None
        };
        let first = FirstMove::new(raw, plan.format, vk::ImageLayout::UNDEFINED, None);
        let aspects: vk::ImageAspectFlags = image_aspects(plan.format).into();
        let image = Image {
            device: self.clone(),
            raw: Raw::new(raw),
            owner: owner(placement, raw),
            plan,
            options: descriptor.resource_options,
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            twin: None,
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            between: OnceLock::new(),
            signed_depth: AtomicBool::new(false),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies,
            linear: false,
            staging_copies,
            objects: ObjectsToken::default(),
        };
        if !held && !moved && !clear && zero.is_none() {
            // The image cannot be written by a clear or a copy, so it cannot start as Metal's
            // does.
            return Err(Error::NotImplemented(RuntimeGap::Textures));
        }
        // Dropping `image` on failure releases everything above.
        if let (true, Some((length, copies)), Some(functions)) =
            (host_copies, &zero, self.shared().host_copies.as_ref())
        {
            let zeros = vec![0; *length as usize];
            let copies: Vec<vk::BufferImageCopy> = copies.iter().map(buffer_image_copy).collect();
            // SAFETY: the image is live, made with host-transfer usage and bound to memory of its
            // own, which nothing used yet; every copy lies within it and reads within `zeros`.
            unsafe {
                host_copy::start(
                    functions,
                    self.shared().vk().handle(),
                    raw,
                    aspects,
                    &zeros,
                    &copies,
                )
            }?;
            self.shared().counts.image_started_on_the_host();
            return Ok(image);
        }
        // Its texels are written after (`Start::Moved`). Dropping `image` on failure releases
        // everything above.
        if let (true, true, Some(functions)) =
            (moved, host_copies, self.shared().host_copies.as_ref())
        {
            // SAFETY: the image is live, made with host-transfer usage and bound to memory of its
            // own, which nothing used yet.
            unsafe { host_copy::transition(functions, self.shared().vk().handle(), raw, aspects) }?;
            self.shared().counts.image_started_on_the_host();
            return Ok(image);
        }
        if moved && staging_copies {
            // SAFETY: the staging command buffer is recording; the image is live, bound to
            // memory of its own, which nothing used yet, in `UNDEFINED`, which the move leaves.
            self.submit_staging(|raw, command_buffer, family| unsafe {
                first.record_alone(raw, command_buffer, family)
            })?;
            return Ok(image);
        }
        // Its zeros are a staging buffer's, which nothing else uses: the device's own zeros are
        // exclusive to its families. Dropping `image` on failure releases everything above.
        if let (true, Some((length, copies))) = (staging_copies, &zero) {
            let mut zeros = Staging::new(self, *length)?;
            zeros.bytes().fill(0);
            let first = first.zeroed(zeros.raw, copies);
            // SAFETY: the staging command buffer is recording; the image is live, bound to
            // memory of its own, which nothing used yet, and made with transfer usage; every
            // copy lies within it and reads within the zeros.
            self.submit_staging(|raw, command_buffer, family| unsafe {
                first.record_alone(raw, command_buffer, family)
            })?;
            return Ok(image);
        }
        // A `Private` texture of the program's (the runtime's own images are held). Dropping
        // `image` on failure releases everything above.
        let patterned = match private && !held {
            true => {
                let key = image.identity.key;
                self.fresh_patterned(&image.plan, layout, descriptor, clear, key)?
            }
            false => None,
        };
        let first = match patterned {
            Some(patterned) => first.patterned(patterned),
            None => first,
        };
        // Dropping `image` on failure releases everything above.
        let first = match zero {
            Some((length, copies)) => {
                let zeros = self.shared().textures.first_moves.zeros(self, length)?;
                first.zeroed(zeros, &copies)
            }
            None if clear => first.cleared(),
            None => first,
        };
        *lock(&image.first_move) = Some(first);
        Ok(image)
    }

    /// How a new `Private` texture of `descriptor` (checked as `layout`) held in an image of
    /// `plan` (`clear`: one a clear writes, [`Device::clears_to_zero`]) starts as Metal's: the
    /// pattern Metal's compressed texture reads over its first levels
    /// (`heap_tiled::fresh_pattern`, `pattern_levels`), written exactly by a clear to a value
    /// that converts to it, or by copies of its bytes where the image holds Metal's bytes; `None`
    /// where Metal reads zero throughout. Where neither writes it exactly (a NaN no clear writes
    /// into a multisampled image, a depth a copy may store clamped, an image holding other bytes
    /// than Metal's), the texture (`key`) reads zero, and the device announces
    /// `fresh-pattern-zeroed` where a command first reads those levels before a clear covers them
    /// (`fresh_unwritten.rs`). A render target's 32-bit float depth written by copies of the
    /// pattern is noted for the passes that load it unwritten (`FreshKind::SignedDepth`).
    fn fresh_patterned(
        &self,
        plan: &ImagePlan,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        clear: bool,
        key: ResourceKey,
    ) -> Result<Option<Patterned>, Error> {
        let pf = layout.pixel_format;
        let [width, height, _] = layout.extent.map(u64::from);
        let one_row = matches!(
            layout.texture_type,
            TextureType::Type1D | TextureType::Type1DArray
        );
        let level0 = [width, if one_row { 1 } else { height }];
        let shape = match layout.samples {
            1 => Some(level0),
            samples => heap_tiled::sample_grid(samples, level0),
        };
        let (Some(pattern), Some(shape)) = (heap_tiled::fresh_pattern(pf), shape) else {
            return Ok(None);
        };
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let colour = !is_depth_stencil(pf);
        let optimized = descriptor.allow_gpu_optimized_contents;
        let levels = heap_tiled::pattern_levels(colour, usage, optimized, shape, layout.levels);
        if levels == 0 {
            return Ok(None);
        }
        let single = layout.samples == 1 && plan.usage.contains(ImageUsageFlags::TRANSFER_DST);
        let copies = |texel: &'static [u8]| -> Result<Pattern, Error> {
            let units = copy_units(plan.format, pf);
            let volume = layout.texture_type == TextureType::Type3D;
            let (length, copies) = zero_copies(
                plan.extent,
                levels,
                plan.layers,
                volume,
                &units,
                PATTERN_BYTES,
            );
            let source = (self.shared().textures.first_moves).pattern(self, texel, length)?;
            Ok(Patterned::copies(source, &copies))
        };
        let written = match pattern {
            FreshPattern::Color { color, .. } if clear && color.is_some() => color
                .and_then(|color| {
                    clear_value::held_color(pf, plan.format, color, ClearRoute::ThroughHalf)
                })
                .map(Pattern::Color),
            FreshPattern::Color { texel, .. }
                if single && heap_tiled::holds_metal_texels(pf, plan.format, texel.len()) =>
            {
                Some(copies(texel)?)
            }
            FreshPattern::Color { texel, .. }
                if layout.samples > 1
                    && plan.usage.contains(ImageUsageFlags::TRANSFER_DST)
                    && heap_tiled::holds_metal_texels(pf, plan.format, texel.len()) =>
            {
                self.scratch_pattern(plan, texel)?
            }
            FreshPattern::Color { .. } => None,
            // Every depth and stencil pattern is all ones: copies of ones write each aspect. A
            // render target's 32-bit float depth then holds a NaN with the sign set, which a pass
            // that loads it before a write covers it switches on (`fresh_unwritten.rs`).
            FreshPattern::DepthStencil { depth, .. }
                if single && self.ones_write(plan.format, pf) =>
            {
                let signed = depth == Some(FreshDepth::Nan)
                    && matches!(plan.format, Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT)
                    && usage.contains(TextureUsage::RENDER_TARGET);
                if signed {
                    let notice = Notice {
                        layout: *layout,
                        format: plan.format,
                        kind: FreshKind::SignedDepth,
                    };
                    let (aspect, extent) = (vk::ImageAspectFlags::DEPTH, plan.extent);
                    (self.shared().fresh).note(key, levels, plan.layers, aspect, extent, notice);
                }
                Some(copies(&[0xff])?)
            }
            FreshPattern::DepthStencil { depth, stencil } if clear => match depth {
                Some(FreshDepth::Nan) => None,
                _ => Some(Pattern::DepthStencil(vk::ClearDepthStencilValue {
                    depth: match depth {
                        Some(FreshDepth::Value(value)) => clear_value::depth(pf, value),
                        _ => 0.0,
                    },
                    stencil: stencil.map_or(0, clear_value::stencil),
                })),
            },
            FreshPattern::DepthStencil { .. } => None,
        };
        let Some(written) = written else {
            let notice = Notice {
                layout: *layout,
                format: plan.format,
                kind: FreshKind::Zeroed,
            };
            let (aspects, extent) = (texture_aspects(plan), plan.extent);
            (self.shared().fresh).note(key, levels, plan.layers, aspects, extent, notice);
            return Ok(None);
        };
        // The scratch's copy into a multisampled float image whose NaNs the device's copy
        // converts: the samples hold the device's NaNs, announced where a command first reads
        // them before a clear covers them.
        if matches!(written, Pattern::Scratch { .. })
            && nan_clear::converting_copy(self, plan.format, layout.samples)
        {
            let notice = Notice {
                layout: *layout,
                format: plan.format,
                kind: FreshKind::NansConverted,
            };
            let (aspects, extent) = (texture_aspects(plan), plan.extent);
            (self.shared().fresh).note(key, levels, plan.layers, aspects, extent, notice);
        }
        Ok(Some(Patterned::new(written, levels, plan.levels)))
    }

    /// The pattern `texel` of a new multisampled texture held in an image of `plan` that holds
    /// Metal's texels, where no clear of the image's format writes it (a NaN): copies of the
    /// device's scratch image of its samples, of the unsigned-integer format of the texel's size,
    /// cleared to its bits (`FirstMoves::scratch`, `Pattern::Scratch`). `None` where the texel's
    /// size has no such format or the device makes no such scratch.
    fn scratch_pattern(&self, plan: &ImagePlan, texel: &[u8]) -> Result<Option<Pattern>, Error> {
        let Some((format, bits)) = clear_value::uint_texel(texel) else {
            return Ok(None);
        };
        let moves = &self.shared().textures.first_moves;
        let (format, target, samples) = (format.into(), plan.format.into(), plan.samples.into());
        let Some(image) = moves.scratch(self, format, target, samples)? else {
            return Ok(None);
        };
        let [width, height, _] = plan.extent;
        Ok(Some(Patterned::scratch(
            image,
            bits,
            [width, height],
            plan.layers,
        )))
    }

    /// Whether copies of all-ones bytes write a new `pf` texture's depth and stencil pattern
    /// (`heap_tiled::fresh_pattern`, all ones) exactly into an image of `format`: each of the
    /// image's copy units is a stencil one, a depth one of `pf`'s depth bytes (a 32-bit float
    /// depth on a device whose depth images keep a copied value outside `[0, 1]`,
    /// `ImageFacts::depth_clamps`; its NaN too), or a colour one of them (a depth twin's
    /// `R32_SFLOAT` image).
    fn ones_write(&self, format: Format, pf: PixelFormat) -> bool {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let depth = match pf {
            PixelFormat::Depth16Unorm => 2,
            PixelFormat::Stencil8 => 0,
            _ => 4,
        };
        let units = copy_units(format, pf);
        !units.is_empty()
            && units.iter().all(|&(aspect, bytes, block)| {
                block == [1, 1]
                    && match aspect {
                        ImageAspectFlags::STENCIL => bytes == 1,
                        ImageAspectFlags::DEPTH => {
                            bytes == depth && (depth == 2 || !facts.depth_clamps())
                        }
                        ImageAspectFlags::COLOR => bytes == depth,
                        _ => false,
                    }
            })
    }

    /// Announce `fresh-pattern-zeroed` for a new `Private` texture of `notice`'s layout in an
    /// image of its format, which Metal reads its pattern in, which reads zero here, and which a
    /// command now reads before a clear covered it (`fresh_unwritten.rs`).
    pub(super) fn degrade_fresh_pattern(&self, notice: Notice) {
        let Notice {
            layout,
            format,
            kind,
        } = notice;
        if kind == FreshKind::NansConverted {
            let [width, height, depth] = layout.extent;
            return self.degrade(Degradation::DeviceNanEncoding, || {
                (
                    format!(
                        "a new Private {:?} {:?} texture of {width}x{height}x{depth}, {} samples",
                        layout.pixel_format, layout.texture_type, layout.samples
                    ),
                    format!(
                        "Metal's compressed texture reads a texel with a NaN, which the device's \
                         copy into a multisampled {format:?} image stores as it converts it \
                         (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`)"
                    ),
                )
            });
        }
        self.degrade(Degradation::FreshPatternZeroed, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "a new Private {:?} {:?} texture of {width}x{height}x{depth}, {} samples",
                layout.pixel_format, layout.texture_type, layout.samples
            );
            let why = match is_depth_stencil(layout.pixel_format) {
                true if layout.samples > 1 => format!(
                    "Metal's compressed texture reads a NaN depth (all ones), which no clear of \
                     a multisampled {format:?} image writes"
                ),
                true => format!(
                    "Metal's compressed texture reads a NaN depth (all ones), which a copy into a \
                     {format:?} image may store clamped on this device"
                ),
                false if layout.samples > 1 => format!(
                    "Metal's compressed texture reads a texel with a NaN, whose bits no clear of \
                     a multisampled {format:?} image controls, and no copy writes them here: the \
                     device makes no multisampled unsigned-integer image of the texel's size, or \
                     this image takes no copies of Metal's bytes"
                ),
                false => format!(
                    "Metal's compressed texture reads a fixed texel, and the {format:?} image \
                     holds other bytes than Metal's texel"
                ),
            };
            (object, why)
        });
    }

    /// Whether a clear writes every texel of an image of `plan` with zero: the image takes
    /// transfers, and its format is a color, depth or stencil attachment's at optimal tiling,
    /// which no block-compressed or multi-planar format is (no clear writes those).
    fn clears_to_zero(&self, plan: &ImagePlan) -> bool {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let features = facts.optimal(plan.format);
        let attachment = FormatFeatureFlags2::COLOR_ATTACHMENT
            .union(FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT);
        plan.usage.contains(ImageUsageFlags::TRANSFER_DST)
            && features.contains(FormatFeatureFlags2::TRANSFER_DST)
            && features.intersects(attachment)
    }
}

impl Device {
    /// Announce `depth-twin` for a texture of `layout` whose texels are held in an `R32_SFLOAT`
    /// image on a device with `facts`: why the texture is
    /// (`plan_depth_twin` and `holds_multisample_depth_in_colour` hold the texels there only
    /// where the facts state a loss), and, where `twin`, that a comparison reads the `D32_SFLOAT`
    /// twin beside it. A multisampled texture has no twin: Metal reads a `depth2d_ms` and
    /// compares nothing.
    pub(super) fn degrade_depth_twin(
        &self,
        layout: &TextureLayout,
        facts: &ImageFacts<'_>,
        twin: bool,
    ) {
        let loss = facts.depth_loss().unwrap_or_default();
        self.degrade(Degradation::DepthTwin, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "{:?} {:?} texture of {width}x{height}x{depth}",
                layout.pixel_format, layout.texture_type
            );
            let compared = match twin {
                true => ", and the comparisons the device makes read a D32_SFLOAT twin",
                false => ", which every read of it reads",
            };
            let why = format!("{loss}: its texels are held in an R32_SFLOAT image{compared}");
            (object, why)
        });
    }

    /// Announce `compressed-texel-decode` for a texture of `layout` in a BC format made on a
    /// device whose driver is not measured to decode its texels as Metal does (the format's
    /// [`block_decode_fact`]).
    pub(super) fn degrade_block_decode(&self, layout: &TextureLayout) {
        self.degrade_device_decode(layout.pixel_format, layout.texture_type, layout.extent);
    }

    /// [`degrade_block_decode`](Self::degrade_block_decode) of a texture or view of
    /// `pixel_format`, `texture_type` and `extent` (a view of another decode than its texture's
    /// decoded twin's too).
    pub(super) fn degrade_device_decode(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        [width, height, depth]: [u32; 3],
    ) {
        let Some(fact) = block_decode_fact(pixel_format) else {
            return;
        };
        if self.properties().driver_facts.contains(&fact) {
            return;
        }
        self.degrade(Degradation::CompressedTexelDecode, || {
            let object =
                format!("a {pixel_format:?} {texture_type:?} texture of {width}x{height}x{depth}");
            let why = format!(
                "the device's driver is not measured to decode the format's texels as Metal \
                 does (DriverFact::{fact:?}): Vulkan leaves the decode's arithmetic to the \
                 implementation"
            );
            (object, why)
        });
    }

    /// Whether a texture of `descriptor` (checked as `layout`) is a depth or stencil texture
    /// without GPU-optimized contents on a device whose driver keeps such images GPU-optimized
    /// (`DriverFact::DepthStencilImagesStayGpuOptimized`), which `optimized-depth-stencil-texture`
    /// announces ([`Device::degrade_optimized_layout`]).
    pub(super) fn keeps_optimized_layout(
        &self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
    ) -> bool {
        !descriptor.allow_gpu_optimized_contents
            && is_depth_stencil(layout.pixel_format)
            && self
                .properties()
                .driver_facts
                .contains(&DriverFact::DepthStencilImagesStayGpuOptimized)
    }

    /// Announce `optimized-depth-stencil-texture` for a depth or stencil texture of `descriptor`
    /// (checked as `layout`) sized or made in a heap without GPU-optimized contents on a device
    /// whose driver keeps such images GPU-optimized
    /// (`DriverFact::DepthStencilImagesStayGpuOptimized`): Metal lays the texture out without its
    /// compression metadata, which the image keeps, and the heap's size answer and its placement
    /// show the layout. A device's own texture, or a shared one, announces nothing: nothing shows
    /// its layout.
    pub(super) fn degrade_optimized_layout(
        &self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
    ) {
        if !self.keeps_optimized_layout(descriptor, layout) {
            return;
        }
        self.degrade(Degradation::OptimizedDepthStencilTexture, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "a {:?} {:?} texture of {width}x{height}x{depth} without GPU-optimized contents",
                layout.pixel_format, layout.texture_type
            );
            let why = "the device's driver keeps depth and stencil images GPU-optimized whatever \
                       their create info (DriverFact::DepthStencilImagesStayGpuOptimized): no \
                       Vulkan request drops the layout Metal drops"
                .to_owned();
            (object, why)
        });
    }
}

impl Heap {
    /// `newTextureWithDescriptor:` of an automatic or a sparse heap.
    ///
    /// An automatic heap's texture (`texture/heap.rs`) lies at the lowest offset with room that is
    /// a multiple of [`Device::heap_texture_size_and_align`]'s alignment, taking that call's size
    /// of the heap, its bytes the heap's memory as they are; its options are the heap's (a tracking
    /// mode named other than the heap's becomes the heap's, as plain Metal makes it). A texture
    /// buffer is a buffer of the heap's there with the texture over its bytes, and a texture the
    /// device holds with a depth twin takes the twin's bytes too. Refused: what
    /// `heap_texture_size_and_align` refuses; a storage or CPU cache mode other than the heap's
    /// (`Invalid`, as Metal ends the process); a heap whose memory the image cannot lie in
    /// (`NotImplemented(HeapTextureMemory)`); no room (`OutOfMemory(Heap)`).
    ///
    /// A sparse heap's texture is a sparse resident image with nothing mapped, with a tile map
    /// where its type has one (a single-sampled 2D, 2D array or 3D texture), from which a function
    /// reads the residency of Metal's tiles on every device. A device unit (a sparse
    /// block, 64 KiB on every measured device) is bound while any of Metal's tiles it holds texels
    /// of is mapped, so a 16-KiB tile maps alone, and the runtime keeps the texels of the block's
    /// unmapped tiles at zero, Metal's answer. A block that holds no mapped tile reads the device's
    /// value: zero, and its writes dropped, on a `residencyNonResidentStrict` device (the device
    /// announces `non-strict-residency` once otherwise), though one device's zero (RADV's) has no
    /// alpha of 1 where Metal's one-channel format has. Where the device makes no sparse image of
    /// the texture and no function reads an image's own residency (the device does not enable
    /// `shaderResourceResidency`; MoltenVK, which binds no sparse memory), it is an ordinary image
    /// whose unmapped tiles the runtime keeps at zero: Metal's answers. A blit's or a render pass's
    /// writes into unmapped tiles are zeroed again after it, as Metal drops them: measured for a
    /// blit as well as for a shader's read on an Apple M3 Max under macOS 27.0.1 (26A434), three
    /// runs, where a `copyFromBuffer:toTexture:` over a checkerboard-mapped level leaves every
    /// texel of each unmapped tile at zero under `copyFromTexture:toBuffer:` and under a kernel's
    /// `read()` alike, and the mapped tiles holding the copy's words (the probe
    /// `probes/sparse_unmapped_blit.m`). What that measurement also says, and what the runtime does
    /// not yet answer, is in `docs/FAST-PATHS.md`'s known gaps: a tile **mapped after** such a blit
    /// reads Metal's unwritten pattern, not zero. The runtime
    /// zeroes a single-sampled texture by copies, and a multisampled one, which no copy writes, by
    /// clears in a render pass, its image made an attachment of its format, as Metal's reads zero
    /// where unmapped (the M3 Max at 2x and 4x). Its storage is the heap's (`Private`), its CPU cache mode the descriptor's, its tracking
    /// mode the heap's, as Metal's are. Refused: what [`Device::new_texture`] refuses; a storage
    /// mode other than the heap's (stopped by Metal even without validation) and a dual-plane
    /// format (validation stops it; the stencil's mapping is not measured) (`Invalid`); a 1D, 1D
    /// array or texture buffer type (`Nil`, as Metal answers nil); a texture the device does not
    /// make sparse where a function may read the image's residency
    /// (`Unsupported(Feature::SparseTextures)`, naming what it lacks, or
    /// `NotImplemented(SparseTileMaps)` where it binds no sparse memory, as [`Device::new_heap`]
    /// refuses); a texture whose unmapped texels the runtime cannot zero: a single-sampled one of a
    /// format the device copies nothing into, a multisampled one of a format it does not clear and
    /// render (`NotImplemented(SparseTileMaps)`), or past its framebuffers
    /// (`Unsupported(Feature::RenderTargetSize)`); a `Depth32Float` texture that would need a depth twin
    /// (`NotImplemented(SparseHeaps)`); a two-plane extended-range format (`RGB10A8_2P_XR10` and
    /// its twin, `NotImplemented(Format)`).
    ///
    /// Refused for a placement heap, whose textures take an offset (`Invalid`, as for its
    /// buffers).
    pub fn new_texture(&self, descriptor: &TextureDescriptor) -> Result<Texture, Error> {
        let descriptor = &self.device().texture_options("new_texture", descriptor)?;
        match self.heap_type() {
            HeapType::Sparse => self.device().new_sparse_texture(self, descriptor),
            HeapType::Automatic => self.device().new_heap_texture(self, descriptor, None),
            heap_type => {
                let problem = Problem::Heap(HeapProblem::HeapType { heap_type });
                Err(invalid("new_texture", problem))
            }
        }
    }

    /// `newTextureWithDescriptor:offset:` of a placement heap: a texture whose image lies `offset`
    /// bytes into the heap's memory, over whatever else lies there, taking
    /// [`Device::heap_texture_size_and_align`]'s size; its bytes are the heap's as they are, and
    /// its options the heap's.
    ///
    /// Refused: another heap type (`Invalid`); an offset that is not a multiple of the answered
    /// alignment, bytes past the heap's end (`Nil`, as plain Metal answers nil, its validation
    /// stopping them); a two-plane extended-range cube or cube array, which Metal makes here
    /// (`NotImplemented(Format)`); what [`Heap::new_texture`] refuses of an
    /// automatic heap's texture. A descriptor plain Metal answers nil for answers `Nil` before
    /// the heap's own rules (its storage mode, `framebuffer_only`, the offset), and one Metal
    /// stops `Invalid` before them.
    ///
    /// Where it answers `Nil` or `OutOfMemory` for an offset inside the heap, the device also says
    /// so in an informational message (id name `metal2vulkan-heap-texture-refused`, `Info`
    /// severity) naming the descriptor, the offset, the heap, the size and alignment
    /// [`Device::heap_texture_size_and_align`] answers, what the making planned and the refusal: a
    /// diagnostic, as Metal says nothing.
    pub fn new_texture_at_offset(
        &self,
        descriptor: &TextureDescriptor,
        offset: u64,
    ) -> Result<Texture, Error> {
        let device = self.device();
        let descriptor = &device.texture_options("new_texture_at_offset", descriptor)?;
        device.new_heap_texture(self, descriptor, Some(offset))
    }
}

/// The driver fact under which a texture of `pf` reads its texels as Metal's, where `pf` is a
/// BC format, whose decode Vulkan leaves to the implementation: BC1 to BC3 (the sRGB forms too)
/// `ColorBlockDecodeIsMetals`, BC4 and BC5 `ChannelBlockDecodeIsMetals`. NVIDIA's decodes both
/// otherwise, RADV's the second. BC6H `Bc6hBlockDecodeIsMetals` and BC7 (its sRGB form through its
/// unorm view and Metal's table, `metal::texel_format::srgb_raw`) `Bc7BlockDecodeIsMetals`,
/// measured as the others, where every driver was taken to decode them as Metal does (lavapipe's
/// BC7 lies a `float` ulp off). Every other format is `None`.
pub(super) fn block_decode_fact(pf: PixelFormat) -> Option<DriverFact> {
    use PixelFormat as P;
    match pf {
        P::BC1RGBA | P::BC1RGBASRGB | P::BC2RGBA | P::BC2RGBASRGB | P::BC3RGBA | P::BC3RGBASRGB => {
            Some(DriverFact::ColorBlockDecodeIsMetals)
        }
        P::BC4RUnorm | P::BC4RSnorm | P::BC5RGUnorm | P::BC5RGSnorm => {
            Some(DriverFact::ChannelBlockDecodeIsMetals)
        }
        P::BC6HRGBFloat | P::BC6HRGBUfloat => Some(DriverFact::Bc6hBlockDecodeIsMetals),
        P::BC7RGBAUnorm | P::BC7RGBAUnormSRGB => Some(DriverFact::Bc7BlockDecodeIsMetals),
        _ => None,
    }
}

/// A sparse texture of a two-plane extended-range format, `pf`, is not built: its image holds both
/// planes side by side, not the tiles Metal's texture would map. The refusal names the format.
fn two_planes_refused(pf: PixelFormat) -> Result<(), Error> {
    if ExtendedRange::of(pf).is_some_and(|f| !f.holds_metal_bytes()) {
        return Err(Error::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(pf),
        ));
    }
    Ok(())
}

/// The bytes of the device's zeros, which a new resource's copies of zeros read
/// (`texture/first_move.rs`): the most one copy reads.
const ZERO_BYTES: u64 = 4 << 20;

/// How the runtime holds a sparse texture (`Device::plan_sparse_texture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SparseBacking {
    /// A sparse resident image: the device makes one of the texture's image.
    Bound,
    /// An ordinary image whose unmapped tiles the runtime keeps at zero: the device makes no
    /// sparse image of the texture, lacking `need`.
    Emulated { need: SparseNeed },
}

/// A sparse texture planned (`Device::plan_sparse_texture`): its layout, its image's plan, its
/// shape and aspects, Metal's tile and `firstMipmapInTail`, how the runtime zeroes its texels, and
/// how it holds it.
pub(crate) struct SparsePlan {
    layout: TextureLayout,
    plan: ImagePlan,
    shape: SparseShape,
    mask: vk::ImageAspectFlags,
    tile: [u64; 3],
    first: u64,
    zeroing: Zeroing,
    pub(crate) backing: SparseBacking,
}

/// `info` with the sparse flags: never 2D-array compatible (`VUID-VkImageCreateInfo-flags-09403`),
/// so a sparse 3D texture is no attachment (`render_pass.rs`).
fn sparse_create_info(info: vk::ImageCreateInfo<'_>) -> vk::ImageCreateInfo<'_> {
    info.flags(
        (info.flags & !vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE)
            | vk::ImageCreateFlags::SPARSE_BINDING
            | vk::ImageCreateFlags::SPARSE_RESIDENCY,
    )
}

/// What one texel block of each aspect of an image of `format` (presenting `pf`) is in a copy's
/// buffer: the aspect, its bytes, and its extent in the image's texels (a PVRTC carrier's 4 x 4,
/// `pvrtc::image_block_extent`).
pub(super) fn copy_units(
    format: Format,
    pf: PixelFormat,
) -> Vec<(ImageAspectFlags, u32, [u32; 2])> {
    let Some(block) = format.block() else {
        return Vec::new();
    };
    let mut units = Vec::new();
    if block.depth.is_some() {
        let bytes = match format {
            Format::D16_UNORM | Format::D16_UNORM_S8_UINT => 2,
            _ => 4,
        };
        units.push((ImageAspectFlags::DEPTH, bytes, [1, 1]));
    }
    if block.stencil.is_some() {
        units.push((ImageAspectFlags::STENCIL, 1, [1, 1]));
    }
    if units.is_empty() {
        let extent = image_block_extent(format, pf).unwrap_or([1, 1]);
        units.push((ImageAspectFlags::COLOR, block.size, extent));
    }
    units
}

/// How a new image starts ([`Device::new_image_starting`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    /// Holding what its memory held.
    Held,
    /// Zero in every texel.
    Zero,
    /// Moved to `GENERAL` alone, its texels written after (a decoded twin's, by the decode its
    /// image stages, `texture/decoded_twin.rs`): on the host or the staging queue where it takes
    /// their copies, as it is made; else by the next submission, as a held image's.
    Moved,
}

impl Start {
    /// [`Start::Held`] where `held`, else [`Start::Zero`].
    fn held_or_zero(held: bool) -> Start {
        match held {
            true => Start::Held,
            false => Start::Zero,
        }
    }
}

/// The copies that write every texel of an image of level-0 `extent`, `levels` and `layers`
/// (`volume`: a 3D image, whose depth slices are copied one by one) from the start of one buffer
/// of zeros, for each aspect's copy unit in `units`: bands of whole rows of at most `budget`
/// bytes (one row of blocks when a row is larger). The buffer's length, and the copies.
pub(super) fn zero_copies(
    extent: [u32; 3],
    levels: u32,
    layers: u32,
    volume: bool,
    units: &[(ImageAspectFlags, u32, [u32; 2])],
    budget: u64,
) -> (u64, Vec<ImageRegion>) {
    let mut length = 0;
    let mut copies = Vec::new();
    for &(aspect, bytes, [bw, bh]) in units {
        for level in 0..levels {
            let [w, h, d] = extent.map(|axis| (axis >> level.min(31)).max(1));
            let row_bytes = u64::from(w.div_ceil(bw)) * u64::from(bytes);
            let band = (budget / row_bytes).max(1);
            let rows = u64::from(h.div_ceil(bh));
            let slices = if volume { d } else { 1 };
            for layer in 0..layers {
                for z in 0..slices {
                    let mut y = 0;
                    while y < rows {
                        let n = band.min(rows - y);
                        length = length.max(n * row_bytes);
                        let top = y as u32 * bh;
                        let height = ((y + n) as u32 * bh).min(h) - top;
                        copies.push(ImageRegion {
                            aspect,
                            level,
                            layer,
                            layers: 1,
                            origin: [0, top, z],
                            size: [w, height, 1],
                        });
                        y += n;
                    }
                }
            }
        }
    }
    (length, copies)
}

/// A host-visible buffer `replace_region` and `get_bytes` copy through, released when dropped.
struct Staging<'a> {
    device: &'a Device,
    raw: vk::Buffer,
    placement: Placement,
    len: usize,
}

impl Staging<'_> {
    /// A staging buffer of `len` bytes.
    fn new(device: &Device, len: u64) -> Result<Staging<'_>, Error> {
        let shared = device.shared();
        let raw_device = shared.vk();
        let size = usize::try_from(len).map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(len)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Shared,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let staging = Staging {
            device,
            raw,
            placement,
            len: size,
        };
        if staging.placement.mapped.is_none() {
            return Err(Error::OutOfMemory(MemoryKind::Host));
        }
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        unsafe {
            crate::creation::bind_buffer(
                raw_device,
                raw,
                staging.placement.memory,
                staging.placement.offset,
            )
        }
        .map_err(vulkan)?;
        Ok(staging)
    }

    /// The staging bytes, through the mapping.
    fn bytes(&mut self) -> &mut [u8] {
        let Some(mapped) = self.placement.mapped else {
            unreachable!("checked when the staging buffer was made");
        };
        // SAFETY: the mapping covers the placement's `len` bytes, which only this staging buffer
        // uses; the memory is host-coherent, and no submission uses it while the slice lives
        // (`submit_internal` waits for its submission).
        unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), self.len) }
    }
}

impl Drop for Staging<'_> {
    fn drop(&mut self) {
        // Once the process is exiting, the buffer is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        // SAFETY: every submission that used the buffer completed.
        unsafe { release::object(shared.vk(), self.raw) };
        lock(&shared.memory).free(shared.vk(), &self.placement);
    }
}

/// The host pages an image is bound to, as its first move notes them (`texture/first_move.rs`).
fn pages_hold(pages: &Arc<HostPages>) -> Option<Weak<dyn Any + Send + Sync>> {
    let pages: Weak<dyn Any + Send + Sync> = Arc::downgrade(pages) as _;
    Some(pages)
}

/// The barrier ordering a copy of the internal submission after every earlier access of the
/// device's memory, and every later access after the copy.
pub(super) fn transfer_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    before: bool,
) {
    let (src_stage, src_access, dst_stage, dst_access) = if before {
        (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        )
    } else {
        (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::AccessFlags::MEMORY_READ
                | vk::AccessFlags::MEMORY_WRITE
                | vk::AccessFlags::HOST_READ,
        )
    };
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    // SAFETY: the internal command buffer is recording.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// `copy`, a copy between a buffer and the `R32_SFLOAT` image of a texture with a depth twin,
/// as the twin takes it: the same texels in its depth aspect.
pub(super) fn twin_copy(copy: vk::BufferImageCopy) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            ..copy.image_subresource
        },
        ..copy
    }
}

/// The Vulkan copy region of `region`, from the start of a tightly packed buffer.
pub(super) fn buffer_image_copy(region: &ImageRegion) -> vk::BufferImageCopy {
    let [x, y, z] = region.origin;
    let [width, height, depth] = region.size;
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: region.aspect.into(),
            mip_level: region.level,
            base_array_layer: region.layer,
            layer_count: region.layers,
        },
        image_offset: vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: z as i32,
        },
        image_extent: vk::Extent3D {
            width,
            height,
            depth,
        },
    }
}

/// `region` of `level` and `slice`, in the `u32`s Vulkan counts in; `None` when a value is past
/// any texture.
pub(super) fn texture_region(region: &Region, level: u64, slice: u64) -> Option<TextureRegion> {
    let n = |v: u64| u32::try_from(v).ok();
    let (o, s) = (region.origin, region.size);
    Some(TextureRegion {
        level: n(level)?,
        slice: n(slice)?,
        origin: [n(o.x)?, n(o.y)?, n(o.z)?],
        size: [n(s.width)?, n(s.height)?, n(s.depth)?],
    })
}

/// A Metal range of levels or slices in `u32`s; `None` when it is past any texture.
pub(super) fn narrow_range(range: &Range<u64>) -> Option<Range<u32>> {
    Some(u32::try_from(range.start).ok()?..u32::try_from(range.end).ok()?)
}

/// Every aspect of an image of `plan`: hazard tracking keys a texture's uses on all of them, so
/// a depth/stencil image's two aspects are never taken apart.
pub(super) fn texture_aspects(plan: &ImagePlan) -> vk::ImageAspectFlags {
    image_aspects(plan.format).into()
}

/// What owns the memory a texture's texels lie in (`Texture::memory`).
pub(super) enum TextureMemory<'a> {
    /// A heap, the texture's or its buffer's.
    Heap(&'a Heap),
    /// The buffer the texture is over.
    Buffer(&'a Buffer),
    /// The texture with the image: the texture itself, or the one its view was made from.
    Own(&'a Texture),
}

/// Whose a new texture object is, for its resource ID (`resource_ids.rs`).
#[derive(Clone, Copy, Debug)]
pub(super) enum IdUse {
    /// The program's, at a placement the caller checked: its ID holds its slot, so a function
    /// reading the ID reaches it; at [`IdPlacement::Outside`], no ID and no slot, though it is
    /// the program's resource still (`used_resources.rs`).
    Program(IdPlacement),
    /// The runtime's own (a scratch or check image): no ID (`IdHold::internal`).
    Runtime,
}

impl Texture {
    /// A new texture object of `device` over `storage`, presenting `record`, a view of `parent`
    /// if any, with its resource ID as `id_use` says. A program's texture takes its ID's slot,
    /// and the device's texture table and heap sets hold it there (`heap_sets::place`).
    fn made(
        device: &Device,
        storage: Storage,
        record: ViewRecord,
        parent: Option<Texture>,
        id_use: IdUse,
    ) -> Texture {
        let ids = &device.shared().resource_ids;
        let id = match id_use {
            IdUse::Program(placement) => IdHold::take(ids, IdSpace::Texture, placement),
            IdUse::Runtime => IdHold::internal(ids, IdSpace::Texture),
        };
        // A heap's image's heap lives while a texture is made from it, or while a texture of it a
        // view is made of lives.
        let heap = match &storage {
            Storage::Image(image) => match &image.owner {
                Owner::Sparse(sparse) => sparse.heap.upgrade(),
                owner => owner.heap_place().and_then(|place| place.heap.upgrade()),
            },
            Storage::Texels(_) | Storage::Alias { .. } | Storage::Shadow { .. } => None,
        };
        let texture = Texture(Arc::new(TextureShared {
            storage,
            record,
            parent,
            id,
            label: Mutex::new(None),
            object_id: ObjectId::new(),
            held_by: HeldBy::default(),
            usage_shadow: Mutex::new(None),
            snapshot: Mutex::new(Vec::new()),
            kept_backings: Mutex::new(Vec::new()),
            render_shadow: Mutex::new(None),
            heap,
            facts: OnceLock::new(),
        }));
        if let Some(objects) = texture.0.storage.objects() {
            objects.object_made();
        }
        // A texture over host pages' bytes, or in a heap over them (a view of one too): a
        // detach of them makes its image present a stand-in (`stand_in.rs`).
        if let Some((bytes, image, exact)) = texture.image_over_bytes() {
            let sink = image.plan().usage.contains(ImageUsageFlags::STORAGE);
            let shared: Arc<dyn Any + Send + Sync> = Arc::clone(&texture.0) as _;
            let image: Arc<dyn Any + Send + Sync> = Arc::clone(image) as _;
            bytes.note_texture(NotedTexture::new(
                Arc::downgrade(&shared),
                Arc::downgrade(&image),
                texture.0.object_id,
                (exact, sink),
            ));
        }
        if let IdUse::Program(placement) = id_use {
            if texture.0.parent.is_none() {
                device.shared().owners.texture(&texture);
            }
            // An `Outside` texture has no ID, so no slot (`IdHold::take`).
            if placement != IdPlacement::Outside {
                texture
                    .0
                    .id
                    .hold(Holder::Texture(Arc::downgrade(&texture.0)));
                heap_sets::place(device, &texture);
            }
        }
        texture
    }
}

impl Texture {
    /// Keep `token` for the texture's range, dropped as the last texture object over its image
    /// or a texture buffer's texels goes (`RangeToken`, `ObjectsToken`), as a creator makes it.
    /// Internal for a texture over another resource's bytes or a view (no creator gives one a
    /// token), and for one that keeps a token already.
    pub(super) fn keep_range_token(&self, token: RangeToken) -> Result<(), Error> {
        if self.0.parent.is_some() {
            return Err(internal("a range token for a texture view"));
        }
        match self.0.storage.objects() {
            Some(objects) => objects.keep(token),
            None => Err(internal("a range token for a texture over a buffer")),
        }
    }

    /// The texture object `shared` names, while it lives (a slot's holder, `resource_ids.rs`).
    pub(super) fn upgrade(shared: &Weak<TextureShared>) -> Option<Texture> {
        shared.upgrade().map(Texture)
    }

    /// The texture object, held while something else keeps it (a heap's, `heap.rs`).
    pub(super) fn downgrade(&self) -> Weak<TextureShared> {
        Arc::downgrade(&self.0)
    }

    /// The memoryless backings it keeps (`attachment_stand_in::kept_backing`).
    pub(super) fn kept_backings(&self) -> &Mutex<Vec<KeptBacking>> {
        &self.0.kept_backings
    }

    /// A texture over `texels`, presenting `record` of them, its ID as `id_use` says.
    pub(super) fn of_texels(texels: Texels, record: ViewRecord, id_use: IdUse) -> Texture {
        let device = texels.buffer.device().clone();
        let storage = Storage::Texels(Arc::new(texels));
        Texture::made(&device, storage, record, None, id_use)
    }

    /// Refused (`Invalid`, `StorageMode`) for a memoryless texture: Metal's validation stops every
    /// use of one but as a render pass's attachment (a Metal probe on an M4: CPU access, blits,
    /// mipmaps, views and bindings).
    pub(super) fn check_not_memoryless(&self, call: &'static str) -> Result<(), Error> {
        if self.storage_mode() == StorageMode::Memoryless {
            return Err(invalid(call, Problem::StorageMode));
        }
        Ok(())
    }

    /// Refused as [`check_not_memoryless`](Self::check_not_memoryless) refuses, and
    /// (`NotImplemented(RotatedTexture)`) for a rotated texture whose bytes its CPU access and
    /// blits address run past its buffer (`texture/rotation.rs`): the uses of its bytes.
    pub(super) fn check_bytes_reached(&self, call: &'static str) -> Result<(), Error> {
        self.check_not_memoryless(call)?;
        self.check_rotated_reach()
    }

    /// What the texture presents of its image.
    pub(super) fn record(&self) -> &ViewRecord {
        &self.0.record
    }

    /// The texture's image. Refused for a texture buffer, which has none (`NotImplemented`): a
    /// call that needs one is not built on texture buffers.
    fn image(&self) -> Result<&Arc<Image>, Error> {
        match &self.0.storage {
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => Ok(image),
            Storage::Texels(_) => Err(Error::NotImplemented(RuntimeGap::Textures)),
        }
    }

    /// The buffer's bytes a texture buffer is, or `None` for a texture with an image.
    pub(super) fn texels(&self) -> Option<&Texels> {
        match &self.0.storage {
            Storage::Image(_) | Storage::Alias { .. } | Storage::Shadow { .. } => None,
            Storage::Texels(texels) => Some(texels),
        }
    }

    /// The buffer's bytes a texture over a buffer holds its texels in (a texture buffer, or a 2D
    /// texture or array with an aliasing or a shadow image), or `None` for a texture of an image
    /// alone.
    pub(super) fn buffer_texels(&self) -> Option<&Texels> {
        match &self.0.storage {
            Storage::Image(_) => None,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => Some(texels),
        }
    }

    /// Whether the texture's image is a linear image over its buffer's bytes now: an aliasing
    /// image (not a shadow, not a texture buffer), not presenting a stand-in while its bytes are
    /// detached (`detach.rs`), so a write of those bytes is a write of its texels.
    pub(super) fn aliases_buffer(&self) -> bool {
        matches!(self.0.storage, Storage::Alias { .. }) && self.linear_image().unwrap_or(false)
    }

    /// Whether the texture's image lies over its buffer's bytes now: an aliasing image, not
    /// presenting a stand-in while its bytes are detached (`detach.rs`), so a pass that renders it
    /// writes those bytes.
    pub(super) fn image_over_buffer(&self) -> bool {
        match &self.0.storage {
            Storage::Alias { image, .. } => lock(&image.detached).is_none(),
            _ => false,
        }
    }

    /// Whether the texture is a buffer's bytes: a texture buffer or a shadowed texture.
    pub(super) fn over_buffer(&self) -> bool {
        !matches!(self.0.storage, Storage::Image(_))
    }

    /// The copies of a shadowed texture, of a texture buffer viewed through a staging buffer
    /// (`Texels::texture_buffer`), or of a heap's depth or stencil texture held as a shadow of
    /// the heap's bytes (`texture/heap.rs`), or `None` for another texture.
    pub(super) fn shadow(&self) -> Option<&Arc<ShadowCopies>> {
        match &self.0.storage {
            Storage::Shadow { copies, .. } => Some(copies),
            Storage::Texels(texels) => texels.staged(),
            Storage::Image(image) => match &image.owner {
                Owner::HeapShadow(shadow) => Some(&shadow.copies),
                Owner::ViewShadow(shadow) => Some(&shadow.copies),
                _ => None,
            },
            Storage::Alias { .. } => None,
        }
    }

    /// Whether two textures (or views) hold their texels in one image or buffer.
    pub(super) fn same_storage(&self, other: &Texture) -> bool {
        match (&self.0.storage, &other.0.storage) {
            (Storage::Image(a), Storage::Image(b)) => {
                Arc::ptr_eq(view_shadow::truth(a), view_shadow::truth(b))
            }
            (Storage::Texels(a), Storage::Texels(b)) => Arc::ptr_eq(a, b),
            (Storage::Shadow { image: a, .. }, Storage::Shadow { image: b, .. })
            | (Storage::Alias { image: a, .. }, Storage::Alias { image: b, .. }) => {
                Arc::ptr_eq(a, b)
            }
            _ => false,
        }
    }

    /// Whether a blit wrote bytes outside the texels of the texture's image, which Metal's blit
    /// of a two-plane extended-range texture keeps and reads back (`two_planes.rs`); `false` for
    /// a texture buffer.
    pub(super) fn kept_outside(&self) -> bool {
        self.kept_image()
            .is_some_and(|image| image.kept_outside.load(Ordering::Relaxed))
    }

    /// Note that a blit wrote bytes outside the texels of the texture's image.
    pub(super) fn keep_outside(&self) {
        if let Some(image) = self.kept_image() {
            image.kept_outside.store(true, Ordering::Relaxed);
        }
    }

    /// Whether bytes outside the texels of the texture's image were written: ones the runtime
    /// holds ([`Texture::between_rows`]) or not ([`Texture::kept_outside`]).
    pub(super) fn outside_written(&self) -> bool {
        self.kept_outside() || self.between_rows().is_some()
    }

    /// The bytes between the rows of a two-plane extended-range texture's plane 0 that its fills
    /// and blits with a buffer wrote (`two_planes/buffer_rows.rs`); `None` before the first.
    pub(super) fn between_rows(&self) -> Option<&Buffer> {
        self.kept_image()?.between.get()
    }

    /// [`Texture::between_rows`], made by `make` where none is yet: the buffer, and whether
    /// `make` made it (its bytes, then, the caller's to zero).
    pub(super) fn hold_between_rows(
        &self,
        make: impl FnOnce() -> Result<Buffer, Error>,
    ) -> Result<(Buffer, bool), Error> {
        let image = self
            .kept_image()
            .ok_or_else(|| internal("bytes between the rows of a texture without an image"))?;
        if let Some(held) = image.between.get() {
            return Ok((held.clone(), false));
        }
        let made = make()?;
        match image.between.set(made.clone()) {
            Ok(()) => Ok((made, true)),
            Err(_) => image
                .between
                .get()
                .cloned()
                .map(|held| (held, false))
                .ok_or_else(|| internal("bytes between rows neither made nor held")),
        }
    }

    /// Where the texture's two planes lie in its heap's bytes, as Metal's blits address them (a
    /// heap's one-level 2D two-plane texture, `texture/heap_two_planes.rs`): the heap, and the
    /// texture's first byte in it; `None` for any other texture, a view and a two-plane texture
    /// of more levels or slices among them.
    pub(super) fn heap_planes(&self) -> Option<(&Heap, u64)> {
        match &self.0.storage {
            Storage::Image(image) if self.0.parent.is_none() => match &image.owner {
                Owner::HeapShadow(shadow) if shadow.copies.holds_blit_rows() => {
                    Some((self.0.heap.as_ref()?, shadow.place.offset))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Where the texture's first level's planes lie in its heap's bytes (a heap's two-plane
    /// texture of any shape the runtime holds over them, `texture/heap_two_planes.rs`): the heap,
    /// the texture's first byte in it, and their layout; `None` for any other texture, a view
    /// among them.
    pub(super) fn heap_two_planes(&self) -> Option<(&Heap, u64, &TwoPlaneHeapLayout)> {
        match &self.0.storage {
            Storage::Image(image) if self.0.parent.is_none() => match &image.owner {
                Owner::HeapShadow(shadow) => {
                    let layout = shadow.copies.plane_layout()?;
                    Some((self.0.heap.as_ref()?, shadow.place.offset, layout))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The image whose bytes a blit of the texture addresses: a view shadow's texture's.
    fn kept_image(&self) -> Option<&Arc<Image>> {
        match &self.0.storage {
            Storage::Image(image) => Some(view_shadow::truth(image)),
            Storage::Alias { image, .. } | Storage::Shadow { image, .. } => Some(image),
            Storage::Texels(_) => None,
        }
    }

    /// The plan of the texture's image; refused as [`image`](Self::image) is.
    pub(super) fn image_plan(&self) -> Result<&ImagePlan, Error> {
        Ok(self.image()?.plan())
    }

    /// Whether the texture's image is linear now: a linear image (`Image::linear`) that is not
    /// presenting a stand-in while the texture's bytes are detached (`detach.rs`). Refused as
    /// [`image`](Self::image) is.
    pub(super) fn linear_image(&self) -> Result<bool, Error> {
        let image = self.image()?;
        Ok(image.linear && lock(&image.detached).is_none())
    }

    /// The plan of the image holding the texture's texels in its format's aspects: its depth
    /// twin's for a texture with one, else its image's. Refused as [`image`](Self::image) is.
    pub(super) fn depth_plan(&self) -> Result<&ImagePlan, Error> {
        let image = self.image()?;
        Ok(image.twin.as_ref().map_or(image.plan(), |twin| twin.plan()))
    }

    /// The bytes of device memory the runtime placed the texture's image in: `None` for an image
    /// in a buffer's, a heap's or the caller's memory, a sparse image (bound by unit), and where
    /// [`image`](Self::image) refuses. A shadowed view's image is in memory of its own
    /// (`texture/view_shadow.rs`).
    pub(super) fn placed_bytes(&self) -> Option<u64> {
        match &self.image().ok()?.owner {
            Owner::Runtime { placement, .. } => Some(placement.size()),
            Owner::ViewShadow(shadow) => Some(shadow.placement.size()),
            Owner::Buffer { .. }
            | Owner::Caller(_)
            | Owner::Sparse(_)
            | Owner::Heap(_)
            | Owner::HeapTwin
            | Owner::HeapShadow(_)
            | Owner::HeapUnaliased(_) => None,
        }
    }

    /// Whether the texture's image is the caller's (`new_texture_with_vk_image`,
    /// `texture/wrapped.rs`).
    pub(super) fn callers_image(&self) -> bool {
        self.image()
            .is_ok_and(|image| matches!(image.owner, Owner::Caller(_)))
    }

    /// Whether the texture's image lies in lazily allocated memory: a memoryless texture's,
    /// where the device has such memory (`image_plan.rs`). MoltenVK runs such an image as Metal's
    /// memoryless storage, which keeps no texels past the render pass that renders them.
    pub(super) fn lazily_allocated(&self) -> bool {
        let Ok(image) = self.image() else {
            return false;
        };
        let (_, placement) = image.owner.releases();
        placement.is_some_and(|placement| {
            let memory = lock(&image.device.shared().memory);
            memory
                .type_flags(placement.type_index())
                .is_some_and(|flags| flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED))
        })
    }

    /// The texture's `VkImage`; refused as [`image`](Self::image) is.
    pub(super) fn raw_image(&self) -> Result<vk::Image, Error> {
        Ok(self.image()?.raw.get())
    }

    /// Whether the texture's texels are in a colour image: a `Depth32Float` texture held in
    /// `R32_SFLOAT`, which keeps every `float` a copy writes where the device's depth images may
    /// not ([`holds_depth_in_colour`] for which plans those are). `false` for every other
    /// texture.
    pub(super) fn holds_depth_in_colour(&self) -> bool {
        self.image()
            .is_ok_and(|image| holds_depth_in_colour(image.plan()))
    }

    /// The `VkImage` of the texture's depth twin, or `None` for a texture without one.
    pub(super) fn twin_image(&self) -> Option<vk::Image> {
        Some(self.image().ok()?.twin.as_ref()?.raw.get())
    }

    /// Announce what `call`'s upload into `aspect` of the texture's image may store otherwise than
    /// Metal's, in the depth of a 32-bit float depth image: on a device without
    /// `VK_EXT_depth_range_unrestricted` enabled whose driver is not measured to keep a value
    /// outside `[0, 1]` there (`ImageFacts::depth_clamps`), `restricted-depth-upload`. On any
    /// other, an upload that may store a negative value or a NaN (`signed`) in a render target's
    /// depth, which a depth test reads, switches it on to render over its magnitude stand-in, so
    /// the device's test answers as Metal's, which compares the value's magnitude
    /// (`texture/signed_depth.rs`; `encoding` where a command buffer of the program's records the
    /// upload); a target that takes no stand-in announces `signed-depth-test`. A texture with a
    /// depth twin holds its texels in colour (its twin's clamping is announced with the twin).
    pub(super) fn degrade_depth_upload(
        &self,
        call: &'static str,
        aspect: ImageAspectFlags,
        signed: bool,
        encoding: bool,
    ) {
        let Ok(image) = self.image() else {
            return;
        };
        let float_depth = matches!(
            image.plan().format,
            Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT
        );
        if aspect != ImageAspectFlags::DEPTH || !float_depth {
            return;
        }
        let device = self.device();
        let profile = device.compile_device().profile();
        let restricted = device.shared().textures.facts(profile).depth_clamps();
        let format = self.pixel_format();
        if restricted {
            device.degrade(Degradation::RestrictedDepthUpload, || {
                let why = format!(
                    "{call} wrote its depth image on a device without \
                     VK_EXT_depth_range_unrestricted"
                );
                (format!("a {format:?} texture"), why)
            });
        } else if signed {
            self.degrade_signed_depth(call, encoding);
        }
    }

    /// Switch the texture's depth on for `call`'s write into it, which may store a negative value
    /// or a NaN, on a device that keeps such a value in a depth image (`texture/signed_depth.rs`;
    /// `encoding` where a command buffer of the program's records the write): a render target's
    /// 32-bit float depth, which a depth test reads and a pass with depth writes stores. One whose
    /// device's depth test then answers as Metal's (it takes the magnitude stand-in, or the
    /// device's test is Metal's own, `DriverFact::DepthTestComparesMagnitudes`) announces
    /// nothing; one that takes no stand-in (multisampled, or a combined depth/stencil format)
    /// announces `signed-depth-test`, as the device's test compares the value as it is, where
    /// Metal's compares its magnitude, and names whether its passes' stores are Metal's (the
    /// strip) or keep the sign.
    pub(super) fn degrade_signed_depth(&self, call: &'static str, encoding: bool) {
        let Ok(image) = self.image() else {
            return;
        };
        let float_depth = matches!(
            image.plan().format,
            Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT
        );
        if !float_depth
            || !self.usage().contains(TextureUsage::RENDER_TARGET)
            || self.switch_on_signed_depth(call, encoding)
        {
            return;
        }
        let device = self.device();
        let profile = device.compile_device().profile();
        let facts = device.shared().textures.facts(profile);
        if facts.depth_clamps() {
            return;
        }
        let stripped = self.magnitude_form().is_some();
        let format = self.pixel_format();
        device.degrade(Degradation::SignedDepthTest, || {
            let why = match (facts.depth_images.test_magnitudes, stripped) {
                (false, true) => format!(
                    "{call} may store a negative or NaN depth in a render target's depth image, \
                     which the device's depth test compares as it is, where Metal's compares its \
                     magnitude: a multisampled or combined depth/stencil target renders over no \
                     magnitude stand-in (a pass with depth writes stores it without its sign, as \
                     Metal's)"
                ),
                (false, false) => format!(
                    "{call} may store a negative or NaN depth in a render target's depth image, \
                     which the device's depth test compares as it is, where Metal's compares its \
                     magnitude, and which a pass with depth writes keeps with its sign, where \
                     Metal stores it without"
                ),
                (true, _) => format!(
                    "{call} may store a negative or NaN depth in a render target's depth image, \
                     which a pass with depth writes that draws under none keeps with its sign, \
                     where Metal stores it without"
                ),
            };
            (format!("a {format:?} texture"), why)
        });
    }

    /// Announce `depth-denormals` for `call`'s write into `aspect` of the texture's image that may
    /// store a denormal (`on_host`: by a host copy): the 32-bit float depth of a depth image, on
    /// a device whose driver may store it or test it otherwise than Metal's
    /// ([`ImageFacts::denormal_loss`]). A texture with a depth twin holds its texels in colour.
    /// Where the driver stores it and only its depth test compares it otherwise, a render target
    /// that takes a magnitude stand-in is switched on (`texture/signed_depth.rs`, `encoding` as
    /// [`degrade_depth_upload`](Self::degrade_depth_upload) takes it): its passes test the
    /// smallest normal in a denormal's place, Metal's answer but for a fragment at exactly 2^-126,
    /// which the notice names. A host copy's denormal on a driver whose GPU copies store one as a
    /// zero switches it on too, but the stand-in's copies read it as a zero: the notice names the
    /// loss.
    pub(super) fn degrade_depth_denormals(
        &self,
        call: &'static str,
        aspect: ImageAspectFlags,
        on_host: bool,
        encoding: bool,
    ) {
        let Ok(image) = self.image() else {
            return;
        };
        let float_depth = matches!(
            image.plan().format,
            Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT
        );
        if aspect != ImageAspectFlags::DEPTH || !float_depth {
            return;
        }
        let device = self.device();
        let profile = device.compile_device().profile();
        let facts = device.shared().textures.facts(profile);
        let Some(loss) = facts.denormal_loss(on_host) else {
            return;
        };
        let stored = on_host || facts.depth_images.denormals;
        let switched = stored
            && self.usage().contains(TextureUsage::RENDER_TARGET)
            && self.switch_on_signed_depth(call, encoding)
            && self.takes_magnitude_stand_in();
        // The stand-in's copies keep a denormal only where the device's GPU copies do.
        let tested = switched && facts.depth_images.denormals;
        device.degrade(Degradation::DepthDenormals, || {
            let format = self.pixel_format();
            let why = match tested {
                true => format!(
                    "{call} may store a denormal depth in a render target, whose passes test it \
                     over the smallest normal, as the device's depth test compares a stored \
                     denormal as zero: a fragment at exactly 2^-126 passes `Equal` and fails \
                     `Greater` against it, where Metal's fails and passes"
                ),
                false => format!("{call} may store a denormal depth: {loss}"),
            };
            (format!("a {format:?} texture"), why)
        });
    }

    /// The `VkImageView` a binding write of `call` needs of the texture: `desc` is the binding
    /// plan's view, `storage` whether the write is a storage image's. Made at the first bind
    /// that needs it, then kept with the image. Refused as `binding_view` refuses.
    pub(super) fn bound_view(
        &self,
        call: &'static str,
        desc: &ImageViewDesc,
        storage: bool,
    ) -> Result<vk::ImageView, Error> {
        let mut image: &Image = self.image()?;
        // A sampled view of a texture with a decoded twin presents the twin.
        if !storage {
            if let Some(view) = self.decoded_view(call, image, desc)? {
                return Ok(view);
            }
        }
        // A view a comparison reads presents the texture's depth twin.
        if desc.twin {
            image = image
                .twin
                .as_deref()
                .ok_or_else(|| internal("a depth twin's view of a texture without one"))?;
        }
        let view = binding_view(image.plan(), &self.0.record.bound(), desc, storage)
            .map_err(|refusal| view_error(call, refusal))?;
        image.view(&view)
    }

    /// The `VkImageView` `view` of the texture's image, made at its first use and then kept with
    /// the image. The caller checked `view` against the image's plan (format, usage, levels and
    /// layers). Refused as [`image`](Self::image) is.
    pub(super) fn image_view(&self, view: &VkView) -> Result<vk::ImageView, Error> {
        self.image()?.view(view)
    }

    /// The objects of exact color resolves into the texture's image that its passes made
    /// (`color_resolve.rs`), which the image keeps until it drops. Refused as
    /// [`image`](Self::image) is.
    pub(super) fn kept_resolves(&self) -> Result<&KeptResolves, Error> {
        Ok(&self.image()?.resolves)
    }

    /// The facts a binding plan reads of the texture, worked out once and kept
    /// (`TextureShared::facts`).
    pub(super) fn facts(&self) -> TextureFacts {
        *self.0.facts.get_or_init(|| self.worked_out_facts())
    }

    /// The facts a binding plan reads of the texture, worked out from what it was made with.
    pub(super) fn worked_out_facts(&self) -> TextureFacts {
        let record = self.0.record.bound();
        let mut facts = texture_facts(&record);
        // A `Depth32Float` texture whose texels are in an `R32_SFLOAT` image: a sampled view
        // reads the colour image, and a compared one the depth twin a single-sampled one has
        // beside it (`image_plan::plan_depth_twin`). A multisampled one has no twin and no
        // comparison to serve: Metal reads a `depth2d_ms` and compares nothing
        // (`MultisampleDepthTwinPolicy::Exact`).
        facts.depth_twin = self.holds_depth_in_colour();
        facts.sparse = self.is_sparse();
        // A wrapped image made without `MUTABLE_FORMAT` takes no view in a storage binding's
        // other formats (a half-float texture's unsigned integer alias): its storage slots keep
        // its own format's view. A texture whose storage binding takes the usage shadow binds
        // the shadow, made with them.
        facts.own_format_views = !self.needs_usage_shadow(true)
            && self.image().is_ok_and(|image| {
                let profile = self.device().compile_device().profile();
                storage_view_formats(record.pixel_format, profile)
                    .iter()
                    .any(|&format| !views_as(image.plan(), format, false))
            });
        facts
    }

    /// A sparse texture's image beyond the `VkImage` (`sparse/texture.rs`); `None` for a texture
    /// that is not sparse.
    pub(super) fn sparse_image(&self) -> Option<&SparseImage> {
        match &self.0.storage {
            Storage::Image(image) => match &image.owner {
                Owner::Sparse(sparse) => Some(sparse),
                _ => None,
            },
            Storage::Texels(_) | Storage::Alias { .. } | Storage::Shadow { .. } => None,
        }
    }

    /// A use of everything the texture presents (its levels and slices of its image, or a
    /// texture buffer's bytes of its buffer), reading by `read` and writing by `write` (either
    /// may be `Access::NONE`).
    pub(super) fn bound_use(&self, read: Access, write: Access) -> Use<ResourceKey> {
        let record = &self.0.record;
        let image = match &self.0.storage {
            Storage::Image(image) | Storage::Shadow { image, .. } => image,
            // An aliasing image's uses are uses of the buffer's bytes it lies in.
            Storage::Texels(texels) | Storage::Alias { texels, .. } => {
                return texels.bound_use(read, write)
            }
        };
        let identity = image.identity;
        Use {
            key: identity.key,
            tracked: identity.tracked,
            range: UseRange::Image {
                aspects: texture_aspects(image.plan()),
                levels: record.base_level..record.base_level + record.level_count,
                layers: record.base_layer..record.base_layer + record.layer_count,
            },
            read,
            write,
        }
    }

    /// A read (`write` false) or write of `levels` and `layers` of the texture's image, counted
    /// in the image's own levels and layers, by `access`. A view's uses are its texture's; an
    /// aliasing image's are of the buffer's bytes it lies in, all of them. Refused as
    /// [`image`](Self::image) is.
    pub(super) fn image_use(
        &self,
        levels: Range<u32>,
        layers: Range<u32>,
        access: Access,
        write: bool,
    ) -> Result<Use<ResourceKey>, Error> {
        if let Storage::Alias { texels, .. } = &self.0.storage {
            return Ok(if write {
                texels.bound_use(Access::NONE, access)
            } else {
                texels.bound_use(access, Access::NONE)
            });
        }
        let image = self.image()?;
        let identity = image.identity;
        // A 3D image has one layer: a use of its depth planes (a 2D view of them, an attachment)
        // is a use of the level.
        let layers = match image.plan().image_type {
            metal2vulkan::vk::ImageType::TYPE_3D => 0..1,
            _ => layers,
        };
        let range = UseRange::Image {
            aspects: texture_aspects(image.plan()),
            levels,
            layers,
        };
        Ok(if write {
            Use::writes(identity.key, identity.tracked, range, access)
        } else {
            Use::reads(identity.key, identity.tracked, range, access)
        })
    }
}

impl Texture {
    /// `device`.
    pub fn device(&self) -> &Device {
        match &self.0.storage {
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => &image.device,
            Storage::Texels(texels) => texels.buffer.device(),
        }
    }

    /// `textureType`.
    pub fn texture_type(&self) -> TextureType {
        self.0.record.texture_type
    }

    /// `pixelFormat`.
    pub fn pixel_format(&self) -> PixelFormat {
        self.0.record.pixel_format
    }

    /// `width`: of the texture's first level.
    pub fn width(&self) -> u64 {
        u64::from(self.0.record.extent()[0])
    }

    /// `height`: of the texture's first level.
    pub fn height(&self) -> u64 {
        u64::from(self.0.record.extent()[1])
    }

    /// `depth`: of the texture's first level.
    pub fn depth(&self) -> u64 {
        u64::from(self.0.record.extent()[2])
    }

    /// `mipmapLevelCount`.
    pub fn mipmap_level_count(&self) -> u64 {
        u64::from(self.0.record.level_count)
    }

    /// `arrayLength`: the slices, or the cubes of a cube type.
    pub fn array_length(&self) -> u64 {
        u64::from(self.0.record.array_length())
    }

    /// `sampleCount`.
    pub fn sample_count(&self) -> u64 {
        u64::from(self.0.record.samples)
    }

    /// `usage`: a texture made with `Unknown` usage reports `ShaderRead | ShaderWrite |
    /// RenderTarget | PixelFormatView`, as Metal does; a swizzled view keeps only `ShaderRead`
    /// and `PixelFormatView` of its texture's.
    ///
    /// Bits Metal's texture drops from its descriptor's are not reported: a texture made with 0x8
    /// alone reports no usage, one of 0x9 `ShaderRead`, as Metal's do.
    ///
    /// `ShaderAtomic` is dropped the same way, and is not `TextureUsage::DROPPED`'s: that constant
    /// is the bits the SDK's header declares no name for, and `MTLTextureUsageShaderAtomic` has a
    /// name. Metal takes it on the descriptor, lets it decide the texture's layout -- a 64 by 64
    /// `R32Uint` texture of `ShaderRead | ShaderAtomic` answers the uncompressed 16384 bytes where
    /// `ShaderRead` alone answers 16512 -- and then reports the usage **without** it. Measured on
    /// an Apple M3 Max under macOS 27.0.1 over every combination `R32Uint` accepts, which is the
    /// one format Metal takes the bit on at all: 0x20 reports 0, 0x21 reports 0x1, 0x22 0x2, 0x23
    /// 0x3, 0x25 0x5, 0x31 0x11 and 0x4021 0x4001. So the bit is kept in the record, where the
    /// layout reads it, and taken off here, where Metal takes it off.
    pub fn usage(&self) -> TextureUsage {
        self.0
            .record
            .usage
            .without_dropped()
            .difference(TextureUsage::SHADER_ATOMIC)
    }

    /// `swizzle`.
    pub fn swizzle(&self) -> TextureSwizzleChannels {
        self.0.record.swizzle
    }

    /// Whether shaders read the texture's alpha as 1 (a runtime extension,
    /// `TextureDescriptor::alpha_reads_one`): its descriptor's, or a view's texture's.
    pub fn alpha_reads_one(&self) -> bool {
        self.0.record.alpha_reads_one
    }

    /// The address modes the texture is sampled with (a runtime extension,
    /// `TextureDescriptor::address_mode_map`): its descriptor's, or a view's texture's.
    pub fn address_mode_map(&self) -> AddressModeMap {
        self.0.record.address_mode_map
    }

    /// `resourceOptions`: the texture's (a view has its texture's; a texture over a buffer has
    /// the buffer's, but a texture of an IOSurface plane its descriptor's), with the hazard
    /// tracking mode it resolved rather than its descriptor's.
    ///
    /// Metal reports no resource's options as `Default`: a texture outside a heap reads
    /// `0x200` shared and `0x220` private, a texture in one reads the mode the heap gave it
    /// ([`hazard_tracking_mode`](Self::hazard_tracking_mode), `Identity::new`). Measured on an
    /// Apple M3 Max under macOS 27.0.1.
    pub fn resource_options(&self) -> ResourceOptions {
        let options = match &self.0.storage {
            Storage::Image(image) => image.options,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => texels.options(),
        };
        options.with_hazard_tracking_mode(self.hazard_tracking_mode())
    }

    /// `storageMode`.
    pub fn storage_mode(&self) -> StorageMode {
        self.resource_options()
            .storage_mode()
            .expect("checked when the texture was made")
    }

    /// `cpuCacheMode`.
    pub fn cpu_cache_mode(&self) -> CpuCacheMode {
        self.resource_options()
            .cpu_cache_mode()
            .expect("checked when the texture was made")
    }

    /// `hazardTrackingMode`: `Tracked` or `Untracked`, `Default` resolved as Metal resolves it.
    pub fn hazard_tracking_mode(&self) -> HazardTrackingMode {
        let tracked = match &self.0.storage {
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => image.identity.tracked,
            Storage::Texels(texels) => texels.tracked(),
        };
        if tracked {
            HazardTrackingMode::Tracked
        } else {
            HazardTrackingMode::Untracked
        }
    }

    /// `allowGPUOptimizedContents`.
    pub fn allow_gpu_optimized_contents(&self) -> bool {
        match &self.0.storage {
            Storage::Image(image) => image.allow_gpu_optimized_contents,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => texels.allow_gpu_optimized_contents,
        }
    }

    /// The texture's identity, a runtime extension ([`ObjectId`]): Metal names a texture by its
    /// pointer; this is unique among the process's buffers, textures and heaps for the texture's
    /// life and never reused, where a `gpu_resource_id` is taken again once its texture is freed.
    /// A view has its own.
    pub fn object_id(&self) -> ObjectId {
        self.0.object_id
    }

    /// `gpuResourceID`: what an argument buffer holds to name the texture (`resource_ids.rs`); a
    /// view has its own.
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.0.id.id()
    }

    /// `isSparse`: the texture was made from a sparse heap (a view of such a texture is sparse
    /// too).
    pub fn is_sparse(&self) -> bool {
        self.sparse_image().is_some()
    }

    /// `firstMipmapInTail`: a sparse texture's first level in its mip tail, the level count
    /// without one (Metal's rule; a view answers its texture's); `None` for a texture that is not
    /// sparse, whose answer is not measured.
    pub fn first_mipmap_in_tail(&self) -> Option<u64> {
        self.sparse_image()
            .map(|sparse| sparse.grid.first_mipmap_in_tail)
    }

    /// `tailSizeInBytes`: a sparse texture's mip tail's bytes per slice, whole pages of its
    /// heap, 0 without a tail (Metal's rule; a view answers its texture's); `None` for a texture
    /// that is not sparse, whose answer is not measured.
    pub fn tail_size_in_bytes(&self) -> Option<u64> {
        self.sparse_image().map(|sparse| sparse.tail_size_in_bytes)
    }

    /// `heap`: the heap the texture was made from, if any.
    pub fn heap(&self) -> Option<&Heap> {
        if let Some(heap) = &self.0.heap {
            return Some(heap);
        }
        self.heap_texels().and_then(Buffer::heap)
    }

    /// `heapOffset`: where the texture lies in a placement heap; 0 for a texture of an automatic or
    /// sparse heap, as Metal answers (and a Metal heap probe on the M3 Max and an M4
    /// for texture buffers), and for one outside a heap.
    pub fn heap_offset(&self) -> u64 {
        if let Some(buffer) = self.heap_texels() {
            return buffer.heap_offset();
        }
        let placement =
            (self.0.heap.as_ref()).is_some_and(|heap| heap.heap_type() == HeapType::Placement);
        self.heap_place()
            .filter(|_| placement)
            .map_or(0, |place| place.offset)
    }

    /// `allocatedSize`, for a texture made from an automatic or placement heap (or a view of one):
    /// the bytes its range of the heap takes, the size [`Device::heap_texture_size_and_align`]
    /// answers for its descriptor, which placed it, with no query. Metal answers so (an M3 Max
    /// under macOS 27.0.1, plain and validated, in both kinds of heap): a heap texture's
    /// `allocatedSize` is `heapTextureSizeAndAlignWithDescriptor:`'s size, a view's is its
    /// texture's, and a texture buffer's is its row of texels. So it is Metal's where the size
    /// query's answer is (larger where the device's image needs more, as that answer is).
    ///
    /// `None` for any other texture: outside a heap, Metal answers its GPU's own allocation (the
    /// layout rounded up to 16-KiB pages past 16 KiB), which the runtime does not model, and a
    /// sparse heap's texture's answer is not measured.
    pub fn allocated_size(&self) -> Option<u64> {
        if let Some(buffer) = self.heap_texels() {
            return Some(buffer.length());
        }
        self.0.heap.as_ref()?;
        self.heap_place().map(|place| place.size)
    }

    /// `makeAliasable`: a heap texture's bytes may be another resource's from now on. An
    /// automatic heap gives its range to the next resource it places; a placement heap's texture
    /// is aliasable from the start, so nothing changes. Using the texture after is the program's
    /// error, as in Metal.
    ///
    /// A texture outside a heap stays as it is, its texels kept and not aliasable, as plain Metal
    /// leaves a `Shared` and a `Private` one where its validation stops the call. Refused for a
    /// sparse heap's texture (`NotImplemented(SparseHeaps)`, not measured).
    pub fn make_aliasable(&self) -> Result<(), Error> {
        if let (Some(place), Some(heap)) = (self.heap_place(), &self.0.heap) {
            heap.make_aliasable(place.offset, place.size, &place.aliasable);
            return Ok(());
        }
        if let Some(buffer) = self.heap_texels() {
            return buffer.make_aliasable();
        }
        if self.sparse_image().is_some() {
            return Err(Error::NotImplemented(RuntimeGap::SparseHeaps));
        }
        Ok(())
    }

    /// `isAliasable`: whether `make_aliasable` was called on an automatic heap's texture (or a
    /// texture it views); true for a placement heap's texture and its views from the start, and
    /// false otherwise, as Metal answers.
    pub fn is_aliasable(&self) -> bool {
        if let Some(buffer) = self.heap_texels() {
            return buffer.is_aliasable();
        }
        self.heap_place()
            .is_some_and(|place| place.aliasable.load(std::sync::atomic::Ordering::Acquire))
    }

    /// The buffer a heap's texture buffer (or a view of one) is the bytes of, placed in the heap
    /// for it (`texture/heap.rs`); `None` for any other texture, a texture over a buffer the
    /// program placed in a heap included.
    fn heap_texels(&self) -> Option<&Buffer> {
        match &self.0.storage {
            Storage::Texels(texels) if texels.home == TexelsHome::Heap => Some(&texels.buffer),
            Storage::Image(_)
            | Storage::Texels(_)
            | Storage::Alias { .. }
            | Storage::Shadow { .. } => None,
        }
    }

    /// What owns the memory the texture's texels lie in (`used_resources.rs`): its heap, the
    /// buffer it is over, or else the texture a view was made from, or the texture itself.
    pub(super) fn memory(&self) -> TextureMemory<'_> {
        if let Some(heap) = self.heap() {
            return TextureMemory::Heap(heap);
        }
        match &self.0.storage {
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => TextureMemory::Buffer(&texels.buffer),
            Storage::Image(_) => {
                let mut root = self;
                while let Some(parent) = root.parent_texture() {
                    root = parent;
                }
                TextureMemory::Own(root)
            }
        }
    }

    /// The key of the buffer whose bytes the texture's images and texel views lie in or are
    /// copied from ([`memory`](Self::memory)): the buffer it is over, or its heap's own buffer
    /// (a view's too). A rebind or a detach of those bytes remakes them, and no other texture's
    /// (`heap_sets::renew_over`). `None` for a texture in memory of its own or of a sparse heap.
    pub(super) fn bytes_key(&self) -> Option<ResourceKey> {
        match self.memory() {
            TextureMemory::Heap(heap) => heap.bytes().map(|bytes| bytes.identity().key),
            TextureMemory::Buffer(buffer) => Some(buffer.identity().key),
            TextureMemory::Own(_) => None,
        }
    }

    /// Another reference to the texture, which a command buffer holds it by
    /// (`EncoderCore::retain_shared`).
    pub(super) fn held(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    /// The texture's address: no other object's while it lives.
    pub(super) fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`).
    pub(super) fn held_by(&self) -> &HeldBy {
        &self.0.held_by
    }

    /// The key the texture's uses name (`bound_use`): its image's, or the bytes' of the buffer
    /// it is over, where its image aliases them or it has none.
    pub(super) fn use_key(&self) -> ResourceKey {
        self.bound_use(Access::NONE, Access::NONE).key
    }

    /// Refused (`NotImplemented(DetachedTexelAccess)`) while the bytes the texture is over (its
    /// root's, for a view) are detached (`detach.rs`): Metal's CPU access reaches the pages the
    /// texture took as it was made, which the runtime does not keep.
    fn refuse_detached_texels(&self) -> Result<(), Error> {
        let root = self.root();
        let detached = match &root.0.storage {
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => texels.buffer.detached(),
            Storage::Image(_) => (root.heap_place())
                .and(root.0.heap.as_ref())
                .and_then(Heap::bytes)
                .is_some_and(Buffer::detached),
        };
        match detached {
            true => Err(Error::NotImplemented(RuntimeGap::DetachedTexelAccess)),
            false => Ok(()),
        }
    }

    /// The texture's image and the buffer whose bytes it is over (its root's, for a view),
    /// where it has both: the buffer an aliasing or shadowed image is over, or the heap's own
    /// buffer for a heap's texture whose image is placed in the heap (or shadows its bytes
    /// there); and whether a stand-in of zero bytes presents the image exactly (not one of
    /// texels its copies convert from the bytes, `stand_in.rs`).
    fn image_over_bytes(&self) -> Option<(&Buffer, &Arc<Image>, bool)> {
        let (image, exact) = match &self.0.storage {
            Storage::Alias { image, .. } | Storage::Image(image) => (image, true),
            Storage::Shadow { image, copies, .. } => (image, !copies.converts()),
            Storage::Texels(_) => return None,
        };
        let root = self.root();
        let bytes = match &root.0.storage {
            Storage::Alias { texels, .. } | Storage::Shadow { texels, .. } => &texels.buffer,
            Storage::Image(_) => root.heap_place().and(root.0.heap.as_ref())?.bytes()?,
            Storage::Texels(_) => return None,
        };
        Some((bytes, image, exact))
    }

    /// Where an automatic or placement heap's texture (or a view of one) lies.
    fn heap_place(&self) -> Option<&heap::HeapPlace> {
        self.0.storage.heap_place()
    }

    /// `parentTexture`: the texture a view was made from (a view of a view has that view).
    pub fn parent_texture(&self) -> Option<&Texture> {
        self.0.parent.as_ref()
    }

    /// `parentRelativeLevel`: the view's first level among its parent's (0 for a texture).
    pub fn parent_relative_level(&self) -> u64 {
        self.0.parent.as_ref().map_or(0, |parent| {
            u64::from(self.0.record.base_level - parent.0.record.base_level)
        })
    }

    /// `parentRelativeSlice`: the view's first slice among its parent's (0 for a texture).
    pub fn parent_relative_slice(&self) -> u64 {
        self.0.parent.as_ref().map_or(0, |parent| {
            u64::from(self.0.record.base_layer - parent.0.record.base_layer)
        })
    }

    /// `newTextureViewWithPixelFormat:`: a view of the whole texture as `pixel_format`.
    pub fn new_texture_view(&self, pixel_format: PixelFormat) -> Result<Texture, Error> {
        let request = ViewRequest::of_format(&self.0.record, pixel_format);
        self.view("new_texture_view", &request, IdPlacement::Any)
    }

    /// `newTextureViewWithPixelFormat:resourceIndex:` (Apple's private call, a runtime
    /// extension): as [`new_texture_view`](Self::new_texture_view), the view's resource ID at
    /// `placement` (`resource_ids.rs`). Refused as that is, and as [`Device::new_texture`]
    /// refuses a placement.
    pub fn new_texture_view_with_placement(
        &self,
        pixel_format: PixelFormat,
        placement: IdPlacement,
    ) -> Result<Texture, Error> {
        let request = ViewRequest::of_format(&self.0.record, pixel_format);
        self.view("new_texture_view_with_placement", &request, placement)
    }

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:`: a view of `levels` and
    /// `slices` (counted from this texture's own first) as `pixel_format` and `texture_type`.
    pub fn new_texture_view_with_levels_slices(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u64>,
        slices: Range<u64>,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture_view_with_levels_slices";
        let request = self.subview(CALL, pixel_format, texture_type, levels, slices, None)?;
        self.view(CALL, &request, IdPlacement::Any)
    }

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`: as
    /// [`new_texture_view_with_levels_slices`](Self::new_texture_view_with_levels_slices), with
    /// `swizzle` replacing the texture's.
    pub fn new_texture_view_with_swizzle(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u64>,
        slices: Range<u64>,
        swizzle: TextureSwizzleChannels,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture_view_with_swizzle";
        let request = self.subview(
            CALL,
            pixel_format,
            texture_type,
            levels,
            slices,
            Some(swizzle),
        )?;
        self.view(CALL, &request, IdPlacement::Any)
    }

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:` and, with a
    /// `swizzle`, `...:swizzle:resourceIndex:` (Apple's private calls, a runtime extension): as
    /// [`new_texture_view_with_swizzle`](Self::new_texture_view_with_swizzle), the texture's own
    /// swizzle where `swizzle` is `None`, the view's resource ID at `placement`
    /// (`resource_ids.rs`). Refused as that is, and as [`Device::new_texture`] refuses a
    /// placement.
    pub fn new_texture_subview_with_placement(
        &self,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u64>,
        slices: Range<u64>,
        swizzle: Option<TextureSwizzleChannels>,
        placement: IdPlacement,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture_subview_with_placement";
        let request = self.subview(CALL, pixel_format, texture_type, levels, slices, swizzle)?;
        self.view(CALL, &request, placement)
    }

    /// The request of a view of `levels` and `slices` as `pixel_format` and `texture_type`, with
    /// `swizzle` replacing the texture's when given, for `call`.
    fn subview(
        &self,
        call: &'static str,
        pixel_format: PixelFormat,
        texture_type: TextureType,
        levels: Range<u64>,
        slices: Range<u64>,
        swizzle: Option<TextureSwizzleChannels>,
    ) -> Result<ViewRequest, Error> {
        use super::view_plan::ViewProblem;
        self.check_bytes_reached(call)?;
        let levels = narrow_range(&levels)
            .ok_or_else(|| invalid(call, Problem::TextureView(ViewProblem::LevelRange)))?;
        let slices = narrow_range(&slices)
            .ok_or_else(|| invalid(call, Problem::TextureView(ViewProblem::SliceRange)))?;
        Ok(ViewRequest {
            pixel_format,
            texture_type,
            levels,
            slices,
            swizzle,
        })
    }

    fn view(
        &self,
        call: &'static str,
        request: &ViewRequest,
        placement: IdPlacement,
    ) -> Result<Texture, Error> {
        self.check_bytes_reached(call)?;
        self.device().check_texture_placement(
            call,
            placement,
            self.0.record.usage,
            request.texture_type,
        )?;
        let profile = self.device().compile_device().profile();
        let record = check_view(&self.0.record, request, profile)
            .map_err(|refusal| view_error(call, refusal))?;
        // A view of a rotated texture acts as the texture does in its own format; only the views
        // measured are made (`texture/rotation.rs`).
        if record.presented.rotation != 0 && !rotation::view_measured(&self.0.record, request) {
            return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
        }
        // A texture made in `X32_Stencil8` holds a stencil texture's bytes: plain Metal views it
        // as `Depth32Float_Stencil8` over them in its GPU's own layout, no defined answer (an M4
        // under macOS 27.0.1), and stops every other format ("not castable"). Only its own format
        // views it.
        let own_x32 = matches!(&self.0.storage,
            Storage::Image(image) if image.plan().pixel_format == PixelFormat::X32Stencil8);
        if own_x32 && record.pixel_format != PixelFormat::X32Stencil8 {
            use super::view_plan::ViewProblem;
            return Err(invalid(
                call,
                Problem::TextureView(ViewProblem::FormatIncompatible),
            ));
        }
        // Metal stops a view of more than one slice of an array over a buffer, plain and
        // validated.
        if self.over_buffer() && record.layer_count > 1 {
            use super::view_plan::ViewProblem;
            return Err(invalid(
                call,
                Problem::TextureView(ViewProblem::BufferSlices),
            ));
        }
        // A carrier's decoded twin holds its own blocks' decode alone: a view of another decode
        // (an ASTC format's HDR form or another footprint, another PVRTC block, an EAC 11-bit
        // format's other signedness) reads no twin (`astc.rs`, `pvrtc.rs`, `etc.rs`; a PVRTC,
        // ETC2 or EAC format's sRGB form's codes are its linear form's). A view of an ASTC
        // texture in its other colour space reads the carrier's second twin, made now where it
        // has none (`texture/decoded_twin.rs`).
        if let Storage::Image(image) = &self.0.storage {
            let truth = view_shadow::truth(image);
            let pf = record.pixel_format;
            let astc = super::astc::decode_of(truth.plan()).map(|held| (held, AstcDecode::of(pf)));
            let other_space = matches!(astc, Some((held, Some(view)))
                if view.srgb != held.srgb && [view.width, view.height] == [held.width, held.height]);
            let astc = astc.is_some_and(|(held, view)| view != Some(held)) && !other_space;
            let pvrtc = super::pvrtc::decode_of(truth.plan())
                .is_some_and(|held| PvrtcDecode::of(pf).map(PvrtcDecode::linear) != Some(held));
            let etc = super::etc::decode_of(truth.plan())
                .is_some_and(|held| EtcDecode::of(pf) != Some(held));
            if astc || pvrtc || etc {
                return Err(Error::Unsupported(Unsupported::not_implemented(
                    Feature::TextureView,
                )));
            }
            if other_space {
                truth.make_second_twin(pf)?;
            }
        }
        // A view shares its texture's storage: its image, its buffer's bytes, and a shadowed
        // texture's copies, so its uses are its texture's. Where the image takes no Vulkan view
        // of the view's format (the device reinterprets no image's components, or the texture's
        // image was made without `PixelFormatView` usage for other formats), the view has storage
        // of its own (`texture/view_shadow.rs`), but for one in other channels of its image's
        // bits, which its bindings take as the image in its own format through a swizzle and a
        // pass renders through a shadow; a binding's own view is checked again where it is made. A texture buffer's view is a
        // texel buffer view of the same bytes, which takes any format of its class.
        let apart = match &self.0.storage {
            Storage::Texels(_) => false,
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => {
                !self.device().image_takes_view(image, record.pixel_format)
            }
        };
        let mut record = record;
        // A two-plane YCbCr texture's view as another type reads its luma codes alone
        // (`texture/ycbcr_image.rs`).
        let luma = self.luma_view(call, &record).transpose()?;
        let storage = if let Some((storage, read_as)) = luma {
            record.read_as = Some(read_as);
            storage
        } else if !apart {
            self.0.storage.clone()
        } else if let Some((image, read_as)) = self.read_through(&record) {
            self.device().shared().counts.view_through_swizzle_made();
            record.read_as = Some(read_as);
            Storage::Image(image)
        } else {
            self.reinterpreted(call, &record)?
        };
        // A view of a texture with a decoded twin that does not present it reads the blocks.
        self.degrade_view_decode(&storage, &record);
        let parent = Some(self.clone());
        let id_use = IdUse::Program(placement);
        Ok(Texture::made(
            self.device(),
            storage,
            record,
            parent,
            id_use,
        ))
    }

    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`: write `region`
    /// of the texture's `level` and `slice` from `bytes`, rows `bytes_per_row` apart and images
    /// `bytes_per_image` apart (read only for a region of more than one image).
    ///
    /// A region with a size of 0 on any axis writes nothing, as plain Metal does (its validation
    /// stops it), a compressed region off its blocks writes the blocks it starts in, and a
    /// `bytes_per_row` off Metal's rules is taken as plain Metal takes it: a pitch of a row or more
    /// exactly, 0 as every row at the image's first byte, and one between 0 and a row writes
    /// nothing. A `bytes_per_image` off Metal's rules, off the pixel or below the region's rows, is
    /// taken exactly, as plain Metal takes it. A combined depth/stencil format writes its depth, 4
    /// bytes a texel, its stencil left as it is, and its stencil view (`X32Stencil8`) the
    /// stencil, a byte a texel, its depth left as it is, as plain Metal does (its validation stops
    /// both). Refused: what Metal stops (`Invalid`: with a [`CopyProblem`], a private texture but
    /// an IOSurface plane's over a buffer the CPU reaches, whose bytes it reads and writes as plain
    /// Metal does, a region outside the level, a `bytes_per_row` off Metal's rules of a texture
    /// over a buffer or of 0 for more than one row and image (not measured), `bytes` shorter than
    /// the layout reads).
    ///
    /// A PVRTC texture's whole level takes its blocks in plain Metal's CPU order, Z order with
    /// the y bit lowest, at a `bytes_per_row` of 0 or of the level's row of blocks and above,
    /// where plain Metal moves the level's bytes contiguously and ignores the pitch
    /// (`pvrtc::cpu_order_position`). Refused `NotImplemented(PvrtcBlockOrder)`: a region other
    /// than a whole level, whose blocks plain Metal does not place by the region at all (it
    /// ignores the origin, reads blocks outside the region, and writes past the bytes the region
    /// accounts for), and a nonzero `bytes_per_row` below the level's row, where the driver
    /// asserts and the call moves nothing.
    ///
    /// 32-bit float depth is copied as Metal holds it, any `float`, on every device: without
    /// `VK_EXT_depth_range_unrestricted` Vulkan leaves a copied value outside `[0, 1]` undefined
    /// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and a device may store such a value clamped:
    /// an upload into a depth image there announces `restricted-depth-upload` unless the device's
    /// driver is measured to keep it (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`). A
    /// texture with a depth twin keeps it: its image is `R32_SFLOAT`, and its twin, which its
    /// comparisons read, takes the same bytes.
    pub fn replace_region(
        &self,
        region: Region,
        level: u64,
        slice: u64,
        bytes: &[u8],
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<(), Error> {
        let upload = Upload::Slice(bytes);
        self.replace(region, level, slice, upload, bytes_per_row, bytes_per_image)
    }

    /// [`replace_region`](Self::replace_region) from bytes at a pointer (a runtime extension,
    /// the same selector): memory that may change while the call reads it (a guest's), which the
    /// call copies through the pointer, each byte at most once, into memory of its own
    /// ([`HostBytes`]), with no copy of the caller's first. The rows of a texture whose image the
    /// copy reaches through a staging buffer (most) go straight into it; the other forms (a
    /// PVRTC level, texels over a buffer or a heap, a two-plane texture, a depth texture's checks,
    /// a shadowed view) read the bytes once into a copy, then as `replace_region` reads a slice.
    /// Refused as `replace_region`.
    pub fn replace_region_with_host_bytes(
        &self,
        region: Region,
        level: u64,
        slice: u64,
        bytes: HostBytes<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<(), Error> {
        let upload = Upload::Host {
            bytes,
            copy: OnceCell::new(),
        };
        self.replace(region, level, slice, upload, bytes_per_row, bytes_per_image)
    }

    /// `replace_region` from `bytes`, whichever form (the two above).
    fn replace(
        &self,
        region: Region,
        level: u64,
        slice: u64,
        bytes: Upload<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "replace_region";
        let Some(copy) =
            self.host_copy(CALL, &region, level, slice, bytes_per_row, bytes_per_image)?
        else {
            return Ok(());
        };
        self.refuse_detached_texels()?;
        if copy
            .host
            .host_len()
            .is_none_or(|len| (bytes.len() as u64) < len)
        {
            return Err(invalid(CALL, Problem::Copy(CopyProblem::BufferRange)));
        }
        let HostCopy {
            region: image_region,
            host,
            cpu_order,
            planes,
            ..
        } = copy.first_image();
        // A PVRTC level's blocks in Metal's CPU order, as the rows the copy packs.
        let rows;
        let bytes = match cpu_order {
            Some(blocks) => {
                rows = rows_of_cpu_order(bytes.slice(), blocks);
                Upload::Slice(&rows)
            }
            None => bytes,
        };
        let image = match &self.0.storage {
            Storage::Image(image) => image,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => {
                return texels.write(CALL, &image_region, &host, bytes.slice())
            }
        };
        // A heap shadow's texels are the heap's bytes (but a two-plane texture's later levels').
        if let Owner::HeapShadow(shadow) = &image.owner {
            if shadow.holds(image.plan(), &image_region) {
                return shadow.write(CALL, image.plan(), &image_region, &host, bytes.slice());
            }
        }
        // A two-plane texture's color words over its texels as they are, the alpha words kept.
        if planes {
            let mut words = vec![0; host.packed_len() as usize];
            bytes.pack(&host, &mut words);
            return self.write_color_words(image, &image_region, &words);
        }
        // Bytes that hold a negative or NaN depth, which a depth test compares otherwise.
        let signed = image_region.aspect == ImageAspectFlags::DEPTH
            && host.holds_negative_or_nan(bytes.slice());
        self.degrade_depth_upload(CALL, image_region.aspect, signed, false);
        // Bytes whose magnitude stand-in is not themselves (-0, a negative denormal, and, where the
        // device's depth test compares a denormal as zero, a positive one) switch a render
        // target's depth on too (`texture/signed_depth.rs`): none of them is announced as a sign.
        if image_region.aspect == ImageAspectFlags::DEPTH
            && !signed
            && self.usage().contains(TextureUsage::RENDER_TARGET)
        {
            let flush = signed_depth::test_flushes_denormals(self.device());
            if host.holds_unlike_its_stand_in(bytes.slice(), flush) {
                self.switch_on_signed_depth(CALL, false);
            }
        }
        // Bytes that hold a denormal depth, announced as the copy below stores them.
        let denormal =
            image_region.aspect == ImageAspectFlags::DEPTH && host.holds_denormal(bytes.slice());
        // A shadowed view's texels are its texture's (`texture/view_shadow.rs`).
        let truth = view_shadow::truth(image);
        let copy = buffer_image_copy(&image_region);
        // The caller's bytes packed, where a copy or a decoded twin's decode takes them so.
        let len = host.packed_len() as usize;
        let pack = || {
            let mut packed = vec![0; len];
            bytes.pack(&host, &mut packed);
            packed
        };
        // Copied on the host from the caller's bytes where they are laid out packed and held as
        // a slice, else from packed bytes (host bytes read once, through the pointer).
        if truth.takes_host_copies() {
            let packed = match (host.is_packed(), bytes.held()) {
                (true, Some(held)) => Cow::Borrowed(&held[..len]),
                _ => Cow::Owned(pack()),
            };
            if truth.host_write(copy, &packed)? {
                if denormal {
                    self.degrade_depth_denormals(CALL, image_region.aspect, true, false);
                }
                return truth.decode_written(&image_region, || packed.into_owned());
            }
        }
        if denormal {
            self.degrade_depth_denormals(CALL, image_region.aspect, false, false);
        }
        // Packed into the staging buffer from the caller's bytes.
        if truth.takes_staging_copies()
            && truth.staging_write(copy, len as u64, |staged| bytes.pack(&host, staged))?
        {
            // A decoded twin's decode, staged; on the CPU while a batch may read it.
            return truth.decode_written(&image_region, pack);
        }
        let device = self.device();
        if truth.defers_copies() {
            let staging = device.own_buffer(host.packed_len(), ResourceOptions::shared())?;
            let contents = (staging.contents())
                .ok_or_else(|| internal("a Shared staging buffer without contents"))?;
            // SAFETY: the staging buffer's bytes, at least `packed_len` from `contents`, in
            // host-coherent memory that nothing else uses yet.
            let packed = unsafe {
                std::slice::from_raw_parts_mut(contents.as_ptr(), host.packed_len() as usize)
            };
            bytes.pack(&host, packed);
            let (source, start) = staging.at(0);
            let copy = vk::BufferImageCopy {
                buffer_offset: copy.buffer_offset + start,
                ..copy
            };
            let mut copies = vec![(truth.raw.get(), copy)];
            // The depth twin takes the same texels.
            if let Some(twin) = self.twin_image() {
                copies.push((twin, twin_copy(copy)));
            }
            let staged = TexelCopy::new(source, host.packed_len(), copies, Box::new(staging));
            // The decoded twin's texels of the region, after the copy.
            let regions = std::slice::from_ref(&image_region);
            let staged = match truth.decode_pass(regions, PassScratch::Device)? {
                Some(pass) => staged.decoding(pass),
                None => staged,
            };
            let keeper: Weak<Image> = Arc::downgrade(truth);
            let firsts = &device.shared().textures.first_moves;
            // Past the bound, the copies staged so far are made now, by a submission waited for.
            if firsts.register_texels(keeper, staged) {
                truth.submit_copy(|_, _| {})?;
            }
            // An ASTC carrier's second twin's decode, for the next batch, after the copy.
            return truth.queue_second_decode(regions);
        }
        let mut staging = Staging::new(device, host.packed_len())?;
        bytes.pack(&host, staging.bytes());
        let (buffer, twin, image) = (staging.raw, self.twin_image(), truth.raw.get());
        // The decoded twin's texels of the region, after the copy; the submission is waited for.
        let decode = truth.decode_pass(std::slice::from_ref(&image_region), PassScratch::Own)?;
        truth.submit_copy(|raw, command_buffer| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the internal command buffer is recording; the staging buffer and the images
            // are live, and the images are in the `GENERAL` layout.
            unsafe {
                raw.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &[copy],
                );
                // The depth twin takes the same texels.
                if let Some(twin) = twin {
                    raw.cmd_copy_buffer_to_image(
                        command_buffer,
                        buffer,
                        twin,
                        vk::ImageLayout::GENERAL,
                        &[twin_copy(copy)],
                    );
                }
            }
            transfer_barrier(raw, command_buffer, false);
            if let Some(decode) = &decode {
                // SAFETY: as above; the pass lives until the submission, waited for, completed.
                unsafe { decode.record(raw, command_buffer) };
            }
        })?;
        // An ASTC carrier's second twin's decode, for the next batch.
        truth.queue_second_decode(std::slice::from_ref(&image_region))
    }

    /// `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:`: read `region` of
    /// the texture's `level` and `slice` into `bytes`, rows `bytes_per_row` apart and images
    /// `bytes_per_image` apart (read only for a region of more than one image). The bytes
    /// between rows and images are left as they are; with a `bytes_per_row` of 0 each row is
    /// written at the image's first byte in turn, so the last one stays, as plain Metal writes
    /// them, and images closer than a region's rows are written in turn, the later image's bytes
    /// staying. A combined depth/stencil format reads its depth, and its stencil view
    /// (`X32Stencil8`) the stencil, a byte a texel. A PVRTC texture's whole level gives its blocks
    /// in plain Metal's CPU order.
    ///
    /// Refused as [`replace_region`](Self::replace_region) is.
    pub fn get_bytes(
        &self,
        bytes: &mut [u8],
        bytes_per_row: u64,
        bytes_per_image: u64,
        region: Region,
        level: u64,
        slice: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "get_bytes";
        let Some(copy) =
            self.host_copy(CALL, &region, level, slice, bytes_per_row, bytes_per_image)?
        else {
            return Ok(());
        };
        self.refuse_detached_texels()?;
        if copy
            .host
            .host_len()
            .is_none_or(|len| (bytes.len() as u64) < len)
        {
            return Err(invalid(CALL, Problem::Copy(CopyProblem::BufferRange)));
        }
        let HostCopy {
            region: image_region,
            host,
            cpu_order,
            planes,
            ..
        } = copy.first_image();
        // A two-plane texture's color words, each texel's first word; those a heap shadow holds
        // are the heap's bytes, which `read_region` reads (as `replace_region` writes them), and
        // a level it does not hold (a mipmapped texture's later one) lies in the image alone.
        let heap_shadow = matches!(
            &self.0.storage,
            Storage::Image(image) if matches!(
                &image.owner,
                Owner::HeapShadow(shadow) if shadow.holds(image.plan(), &image_region)
            )
        );
        if planes && !heap_shadow {
            let mut words = vec![0; host.packed_len() as usize];
            self.read_color_words(&image_region, &mut words)?;
            host.unpack(&words, bytes);
            return Ok(());
        }
        // A PVRTC level's rows, read whole, then laid out in Metal's CPU order.
        let Some(blocks) = cpu_order else {
            return self.read_region(CALL, &image_region, &host, bytes);
        };
        let mut rows = vec![0; host.packed_len() as usize];
        self.read_region(CALL, &image_region, &host, &mut rows)?;
        cpu_order_of_rows(&rows, blocks, bytes);
        Ok(())
    }

    /// A staging buffer of `region`'s texels of a two-plane texture's `image`, 8 bytes each,
    /// holding `words` color words (`two_planes.rs`).
    fn color_texels(
        &self,
        image: &Arc<Image>,
        region: &ImageRegion,
        words: u64,
    ) -> Result<Staging<'_>, Error> {
        let device = self.device();
        let staging = Staging::new(device, 2 * words)?;
        // A shadowed view's texels are its texture's (`texture/view_shadow.rs`).
        let truth = view_shadow::truth(image);
        let (buffer, raw) = (staging.raw, truth.raw.get());
        let copy = buffer_image_copy(region);
        truth.submit_copy(|commands, command_buffer| {
            transfer_barrier(commands, command_buffer, true);
            // SAFETY: the internal command buffer is recording; the staging buffer and the image
            // are live, and the image is in the `GENERAL` layout.
            unsafe {
                commands.cmd_copy_image_to_buffer(
                    command_buffer,
                    raw,
                    vk::ImageLayout::GENERAL,
                    buffer,
                    &[copy],
                );
            }
            transfer_barrier(commands, command_buffer, false);
        })?;
        Ok(staging)
    }

    /// The color words of `region` of a two-plane texture's image into `words`, packed 4 bytes a
    /// texel: each texel's first word.
    ///
    /// Read on the host where the image copies there (`texture/host_copy.rs`), with no submission
    /// and no wait, as Metal's `getBytes` of such a texture waits for no GPU work; else through a
    /// staging buffer, by a submission waited for.
    fn read_color_words(&self, region: &ImageRegion, words: &mut [u8]) -> Result<(), Error> {
        let Storage::Image(image) = &self.0.storage else {
            return Err(internal("a two-plane texture's copy without its image"));
        };
        // A shadowed view's texels are its texture's (`texture/view_shadow.rs`).
        let truth = view_shadow::truth(image);
        if truth.takes_host_copies() {
            let mut texels = vec![0; 2 * words.len()];
            if truth.host_read(buffer_image_copy(region), &mut texels)? {
                two_planes::narrow(&texels, words);
                return Ok(());
            }
        }
        let mut staging = self.color_texels(image, region, words.len() as u64)?;
        two_planes::narrow(staging.bytes(), words);
        Ok(())
    }

    /// `words`, packed 4 bytes a texel, over the first word of each texel of `region` of a
    /// two-plane texture's `image`: its texels read first and written back, the alpha words as
    /// they were.
    ///
    /// Read and written back on the host where the image copies there (`texture/host_copy.rs`),
    /// with no submission and no wait, as Metal's `replaceRegion` of such a texture waits for no
    /// GPU work; else through a staging buffer, by submissions waited for.
    fn write_color_words(
        &self,
        image: &Arc<Image>,
        region: &ImageRegion,
        words: &[u8],
    ) -> Result<(), Error> {
        // A shadowed view's texels are its texture's (`texture/view_shadow.rs`).
        let truth = view_shadow::truth(image);
        let copy = buffer_image_copy(region);
        if truth.takes_host_copies() {
            let mut texels = vec![0; 2 * words.len()];
            if truth.host_read(copy, &mut texels)? {
                two_planes::widen(words, &mut texels);
                // A write the image declines (it presents a stand-in now, its texture's bytes
                // detached meanwhile) goes through the staging buffer below, which reads its
                // texels again.
                if truth.host_write(copy, &texels)? {
                    return Ok(());
                }
            }
        }
        let mut staging = self.color_texels(image, region, words.len() as u64)?;
        two_planes::widen(words, staging.bytes());
        let (buffer, raw) = (staging.raw, truth.raw.get());
        truth.submit_copy(|commands, command_buffer| {
            transfer_barrier(commands, command_buffer, true);
            // SAFETY: the internal command buffer is recording; the staging buffer and the image
            // are live, and the image is in the `GENERAL` layout.
            unsafe {
                commands.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    raw,
                    vk::ImageLayout::GENERAL,
                    &[copy],
                );
            }
            transfer_barrier(commands, command_buffer, false);
        })
    }

    /// `region` of the texture's image into `bytes` laid out as `host` (checked by `get_bytes`).
    fn read_region(
        &self,
        call: &'static str,
        image_region: &ImageRegion,
        host: &HostLayout,
        bytes: &mut [u8],
    ) -> Result<(), Error> {
        let image = match &self.0.storage {
            Storage::Image(image) => image,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => {
                return texels.read(call, image_region, host, bytes)
            }
        };
        // A heap shadow's texels are the heap's bytes (but a two-plane texture's later levels').
        if let Owner::HeapShadow(shadow) = &image.owner {
            if shadow.holds(image.plan(), image_region) {
                return shadow.read(call, image.plan(), image_region, host, bytes);
            }
        }
        // A shadowed view's texels are its texture's (`texture/view_shadow.rs`).
        let truth = view_shadow::truth(image);
        let copy = buffer_image_copy(image_region);
        // Copied on the host into the caller's bytes where they are laid out packed, else into
        // packed bytes laid out from there.
        if truth.takes_host_copies() {
            let read = match host.is_packed() {
                true => truth.host_read(copy, &mut bytes[..host.packed_len() as usize])?,
                false => {
                    let mut packed = vec![0; host.packed_len() as usize];
                    let read = truth.host_read(copy, &mut packed)?;
                    if read {
                        host.unpack(&packed, bytes);
                    }
                    read
                }
            };
            if read {
                return Ok(());
            }
        }
        // Laid out from the staging buffer.
        if truth.takes_staging_copies()
            && truth.staging_read(copy, host.packed_len(), |staged| host.unpack(staged, bytes))?
        {
            return Ok(());
        }
        let device = self.device();
        let mut staging = Staging::new(device, host.packed_len())?;
        let (buffer, image) = (staging.raw, truth.raw.get());
        truth.submit_copy(|raw, command_buffer| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the internal command buffer is recording; the staging buffer and the image
            // are live, and the image is in the `GENERAL` layout.
            unsafe {
                raw.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    buffer,
                    &[copy],
                );
            }
            transfer_barrier(raw, command_buffer, false);
        })?;
        host.unpack(staging.bytes(), bytes);
        Ok(())
    }

    /// Metal's checks of a CPU copy of `region` of `level` and `slice`, and its plan; `None` for
    /// a region with a size of 0 on any axis, whatever the texture's type, which copies nothing
    /// as plain Metal does.
    fn host_copy(
        &self,
        call: &'static str,
        region: &Region,
        level: u64,
        slice: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<Option<HostCopy>, Error> {
        let record = &self.0.record;
        self.check_bytes_reached(call)?;
        // A texture over a buffer the CPU reaches has its bytes there whatever its storage mode:
        // a `Private` IOSurface plane's `getBytes` and `replaceRegion` read and write the
        // surface's bytes on plain Metal. A texture over a `Private` buffer takes
        // its storage mode, and is refused as a private texture is.
        let reached = match &self.0.storage {
            Storage::Image(_) => false,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => texels.buffer.contents().is_some(),
        };
        if self.storage_mode() == StorageMode::Private && !reached {
            return Err(invalid(call, Problem::Copy(CopyProblem::CpuAccessPrivate)));
        }
        let combined = |pf: PixelFormat| {
            aspects(pf)
                .is_some_and(|a| a.contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL))
        };
        let (unit, over_buffer) = match &self.0.storage {
            // A two-plane texture's CPU copies move plane 0's color words (`two_planes.rs`); a
            // heap's lies apart from the heap's bytes (`texture/heap.rs`).
            Storage::Image(_) if two_planes::holds_planes(record.pixel_format) => {
                (two_planes::UNIT, false)
            }
            Storage::Image(image) => {
                // The stencil view of a combined depth/stencil texture (`X32Stencil8` of
                // `Depth32FloatStencil8`, the one `new_texture_view` makes) moves the stencil, a
                // byte a texel, its depth kept, as its blits do (`copy_unit`): plain Metal's
                // answer on an M3 Max under macOS 27.0.1 (a probe; its validation stops the
                // CPU access). Another depth or stencil view of a combined texture is not
                // measured.
                let of = image.plan().pixel_format;
                let stencil_view = record.pixel_format == PixelFormat::X32Stencil8
                    && of == PixelFormat::Depth32FloatStencil8;
                if is_depth_stencil(record.pixel_format)
                    && !combined(record.pixel_format)
                    && combined(of)
                    && !stencil_view
                {
                    return Err(Error::NotImplemented(RuntimeGap::Textures));
                }
                // A combined depth/stencil format copies its depth, as a blit without an option
                // does: plain Metal's answer (its validation stops it).
                let unit = copy_unit(record.pixel_format, &image.plan().mapping, BlitOption::None)
                    .map_err(|refusal| copy_error(call, refusal))?;
                (unit, false)
            }
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => (texels.unit(), true),
        };
        let region = texture_region(region, level, slice)
            .ok_or_else(|| invalid(call, Problem::Copy(CopyProblem::Region)))?;
        if region.size.contains(&0) {
            return Ok(None);
        }
        // A packed two-plane YCbCr texture's CPU calls move nothing: Metal's `getBytes` leaves
        // the bytes and its `replaceRegion` the planes as they were, at any bytes per row.
        if ycbcr_format(record.pixel_format).is_some_and(|f| f.packed) {
            check_region(record, &region, &unit)
                .map_err(|problem| invalid(call, Problem::Copy(problem)))?;
            return Ok(None);
        }
        // A PVRTC level's blocks across and down: the CPU copies a whole level in Metal's order,
        // through rows of them.
        let extent = level_extent(record, region.level);
        let cpu_order = is_pvrtc(record.pixel_format)
            .then(|| [0, 1].map(|axis| u64::from(extent[axis].div_ceil(unit.extent[axis]))));
        let pitch = cpu_order.map_or(bytes_per_row, |[across, _]| across * u64::from(unit.bytes));
        // Metal's answer for a two-plane texture is measured for rows and images a whole row
        // and image apart or more, by 4-byte multiples.
        let planes = matches!(self.0.storage, Storage::Image(_))
            && two_planes::holds_planes(record.pixel_format);
        if planes {
            let s = super::copy_plan::shape(region.size, &unit);
            let layout = HostLayout {
                row_bytes: s.row_bytes,
                rows: s.rows,
                images: s.images,
                bytes_per_row,
                bytes_per_image,
            };
            if !two_planes::measured_host_layout(&layout) {
                return Err(Error::Unsupported(two_planes::unmeasured(
                    record.pixel_format,
                )));
            }
        }
        // A private texture the CPU does not reach was refused above.
        let Some((planned, host)) = plan_host_copy(
            record,
            &region,
            &unit,
            pitch,
            bytes_per_image,
            false,
            over_buffer,
        )
        .map_err(|problem| invalid(call, Problem::Copy(problem)))?
        else {
            return Ok(None);
        };
        if cpu_order.is_some() {
            // A whole block of a level smaller than one is the whole level (plain Metal's bytes
            // on an Apple M3 Max under macOS 26.7, a probe; its validation stops it).
            let reaches = region.size.iter().zip(&extent).all(|(s, e)| s >= e);
            let whole = region.origin == [0; 3] && reaches;
            // A whole level takes any `bytesPerRow` of 0 or of the level's row of blocks and
            // above: plain Metal moves the level's bytes contiguously there, the pitch ignored.
            // Measured on this M3 Max (macOS 27.0.1, plain Metal, no validation layer, a 32x32
            // texture of both 4- and 2-bit formats, `replaceRegion` and `getBytes` alike, two
            // runs byte-identical): at the level's row, row + 8, twice the row, twice + 8 and
            // seven times it, every byte is the byte a pitch of 0 moves, so a padded pitch reads
            // and writes the padding bytes as block data. `pitch` above already substitutes the
            // level's own row for the caller's value, so the copy planned below moves exactly
            // those bytes. Below the level's row and nonzero the AGX driver logs
            // "Texture read/write assertion failed: bytes_per_row >= used_bytes_per_row" and the
            // call moves nothing, which stays refused rather than emulated.
            //
            // A region other than a whole level stays refused: plain Metal does not copy the
            // region's blocks at all. Measured in the same probe, the origin is ignored (a 2x2
            // block region at block (2,1) and at the origin answer identically), the block set
            // follows the region's extent and takes blocks outside the region (a left-half
            // request answered with the top half's blocks), and for an 8x7-block region
            // `getBytes` wrote 224 bytes past the 448 the region accounts for. There is no rule
            // mapping the call to its result that a consumer could use, and matching Metal would
            // mean writing past the caller's buffer.
            if !whole || (bytes_per_row != 0 && bytes_per_row < pitch) {
                return Err(Error::NotImplemented(RuntimeGap::PvrtcBlockOrder));
            }
        }
        // A 2-bit PVRTC texture's carrier halves its regions' x (`pvrtc.rs`); an ASTC texture's
        // carrier holds each block in 4 x 4 of its texels (`astc.rs`).
        let planned = match &self.0.storage {
            Storage::Image(image)
                if holds_blocks(image.plan()) || super::astc::decode_of(image.plan()).is_some() =>
            {
                image_region(image.plan(), planned)
            }
            _ => planned,
        };
        Ok(Some(HostCopy {
            region: planned,
            host,
            cpu_order,
            planes,
            volume: planes && record.texture_type == TextureType::Type3D,
        }))
    }
}

/// A CPU copy's plan (`Texture::host_copy`): the region of the texture's image, the host bytes'
/// layout, and for a PVRTC level the blocks across and down it holds, which the host bytes take
/// in plain Metal's CPU order (`pvrtc::cpu_order_position`) and `host` as rows.
struct HostCopy {
    region: ImageRegion,
    host: HostLayout,
    cpu_order: Option<[u64; 2]>,
    /// A two-plane extended-range texture's color words, its image's texels 8 bytes each
    /// (`two_planes.rs`).
    planes: bool,
    /// Such a texture is 3D: its copy moves the region's first image alone, at its first slice
    /// ([`HostCopy::first_image`]).
    volume: bool,
}

impl HostCopy {
    /// The copy of a two-plane 3D texture's region ([`HostCopy::volume`]), as Metal makes it (an
    /// Apple M3 Max under macOS 27.0.1, `Shared` and in a heap): `replaceRegion` writes the
    /// region's first image over texel `(x, y)` of the first slice whatever the region's `z`, and
    /// `getBytes` reads that slice's texels into the first image, the caller's other images as
    /// they were. The checks before it took the whole region.
    fn first_image(self) -> HostCopy {
        match self.volume {
            true => HostCopy {
                region: ImageRegion {
                    origin: [self.region.origin[0], self.region.origin[1], 0],
                    size: [self.region.size[0], self.region.size[1], 1],
                    ..self.region
                },
                host: HostLayout {
                    images: 1,
                    ..self.host
                },
                ..self
            },
            false => self,
        }
    }
}

/// What `replaceRegion` reads (`Texture::replace`): a slice, or bytes at a pointer
/// ([`HostBytes`]), read once: straight into the packed rows where nothing read them before,
/// else into `copy` the first time a form needs them whole, which every later read reads.
enum Upload<'a> {
    Slice(&'a [u8]),
    Host {
        bytes: HostBytes<'a>,
        copy: OnceCell<Vec<u8>>,
    },
}

impl Upload<'_> {
    /// How many bytes it holds.
    fn len(&self) -> usize {
        match self {
            Upload::Slice(bytes) => bytes.len(),
            Upload::Host { bytes, .. } => bytes.len(),
        }
    }

    /// Its bytes whole: host bytes read once, here, into a copy of the runtime's own.
    fn slice(&self) -> &[u8] {
        match self {
            Upload::Slice(bytes) => bytes,
            Upload::Host { bytes, copy } => copy.get_or_init(|| bytes.to_vec()),
        }
    }

    /// Its bytes as a slice where it holds one: a slice's, or host bytes once read into a copy;
    /// `None` for host bytes not read yet, which no reference may cover.
    fn held(&self) -> Option<&[u8]> {
        match self {
            Upload::Slice(bytes) => Some(bytes),
            Upload::Host { copy, .. } => copy.get().map(Vec::as_slice),
        }
    }

    /// `host`'s rows of it, packed into `packed`: host bytes not read yet straight from the
    /// pointer, each row once.
    fn pack(&self, host: &HostLayout, packed: &mut [u8]) {
        match self {
            Upload::Slice(bytes) => host.pack(bytes, packed),
            Upload::Host { bytes, copy } => match copy.get() {
                Some(copy) => host.pack(copy, packed),
                None => host.pack_from(*bytes, packed),
            },
        }
    }
}
