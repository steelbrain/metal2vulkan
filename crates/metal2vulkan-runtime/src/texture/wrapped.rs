//! Textures over images the caller made: `Device::new_texture_with_vk_image`, for example a
//! swapchain image of an adopted device.
//!
//! - **Ownership.** The caller made the `VkImage` and its memory and keeps both: the runtime never
//!   destroys or frees them ([`Owner::Caller`]). It makes and destroys only its own views of the
//!   image.
//! - **What the image is.** Vulkan has no query for an image's creation, so the caller states it:
//!   the `VkImageCreateInfo` it made the image with. The image must be what the runtime would make
//!   for the descriptor on this device ([`wrap_plan`]): the same type, format, extent, levels,
//!   layers and samples, optimal tiling, every usage and the cube compatibility the runtime's own
//!   texture of the descriptor would have (transfer usage among them where the format has it:
//!   Metal copies every texture with blits). Two the runtime gives against uses no texture can
//!   foresee may be missing, and a pass that needs one renders the texture through a stand-in
//!   that has it (`attachment_stand_in.rs`): input attachment usage, and the custom sample
//!   location flag of a multisampled depth render target. A contradiction is refused, naming it
//!   ([`WrappedImageProblem`]). An image that is not mutable-format is viewed in its own format
//!   only: a view or binding in another one (the sRGB twin Metal views without
//!   `PixelFormatView`) is refused when made.
//! - **The layout contract** ([`VkImageLayouts`]). The image is the caller's between the
//!   runtime's batches, in its `home` layout. Each batch of the runtime's submissions that uses
//!   the image moves it to `GENERAL` at its start, where every image of the runtime rests, and
//!   back to `home` at its end, before its signals; the first batch moves it from `initial` (its
//!   layout when wrapped) instead. A layout `GENERAL` on both sides makes no transition. The moves
//!   are writes the hazard tracker orders as every write ([`WrappedImage::transition_use`]), so a
//!   texture that moves needs tracked hazards. Storage is `Private`: the runtime gives no CPU access to the
//!   caller's memory. Once an image rests in `GENERAL` and its home is `GENERAL` (an `initial` of
//!   `UNDEFINED` or `PREINITIALIZED` moved by its first batch), no batch moves it again, and the
//!   scheduler forgets it.

use super::super::address::ObjectsToken;
use super::super::error::RuntimeGap;
use super::super::error::{invalid, Error, Problem};
use super::super::image_barrier::{image_barrier, ImageDst};
use super::super::image_plan::{compatible, plan_image, ImagePlan};
use super::super::key_hash::KeyMap;
use super::super::resource::{Identity, ResourceKey};
use super::super::resource_ids::IdPlacement;
use super::super::sharing::Sharing;
use super::super::texture_descriptor::{texture_usage, TextureDescriptor};
use super::super::tracker::{Access, Use, UseRange};
use super::super::values::StorageMode;
use super::super::view_plan::ViewRecord;
use super::super::Device;
use super::{image_aspects, image_error, IdUse, Image, KeptResolves, Owner, Storage, Texture};
use ash::vk;
use metal2vulkan::metal::{TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, ImageAspectFlags, ImageCreateFlags, ImageLayout, ImageTiling, ImageType,
    ImageUsageFlags, SampleCountFlags,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

/// No Metal counterpart (an interop hook): the layouts of an image the caller owns around the
/// runtime's work on it. The runtime finds the image in `initial` at the first batch that uses it,
/// in `home` at every later one, and leaves it in `home` after each.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkImageLayouts {
    /// The image's layout when wrapped. `UNDEFINED` (a new swapchain image's) lets the first use
    /// discard the contents; `PREINITIALIZED` keeps them, as every other layout does.
    pub initial: ImageLayout,
    /// The image's layout between the runtime's batches: `PRESENT_SRC_KHR` for a swapchain image
    /// the caller presents, `GENERAL` for one it leaves to the runtime.
    pub home: ImageLayout,
}

impl VkImageLayouts {
    /// The image is in `initial` now, and in `home` between the runtime's batches.
    pub fn new(initial: ImageLayout, home: ImageLayout) -> VkImageLayouts {
        VkImageLayouts { initial, home }
    }

    /// The image rests in `GENERAL`, as the runtime's own images do: no transition at all.
    pub fn general() -> VkImageLayouts {
        VkImageLayouts::new(ImageLayout::GENERAL, ImageLayout::GENERAL)
    }

    /// Whether any batch moves the image between layouts.
    pub(crate) fn moves(self) -> bool {
        self.initial != ImageLayout::GENERAL || self.home != ImageLayout::GENERAL
    }
}

/// Why an image the caller made cannot be wrapped as a texture of the descriptor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrappedImageProblem {
    /// A null image handle.
    NullImage,
    /// The descriptor's type has no image here (`TextureBuffer`), or the image's type is another.
    ImageType {
        /// The image's type.
        image: ImageType,
    },
    /// The image's format is not the one the runtime makes for the descriptor's pixel format on
    /// this device.
    Format {
        /// The image's format.
        image: Format,
        /// The format the runtime makes for the descriptor.
        texture: Format,
    },
    /// The image's extent is not the descriptor's.
    Extent {
        /// The image's extent.
        image: [u32; 3],
        /// The descriptor's extent.
        texture: [u32; 3],
    },
    /// The image's level count is not the descriptor's.
    Levels {
        /// The image's level count.
        image: u32,
        /// The descriptor's level count.
        texture: u32,
    },
    /// The image's array layers are not the descriptor's (slices, six per cube).
    Layers {
        /// The image's array layers.
        image: u32,
        /// The descriptor's layers.
        texture: u32,
    },
    /// The image's sample count is not the descriptor's.
    Samples {
        /// The image's sample count.
        image: SampleCountFlags,
        /// The descriptor's sample count.
        texture: SampleCountFlags,
    },
    /// The image is not optimal-tiled.
    Tiling {
        /// The image's tiling.
        image: ImageTiling,
    },
    /// The image lacks usages the runtime's own texture of the descriptor has, other than the
    /// input attachment usage only a fetching pass needs (that pass renders into a stand-in).
    Usage {
        /// The usages the image lacks.
        missing: ImageUsageFlags,
    },
    /// The image lacks creation flags the texture needs (cube compatibility, or extended usage
    /// for a storage view in another format).
    Flags {
        /// The creation flags the image lacks.
        missing: ImageCreateFlags,
    },
    /// A protected image: the runtime makes no protected submissions.
    Protected,
    /// An image not shared with every queue family of the device: a concurrent one whose
    /// families leave out one of the device's, or, on a device of several families, an exclusive
    /// one, as the runtime transfers no ownership between them.
    QueueFamily,
    /// A layout the contract cannot take: `UNDEFINED` or `PREINITIALIZED` as `home`, a layout
    /// the image's usage or aspects do not allow, or one the runtime does not move images
    /// through.
    Layout {
        /// The layout given.
        layout: ImageLayout,
    },
    /// A texture whose image moves between layouts made with `Untracked` hazards: the moves are
    /// writes only the tracker orders.
    Untracked,
    /// The plane asked for is not one of the image's: its format has `planes` planes (0 for a
    /// format that is not multi-planar), numbered from 0.
    Plane {
        /// The plane asked for.
        plane: u32,
        /// The planes of the image's format.
        planes: u32,
    },
    /// The descriptor's format is not compatible with the plane's (the plane's format's
    /// compatibility class, `VUID-VkImageViewCreateInfo-image-01586`).
    PlaneFormat {
        /// The plane's format, as the registry gives it.
        plane: Format,
        /// The format the runtime makes for the descriptor.
        texture: Format,
    },
}

/// The runtime's view of an image the caller owns: its key, and where its layout stands.
pub(crate) struct WrappedImage {
    /// What the hazard tracker knows the image by: its texture's key.
    pub(crate) key: ResourceKey,
    raw: vk::Image,
    aspects: ImageAspectFlags,
    levels: u32,
    layers: u32,
    home: ImageLayout,
    /// The image as its degrade notice names it: its handle, format and extent.
    object: String,
    /// The plane of a multi-planar image the texture presents
    /// (`Device::new_texture_with_vk_image_plane`): its views name this aspect where the runtime's
    /// name the color aspect; `None` for a whole image.
    pub(crate) plane: Option<ImageAspectFlags>,
    /// The layout the image is in once every batch submitted so far completed: `initial` until
    /// the first batch that uses it is submitted, `home` after. The scheduler's thread reads and
    /// writes it as it submits, under its lock.
    current: Mutex<ImageLayout>,
}

/// One layout transition of a caller's image, recorded as one `VkImageMemoryBarrier` over all of
/// it that orders everything before it before everything after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Transition {
    pub(crate) image: vk::Image,
    pub(crate) aspects: ImageAspectFlags,
    pub(crate) old: ImageLayout,
    pub(crate) new: ImageLayout,
}

impl Transition {
    /// The barrier recording the transition for a family of `family`'s queue flags, before every
    /// access of the device's there (`image_barrier.rs`).
    pub(crate) fn barrier(&self, family: vk::QueueFlags) -> vk::ImageMemoryBarrier<'static> {
        let range = vk::ImageSubresourceRange {
            aspect_mask: self.aspects.into(),
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        let layouts = (self.old.into(), self.new.into());
        let write = vk::AccessFlags::MEMORY_WRITE;
        image_barrier(self.image, range, layouts, write, ImageDst::Every, family)
    }
}

/// The transitions a batch makes of an image now in `current` whose home is `home`: into
/// `GENERAL` at the batch's start, and back home at its end.
fn moves(
    current: ImageLayout,
    home: ImageLayout,
) -> (
    Option<(ImageLayout, ImageLayout)>,
    Option<(ImageLayout, ImageLayout)>,
) {
    let general = ImageLayout::GENERAL;
    (
        (current != general).then_some((current, general)),
        (home != general).then_some((general, home)),
    )
}

impl WrappedImage {
    /// The image `raw` of the texture known by `key`, with `levels` and `layers` of `aspects`,
    /// in `layouts.initial` now; `object` names it in its degrade notice, and `plane` is the plane
    /// the texture presents of a multi-planar image.
    pub(crate) fn new(
        key: ResourceKey,
        raw: vk::Image,
        aspects: ImageAspectFlags,
        levels: u32,
        layers: u32,
        layouts: VkImageLayouts,
        object: String,
        plane: Option<ImageAspectFlags>,
    ) -> WrappedImage {
        WrappedImage {
            key,
            raw,
            aspects,
            levels,
            layers,
            home: layouts.home,
            object,
            plane,
            current: Mutex::new(layouts.initial),
        }
    }

    /// Whether the image rests out of `GENERAL` between batches: each batch that uses it moves
    /// it into `GENERAL` and back (`Degradation::WrappedImageLayoutTransitions`).
    pub(crate) fn rests_out_of_general(&self) -> bool {
        self.home != ImageLayout::GENERAL
    }

    /// The degrade notice's object and reason for an image that rests out of `GENERAL`.
    pub(crate) fn notice(&self) -> (String, String) {
        (
            self.object.clone(),
            format!(
                "its home layout is {:?}, not GENERAL: each batch that uses it moves it into \
                 GENERAL before its first encoder and back at its end, 2 whole-image barriers \
                 per batch",
                self.home
            ),
        )
    }

    /// The transitions a batch submitted now makes: at its start, and at its end.
    pub(crate) fn transitions(&self) -> (Option<Transition>, Option<Transition>) {
        let current = *self.current.lock().unwrap_or_else(|p| p.into_inner());
        let (entry, exit) = moves(current, self.home);
        let transition = |(old, new)| Transition {
            image: self.raw,
            aspects: self.aspects,
            old,
            new,
        };
        (entry.map(transition), exit.map(transition))
    }

    /// A batch that used the image was submitted: it leaves the image home.
    pub(crate) fn submitted(&self) {
        *self.current.lock().unwrap_or_else(|p| p.into_inner()) = self.home;
    }

    /// Whether no batch moves the image again: it rests in `GENERAL`, which is its home.
    pub(crate) fn settled(&self) -> bool {
        let current = *self.current.lock().unwrap_or_else(|p| p.into_inner());
        moves(current, self.home) == (None, None)
    }

    /// A transition's use of the whole image, for the hazard tracker: a write in every stage.
    pub(crate) fn transition_use(&self) -> Use<ResourceKey> {
        Use::writes(
            self.key,
            true,
            UseRange::Image {
                aspects: self.aspects.into(),
                levels: 0..self.levels,
                layers: 0..self.layers,
            },
            Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
        )
    }
}

/// The wrapped images of one device whose batches move them between layouts, by key. The
/// scheduler finds each batch's among its encoders' uses, and forgets each once it is
/// [`settled`](WrappedImage::settled).
#[derive(Default)]
pub(crate) struct Wrapped {
    images: Mutex<KeyMap<ResourceKey, Arc<WrappedImage>>>,
}

impl Wrapped {
    /// `image` moves between layouts: the scheduler looks for it in batches.
    pub(crate) fn insert(&self, image: &Arc<WrappedImage>) {
        self.lock().insert(image.key, Arc::clone(image));
    }

    /// The image of `wrapped` is gone, or settled.
    pub(crate) fn forget(&self, wrapped: &WrappedImage) {
        self.lock().remove(&wrapped.key);
    }

    /// The moving images among `keys`, each once, in key order.
    pub(crate) fn among(
        &self,
        keys: impl IntoIterator<Item = ResourceKey>,
    ) -> Vec<Arc<WrappedImage>> {
        let images = self.lock();
        if images.is_empty() {
            return Vec::new();
        }
        let mut found: Vec<Arc<WrappedImage>> = keys
            .into_iter()
            .filter_map(|key| images.get(&key).cloned())
            .collect();
        found.sort_by_key(|image| image.key);
        found.dedup_by_key(|image| image.key);
        found
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, KeyMap<ResourceKey, Arc<WrappedImage>>> {
        self.images.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// The plan of a texture of `plan` (the runtime's own image for the descriptor) over an image the
/// caller made with `info`, shared with the queue families `families` (read for a concurrent
/// image), on a device whose images are shared as `sharing` says: `plan` without the format list
/// where the image is not mutable-format. Refused, naming the first contradiction.
pub(crate) fn wrap_plan(
    mut plan: ImagePlan,
    info: &vk::ImageCreateInfo<'_>,
    families: &[u32],
    sharing: Sharing<'_>,
) -> Result<ImagePlan, WrappedImageProblem> {
    use WrappedImageProblem as P;
    let image_type = ImageType::from(info.image_type);
    if image_type != plan.image_type {
        return Err(P::ImageType { image: image_type });
    }
    let format = Format::from(info.format);
    if format != plan.format {
        return Err(P::Format {
            image: format,
            texture: plan.format,
        });
    }
    let extent = [info.extent.width, info.extent.height, info.extent.depth];
    if extent != plan.extent {
        return Err(P::Extent {
            image: extent,
            texture: plan.extent,
        });
    }
    if info.mip_levels != plan.levels {
        return Err(P::Levels {
            image: info.mip_levels,
            texture: plan.levels,
        });
    }
    if info.array_layers != plan.layers {
        return Err(P::Layers {
            image: info.array_layers,
            texture: plan.layers,
        });
    }
    let samples = SampleCountFlags::from(info.samples);
    if samples != plan.samples {
        return Err(P::Samples {
            image: samples,
            texture: plan.samples,
        });
    }
    let tiling = ImageTiling::from(info.tiling);
    if tiling != ImageTiling::OPTIMAL {
        return Err(P::Tiling { image: tiling });
    }
    let flags = ImageCreateFlags::from(info.flags);
    if flags.contains(ImageCreateFlags::PROTECTED) {
        return Err(P::Protected);
    }
    let usage = ImageUsageFlags::from(info.usage);
    // Where framebuffer fetch reads input attachments, the runtime's color render targets take
    // input attachment usage against a fetch no texture can foresee (`image_plan.rs`): an image
    // without it is wrapped without it, and a pass that fetches it renders into a stand-in that
    // has it (`attachment_stand_in.rs`).
    if !usage.contains(ImageUsageFlags::INPUT_ATTACHMENT) {
        plan.usage = plan.usage.difference(ImageUsageFlags::INPUT_ATTACHMENT);
    }
    let missing = plan.usage.difference(usage);
    if !missing.is_empty() {
        return Err(P::Usage { missing });
    }
    // So does the flag custom sample positions need of a multisampled depth render target
    // (`sample_positions::compatible_depth`), against positions no texture can foresee: an image
    // without it is wrapped without it, and a pass at custom positions renders it through a
    // stand-in made with it (`attachment_stand_in.rs`).
    let located = ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT;
    if !flags.contains(located) {
        plan.flags = plan.flags.difference(located);
    }
    // Cube compatibility, and extended usage where a storage view in another format needs it,
    // are the image's to have; the format list of a mutable-format image is the runtime's choice
    // of views, so an image that is not mutable is viewed in its own format alone.
    let mutable = ImageCreateFlags::MUTABLE_FORMAT;
    let mut needed = plan.flags.difference(mutable);
    if plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE) {
        needed |= mutable;
    }
    let missing = needed.difference(flags);
    if !missing.is_empty() {
        return Err(P::Flags { missing });
    }
    if !sharing.admits(info.sharing_mode, families) {
        return Err(P::QueueFamily);
    }
    if !flags.contains(mutable) {
        plan.flags = plan.flags.difference(mutable);
        plan.view_formats.clear();
    }
    Ok(plan)
}

/// Whether `layouts` meet the contract for an image of `usage` and `aspects`: `home` a layout the
/// runtime moves images out of and back into, `initial` such a layout or `UNDEFINED` or
/// `PREINITIALIZED`, each allowed by the image's usage and aspects (the `VkImageMemoryBarrier`
/// layout rules, VUID-VkImageMemoryBarrier-oldLayout-01208 to 01213).
pub(crate) fn check_layouts(
    layouts: VkImageLayouts,
    usage: ImageUsageFlags,
    aspects: ImageAspectFlags,
) -> Result<(), WrappedImageProblem> {
    let start = matches!(
        layouts.initial,
        ImageLayout::UNDEFINED | ImageLayout::PREINITIALIZED
    );
    for (layout, may_start) in [(layouts.initial, start), (layouts.home, false)] {
        if !(may_start || layout_allowed(layout, usage, aspects)) {
            return Err(WrappedImageProblem::Layout { layout });
        }
    }
    Ok(())
}

/// Whether the runtime moves an image of `usage` and `aspects` out of `layout` and back.
fn layout_allowed(layout: ImageLayout, usage: ImageUsageFlags, aspects: ImageAspectFlags) -> bool {
    use ImageLayout as L;
    use ImageUsageFlags as U;
    let color = aspects.contains(ImageAspectFlags::COLOR);
    match layout {
        L::GENERAL => true,
        L::COLOR_ATTACHMENT_OPTIMAL => color && usage.contains(U::COLOR_ATTACHMENT),
        L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            !color && usage.contains(U::DEPTH_STENCIL_ATTACHMENT)
        }
        L::DEPTH_STENCIL_READ_ONLY_OPTIMAL => {
            !color
                && usage.intersects(U::DEPTH_STENCIL_ATTACHMENT | U::SAMPLED | U::INPUT_ATTACHMENT)
        }
        L::SHADER_READ_ONLY_OPTIMAL => usage.intersects(U::SAMPLED | U::INPUT_ATTACHMENT),
        L::TRANSFER_SRC_OPTIMAL => usage.contains(U::TRANSFER_SRC),
        L::TRANSFER_DST_OPTIMAL => usage.contains(U::TRANSFER_DST),
        L::PRESENT_SRC_KHR => color,
        _ => false,
    }
}

/// Each plane of a multi-planar format, from plane 0: its format and the divisors of the image's
/// width and height that give its extent (the registry's `<plane>` elements of `vk.xml`, the
/// formats of Vulkan 1.1's sampler Y′CbCr conversion and Vulkan 1.3's two-plane 4:4:4 ones). Empty
/// for every other format.
pub(crate) fn planes_of(format: Format) -> &'static [(Format, u32, u32)] {
    use Format as F;
    match format {
        F::G8_B8_R8_3PLANE_420_UNORM => &[
            (F::R8_UNORM, 1, 1),
            (F::R8_UNORM, 2, 2),
            (F::R8_UNORM, 2, 2),
        ],
        F::G8_B8R8_2PLANE_420_UNORM => &[(F::R8_UNORM, 1, 1), (F::R8G8_UNORM, 2, 2)],
        F::G8_B8_R8_3PLANE_422_UNORM => &[
            (F::R8_UNORM, 1, 1),
            (F::R8_UNORM, 2, 1),
            (F::R8_UNORM, 2, 1),
        ],
        F::G8_B8R8_2PLANE_422_UNORM => &[(F::R8_UNORM, 1, 1), (F::R8G8_UNORM, 2, 1)],
        F::G8_B8_R8_3PLANE_444_UNORM => &[
            (F::R8_UNORM, 1, 1),
            (F::R8_UNORM, 1, 1),
            (F::R8_UNORM, 1, 1),
        ],
        F::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6_UNORM_PACK16, 2, 2),
            (F::R10X6_UNORM_PACK16, 2, 2),
        ],
        F::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6G10X6_UNORM_2PACK16, 2, 2),
        ],
        F::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6_UNORM_PACK16, 2, 1),
            (F::R10X6_UNORM_PACK16, 2, 1),
        ],
        F::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6G10X6_UNORM_2PACK16, 2, 1),
        ],
        F::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6_UNORM_PACK16, 1, 1),
        ],
        F::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4_UNORM_PACK16, 2, 2),
            (F::R12X4_UNORM_PACK16, 2, 2),
        ],
        F::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4G12X4_UNORM_2PACK16, 2, 2),
        ],
        F::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4_UNORM_PACK16, 2, 1),
            (F::R12X4_UNORM_PACK16, 2, 1),
        ],
        F::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4G12X4_UNORM_2PACK16, 2, 1),
        ],
        F::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4_UNORM_PACK16, 1, 1),
        ],
        F::G16_B16_R16_3PLANE_420_UNORM => &[
            (F::R16_UNORM, 1, 1),
            (F::R16_UNORM, 2, 2),
            (F::R16_UNORM, 2, 2),
        ],
        F::G16_B16R16_2PLANE_420_UNORM => &[(F::R16_UNORM, 1, 1), (F::R16G16_UNORM, 2, 2)],
        F::G16_B16_R16_3PLANE_422_UNORM => &[
            (F::R16_UNORM, 1, 1),
            (F::R16_UNORM, 2, 1),
            (F::R16_UNORM, 2, 1),
        ],
        F::G16_B16R16_2PLANE_422_UNORM => &[(F::R16_UNORM, 1, 1), (F::R16G16_UNORM, 2, 1)],
        F::G16_B16_R16_3PLANE_444_UNORM => &[
            (F::R16_UNORM, 1, 1),
            (F::R16_UNORM, 1, 1),
            (F::R16_UNORM, 1, 1),
        ],
        F::G8_B8R8_2PLANE_444_UNORM => &[(F::R8_UNORM, 1, 1), (F::R8G8_UNORM, 1, 1)],
        F::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16 => &[
            (F::R10X6_UNORM_PACK16, 1, 1),
            (F::R10X6G10X6_UNORM_2PACK16, 1, 1),
        ],
        F::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16 => &[
            (F::R12X4_UNORM_PACK16, 1, 1),
            (F::R12X4G12X4_UNORM_2PACK16, 1, 1),
        ],
        F::G16_B16R16_2PLANE_444_UNORM => &[(F::R16_UNORM, 1, 1), (F::R16G16_UNORM, 1, 1)],
        _ => &[],
    }
}

/// The plan of a texture of `plan` (the runtime's own image for the descriptor) over plane
/// `plane` of a multi-planar image the caller made with `info`, shared with the queue families
/// `families` (read for a concurrent image), on a device whose images are shared as `sharing`
/// says, for a texture of Metal usage `usage`; with the plane's aspect, which its views name. The
/// plan's format is the descriptor's, its extent the plane's, and its usage the image's sampled
/// and storage usages, which its views serve: no copy, fill or render pass of it is made
/// (`RuntimeGap::VkImagePlaneCopy`). Refused, naming the first contradiction.
pub(crate) fn plane_plan(
    mut plan: ImagePlan,
    info: &vk::ImageCreateInfo<'_>,
    plane: u32,
    usage: TextureUsage,
    families: &[u32],
    sharing: Sharing<'_>,
) -> Result<(ImagePlan, ImageAspectFlags), WrappedImageProblem> {
    use WrappedImageProblem as P;
    let image_type = ImageType::from(info.image_type);
    if image_type != ImageType::TYPE_2D || plan.image_type != ImageType::TYPE_2D {
        return Err(P::ImageType { image: image_type });
    }
    let planes = planes_of(Format::from(info.format));
    let Some(&(format, width, height)) = planes.get(plane as usize) else {
        return Err(P::Plane {
            plane,
            planes: planes.len() as u32,
        });
    };
    if !compatible(format, plan.format, false) {
        return Err(P::PlaneFormat {
            plane: format,
            texture: plan.format,
        });
    }
    let extent = [
        info.extent.width / width,
        info.extent.height / height,
        info.extent.depth,
    ];
    if extent != plan.extent {
        return Err(P::Extent {
            image: extent,
            texture: plan.extent,
        });
    }
    if info.mip_levels != plan.levels {
        return Err(P::Levels {
            image: info.mip_levels,
            texture: plan.levels,
        });
    }
    if info.array_layers != plan.layers {
        return Err(P::Layers {
            image: info.array_layers,
            texture: plan.layers,
        });
    }
    let samples = SampleCountFlags::from(info.samples);
    if samples != plan.samples {
        return Err(P::Samples {
            image: samples,
            texture: plan.samples,
        });
    }
    let tiling = ImageTiling::from(info.tiling);
    if tiling != ImageTiling::OPTIMAL {
        return Err(P::Tiling { image: tiling });
    }
    let flags = ImageCreateFlags::from(info.flags);
    if flags.contains(ImageCreateFlags::PROTECTED) {
        return Err(P::Protected);
    }
    // A view of one plane in its own format takes a mutable-format image
    // (`VUID-VkImageViewCreateInfo-image-01762`), and its usage beyond the multi-planar
    // format's, extended usage.
    let needed = ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE;
    let missing = needed.difference(flags);
    if !missing.is_empty() {
        return Err(P::Flags { missing });
    }
    let shader = ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE;
    let mut wanted = ImageUsageFlags::empty();
    if usage.contains(TextureUsage::SHADER_READ) {
        wanted |= ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::SHADER_WRITE) {
        wanted |= ImageUsageFlags::STORAGE;
    }
    let image_usage = ImageUsageFlags::from(info.usage);
    let missing = wanted.difference(image_usage);
    if !missing.is_empty() {
        return Err(P::Usage { missing });
    }
    if !sharing.admits(info.sharing_mode, families) {
        return Err(P::QueueFamily);
    }
    plan.extent = extent;
    plan.usage = image_usage & shader;
    plan.flags = needed;
    plan.view_formats.clear();
    let aspect = match plane {
        0 => ImageAspectFlags::PLANE_0,
        1 => ImageAspectFlags::PLANE_1,
        _ => ImageAspectFlags::PLANE_2,
    };
    Ok((plan, aspect))
}

impl Device {
    /// A texture of `descriptor` over `image`, which the caller made with `info` and keeps: a
    /// swapchain image of an adopted device, for example. The runtime never destroys the image or
    /// frees its memory; it finds the image in `layouts.initial` at the first batch that uses it,
    /// in `layouts.home` at every later one, and leaves it in `layouts.home` after each
    /// ([`VkImageLayouts`]). Hazard tracking orders its uses as it orders a texture the runtime
    /// made.
    ///
    /// The image has every usage the runtime's own texture of `descriptor` has, the transfer
    /// usages included wherever the device copies the format, whatever the descriptor's usage
    /// (a blit encoder's copies and an exact resolve's copy of the samples use them). Only the
    /// input attachment usage may be missing: a pass that fetches the texture renders into a
    /// stand-in that has it.
    ///
    /// A runtime extension: Metal makes no texture over a Vulkan image. The descriptor's
    /// `placement` gives the texture's resource ID as `new_texture`'s does
    /// (`resource_ids.rs`), and [`IdPlacement::Outside`] gives it
    /// none, as there: it answers 0, holds no slot, and counts in no plain allocation or
    /// reservation, so a swapchain image moves no program texture's ID. Unlike `new_texture`,
    /// the call takes no sampler ID for the device at any placement.
    ///
    /// Refused: a descriptor Metal stops, as [`Device::new_texture`] refuses it; storage other
    /// than `Private` (`Invalid`, `StorageMode`); an image that contradicts the descriptor (a
    /// usage missing, [`WrappedImageProblem::Usage`]), a layout outside the contract, or a texture
    /// that moves between layouts with `Untracked` hazards (`Invalid` with a
    /// [`WrappedImageProblem`]); a format or size the device lacks for the descriptor
    /// (`Unsupported`, as `new_texture`).
    ///
    /// # Safety
    ///
    /// - `image` is a live image of this device, made with exactly `info` (its `pNext` chain is
    ///   not read; a format list there includes every format of the runtime's views of a
    ///   mutable-format image), bound to memory, in `layouts.initial` now, and owned by the
    ///   device's queue family whenever a batch of the runtime that uses it runs.
    /// - The image and its memory outlive the texture, every view of it, and every command
    ///   buffer that uses them.
    /// - Between the runtime's batches the caller uses the image only in `layouts.home`, ordered
    ///   against the runtime's batches that use it (by the binary semaphores of
    ///   [`CommandBuffer::encode_wait_for_vk_semaphore`](super::super::CommandBuffer::encode_wait_for_vk_semaphore)
    ///   and
    ///   [`encode_signal_vk_semaphore`](super::super::CommandBuffer::encode_signal_vk_semaphore),
    ///   or by waiting for their completion).
    pub unsafe fn new_texture_with_vk_image(
        &self,
        descriptor: &TextureDescriptor,
        image: vk::Image,
        info: &vk::ImageCreateInfo<'_>,
        layouts: VkImageLayouts,
    ) -> Result<Texture, Error> {
        // SAFETY: the caller's contract.
        unsafe {
            self.current()
                .new_texture_with_vk_image_now(descriptor, image, info, layouts)
        }
    }

    /// [`Device::new_texture_with_vk_image`] on the generation that serves (`device/line.rs`).
    pub(crate) unsafe fn new_texture_with_vk_image_now(
        &self,
        descriptor: &TextureDescriptor,
        image: vk::Image,
        info: &vk::ImageCreateInfo<'_>,
        layouts: VkImageLayouts,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture_with_vk_image";
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
        let wrapped = |problem| invalid(CALL, Problem::WrappedImage(problem));
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        let hazard = options.hazard_tracking_mode()?;
        options.cpu_cache_mode()?;
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error(CALL))?;
        if storage != StorageMode::Private {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        if image == vk::Image::null() {
            return Err(wrapped(WrappedImageProblem::NullImage));
        }
        if layout.texture_type == TextureType::TextureBuffer {
            return Err(wrapped(WrappedImageProblem::ImageType {
                image: info.image_type.into(),
            }));
        }
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(CALL, refusal))?;
        let families: &[u32] = if info.sharing_mode == vk::SharingMode::CONCURRENT
            && !info.p_queue_family_indices.is_null()
        {
            // SAFETY: the caller's contract: `info` is the image's creation, whose concurrent
            // queue families are `queue_family_index_count` at `p_queue_family_indices`.
            unsafe {
                std::slice::from_raw_parts(
                    info.p_queue_family_indices,
                    info.queue_family_index_count as usize,
                )
            }
        } else {
            &[]
        };
        let plan = wrap_plan(plan, info, families, self.shared().sharing()).map_err(wrapped)?;
        let aspects = image_aspects(plan.format);
        check_layouts(layouts, info.usage.into(), aspects).map_err(wrapped)?;
        let identity = Identity::new(hazard, false);
        if layouts.moves() && !identity.tracked {
            return Err(wrapped(WrappedImageProblem::Untracked));
        }
        let [width, height, depth] = plan.extent;
        let caller = Arc::new(WrappedImage::new(
            identity.key,
            image,
            aspects,
            plan.levels,
            plan.layers,
            layouts,
            format!(
                "texture over the caller's VkImage {image:?}, {:?}, {width}x{height}x{depth}",
                plan.format
            ),
            None,
        ));
        if layouts.moves() {
            self.shared().interop.wrapped.insert(&caller);
        }
        let image = Image {
            device: self.clone(),
            raw: super::Raw::new(image),
            owner: Owner::Caller(caller),
            plan,
            options,
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            decoded: None,
            second: std::sync::OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            released: AtomicBool::new(false),
            host_shadow: Mutex::default(),
            between: OnceLock::new(),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: false,
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        self.degrade_block_decode(&layout);
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let storage = Storage::Image(Arc::new(image));
        let id_use = IdUse::Program(descriptor.placement);
        Ok(Texture::made(self, storage, record, None, id_use))
    }
}

impl Device {
    /// A texture of `descriptor` over plane `plane` of `image`, a multi-planar image the caller
    /// made with `info` and keeps (a video decoder's output picture, whose
    /// `VkVideoProfileListInfoKHR` and decoded picture buffer role only the caller knows). The
    /// counterpart of Metal's `newTextureWithDescriptor:iosurface:plane:`, over a Vulkan image's
    /// plane rather than an IOSurface's: the texture reads and writes the plane's texels in place,
    /// through views of that plane in the descriptor's format (`R8Unorm` or `RG8Unorm` for an
    /// NV12 image's luma and chroma planes, `R16Unorm` or `RG16Unorm` for a P010 image's). Hazard
    /// tracking orders its uses as it orders a texture the runtime made, and it lives as such a
    /// texture does, holding nothing of the image but its own views.
    ///
    /// The descriptor's type is `Type2D` (or `Type2DArray` over an image of several layers), its
    /// width and height the plane's (the image's divided as the format's planes are: half of each
    /// for a 4:2:0 format's chroma plane), its levels and layers the image's, its format
    /// compatible with the plane's (`VUID-VkImageViewCreateInfo-image-01586`), its storage
    /// `Private`. Its `ShaderRead` usage takes the image's `SAMPLED` usage, `ShaderWrite` its
    /// `STORAGE`.
    ///
    /// What differs from Metal's: nothing a function's reads and writes see, where the plane
    /// holds the texels Metal's surface would; the runtime makes no copy of it, so a blit copy,
    /// fill or mipmap generation of the texture, a render pass that renders it, and a view or
    /// binding that would hold its texels apart (a view of another component layout on a device
    /// that reinterprets none, a binding beyond the texture's usage) are refused
    /// (`NotImplemented(VkImagePlaneCopy)`), as is a descriptor with `RenderTarget` usage. Metal
    /// copies and renders a surface's plane. The descriptor's `placement` gives the texture's
    /// resource ID as [`new_texture_with_vk_image`](Self::new_texture_with_vk_image)'s does.
    ///
    /// Refused: a descriptor Metal stops, as [`Device::new_texture`] refuses it; storage other
    /// than `Private` (`Invalid`, `StorageMode`); an image that contradicts the descriptor or the
    /// contract below (`Invalid` with a [`WrappedImageProblem`]: not multi-planar or no such
    /// plane, `Plane`; a format incompatible with the plane's, `PlaneFormat`; an extent other than
    /// the plane's; another tiling than optimal; `MUTABLE_FORMAT` or `EXTENDED_USAGE` missing,
    /// `Flags`; the usage missing; a protected image; one not shared with every queue family of
    /// the device); a format or size the device lacks for the descriptor (`Unsupported`, as
    /// `new_texture`).
    ///
    /// # Safety
    ///
    /// - `image` is a live image of this device, made with exactly `info` (its `pNext` chain is
    ///   not read), bound to memory, and created with `CONCURRENT` sharing among the device's
    ///   queue families ([`DeviceHandles::queue_families`](crate::DeviceHandles::queue_families))
    ///   and every family the caller uses it on (an extra queue's,
    ///   [`DeviceHandles::extra_queues`](crate::DeviceHandles::extra_queues)), with
    ///   `MUTABLE_FORMAT`, `EXTENDED_USAGE` and the usages above.
    /// - It is in `VK_IMAGE_LAYOUT_GENERAL` whenever a batch of the runtime that uses it runs: the
    ///   runtime never moves it between layouts and binds its views in `GENERAL`. With
    ///   `VK_KHR_unified_image_layouts` (`unifiedImageLayoutsVideo`), `GENERAL` serves a video
    ///   decode's output and reference pictures too; without it, the caller moves the image to
    ///   `GENERAL` before the runtime's use and back after.
    /// - The runtime never takes queue family ownership of it, never moves its layout, and never
    ///   destroys it or frees its memory: the image and its memory outlive the texture, every
    ///   view of it, and every command buffer that uses them.
    /// - The caller's own uses of the image are ordered against the runtime's batches that use
    ///   it: by a shared event's semaphore ([`SharedEvent::vk_wait`](crate::SharedEvent::vk_wait),
    ///   [`SharedEvent::reserve_vk_signal`](crate::SharedEvent::reserve_vk_signal)), by the binary
    ///   semaphores of
    ///   [`CommandBuffer::encode_wait_for_vk_semaphore`](crate::CommandBuffer::encode_wait_for_vk_semaphore)
    ///   and
    ///   [`encode_signal_vk_semaphore`](crate::CommandBuffer::encode_signal_vk_semaphore), or by
    ///   waiting for their completion.
    pub unsafe fn new_texture_with_vk_image_plane(
        &self,
        descriptor: &TextureDescriptor,
        image: vk::Image,
        info: &vk::ImageCreateInfo<'_>,
        plane: u32,
    ) -> Result<Texture, Error> {
        // SAFETY: the caller's contract.
        unsafe {
            self.current()
                .new_texture_with_vk_image_plane_now(descriptor, image, info, plane)
        }
    }

    /// [`Device::new_texture_with_vk_image_plane`] on the generation that serves
    /// (`device/line.rs`).
    pub(crate) unsafe fn new_texture_with_vk_image_plane_now(
        &self,
        descriptor: &TextureDescriptor,
        image: vk::Image,
        info: &vk::ImageCreateInfo<'_>,
        plane: u32,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_texture_with_vk_image_plane";
        let descriptor = &self.texture_options(CALL, descriptor)?;
        if descriptor.placement != IdPlacement::Outside {
            self.check_texture_placement(
                CALL,
                descriptor.placement,
                descriptor.usage,
                descriptor.texture_type,
            )?;
        }
        let wrapped = |problem| invalid(CALL, Problem::WrappedImage(problem));
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        let hazard = options.hazard_tracking_mode()?;
        options.cpu_cache_mode()?;
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error(CALL))?;
        if storage != StorageMode::Private {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        if image == vk::Image::null() {
            return Err(wrapped(WrappedImageProblem::NullImage));
        }
        if !matches!(
            layout.texture_type,
            TextureType::Type2D | TextureType::Type2DArray
        ) {
            return Err(wrapped(WrappedImageProblem::ImageType {
                image: info.image_type.into(),
            }));
        }
        if descriptor.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(Error::NotImplemented(RuntimeGap::VkImagePlaneCopy));
        }
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(CALL, refusal))?;
        let families: &[u32] = if info.sharing_mode == vk::SharingMode::CONCURRENT
            && !info.p_queue_family_indices.is_null()
        {
            // SAFETY: the caller's contract: `info` is the image's creation, whose concurrent
            // queue families are `queue_family_index_count` at `p_queue_family_indices`.
            unsafe {
                std::slice::from_raw_parts(
                    info.p_queue_family_indices,
                    info.queue_family_index_count as usize,
                )
            }
        } else {
            &[]
        };
        let (plan, aspect) = plane_plan(
            plan,
            info,
            plane,
            descriptor.usage,
            families,
            self.shared().sharing(),
        )
        .map_err(wrapped)?;
        let identity = Identity::new(hazard, false);
        let [width, height, _] = plan.extent;
        // The tracker's barriers over it name the colour aspect, which Vulkan takes of a
        // multi-planar image (`VUID-VkImageMemoryBarrier-image-09242`), and cover every plane.
        let caller = Arc::new(WrappedImage::new(
            identity.key,
            image,
            ImageAspectFlags::COLOR,
            plan.levels,
            plan.layers,
            VkImageLayouts::general(),
            format!(
                "plane {plane} of the caller's VkImage {image:?}, {:?} as {:?}, {width}x{height}",
                Format::from(info.format),
                plan.format
            ),
            Some(aspect),
        ));
        let image = Image {
            device: self.clone(),
            raw: super::Raw::new(image),
            owner: Owner::Caller(caller),
            plan,
            options,
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            identity,
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            decoded: None,
            second: std::sync::OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            released: AtomicBool::new(false),
            host_shadow: Mutex::default(),
            between: OnceLock::new(),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: false,
            staging_copies: false,
            objects: ObjectsToken::default(),
        };
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let storage = Storage::Image(Arc::new(image));
        let id_use = IdUse::Program(descriptor.placement);
        Ok(Texture::made(self, storage, record, None, id_use))
    }
}
