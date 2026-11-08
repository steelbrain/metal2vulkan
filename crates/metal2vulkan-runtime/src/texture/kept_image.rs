//! A merged shadow's kept image (`shadow_merge.rs`): the texels of the shadow before the commands,
//! copied from it by `vkCmdCopyImage` at their own coordinates, which the merge's kernel fetches
//! beside the shadow's.
//!
//! A tracked `Private` 2D image of one level and slice, of the shadow's extent and of the
//! unsigned-integer format of its word view (`shadow_copy_back.rs`), so a copy of the shadow's
//! texels holds their bits ("compatible" formats, `vkCmdCopyImage`) and the kernel reads them as
//! it reads the shadow's: sampled and copied into, nothing else. It is made in memory of its own,
//! as its memory lies (each texel the merge reads, the keep before it writes), and starts in
//! `UNDEFINED`, which the device's next submission leaves for `GENERAL` (`first_move.rs`): making
//! it waits for no GPU work. Its view is one of its own (`Image::own_view`), which goes with it.
//!
//! Each command queue keeps the kept images its keeps made ([`KeptImages`]), for any merge of its
//! of the same view and extent, whatever its shadow (a texture made each frame over the same
//! bytes takes the last frame's): the encoder that took one holds it to its end, or the run of
//! passes it begins to the run's end (`pass_runs.rs`; [`KeptLease`]), and the command buffers that used it hold it (`Arc`) until they complete.
//! `WorkCounts::shadow_merge_kept_images_made`, `shadow_merge_kept_images_followed` and
//! `shadow_merge_kept_images_live` count them.

use super::super::binding_view::VkView;
use super::super::device::lock;
use super::super::error::Error;
use super::super::resource::Identity;
use super::super::texture_descriptor::TextureDescriptor;
use super::super::values::{HazardTrackingMode, ResourceOptions};
use super::super::Device;
use super::{Image, ImagePlan};
use ash::vk;
use metal2vulkan::bind::ShadowTexelView;
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
    ImageViewType, SampleCountFlags,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The most kept images of one view and extent a queue keeps that no open encoder holds
/// ([`KeptImages`]).
const KEPT_SPARES: usize = 2;

/// The takes of its queue's kept images over which one that no open encoder holds stays untaken
/// ([`KeptImages`]): the runtime has no idle trim, so the images of a shape no longer merged (a
/// window resized) go as the queue's merges go on.
const KEPT_TAKES: u64 = 256;

/// A merged shadow's kept image (the module docs), of one view of a texel and an extent. Cheap to
/// clone (a retain): its queue's pool and the merges that use it hold it.
#[derive(Clone)]
pub(crate) struct KeptImage(Arc<Kept>);

/// What a [`KeptImage`] holds.
struct Kept {
    image: Arc<Image>,
    view: vk::ImageView,
    texel: ShadowTexelView,
    extent: [u32; 2],
}

impl Drop for Kept {
    /// One fewer live (`WorkCounts::shadow_merge_kept_images_live`): its view and its image go
    /// with the image's drop.
    fn drop(&mut self) {
        self.image
            .device
            .shared()
            .counts
            .shadow_merge_kept_image_gone();
    }
}

impl KeptImage {
    /// Its `VkImage`.
    pub(crate) fn raw(&self) -> vk::Image {
        self.0.image.raw.get()
    }

    /// Its view of its own format, the merge kernel's sampled image.
    pub(crate) fn view(&self) -> vk::ImageView {
        self.0.view
    }

    /// What hazard tracking knows of it: tracked.
    pub(crate) fn identity(&self) -> Identity {
        self.0.image.identity
    }

    /// Whether it is of the view `texel` and `extent`.
    pub(crate) fn fits(&self, texel: ShadowTexelView, extent: [u32; 2]) -> bool {
        self.0.texel == texel && self.0.extent == extent
    }

    /// Whether it is of `other`'s view and extent.
    fn alike(&self, other: &KeptImage) -> bool {
        self.fits(other.0.texel, other.0.extent)
    }
}

/// The kept images one command queue keeps for its merges' keeps (the module docs), each with
/// the hold of the encoder that took it last ([`KeptLease`]).
///
/// A keep takes one of its view and extent that no open encoder holds, without waiting for the
/// command buffers that used it to complete (they hold it until they do), and so the order of
/// its uses is the queue's:
/// 1. A keep and its merge lie in one encoder: a merged-back shadow is merged before the
///    encoder's next command that touches its bytes another way or at the encoder's end, and a
///    command writing two ways is merged before the next command (`shadow.rs`); or in one run of
///    render passes, in one batch, which holds the image from its first pass's keep until it ends
///    or its command buffer commits, after its last merge in submission order (`pass_runs.rs`).
///    An encoder a rebind encodes again takes no image of the pool (`ShadowMerge::new`).
/// 2. So the spans from a keep to its merge of two encoders (or runs) that took the image never
///    interleave in submission order: one encoder of a command buffer is open at a time, and they
///    go in order; each span lies in one batch, and a logical queue's batches go to its one
///    `VkQueue` in its order, a command buffer's whole.
/// 3. The span submitted second keeps after the first's merge, which ends with a barrier from
///    every command to every command (`Barrier::everything`, `shadow_merge.rs`): outside a render
///    pass, a pipeline barrier's first synchronization scope is every command before it in
///    submission order on the queue and its second every command after it, in later command
///    buffers too (`vkCmdPipelineBarrier2`). The keep's writes and the merge's reads are uses of a
///    tracked image besides, which the device's tracker orders between command buffers as any
///    texture's.
///
/// At most [`KEPT_SPARES`] images of one view and extent that no open encoder holds, those taken
/// least recently going first, and none of them untaken over [`KEPT_TAKES`] takes; one let go of
/// here goes once the command buffers that used it completed. The rest go with the queue. A leaf
/// lock: an image let go of goes after it.
#[derive(Default)]
pub(crate) struct KeptImages(Mutex<Pool>);

/// What [`KeptImages`] holds: the images, and the takes so far, the clock each image's `taken`
/// reads.
#[derive(Default)]
struct Pool {
    kept: Vec<Pooled>,
    takes: u64,
}

/// A kept image of a queue's pool: the take that holds it (0 for none; [`KeptLease`]), and the
/// pool's count of takes at its last take.
struct Pooled {
    image: KeptImage,
    held: Arc<AtomicU64>,
    taken: u64,
}

impl Pooled {
    /// Whether no open encoder holds it.
    fn free(&self) -> bool {
        self.held.load(Ordering::Acquire) == 0
    }
}

/// An open encoder's hold on a kept image it took from its queue's pool ([`KeptImages`]): no
/// other keep takes the image while it lives. The encoder holds it to its end, or its drop; a run
/// of passes over the shadowed texture, past the encoder's end to the run's (`pass_runs.rs`).
pub(crate) struct KeptLease {
    held: Arc<AtomicU64>,
    take: u64,
}

impl Drop for KeptLease {
    fn drop(&mut self) {
        // Its take's own: the image is taken again only once this one let go.
        let _ = (self.held).compare_exchange(self.take, 0, Ordering::AcqRel, Ordering::Relaxed);
    }
}

/// A kept image taken from its queue's pool ([`KeptImages::take`]).
pub(crate) struct TakenImage {
    pub(crate) image: KeptImage,
    pub(crate) lease: KeptLease,
    /// Whether a command buffer not yet completed holds it still: taken in its queue's order
    /// ahead of that completion (`WorkCounts::shadow_merge_kept_images_followed`).
    pub(crate) followed: bool,
}

impl KeptImages {
    /// A kept image of the view `texel` and `extent` that no open encoder holds, the one taken
    /// last, taken and held by the lease returned; none where the queue keeps no such image.
    pub(crate) fn take(&self, texel: ShadowTexelView, extent: [u32; 2]) -> Option<TakenImage> {
        let mut gone = Vec::new();
        let taken = {
            let mut pool = lock(&self.0);
            pool.takes += 1;
            let take = pool.takes;
            let at = (pool.kept.iter().enumerate())
                .filter(|(_, k)| k.image.fits(texel, extent) && k.free())
                .max_by_key(|(_, k)| k.taken)
                .map(|(at, _)| at);
            let taken = at.map(|at| {
                let k = &mut pool.kept[at];
                k.held.store(take, Ordering::Release);
                k.taken = take;
                // The pool holds one; any other is a merge's, whose command buffer holds it
                // until it completes.
                let followed = Arc::strong_count(&k.image.0) > 1;
                TakenImage {
                    image: k.image.clone(),
                    lease: KeptLease {
                        held: Arc::clone(&k.held),
                        take,
                    },
                    followed,
                }
            });
            pool.bound(&mut gone);
            taken
        };
        drop(gone);
        taken
    }

    /// Keep `image`, made for a keep that took none, held by the lease returned.
    pub(crate) fn keep(&self, image: KeptImage) -> KeptLease {
        let mut gone = Vec::new();
        let lease = {
            let mut pool = lock(&self.0);
            pool.takes += 1;
            let take = pool.takes;
            let held = Arc::new(AtomicU64::new(take));
            let lease = KeptLease {
                held: Arc::clone(&held),
                take,
            };
            pool.kept.push(Pooled {
                image,
                held,
                taken: take,
            });
            pool.bound(&mut gone);
            lease
        };
        drop(gone);
        lease
    }
}

impl Pool {
    /// Let go of the images that no open encoder holds past the bounds ([`KeptImages`]), those
    /// taken least recently first, into `gone`, which the caller drops without the lock.
    fn bound(&mut self, gone: &mut Vec<KeptImage>) {
        loop {
            let over = (0..self.kept.len())
                .filter(|&at| self.over(at))
                .min_by_key(|&at| self.kept[at].taken);
            let Some(at) = over else {
                return;
            };
            gone.push(self.kept.swap_remove(at).image);
        }
    }

    /// Whether the image at `at` is past the bounds: no open encoder holds it, and it went
    /// untaken over [`KEPT_TAKES`] takes or more than [`KEPT_SPARES`] alike are free.
    fn over(&self, at: usize) -> bool {
        let k = &self.kept[at];
        let alike = || {
            (self.kept.iter())
                .filter(|other| other.free() && other.image.alike(&k.image))
                .count()
        };
        k.free() && (k.taken + KEPT_TAKES <= self.takes || alike() > KEPT_SPARES)
    }
}

/// The format of `texel`'s view, and Metal's pixel format of it.
fn formats(texel: ShadowTexelView) -> (Format, PixelFormat) {
    match texel {
        ShadowTexelView::R32Uint => (Format::R32_UINT, PixelFormat::R32Uint),
        ShadowTexelView::R8G8B8A8Uint => (Format::R8G8B8A8_UINT, PixelFormat::RGBA8Uint),
        ShadowTexelView::R32G32Uint => (Format::R32G32_UINT, PixelFormat::RG32Uint),
        ShadowTexelView::R32G32B32A32Uint => (Format::R32G32B32A32_UINT, PixelFormat::RGBA32Uint),
        ShadowTexelView::R8Uint => (Format::R8_UINT, PixelFormat::R8Uint),
        ShadowTexelView::R16Uint => (Format::R16_UINT, PixelFormat::R16Uint),
    }
}

impl Device {
    /// A new kept image (the module docs) of the view `texel` and `extent`, with its view.
    /// Refused as `new_texture` refuses its image.
    pub(crate) fn new_kept_image(
        &self,
        texel: ShadowTexelView,
        extent: [u32; 2],
    ) -> Result<KeptImage, Error> {
        let (format, pixel_format) = formats(texel);
        let [width, height] = extent;
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format,
            width: width.into(),
            height: height.into(),
            usage: TextureUsage::SHADER_READ,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error("new_texture"))?;
        let usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST;
        let plan = ImagePlan {
            pixel_format,
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::TYPE_2D,
            format,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: SampleCountFlags::from_raw(1),
            usage,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_WRITE
                | TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let identity = Identity::new(HazardTrackingMode::Tracked, false);
        let image =
            self.new_image_as_it_lies(plan, &layout, &descriptor, identity, Device::runtime_owner)?;
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::SAMPLED,
        })?;
        self.shared().counts.shadow_merge_kept_image_made();
        Ok(KeptImage(Arc::new(Kept {
            image: image.shared(),
            view,
            texel,
            extent,
        })))
    }
}
