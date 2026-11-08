//! The images a texture over detached bytes presents: a stand-in of its image's plan, zero and
//! never written, which its views, its heap-set elements and its attachments name while its bytes
//! are detached (`detach.rs`). A tiled texture of more than one level keeps its own image instead,
//! its first level loaded with the bytes' last texels ([`keep_first_levels`]).
//!
//! - **What Metal answers** (an M4 under macOS 27.0.1): a texture over detached
//!   bytes keeps its descriptor's metadata (every size and level query answers it, its views'
//!   too), and a function that reaches it unnamed, where nothing made its pages resident, reads
//!   zero bytes in every texel and drops what it writes. So the stand-in is of the image's own
//!   plan (its extent, levels, layers, samples, format and view formats), its texels zero bytes.
//! - **Exact by construction** (the coordinator's ruling (b)): one stand-in per distinct plan,
//!   shared by every detached texture of that plan, in memory of its own, zeroed once by copies
//!   of zeros (every texel of every aspect, level and layer; by the device's next submission,
//!   which first moves it to `GENERAL`, `texture/first_move.rs`) and never written after: nothing
//!   that runs writes it, so every read of it reads zero bytes on every device. No stand-in
//!   aliases another's memory: Vulkan leaves an optimal image's contents undefined over memory
//!   another image wrote.
//! - **Writes go to a sink.** The read stand-in has no `STORAGE` usage, so no shader writes it.
//!   A plan with `STORAGE` usage also has a sink: an image of the same plan the storage views (a
//!   heap set's storage-class elements) name, so an unnamed write lands where no sampled read
//!   looks. A read through such a storage view reads the sink.
//! - **Uses that run** never name it: a command that names a texture over detached bytes is
//!   never submitted (`EncoderCore::skips`); its recording may name the stand-in, so the read
//!   stand-in keeps the plan's other usages (attachments, transfers).
//! - **Kept** by the images that present it, and by the device weakly: a stand-in no image
//!   presents goes, its memory with it.

use super::super::buffer::Buffer;
use super::super::device::lock;
use super::super::error::{vulkan, Error, RuntimeGap};
use super::super::memory::{MemoryClass, Placement};
use super::super::{exit_guard, release, Device};
use super::first_move::{FirstMove, Fresh};
use super::{copy_units, zero_copies, Image, ImageObjects, ImagePlan};
use crate::resource::ResourceKey;
use ash::vk;
use metal2vulkan::vk::{ImageType, ImageUsageFlags, SampleCountFlags};
use std::any::Any;
use std::sync::{Arc, Mutex, Weak};

/// The stand-in images of one plan: the read one and, for a plan with `STORAGE` usage, the sink.
pub(crate) struct StandIn {
    device: Device,
    read: StandInImage,
    sink: Option<StandInImage>,
    /// Its images' moves to `GENERAL` and fills with zeros, until the device's next submission
    /// takes them (`texture/first_move.rs`).
    first_moves: Mutex<Vec<FirstMove>>,
}

/// One stand-in `VkImage` in memory of its own.
struct StandInImage {
    raw: vk::Image,
    placement: Placement,
}

impl StandIn {
    /// The image a sampled view, an attachment or a transfer names.
    pub(in crate::texture) fn read(&self) -> vk::Image {
        self.read.raw
    }

    /// The image a view of `usage` names: the sink for a storage view, else the read image.
    pub(in crate::texture) fn for_usage(&self, usage: ImageUsageFlags) -> vk::Image {
        match &self.sink {
            Some(sink) if usage.contains(ImageUsageFlags::STORAGE) => sink.raw,
            _ => self.read.raw,
        }
    }

    /// The bytes of memory it holds.
    fn bytes(&self) -> u64 {
        self.read.placement.size() + self.sink.as_ref().map_or(0, |sink| sink.placement.size())
    }
}

impl Drop for StandIn {
    fn drop(&mut self) {
        // Once the process is exiting, the images are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        shared.counts.stand_in_released(self.bytes());
        for image in std::iter::once(&self.read).chain(&self.sink) {
            // SAFETY: no image presents the stand-in any more, and a command buffer that may read
            // it holds the image that presented it until it completes.
            unsafe { release::object(shared.vk(), image.raw) };
            lock(&shared.memory).free(shared.vk(), &image.placement);
        }
    }
}

impl Fresh for StandIn {
    fn take(&self) -> Vec<FirstMove> {
        lock(&self.first_moves).drain(..).collect()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        lock(&self.first_moves).extend(moves);
    }

    fn rebindable(&self) -> Option<ResourceKey> {
        None
    }
}

/// A device's live stand-ins, by plan, held weakly.
#[derive(Default)]
pub(crate) struct StandIns {
    by_plan: Mutex<Vec<(ImagePlan, Weak<StandIn>)>>,
}

impl Device {
    /// The stand-in of `plan` (the module docs): the live one, else a new one, zero and in the
    /// `GENERAL` layout. Refused (`NotImplemented(DetachTextures)`) for a plan whose texels no
    /// copy of zeros reaches (a multi-planar format; a multisample one), so no exact stand-in
    /// exists.
    pub(in crate::texture) fn stand_in(&self, plan: &ImagePlan) -> Result<Arc<StandIn>, Error> {
        let shared = self.shared();
        let mut by_plan = lock(&shared.rebinds.stand_ins.by_plan);
        by_plan.retain(|(_, stand_in)| stand_in.strong_count() > 0);
        if let Some(stand_in) = by_plan
            .iter()
            .find(|(held, _)| held == plan)
            .and_then(|(_, stand_in)| stand_in.upgrade())
        {
            return Ok(stand_in);
        }
        let stand_in = Arc::new(self.new_stand_in(plan)?);
        let weak: Weak<StandIn> = Arc::downgrade(&stand_in);
        let fresh: Weak<dyn Fresh> = weak;
        shared.textures.first_moves.register(fresh);
        shared.counts.stand_in_made(stand_in.bytes());
        by_plan.push((plan.clone(), Arc::downgrade(&stand_in)));
        Ok(stand_in)
    }

    /// A new stand-in of `plan`.
    fn new_stand_in(&self, plan: &ImagePlan) -> Result<StandIn, Error> {
        if copy_units(plan.format, plan.pixel_format).is_empty()
            || plan.samples != SampleCountFlags::COUNT_1
        {
            return Err(Error::NotImplemented(RuntimeGap::DetachTextures));
        }
        let usage = plan.usage.difference(ImageUsageFlags::STORAGE) | ImageUsageFlags::TRANSFER_DST;
        let (read, read_move) = self.stand_in_image(plan, usage)?;
        let sink = match plan.usage.contains(ImageUsageFlags::STORAGE) {
            true => Some(self.stand_in_image(plan, plan.usage | ImageUsageFlags::TRANSFER_DST)),
            false => None,
        };
        let sink = match sink.transpose() {
            Ok(sink) => sink,
            Err(error) => {
                self.drop_stand_in_image(&read);
                return Err(error);
            }
        };
        let (sink, sink_move) = sink.unzip();
        let stand_in = StandIn {
            device: self.clone(),
            read,
            sink,
            first_moves: Mutex::new(std::iter::once(read_move).chain(sink_move).collect()),
        };
        Ok(stand_in)
    }

    /// One stand-in image of `plan` with `usage`, and its move to the `GENERAL` layout that
    /// writes every texel with zero bytes.
    fn stand_in_image(
        &self,
        plan: &ImagePlan,
        usage: ImageUsageFlags,
    ) -> Result<(StandInImage, FirstMove), Error> {
        let plan = ImagePlan {
            usage,
            ..plan.clone()
        };
        let shared = self.shared();
        let raw_device = shared.vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan.create_info(self.shared().sharing(), &view_formats, &mut list);
        // SAFETY: the device is live and `info` (with its format list) valid: the plan's, which
        // made the texture's image, with fewer or more transfer usages.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        let requirements = shared.textures.placement().place(requirements);
        let allocated = crate::memory::allocate_image(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        );
        let placement = match allocated {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let image = StandInImage { raw, placement };
        // SAFETY: the placement meets the image's requirements and is bound nowhere else.
        if let Err(error) = unsafe {
            crate::creation::bind_image(
                raw_device,
                raw,
                image.placement.memory,
                image.placement.offset,
            )
        } {
            self.drop_stand_in_image(&image);
            return Err(vulkan(error));
        }
        let units = copy_units(plan.format, plan.pixel_format);
        let volume = plan.image_type == ImageType::TYPE_3D;
        let (length, copies) = zero_copies(
            plan.extent,
            plan.levels,
            plan.layers,
            volume,
            &units,
            super::ZERO_BYTES,
        );
        let zeros = match shared.textures.first_moves.zeros(self, length) {
            Ok(zeros) => zeros,
            Err(error) => {
                self.drop_stand_in_image(&image);
                return Err(error);
            }
        };
        let undefined = vk::ImageLayout::UNDEFINED;
        let first = FirstMove::new(raw, plan.format, undefined, None).zeroed(zeros, &copies);
        Ok((image, first))
    }

    /// `image`, made over the bytes of `bytes` while they are detached (`detach.rs`), in memory
    /// of its own, presents the stand-in of its plan from now on: called before its texture is
    /// made, so nothing names its own `VkImage` first, and after its copies took that image.
    /// Nothing where the bytes lie in pages. Refused as `Device::stand_in` refuses, and
    /// (`NotImplemented(DetachTextures)`) for an image with a decoded twin.
    pub(in crate::texture) fn stand_in_while_detached(
        &self,
        bytes: &Buffer,
        image: &Image,
    ) -> Result<(), Error> {
        if !bytes.detached() {
            return Ok(());
        }
        if image.decoded.is_some() {
            return Err(Error::NotImplemented(RuntimeGap::DetachTextures));
        }
        let stand_in = self.stand_in(image.plan())?;
        // Its own `VkImage` stays its own, with the views its copies hold of it
        // (`Image::own_view`); it has no other views yet.
        drop(image.present_stand_in(stand_in));
        Ok(())
    }

    /// Release `image`, unused.
    fn drop_stand_in_image(&self, image: &StandInImage) {
        let shared = self.shared();
        // SAFETY: no command uses the image.
        unsafe { release::object(shared.vk(), image.raw) };
        lock(&shared.memory).free(shared.vk(), &image.placement);
    }
}

/// Load the first level of each tiled texture of `textures` (a buffer's, by type erased:
/// `buffer::TexturesOver::keeping`) into its image as the bytes hold it now, before a detach lets
/// go of them (`detach.rs`, `Texture::keep_first_level`): such a texture keeps its own image while
/// they are detached, presenting no stand-in. Refused as the loads are (out of memory), before
/// anything is detached.
pub(crate) fn keep_first_levels(textures: Vec<Arc<dyn Any + Send + Sync>>) -> Result<(), Error> {
    for erased in textures {
        if let Ok(shared) = erased.downcast::<super::TextureShared>() {
            super::Texture(shared).keep_first_level()?;
        }
    }
    Ok(())
}

/// The images of the textures over a buffer's host pages (`buffer::TexturesOver`), as a detach
/// and a reattach take them (`detach.rs`, `buffer_descriptor.rs`), and whether a stand-in of
/// zero bytes presents each exactly.
pub(crate) struct TextureImages {
    images: Vec<Arc<Image>>,
    exact: bool,
}

impl TextureImages {
    /// The images among `erased`, the noted images of textures over the bytes, each once, and
    /// whether a stand-in presents each exactly (`TexturesOver::images`).
    pub(crate) fn of((erased, exact): (Vec<Arc<dyn Any + Send + Sync>>, bool)) -> TextureImages {
        let images = erased
            .into_iter()
            .filter_map(|image| image.downcast::<Image>().ok())
            .collect();
        TextureImages { images, exact }
    }

    /// Whether there is none.
    pub(crate) fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// The stand-in each image will present, of its plan, made now, before the detach: a
    /// failure leaves everything as it was (an image presenting one already takes it again).
    /// Refused (`NotImplemented(DetachTextures)`) where no stand-in presents an image exactly:
    /// one of texels its copies convert from the bytes (a two-plane YCbCr texture's), and one
    /// with a decoded twin, whose sampled views the stand-in does not present.
    pub(crate) fn stand_ins(&self, device: &Device) -> Result<Vec<Arc<StandIn>>, Error> {
        if !self.exact {
            return Err(Error::NotImplemented(RuntimeGap::DetachTextures));
        }
        self.images
            .iter()
            .map(|image| match image.decoded.is_some() {
                true => Err(Error::NotImplemented(RuntimeGap::DetachTextures)),
                false => device.stand_in(image.plan()),
            })
            .collect()
    }

    /// Each image presents its stand-in of `stand_ins` (in order) from now on: the objects that
    /// go, its views and, for an image bound to the detached pages, its `VkImage`, for those
    /// pages to keep until no command buffer holds them.
    pub(crate) fn present_stand_ins(&self, stand_ins: Vec<Arc<StandIn>>) -> Vec<ImageObjects> {
        self.images
            .iter()
            .zip(stand_ins)
            .filter_map(|(image, stand_in)| image.present_stand_in(stand_in))
            .collect()
    }

    /// Each image in memory of its own presents its own `VkImage` again, the bytes given pages
    /// again: the stand-ins' objects that go. An image bound to the pages takes the image a
    /// rebind made over the new pages instead (`ImageSwap`).
    pub(crate) fn present_own(&self) -> Vec<ImageObjects> {
        self.images
            .iter()
            .filter_map(|image| image.present_own())
            .collect()
    }

    /// Whether an image still presents a stand-in.
    pub(crate) fn standing_in(&self) -> bool {
        self.images.iter().any(|image| image.presents_stand_in())
    }
}
