//! Images in place over a no-copy buffer's dma-buf pages, kept as their texture goes for the next
//! texture in place over the same pages: a runtime-internal reuse, which nothing of Metal's
//! observes.
//!
//! A texture in place over such pages is a `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated
//! import of a dma-buf of its own pages, which the buffer's source makes as the texture is made
//! (`texture/in_place_image.rs`, `dma_buf.rs`). Each such import costs the kernel work for every
//! page: the source's `udmabuf` pins them, and RADV maps the whole dma-buf at a GPU address range
//! of its own, its first submission waiting for the page tables. On a Radeon 780M (RADV of Mesa
//! 26.2.6, Linux 7.3) a texture of 84 MiB over 4 KiB pages took 1.4 ms to make (1.1 ms of it the
//! source's), and a command buffer sampling it, of 1.7 ms of GPU work, 7.1 ms to complete, where
//! the same over a texture in the device's own memory took 2.2 ms (medians,
//! `tests::texture_shadow_bench`): a guest that makes a texture over the same pages for each
//! frame paid that each frame.
//!
//! - **Parked.** As an image in place over a no-copy buffer's dma-buf pages goes (its texture
//!   objects went, and every command buffer that used it completed: a committed command buffer
//!   holds the textures it uses until then), it is kept with its import under the import of its
//!   buffer's pages (`buffer.rs`'s `HostImport`, which every no-copy buffer over the same bytes
//!   lies in, `host_aliases.rs`), keyed by what it was made of beside the device ([`ParkKey`]),
//!   where its `VkImage` was never handed out (`Texture::vk_image`), it was made on the
//!   device's own threads (not its check's, `in_place_check.rs`), no rebind or detach replaced
//!   it, the pages' import still lives, and the device parks fewer than
//!   [`PARKED_MOST`](crate::host_residency::PARKED_MOST) images. Else it is destroyed and its
//!   import freed, as it was before.
//! - **Taken.** A texture made in place over pages lying in that import tries each plan of its
//!   image in the order its making tries them (`Device::in_place_image`: wider usages first):
//!   a parked image of the plan at its texels is taken before the source is asked, so the source
//!   is not called, nothing is imported and no `VkImage` is made; else that plan's image is made
//!   as before. So it ends with the first plan parked or made, which is the making's own where
//!   none is parked.
//! - **Exact.** A parked image is the image the making would make of its key: the same create
//!   info (the plan, the formats the device lists for its views, the plane's offset in the first
//!   page and its pitches), bound to a dedicated import of the same memory type of a dma-buf of
//!   the same pages (the source's contract, `HostMemory::with_dma_buf`: the same pages for every
//!   request over the same pages). Its views and resolve objects went with the texture's image;
//!   the new texture makes its own at its binds. Its layout: one whose first move a submission
//!   made rests in `GENERAL`, owned by the device's queue family, which acquired it from
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT` (`texture/first_move.rs`). An exclusive resource stays owned by
//!   the family that acquired it, a transfer making its contents available only to another
//!   family (Vulkan 1.4, 12.10 "Resource Sharing Mode", and 7.7.6 "Queue Family Ownership
//!   Transfer"), and no family of another user ever reached it: so the taken image needs no move,
//!   and rests as a live image rests between its submissions, where the CPU's writes through
//!   `contents` meanwhile are host writes each later submission sees (7.9 "Host Write Ordering
//!   Guarantees"), as `dma_buf.rs` ("Their layouts") has it for every such image. One that no
//!   submission moved is still `UNDEFINED`, its acquire not made: the taken image notes the move
//!   a new one notes (`FirstMove::acquired`). The new texture's commands are ordered against the
//!   buffer's other uses by the bytes' identity, as a new image's are.
//! - **Released** untaken, its `VkImage` destroyed and its import freed: by the device's
//!   residency thread (`host_residency.rs`) at its first pass after the park (every 0.5 s) on a
//!   device whose held imports cost every submission (`Device::pointer_imports_tax_submissions`:
//!   there each parked image is one more import on every submission's list, about 0.46 us a
//!   submission on a Radeon 780M, where a take saves 0.13 to 2.6 ms of making over 2 MiB pages),
//!   else once parked for [`IDLE`](crate::host_residency::IDLE); and on a device whose held
//!   imports cost every submission past the most listed imports it keeps
//!   ([`RESIDENT_MOST`](crate::host_residency::RESIDENT_MOST)), each parked image one there,
//!   released in the order of its park among the releasable imports' groups by their last use;
//!   on the caller's thread as `Device::host_range_remapped` names a page of its buffer's bytes;
//!   and as the import of the pages goes, before the caller's deallocator and keep-alive go.
//! - **Counted.** `WorkCounts::dma_buf_images_parked`, `dma_buf_images_taken`,
//!   `dma_buf_images_missed` (made anew: none parked of the plan at the texels) and
//!   `dma_buf_images_parked_released`. A parked image's import stays in
//!   `dma_buf_image_memories_live`.
//! - **Over the runtime's exported memory.** A texture in place over a buffer of the runtime's
//!   memory exported as a dma-buf, or in a block of its buffers allocated exportable as one,
//!   where no image binds in that memory itself (`texture.rs`'s `exported_dma_buf_alias`), is a
//!   `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated import of the memory's own dma-buf
//!   (`dma_buf.rs`'s `bind_exported_dma_buf_image`): a `vkGetMemoryFdKHR` and an import for each
//!   texture, about 0.33 ms of import on a Radeon 780M (RADV) where a guest makes and frees two
//!   such textures each frame. It parks as above, under the buffer's pages (`buffer.rs`'s
//!   `FixedPages`, which no rebind or detach replaces) rather than a host import, keyed by the
//!   texels' offset in the memory, and the next texture over the same buffer of the same plan at
//!   the same texels and pitches takes it, so no export, import or `VkImage` is made: within the
//!   same bound and windows as the images over host pages, released untaken by the residency as
//!   they are (it never meets a host range a remap names), and as the buffer's pages go, before
//!   their memory is freed or recycled. Counted apart:
//!   `WorkCounts::exported_dma_buf_images_parked`, `exported_dma_buf_images_taken` and
//!   `exported_dma_buf_images_parked_released`.

use super::{DmaBufImage, DmaBufPages};
use crate::buffer::{FixedPages, HostImport};
use crate::device::lock;
use crate::exit_guard;
use crate::host_residency::now_ms;
use crate::image_plan::ImagePlan;
use crate::{release, Device};
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// What a parked image is taken by: everything its `VkImage` and its import were made of beside
/// the device, whose formats, features and plane layouts answer the same for the same key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParkKey {
    /// The image's plan: its format, extent, usage, flags and view formats among the rest.
    pub(crate) plan: ImagePlan,
    /// The host address of its texels' first byte among the pages the source makes dma-bufs of
    /// (`DmaBufTexels::at`); over the runtime's exported memory, their offset in that memory
    /// (`ExportedTexels::at`).
    pub(crate) at: u64,
    /// The bytes from a row's first texel to the next row's, and from a slice's to the next
    /// slice's.
    pub(crate) row_pitch: u64,
    pub(crate) slice_pitch: u64,
    /// The memory type of the pages' imports, which the image's import takes.
    pub(crate) type_index: u32,
}

/// What images park under: the import of a no-copy buffer's host pages (`buffer.rs`'s
/// `HostImport`), or the pages of a buffer of the runtime's exported memory (`FixedPages`). Each
/// releases what it parks as it goes; the residency notes it weakly (`host_residency.rs`).
pub(crate) trait ParkHolder: Send + Sync {
    /// The images parked under it.
    fn parked_images(&self) -> &ParkedImages;
}

impl ParkHolder for HostImport {
    fn parked_images(&self) -> &ParkedImages {
        HostImport::parked_images(self)
    }
}

impl ParkHolder for FixedPages {
    fn parked_images(&self) -> &ParkedImages {
        FixedPages::parked_images(self)
    }
}

/// Where an image parks as it goes: under the import of its no-copy buffer's dma-buf pages, or
/// under the pages of its buffer of the runtime's exported memory, weakly (it parks nowhere once
/// that went), what it is taken by, and the host bytes of its buffer, a remap of any of whose
/// pages releases it (none over exported memory).
pub(crate) struct ParkSpot {
    holder: Weak<dyn ParkHolder>,
    key: ParkKey,
    bytes: Range<u64>,
    /// Over the runtime's exported memory: counted apart.
    exported: bool,
}

impl ParkSpot {
    /// The spot under `import` of an image of `key` over a buffer of the host `bytes`.
    pub(crate) fn new(import: &Arc<HostImport>, key: ParkKey, bytes: Range<u64>) -> ParkSpot {
        let held: Weak<HostImport> = Arc::downgrade(import);
        let holder: Weak<dyn ParkHolder> = held;
        ParkSpot {
            holder,
            key,
            bytes,
            exported: false,
        }
    }

    /// The spot under `pages`, a buffer's pages of the runtime's exported memory, of an image of
    /// `key` over them.
    pub(crate) fn exported(pages: &Arc<FixedPages>, key: ParkKey) -> ParkSpot {
        let held: Weak<FixedPages> = Arc::downgrade(pages);
        let holder: Weak<dyn ParkHolder> = held;
        ParkSpot {
            holder,
            key,
            bytes: 0..0,
            exported: true,
        }
    }

    /// Park `raw`, bound to `import`, its dedicated import, which no command uses any more:
    /// `moved` where a submission made its first move. Given back where it does not park: what it
    /// parks under went, or the device parks as many as it parks at most.
    pub(crate) fn park(
        self,
        device: &Device,
        raw: vk::Image,
        import: Arc<DmaBufPages>,
        moved: bool,
    ) -> Result<(), (vk::Image, Arc<DmaBufPages>)> {
        let Some(holder) = self.holder.upgrade() else {
            return Err((raw, import));
        };
        let shared = device.shared();
        if !shared.residency.park(device, &holder) {
            return Err((raw, import));
        }
        match self.exported {
            true => shared.counts.exported_dma_buf_image_parked(),
            false => shared.counts.dma_buf_image_parked(),
        }
        holder.parked_images().push(ParkedImage {
            raw,
            import,
            key: self.key,
            bytes: self.bytes,
            moved,
            since: now_ms(),
            exported: self.exported,
        });
        // What it parks under may go here, its last holder: it releases what it parks.
        drop(holder);
        Ok(())
    }
}

/// An image parked under the import of its pages (the module docs).
pub(crate) struct ParkedImage {
    raw: vk::Image,
    import: Arc<DmaBufPages>,
    key: ParkKey,
    bytes: Range<u64>,
    /// A submission made its first move: it rests in `GENERAL` on the device's queue family.
    moved: bool,
    /// When it was parked, in the residency's milliseconds.
    since: u64,
    /// Over the runtime's exported memory: counted apart.
    exported: bool,
}

impl ParkedImage {
    /// When it was parked, in the residency's milliseconds.
    pub(crate) fn since(&self) -> u64 {
        self.since
    }

    /// Its device.
    pub(crate) fn device(&self) -> &Device {
        &self.import.device
    }

    /// Its `VkImage`, which names it among the parked images while it is parked.
    pub(crate) fn raw(&self) -> vk::Image {
        self.raw
    }

    /// Whether a page of the host bytes `start..end`, rounded out to `unit`, is one of its
    /// buffer's bytes' pages.
    pub(crate) fn meets(&self, (start, end): (u64, u64), unit: u64) -> bool {
        let unit = unit.max(1);
        let round = |range: Range<u64>| {
            let from = range.start - range.start % unit;
            (from, range.end.div_ceil(unit).saturating_mul(unit))
        };
        let (from, to) = round(start..end);
        let (first, last) = round(self.bytes.clone());
        start < end && first < to && from < last
    }

    /// The image, taken: its `VkImage` and import, and whether a submission made its first move.
    pub(crate) fn taken(self) -> (DmaBufImage, bool) {
        let ParkedImage {
            raw, import, moved, ..
        } = self;
        (DmaBufImage { raw, import }, moved)
    }
}

/// The images parked under the import of a no-copy buffer's pages (the module docs), the latest
/// last.
#[derive(Default)]
pub(crate) struct ParkedImages {
    images: Mutex<Vec<ParkedImage>>,
    /// How many, read without the lock by a making over pages that park none.
    count: AtomicUsize,
}

impl ParkedImages {
    /// Whether none is parked.
    pub(crate) fn is_empty(&self) -> bool {
        self.count.load(Ordering::Acquire) == 0
    }

    fn push(&self, image: ParkedImage) {
        let mut images = lock(&self.images);
        images.push(image);
        self.count.store(images.len(), Ordering::Release);
    }

    /// The latest parked image of `key`, taken out.
    pub(crate) fn take(&self, key: &ParkKey) -> Option<ParkedImage> {
        let mut images = lock(&self.images);
        let at = images.iter().rposition(|image| image.key == *key)?;
        let image = images.remove(at);
        self.count.store(images.len(), Ordering::Release);
        Some(image)
    }

    /// The parked images `which` names, taken out, in the order of their parks.
    pub(crate) fn take_where(&self, which: impl Fn(&ParkedImage) -> bool) -> Vec<ParkedImage> {
        let mut images = lock(&self.images);
        let (taken, kept): (Vec<ParkedImage>, Vec<ParkedImage>) =
            images.drain(..).partition(|image| which(image));
        *images = kept;
        self.count.store(images.len(), Ordering::Release);
        taken
    }

    /// When each was parked, with its `VkImage`.
    pub(crate) fn ages(&self) -> Vec<(u64, vk::Image)> {
        (lock(&self.images).iter())
            .map(|image| (image.since, image.raw))
            .collect()
    }

    /// The device of the images parked, `None` where none is.
    pub(crate) fn device(&self) -> Option<Device> {
        (lock(&self.images).first()).map(|image| image.device().clone())
    }
}

/// Release `images`, parked images taken out untaken: each `VkImage` destroyed, then its import
/// freed, each counted (`WorkCounts::dma_buf_images_parked_released`, or
/// `exported_dma_buf_images_parked_released`) and let go of by the residency. No command uses
/// them. On any thread, the residency's among them: each destruction is a section of the exit
/// guard, as every Vulkan call of a runtime thread is (`exit_guard::check`), and while the process
/// exits the image is left to its end, as its import is (`DmaBufPages`'s drop).
pub(crate) fn release(images: Vec<ParkedImage>) {
    for image in images {
        let ParkedImage {
            raw,
            import,
            exported,
            ..
        } = image;
        let device = import.device.clone();
        let shared = device.shared();
        if let Some(busy) = exit_guard::enter() {
            // SAFETY: the image is the device's, and no command uses it: every command buffer
            // that used it completed before it was parked, and none took it since.
            unsafe { release::object(shared.vk(), raw) };
            drop(busy);
        }
        // Its import goes after it.
        drop(import);
        match exported {
            true => shared.counts.exported_dma_buf_images_parked_released(1),
            false => shared.counts.dma_buf_images_parked_released(1),
        }
        shared.residency.unparked(&device, 1);
    }
}
