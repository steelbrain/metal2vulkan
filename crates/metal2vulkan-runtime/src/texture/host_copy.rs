//! A `Shared` or `Managed` texture's CPU copies on the host: Metal's `replaceRegion` and `getBytes`
//! of such a texture are CPU copies that wait for no GPU work, and so are the runtime's where the
//! device has `VK_EXT_host_image_copy`.
//!
//! - **Which images.** Where the device enabled `hostImageCopy` and copies on the host to and
//!   from `GENERAL` ([`reaches_general`]), a new `Shared` or `Managed` texture's image in memory
//!   of its own (`Device::new_image_owned`) takes host-transfer usage where its format has
//!   `HOST_IMAGE_TRANSFER` at optimal tiling and the device makes and places the image with it,
//!   in memory of its class the host caches; elsewhere it is made as before, and its copies go
//!   through a submission. A multisampled image never takes the usage: Vulkan copies nothing into
//!   or out of one, and Metal makes no CPU copy of such a texture; a clear writes its zeros.
//!   Metal's `Shared` and `Managed` textures are laid out without lossless compression; the usage
//!   may lay the image out so too (RADV leaves out DCC).
//! - **An image whose texel block holds more than one texel takes the usage only where its
//!   driver's copies of such an image are its own** ([`reaches_format`]). The usage puts the image
//!   in host-visible memory, and a driver may then copy the whole image out of that memory as the
//!   memory is bound, with no copy the program asked for, through a row pitch of its own whose own
//!   check it then fails: `RuntimeDriverDefect::HostTransferImagesCopiedAtBind`, which
//!   `driver_defects.rs` measures on MoltenVK with the backtrace and the numbers. There such an
//!   image is made as before and its CPU copies go through a submission, as on a device without
//!   `VK_EXT_host_image_copy`; on every other driver a block format's CPU copies stay on the
//!   host.
//! - **In host-cached memory where the device has it**. A host copy reads and writes through the
//!   host's mapping of the image's memory, and the host reads uncached memory slowly
//!   (`memory::host_cached`): NVIDIA places a texture's host-transfer image in resizable BAR, its
//!   one device-local type for it, where an RTX 3090's `vkCopyImageToMemoryEXT` read a 1920 x 1080
//!   `BGRA8Unorm` image's 8 MiB in 758 ms at best (the consumer's RTX 5080 laptop: 1.0 to 2.55 s),
//!   and RADV on a Radeon 780M in its uncached device-local type (144 ms). The image takes memory
//!   of its class the host caches where the device has it among the image's types
//!   (`memory::host_cached_placement`; MoltenVK's shared type on Apple GPUs and lavapipe's one type
//!   are), only of a coherent type: the runtime flushes no mapped range, and ANV's cached type on
//!   a GPU without a cache shared with the host (Arrow Lake-S's Xe-LPG), which is not coherent,
//!   left the GPU reading stale 64-byte lines of such images (`memory::host_cached`); else, on a
//!   device with unified memory, `Shared` memory the host caches (a Radeon 780M's
//!   system memory, where its GPU uses the image as fast as in device-local memory or faster, and a
//!   1080p `get_bytes` takes 2.6 ms where it took 108 ms from RADV's uncached device-local type);
//!   else the class's memory as before, counted (`WorkCounts::uncached_host_images`): the copies
//!   stay on the host there too, slowly, since a copy through a submission on the device's first
//!   queue would wait behind the program's work, and forever behind a command buffer waiting on an
//!   event only a later command of the calling thread signals. Where the device reserved a staging
//!   queue on a copy engine (`device/staging.rs`; NVIDIA), a color texture copies on the host only
//!   from host-cached memory, and else on that queue (`texture/staging_copy.rs`).
//! - **Made.** The making moves the image to `GENERAL` and writes zeros over every texel on the
//!   host ([`start`]), as Metal's new `Shared` and `Managed` textures read zero: no first move
//!   waits for a submission (`first_move.rs`). Where the device enabled
//!   `VK_EXT_zero_initialize_device_memory`'s `zeroInitializeDeviceMemory`, its blocks of images
//!   are allocated zero-initialized (`Memory::zero_image_blocks`), the image is made in
//!   `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT`, and where it lies past every range placed in its
//!   block before (`Placement::fresh`) it moves to `GENERAL` out of that layout on the host
//!   ([`transition_from`]): its zeros are the driver's, and no write of the host's touches its
//!   memory as it is made. Elsewhere it moves out of `UNDEFINED`, which any image may whatever
//!   its layout, and is zeroed as above. On an RX 7900 XTX (RADV, Mesa 26.2.6) the host's write
//!   of zeros into new memory waited for the program's queued GPU work, 0.2 to 1.1 s behind
//!   about 1.5 s of fills, and the zeroed start took 0.3 ms (`tests/zeroed_start.rs`). Counted:
//!   `WorkCounts::images_started_zeroed`. Where the GPU had idled into runtime suspend, the first
//!   CPU write into a block made then waited there instead, 516 to 684 ms, in the kernel's clear
//!   of the block's memory; the blocks of the pool these images place in are made ready ahead,
//!   every page of their mapping read before any image is placed in them (`memory/ready.rs`).
//! - **Copied.** `replace_region` copies the caller's bytes into the image on the host, straight
//!   from them where they are laid out packed
//!   (`HostLayout::is_packed`), else from packed bytes, and `get_bytes` copies the image's texels
//!   out, into the
//!   caller's bytes where they are laid out packed (`HostLayout::is_packed`), else into packed
//!   bytes it lays out from: no submission, no wait. The host's writes are visible to every command submitted after
//!   the call (a submission orders the host's writes before it); a batch's writes are visible to
//!   the host once it completed (every batch ends with its writes made available to the host,
//!   `tracker.rs`).
//! - **A race is the program's.** A copy waits for no command that uses the texture, as Metal's
//!   does not: a command in flight meanwhile reads or writes texels the copy may or may not have
//!   moved. The copy reaches only the texture's own image and the caller's bytes: a race gives
//!   that texture wrong texels, never another resource's bytes or a crash. A copy runs under the
//!   lock a detach takes to present a stand-in (`detach.rs`), so it never reaches a `VkImage` that
//!   went.

use super::Image;
use crate::device::lock;
use crate::error::{vulkan, Error};
use crate::exit_guard;
use ash::ext::host_image_copy::DeviceFn;
use ash::vk;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::Format;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The defect that keeps a block format's image off host copies (the module docs): the driver
/// copies such an image out of its host-visible memory as the memory is bound, through a row pitch
/// of its own, and its own check of that copy fails.
const COPIED_AT_BIND: RuntimeDriverDefect = RuntimeDriverDefect::HostTransferImagesCopiedAtBind;

/// Whether host copies reach an image of `format` on a device whose driver has `defects`: every
/// image, unless its texel block holds more than one texel and the driver copies such an image out
/// of its memory as the memory is bound ([`COPIED_AT_BIND`], the module docs). A format with no
/// block is no image format and takes none either.
pub(crate) fn reaches_format(format: Format, defects: &BTreeSet<RuntimeDriverDefect>) -> bool {
    let Some(block) = format.block() else {
        return false;
    };
    block.texels == 1 || !defects.contains(&COPIED_AT_BIND)
}

/// Whether the device of `physical` copies on the host to and from images in `GENERAL`: both of
/// its lists of layouts for host copies hold it.
///
/// # Safety
///
/// `physical` belongs to `instance`, created at Vulkan 1.1 or later, and has
/// `VK_EXT_host_image_copy` (or Vulkan 1.4).
pub(crate) unsafe fn reaches_general(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut counts = vk::PhysicalDeviceHostImageCopyPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut counts);
    // SAFETY: the caller's contract; the structure chained is the extension's.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let mut sources = vec![vk::ImageLayout::UNDEFINED; counts.copy_src_layout_count as usize];
    let mut destinations = vec![vk::ImageLayout::UNDEFINED; counts.copy_dst_layout_count as usize];
    let mut layouts = vk::PhysicalDeviceHostImageCopyPropertiesEXT::default()
        .copy_src_layouts(&mut sources)
        .copy_dst_layouts(&mut destinations);
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut layouts);
    // SAFETY: as above; the arrays hold the counts the device answered.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let written = (
        layouts.copy_src_layout_count as usize,
        layouts.copy_dst_layout_count as usize,
    );
    let general = vk::ImageLayout::GENERAL;
    sources[..written.0.min(sources.len())].contains(&general)
        && destinations[..written.1.min(destinations.len())].contains(&general)
}

/// Move the new `image` (no command or host copy used it yet) of `aspects` from `UNDEFINED` to
/// `GENERAL` on the host, whatever layout it was made in, then copy `zeros` over every texel
/// `copies` reach (`zero_copies`, each within `zeros` from its start).
///
/// # Safety
///
/// `functions` are `device`'s, `image` is a live image of it with host-transfer usage, bound to
/// memory, which nothing else uses.
pub(super) unsafe fn start(
    functions: &DeviceFn,
    device: vk::Device,
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
    zeros: &[u8],
    copies: &[vk::BufferImageCopy],
) -> Result<(), Error> {
    // SAFETY: the caller's contract.
    unsafe { transition(functions, device, image, aspects) }?;
    // SAFETY: the caller's contract; the image is in `GENERAL` now, and every copy reads within
    // `zeros`.
    unsafe { write(functions, device, image, zeros, copies) }
}

/// Move the new `image` (no command or host copy used it yet) of `aspects` from `UNDEFINED` to
/// `GENERAL` on the host, writing no texel (a decoded twin's, `Start::Moved`).
///
/// # Safety
///
/// As [`start`]'s.
pub(super) unsafe fn transition(
    functions: &DeviceFn,
    device: vk::Device,
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
) -> Result<(), Error> {
    // SAFETY: the caller's contract.
    unsafe {
        transition_from(
            functions,
            device,
            image,
            aspects,
            vk::ImageLayout::UNDEFINED,
        )
    }
}

/// `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` (`VK_EXT_zero_initialize_device_memory`), which
/// `ash 0.38` predates: the layout of a new image whose memory reads zero, which a move out of it
/// keeps.
pub(super) const ZERO_INITIALIZED: vk::ImageLayout = vk::ImageLayout::from_raw(1_000_620_000);

/// [`transition`] of an image in `from` (`UNDEFINED`, which any image may move out of whatever
/// its layout, or [`ZERO_INITIALIZED`] for an image over bytes that read zero, which the move
/// keeps).
///
/// # Safety
///
/// As [`start`]'s; the image is in `from` or `from` is `UNDEFINED`, and where `from` is
/// [`ZERO_INITIALIZED`], the device enabled `zeroInitializeDeviceMemory` and the image is bound
/// to bytes of memory allocated with `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` that no
/// resource used.
pub(super) unsafe fn transition_from(
    functions: &DeviceFn,
    device: vk::Device,
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
    from: vk::ImageLayout,
) -> Result<(), Error> {
    let transition = vk::HostImageLayoutTransitionInfoEXT::default()
        .image(image)
        .old_layout(from)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspects,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        });
    // SAFETY: the caller's contract; the image is in `from`, or `from` is `UNDEFINED`, which a
    // move may name in place of any layout, and the move takes every subresource of it.
    unsafe { (functions.transition_image_layout_ext)(device, 1, &transition) }
        .result()
        .map_err(vulkan)
}

/// Copy `bytes` into `image`, in `GENERAL`, as `copies` read them from a buffer of `bytes`.
///
/// # Safety
///
/// `functions` are `device`'s, `image` is a live image of it with host-transfer usage, in
/// `GENERAL`; every copy lies within the image and reads within `bytes`.
unsafe fn write(
    functions: &DeviceFn,
    device: vk::Device,
    image: vk::Image,
    bytes: &[u8],
    copies: &[vk::BufferImageCopy],
) -> Result<(), Error> {
    let regions: Vec<vk::MemoryToImageCopyEXT<'_>> = (copies.iter())
        .map(|copy| vk::MemoryToImageCopyEXT {
            p_host_pointer: bytes[copy.buffer_offset as usize..].as_ptr().cast(),
            memory_row_length: copy.buffer_row_length,
            memory_image_height: copy.buffer_image_height,
            image_subresource: copy.image_subresource,
            image_offset: copy.image_offset,
            image_extent: copy.image_extent,
            ..Default::default()
        })
        .collect();
    let info = vk::CopyMemoryToImageInfoEXT::default()
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .regions(&regions);
    // SAFETY: the caller's contract.
    unsafe { (functions.copy_memory_to_image_ext)(device, &info) }
        .result()
        .map_err(vulkan)
}

impl crate::Device {
    /// Whether the device copies a color texture on the host: it copies on the host
    /// ([`reaches_general`]), host copies reach `R8G8B8A8_UNORM` ([`reaches_format`]), which has
    /// `HOST_IMAGE_TRANSFER` at optimal tiling, and a color texture's copies do not go to the
    /// staging queue for want of memory the host caches (`Device::host_placed_image`'s tests,
    /// `texture/staging_copy.rs`). The pool such images place in gets its blocks made ready ahead
    /// (`memory/ready.rs`).
    pub(crate) fn copies_color_on_the_host(&self) -> bool {
        let shared = self.shared();
        let format = Format::R8G8B8A8_UNORM;
        let staged = self.has_staging_queue()
            && !lock(&shared.memory).has_host_cached(crate::memory::MemoryClass::Private);
        shared.host_copies.is_some()
            && reaches_format(format, &self.properties().driver_defects)
            && !staged
            && (shared.textures.facts(self.compile_device().profile()))
                .optimal(format)
                .contains(metal2vulkan::vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER)
    }
}

impl Image {
    /// Whether its CPU copies run on the host: it was made with host-transfer usage, moved and
    /// zeroed on the host (the module docs). With a decoded twin too: a read copies the carrier's
    /// blocks, and a write's decode into the twin is made on the CPU and copied into the twin on
    /// the host (`Image::decode_written`).
    pub(super) fn takes_host_copies(&self) -> bool {
        self.host_copies && self.device.shared().host_copies.is_some()
    }

    /// Copy `bytes`, laid out as `copy` reads them from a buffer, into the image on the host;
    /// the image [takes host copies](Self::takes_host_copies). Whether it did: not while the image
    /// presents a stand-in (its texture's bytes detached meanwhile), whose image takes none.
    pub(super) fn host_write(
        &self,
        copy: vk::BufferImageCopy,
        bytes: &[u8],
    ) -> Result<bool, Error> {
        let shared = self.device.shared();
        let Some(functions) = shared.host_copies.as_ref() else {
            return Ok(false);
        };
        let device = shared.vk().handle();
        let detached = lock(&self.detached);
        if detached.is_some() {
            return Ok(false);
        }
        // SAFETY: the image is live, made with host-transfer usage and in `GENERAL` since it
        // was made; the caller checked the copy against the image's plan and `bytes` against its
        // layout. The lock keeps the image's own `VkImage` presented.
        unsafe { write(functions, device, self.raw.get(), bytes, &[copy])? };
        drop(detached);
        shared.counts.host_texel_copy();
        Ok(true)
    }

    /// [`host_write`](Self::host_write) of `copies`, each reading `bytes` at its offset, into
    /// disjoint texels of an image (a decoded twin), on up to `threads`
    /// threads at once: the Vulkan specification makes host copies race only where they access
    /// overlapping image memory, which disjoint texels never share. Whether it did: not while
    /// the image presents a stand-in.
    pub(super) fn host_write_disjoint(
        &self,
        copies: &[vk::BufferImageCopy],
        bytes: &[u8],
        threads: usize,
    ) -> Result<bool, Error> {
        let shared = self.device.shared();
        let Some(functions) = shared.host_copies.as_ref() else {
            return Ok(false);
        };
        let device = shared.vk().handle();
        let detached = lock(&self.detached);
        if detached.is_some() {
            return Ok(false);
        }
        let image = self.raw.get();
        let next = AtomicUsize::new(0);
        let failed = Mutex::new(None);
        let work = || {
            while let Some(copy) = copies.get(next.fetch_add(1, Ordering::Relaxed)) {
                // SAFETY: the image is live, made with host-transfer usage and in `GENERAL` since
                // it was made; the caller checked each copy against the image's plan and `bytes`
                // against its layout, and the copies' texels are disjoint. The lock keeps the
                // image's own `VkImage` presented.
                let wrote =
                    unsafe { write(functions, device, image, bytes, std::slice::from_ref(copy)) };
                if let Err(error) = wrote {
                    *lock(&failed) = Some(error);
                }
            }
        };
        std::thread::scope(|scope| {
            for _ in 1..threads.min(copies.len()) {
                let builder = std::thread::Builder::new().name("metal2vulkan-twin-decode".into());
                if exit_guard::spawn_scoped(scope, builder, work).is_err() {
                    break;
                }
            }
            work();
        });
        drop(detached);
        if let Some(error) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(error);
        }
        shared.counts.host_texel_copy();
        Ok(true)
    }

    /// Copy the image's texels `copy` reaches into `bytes` on the host, laid out as `copy` writes
    /// them to a buffer; the image [takes host copies](Self::takes_host_copies). Whether it did:
    /// not while the image presents a stand-in.
    pub(super) fn host_read(
        &self,
        copy: vk::BufferImageCopy,
        bytes: &mut [u8],
    ) -> Result<bool, Error> {
        let shared = self.device.shared();
        let Some(functions) = shared.host_copies.as_ref() else {
            return Ok(false);
        };
        let device = shared.vk().handle();
        let detached = lock(&self.detached);
        if detached.is_some() {
            return Ok(false);
        }
        let region = vk::ImageToMemoryCopyEXT {
            p_host_pointer: bytes[copy.buffer_offset as usize..].as_mut_ptr().cast(),
            memory_row_length: copy.buffer_row_length,
            memory_image_height: copy.buffer_image_height,
            image_subresource: copy.image_subresource,
            image_offset: copy.image_offset,
            image_extent: copy.image_extent,
            ..Default::default()
        };
        let info = vk::CopyImageToMemoryInfoEXT::default()
            .src_image(self.raw.get())
            .src_image_layout(vk::ImageLayout::GENERAL)
            .regions(std::slice::from_ref(&region));
        // SAFETY: as `host_write`'s; the copy writes within `bytes`, which the caller sized to
        // its layout.
        unsafe { (functions.copy_image_to_memory_ext)(device, &info) }
            .result()
            .map_err(vulkan)?;
        drop(detached);
        shared.counts.host_texel_copy();
        Ok(true)
    }
}
