//! A `Shared` or `Managed` texture's CPU copies on the device's staging queue, a copy engine
//! (`device/staging.rs`): where the device has one and the texture copies none on the host from
//! memory the host caches (NVIDIA: an RTX 3090's host-transfer image lies in resizable BAR,
//! `texture/host_copy.rs`), Metal's `replaceRegion` and `getBytes` of such a texture wait for no
//! GPU work of the program's, as Metal's do.
//!
//! - **Which images** ([`Device::stages`]). A new `Shared` or `Managed` texture's image in memory
//!   of its own, of a color format (a queue of transfers alone copies no depth or stencil aspect,
//!   so a depth texture keeps the host's or the submission's copies), made for
//!   copies both ways, of one sample (Vulkan copies nothing into a multisampled image; a clear
//!   writes its zeros). It is made concurrent among the device's families and the staging queue's,
//!   so its uses change no owner: measured on the RTX 3090, 50 clears and 50 copies of a 1080p
//!   image cost the same either way (0.24 ms and 1.32 ms).
//! - **Made.** It is moved to `GENERAL` and zeroed on the staging queue as it is made, from a
//!   staging buffer of zeros (a copy engine clears nothing), waited for
//!   (`FirstMove::record_alone`): no batch finds its move waiting.
//! - **Copied in.** `replace_region` packs the caller's bytes into a staging buffer of their own
//!   before it returns, and queues the copy into the image for the staging queue's next
//!   submission ([`Queued`]), which carries every copy queued so far, in the order they were
//!   queued, a barrier between two into one image: a burst of calls makes one submission, not one
//!   each. That submission is made by the first of the device's next batch or internal
//!   submission, on any `VkQueue`, before its own, which waits for it on the GPU
//!   (`Device::staging_wait`, `device/staging.rs`); a `get_bytes` of any image that copies here,
//!   which carries them before its read; a texture's `VkImage` handed out, and the program's own
//!   use of a `VkQueue`, which wait for it on the host; and the call that queues past
//!   [`STAGED_COPIES`] copies or [`STAGED_BYTES`] bytes, which waits for it on the host. So a
//!   later command buffer's commands, the device's own copies of the texture, and every CPU read
//!   find the bytes written. Counted: `WorkCounts::staging_copies_queued`,
//!   `WorkCounts::staging_flushes`, `WorkCounts::staging_gpu_waits`.
//! - **Copied out.** `get_bytes` copies the image's texels into a staging buffer on the staging
//!   queue, after the copies queued, and lays them out from it into the caller's bytes, waiting
//!   for that submission alone: the copies and one submission's round trip, whatever the program
//!   has in flight. It reads what every command buffer the host saw complete wrote.
//! - **A race is the program's,** as on the host (`host_copy.rs`): a command in flight meanwhile
//!   reads or writes texels the copy may or may not have moved, and one submitted before the
//!   queued copies were (a command buffer committed before the call, still running or waiting on
//!   an event) may read the texels before or after them, as Metal leaves a `replaceRegion` while
//!   the GPU may use the texture. A `get_bytes` runs under the lock a detach takes, so it never
//!   reaches a `VkImage` that went; a queued copy goes into the `VkImage` the image presented at
//!   the call, its own, which a detach keeps for the reattach, and the submission that carries it
//!   holds the image until it completed. While the image presents a stand-in, or a host call's
//!   copy into it still waits for a batch (made while it presented one), the call takes the
//!   submission's path.
//! - **With a decoded twin** (`decoded_twin.rs`; a carried texture) too: `get_bytes` copies its
//!   carrier's blocks, which hold its texels, and `replace_region` copies the blocks into it, then
//!   queues their decode on the CPU, which copies the texels into the twin on the staging queue
//!   (`Image::decode_written`): the twin is made to take the staging queue's copies where its
//!   carrier does.

use super::first_move::{STAGED_BYTES, STAGED_COPIES};
use super::{image_aspects, staging_buffer, transfer_barrier, Image, ImagePlan, Staging};
use crate::device::{lock, Device};
use crate::error::Error;
use crate::exit_guard::{self, Call};
use crate::memory::Placement;
use crate::release;
use ash::vk;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{ImageAspectFlags, ImageUsageFlags};
use std::sync::{Arc, Weak};

/// The work queued for the staging queue's next submission (the module docs), in the device's
/// first moves (`FirstMoves::register_staging`), and what bounds it.
#[derive(Default)]
pub(crate) struct Queued {
    work: Vec<Work>,
    /// The bytes of the staging buffers `work` holds.
    bytes: u64,
    /// Submissions that took work from here and have not yet stored the value they signal, under
    /// the staging queue's lock: while one has not, a submission elsewhere that finds nothing
    /// queued takes the lock before it reads that value ([`Device::flush_staging`]).
    draining: usize,
}

/// One copy queued: of `copy` from the staging buffer `bytes` into `raw`, the `VkImage` of the
/// image `keeper` keeps (its own, or its decoded twin's), held weakly: an image gone takes its
/// copy with it.
struct Work {
    keeper: Weak<Image>,
    raw: vk::Image,
    bytes: StagedBytes,
    copy: vk::BufferImageCopy,
}

/// A staging buffer's objects, which nothing frees as they drop: the queue that holds them
/// frees them ([`StagedBytes::free`]), as it holds no device.
pub(super) struct StagedBytes {
    raw: vk::Buffer,
    placement: Placement,
    len: u64,
}

impl StagedBytes {
    /// A staging buffer of `len` bytes, host-visible, shared as `Staging`'s.
    fn new(device: &Device, len: u64) -> Result<StagedBytes, Error> {
        let (raw, placement) = staging_buffer(device, len)?;
        Ok(StagedBytes {
            raw,
            placement,
            len,
        })
    }

    /// Its bytes, through the mapping.
    fn bytes(&mut self) -> &mut [u8] {
        let Some(mapped) = self.placement.mapped else {
            unreachable!("checked when the staging buffer was made");
        };
        // SAFETY: the mapping covers the placement's `len` bytes, which only this staging buffer
        // uses; the memory is host-coherent, and no submission uses it yet.
        unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), self.len as usize) }
    }

    /// Free it; no pending submission uses it.
    fn free(self, device: &Device) {
        // Once the process is exiting, the buffer is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = device.shared();
        // SAFETY: the caller's.
        unsafe { release::object(shared.vk(), self.raw) };
        lock(&shared.memory).free(shared.vk(), &self.placement);
    }
}

impl Queued {
    /// Queue `work`; whether the work queued now passed [`STAGED_COPIES`] or [`STAGED_BYTES`].
    fn push(&mut self, work: Work) -> bool {
        self.bytes += work.bytes.len;
        self.work.push(work);
        self.work.len() > STAGED_COPIES || self.bytes > STAGED_BYTES
    }

    /// Whether a submission elsewhere has queued work to carry or to wait for (the docs of
    /// `draining`).
    pub(super) fn waits(&self) -> bool {
        !self.work.is_empty() || self.draining > 0
    }

    /// The device's teardown: free the staging buffers queued; the device is idle.
    pub(super) fn destroy(&mut self, device: &ash::Device, memory: &mut crate::memory::Memory) {
        for work in self.work.drain(..) {
            // SAFETY: the device is idle and no submission used the buffer.
            unsafe { release::object(device, work.bytes.raw) };
            memory.free(device, &work.bytes.placement);
        }
        self.bytes = 0;
    }
}

/// What a staging submission that carried queued work holds until it completed: the images it
/// copied into and the staging buffers it copied from, then freed. Its drop waits on the host for
/// the submission where the host has not seen it complete (a batch that holds it completed after
/// it, so that wait returns at once).
pub(crate) struct Flushed {
    device: Device,
    value: u64,
    bytes: Vec<StagedBytes>,
    _images: Vec<Arc<Image>>,
}

impl Drop for Flushed {
    fn drop(&mut self) {
        // A failed wait (a lost device) leaves nothing running that uses them.
        let _ = self.device.wait_staging(self.value);
        for bytes in self.bytes.drain(..) {
            bytes.free(&self.device);
        }
    }
}

/// The queued work taken by one submission: each copy whose image is still there, and the
/// staging buffers of those gone, to free.
struct Taken {
    live: Vec<(Arc<Image>, Work)>,
    gone: Vec<StagedBytes>,
}

impl Taken {
    /// Record the copies in order, a barrier before each into an image an earlier one wrote:
    /// copies into one image keep their order.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording on the staging queue, after a barrier from every command
    /// before; every image is live, in `GENERAL`, shared with the staging queue; every staging
    /// buffer holds its bytes.
    unsafe fn record(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        let mut written: Vec<vk::Image> = Vec::new();
        for (_, work) in &self.live {
            if written.contains(&work.raw) {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                // SAFETY: the caller's.
                unsafe {
                    raw.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                exit_guard::called(Call::CmdPipelineBarrier);
                written.clear();
            }
            // SAFETY: the caller's.
            unsafe {
                raw.cmd_copy_buffer_to_image(
                    command_buffer,
                    work.bytes.raw,
                    work.raw,
                    vk::ImageLayout::GENERAL,
                    &[work.copy],
                );
            }
            exit_guard::called(Call::CmdCopyBufferToImage);
            written.push(work.raw);
        }
    }

    /// What the submission that carried it holds until it completed, the value it signals.
    fn flushed(self, device: &Device, value: u64) -> Flushed {
        for bytes in self.gone {
            bytes.free(device);
        }
        let (images, bytes) = self.live.into_iter().map(|(i, w)| (i, w.bytes)).unzip();
        Flushed {
            device: device.clone(),
            value,
            bytes,
            _images: images,
        }
    }
}

impl Device {
    /// Whether a new `Shared` or `Managed` texture's image of `plan`, for a texture of
    /// `pixel_format`, copies on the staging queue where it takes no host copies from host-cached
    /// memory (the module docs): the device has a staging queue, the image is of color alone and
    /// not a depth texture's, and it is made for copies both ways.
    pub(super) fn stages(&self, plan: &ImagePlan, pixel_format: PixelFormat) -> bool {
        let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        self.has_staging_queue()
            && image_aspects(plan.format) == ImageAspectFlags::COLOR
            && !is_depth_stencil(pixel_format)
            && plan.usage.contains(both)
    }

    /// Make the staging queue's submission of the work queued so far (the module docs), for a
    /// submission of the device's about to be made, which waits for it on the GPU
    /// ([`Device::staging_wait`]); what that submission holds until it completed, `None` where
    /// nothing was queued. Where another submission took queued work and has not yet stored the
    /// value it signals, this waits for the staging queue's lock until it has.
    pub(crate) fn flush_staging(&self) -> Result<Option<Flushed>, Error> {
        if !self.has_staging_queue() {
            return Ok(None);
        }
        let firsts = &self.shared().textures.first_moves;
        if !lock(&firsts.staging).waits() {
            return Ok(None);
        }
        self.submit_queued(None::<fn(&ash::Device, vk::CommandBuffer)>)
    }

    /// [`flush_staging`](Self::flush_staging), waited for on the host: for a host call whose
    /// result needs what it copied (a texture's `VkImage` handed out, the program's own use of a
    /// `VkQueue`), or that queued past the bound.
    pub(crate) fn flush_staging_waited(&self) -> Result<(), Error> {
        if let Some(flushed) = self.flush_staging()? {
            self.wait_staging(flushed.value)?;
        }
        Ok(())
    }

    /// Take the queued work under the staging queue's lock and submit it there in one
    /// submission, then `after` (a read of the caller's) behind a barrier from it: copies in
    /// queued order, a barrier before each into an image an earlier one wrote. What the
    /// submission holds until it completed, and the value it signals; `None` where nothing was
    /// queued (no copy into an image still there) and there is no `after`, and nothing is
    /// submitted. Work taken and not submitted goes back to the queue's front.
    fn submit_queued(
        &self,
        after: Option<impl FnOnce(&ash::Device, vk::CommandBuffer)>,
    ) -> Result<Option<Flushed>, Error> {
        let firsts = &self.shared().textures.first_moves;
        let mut staging = self.staging_lock()?;
        let work = {
            let mut queued = lock(&firsts.staging);
            if queued.work.is_empty() && after.is_none() {
                return Ok(None);
            }
            queued.bytes = 0;
            queued.draining += 1;
            std::mem::take(&mut queued.work)
        };
        let mut taken = Taken {
            live: Vec::with_capacity(work.len()),
            gone: Vec::new(),
        };
        for work in work {
            match work.keeper.upgrade() {
                Some(image) => taken.live.push((image, work)),
                None => taken.gone.push(work.bytes),
            }
        }
        let copies = taken.live.len();
        let submitted = match (copies, after) {
            (0, None) => Ok(None),
            (copies, after) => staging
                .submit(|raw, command_buffer, _| {
                    if copies > 0 {
                        transfer_barrier(raw, command_buffer, true);
                        // SAFETY: recording on the staging queue after a barrier from every
                        // command before; each image is held live (`taken`), rests in `GENERAL`
                        // (moved as it was made) and is shared with the staging queue; each
                        // staging buffer holds the bytes packed into it.
                        unsafe { taken.record(raw, command_buffer) };
                    }
                    match after {
                        Some(after) => after(raw, command_buffer),
                        None => transfer_barrier(raw, command_buffer, false),
                    }
                })
                .map(Some),
        };
        // The value is stored (or no submission made) before another submission may read it.
        lock(&firsts.staging).draining -= 1;
        drop(staging);
        match submitted {
            Ok(Some(value)) => {
                if copies > 0 {
                    self.shared().counts.staging_flush();
                }
                Ok(Some(taken.flushed(self, value)))
            }
            Ok(None) => {
                for bytes in taken.gone {
                    bytes.free(self);
                }
                Ok(None)
            }
            Err(error) => {
                let mut queued = lock(&firsts.staging);
                let later = std::mem::take(&mut queued.work);
                queued.bytes = 0;
                for (_, work) in taken.live {
                    queued.push(work);
                }
                for work in later {
                    queued.push(work);
                }
                drop(queued);
                for bytes in taken.gone {
                    bytes.free(self);
                }
                Err(error)
            }
        }
    }
}

impl Image {
    /// Whether its CPU copies may run on the staging queue: it was made so (a decoded twin's
    /// decode is staged, the module docs).
    pub(super) fn takes_staging_copies(&self) -> bool {
        self.staging_copies
    }

    /// Whether nothing of a host call's copy into it waits for a batch: no copy staged while it
    /// presented a stand-in (`texture/first_move.rs`).
    fn nothing_staged(&self) -> bool {
        let at: *const Image = self;
        !(self.device.shared().textures.first_moves).texels_wait(at.cast())
    }

    /// Queue the copy of `len` bytes, laid out as `copy` reads them from a buffer, which `fill`
    /// writes into a staging buffer of their own before this returns, into the image for the
    /// staging queue's next submission (the module docs); the image [takes staging
    /// copies](Self::takes_staging_copies), and `keeper` is what holds it (itself, or the texture
    /// whose decoded twin it is). Whether it did: not while it presents a stand-in, and then
    /// `fill` is not called. Where the image's `VkImage` was handed out (`Texture::vk_image`), whose
    /// user the runtime does not see, or the work queued passed its bound, the submission is
    /// made now and waited for.
    pub(super) fn staging_write(
        &self,
        keeper: &Arc<Image>,
        copy: vk::BufferImageCopy,
        len: u64,
        fill: impl FnOnce(&mut [u8]),
    ) -> Result<bool, Error> {
        let raw = {
            let detached = lock(&self.detached);
            if detached.is_some() || !self.nothing_staged() {
                return Ok(false);
            }
            self.raw.get()
        };
        let mut bytes = StagedBytes::new(&self.device, len)?;
        fill(bytes.bytes());
        let work = Work {
            keeper: Arc::downgrade(keeper),
            raw,
            bytes,
            copy,
        };
        let shared = self.device.shared();
        let past = lock(&shared.textures.first_moves.staging).push(work);
        shared.counts.staging_copy_queued();
        if past || self.handed_out() {
            self.device.flush_staging_waited()?;
        }
        Ok(true)
    }

    /// Copy the image's texels `copy` reaches, `len` bytes laid out as `copy` writes them to a
    /// buffer, on the staging queue, after the copies queued for it there, and hand them to
    /// `take` where they lie, in the staging buffer; the image [takes staging
    /// copies](Self::takes_staging_copies). Whether it did: not while it presents a stand-in.
    pub(super) fn staging_read(
        &self,
        copy: vk::BufferImageCopy,
        len: u64,
        take: impl FnOnce(&[u8]),
    ) -> Result<bool, Error> {
        let detached = lock(&self.detached);
        if detached.is_some() || !self.nothing_staged() {
            return Ok(false);
        }
        let mut staging = Staging::new(&self.device, len)?;
        let (buffer, image) = (staging.raw, self.raw.get());
        let read = |raw: &ash::Device, command_buffer: vk::CommandBuffer| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the staging command buffer is recording; the staging buffer and the image
            // are live (the detach lock keeps the image's own `VkImage` presented), and the image
            // rests in `GENERAL` (moved as it was made), shared with the staging queue.
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
        };
        let Some(flushed) = self.device.submit_queued(Some(read))? else {
            unreachable!("a submission with a read is made");
        };
        self.device.wait_staging(flushed.value)?;
        drop(flushed);
        drop(detached);
        take(staging.bytes());
        Ok(true)
    }
}
