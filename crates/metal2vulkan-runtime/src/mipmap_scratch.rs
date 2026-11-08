//! The scratch buffer `generate_mipmaps`' compute filter runs in (`mipmap_filter.rs`): kept by
//! the device between calls and grown to the largest chain, so that a call makes no buffer, and
//! clears none, once the device holds one large enough.
//!
//! - **Leases.** A call takes the device's spare block where it holds at least the chain's
//!   bytes, else makes a block of the chain's bytes (releasing a spare too small). Its command
//!   buffer holds the [`Lease`] until it completes, or until it is dropped unsubmitted; then the
//!   block is given back. No other call gets a block while it is leased, so a block an
//!   uncompleted command buffer uses is never reused, and calls in flight at once, from any
//!   thread, each have their own.
//! - **Bounded.** The device keeps at most one spare block: given back beside another, the
//!   smaller is released ([`Spare`]). Leased blocks are bounded by the work in flight.
//! - **Raw.** A block is a `VkBuffer` bound to private memory, outside the Metal address space,
//!   not a `Buffer`, which would hold the device from the device's own spare. The device
//!   releases its spare as it is destroyed.
//! - **Not cleared.** The chain's command writes every byte it reads before reading it (the first
//!   level copied in, every later level filled with zeros where its texels are narrower than a
//!   word and else written whole by its dispatch, each level's words uploaded), so a block's
//!   earlier bytes are never read.

use super::device::{lock, DeviceShared};
use super::error::{vulkan, Error};
use super::exit_guard;
use super::memory::{Memory, MemoryClass, Placement};
use super::Device;
use crate::release;
use ash::vk;

/// A filter chain's block: storage, and a transfer's source and destination.
const MIPMAP_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// The spare-block decisions, pure so that the CPU tests cover them: the one block of `T` the
/// device keeps between leases, with its size in bytes.
#[derive(Debug)]
pub(super) struct Spare<T> {
    idle: Option<(T, u64)>,
}

impl<T> Default for Spare<T> {
    fn default() -> Spare<T> {
        Spare { idle: None }
    }
}

/// What a call for a block does ([`Spare::take`]).
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Take<T> {
    /// Leases the spare, of its size.
    Reuse(T, u64),
    /// Makes a block, after releasing the spare too small, where there was one.
    Make(Option<T>),
}

impl<T> Spare<T> {
    /// A block of at least `need` bytes for a lease: the spare where it is that large, taken out
    /// until it is given back.
    pub(super) fn take(&mut self, need: u64) -> Take<T> {
        match self.idle.take() {
            Some((block, size)) if size >= need => Take::Reuse(block, size),
            stale => Take::Make(stale.map(|(block, _)| block)),
        }
    }

    /// Give back `block` of `size` bytes as its lease ends: the block to release, the smaller of
    /// it and the spare, where there is one.
    pub(super) fn give(&mut self, block: T, size: u64) -> Option<T> {
        match self.idle.take() {
            Some((idle, idle_size)) if idle_size >= size => {
                self.idle = Some((idle, idle_size));
                Some(block)
            }
            smaller => {
                self.idle = Some((block, size));
                smaller.map(|(block, _)| block)
            }
        }
    }
}

/// A block: its buffer and the memory it is bound to.
#[derive(Debug)]
pub(super) struct Block {
    raw: vk::Buffer,
    placement: Placement,
}

impl Block {
    /// The block's buffer.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// The device address of the block's byte 0, where it was made addressed
    /// ([`Device::make_block`]) on a device that enables `bufferDeviceAddress`.
    pub(super) fn address(&self) -> Option<u64> {
        self.placement.address
    }

    /// Release the buffer and its memory.
    ///
    /// # Safety
    ///
    /// No command that uses the buffer is pending.
    pub(super) unsafe fn release(self, device: &ash::Device, memory: &mut Memory) {
        // SAFETY: the caller's.
        unsafe { release::object(device, self.raw) };
        memory.free(device, &self.placement);
    }

    /// [`release`](Self::release) it with `shared`'s functions, reached inside a section of the
    /// exit guard, whatever thread lets it go: a lease ends as its command buffer completes, on
    /// the queue's waiter that finishes it, a runtime thread. Once the process is exiting, the
    /// buffer and its memory are left to its end. Whether it was released.
    ///
    /// # Safety
    ///
    /// As [`release`](Self::release).
    pub(super) unsafe fn release_in_section(self, shared: &DeviceShared) -> bool {
        // The Vulkan teardown is a section of the exit guard.
        let Some(_busy) = exit_guard::enter() else {
            return false;
        };
        // SAFETY: the caller's.
        unsafe { self.release(shared.vk(), &mut lock(&shared.memory)) };
        true
    }
}

/// The device's spare block ([`DeviceShared::mipmap_scratch`](super::device::DeviceShared)).
pub(super) type MipmapScratch = Spare<Block>;

impl Spare<Block> {
    /// Release the spare as the device is destroyed.
    pub(super) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        if let Some((block, _)) = self.idle.take() {
            // SAFETY: the device is idle, so no command uses the block.
            unsafe { block.release(device, memory) };
        }
    }
}

/// A leased block of `size` bytes, given back to its device as it drops (the module docs).
pub(super) struct Lease {
    device: Device,
    block: Option<Block>,
    size: u64,
}

impl Lease {
    /// The leased buffer.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.block
            .as_ref()
            .map_or(vk::Buffer::null(), |block| block.raw)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        let shared = self.device.shared();
        let released = lock(&shared.mipmap_scratch).give(block, self.size);
        if let Some(block) = released {
            // SAFETY: a lease ends when its command buffer completes or is dropped unsubmitted,
            // and the spare is leased to none, so no command uses either block.
            unsafe { block.release_in_section(shared) };
        }
    }
}

impl Device {
    /// A lease of a scratch block of at least `need` bytes for a mipmap filter chain (the module
    /// docs); `None` where the device's memory has no room for it.
    pub(super) fn mipmap_scratch(&self, need: u64) -> Result<Option<Lease>, Error> {
        let shared = self.shared();
        let taken = lock(&shared.mipmap_scratch).take(need);
        let (block, size) = match taken {
            Take::Reuse(block, size) => (block, size),
            Take::Make(stale) => {
                if let Some(block) = stale {
                    // SAFETY: the spare was leased to none, so no command uses it.
                    unsafe { block.release_in_section(shared) };
                }
                match self.make_block(need, MIPMAP_USAGE, false) {
                    Ok(block) => (block, need),
                    Err(Error::OutOfMemory(_)) => return Ok(None),
                    Err(error) => return Err(error),
                }
            }
        };
        Ok(Some(Lease {
            device: self.clone(),
            block: Some(block),
            size,
        }))
    }

    /// A block of `size` bytes in private memory, of `usage` (a filter chain's
    /// [`MIPMAP_USAGE`]; a command buffer's pool's, `gpu_pools.rs`), with a device address where
    /// `addressed` and the device has device addresses (`memory::allocate_addressed`).
    pub(super) fn make_block(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        addressed: bool,
    ) -> Result<Block, Error> {
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared
            .sharing()
            .buffer(vk::BufferCreateInfo::default().size(size).usage(usage));
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::command_buffer::completion::trace::mark("block: buffer made");
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let allocated = if addressed {
            crate::memory::allocate_addressed(
                &shared.memory,
                raw_device,
                &requirements,
                MemoryClass::Private,
                &shared.counts,
            )
        } else {
            crate::memory::allocate(
                &shared.memory,
                raw_device,
                &requirements,
                MemoryClass::Private,
            )
        };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::command_buffer::completion::trace::mark("block: memory placed");
        let placed = allocated.and_then(|placement| {
            // SAFETY: the placement meets the buffer's requirements and is bound nowhere
            // else.
            match unsafe {
                crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
            } {
                Ok(()) => Ok(placement),
                Err(error) => {
                    lock(&shared.memory).free(raw_device, &placement);
                    Err(vulkan(error))
                }
            }
        });
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::command_buffer::completion::trace::mark("block: bound");
        match placed {
            Ok(placement) => Ok(Block { raw, placement }),
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                Err(error)
            }
        }
    }
}
