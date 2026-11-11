//! The device's memory of a copied host buffer's base (`copied.rs`, "A base once it may"): the
//! bytes the pages and the copy last agreed on, in device-local memory the host maps, where the
//! device has such memory with room ([`memory::allocate_copied_base`]) and a staging queue
//! (`device/staging.rs`), so that a copy the GPU writes holds no second copy of its bytes in the
//! host's memory. The host writes the base a block at a time where a block changed, and reads it
//! only for a block the CPU and the GPU both changed since they agreed: itself, uncached, a block
//! at a time, where a comparison reads few, and through one copy into host memory on the staging
//! queue where it reads many ([`Held::read_back`]). The staging queue waits for no work of the
//! program's, so a completion's merge back that copies there waits for its copy alone.

use crate::copied::{BaseBytes, Held};
use crate::device::lock;
use crate::exit_guard;
use crate::memory::{self, Placement};
use crate::release;
use crate::texture::{staging_buffer, transfer_barrier};
use crate::Device;
use ash::vk;
use std::ops::{Deref, Range};
use std::ptr::NonNull;

/// A base's bytes in the device's memory, and the buffer over them its reads back copy from:
/// freed as it goes.
struct DeviceBase {
    device: Device,
    raw: vk::Buffer,
    placement: Placement,
}

impl Drop for DeviceBase {
    fn drop(&mut self) {
        // The process is exiting: the memory goes with the device.
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        // SAFETY: no submission uses the buffer: each read back was waited for before it
        // returned, and nothing else uses it.
        unsafe { release::object(shared.vk(), self.raw) };
        lock(&shared.memory).free(shared.vk(), &self.placement);
        drop(busy);
    }
}

impl Held for DeviceBase {
    /// The base's bytes of `runs`, copied on the device's staging queue into a host-visible
    /// buffer of the device's in memory the host caches where it has such (`MemoryClass::Shared`),
    /// in one submission waited for here: `None` where the process is exiting, the device is
    /// lost, or no memory holds the buffer.
    fn read_back(&self, runs: &[Range<usize>]) -> Option<Box<dyn Deref<Target = [u8]>>> {
        let _busy = exit_guard::enter()?;
        let len: usize = runs.iter().map(|run| run.end - run.start).sum();
        let read = ReadBack::new(&self.device, len)?;
        let mut at = 0u64;
        let regions: Vec<vk::BufferCopy> = (runs.iter())
            .map(|run| {
                let size = (run.end - run.start) as u64;
                let region = vk::BufferCopy {
                    src_offset: run.start as u64,
                    dst_offset: at,
                    size,
                };
                at += size;
                region
            })
            .collect();
        let (from, to) = (self.raw, read.raw);
        let mut staging = self.device.staging_lock().ok()?;
        let value = staging.submit(|raw, command_buffer, _| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the staging command buffer is recording, outside a render pass; both
            // buffers are live and shared with the staging queue's family, the base's bytes
            // written by the host before this submission, and the regions lie in each.
            unsafe { raw.cmd_copy_buffer(command_buffer, from, to, &regions) };
            transfer_barrier(raw, command_buffer, false);
        });
        drop(staging);
        self.device.wait_staging(value.ok()?).ok()?;
        Some(Box::new(read))
    }
}

/// A host-visible buffer a base's read back copies into, released when dropped.
struct ReadBack {
    device: Device,
    raw: vk::Buffer,
    placement: Placement,
    len: usize,
}

impl ReadBack {
    /// A buffer of `len` bytes; `None` where no memory holds it.
    fn new(device: &Device, len: usize) -> Option<ReadBack> {
        let (raw, placement) = staging_buffer(device, len as u64).ok()?;
        Some(ReadBack {
            device: device.clone(),
            raw,
            placement,
            len,
        })
    }
}

impl Deref for ReadBack {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        let Some(mapped) = self.placement.mapped else {
            unreachable!("checked when the staging buffer was made");
        };
        // SAFETY: the mapping covers the placement's `len` bytes, which only this buffer uses;
        // the memory is host-coherent, and the copy into it completed, seen by the host
        // (`Device::wait_staging`), before it was handed out.
        unsafe { std::slice::from_raw_parts(mapped.as_ptr(), self.len) }
    }
}

impl Drop for ReadBack {
    fn drop(&mut self) {
        // Once the process is exiting, the buffer is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        // SAFETY: the submission that used it completed, or the device was lost.
        unsafe { release::object(shared.vk(), self.raw) };
        lock(&shared.memory).free(shared.vk(), &self.placement);
    }
}

impl Device {
    /// The bytes of a copied host buffer's base of `length` bytes in the device's memory
    /// ([`memory::allocate_copied_base`]), with a buffer over them that its reads back copy from
    /// on the staging queue: `None` where the device has no staging queue or no such memory with
    /// room for them, and they lie in the host's.
    pub(crate) fn copied_base(&self, length: usize) -> Option<BaseBytes> {
        if !self.has_staging_queue() {
            return None;
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(length as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.ok()?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let Some(placement) =
            memory::allocate_copied_base(&shared.memory, raw_device, &requirements)
        else {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
            return None;
        };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        let bound = unsafe {
            crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
        };
        let at = placement.mapped.map(|mapped| mapped.as_ptr());
        let held = DeviceBase {
            device: self.clone(),
            raw,
            placement,
        };
        bound.ok()?;
        let at = NonNull::new(at?)?;
        // SAFETY: the placement is `length` bytes of device-local memory the host maps, coherent
        // (`Memory::copied_base_type`), that nothing else uses, mapped until `held` frees it,
        // and the buffer over it is `held`'s, which reads it back.
        Some(unsafe { BaseBytes::device(at, length, Box::new(held)) })
    }
}
