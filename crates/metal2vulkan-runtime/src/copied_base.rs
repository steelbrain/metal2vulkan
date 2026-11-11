//! The device's memory of a copied host buffer's base (`copied.rs`, "A base once it may"): the
//! bytes the pages and the copy last agreed on, in device-local memory the host maps, where the
//! device has such memory with room ([`memory::allocate_copied_base`]), so that a copy the GPU
//! writes holds no second copy of its bytes in the host's memory. The host writes the base a
//! block at a time where a block changed, and reads it only for a block the CPU and the GPU both
//! changed since they agreed.

use crate::copied::BaseBytes;
use crate::device::lock;
use crate::exit_guard;
use crate::memory::{self, Placement};
use crate::Device;

/// A base's bytes in the device's memory: freed as it goes.
struct DeviceBase {
    device: Device,
    placement: Placement,
}

impl Drop for DeviceBase {
    fn drop(&mut self) {
        // The process is exiting: the memory goes with the device.
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        // No command uses it: only the host reads and writes a base.
        lock(&shared.memory).free(shared.vk(), &self.placement);
        drop(busy);
    }
}

impl Device {
    /// The bytes of a copied host buffer's base of `length` bytes in the device's memory
    /// ([`memory::allocate_copied_base`]): `None` where the device has no such memory with room
    /// for them, and they lie in the host's.
    pub(crate) fn copied_base(&self, length: usize) -> Option<BaseBytes> {
        let shared = self.shared();
        let placement = memory::allocate_copied_base(&shared.memory, shared.vk(), length as u64)?;
        let at = placement.mapped.map(|mapped| mapped.as_ptr());
        let held = DeviceBase {
            device: self.clone(),
            placement,
        };
        let at = std::ptr::NonNull::new(at?)?;
        // SAFETY: the placement is `length` bytes of device-local memory the host maps, coherent
        // (`Memory::copied_base_type`), that nothing else uses, mapped until `held` frees it.
        Some(unsafe { BaseBytes::device(at, length, Box::new(held)) })
    }
}
