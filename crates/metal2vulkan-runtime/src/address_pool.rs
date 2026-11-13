//! A pool of the Metal address space whose slots buffers are placed in (the consumer's ask
//! "sub-page placement", `address.rs`' module docs).
//!
//! Metal makes each buffer of its own memory shorter than a page in a pool of its own, at a
//! 256-byte-aligned slot of `round_up(length, 256)` bytes, so several buffers share a 16-KiB page
//! (the consumer's `vaprobe smallpool` on an M3 Max under macOS 26.7). A caller that models those
//! pools reserves each ([`Device::reserve_address_pool`]) and places each small buffer at the slot
//! its model gives (`BufferDescriptor::with_pool_slot`): the buffer answers Metal's address.

use super::address::{AddressPlacement, AddressSpace, GpuAddress};
use super::device::carry::LineDevice;
use super::device::lock;
use super::error::Error;
use super::Device;
use std::sync::{Arc, Mutex};

/// A reserved range of the Metal address space whose slots buffers are placed in
/// (`BufferDescriptor::with_pool_slot`; a runtime extension, no Metal counterpart: Metal's own
/// buffer pools, which hold every buffer shorter than a page, are not objects of its API). While
/// it lives, no other buffer or heap, pinned or not, lands in its span. Dropping it lets the span
/// go, at once where no buffer lies in a slot of it, else once the last such buffer is gone.
///
/// Where the device's recovery makes it a new `VkDevice` (`Degradation::DeviceRecovered`), the
/// pool goes on there: it is a span of the device's address space, which every `VkDevice` of the
/// device shares, and holds no Vulkan object, so buffers made after the recovery take its slots
/// as before.
pub struct AddressPool {
    /// The device: its root handle, which a buffer's placement in a slot matches by its line,
    /// and the generation handle `device` answers.
    device: LineDevice,
    /// The device's address space, which its span is of.
    addresses: Arc<Mutex<AddressSpace>>,
    address: GpuAddress,
    length: u64,
}

impl AddressPool {
    /// The pool's first byte: a slot's buffer at `offset` answers this plus `offset`.
    pub fn gpu_address(&self) -> GpuAddress {
        self.address
    }

    /// The pool's bytes, which its slots lie in.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// The device whose address space the pool is of: a generation handle of the device: of the
    /// generation the pool was made on, then of each one the device's recoveries carried the pool
    /// to. The handle is the caller's: it keeps meaning that generation while held, past a recovery
    /// that carries the pool on, its raw handles that generation's `VkDevice`'s, kept alive while
    /// held.
    pub fn device(&self) -> Device {
        self.device.generation().clone()
    }

    /// A root handle of the pool's device, which a buffer's placement in a slot matches by its
    /// line.
    pub(crate) fn root_device(&self) -> &Device {
        self.device.root()
    }
}

impl std::fmt::Debug for AddressPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressPool")
            .field("gpu_address", &self.address)
            .field("length", &self.length)
            .finish()
    }
}

impl Drop for AddressPool {
    fn drop(&mut self) {
        lock(&self.addresses).release_pool(self.address);
    }
}

impl Device {
    /// Reserve a pool of `length` bytes of the Metal address space at `placement` (a runtime
    /// extension, no Metal counterpart; `address.rs`' module docs): the pool holds the span a
    /// buffer of `length` bytes pinned at its address holds (the pages its bytes touch and one
    /// guard page, rounded up to 32 KiB), so no other placement, pin or range the runtime picks
    /// lands in it while it lives. Buffers go in its slots (`BufferDescriptor::with_pool_slot`).
    ///
    /// At `AddressPlacement::At`, refused as a pin is (`PlacementConflict`: `Held`,
    /// `OffGranule`, `OutsideSpace`); at `Any` or `Outside`, at a 32-KiB granule the runtime
    /// picks in the space or outside it, refused `OutOfMemory` where no span is free.
    pub fn reserve_address_pool(
        &self,
        placement: AddressPlacement,
        length: u64,
    ) -> Result<AddressPool, Error> {
        self.at_serving(|device| device.reserve_address_pool_now(placement, length))
    }

    /// [`Device::reserve_address_pool`] on the generation that serves (`device/line.rs`).
    pub(crate) fn reserve_address_pool_now(
        &self,
        placement: AddressPlacement,
        length: u64,
    ) -> Result<AddressPool, Error> {
        let address = (self.address_space_for(placement)).reserve_pool(placement, length)?;
        Ok(AddressPool {
            device: LineDevice::new(self),
            addresses: Arc::clone(&self.shared().addresses),
            address,
            length,
        })
    }
}
