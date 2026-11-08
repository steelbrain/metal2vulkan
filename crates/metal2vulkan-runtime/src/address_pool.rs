//! A pool of the Metal address space whose slots buffers are placed in (the consumer's ask
//! "sub-page placement", `address.rs`' module docs).
//!
//! Metal makes each buffer of its own memory shorter than a page in a pool of its own, at a
//! 256-byte-aligned slot of `round_up(length, 256)` bytes, so several buffers share a 16-KiB page
//! (the consumer's `vaprobe smallpool` on an M3 Max under macOS 26.7). A caller that models those
//! pools reserves each ([`Device::reserve_address_pool`]) and places each small buffer at the slot
//! its model gives (`BufferDescriptor::with_pool_slot`): the buffer answers Metal's address.

use super::address::{AddressPlacement, GpuAddress};
use super::device::lock;
use super::error::Error;
use super::Device;

/// A reserved range of the Metal address space whose slots buffers are placed in
/// (`BufferDescriptor::with_pool_slot`; a runtime extension, no Metal counterpart: Metal's own
/// buffer pools, which hold every buffer shorter than a page, are not objects of its API). While
/// it lives, no other buffer or heap, pinned or not, lands in its span. Dropping it lets the span
/// go, at once where no buffer lies in a slot of it, else once the last such buffer is gone.
pub struct AddressPool {
    device: Device,
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

    /// The device whose address space the pool is of.
    pub fn device(&self) -> &Device {
        &self.device
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
        lock(&self.device.shared().addresses).release_pool(self.address);
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
        self.current().reserve_address_pool_now(placement, length)
    }

    /// [`Device::reserve_address_pool`] on the generation that serves (`device/line.rs`).
    pub(crate) fn reserve_address_pool_now(
        &self,
        placement: AddressPlacement,
        length: u64,
    ) -> Result<AddressPool, Error> {
        let address = lock(&self.shared().addresses).reserve_pool(placement, length)?;
        Ok(AddressPool {
            device: self.clone(),
            address,
            length,
        })
    }
}
