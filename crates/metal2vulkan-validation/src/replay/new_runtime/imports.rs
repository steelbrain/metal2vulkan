//! The memory of the stream's no-copy buffers (`newBufferWithBytesNoCopy:length:options:deallocator:`,
//! and `newBufferWithDescriptor:` with `noCopy`): host pages of the player's own, standing for the
//! program's, imported as a block (`rt::Device::import_host_memory`) as the product imports the
//! program's pages. A texture over a buffer in such a block is then the runtime's counted copy
//! (`Metrics::over_copies`).

use metal2vulkan_rt as rt;
use std::alloc::Layout;
use std::ptr::NonNull;

/// The device's `minImportedHostPointerAlignment`, where it imports host memory
/// (`VK_EXT_external_memory_host` enabled): an import's address and length are multiples of it.
pub(super) fn host_alignment(device: &rt::Device) -> Option<u64> {
    let name = ash::ext::external_memory_host::NAME;
    let mut enabled = device.extensions().iter();
    if !enabled.any(|extension| extension.as_c_str() == name) {
        return None;
    }
    let raw = device.raw();
    let mut host = ash::vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut host);
    // SAFETY: the physical device is the instance's, and the extension the chained structure
    // belongs to is enabled on the device made from it.
    unsafe { (raw.instance).get_physical_device_properties2(raw.physical_device, &mut properties) };
    Some(host.min_imported_host_pointer_alignment.max(1))
}

/// Zeroed host pages of the player's own, which an import's release frees once the runtime has
/// freed the import.
#[derive(Clone, Copy)]
struct HostPages {
    pointer: NonNull<u8>,
    layout: Layout,
}

// SAFETY: the pages are plain memory the player allocated; the import's release takes them to the
// thread that frees them, once, after the runtime is done with them.
unsafe impl Send for HostPages {}

impl HostPages {
    /// `len` bytes at a multiple of `alignment`, zeroed; none where the allocator has no room.
    fn new(len: u64, alignment: u64) -> Option<HostPages> {
        let layout = Layout::from_size_align(len.try_into().ok()?, alignment.try_into().ok()?);
        let layout = layout.ok()?;
        // SAFETY: the layout's size is not zero (`imported` takes at least one alignment).
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        Some(HostPages { pointer, layout })
    }

    /// Frees the pages.
    ///
    /// # Safety
    ///
    /// Nothing reads or writes them any more, and they are freed once.
    unsafe fn free(self) {
        // SAFETY: allocated with this layout (`new`); the caller's.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

/// A block over zeroed host pages of the player's own for a no-copy buffer of `length` bytes,
/// and the bytes imported: `length` rounded up to `alignment` (the device's
/// `minImportedHostPointerAlignment`), imported (`rt::Device::import_host_memory`). The pages are
/// freed after the runtime frees the import, which it does once the block's last handle, buffer
/// and use are gone. Err says why there is none: the host allocator's refusal or the runtime's.
pub(super) fn imported(
    device: &rt::Device,
    length: u64,
    alignment: u64,
) -> Result<(rt::Block, u64), String> {
    let len = length.max(1).next_multiple_of(alignment);
    let Some(pages) = HostPages::new(len, alignment) else {
        return Err(format!(
            "no {len} bytes of host pages at a multiple of {alignment}"
        ));
    };
    // SAFETY: the release runs once, after the runtime freed the import (its contract), and
    // nothing of the player's touches the pages after the block's buffer is released.
    let release: Box<dyn FnOnce() + Send> = Box::new(move || unsafe { pages.free() });
    // SAFETY: the pages are mapped readable and writable, the player's own, and stay so until the
    // release frees them.
    match unsafe { device.import_host_memory(pages.pointer, len, Some(release)) } {
        Ok((block, _)) => Ok((block, len)),
        Err(error) => {
            // On an error the release was dropped unrun: the pages are the player's again.
            // SAFETY: nothing imported them, and this is their one free.
            unsafe { pages.free() };
            Err(error.to_string())
        }
    }
}
