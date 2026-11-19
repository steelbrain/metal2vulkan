//! The memory of the stream's no-copy buffers (`newBufferWithBytesNoCopy:length:options:deallocator:`,
//! and `newBufferWithDescriptor:` with `noCopy`) that are imports in the product: host pages of the
//! player's own, standing for the program's, imported as a block (`rt::Device::import_host_memory`)
//! as the product imports the program's pages. A texture over a buffer in such a block is then the
//! runtime's counted copy (`Metrics::over_copies_imported`). A descriptor-form one inside a
//! recorded block is that block's storage, and no import ([`NoCopies`]).

use metal2vulkan_rt as rt;
use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
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

/// Which creator made a no-copy buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NoCopy {
    /// `newBufferWithBytesNoCopy:length:options:deallocator:`: in the recordings, the host's own
    /// buffers, never one of the program's objects.
    Bytes,
    /// `newBufferWithDescriptor:` with `noCopy`.
    Descriptor,
}

/// The stream's no-copy buffers by how they play, and the recorded ranges of the stream's blocks
/// (`m2v.newBlockWithLength:storageMode:`) a descriptor-form one may lie in: one whose recorded
/// address and length lie inside a live block's recorded range is the program's buffer in that
/// block's storage, and plays as a buffer of the block at that offset; any other is imported.
#[derive(Default)]
pub(super) struct NoCopies {
    /// Each live block's recorded range by its first byte: its end and its object.
    blocks: BTreeMap<u64, (u64, u64)>,
    /// Each live block's recorded first byte, by its object.
    starts: HashMap<u64, u64>,
    /// Descriptor-form ones played as buffers of a block.
    in_block: u64,
    /// Descriptor-form ones outside every block, imported, and their bytes imported.
    outside: u64,
    outside_bytes: u64,
    /// Bytes-form ones, imported, and their bytes imported.
    bytes_form: u64,
    bytes_form_bytes: u64,
}

impl NoCopies {
    /// Block `object` made, recorded at `start` for `len` bytes.
    pub(super) fn block_made(&mut self, object: u64, start: u64, len: u64) {
        if let Some(earlier) = self.starts.insert(object, start) {
            self.blocks.remove(&earlier);
        }
        self.blocks
            .insert(start, (start.saturating_add(len), object));
    }

    /// Block `object` released.
    pub(super) fn block_released(&mut self, object: u64) {
        if let Some(start) = self.starts.remove(&object) {
            self.blocks.remove(&start);
        }
    }

    /// The live block whose recorded range holds the `length` bytes recorded at `address`, and
    /// their offset in it.
    pub(super) fn holding(&self, address: u64, length: u64) -> Option<(u64, u64)> {
        let (&start, &(end, object)) = self.blocks.range(..=address).next_back()?;
        (address.checked_add(length)? <= end).then_some((object, address - start))
    }

    /// A descriptor-form one played as a buffer of a block.
    pub(super) fn in_block(&mut self) {
        self.in_block += 1;
    }

    /// One of `kind` imported, `bytes` bytes.
    pub(super) fn imported(&mut self, kind: NoCopy, bytes: u64) {
        let (count, sum) = match kind {
            NoCopy::Bytes => (&mut self.bytes_form, &mut self.bytes_form_bytes),
            NoCopy::Descriptor => (&mut self.outside, &mut self.outside_bytes),
        };
        *count += 1;
        *sum += bytes;
    }

    /// The report's line.
    pub(super) fn line(&self) -> String {
        format!(
            "no-copy buffers: descriptor-form {} in a recorded block (buffers of it, no import), \
             {} outside every block (each imported, {} bytes in all); bytes-form {} (the host's \
             own objects, each imported, {} bytes in all); a texture over an imported one is the \
             runtime's counted copy (over_copies_imported, over_copy_bytes_imported)",
            self.in_block, self.outside, self.outside_bytes, self.bytes_form, self.bytes_form_bytes
        )
    }
}
