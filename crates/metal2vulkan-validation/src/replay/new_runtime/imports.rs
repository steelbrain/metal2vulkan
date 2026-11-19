//! The stream's no-copy buffers. A bytes-form one (`newBufferWithBytesNoCopy:length:options:
//! deallocator:`), the host's own memory in the recordings, is made over host pages of the
//! player's own, imported as a block (`rt::Device::import_host_memory`) as the product imports
//! such pages; a texture over it is then the runtime's counted copy
//! (`Metrics::over_copies_imported`). A descriptor-form one (`newBufferWithDescriptor:` with
//! `noCopy`) is a pool buffer, with no import ([`NoCopies`]).

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
    /// `newBufferWithDescriptor:` with `noCopy`; `pinned_unzeroed` where its descriptor has a
    /// `pinnedGpuAddress` and no `zeroFrom`, the one kind the host's recorder makes over the
    /// program's own no-copy memory (one with `zeroFrom` or no pinned address is the host's own).
    Descriptor { pinned_unzeroed: bool },
}

/// The stream's no-copy buffers by how they play, and the recorded ranges of the stream's blocks
/// (`m2v.newBlockWithLength:storageMode:`) a descriptor-form one may lie in. A descriptor-form one
/// is a pool buffer in the product, the program's buffer in a block's storage: where its recorded
/// address and length lie inside a live block's recorded range it plays as a buffer of the block
/// at that offset, and elsewhere in a place of its own. The recordings do not tell the program's
/// own no-copy memory, which the product imports, from these, so such memory plays as a pool
/// buffer too (the report's gap line, bounded by the pinned ones with no `zeroFrom` outside every
/// block). A bytes-form one, the host's own, is imported.
#[derive(Default)]
pub(super) struct NoCopies {
    /// Each live block's recorded range by its first byte: its end and its object.
    blocks: BTreeMap<u64, (u64, u64)>,
    /// Each live block's recorded first byte, by its object.
    starts: HashMap<u64, u64>,
    /// Descriptor-form ones played as buffers of a block.
    in_block: u64,
    /// Descriptor-form ones outside every block, each in a place of its own, and their bytes.
    outside: u64,
    outside_bytes: u64,
    /// Of those, the ones that may be the program's own memory (`NoCopy::Descriptor`'s
    /// `pinned_unzeroed`), and their bytes.
    may_be_programs: u64,
    may_be_programs_bytes: u64,
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

    /// A bytes-form one imported, `bytes` bytes.
    pub(super) fn imported(&mut self, bytes: u64) {
        self.bytes_form += 1;
        self.bytes_form_bytes += bytes;
    }

    /// A descriptor-form one outside every block, in a place of its own, `bytes` long; one that
    /// may be the program's own memory where `pinned_unzeroed`.
    pub(super) fn pooled(&mut self, bytes: u64, pinned_unzeroed: bool) {
        self.outside += 1;
        self.outside_bytes += bytes;
        if pinned_unzeroed {
            self.may_be_programs += 1;
            self.may_be_programs_bytes += bytes;
        }
    }

    /// The report's lines: the counts, and where a descriptor-form one that may be the program's
    /// own memory played outside every block, the gap and its bound.
    pub(super) fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "no-copy buffers: descriptor-form {} in a recorded block (buffers of it) and {} \
             outside every block (places of their own, {} bytes in all), pool buffers with no \
             import; bytes-form {} (the host's own objects, each imported, {} bytes in all); a \
             texture over an imported one is the runtime's counted copy (over_copies_imported, \
             over_copy_bytes_imported)",
            self.in_block, self.outside, self.outside_bytes, self.bytes_form, self.bytes_form_bytes
        )];
        if self.may_be_programs > 0 {
            lines.push(format!(
                "gap: descriptor-form buffers that may be the app's own no-copy memory are played \
                 as pool buffers, count unknown: at most {}, those outside every block with a \
                 pinnedGpuAddress and no zeroFrom ({} bytes in all)",
                self.may_be_programs, self.may_be_programs_bytes
            ));
        }
        lines
    }
}
