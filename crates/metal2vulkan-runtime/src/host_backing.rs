//! What the GPU uses for a buffer over the caller's host memory ([`Buffer::host_backing`], a
//! runtime extension): the caller's pages themselves, through one of the device's imports of
//! them, or a copy of them (`copied.rs`). Metal has no counterpart: a no-copy buffer's bytes are
//! always the caller's pages there.
//!
//! The answer is read from where the buffer lies now (`buffer::Lying`): its import's shape
//! (`buffer::Shape`: dma-bufs, a guest RAM's chunk, else a pointer) and whether that import is a
//! copy. It reads no page and takes no import's objects, so it keeps no import imported
//! (`host_residency.rs`).

use super::buffer::{Buffer, Lying};

/// What the GPU reads and writes for a buffer over the caller's host memory
/// ([`Buffer::host_backing`], a runtime extension): the caller's pages, through one of the
/// device's imports of them, or a copy of them. Metal has no counterpart: a no-copy buffer's bytes
/// are always the caller's pages there (`newBufferWithBytesNoCopy:length:options:deallocator:`),
/// so [`HostBacking::Copied`] is the one form that differs from Metal's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostBacking {
    /// In place: the device imported the pages by their pointer (`VK_EXT_external_memory_host`),
    /// or for memory whose GPU writes Metal's private option bit 24 drops, by the pointer of a
    /// read-only view of them. An import the device may release while idle and import again at
    /// its next use (a runtime-internal residency, which nothing of Metal's observes) answers so
    /// throughout.
    ImportedByPointer,
    /// In place: the device imported dma-bufs of the pages that the caller's source made
    /// ([`HostMemory::with_dma_buf`](crate::HostMemory::with_dma_buf)), where it refused their
    /// pointer. Released while idle and imported again as an import by pointer may be.
    ImportedAsDmaBuf,
    /// In place: bound in the device's import of a chunk of a guest RAM it uses
    /// ([`Device::use_host_ram`](crate::Device::use_host_ram),
    /// [`HostMemory::with_runs`](crate::HostMemory::with_runs)), with no import of its own.
    InHostRam,
    /// A copy of the pages in memory of the device's own, which the GPU reads and writes in their
    /// place: the caller's gate refused their import
    /// ([`HostMemory::with_import_gate`](crate::HostMemory::with_import_gate)), the import the
    /// device and the gate admitted failed, or a rebind's new bytes lie at another offset in
    /// their page than the old under a texture over the buffer. Announced as the
    /// [`copied-host-pages`](crate::Degradation::CopiedHostPages) degradation: each commit writes
    /// into the copy the bytes the CPU changed in the pages, and each batch's completion writes
    /// back the bytes the GPU changed, where Metal's GPU uses the pages themselves.
    Copied,
    /// In no pages: detached ([`Buffer::detach_backing`]) until a rebind gives it some.
    Detached,
}

impl Buffer {
    /// What the GPU uses for the buffer's bytes now, where they are the caller's host memory: a
    /// no-copy buffer's, or for a buffer placed in a heap over the caller's host memory
    /// (`Device::new_heap_with_bytes_no_copy`) its heap's ([`HostBacking`]). `None` for any other
    /// buffer. A runtime extension; Metal has no counterpart, as a no-copy buffer's bytes are
    /// always the caller's pages there: [`HostBacking::Copied`] answers where they are a copy
    /// instead, announced as the
    /// [`copied-host-pages`](crate::Degradation::CopiedHostPages) degradation.
    ///
    /// The answer changes only where the buffer's pages do: at its
    /// [`rebind_bytes_no_copy`](Buffer::rebind_bytes_no_copy), which may move it between an import
    /// and a copy, and its [`detach_backing`](Buffer::detach_backing) (for a heap's buffer, its
    /// heap's); between them it stays. A query alone: it reads no page and keeps no import
    /// imported, unlike [`gpu_address`](Buffer::gpu_address).
    pub fn host_backing(&self) -> Option<HostBacking> {
        let pages = match self.pages_cell()?.lying() {
            Lying::In(pages) => pages,
            Lying::Detached(_) => return Some(HostBacking::Detached),
        };
        let backing = if pages.copied() {
            HostBacking::Copied
        } else if pages.over_dma_buf() {
            HostBacking::ImportedAsDmaBuf
        } else if pages.in_ram_chunk() {
            HostBacking::InHostRam
        } else {
            HostBacking::ImportedByPointer
        };
        Some(backing)
    }
}
