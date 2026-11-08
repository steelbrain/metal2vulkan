//! A command buffer's upload arena: host-visible memory the runtime writes a command's own data
//! into while the command is encoded — argument, handle and root tables, constant blobs, neutral
//! buffers, and the command buffer's execution status words.
//!
//! - **Chunks.** The arena is a list of `Shared` buffers of the device in the default CPU cache
//!   mode, with Vulkan device addresses where the device has them, each filled from its start:
//!   host-coherent, host-cached where the device has such memory, else device-local where it maps
//!   such (`MemoryClass::Shared`), so system memory on a discrete GPU. The encoding thread writes
//!   them, and a write into device-local memory waits for a powered-down GPU to wake, where one
//!   into system memory does not (an RTX 5080 laptop GPU in NVIDIA's runtime D3 took 970 to 980 ms
//!   for its first write into device-local memory after 20 s idle, against 0.01 ms into system
//!   memory). The host writes host-cached memory faster too, and the GPU reads it as fast: on an RX
//!   7900 XTX (RADV), 3.7 us of thread time a draw that sets its bytes, against 4.3 in device-local
//!   memory, and a vertex-heavy draw's GPU time the same (28.19 ms against 28.18). A region is
//!   never reused while the command buffer lives: every command reads data no other command wrote,
//!   and the host writes it before the command's batch is submitted (a host step's, after the
//!   batches before it). The regions the GPU writes: a staged scatter's staging, which a host
//!   step takes and the GPU's copy writes, and which a host step of a later batch reads once
//!   that copy completed (`host_ram/copy.rs`); where a pass's visibility results land before
//!   they are copied on into a no-copy buffer (`visibility.rs`), within one command of barriers,
//!   and an index bounds pass's bounds words and the command buffer's index pool
//!   (`index_bounds.rs`), written before the pass between barriers; and a flat dispatch's region
//!   record and `VkDispatchIndirectCommand` (`binding::Tables::region_record`), which its
//!   preprocess writes before the dispatch between barriers; and an execution's device replay
//!   scratch on the GPU alone (its slot tables, region records, arguments or sequences and counts,
//!   and the sink; `gpu_execute.rs`), which the replay's preprocess writes before the commands
//!   between barriers; and an emulated tessellation draw's work buffer, whose first words an
//!   indirect draw's copy of its arguments and whose header and chunk slots its scan write before
//!   its draws between barriers (`render/tessellation_scan.rs`): every chunk carries indirect and
//!   storage usage and, where the device has them, a device address. The command buffer releases
//!   the arena when it completes.
//! - **Sizes.** A command buffer's first chunk holds [`FIRST_CHUNK_BYTES`], and each next one
//!   twice as many, up to [`CHUNK_BYTES`]; a region of more than half of that takes a chunk of its
//!   own. A chunk's bytes hold no value as it is made (`Device::unzeroed_buffer_with`): what must
//!   read zero is the part of a region its data does not cover, which the region zeroes as it is
//!   handed out, and nothing reads a byte of a chunk outside a region. So a command buffer clears
//!   its regions' unwritten bytes, at most its chunks' bytes and in practice far less: a command
//!   whose tables are as long as their data clears nothing, where a clear of each chunk wrote
//!   4 KiB for the first region of a command buffer whatever it held (clearing 64 KiB took 11.5 us
//!   of an RTX 3090's host-visible memory and 2.6 us of a Radeon 780M's). A chunk long enough to
//!   lie in fresh pages reads zero unwritten, and a region of one writes nothing past its data.
//!   Counted: `WorkCounts::new_buffer_bytes_cleared`, the same count a buffer's own clear moves.
//! - **Alignment.** Every region starts at a multiple of the device's upload alignment (16 bytes,
//!   and the storage and uniform buffer offset alignments), so it can be bound at its offset and
//!   its device address is 16-aligned. A root table and the resident set it searches start at a
//!   device address that is a multiple of 32 too (`upload_aligned`). A region's bytes past the data
//!   are zero: [`Uploads::upload`] sets them, since the chunk holds no value as it is made; but an
//!   unset region's ([`Uploads::unset`]: a staged gather's pages, which a host step writes before
//!   its batch is submitted, every byte its command reads among them; a staged scatter's bytes,
//!   which its copy writes, every byte the host reads after it among them). The bytes an alignment
//!   leaves between two regions belong to no region, and nothing reads them.

use super::buffer::{Buffer, NewBytes};
use super::device::lock;
use super::error::{Error, MemoryKind};
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use ash::vk;
use std::sync::{Arc, Mutex};

/// The size of a command buffer's first ordinary chunk.
pub(super) const FIRST_CHUNK_BYTES: u64 = 4 * 1024;

/// The size of the largest ordinary chunk; a region of more than half of it gets a chunk of its
/// own.
const CHUNK_BYTES: u64 = 64 * 1024;

/// The arena of one command buffer.
pub(super) struct Uploads {
    /// The chunk filled now is the last; dedicated chunks go before it.
    chunks: Vec<Chunk>,
    /// The ordinary chunks opened: the next is [`FIRST_CHUNK_BYTES`] times two to this power,
    /// at most [`CHUNK_BYTES`].
    opened: u32,
    /// The arena of a predicated loop's check, lent while it records a run again
    /// (`command_buffer/rerun.rs`): every region comes from it meanwhile, and goes once that run
    /// completed.
    lent: Option<Arc<Mutex<Uploads>>>,
}

struct Chunk {
    buffer: Buffer,
    /// Bytes from the chunk's start that regions took.
    used: u64,
    /// Whether every byte of it nothing wrote reads zero: it was made in fresh pages
    /// (`fresh_pages.rs`), so a region of it needs no zeros written past its data.
    zero_unwritten: bool,
}

/// A region of the arena: `len` bytes of `buffer` from `offset`.
#[derive(Clone, Debug)]
pub(super) struct Upload {
    buffer: Buffer,
    offset: u64,
    len: u64,
}

impl Uploads {
    /// An empty arena.
    pub(super) fn new() -> Uploads {
        Uploads {
            chunks: Vec::new(),
            opened: 0,
            lent: None,
        }
    }

    /// Take every region from `lent`, a predicated loop's check's arena, while it is lent; `None`:
    /// from this one again.
    pub(super) fn lend(&mut self, lent: Option<Arc<Mutex<Uploads>>>) {
        self.lent = lent;
    }

    /// Every chunk, taken out: no command reads the arena any more, and the chunks go as the
    /// arena returned drops.
    pub(super) fn take(&mut self) -> Uploads {
        let opened = self.opened;
        self.opened = 0;
        Uploads {
            chunks: std::mem::take(&mut self.chunks),
            opened,
            lent: None,
        }
    }

    /// The bytes its regions took of its chunks (a device test's measure).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn bytes(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.used).sum()
    }

    /// A region of `len` bytes (at least `bytes.len()`) holding `bytes` and then zeros.
    pub(super) fn upload(
        &mut self,
        device: &Device,
        bytes: &[u8],
        len: u64,
    ) -> Result<Upload, Error> {
        self.upload_aligned(device, bytes, len, 1)
    }

    /// [`upload`](Self::upload) at a device address that is a multiple of `address_alignment`
    /// too, a power of two (a root table and the resident set it searches, 32 bytes): on a
    /// device without device addresses, at an offset that is.
    pub(super) fn upload_aligned(
        &mut self,
        device: &Device,
        bytes: &[u8],
        len: u64,
        address_alignment: u64,
    ) -> Result<Upload, Error> {
        self.place(device, bytes, (len, address_alignment), true)
    }

    /// A region of `len` bytes whose bytes hold no value until written ([`Upload::write_at`]):
    /// the caller writes every byte a command reads before the batch that reads them is
    /// submitted (a host step's, a staged gather's pages, `host_ram/copy.rs`), or every byte the
    /// host reads by a command that completed before (a staged scatter's copy).
    pub(super) fn unset(&mut self, device: &Device, len: u64) -> Result<Upload, Error> {
        self.place(device, &[], (len, 1), false)
    }

    /// A region of `len` bytes (at least `bytes.len()`) holding `bytes`, at a device address
    /// that is a multiple of `address_alignment`, its bytes past them zero where `clear` says.
    fn place(
        &mut self,
        device: &Device,
        bytes: &[u8],
        (len, address_alignment): (u64, u64),
        clear: bool,
    ) -> Result<Upload, Error> {
        if let Some(lent) = &self.lent {
            return lock(lent).place(device, bytes, (len, address_alignment), clear);
        }
        let len = len.max(bytes.len() as u64);
        let alignment = device.shared().upload_alignment;
        // The first offset from `from`, a multiple of the upload alignment, whose device address
        // in `chunk` is a multiple of `address_alignment`: both are powers of two, so the walk
        // takes at most `address_alignment / alignment` steps.
        let aligned = |chunk: &Chunk, from: u64| {
            let base = chunk.buffer.device_address().unwrap_or(0);
            let mut offset = from.checked_next_multiple_of(alignment)?;
            while !base.wrapping_add(offset).is_multiple_of(address_alignment) {
                offset = offset.checked_add(alignment)?;
            }
            Some(offset)
        };
        let fits = |chunk: &Chunk| {
            aligned(chunk, chunk.used)
                .and_then(|offset| Some((offset, offset.checked_add(len)?)))
                .filter(|&(_, end)| end <= chunk.buffer.length())
        };
        let current = self
            .chunks
            .last()
            .and_then(fits)
            .map(|(offset, end)| (self.chunks.len() - 1, offset, end));
        let (index, offset, end) = match current {
            Some(found) => found,
            None => {
                // Room for the region past the first aligned address.
                let need = len
                    .checked_add(address_alignment - 1)
                    .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
                let dedicated = need > CHUNK_BYTES / 2;
                let size = if dedicated {
                    need.checked_next_multiple_of(CHUNK_BYTES)
                        .ok_or(Error::OutOfMemory(MemoryKind::Device))?
                } else {
                    // At most `CHUNK_BYTES`: `need` is at most half of it.
                    (FIRST_CHUNK_BYTES << self.opened.min(4)).max(need.next_power_of_two())
                };
                // The encoding thread writes the chunk: host-cached memory where the device has
                // such (the module docs). Its bytes hold no value as it is made: a region zeroes
                // what its own data does not cover as it is handed out, and nothing reads a byte
                // of the chunk outside a region, so a clear of the whole chunk would write bytes
                // no reader can reach (the module docs' "Sizes").
                let (buffer, new_bytes) = device.unzeroed_buffer_with(
                    size,
                    ResourceOptions::new(
                        StorageMode::Shared,
                        CpuCacheMode::DefaultCache,
                        HazardTrackingMode::Untracked,
                    ),
                )?;
                if !device.properties().has_unified_memory
                    && buffer.own_pages_device_local() == Some(true)
                {
                    device.shared().counts.upload_chunk_in_device_memory();
                }
                // A large region takes a chunk of its own, leaving the current chunk current.
                let at = if dedicated && !self.chunks.is_empty() {
                    self.chunks.len() - 1
                } else {
                    self.chunks.len()
                };
                if !dedicated {
                    self.opened += 1;
                }
                let chunk = Chunk {
                    buffer,
                    used: 0,
                    zero_unwritten: new_bytes == NewBytes::Zero,
                };
                let offset = aligned(&chunk, 0).ok_or(Error::OutOfMemory(MemoryKind::Device))?;
                self.chunks.insert(at, chunk);
                (at, offset, offset + len)
            }
        };
        let chunk = &mut self.chunks[index];
        let Some(mapped) = chunk.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        // SAFETY: `[offset, offset + bytes.len())` is inside the chunk's mapping (`fits`, or a
        // new chunk of at least `len` bytes), no region overlaps it, and no submitted command
        // reads it: regions are written while the command buffer is encoded, or by a host step
        // for a batch not submitted yet, past every region submitted before.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add(offset as usize),
                bytes.len(),
            );
        }
        // The region's bytes past the data are zero (the module docs' "Alignment"), and the chunk
        // holds no value as it is made, so the region sets them: at most the chunk's bytes over
        // all its regions, where a clear of the chunk wrote every byte of it whether a region took
        // it or not. A chunk made in fresh pages already reads zero where nothing wrote, and no
        // region writes another's bytes, so a region of one writes none of them: else a replay
        // scratch of tens of MiB, whose head alone is uploaded and whose rest the GPU writes,
        // would pay a memset its creation never paid (`fresh_pages.rs`).
        let tail = len - bytes.len() as u64;
        if clear && tail > 0 && !chunk.zero_unwritten {
            let at = offset + bytes.len() as u64;
            // SAFETY: `[at, at + tail)` is the rest of the region, inside the chunk's mapping as
            // above, and no region overlaps it.
            unsafe { std::ptr::write_bytes(mapped.as_ptr().add(at as usize), 0, tail as usize) };
            device.shared().counts.new_buffer_cleared(tail);
        }
        chunk.used = end;
        Ok(Upload {
            buffer: chunk.buffer.clone(),
            offset,
            len,
        })
    }
}

impl Upload {
    /// The `VkBuffer` the region is in.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer.at(self.offset).0
    }

    /// The region's offset in its `VkBuffer`.
    pub(super) fn offset(&self) -> u64 {
        self.buffer.at(self.offset).1
    }

    /// The buffer the region is in, and the region's offset there: a draw replay's scratch is
    /// its draws' indirect buffer (`render/executions/gpu.rs`).
    pub(super) fn buffer(&self) -> (&Buffer, u64) {
        (&self.buffer, self.offset)
    }

    /// The region's length.
    pub(super) fn len(&self) -> u64 {
        self.len
    }

    /// `len` bytes of the region from byte `at`, as a region of their own: a gate in a bound
    /// predication step's record (`predication.rs`). A part past the region is cut to it.
    pub(super) fn part(&self, at: u64, len: u64) -> Upload {
        let at = at.min(self.len);
        Upload {
            buffer: self.buffer.clone(),
            offset: self.offset + at,
            len: len.min(self.len - at),
        }
    }

    /// Write `bytes` at the region's start again, before any batch that reads the region is
    /// submitted: as the command buffer commits (a function-table snapshot). A write longer than
    /// the region is cut to it.
    pub(super) fn write(&self, bytes: &[u8]) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = bytes.len().min(self.len as usize);
        // SAFETY: `[offset, offset + len)` is inside the region, which no other region overlaps;
        // the caller writes it before the batches that read it are submitted, so no command
        // reads it meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add(self.offset as usize),
                len,
            );
        }
    }

    /// [`write`](Self::write) at byte `at` of the region: a root table's resident-set fields as
    /// the command buffer commits, or as a host step uploads it. A write past the region is cut.
    pub(super) fn write_at(&self, at: u64, bytes: &[u8]) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = (bytes.len() as u64).min(self.len.saturating_sub(at)) as usize;
        // SAFETY: `[offset + at, offset + at + len)` is inside the region, which no other region
        // overlaps; the caller writes it before the batches that read it are submitted.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add((self.offset + at) as usize),
                len,
            );
        }
    }

    /// Copy the `len` bytes at `from` into the region from its byte `at`, before any batch that
    /// reads the region is submitted: a staged gather's pages, by a host step (`host_ram/copy.rs`).
    /// A copy past the region is cut to it.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `from` stay mapped and readable while the call runs, and lie outside
    /// the region.
    pub(super) unsafe fn copy_in(&self, at: u64, from: *const u8, len: u64) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = len.min(self.len.saturating_sub(at)) as usize;
        // SAFETY: `[offset + at, offset + at + len)` is inside the region, which no other region
        // overlaps and no submitted command reads (the caller writes it before the batches that
        // read it are submitted); the bytes at `from` are readable and apart from it (the
        // caller's contract). A raw copy, as another thread may write those bytes meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(
                from,
                mapped.as_ptr().add((self.offset + at) as usize),
                len,
            );
        }
    }

    /// Copy the `len` bytes of the region from its byte `at` to `to`: a staged scatter's bytes
    /// into the pages, by a host step (`host_ram/copy.rs`). A copy past the region is cut to it.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `to` stay mapped and writable while the call runs, and lie outside the
    /// region; every command that writes those bytes of the region completed, and their writes
    /// are visible to the host (its batch's end, `tracker::Barrier::to_host`).
    pub(super) unsafe fn copy_out(&self, at: u64, to: *mut u8, len: u64) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = len.min(self.len.saturating_sub(at)) as usize;
        // SAFETY: `[offset + at, offset + at + len)` is inside the region, which no other region
        // overlaps and no command writes any more (the caller's contract), in host-coherent
        // memory; the bytes at `to` are writable and apart from it (the caller's contract). A raw
        // copy, as another thread may read or write those bytes meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().add((self.offset + at) as usize),
                to,
                len,
            );
        }
    }

    /// The region's Vulkan device address, where the device has device addresses.
    pub(super) fn device_address(&self) -> Option<u64> {
        self.buffer
            .device_address()
            .map(|address| address + self.offset)
    }

    /// The region's first `N` bytes as the device left them. The caller has waited for every
    /// command that writes them, and made their writes visible to the host.
    pub(super) fn read<const N: usize>(&self) -> [u8; N] {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        assert!(N as u64 <= self.len, "a read past the region");
        let mut bytes = [0; N];
        // SAFETY: the region holds at least `N` bytes of the mapping, and no command writes them
        // any more (the caller's contract); the memory is host-coherent.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().add(self.offset as usize),
                bytes.as_mut_ptr(),
                N,
            );
        }
        bytes
    }
}
