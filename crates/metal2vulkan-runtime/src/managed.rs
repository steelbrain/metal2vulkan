//! Managed storage (`MTLStorageModeManaged`): `Buffer::did_modify_range`.
//! `BlitCommandEncoder::synchronize_resource` is in `blit.rs`.
//!
//! A managed buffer follows Metal's contract for the device's memory model
//! ([`DeviceProperties::has_unified_memory`](super::DeviceProperties::has_unified_memory)):
//!
//! - **With unified memory** (an integrated GPU, a CPU implementation), it lives in host-visible
//!   coherent memory, cached where the device has such
//!   ([`MemoryClass::Managed`](super::memory::MemoryClass)), which the GPU uses itself, as Apple
//!   silicon keeps managed resources in memory the CPU and GPU share. Coherent memory needs no
//!   flush and no invalidate: the CPU's writes reach the GPU at the next submission, and every
//!   command buffer ends with a barrier to the host. So these calls check a managed buffer's
//!   range and record nothing more. On a resource of another storage mode they do nothing, as
//!   plain Metal does (its validation stops them).
//! - **Without unified memory** (a discrete GPU), it has two copies, as Metal's managed resources
//!   have on such a device: its own bytes in device-local memory, which the GPU uses, and a host
//!   copy in such cached memory, which `contents` maps (`buffer.rs`'s `Mirror`). `did_modify_range`
//!   copies the range from the host copy to the device's by the device's next submission, before
//!   any of its commands, and does not wait for it (it waited before); `synchronize_resource`
//!   copies the buffer back where the command buffer runs it. A program that skips them sees stale
//!   bytes there, as on a Mac with a discrete GPU. A GPU writing host-cached memory across the bus
//!   ran a kernel's writes at 0.03 to 0.1 GB/s on an RTX 3090 and an RTX 5080, where a device-local
//!   write and a copy back ran at several GB/s. A no-copy buffer is the caller's
//!   memory and has no device copy.

use super::buffer::Buffer;
use super::error::{invalid, Error, Problem};
use super::values::StorageMode;
use std::ops::Range;

impl Buffer {
    /// `didModifyRange:`: the CPU wrote `range` of a managed buffer; command buffers committed
    /// after this call see it. Where the buffer has a host copy (a device without unified memory),
    /// the range is copied to its own bytes by the device's next submission, before any of its
    /// commands; the call does not wait for the GPU. On a buffer of another storage mode it does nothing, whatever the range, as
    /// plain Metal does (RT2 `managed.didmodify.shared`: `Shared` and `Private`
    /// buffers, in range and past it; Metal's validation stops it). Refused for a managed
    /// buffer's range past the buffer.
    pub fn did_modify_range(&self, range: Range<u64>) -> Result<(), Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        const CALL: &str = "did_modify_range";
        if self.storage_mode() != StorageMode::Managed {
            return Ok(());
        }
        let length = self.length();
        if range.start > range.end || range.end > length {
            return Err(invalid(
                CALL,
                Problem::RangeOutOfBounds {
                    offset: range.start,
                    size: range.end.saturating_sub(range.start),
                    length,
                },
            ));
        }
        self.upload(range)
    }
}
