//! Where a draw reads its index buffer and its indirect arguments.
//!
//! Vulkan binds an index buffer at a multiple of the index size of its `VkBuffer` and reads
//! indirect arguments at a multiple of 4, and the binding layer bounds an index fetch at a
//! multiple of 4 bytes from the buffer's byte 0 (`IndexBufferRef::fetch_end`). Metal's offsets
//! meet these rules, but a no-copy buffer whose pointer misses 4 bytes has its byte 0 at no
//! multiple of 4 of its `VkBuffer` (`Buffer::base`), so the draw's offset there can miss them.
//! Such a draw reads a copy instead: `vkCmdCopyBuffer` takes any offset, so the bytes the draw
//! can read are copied into a new write-combined `Shared` buffer (`Device::write_combined_buffer`),
//! and the draw reads them there.
//!
//! - **Index fetches past the buffer**. Metal reads an index's bytes up to the buffer's length
//!   rounded up to 4 bytes (a partial last index reads the bytes past the length there: the
//!   padding, or a heap's bytes after a placed buffer), and draws each position from there as index
//!   0 plus the base vertex, bounded by the buffer even inside a heap (measured on an M3 Max and an
//!   M4). Vulkan states no value for an index fetched past the bound index buffer range, even under
//!   `robustBufferAccess2`, which only makes the fetch valid, so a direct draw whose positions run
//!   past where Metal's fetch stops reads a zero-padded copy ([`DrawSource::padded`]): the bytes
//!   before there copied to the start of a new buffer that holds every position, the rest zero, so
//!   every fetch lies inside the bound range and reads Metal's index on every device
//!   (`Degradation::IsolatedIndexBytes`). An indexed indirect draw, whose positions the arguments
//!   choose on the GPU, is bounded by the device's range only on a driver measured to read zero
//!   past it under `robustBufferAccess2` (`DriverFact::RobustIndexFetchReadsZero`), to which the
//!   vertex offset is added; elsewhere the index bounds pass draws it (`index_bounds.rs`). Where
//!   the device enabled `maintenance5`, that draw binds the range of the indices it reads alone
//!   ([`SizedIndexBind`], `vkCmdBindIndexBuffer2`), so the range ends where Metal's fetch stops
//!   reading the buffer, wherever its `VkBuffer` runs on (a heap's, past a placed buffer). Without
//!   it, `vkCmdBindIndexBuffer`'s range runs to the end of the `VkBuffer`, and a fetch past the
//!   buffer would read its padding, or the heap's bytes after a placed buffer: such a draw
//!   (`isolate`) reads a copy of its indices instead (`Degradation::IsolatedIndexBytes`).
//! - **The copy** of a misaligned byte 0 or an isolated fetch ends where its new buffer's
//!   `VkBuffer` ends: the new buffer is a multiple of 16
//!   bytes long, so its `VkBuffer` has no padding, and the bytes are copied to its end. An index
//!   position past them is then past the `VkBuffer`, where `robustBufferAccess2` reads zero. Where
//!   the draw reads indices to where Metal's fetch stops, which lies at a multiple of 4 from the
//!   buffer's byte 0, the byte 0 lies at a multiple of 4 in the copy too (elsewhere the draw reads
//!   no position past its bytes, so the binding layer's bound has nothing to bound). A copy of no
//!   byte (no whole index lies past the offset) leaves the 16 bytes zero and is read from their
//!   start: every position reads 0 there, and past them `robustBufferAccess2` reads 0 again. A
//!   padded copy lands at its new buffer's byte 0 (a multiple of 16 of its `VkBuffer`), and the new
//!   buffer holds every position the draw reads.
//! - **Ordering.** The copy is a command of the encoder's own command buffer, so it runs before
//!   the pass, between a barrier from everything before and one to everything after: it reads
//!   the bytes every earlier command left, as the draw itself would. Metal orders a render
//!   encoder's draws against each other's memory writes only through the program's memory
//!   barriers (`fence.rs`): where an earlier draw or tile dispatch of the encoder writes the
//!   bytes behind one, the copy runs before the draw's own render pass instance instead, after
//!   the writer's ([`CopySteps`]), and reads the write, as the draw does in Metal. A write since
//!   the last barrier races the draw's read in Metal. Before any barrier of the encoder, Metal's
//!   indexed indirect draw read the bytes from before the write in every run measured (a probe, an
//!   M3 Max and an M4), so such a draw whose arguments or indices a draw of the
//!   encoder wrote reads a copy of them taken before the pass for that reason alone
//!   ([`DrawSource::racing`]), announced once as `Degradation::UnorderedDrawBytes`. After a
//!   barrier, a rewrite since it was read in some runs, so no copy is taken
//!   there.
//! - **Bounds.** The copy holds every byte the draw reads inside the buffer: a direct draw's
//!   indices that lie wholly inside it, an indirect draw's arguments, or an indexed indirect
//!   draw's whole indices from its offset to where Metal's fetch stops, whose positions the
//!   arguments choose on the GPU. The binding layer is given the buffer's length past the offset
//!   (`DrawSource::length`), so it bounds the draw as it bounds it in the buffer; a padded copy's
//!   length is every position's, so it bounds none. The tracker takes the draw's read of the Metal
//!   buffer up to where Metal's fetch stops at most (`binding::command_uses`).

use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::degradation::Degradation;
use super::encoder::EncoderCore;
use super::error::Error;
use super::resource::ResourceKey;
use super::tracker::{Barrier, Use};
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::device::{EnableSet, VkExtension, VkFeature};
use std::ffi::CStr;

/// `vkCmdBindIndexBuffer2` (`VK_KHR_maintenance5`, core in Vulkan 1.4): binds an index buffer
/// range of a size, past which a fetch is out of range wherever the `VkBuffer` runs on.
pub(super) type SizedIndexBind = vk::PFN_vkCmdBindIndexBuffer2KHR;

/// The device's [`SizedIndexBind`], where it enabled `maintenance5`: `VK_KHR_maintenance5`'s
/// command where it enabled the extension, else Vulkan 1.4's, which enables the feature without
/// it.
///
/// # Safety
///
/// `device` was made from `instance` with `enabled` enabled, and the loader serves
/// `vkGetDeviceProcAddr` for it (its loader word is in place).
pub(super) unsafe fn sized_index_bind(
    instance: &ash::Instance,
    device: &ash::Device,
    enabled: &EnableSet,
) -> Option<SizedIndexBind> {
    if !enabled.features.contains(&VkFeature::Maintenance5) {
        return None;
    }
    let name: &CStr = if enabled.extensions.contains(&VkExtension::KhrMaintenance5) {
        c"vkCmdBindIndexBuffer2KHR"
    } else {
        c"vkCmdBindIndexBuffer2"
    };
    // SAFETY: the caller's; a command the device does not have is null.
    let command = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }?;
    // SAFETY: the device gave the command of this name, and both names share its signature.
    Some(unsafe { std::mem::transmute::<unsafe extern "system" fn(), SizedIndexBind>(command) })
}

/// How a draw binds its index buffer: at the draw's index buffer offset of `buffer`, and where
/// the device binds an index range of a size, the bytes of the whole indices the draw reads from
/// there (the module docs).
#[derive(Clone, Copy)]
pub(super) struct IndexBind {
    pub(super) buffer: vk::Buffer,
    pub(super) offset: u64,
    pub(super) index_type: vk::IndexType,
    pub(super) sized: Option<(SizedIndexBind, u64)>,
}

impl IndexBind {
    /// How `draw` binds `source`'s bytes, through `sized` where the device has it; `None` for a
    /// draw that binds no index buffer.
    pub(super) fn of(
        source: &DrawSource<'_>,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<IndexBind> {
        let (metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_type,
            index_buffer_offset,
            ..
        }
        | metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
            index_type,
            index_buffer_offset,
            ..
        }) = draw
        else {
            return None;
        };
        Some(IndexBind {
            buffer: source.raw,
            offset: index_buffer_offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, source.bound(index_buffer_offset))),
        })
    }
}

/// The bytes a draw reads at an offset of a buffer, as the draw reads them: in the buffer's
/// `VkBuffer` where Vulkan takes the offset there and bounds what the draw reads past them,
/// else in a copy.
pub(super) struct DrawSource<'a> {
    /// The Metal buffer.
    pub(super) buffer: &'a Buffer,
    /// The draw's offset in the Metal buffer.
    pub(super) offset: u64,
    /// The `VkBuffer` the draw reads.
    pub(super) raw: vk::Buffer,
    /// The draw's offset in `raw`.
    pub(super) at: u64,
    /// The end of the Metal buffer's length, in `raw`'s offsets.
    pub(super) length: u64,
    /// The end of the bytes the draw reads in `raw`.
    end: u64,
    /// The bytes are copied because a draw of the encoder wrote them before any memory barrier
    /// (the module docs).
    racing: bool,
    /// What Vulkan reads the bytes at multiples of: an index source's index size.
    alignment: u64,
    /// The copy into `raw`, when `raw` is a copy's.
    copy: Option<Copied>,
}

/// A draw's bytes read in place (`DrawSource::kept`): what a draw recorded again reads them
/// from (`render/reroute.rs`).
pub(super) struct KeptSource {
    buffer: Buffer,
    offset: u64,
    reach: u64,
    alignment: u64,
}

impl KeptSource {
    /// The device address of the bytes' first, where the device gives the buffer one.
    pub(super) fn address(&self) -> Option<u64> {
        Some(self.buffer.device_address()? + self.offset)
    }

    /// The source again: the same bytes, read in place.
    pub(super) fn source(&self) -> Result<DrawSource<'_>, Error> {
        DrawSource::new(
            &self.buffer,
            self.offset,
            self.reach,
            self.alignment,
            false,
            false,
        )
    }
}

/// A copy of a draw's bytes into a new buffer.
struct Copied {
    scratch: Buffer,
    from: vk::Buffer,
    region: vk::BufferCopy,
    /// Where the copy lands in the new buffer's bytes.
    into: u64,
    why: CopyCause,
}

/// Why a draw reads a copy of its bytes.
#[derive(Clone, Copy)]
enum CopyCause {
    /// The buffer's byte 0 lies at this byte of its `VkBuffer`, at no multiple of 4.
    Misaligned(u64),
    /// The draw can fetch indices past them, which nothing bounds at their end.
    Isolated,
    /// The draw's positions run past where Metal's fetch stops: the copy holds every position,
    /// zero past the bytes.
    Padded,
    /// A draw of the encoder wrote the bytes, before any memory barrier: the copy holds the
    /// bytes from before the write.
    Racing,
}

impl<'a> DrawSource<'a> {
    /// The bytes a draw reads at `offset` of `buffer` (at most the buffer's length rounded up to
    /// 4), `reach` of them (before where Metal's fetch stops reading the buffer for an index
    /// source), where Vulkan reads them at multiples of `alignment` of a `VkBuffer`; copied when
    /// `isolate`, when `racing` (a draw of the encoder wrote them, before any memory barrier) or
    /// where the buffer's byte 0 lies at no multiple of 4, as the module docs say. Refused as
    /// `new_buffer` refuses the copy's buffer.
    pub(super) fn new(
        buffer: &'a Buffer,
        offset: u64,
        reach: u64,
        alignment: u64,
        isolate: bool,
        racing: bool,
    ) -> Result<DrawSource<'a>, Error> {
        let (from, at) = buffer.at(offset);
        let rest = buffer.length().saturating_sub(offset);
        // The buffer's byte 0 in `from`. At a multiple of 4, `at` is a multiple of `alignment`
        // (which divides 4 and Metal's offset), and the binding layer's bound at a multiple of 4
        // from byte 0 is Metal's.
        let zero = at - offset;
        let aligned = zero.is_multiple_of(4);
        if aligned && !isolate && !racing {
            return Ok(DrawSource {
                buffer,
                offset,
                raw: from,
                at,
                length: at + rest,
                end: at + reach,
                racing,
                alignment,
                copy: None,
            });
        }
        // A multiple of 16 bytes, so the new buffer's `VkBuffer` ends where its length does.
        let size = reach.next_multiple_of(16).max(16);
        let scratch = buffer.generation().write_combined_buffer(size)?;
        let (raw, start) = scratch.at(0);
        let into = if reach == 0 { 0 } else { size - reach };
        let region = vk::BufferCopy {
            src_offset: at,
            dst_offset: start + into,
            size: reach,
        };
        let why = match (aligned, isolate) {
            (false, _) => CopyCause::Misaligned(zero),
            (true, true) => CopyCause::Isolated,
            (true, false) => CopyCause::Racing,
        };
        Ok(DrawSource {
            buffer,
            offset,
            raw,
            at: start + into,
            length: start + into + rest,
            end: start + into + reach,
            racing,
            alignment,
            copy: Some(Copied {
                scratch,
                from,
                region,
                into,
                why,
            }),
        })
    }

    /// The indices a direct draw reads at `offset` of `buffer`, whose positions run past where
    /// Metal's fetch stops reading the buffer: `reach` bytes of whole indices before there,
    /// copied to the start of a new buffer of `positions` bytes (every position the draw reads,
    /// of `alignment` bytes each), the rest of which stays zero. Every position then lies inside
    /// the bytes the binding layer is given, and the draw reads Metal's index 0 past the
    /// buffer's bytes on every device, robust or not (the module docs). Refused as `new_buffer`
    /// refuses the new buffer, for which the positions' bytes are its length.
    pub(super) fn padded(
        buffer: &'a Buffer,
        offset: u64,
        reach: u64,
        positions: u64,
        alignment: u64,
    ) -> Result<DrawSource<'a>, Error> {
        let (from, at) = buffer.at(offset);
        // A new buffer reads zero (the padding included), and its `VkBuffer` starts at a multiple
        // of 16 of it, so its byte 0 lies at a multiple of 4.
        let scratch = buffer
            .generation()
            .write_combined_buffer(positions.max(16))?;
        let (raw, start) = scratch.at(0);
        let region = vk::BufferCopy {
            src_offset: at,
            dst_offset: start,
            size: reach,
        };
        Ok(DrawSource {
            buffer,
            offset,
            raw,
            at: start,
            length: start + positions,
            end: start + positions,
            racing: false,
            alignment,
            copy: Some(Copied {
                scratch,
                from,
                region,
                into: 0,
                why: CopyCause::Padded,
            }),
        })
    }

    /// The bytes of whole multiples of the alignment (an index source's whole indices) from
    /// `offset` of `raw` to the end of the bytes the draw reads: the range an index binding of a
    /// size binds there (the module docs).
    pub(super) fn bound(&self, offset: u64) -> u64 {
        self.end.saturating_sub(offset) / self.alignment * self.alignment
    }

    /// The Metal buffer and the draw's offset in it, as the tracker takes the draw's use.
    pub(super) fn metal(&self) -> (&'a Buffer, u64) {
        (self.buffer, self.offset)
    }

    /// The index positions a [`DrawSource::padded`] copy holds past the copied bytes, which
    /// read index 0; `None` for every other source.
    pub(super) fn padding(&self) -> Option<u64> {
        let copy = self.copy.as_ref()?;
        match copy.why {
            CopyCause::Padded => {
                Some((self.end - self.at).saturating_sub(copy.region.size) / self.alignment)
            }
            CopyCause::Misaligned(_) | CopyCause::Isolated | CopyCause::Racing => None,
        }
    }

    /// What makes this source again (`KeptSource::source`) where the draw reads the buffer in
    /// place: `None` for a copy's.
    pub(super) fn kept(&self) -> Option<KeptSource> {
        match (&self.copy, self.racing) {
            (None, false) => Some(KeptSource {
                buffer: self.buffer.clone(),
                offset: self.offset,
                reach: self.end - self.at,
                alignment: self.alignment,
            }),
            _ => None,
        }
    }

    /// Whether the draw reads the Metal buffer itself, not a copy of its bytes.
    pub(super) fn in_place(&self) -> bool {
        self.copy.is_none()
    }

    /// Whether the draw reads a copy because a draw of the encoder wrote the bytes, before any
    /// memory barrier (the module docs).
    pub(super) fn racing(&self) -> bool {
        self.racing
    }

    /// The draw's read of the Metal buffer where it reads a copy of bytes: what the copy reads,
    /// before the pass or before the draw's render pass instance (`render.rs`).
    pub(super) fn copy_reads(&self) -> Option<Use<ResourceKey>> {
        let copy = self.copy.as_ref()?;
        let size = copy.region.size;
        (size > 0).then(|| reads(self.buffer, self.offset..self.offset + size))
    }

    /// Hold the copy's buffer, if the draw reads one, until `core`'s command buffer completes,
    /// and announce it once; the copy of any bytes, which records before the pass or before the
    /// draw's render pass instance (`render.rs`). Call it once the draw records.
    pub(super) fn copy_steps(&self, core: &mut EncoderCore) -> Option<CopySteps> {
        let copy = self.copy.as_ref()?;
        core.retain(copy.scratch.clone());
        let size = copy.region.size;
        if size == 0 {
            return None;
        }
        let (offset, at) = (self.offset, copy.region.src_offset);
        let device = self.buffer.generation();
        match copy.why {
            CopyCause::Misaligned(zero) => device.degrade(Degradation::CopiedDrawBytes, || {
                let object =
                    format!("a draw's index buffer or indirect arguments at offset {offset}");
                let reason = format!(
                    "the buffer's byte 0 lies at byte {zero} of its VkBuffer, at no multiple of \
                     4, where Vulkan reads indirect arguments and the draw bounds its indices \
                     (they start at byte {at})"
                );
                (object, reason)
            }),
            CopyCause::Isolated => device.degrade(Degradation::IsolatedIndexBytes, || {
                let object = format!("an indexed draw's index buffer at offset {offset}");
                let reason = format!(
                    "the draw can fetch indices past its {size} bytes of whole indices, and \
                     without maintenance5 the bound range runs on to the end of the buffer's \
                     VkBuffer"
                );
                (object, reason)
            }),
            CopyCause::Padded => device.degrade(Degradation::IsolatedIndexBytes, || {
                let object = format!("an indexed draw's index buffer at offset {offset}");
                let reason = format!(
                    "the draw's positions run past its {size} bytes of whole indices, and \
                     Vulkan states no index for a fetch past the bound range"
                );
                (object, reason)
            }),
            // Announced below.
            CopyCause::Racing => {}
        }
        if self.racing {
            device.degrade(Degradation::UnorderedDrawBytes, || {
                let object = format!(
                    "an indexed indirect draw's index buffer or arguments at offset {offset}"
                );
                let reason = format!(
                    "a draw of the encoder wrote those {size} bytes with no memory barrier \
                     between, a race Metal does not order; Metal's draw read the bytes from \
                     before the write in every run measured"
                );
                (object, reason)
            });
        }
        let into = copy.into;
        Some(CopySteps {
            from: copy.from,
            to: self.raw,
            region: copy.region,
            uses: [
                reads(self.buffer, self.offset..self.offset + size),
                writes(&copy.scratch, into..into + size),
            ],
        })
    }
}

/// A draw's copy of its bytes into a new buffer ([`DrawSource::copy_steps`]), which the command
/// buffer holds.
pub(super) struct CopySteps {
    from: vk::Buffer,
    to: vk::Buffer,
    region: vk::BufferCopy,
    /// Its read of the Metal buffer, and its write of the new one.
    uses: [Use<ResourceKey>; 2],
}

impl CopySteps {
    /// Its read of the Metal buffer, and its write of the new one.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the copy into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
    pub(super) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &self.uses, |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass.
            unsafe { self.copy(device, command_buffer) }
        });
        core.barrier(Barrier::everything());
    }

    /// Record the copy into `command_buffer` between a barrier from everything before and one to
    /// everything after: before a render pass instance of the encoder's pass, after the
    /// instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.copy(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn copy(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; both buffers are live until the command buffer completes
        // (the encoder holds the Metal buffer, and the command buffer the new one), and the
        // region lies in both: the Metal buffer's length from the offset, and the new buffer's
        // end.
        unsafe { raw.cmd_copy_buffer(command_buffer, self.from, self.to, &[self.region]) };
    }
}
