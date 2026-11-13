//! The blit encoder's indirect command buffer commands (`copyIndirectCommandBuffer:...`,
//! `resetCommandsInBuffer:withRange:`, `optimizeIndirectCommandBuffer:withRange:`).
//!
//! - **What Metal answers** (an M3 Max, with and without `MTL_DEBUG_LAYER`). A copy between buffers
//!   of one descriptor copies the commands: the destination's getters and execution read them.
//!   Between buffers of different descriptors the copy is garbled, and a copy that runs past the
//!   destination's end corrupts its command 0; validation stops neither. A reset resets the
//!   commands as `resetWithRange:` does. An optimize changes nothing a getter or an execution
//!   observes.
//! - **The runtime.** The commands are records of one format whatever the descriptor, so a copy
//!   copies the records' bytes ([`BlitCommandEncoder::copy_from_buffer_to_buffer`], which moves
//!   overlapping ranges of one buffer as `memmove` does); a record the destination's descriptor
//!   does not allow runs nothing where an execution meets it (`ReplayFaults::RECORD`), the defined
//!   answer for Metal's garbled one; it sets the destination commands' written flags
//!   (`indirect_commands::written_flag_offset`) by one fill. A reset writes the reset record, all
//!   zero (`CommandRecord::default`'s bytes), over each command whose written flag is set, and
//!   clears the flag, by dispatches of `bind::icb_reset_module`, one workgroup a command, each
//!   dispatch a run of commands whose records one storage-buffer binding holds: every writer of
//!   the records sets the flag of each record it writes (a kernel's encoder operations, the CPU's
//!   setters, a copy) and the CPU's writes of the reset record clear it, so a clear flag's record
//!   holds the reset record already, and a reset of many commands of which few were written
//!   stores those few records rather than every record of the range (where the records lie in
//!   host memory across a discrete GPU's link, the range's whole bytes over the link). Both write
//!   the destination's records on the GPU as the command buffer runs, so the call marks it
//!   written by the GPU (`IndirectCommandBuffer::written_by_gpu`): its executions run the records
//!   as they stand once the work before them completed. An optimize records nothing. Counted as
//!   encoded, in the records' bytes of the range (`COMMAND_BYTES` a command, whatever the
//!   descriptor, a reset's flagged or not): `WorkCounts::indirect_record_bytes_gpu_copied` and
//!   `indirect_record_bytes_gpu_reset`.
//! - **What a reset costs** (an RTX 2080 on a PCIe 3.0 x4 link, NVIDIA 620.70.06, release, the
//!   median GPU time of 15 command buffers of one reset each, two runs alike). A reset of 45,000
//!   render commands (101 MB of records in host-cached system memory) took 0.26 ms with no command
//!   written since the last reset, 1.74 ms with 1,300 written by the CPU, and 60.5 ms with every
//!   command written, where one fill of the same records took 56.8 ms whatever was written. So a
//!   buffer whose every command is written between resets resets about 6% slower than a fill
//!   would, a known gap; the flags' reads alone took 0.26 ms of it. A kernel's flag stores, one
//!   per operation, took a kernel that encodes 45,000 render commands (two buffers and a draw
//!   each) from 19.80 ms to 20.11 ms.
//! - **A buffer of no command types beside one of command types.** The copy completes and moves
//!   nothing, as plain Metal's does. Metal's own answer, read through what the destination then
//!   paints rather than through its status (an M3 Max, macOS 27.0.1, build 26A434, the validation
//!   layer off, one shape per process, four processes a shape, all four alike, no byte of a 0xEE
//!   guard at the next 256-aligned placement-heap offset past the vertex buffer touched;
//!   `probes/icb_argument_encode_and_copy_destinations.m`): a copy of 4 commands from a buffer of
//!   no command types over a `Draw` buffer whose command 0 held a draw of the whole target
//!   completed, status 4 and no error, and the destination still painted 0.75 — so nothing moved,
//!   not even the source's reset records. Both controls ran live beside it: the same copy from a
//!   fresh `Draw` buffer left -1 (the copy does write), and a copy from a `Draw` buffer that held
//!   the draw into a reset `Draw` buffer painted 0.75 (the copy does carry records). The other
//!   direction is beyond any public Metal call: a buffer of no command types has no setter Metal
//!   survives, no read-back, an execution that paints nothing, and a copy out of it that moves
//!   nothing, so a copy into it cannot be read — `Draw` -> none -> `Draw` painted -1 where
//!   `Draw` -> `Draw` -> `Draw` painted 0.75, which the second copy alone accounts for. Recording
//!   nothing is the one answer that matches every observation there is. Metal places no indirect
//!   command buffer in a placement heap, so no guard can sit beside a copy's destination.
//! - **Refused** (`Invalid`, changing nothing): a buffer of another device, and a range past
//!   either buffer's size, which Metal runs as a corruption no program relies on.
//! - **What a command holds** (the M4 probe, plain Metal). A copy and a reset hold the
//!   records they read and write until the command buffer completes, but not the buffers: a buffer
//!   the program releases meanwhile gives its ID back at once, as Metal handed a blit's buffer's ID
//!   out again while the blit waited. An optimize holds nothing.

use super::blit::{reads, writes, writes_by, BlitCommandEncoder};
use super::error::{internal, Error};
use super::indirect_command_buffer::{IndirectCommandBuffer, Overflow};
use super::owned_kernel::{OwnedBinds, OwnedModule, OwnedSet};
use super::replay::BlitCall;
use super::tracker::Access;
use super::write_routes::GpuWriter;
use ash::vk;
use metal2vulkan::bind::indirect_commands::COMMAND_BYTES;
use metal2vulkan::bind::IcbResetModule;
use std::ops::Range;

impl BlitCommandEncoder {
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`: commands
    /// `source_range` of `source` become commands `destination_index..` of `destination` as the
    /// command buffer runs, as on Metal (the module docs). An empty range copies nothing.
    ///
    /// A buffer of no command types beside one of command types copies nothing, as plain Metal's
    /// does (the module docs); two buffers of no command types copy as any others, as plain
    /// Metal's do.
    ///
    /// Refused (`Invalid`): a buffer of another device, and a range past either buffer's size
    /// (`IndirectCommandBufferProblem::Range`).
    pub fn copy_indirect_command_buffer(
        &mut self,
        source: &IndirectCommandBuffer,
        source_range: Range<u64>,
        destination: &IndirectCommandBuffer,
        destination_index: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyIndirectCommandBuffer(Box::new((
                    source.clone(),
                    source_range.clone(),
                    destination.clone(),
                    destination_index,
                )))
            },
            |this| {
                const CALL: &str = "copy_indirect_command_buffer";
                this.core().check_device(CALL, source.generation())?;
                this.core().check_device(CALL, destination.generation())?;
                let from = source.command_range(CALL, source_range.clone())?;
                let count = u64::from(from.end - from.start);
                let to = destination.command_range(
                    CALL,
                    destination_index..destination_index.saturating_add(count),
                )?;
                // A buffer of no command types beside one of command types: the records of one
                // side are of commands the other's descriptor has no kind for at all, and Metal
                // moves none of them (the module docs), so the copy records nothing.
                if source.has_no_command_types() != destination.has_no_command_types() {
                    return Ok(());
                }
                if from.is_empty() {
                    return Ok(());
                }
                destination.written_by_gpu(GpuWriter::Copy);
                this.core().retain(destination.gpu_write());
                destination.keep_copied(source, from.clone());
                destination.written().copied_from(source.written()).mark();
                // The slots past the records', carried as Metal's copy carries every slot below the
                // count.
                this.copy_overflow(source, from.clone(), destination, to.start)?;
                // The destination commands' written flags, set: a reset writes their records.
                let flags = destination.flag_bytes(to.clone());
                let storage = destination.storage();
                let (raw, at) = storage.at(flags.start);
                let size = flags.end - flags.start;
                this.command(&[writes(storage, flags)], |device, command_buffer| {
                    // SAFETY: the encoder's command buffer is recording; the storage is live and
                    // retained until it completes; the flags lie inside it, their offset and size
                    // multiples of 4 (`u32`s past the records, which end at a multiple of 8).
                    unsafe { device.cmd_fill_buffer(command_buffer, raw, at, size, 1) };
                })?;
                let bytes = IndirectCommandBuffer::record_bytes(from);
                let at = IndirectCommandBuffer::record_bytes(to).start;
                let counts = &this.core().device().shared().counts;
                counts.indirect_record_bytes_gpu_copied(bytes.end - bytes.start);
                this.unlogged(|this| {
                    this.copy_from_buffer_to_buffer(
                        source.storage(),
                        bytes.start,
                        destination.storage(),
                        at,
                        bytes.end - bytes.start,
                    )
                })?;
                this.core().retain(source.storage().clone());
                this.core().retain(destination.storage().clone());
                Ok(())
            },
        )
    }

    /// `resetCommandsInBuffer:withRange:`: commands `range` of `buffer` are reset, as a fresh
    /// buffer's, as the command buffer runs. An empty range resets nothing.
    ///
    /// Refused (`Invalid`): a buffer of another device, and a range past its size
    /// (`IndirectCommandBufferProblem::Range`). A buffer of no command types is taken as any
    /// other, as plain Metal takes it.
    pub fn reset_commands_in_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::ResetCommandsInBuffer(buffer.clone(), range.clone()),
            |this| {
                const CALL: &str = "reset_commands_in_buffer";
                this.core().check_device(CALL, buffer.generation())?;
                let range = buffer.command_range(CALL, range.clone())?;
                if range.is_empty() {
                    return Ok(());
                }
                buffer.written_by_gpu(GpuWriter::Reset);
                this.core().retain(buffer.gpu_write());
                let bytes = IndirectCommandBuffer::record_bytes(range.clone());
                let counts = &this.core().device().shared().counts;
                counts.indirect_record_bytes_gpu_reset(bytes.end - bytes.start);
                this.reset_flagged(buffer, range.clone())?;
                this.core().retain(buffer.storage().clone());
                // The slots past the records', cleared as Metal's reset clears every slot.
                if let Some(overflow) = buffer.overflow() {
                    this.clear_overflow(overflow, overflow.bytes(range))?;
                }
                Ok(())
            },
        )
    }

    /// Reset the flagged commands of `range` of `buffer` (the module docs), by dispatches of the
    /// flagged reset, each a run of commands whose records' and flags' bindings start at the
    /// device's storage-buffer alignment below them and fit its range and workgroup count.
    fn reset_flagged(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u32>,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let kernel = device.owned_kernel(OwnedModule::IcbReset)?;
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let most = u64::from(limits.max_storage_buffer_range);
        // Commands a run takes: its records from below an aligned offset within the range (at
        // least 2^27 bytes, `maxStorageBufferRange`'s least, holds many), and one workgroup each.
        let run = most.saturating_sub(align.max(8)) / COMMAND_BYTES as u64;
        let run = run.min(u64::from(limits.max_compute_work_group_count[0]));
        let run = u32::try_from(run).unwrap_or(u32::MAX).max(1);
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let access = Access::new(
            compute,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let storage = buffer.storage();
        let mut start = range.start;
        while start < range.end {
            let end = start.saturating_add(run).min(range.end);
            let records = IndirectCommandBuffer::record_bytes(start..end);
            let flags = buffer.flag_bytes(start..end);
            let (raw, record_at) = storage.at(records.start);
            let (_, flag_at) = storage.at(flags.start);
            // The records binding holds 8-byte pairs: its offset at a multiple of 8 too, and the
            // records at one (`COMMAND_BYTES`: the storage is 8-byte aligned).
            if !record_at.is_multiple_of(8) {
                return Err(internal(
                    "an indirect command buffer's records not 8-byte aligned",
                ));
            }
            let record_offset = record_at - record_at % align.max(8);
            let flag_offset = flag_at - flag_at % align;
            let set = OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: vec![
                    (
                        raw,
                        record_offset,
                        record_at - record_offset + (records.end - records.start),
                    ),
                    (
                        raw,
                        flag_offset,
                        flag_at - flag_offset + (flags.end - flags.start),
                    ),
                ],
            };
            let binds = OwnedBinds::new(&device, &kernel, &[set])?;
            // Each first place below the alignment, a `u32`.
            let firsts = IcbResetModule::firsts(
                ((record_at - record_offset) / 8) as u32,
                ((flag_at - flag_offset) / 4) as u32,
            );
            let [x, y, z] = IcbResetModule::groups(end - start);
            let uses = [
                writes_by(storage, records, access),
                writes_by(storage, flags, access),
            ];
            self.command_in(compute, &uses, |device, command_buffer| {
                // SAFETY: the encoder's command buffer is recording; the kernel is the device's,
                // and the storage is live and retained until the command buffer completes (the
                // caller); the bindings hold the run's records and flags, within the device's
                // range and alignment, and the dispatch is one workgroup a command of the run.
                unsafe {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline,
                    );
                    binds.bind(device, command_buffer, &kernel, 0);
                    device.cmd_push_constants(
                        command_buffer,
                        kernel.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &firsts,
                    );
                    device.cmd_dispatch(command_buffer, x, y, z);
                }
            })?;
            if let OwnedBinds::Sets(sets) = binds {
                self.core().retain(sets);
            }
            start = end;
        }
        Ok(())
    }

    /// Carry the slots past the records' of `source`'s commands `from` into `destination`'s from
    /// command `to`, as the copy carries the records: every slot of a stage both hold, and the
    /// destination's slots past the source's cleared, which the source holds unset; and the
    /// threadgroup lengths and words past the records', lengths 36 and 37 cleared where the
    /// destination's commands hold pipeline words, as Metal's copy clears them
    /// (`Overflow::copy_clears`).
    fn copy_overflow(
        &mut self,
        source: &IndirectCommandBuffer,
        from: Range<u32>,
        destination: &IndirectCommandBuffer,
        to: u32,
    ) -> Result<(), Error> {
        let Some(into) = destination.overflow() else {
            return Ok(());
        };
        let count = from.end - from.start;
        let held = match source.overflow() {
            None => return self.clear_overflow(into, into.bytes(to..to + count)),
            Some(held) if held.same_layout(into) => {
                let bytes = held.bytes(from);
                let at = into.bytes(to..to).start;
                self.unlogged(|this| {
                    this.copy_from_buffer_to_buffer(
                        held.storage(),
                        bytes.start,
                        into.storage(),
                        at,
                        bytes.end - bytes.start,
                    )
                })?;
                self.core().retain(held.storage().clone());
                self.core().retain(into.storage().clone());
                return self.clear_runs(into, to..to + count);
            }
            Some(held) => held,
        };
        // Two layouts, so two buffers: no copy overlaps another.
        let read = held.bytes(from.clone());
        let (mut copies, mut clears) = (Vec::new(), Vec::new());
        for (offset, index) in from.enumerate() {
            let runs = held.runs(index, into, to + offset as u32);
            copies.extend(runs.copies);
            clears.extend(runs.clears);
        }
        let (source_raw, _) = held.storage().at(0);
        let (raw, _) = into.storage().at(0);
        let regions: Vec<vk::BufferCopy> = copies
            .iter()
            .map(|&(read, write, size)| vk::BufferCopy {
                src_offset: held.storage().at(read).1,
                dst_offset: into.storage().at(write).1,
                size,
            })
            .collect();
        let fills: Vec<(u64, u64)> = clears
            .iter()
            .map(|&(at, size)| (into.storage().at(at).1, size))
            .collect();
        let written = into.bytes(to..to + count);
        let uses = [reads(held.storage(), read), writes(into.storage(), written)];
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording; both buffers are live and
            // retained until it completes; each region lies inside both, and none overlaps
            // another (two buffers).
            unsafe {
                if !regions.is_empty() {
                    device.cmd_copy_buffer(command_buffer, source_raw, raw, &regions);
                }
                for &(at, size) in &fills {
                    device.cmd_fill_buffer(command_buffer, raw, at, size, 0);
                }
            }
        })?;
        self.core().retain(held.storage().clone());
        self.core().retain(into.storage().clone());
        Ok(())
    }

    /// Clear the runs of commands `range` of `overflow` a copy into them clears after it carried
    /// the source's blocks (`Overflow::copy_clears`), by one copy from zeroed bytes of the upload
    /// arena: a fill a run costs the device a command each.
    fn clear_runs(&mut self, overflow: &Overflow, range: Range<u32>) -> Result<(), Error> {
        let runs = overflow.copy_clears(range.clone());
        let Some(&(_, size)) = runs.first() else {
            return Ok(());
        };
        let device = self.core().device().clone();
        let zeros = vec![0u8; size as usize];
        let upload = self
            .core()
            .command_buffer()
            .state()
            .uploads
            .upload(&device, &zeros, size)?;
        let storage = overflow.storage();
        let regions: Vec<vk::BufferCopy> = runs
            .iter()
            .map(|&(at, size)| vk::BufferCopy {
                src_offset: upload.offset(),
                dst_offset: storage.at(at).1,
                size,
            })
            .collect();
        let (from, (raw, _)) = (upload.raw(), storage.at(0));
        self.command(
            &[writes(storage, overflow.bytes(range))],
            |device, command_buffer| {
                // SAFETY: the encoder's command buffer is recording; the overflow is live and
                // retained until it completes, and the arena's region until the command buffer
                // completes; each region lies inside both, and none overlaps another.
                unsafe { device.cmd_copy_buffer(command_buffer, from, raw, &regions) };
            },
        )?;
        self.core().retain(storage.clone());
        Ok(())
    }

    /// Clear `bytes` of `overflow`, the slots of commands a reset or a copy clears.
    fn clear_overflow(&mut self, overflow: &Overflow, bytes: Range<u64>) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        let storage = overflow.storage();
        let (raw, at) = storage.at(bytes.start);
        let size = bytes.end - bytes.start;
        self.command(&[writes(storage, bytes)], |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording; the overflow is live and
            // retained until it completes; the range lies inside it, its offset and size multiples
            // of 4 bytes.
            unsafe { device.cmd_fill_buffer(command_buffer, raw, at, size, 0) };
        })?;
        self.core().retain(storage.clone());
        Ok(())
    }

    /// `optimizeIndirectCommandBuffer:withRange:`: nothing Metal's optimize changes is
    /// observable, so it records nothing.
    ///
    /// Refused (`Invalid`): a buffer of another device, and a range past its size
    /// (`IndirectCommandBufferProblem::Range`). A buffer of no command types is taken as any
    /// other, as plain Metal takes it.
    pub fn optimize_indirect_command_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::OptimizeIndirectCommandBuffer(buffer.clone(), range.clone()),
            |this| {
                const CALL: &str = "optimize_indirect_command_buffer";
                this.core().check_device(CALL, buffer.generation())?;
                buffer.command_range(CALL, range.clone())?;
                Ok(())
            },
        )
    }
}
