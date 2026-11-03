//! Indirect dispatches:
//! `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`, and
//! `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`, the compute encoder's SPI that
//! Metal 4's encoder makes public.
//!
//! - **What Metal takes and runs**, measured by our probes on an M3 Max
//!   (macOS 26.7) and an M4 (macOS 27.0.1), identically. API validation stops an argument offset
//!   that is not a multiple of 4 ("indirectBufferOffset(2) must be a multiple of 4 bytes"), and
//!   arguments that run past the buffer ("(indirectBufferOffset +
//!   sizeof(MTLDispatchThreadsIndirectArguments))(72) must be <= \[indirectBuffer length\](64)").
//!   Plain Metal reads the arguments at the offset rounded down to a multiple of 4, and so does the
//!   runtime, out of the buffer's allocation and not out of its length: arguments past the length
//!   are taken as far as the bytes the runtime holds at Metal's address (`Buffer::copy_reach`, a
//!   no-copy buffer's host pages to the end of its `VkBuffer`, else the length), and refused past
//!   them as the dispatch is encoded ([`arguments_offset`], which carries the measurement).
//!   The dispatch reads its arguments as it runs,
//!   after what the encoder's earlier commands wrote. A grid of `threadsPerGrid` runs its edge
//!   threadgroups at the remainder's size, as `dispatchThreads:` does. A zero axis of the grid or
//!   of the group counts runs nothing, and so does a threadgroup with a zero axis or more threads
//!   than Metal's limit for the state (Apple's 1024, or the function's or the descriptor's lower
//!   one), with no error. One within Metal's limit but past the state's
//!   `maxTotalThreadsPerThreadgroup`, where this device's limit is lower, runs on Metal: the flat
//!   path runs it exactly (its kernel does not observe Metal's threadgroups), and elsewhere its
//!   plan refuses it typed as the device's gap, at the host step as `CommandBufferError::Refused`.
//!   Metal runs every group count up to `2^32 - 1` on x and on y (and `2^24` on z, the largest
//!   measured), which is past every committed Vulkan profile's `maxComputeWorkGroupCount` on some
//!   axis.
//! - **The GPU path** (on the flat variants). Both calls of a kernel that does not observe its
//!   threadgroup partition (`KernelFacts::partition_oblivious`), on a device with device addresses,
//!   are recorded where they are encoded, on the kernel's flat variant
//!   (`ComputePipelineState::settle_flat`) in threadgroups of [`FLAT_THREADS`] threads, or of the
//!   state's most where fewer ([`flat_local`]): the crate's flat-dispatch module
//!   (`bind::flat_dispatch_module`, an owned kernel) reads Metal's arguments by device address as
//!   the command runs and writes the region record and the `VkDispatchIndirectCommand` into 80
//!   bytes of the command buffer's upload arena ([`FlatPreprocess`]); a barrier; then the variant's
//!   `vkCmdDispatchIndirect` from that command, its `ReplayRegionAddress` the record's address. The
//!   module's read of the arguments is one of the command's uses ([`FlatCall::reads`]), so the
//!   barriers the encoder and the command buffer record order it after what wrote them, as Metal's
//!   dispatch reads them when it runs. Where Metal runs nothing the command is zero, and nothing is
//!   reported. A grid the variant's threadgroups cannot spread within the device's group-count
//!   limits, or a threadgroup dispatch above `2^32 - 1` threads on an axis, runs nothing and is the
//!   replay fault `GROUP_COUNT` of the command buffer's `execution_report`, where the host step
//!   splits the first exactly and no measurement says what Metal's grid is for the second. Nothing
//!   is announced.
//! - **The folded path**. `dispatch_threadgroups_indirect` of a kernel that observes its partition,
//!   on a device with device addresses, is recorded where it is encoded too, on the kernel's folded
//!   variant of the call's threadgroup size (`ComputePipelineState::settle_folded`,
//!   `VariantKey::folded_groups`), behind the same preprocess in its folded form
//!   (`FlatArguments::FoldedThreadgroups`): the product of Metal's group counts spreads over the
//!   device's group-count limits, the record holds the grid in threads and the threadgroup size,
//!   and the variant counts Metal's threadgroups from its workgroups, those past the last
//!   returning before the body. So every group count Metal runs, past the device's
//!   `maxComputeWorkGroupCount` too, runs as Metal runs it in one `vkCmdDispatchIndirect`. A grid
//!   above `2^32 - 1` threads on an axis runs nothing and is the replay fault `GROUP_COUNT`, as
//!   on the flat path. Nothing is announced.
//! - **The host step** keeps every other indirect dispatch: `dispatch_threads_indirect` of a
//!   kernel that observes its partition (its threadgroup size is GPU data, and Vulkan fixes a
//!   pipeline's), a kernel that reads a stage-input region or whose encoder set an indirect
//!   one, a threadgroup the folded path's form cannot run (a zero axis, or more threads than
//!   the state's most), and a device without device addresses (the compile
//!   layer's refusals, `Unsupported{GpuEncodedIndirectCommands}`, which the state remembers).
//!   Where the dispatch is encoded, the encoder copies its arguments into a `Shared` buffer of
//!   its own, behind a barrier from everything before, so the copy reads what Metal's dispatch
//!   would. The dispatch is then
//!   a segment of the encoder that starts a part, and a batch, of its own (`deferred.rs`). The
//!   scheduler's host step records it once the work before it completed: from the arguments
//!   then, as `dispatch_threadgroups` or `dispatch_threads` with the encoder's tables at the call
//!   and the command buffer's resident resources, every encoder's (`residency.rs`) records it
//!   ([`ArgumentDispatch::record`]; the regions split a group count past the device's limit
//!   exactly), or as nothing where Metal runs nothing. The device announces it once
//!   (`Degradation::HostReadDispatchArguments`). `commit` compiles the variants the dispatch may
//!   run on for its bind shape, whatever its threadgroup size (`ComputePipelineState::prepare`),
//!   so the step compiles nothing; a threadgroup size its state has not dispatched makes that
//!   `VkPipeline` then.

use super::argument_table::ArgumentTable;
use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::compute::{prepare_variants, record_dispatch, DispatchTables};
use super::degradation::Degradation;
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{internal, invalid, Error, Problem};
use super::owned_kernel::{OwnedKernel, OwnedModule};
use super::pipeline::{BufferForms, ComputePipelineState, Variant};
use super::predication::HostGate;
use super::residency::Residency;
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::Upload;
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    Dispatch, FlatArguments, FlatDispatchModule, FlatIndirect, ImageblockSize, PredicationStep,
    StageInRegion,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The bytes of the copy of a dispatch's arguments: the longest arguments, rounded up to 16
/// bytes, so the buffer's `VkBuffer` ends where its length does.
const COPY_BYTES: u64 = 32;

/// The bytes of `MTLStageInRegionIndirectArguments`: the region's origin, then its size, each
/// three `u32`s (`setStageInRegionWithIndirectBuffer:indirectBufferOffset:`).
pub(super) const STAGE_IN_REGION_BYTES: u64 = 24;

/// The most threads a flat variant's threadgroup holds (the module docs).
const FLAT_THREADS: u32 = 64;

/// What an indirect dispatch's arguments state.
#[derive(Clone, Copy, Debug)]
pub(super) enum Arguments {
    /// `MTLDispatchThreadgroupsIndirectArguments`: the group counts, of threadgroups of the size
    /// the call states.
    Threadgroups { threads_per_group: [u32; 3] },
    /// `MTLDispatchThreadsIndirectArguments`: the grid's threads, then the threadgroup size.
    Threads,
}

impl Arguments {
    /// The arguments' bytes.
    pub(super) fn bytes(self) -> u64 {
        match self {
            Arguments::Threadgroups { .. } => 12,
            Arguments::Threads => 24,
        }
    }

    /// The dispatch the arguments' `words` state for a state of `max_threads` threads a
    /// threadgroup at most, or `None` where Metal runs nothing (the module docs): a zero axis of
    /// the grid or the group counts, or a threadgroup with no thread or more than `max_threads`.
    fn dispatch(self, words: [u32; 6], max_threads: u32) -> Option<Dispatch> {
        let [x, y, z, a, b, c] = words;
        let threads_per_group = match self {
            Arguments::Threadgroups { threads_per_group } => threads_per_group,
            Arguments::Threads => [a, b, c],
        };
        let threads: u64 = threads_per_group.iter().map(|&n| u64::from(n)).product();
        if [x, y, z].contains(&0) || threads == 0 || threads > u64::from(max_threads) {
            return None;
        }
        Some(match self {
            Arguments::Threadgroups { .. } => Dispatch::Threadgroups {
                groups: [x, y, z],
                threads_per_group,
            },
            Arguments::Threads => Dispatch::Threads {
                threads: [x, y, z],
                threads_per_group,
            },
        })
    }
}

/// The offset Metal reads arguments of `bytes` at, given `offset` of a buffer of `length` bytes
/// the runtime holds `reach` of (`Buffer::copy_reach`): `offset` rounded down to a multiple of 4,
/// as plain Metal reads them where its API validation asserts (the module docs).
///
/// Arguments that run past the buffer's `length` are taken as far as the reach, since Metal reads
/// them out of the buffer's allocation and not out of its length. Measured on an Apple M3 Max
/// under macOS 27.0.1: `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:0` on a
/// no-copy buffer of length 8 over a 16 KiB page holding the group counts `(7, 1, 1)` ran, and its
/// kernel counted 7 threadgroups, so Metal read all three `u32`s, the third at byte 8 -- past the
/// length, inside the allocation -- while the same call on a 256-byte buffer of `(3, 1, 1)`
/// counted 3, so each count is read from the buffer and not from a copy the encoder kept. Metal's
/// API validation layer reports it ("(indirectBufferOffset +
/// sizeof(MTLDispatchThreadgroupsIndirectArguments))(12) must be <= \[indirectBuffer length\](8)")
/// and the dispatch ran all the same, as plain Metal's did; a program running with the layer and
/// `MTL_DEBUG_LAYER_ERROR_MODE=assert` ends there instead.
///
/// Refused (`IndirectDispatchRange`) past the reach, where the runtime holds none of the bytes
/// Metal reads at that address, as a blit's read past a buffer's reach is refused
/// (`Buffer::copy_reach`).
pub(super) fn arguments_offset(
    call: &'static str,
    (length, reach): (u64, u64),
    offset: u64,
    bytes: u64,
) -> Result<u64, Error> {
    let offset = offset - offset % 4;
    if offset.checked_add(bytes).is_none_or(|end| end > reach) {
        return Err(invalid(
            call,
            Problem::IndirectDispatchRange {
                offset,
                bytes,
                length,
            },
        ));
    }
    Ok(offset)
}

/// The copy of `bytes` of arguments at `offset` of `buffer`, recorded into `core` where the
/// dispatch is encoded, behind a barrier from everything before (the module docs). Refused as
/// `new_buffer` refuses the copy's buffer, recording nothing.
pub(super) fn copy_arguments(
    core: &mut EncoderCore,
    buffer: &Buffer,
    offset: u64,
    bytes: u64,
) -> Result<Buffer, Error> {
    let copy = buffer
        .device()
        .own_buffer(COPY_BYTES, ResourceOptions::shared())?;
    let (from, at) = buffer.at(offset);
    let (into, start) = copy.at(0);
    let region = ash::vk::BufferCopy {
        src_offset: at,
        dst_offset: start,
        size: bytes,
    };
    let uses = [
        reads(buffer, offset..offset + bytes),
        writes(&copy, 0..bytes),
    ];
    core.barrier(Barrier::everything());
    core.command(TRANSFER, &uses, |device, command_buffer| {
        // SAFETY: the encoder's command buffer is recording; both buffers are live until it
        // completes (the encoder holds them), and the region lies in both: `arguments_offset`
        // held the arguments inside the bytes the buffer's `VkBuffer` holds
        // (`Buffer::copy_reach`), and the copy holds `COPY_BYTES`.
        unsafe { device.cmd_copy_buffer(command_buffer, from, into, &[region]) };
    });
    core.retain(buffer.clone());
    core.retain(copy.clone());
    Ok(copy)
}

/// What a host step dispatches: a dispatch the encoder knew, or one an indirect dispatch's
/// arguments state.
pub(super) enum HostDispatch {
    /// A direct dispatch, deferred only for its stage-in region (`StageInCopy`).
    Direct(Dispatch),
    /// An indirect dispatch's arguments, of `arguments`' form, in `copy`.
    Indirect { arguments: Arguments, copy: Buffer },
}

/// A dispatch left to its host step (the module docs): an indirect dispatch, or any dispatch
/// whose stage-in region is GPU memory (`setStageInRegionWithIndirectBuffer:`).
pub(super) struct ArgumentDispatch {
    /// The Metal call.
    pub(super) call: &'static str,
    pub(super) dispatch: HostDispatch,
    /// The copy of the `MTLStageInRegionIndirectArguments` the encoder's stage-in region was
    /// set to read at the call, which the host step reads in place of `stage_in_region`.
    pub(super) stage_in_copy: Option<Buffer>,
    /// The encoder's tables at the call.
    pub(super) pipeline: ComputePipelineState,
    pub(super) table: ArgumentTable,
    pub(super) threadgroup_lengths: BTreeMap<u32, u32>,
    pub(super) imageblock: ImageblockSize,
    pub(super) stage_in_region: Option<StageInRegion>,
    /// Whether the encoder runs its dispatches in encoded order.
    pub(super) ordered: bool,
    /// The check of the block of command predication it was encoded in, if any
    /// (`predication.rs`): where its counter skips, the host step records nothing.
    pub(super) gate: Option<HostGate>,
}

impl ArgumentDispatch {
    /// The tables the dispatch binds and runs with: the encoder's at the call, in the bound
    /// form, as a direct dispatch's, with `residency`, its command buffer's (`residency.rs`).
    fn tables<'a>(&'a self, residency: &'a Residency) -> DispatchTables<'a> {
        DispatchTables {
            pipeline: &self.pipeline,
            table: &self.table,
            threadgroup_lengths: &self.threadgroup_lengths,
            imageblock: self.imageblock,
            stage_in_region: self.stage_in_region,
            residency,
            forms: BufferForms::Bound,
        }
    }

    /// Settle at `commit`, on its thread, what the host step needs to translate nothing: the
    /// variants the dispatch may run on whatever its threadgroup size, and the device's neutrals
    /// their plans may bind. What cannot be settled now (a state the plan path refuses) is left
    /// to the step, which meets it as it is then.
    pub(super) fn settle(&self, device: &Device, residency: &Residency) -> Vec<Arc<Variant>> {
        prepare_variants(device, &self.tables(residency)).unwrap_or_default()
    }

    /// The host step's recording: the dispatch the arguments state now, as the encoder's
    /// dispatch of them would record it, or nothing where Metal runs nothing (the module docs).
    pub(super) fn record(&self, command_buffer: &CommandBuffer) -> Result<RecordedSegment, Error> {
        let mut core = EncoderCore::segment(command_buffer, self.ordered)?;
        // Inside a block of command predication that skips, it runs nothing.
        if self.gate.as_ref().is_some_and(HostGate::skips) {
            return core.end_segment();
        }
        // Past Metal's limit for the state nothing runs; within it, past the device's, the
        // dispatch's plan refuses it typed.
        let max_threads = self.pipeline.metal_max_total_threads_per_threadgroup();
        let dispatch = match &self.dispatch {
            HostDispatch::Direct(dispatch) => Some(*dispatch),
            HostDispatch::Indirect { arguments, copy } => {
                arguments.dispatch(words(copy, arguments.bytes())?, max_threads)
            }
        };
        if let Some(dispatch) = dispatch {
            let residency = command_buffer.residency()?.kernel();
            let mut tables = self.tables(&residency);
            if let Some(copy) = &self.stage_in_copy {
                let [x, y, z, width, height, depth] = words(copy, STAGE_IN_REGION_BYTES)?;
                tables.stage_in_region =
                    Some(StageInRegion::new([x, y, z], [width, height, depth]));
            }
            record_dispatch(&mut core, &tables, self.call, dispatch)?;
        }
        core.end_segment()
    }

    /// Announce on `device`, once, that the host reads such arguments.
    pub(super) fn degrade(&self, device: &Device) {
        let why = match self.dispatch {
            HostDispatch::Indirect { .. } => {
                "its arguments are GPU memory the work before it may write, which the host reads \
                 once that work completed"
            }
            HostDispatch::Direct(_) => {
                "its stage-in region is GPU memory the work before it may write \
                 (setStageInRegionWithIndirectBuffer:indirectBufferOffset:), which the host reads \
                 once that work completed"
            }
        };
        device.degrade(Degradation::HostReadDispatchArguments, || {
            (
                format!(
                    "`{}` of kernel `{}`",
                    self.call,
                    self.pipeline.function_name()
                ),
                why.to_owned(),
            )
        });
    }
}

/// The first `bytes` of `copy`, a copy of a dispatch's arguments, as words; zero past them.
fn words(copy: &Buffer, bytes: u64) -> Result<[u32; 6], Error> {
    let contents = copy
        .contents()
        .ok_or_else(|| internal("an indirect dispatch's copy without host bytes"))?;
    let mut read = [0u8; 24];
    let length = bytes.min(24) as usize;
    // SAFETY: the copy is a `Shared` buffer of `COPY_BYTES`, mapped; the GPU's write to it
    // completed before the host step (its batch ended with the barrier to the host), and nothing
    // writes it after.
    unsafe {
        std::ptr::copy_nonoverlapping(contents.as_ptr(), read.as_mut_ptr(), length);
    }
    Ok(std::array::from_fn(|i| {
        u32::from_le_bytes([
            read[4 * i],
            read[4 * i + 1],
            read[4 * i + 2],
            read[4 * i + 3],
        ])
    }))
}

/// An indirect dispatch on the GPU path (the module docs): Metal's arguments of `arguments`' form
/// at `offset` of `buffer`.
#[derive(Clone, Copy)]
pub(super) struct FlatCall<'a> {
    pub(super) buffer: &'a Buffer,
    pub(super) offset: u64,
    pub(super) arguments: Arguments,
}

impl FlatCall<'_> {
    /// The dispatch on a flat variant of `pipeline` (the module docs): `None` where the GPU path
    /// does not take it, on a device without device addresses or for a state of more than 1,024
    /// threads a threadgroup (`FlatIndirect::new`).
    pub(super) fn indirect(&self, pipeline: &ComputePipelineState) -> Option<FlatIndirect> {
        let arguments = self.buffer.device_address()?.checked_add(self.offset)?;
        let form = match self.arguments {
            Arguments::Threads => FlatArguments::Threads,
            Arguments::Threadgroups { threads_per_group } => FlatArguments::Threadgroups {
                threads_per_threadgroup: threads_per_group,
            },
        };
        // The variant runs threadgroups the device holds; the command runs nothing past Metal's
        // limit for the state alone, since the kernel does not observe Metal's threadgroups.
        FlatIndirect::new(
            arguments,
            form,
            flat_local(pipeline.pipeline_max_total_threads_per_threadgroup()),
            pipeline.metal_max_total_threads_per_threadgroup(),
        )
    }

    /// The dispatch on a folded variant of `pipeline` (the module docs): the threadgroups form's,
    /// in threadgroups of the call's size. `None` for the threads form, on a device without device
    /// addresses, and for a threadgroup with a zero axis or more threads than the state's
    /// `maxTotalThreadsPerThreadgroup`, which no variant of its size compiles: the encoder runs
    /// nothing past Metal's limit for the state, as Metal does, and refuses past this device's
    /// lower one as its gap.
    pub(super) fn folded(&self, pipeline: &ComputePipelineState) -> Option<FlatIndirect> {
        let Arguments::Threadgroups { threads_per_group } = self.arguments else {
            return None;
        };
        let arguments = self.buffer.device_address()?.checked_add(self.offset)?;
        let max_threads = pipeline.pipeline_max_total_threads_per_threadgroup();
        let threads: u64 = threads_per_group.iter().map(|&n| u64::from(n)).product();
        if threads == 0 || threads > u64::from(max_threads) {
            return None;
        }
        let form = FlatArguments::FoldedThreadgroups {
            threads_per_threadgroup: threads_per_group,
        };
        FlatIndirect::new(arguments, form, threads_per_group, max_threads)
    }

    /// The preprocess's read of Metal's arguments, by device address in the compute stage.
    pub(super) fn reads(&self) -> Use<ResourceKey> {
        let bytes = self.arguments.bytes();
        let (identity, range) = self.buffer.tracked(self.offset..self.offset + bytes);
        let read = Access::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
    }
}

/// The threadgroup size of the flat variant a state of `max_threads` threads a threadgroup at
/// most runs on: [`FLAT_THREADS`], or fewer where the state holds fewer. The kernel computes the
/// same in every partition of its grid (the module docs), so the size changes nothing it writes.
pub(super) fn flat_local(max_threads: u32) -> [u32; 3] {
    [max_threads.clamp(1, FLAT_THREADS), 1, 1]
}

/// The preprocess of one dispatch on the GPU path: its kernel, its scratch in the command
/// buffer's upload arena (`FlatIndirect::SCRATCH_BYTES`, the module's inputs written), and its
/// push constants (the scratch's device address and the command buffer's status words').
pub(super) struct FlatPreprocess {
    kernel: OwnedKernel,
    scratch: Upload,
    push: [u8; FlatDispatchModule::PUSH_BYTES as usize],
    /// The device address of the region record the variant's dispatch reads.
    pub(super) record: u64,
    /// It lies in a predicated loop's laying, which its check may submit again
    /// (`command_buffer/rerun.rs`): its gated recording zeroes the variant's command first.
    rerun: bool,
}

impl FlatPreprocess {
    /// The preprocess of `indirect` in `core`'s command buffer. Refused as the arena or the
    /// kernel refuses it, recording nothing.
    pub(super) fn new(
        core: &EncoderCore,
        indirect: &FlatIndirect,
    ) -> Result<FlatPreprocess, Error> {
        let device = core.device().clone();
        let kernel = device.owned_kernel(OwnedModule::FlatDispatch)?;
        let mut bytes = [0u8; FlatIndirect::SCRATCH_BYTES as usize];
        let inputs = FlatIndirect::INPUTS_OFFSET as usize;
        bytes[inputs..].copy_from_slice(&indirect.inputs());
        let (scratch, status, rerun) = {
            let mut state = core.command_buffer().state();
            let scratch = state
                .uploads
                .upload(&device, &bytes, FlatIndirect::SCRATCH_BYTES)?;
            let rerun = state.predication_mut().laying();
            (scratch, state.status_address(&device)?, rerun)
        };
        let (Some(address), Some(status)) = (scratch.device_address(), status) else {
            return Err(internal(
                "a flat dispatch's scratch without a device address",
            ));
        };
        Ok(FlatPreprocess {
            kernel,
            push: FlatDispatchModule::push_constants(address, status),
            record: address + FlatIndirect::RECORD_OFFSET,
            scratch,
            rerun,
        })
    }

    /// Record the preprocess, then the barrier from its writes to the variant's dispatch, which
    /// reads the command and the record. Inside a block of command predication it runs from its
    /// `gate` (`predication.rs`): where the block skips, it runs nothing, and the variant's command
    /// stays zero, as the arena wrote it; inside a loop's laying, which may run again after a run
    /// that wrote it, the recording zeroes it first.
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `raw`'s device in the recording state, outside a
    /// render pass, that only this thread records; it records the variant's dispatch next. The
    /// gate lies in the command buffer's arena.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        gate: Option<&Upload>,
    ) {
        // SAFETY: the caller's contract; the kernel is the device's, and the scratch lies in the
        // command buffer's arena, which lives until it completes.
        unsafe {
            if gate.is_some() && self.rerun {
                raw.cmd_fill_buffer(
                    command_buffer,
                    self.scratch.raw(),
                    self.scratch.offset() + FlatIndirect::COMMAND_OFFSET,
                    std::mem::size_of::<vk::DispatchIndirectCommand>() as u64,
                    0,
                );
                Barrier::after_fill().record(raw, command_buffer);
            }
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            raw.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.push,
            );
            match gate {
                Some(gate) => raw.cmd_dispatch_indirect(
                    command_buffer,
                    gate.raw(),
                    gate.offset() + PredicationStep::GATE_OUT_OFFSET,
                ),
                None => raw.cmd_dispatch(command_buffer, 1, 1, 1),
            }
            Barrier::between_dispatches().record(raw, command_buffer);
        }
    }

    /// Record the variant's dispatch, from the command the preprocess wrote.
    ///
    /// # Safety
    ///
    /// As [`FlatPreprocess::record`]'s, after it, with the variant's pipeline, sets and push
    /// constants recorded.
    pub(super) unsafe fn dispatch(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the command lies in the scratch, which has indirect
        // usage.
        unsafe {
            raw.cmd_dispatch_indirect(
                command_buffer,
                self.scratch.raw(),
                self.scratch.offset() + FlatIndirect::COMMAND_OFFSET,
            );
        }
    }
}
