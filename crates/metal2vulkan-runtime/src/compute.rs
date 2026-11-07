//! `ComputeCommandEncoder` (`MTLComputeCommandEncoder`).
//!
//! The encoder keeps Metal's argument tables (the pipeline state, buffers with their offsets,
//! threadgroup memory lengths) and the stage-in region, and records each dispatch when it is
//! encoded:
//!
//! 1. Metal's checks: a pipeline state is set (plain Metal faults at a dispatch without one).
//!    Where plain Metal gives a measured answer its API validation asserts at, the encoder gives
//!    it (an M4): a threadgroup whose size, the axes' wrapping 64-bit product, is 0 or above the
//!    state's `max_total_threads_per_threadgroup`, or whose axes modulo `2^32` hold other threads
//!    than that size, runs nothing, and a later dispatch runs; grid, threadgroup count and stage-in
//!    region axes past 32 bits are taken modulo `2^32`, and so is a per-bind attribute stride; a
//!    threadgroup memory length that is not a multiple of 16 is taken (the binder rounds every
//!    length up to 16, as Metal does). `set_imageblock_size` is checked where Metal checks it, at
//!    the call.
//! 2. The tables become a `bind::BindState`, brought in place from the one the encoder's last
//!    dispatch bound (`compute/live.rs`); the pipeline state's kept plans give the plan of its
//!    shape, on the variant the dispatch runs on (compiled when first needed), and
//!    `instantiate_dispatch` the command's regions, bytes, and push holes.
//! 3. [`binding`] gives the descriptors, uploads, and push bytes; the variant
//!    gives set 0 and one `VkPipeline` per region specialization, and the command buffer set 1,
//!    the heap, which its dispatches on the variant share ([`heap_sets`]).
//! 4. Only then is anything recorded, through the tracker's per-command entry
//!    ([`EncoderCore::command`]) with the buffers the dispatch reads and writes: a serial encoder
//!    is an ordered one, so a dispatch gets the barrier its uses need after the encoder's earlier
//!    dispatches (tracked buffers per resource and range, untracked memory as one class); a
//!    concurrent encoder leaves that order to the program's memory barriers. Then each region, with
//!    a barrier between regions (which only adds an ordering Metal's one dispatch allows). A
//!    refused dispatch records nothing and changes nothing.
//!
//! The command buffer retains what the dispatch uses (buffers, textures, sampler states and
//! their Vulkan samplers, pipeline objects, descriptor pool) until it completes, and what
//! `use_resource(s)` and `use_heap(s)` made resident (`residency`): every
//! later dispatch's resident set holds those buffers and heaps' ranges, and its handle tables and
//! heap those textures, each at the element of its resource ID (`HeapPlacement::ResourceId`),
//! with every live sampler state. `set_bytes` copies its bytes into the command buffer's upload
//! arena at the call (`upload.rs`), and every later dispatch binds that copy at the index until
//! the index is bound again.
//!
//! An indirect dispatch (`dispatch_threadgroups_indirect`, `dispatch_threads_indirect`) is
//! checked as Metal checks it when it is encoded. A kernel that does not observe its threadgroup
//! partition, on a device with device addresses, runs on its flat variant there, behind a
//! preprocess that reads the arguments on the device; any other kernel's threadgroup dispatch runs
//! there on its folded variant of the call's threadgroup size, behind the same preprocess; any
//! other dispatch's arguments are copied there, and a host step records it as the direct dispatch
//! they state once the work before it completed (`indirect_dispatch`).

use super::address_map::{self, SamplerMaps};
use super::argument_table::{check_index, ArgumentTable, Stamp};
use super::binding;
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::deferred::Deferred;
use super::degradation::{BoundFormats, CompiledPipeline, Degradation};
use super::device::Device;
use super::encoder::{EncoderCore, Replayed};
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::execution::{ExecutedPipeline, Execution};
use super::fetch_reach;
use super::function_table::VisibleFunctionTable;
use super::heap::Heap;
use super::heap_sets;
use super::indirect_command_buffer::{
    IndirectCommandBuffer, IndirectCommandCheck, IndirectCommandCheckNeeds,
};
use super::indirect_dispatch::{
    self, ArgumentDispatch, Arguments, FlatCall, FlatPreprocess, HostDispatch,
    STAGE_IN_REGION_BYTES,
};
use super::indirect_execute::{self, Execute, ExecutionRange};
use super::pending_dispatch::{PendingDispatch, Recording};
use super::pipeline::{
    flat_dispatch, BufferForms, ComputePipelineState, Planned, RawPipeline, RegionKey,
    RegionPipeline, Variant, VariantObjects,
};
use super::predication::{self, Captures, Event};
use super::pvrtc::degrade_unreadable;
use super::replay::{CallLog, ComputeCall, Replay, Resource, SamplersWithLodClamps};
use super::residency::Residency;
use super::resource::ResourceRef;
use super::sampler::SamplerState;
use super::shadow;
use super::texture::Texture;
use super::tracker::Barrier;
use super::used_resources::EncoderKind;
use super::values::{Region, Size};
use ash::vk;
use metal2vulkan::bind::{
    BindShape, BindState, BindingPlan, Dispatch, DispatchCommands, FlatIndirect, HeapPlacement,
    ImageblockSize, StageInRegion, UseTarget,
};
use metal2vulkan::error::{BindError, RuntimeDriverDefect};
use metal2vulkan::metal::{BarrierScope, DispatchType, RenderStages, ResourceUsage};
use metal2vulkan::pipeline::Stage;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock};

mod live;

use live::LiveDispatch;
pub(crate) use live::SpareLiveDispatches;

/// Entries of a compute encoder's threadgroup memory table.
pub(super) const THREADGROUP_SLOTS: u32 = 31;

/// The most cells `setImageblockWidth:height:` takes (Metal's API validation on an M3 and an M4).
const MAX_IMAGEBLOCK_CELLS: u64 = 1024;

/// The longest side plain Metal's answer to `setImageblockWidth:height:` is measured at: Metal may
/// keep fewer bits of a longer one.
const MAX_MEASURED_IMAGEBLOCK_SIDE: u64 = 65536;

/// The imageblock size Metal's API validation takes, the same on an M3 and an M4 and whatever the
/// pipeline: a width and a height that are powers of two (0 excluded), of at most
/// [`MAX_IMAGEBLOCK_CELLS`] cells; the rest refused. An indirect command's size is held to it,
/// plain Metal's answer there being unmeasured.
pub(super) fn validated_imageblock_size(
    width: u64,
    height: u64,
) -> Result<ImageblockSize, Problem> {
    let refused = Problem::ImageblockSize { width, height };
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(refused);
    }
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(w), Ok(h)) if width.saturating_mul(height) <= MAX_IMAGEBLOCK_CELLS => {
            Ok(ImageblockSize::new(w, h))
        }
        _ => Err(refused),
    }
}

/// The imageblock size `setImageblockWidth:height:` sets on a compute encoder, as plain Metal sets
/// it where its API validation asserts (the M3 Max and the M4 alike): a side of 0 makes it 1 by 1;
/// else each side is rounded up to a power of two, and a size of more than [`MAX_IMAGEBLOCK_CELLS`]
/// cells in one row or one column is 1 by 1. Refused, as validation refuses it: more than
/// [`MAX_IMAGEBLOCK_CELLS`] cells over more rows and columns (Metal reads 1024 by 1 there, its
/// cells reading 0), and a side past [`MAX_MEASURED_IMAGEBLOCK_SIDE`].
pub(super) fn imageblock_size(width: u64, height: u64) -> Result<ImageblockSize, Problem> {
    if width == 0 || height == 0 {
        return Ok(ImageblockSize::new(1, 1));
    }
    if width.max(height) > MAX_MEASURED_IMAGEBLOCK_SIDE {
        return Err(Problem::ImageblockSize { width, height });
    }
    let (w, h) = (width.next_power_of_two(), height.next_power_of_two());
    if w * h > MAX_IMAGEBLOCK_CELLS && (w == 1 || h == 1) {
        return Ok(ImageblockSize::new(1, 1));
    }
    validated_imageblock_size(w, h).map_err(|_| Problem::ImageblockSize { width, height })
}

/// A compute encoder (`MTLComputeCommandEncoder`): `Send`, not `Sync`; `end_encoding` consumes
/// it. Each object a call binds is held until its command buffer completes (or, never committed,
/// is released), as Metal holds what an encoder was handed: one a later call replaced or unbound
/// before any dispatch used it, and one bound after the last dispatch, too.
pub struct ComputeCommandEncoder {
    core: EncoderCore,
    dispatch_type: DispatchType,
    pipeline: Option<ComputePipelineState>,
    table: ArgumentTable,
    threadgroup_lengths: BTreeMap<u32, u32>,
    /// `setImageblockWidth:height:`'s size, 1 by 1 until set: what a kernel of Metal's reads
    /// when no size was set.
    imageblock: ImageblockSize,
    /// `setStageInRegion:`'s region, which a kernel with `[[stage_in]]` reads.
    stage_in_region: Option<StageInRegion>,
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`'s buffer and offset, in place
    /// of `stage_in_region` until the next `setStageInRegion:`.
    indirect_stage_in: Option<(Buffer, u64)>,
    residency: Residency,
    /// The bodies of the predicated loops it opened and has not closed (`predication.rs`).
    captures: Captures,
    /// Its calls so far, where the device had a buffer over host pages as it opened
    /// (`replay.rs`): `None` where it logs nothing, or no longer.
    log: Option<CallLog<ComputeCall>>,
    /// Its dispatches encoded so far: the next one's order among them, which one whose
    /// pipelines are being made keeps for its failure (`FailedPipeline::draw`).
    dispatches: usize,
    /// Its bind state as its last dispatch that settled bound it, with what that one settled
    /// (`compute/live.rs`, [`DispatchMemo`]).
    live: LiveDispatch,
    /// The residency's generation as a command last noted the heaps of what it made resident
    /// (`command_encoded`): no command notes them again before it changes.
    heaps_noted: Option<u64>,
}

/// The state of a compute encoder a command ran with, which a predicated loop's copies replay it
/// with (`predication.rs`): what its setters set. What it made resident is not part of it: a
/// copy runs with what the encoder made resident by the loop's end.
#[derive(Clone)]
pub(super) struct EncoderState {
    pipeline: Option<ComputePipelineState>,
    table: ArgumentTable,
    threadgroup_lengths: BTreeMap<u32, u32>,
    imageblock: ImageblockSize,
    stage_in_region: Option<StageInRegion>,
    indirect_stage_in: Option<(Buffer, u64)>,
}

impl std::fmt::Debug for ComputeCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeCommandEncoder")
            .field("dispatch_type", &self.dispatch_type)
            .finish()
    }
}

impl CommandBuffer {
    /// `computeCommandEncoder`: a serial compute encoder. Refused while another encoder of this
    /// command buffer is open, after one was abandoned, and after commit.
    pub fn compute_command_encoder(&self) -> Result<ComputeCommandEncoder, Error> {
        self.open_compute("compute_command_encoder", DispatchType::Serial)
    }

    /// `computeCommandEncoderWithDispatchType:`: a serial encoder orders its dispatches as
    /// Metal's does (the tracker's order inside an encoder); a concurrent one leaves their order
    /// to the program's memory barriers. Refused as [`CommandBuffer::compute_command_encoder`] is.
    pub fn compute_command_encoder_with_dispatch_type(
        &self,
        dispatch_type: DispatchType,
    ) -> Result<ComputeCommandEncoder, Error> {
        self.open_compute("compute_command_encoder_with_dispatch_type", dispatch_type)
    }

    /// `computeCommandEncoderWithDispatchType:`, or `computeCommandEncoderWithDescriptor:` of a
    /// pass descriptor's `dispatchType`, with the raw `MTLDispatchType` value `raw`, any of its 64
    /// bits: Metal makes a concurrent encoder of 1 and a serial one of every other value (2, 7,
    /// `u32::MAX`, `1 << 32`, `(1 << 32) | 1`, `u64::MAX`, ...), whose `dispatchType` answers
    /// serial and which skips a nil resource of `useResources:count:usage:` as a serial encoder
    /// does, where a concurrent one ends the process (the M3 Max under macOS 27.0.1); so does
    /// the runtime. Refused as [`CommandBuffer::compute_command_encoder`] is.
    pub fn compute_command_encoder_with_raw_dispatch_type(
        &self,
        raw: u64,
    ) -> Result<ComputeCommandEncoder, Error> {
        let dispatch_type = match raw == u64::from(DispatchType::Concurrent.as_raw()) {
            true => DispatchType::Concurrent,
            false => DispatchType::Serial,
        };
        self.open_compute(
            "compute_command_encoder_with_raw_dispatch_type",
            dispatch_type,
        )
    }

    fn open_compute(
        &self,
        call: &'static str,
        dispatch_type: DispatchType,
    ) -> Result<ComputeCommandEncoder, Error> {
        // Metal's process ends here under some protection options (`command_buffer/protection.rs`).
        self.refuse_protected_encoder(call)?;
        let core = EncoderCore::open(
            self,
            call,
            EncoderKind::Compute,
            dispatch_type == DispatchType::Serial,
        )?;
        // It joins its command buffer's unit of command predication, carrying on the loops open
        // across its start, or starts the next.
        let carried = self.state().enter_predication(dispatch_type);
        let mut core = core;
        // The laying of a loop open across its start goes on in a command buffer of its own, which
        // the loop's check may submit again (`predication.rs`).
        if !carried.is_empty() && self.state().predication.laying() {
            let raw = core.cut_reusable()?;
            self.state().predication.laid_into(raw);
        }
        Ok(ComputeCommandEncoder {
            core,
            dispatch_type,
            pipeline: None,
            table: ArgumentTable::compute(),
            threadgroup_lengths: BTreeMap::new(),
            imageblock: ImageblockSize::new(1, 1),
            stage_in_region: None,
            indirect_stage_in: None,
            residency: self.inherited_residency(false),
            captures: Captures::carried(carried),
            log: (self.device().shared().rebinds.any())
                .then(|| self.device().shared().scheduler.spare_logs().compute()),
            dispatches: 0,
            // The bind state the device's last ended compute encoder left (`compute/live.rs`).
            live: self.device().shared().live_dispatches.take(),
            heaps_noted: None,
        })
    }
}

impl ComputeCommandEncoder {
    /// An encoder of `dispatch_type` that encodes an ended one of `command_buffer` again,
    /// starting with `residency` as it started (`replay.rs`): it logs nothing.
    pub(super) fn replaying(
        command_buffer: &CommandBuffer,
        dispatch_type: DispatchType,
        residency: Residency,
    ) -> Result<ComputeCommandEncoder, Error> {
        let core = EncoderCore::replaying(command_buffer, dispatch_type == DispatchType::Serial)?;
        Ok(ComputeCommandEncoder {
            core,
            dispatch_type,
            pipeline: None,
            table: ArgumentTable::compute(),
            threadgroup_lengths: BTreeMap::new(),
            imageblock: ImageblockSize::new(1, 1),
            stage_in_region: None,
            indirect_stage_in: None,
            residency,
            captures: Captures::default(),
            log: None,
            dispatches: 0,
            live: command_buffer.device().shared().live_dispatches.take(),
            heaps_noted: None,
        })
    }

    /// End an encoder made by [`replaying`](Self::replaying): what it recorded, and what it made
    /// resident.
    pub(super) fn finish_replay(mut self) -> Result<(Replayed, Option<Residency>), Error> {
        // Its unit of command predication takes what it encoded again, as at its end
        // (`predication.rs`).
        self.end_predication()?;
        self.table.let_go_all(&mut self.core);
        // Its bind state goes on to the device's next compute encoder (`compute/live.rs`).
        self.core.device().shared().live_dispatches.give(self.live);
        let replayed = self.core.finish_replay()?;
        Ok((replayed, Some(self.residency)))
    }

    /// Log the call `call` makes of the log, before it runs, where the encoder logs
    /// (`replay.rs`): its place in the log.
    pub(super) fn log(
        &mut self,
        call: impl FnOnce(&mut CallLog<ComputeCall>) -> ComputeCall,
    ) -> Option<usize> {
        let log = self.log.as_mut()?;
        let call = call(log);
        let extra = call.extra();
        Some(log.push(call, extra))
    }

    /// Run `run`, a public call, logged first as `call` makes it of the log
    /// ([`log`](Self::log)): where it is refused, its log notes it so (`CallLog::refuse`), and an
    /// encoding again refuses it again and goes on (`replay.rs`).
    pub(super) fn logged<T>(
        &mut self,
        call: impl FnOnce(&mut CallLog<ComputeCall>) -> ComputeCall,
        run: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let at = self.log(call);
        let ran = run(self);
        if let (Err(_), Some(at), Some(log)) = (&ran, at, &mut self.log) {
            log.refuse(at);
        }
        ran
    }

    /// Note in the log the answer `result` of the call logged at `at` (`call` of it), or its
    /// refusal: encoded again, the call must answer so (`replay.rs`).
    pub(super) fn answered(
        &mut self,
        at: Option<usize>,
        result: &Result<bool, Error>,
        call: fn(bool) -> ComputeCall,
    ) {
        let (Some(at), Some(log)) = (at, &mut self.log) else {
            return;
        };
        match result {
            Ok(answer) => log.calls[at] = call(*answer),
            Err(_) => log.refuse(at),
        }
    }

    /// The encoder logs nothing more, and will not be encoded again: it records a loop of
    /// command predication (`predication.rs`), whose laying its check submits again and whose
    /// copies encode its own calls again.
    pub(super) fn stop_logging(&mut self) {
        self.log = None;
    }
}

/// Announce how a recorded dispatch of `kernel` degraded: run as `regions` Vulkan dispatches, or
/// on the kernel's emulated-simdgroups variant (`emulated`).
fn degrade_dispatch(
    device: &Device,
    kernel: &str,
    dispatch: &Dispatch,
    regions: usize,
    emulated: bool,
) {
    let object = || format!("a dispatch of kernel `{kernel}`");
    let threads_per_group = dispatch.threads_per_group();
    if regions > 1 {
        device.degrade(Degradation::SplitDispatch, || {
            let reason = match *dispatch {
                Dispatch::Threads { threads, .. }
                    if threads
                        .iter()
                        .zip(threads_per_group)
                        .any(|(&n, group)| !n.is_multiple_of(group)) =>
                {
                    format!(
                        "its grid of {threads:?} threads is not a whole number of threadgroups \
                         of {threads_per_group:?}"
                    )
                }
                _ => "a group count is past the device's maxComputeWorkGroupCount".to_owned(),
            };
            (object(), reason)
        });
    }
    if emulated {
        device.degrade(Degradation::EmulatedSimdgroups, || {
            let reason = format!(
                "its threadgroups of {threads_per_group:?} threads are not whole subgroups of \
                 the size the kernel's simdgroups run on"
            );
            (object(), reason)
        });
    }
}

impl ComputeCommandEncoder {
    /// `dispatchType`.
    pub fn dispatch_type(&self) -> DispatchType {
        self.dispatch_type
    }

    /// `setComputePipelineState:`. A state of another device is refused.
    pub fn set_compute_pipeline_state(
        &mut self,
        state: &ComputePipelineState,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::PipelineState(state.clone()),
            |this| {
                this.core
                    .check_device("set_compute_pipeline_state", state.device())?;
                this.pipeline = Some(state.clone());
                (this.core.routes).state_set(state.indirect_command_writes(), || {
                    vec![state.function_name().to_owned()]
                });
                Ok(())
            },
        )
    }

    /// `setBuffer:offset:atIndex:` (`None` unbinds the index). A buffer bound past its length is
    /// held, as plain Metal binds it, and a dispatch whose kernel reads the slot is refused
    /// (`Invalid(OffsetOutOfRange)`: Metal's read is past the buffer). Refused: an index past the
    /// table (plain Metal writes past its own table: a texture at 128 faulted), a buffer of another
    /// device.
    pub fn set_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Buffer(buffer.cloned(), offset, index),
            |this| {
                this.table
                    .set_buffer(&mut this.core, "set_buffer", buffer, offset, index)
            },
        )
    }

    /// `setBuffers:offsets:withRange:`: `buffers[i]` at offset `offsets[i]` at index
    /// `first + i`. Refused as [`ComputeCommandEncoder::set_buffer`] is, and when the slices'
    /// lengths differ; a refusal binds none of them.
    pub fn set_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.logged(
            |log| ComputeCall::Buffers(log.buffer_span(buffers), log.word_span(offsets), first),
            |this| {
                this.table
                    .set_buffers(&mut this.core, "set_buffers", buffers, offsets, first)
            },
        )
    }

    /// `setBufferOffset:atIndex:`, which clears the index's attribute stride
    /// ([`set_buffer_offset_with_stride`](Self::set_buffer_offset_with_stride)). An offset past
    /// the buffer's length is held as [`set_buffer`](Self::set_buffer) holds one; at an index
    /// with no buffer it changes nothing, as plain Metal's. At an index holding a visible-function
    /// table, the kernel then reads the null table there (size 0, every entry null) until a table
    /// or buffer is bound at the index again, as plain Metal reads it (an M4). At an index holding
    /// inline bytes ([`set_bytes`](Self::set_bytes)), they then read as zeros at any offset until
    /// the index is bound again, as plain Metal reads them (an M4; its debug layer asserts there is
    /// no buffer). Refused: an index past the table.
    pub fn set_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::BufferOffset(offset, index),
            |this| {
                this.table
                    .set_buffer_offset(&mut this.core, "set_buffer_offset", offset, index)
            },
        )
    }

    /// `setBuffer:offset:attributeStride:atIndex:`: as [`set_buffer`](Self::set_buffer), the
    /// buffer read at `stride` by a kernel's stage-in layout of `MTLBufferLayoutStrideDynamic`.
    /// Only such a layout reads a stride, and one bound without a stride (`set_buffer`) reads at
    /// stride 0; a layout of a fixed stride reads its own whatever the bind's (as an M4 reads
    /// them). A stride past 32 bits is taken modulo `2^32`, as plain Metal reads it and its
    /// validation takes it (an M4 and an M3 Max), and `MTLAttributeStrideStatic` (`u64::MAX`) with
    /// it: such a layout reads the element at the bind offset for the first element and zero for
    /// every element past it, which lies `2^32 - 1` bytes on, past the buffer, as plain Metal on an
    /// M3 Max under macOS 27.0.1 answered it. Refused as `set_buffer` is.
    pub fn set_buffer_with_stride(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::BufferWithStride(buffer.cloned(), offset, stride, index),
            |this| {
                this.table.set_buffer_with_stride(
                    &mut this.core,
                    "set_buffer_with_stride",
                    buffer,
                    offset,
                    stride,
                    index,
                )
            },
        )
    }

    /// `setBuffers:offsets:attributeStrides:withRange:`: `buffers[i]` at offset `offsets[i]` and
    /// stride `strides[i]` at index `first + i`, each as
    /// [`set_buffer_with_stride`](Self::set_buffer_with_stride). Refused as that is, and when the
    /// slices' lengths differ; a refusal binds none of them.
    pub fn set_buffers_with_strides(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        strides: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.logged(
            |log| {
                let (buffers, offsets) = (log.buffer_span(buffers), log.word_span(offsets));
                ComputeCall::BuffersWithStrides(buffers, offsets, log.word_span(strides), first)
            },
            |this| {
                this.table.set_buffers_with_strides(
                    &mut this.core,
                    "set_buffers_with_strides",
                    buffers,
                    offsets,
                    Some(strides),
                    first,
                )
            },
        )
    }

    /// `setBufferOffset:attributeStride:atIndex:`: as [`set_buffer_offset`](Self::set_buffer_offset),
    /// with the index's stride set as [`set_buffer_with_stride`](Self::set_buffer_with_stride)
    /// sets it, `MTLAttributeStrideStatic` included. `set_buffer_offset` clears the stride.
    /// Refused as both are.
    pub fn set_buffer_offset_with_stride(
        &mut self,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::BufferOffsetWithStride(offset, stride, index),
            |this| {
                this.table.set_buffer_offset_with_stride(
                    &mut this.core,
                    "set_buffer_offset_with_stride",
                    offset,
                    stride,
                    index,
                )
            },
        )
    }

    /// `setBytes:length:atIndex:`: `bytes`, copied now, at buffer index `index` in place of
    /// anything bound there, so a later write of the caller's memory reaches no dispatch, as
    /// Metal copies them at the call. Every later dispatch reads the copy until the index is
    /// bound again; a kernel's stage-in layout of `MTLBufferLayoutStrideDynamic` reads it at
    /// stride 0, as a buffer bound without a stride. A slot the kernel reaches through a pointer
    /// loaded from memory (one of a device-address form) cannot read inline bytes:
    /// `Bind(MissingDeviceAddress)`, at the dispatch. Refused: an index past the table, and more
    /// than 4096 bytes, Metal's limit for inline bytes (`Invalid(InlineBytesLength)`).
    pub fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.logged(
            |log| ComputeCall::Bytes(log.byte_span(bytes), index),
            |this| {
                this.table
                    .set_bytes(&mut this.core, "set_bytes", bytes, None, index)
            },
        )
    }

    /// `setBytes:length:attributeStride:atIndex:`: as [`set_bytes`](Self::set_bytes), the
    /// bytes read at `stride`, its low 32 bits, by a kernel's stage-in layout of
    /// `MTLBufferLayoutStrideDynamic`, as [`set_buffer_with_stride`](Self::set_buffer_with_stride)
    /// reads a buffer: at `MTLAttributeStrideStatic` (`u64::MAX`) such a layout reads element 0 of
    /// the bytes and zero past it, as an M4 and an M3 Max read them. Refused as `set_bytes` is.
    pub fn set_bytes_with_stride(
        &mut self,
        bytes: &[u8],
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |log| ComputeCall::BytesWithStride(log.byte_span(bytes), stride, index),
            |this| {
                this.table.set_bytes(
                    &mut this.core,
                    "set_bytes_with_stride",
                    bytes,
                    Some(stride),
                    index,
                )
            },
        )
    }

    /// `setVisibleFunctionTable:atBufferIndex:` (`None` unbinds the index): the table at buffer
    /// index `index`, in place of any buffer there. Refused: an index past the buffer table, a
    /// table of another device. A dispatch whose function reads a buffer at the index, or a table
    /// where a buffer is bound, is refused (`Invalid(FunctionTable)`), as Metal's validation
    /// stops it.
    pub fn set_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::FunctionTable(table.cloned(), index),
            |this| {
                this.table.set_function_tables(
                    &mut this.core,
                    "set_visible_function_table",
                    &[table],
                    index,
                )
            },
        )
    }

    /// `setVisibleFunctionTables:withBufferRange:`: `tables[i]` at buffer index `first + i`.
    /// Refused as [`ComputeCommandEncoder::set_visible_function_table`] is; a refusal binds none
    /// of them.
    pub fn set_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                ComputeCall::FunctionTables(
                    tables.iter().map(|table| table.cloned()).collect(),
                    first,
                )
            },
            |this| {
                this.table.set_function_tables(
                    &mut this.core,
                    "set_visible_function_tables",
                    tables,
                    first,
                )
            },
        )
    }

    /// `setTexture:atIndex:` (`None` unbinds the index). Refused: an index past the table, a
    /// texture of another device. What a bound texture's usage allows is checked at the
    /// dispatch, against the function's use of the slot.
    pub fn set_texture(&mut self, texture: Option<&Texture>, index: u32) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Texture(texture.cloned(), index),
            |this| {
                this.table
                    .set_textures(&mut this.core, "set_texture", &[texture], index)
            },
        )
    }

    /// `setTextures:withRange:`: `textures[i]` at index `first + i`. Refused as
    /// [`ComputeCommandEncoder::set_texture`] is; a refusal binds none of them.
    pub fn set_textures(&mut self, textures: &[Option<&Texture>], first: u32) -> Result<(), Error> {
        self.logged(
            |log| ComputeCall::Textures(log.texture_span(textures), first),
            |this| {
                this.table
                    .set_textures(&mut this.core, "set_textures", textures, first)
            },
        )
    }

    /// `setSamplerState:atIndex:` (`None` unbinds the index). Refused: an index past the table, a
    /// state of another device.
    pub fn set_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Sampler(sampler.cloned(), index),
            |this| {
                this.table.set_samplers(
                    &mut this.core,
                    "set_sampler_state",
                    &[sampler],
                    None,
                    index,
                )
            },
        )
    }

    /// `setSamplerStates:withRange:`: `samplers[i]` at index `first + i`. Refused as
    /// [`ComputeCommandEncoder::set_sampler_state`] is; a refusal binds none of them.
    pub fn set_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        first: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                ComputeCall::Samplers(
                    samplers.iter().map(|sampler| sampler.cloned()).collect(),
                    first,
                )
            },
            |this| {
                this.table
                    .set_samplers(&mut this.core, "set_sampler_states", samplers, None, first)
            },
        )
    }

    /// `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`: the state with its LOD clamps
    /// replaced by these, at `index`. Metal takes any clamps: inverted ones read the upper clamp's
    /// level, as Metal's do; a sampler whose clamps are NaN is refused at the dispatch that samples
    /// through it (what Metal samples then is not measured). Refused as
    /// [`ComputeCommandEncoder::set_sampler_state`] is.
    pub fn set_sampler_state_with_lod_clamps(
        &mut self,
        sampler: Option<&SamplerState>,
        lod_min_clamp: f32,
        lod_max_clamp: f32,
        index: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                ComputeCall::SamplerWithLodClamps(
                    sampler.cloned(),
                    lod_min_clamp,
                    lod_max_clamp,
                    index,
                )
            },
            |this| {
                this.table.set_samplers(
                    &mut this.core,
                    "set_sampler_state_with_lod_clamps",
                    &[sampler],
                    Some((&[lod_min_clamp], &[lod_max_clamp])),
                    index,
                )
            },
        )
    }

    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:`: `samplers[i]` with LOD clamps
    /// `lod_min_clamps[i]` and `lod_max_clamps[i]` at index `first + i`. Refused as
    /// [`ComputeCommandEncoder::set_sampler_state_with_lod_clamps`] is, and when the slices'
    /// lengths differ; a refusal binds none of them.
    pub fn set_sampler_states_with_lod_clamps(
        &mut self,
        samplers: &[Option<&SamplerState>],
        lod_min_clamps: &[f32],
        lod_max_clamps: &[f32],
        first: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                ComputeCall::SamplersWithLodClamps(Box::new(SamplersWithLodClamps {
                    samplers: samplers.iter().map(|sampler| sampler.cloned()).collect(),
                    mins: lod_min_clamps.into(),
                    maxes: lod_max_clamps.into(),
                    first,
                }))
            },
            |this| {
                this.table.set_samplers(
                    &mut this.core,
                    "set_sampler_states_with_lod_clamps",
                    samplers,
                    Some((lod_min_clamps, lod_max_clamps)),
                    first,
                )
            },
        )
    }

    /// `setThreadgroupMemoryLength:atIndex:`. A length that is not a multiple of 16 bytes is
    /// taken, rounded up to 16 at the dispatch, as plain Metal takes it (lengths of 4, 8 and 20
    /// held their threads' words on an M4); Metal's API validation asserts. Refused: an index past
    /// the table. The device's limit is checked at the dispatch.
    pub fn set_threadgroup_memory_length(&mut self, length: u32, index: u32) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::ThreadgroupMemoryLength(length, index),
            |this| {
                const CALL: &str = "set_threadgroup_memory_length";
                check_index(CALL, index, THREADGROUP_SLOTS)?;
                this.threadgroup_lengths.insert(index, length);
                Ok(())
            },
        )
    }

    /// `setImageblockWidth:height:`: the cells of each threadgroup's explicit imageblock in the
    /// encoder's later dispatches, whichever pipeline state they use. Until it is called a
    /// dispatch's imageblock is 1 by 1, as Metal's is. Its bytes count against the device's
    /// threadgroup memory beside the threadgroup memory lengths, checked at the dispatch.
    ///
    /// A side of 0 sets 1 by 1, a side that is not a power of two is rounded up to one, and more
    /// than 1024 cells in one row or column set 1 by 1, as plain Metal sets them (Metal's API
    /// validation stops them). Refused (`Invalid`), as validation refuses it, changing nothing:
    /// more than 1024 cells over more rows and columns, which Metal runs as 1024 by 1 with its
    /// cells reading 0, and a side past 65536 (unmeasured).
    pub fn set_imageblock_size(&mut self, width: u64, height: u64) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::ImageblockSize(width, height),
            |this| {
                this.imageblock = imageblock_size(width, height)
                    .map_err(|p| invalid("set_imageblock_size", p))?;
                Ok(())
            },
        )
    }

    /// `setStageInRegion:`: the region whose origin each thread's position in the grid is added
    /// to, to find the element its `[[stage_in]]` attributes are fetched from, and which
    /// `[[grid_origin]]` and `[[grid_size]]` read. It holds for the encoder's later dispatches. A
    /// dispatch of a kernel that reads it with none set is refused (`Bind(MissingStageInRegion)`).
    /// An origin or size axis past 32 bits is taken modulo `2^32`, as Metal takes it with or
    /// without its API validation (an origin of `2^32 + 1` read 1 and fetched element 1 on an M4).
    pub fn set_stage_in_region(&mut self, region: Region) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::StageInRegion(region),
            |this| {
                let Region { origin, size } = region;
                let origin = Size::new(origin.x, origin.y, origin.z).truncated();
                this.stage_in_region = Some(StageInRegion::new(origin, size.truncated()));
                this.indirect_stage_in = None;
                Ok(())
            },
        )
    }

    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`: the encoder's later
    /// dispatches read their stage-in region from the `MTLStageInRegionIndirectArguments` at
    /// `indirect_buffer_offset` (the origin's three `u32`s, then the size's) as each runs, until a
    /// `set_stage_in_region`. A host step reads them once the work before the dispatch
    /// completed, and records the dispatch with the region they state
    /// (`indirect_dispatch`), as it does an indirect dispatch's
    /// arguments; a region the binder refuses there fails the command buffer, as an indirect
    /// command's dispatch does.
    ///
    /// An offset that is not a multiple of 4 reads at the multiple below, as plain Metal reads
    /// it (Metal's API validation stops it). Arguments past the buffer's length are read out of
    /// its allocation, as an indirect dispatch's are measured to be
    /// (`indirect_dispatch::arguments_offset`, which carries the measurement; the region's own
    /// read is unmeasured, and follows it as the same read of a `u32` at the same kind of
    /// address). Refused, changing nothing (`Invalid`): arguments past the bytes the runtime
    /// holds there (`Buffer::copy_reach`), and a buffer of another device.
    pub fn set_stage_in_region_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::StageInRegionIndirect(indirect_buffer.clone(), indirect_buffer_offset),
            |this| {
                const CALL: &str = "set_stage_in_region_indirect";
                this.core.check_device(CALL, indirect_buffer.device())?;
                let offset = indirect_dispatch::arguments_offset(
                    CALL,
                    (indirect_buffer.length(), indirect_buffer.copy_reach()),
                    indirect_buffer_offset,
                    STAGE_IN_REGION_BYTES,
                )?;
                this.indirect_stage_in = Some((indirect_buffer.clone(), offset));
                this.stage_in_region = None;
                Ok(())
            },
        )
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:`. A threadgroup Metal runs nothing for (its
    /// size 0 or above the state's maximum, the module docs) records nothing; a count axis past
    /// 32 bits is taken modulo `2^32`, and one that is 0 runs nothing. Refused: no pipeline state
    /// set (plain Metal faults).
    pub fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Threadgroups(groups, threads),
            |this| {
                const CALL: &str = "dispatch_threadgroups";
                if this.fails_on_memoryless() {
                    return Ok(());
                }
                let Some(threads_per_group) = this.threadgroup(CALL, threads)? else {
                    return Ok(());
                };
                this.dispatch(
                    CALL,
                    Dispatch::Threadgroups {
                        groups: groups.truncated(),
                        threads_per_group,
                    },
                )?;
                this.worked();
                this.capture(|| Event::Threadgroups(groups, threads));
                Ok(())
            },
        )
    }

    /// `dispatchThreads:threadsPerThreadgroup:`: a grid of `threads`, its edges in partial
    /// threadgroups. Its threadgroup and grid are taken as
    /// [`dispatch_threadgroups`](Self::dispatch_threadgroups) takes its threadgroup and counts.
    pub fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Threads(threads, per_group),
            |this| {
                const CALL: &str = "dispatch_threads";
                if this.fails_on_memoryless() {
                    return Ok(());
                }
                let Some(threads_per_group) = this.threadgroup(CALL, per_group)? else {
                    return Ok(());
                };
                this.dispatch(
                    CALL,
                    Dispatch::Threads {
                        threads: threads.truncated(),
                        threads_per_group,
                    },
                )?;
                this.worked();
                this.capture(|| Event::Threads(threads, per_group));
                Ok(())
            },
        )
    }

    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`:
    /// threadgroups of `threads` threads, as many on each axis as the
    /// `MTLDispatchThreadgroupsIndirectArguments` at `indirect_buffer_offset` state when the
    /// dispatch runs. A host step reads them once the work before the dispatch completed, and
    /// records the dispatch as `dispatch_threadgroups` of the encoder's tables now
    /// (`indirect_dispatch`); a zero count runs nothing.
    ///
    /// An offset that is not a multiple of 4 reads at the multiple below, as plain Metal reads
    /// it. Refused, recording nothing, as `dispatch_threadgroups` refuses `threads`, and as Metal's
    /// API validation refuses arguments past the buffer's length from there (`Invalid`); and for
    /// a buffer of another device.
    pub fn dispatch_threadgroups_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
        threads: Size,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                ComputeCall::ThreadgroupsIndirect(
                    indirect_buffer.clone(),
                    indirect_buffer_offset,
                    threads,
                )
            },
            |this| {
                const CALL: &str = "dispatch_threadgroups_indirect";
                let Some(threads_per_group) = this.threadgroup(CALL, threads)? else {
                    // A threadgroup that runs nothing fails as a direct dispatch's does.
                    this.fails_on_memoryless();
                    return Ok(());
                };
                let arguments = Arguments::Threadgroups { threads_per_group };
                this.dispatch_indirect(CALL, indirect_buffer, indirect_buffer_offset, arguments)?;
                this.worked();
                this.capture(|| {
                    Event::ThreadgroupsIndirect(
                        indirect_buffer.clone(),
                        indirect_buffer_offset,
                        threads,
                    )
                });
                Ok(())
            },
        )
    }

    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` (the encoder's SPI; Metal 4's
    /// `dispatchThreadsWithIndirectBuffer:`): a grid of threads, its edges in partial
    /// threadgroups, as the `MTLDispatchThreadsIndirectArguments` at `indirect_buffer_offset`
    /// state when the dispatch runs (`threadsPerGrid`, then `threadsPerThreadgroup`). A host
    /// step reads them once the work before the dispatch completed, and records the dispatch as
    /// `dispatch_threads` of the encoder's tables now
    /// (`indirect_dispatch`). A zero grid axis, and a threadgroup with a zero axis or more threads
    /// than Metal's limit for the state, run nothing, as Metal's do; one past only the state's
    /// `max_total_threads_per_threadgroup` (this device's lower limit) is refused typed as the
    /// device's gap.
    ///
    /// An offset that is not a multiple of 4 reads at the multiple below, as plain Metal reads
    /// it. Refused, recording nothing, with no pipeline state set, and as Metal's API validation
    /// refuses arguments past the buffer's length from there (`Invalid`); and for a buffer of
    /// another device.
    pub fn dispatch_threads_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::ThreadsIndirect(indirect_buffer.clone(), indirect_buffer_offset),
            |this| {
                const CALL: &str = "dispatch_threads_indirect";
                let arguments = Arguments::Threads;
                this.dispatch_indirect(CALL, indirect_buffer, indirect_buffer_offset, arguments)?;
                this.worked();
                this.capture(|| {
                    Event::ThreadsIndirect(indirect_buffer.clone(), indirect_buffer_offset)
                });
                Ok(())
            },
        )
    }

    /// Check an indirect dispatch of `arguments` at `offset` of `buffer`, and record it on its
    /// kernel's flat variant, or else copy them where it is encoded and leave the dispatch to its
    /// host step (`indirect_dispatch.rs`); with every texture the encoder's heaps hold resident.
    fn dispatch_indirect(
        &mut self,
        call: &'static str,
        buffer: &Buffer,
        offset: u64,
        arguments: Arguments,
    ) -> Result<(), Error> {
        self.sync_heaps();
        let reads = self
            .pipeline
            .as_ref()
            .is_some_and(ComputePipelineState::reads_texture_handles);
        self.dispatch_indirect_resident(call, buffer, offset, arguments)?;
        self.command_encoded(reads);
        self.dispatches += 1;
        Ok(())
    }

    /// [`dispatch_indirect`](Self::dispatch_indirect) with the heaps' textures resident.
    fn dispatch_indirect_resident(
        &mut self,
        call: &'static str,
        buffer: &Buffer,
        offset: u64,
        arguments: Arguments,
    ) -> Result<(), Error> {
        let Some(pipeline) = self.pipeline.clone() else {
            return Err(invalid(call, Problem::NoPipelineState));
        };
        self.core.check_device(call, buffer.device())?;
        let offset = indirect_dispatch::arguments_offset(
            call,
            (buffer.length(), buffer.copy_reach()),
            offset,
            arguments.bytes(),
        )?;
        if self.fails_on_memoryless() {
            return Ok(());
        }
        let tables = DispatchTables {
            pipeline: &pipeline,
            table: &self.table,
            threadgroup_lengths: &self.threadgroup_lengths,
            imageblock: self.imageblock,
            stage_in_region: self.stage_in_region,
            residency: &self.residency,
            forms: BufferForms::Bound,
        };
        let flat = FlatCall {
            buffer,
            offset,
            arguments,
        };
        if self.indirect_stage_in.is_none()
            && record(
                &mut self.core,
                &tables,
                call,
                Counts::Flat(flat),
                Some((self.dispatches, &mut self.live)),
            )?
        {
            return Ok(());
        }
        let copy =
            indirect_dispatch::copy_arguments(&mut self.core, buffer, offset, arguments.bytes())?;
        self.defer_dispatch(call, pipeline, HostDispatch::Indirect { arguments, copy })
    }

    /// Leave `dispatch` of `pipeline` to its host step, with the encoder's tables now and a copy
    /// of its indirect stage-in region where one is set (`indirect_dispatch.rs`).
    fn defer_dispatch(
        &mut self,
        call: &'static str,
        pipeline: ComputePipelineState,
        dispatch: HostDispatch,
    ) -> Result<(), Error> {
        let stage_in_copy = match &self.indirect_stage_in {
            Some((buffer, offset)) => Some(indirect_dispatch::copy_arguments(
                &mut self.core,
                buffer,
                *offset,
                STAGE_IN_REGION_BYTES,
            )?),
            None => None,
        };
        let gate = self
            .core
            .command_buffer()
            .state()
            .predication_mut()
            .host_gate();
        self.core
            .defer(Deferred::Dispatch(Arc::new(ArgumentDispatch {
                call,
                dispatch,
                stage_in_copy,
                pipeline,
                table: self.table.clone(),
                threadgroup_lengths: self.threadgroup_lengths.clone(),
                imageblock: self.imageblock,
                stage_in_region: self.stage_in_region,
                ordered: self.dispatch_type == DispatchType::Serial,
                gate,
            })))
    }

    /// `memoryBarrierWithScope:`: every write of the encoder's earlier dispatches to the kinds
    /// of resource `scope` names is visible to its later dispatches, however they read it
    /// (storage, uniform, or indirect arguments). One global memory barrier covers buffers and
    /// textures alike; render targets are not a compute encoder's, so a scope of only
    /// `RENDER_TARGETS` records nothing. On a serial encoder, which orders its dispatches anyway,
    /// it adds nothing Metal would not do.
    pub fn memory_barrier_with_scope(&mut self, scope: BarrierScope) {
        self.log(|_| ComputeCall::BarrierWithScope(scope));
        if !scope.intersects(BarrierScope::BUFFERS.union(BarrierScope::TEXTURES)) {
            return;
        }
        self.core.barrier(Barrier::between_dispatches());
        self.capture(|| Event::Barrier);
    }

    /// `useResource:usage:`: the encoder's later dispatches may reach `resource` through an
    /// address or a resource ID held in memory, reading or writing as `usage` says
    /// (`residency`). A memoryless texture, a texture buffer too, fails the command buffer as
    /// plain Metal's does, with no dispatch after it too (`CommandBufferError::InvalidResource`:
    /// the encoder's run of compute encoders runs nothing, an M3 Max under macOS 27.0.1).
    ///
    /// Refused, changing nothing: a resource of another device (`Invalid`).
    pub fn use_resource(
        &mut self,
        resource: ResourceRef<'_>,
        usage: ResourceUsage,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::UseResource(Resource::one(resource), usage),
            |this| this.use_all("use_resource", &[resource], usage),
        )
    }

    /// `useResources:count:usage:`: [`use_resource`](Self::use_resource) of each of
    /// `resources`, refused, changing nothing, as it is for any of them.
    pub fn use_resources(
        &mut self,
        resources: &[ResourceRef<'_>],
        usage: ResourceUsage,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::UseResources(Resource::all(resources), usage),
            |this| this.use_all("use_resources", resources, usage),
        )
    }

    fn use_all(
        &mut self,
        call: &'static str,
        resources: &[ResourceRef<'_>],
        usage: ResourceUsage,
    ) -> Result<(), Error> {
        for resource in resources {
            self.core.check_device(call, resource.device())?;
        }
        if resources.iter().any(ResourceRef::is_memoryless) {
            self.core.fail_memoryless_use();
            return Ok(());
        }
        for &resource in resources {
            let used = self
                .residency
                .use_resource(resource, usage, RenderStages::empty());
            if let Some(used) = used {
                self.core.made_resident(used);
            }
            if let ResourceRef::IndirectCommandBuffer(icb) = resource {
                if usage.contains(ResourceUsage::WRITE) {
                    self.core.routes.writable(icb);
                    self.core.retain(icb.gpu_write());
                }
            }
        }
        Ok(())
    }

    /// `useHeap:`: the encoder's later dispatches may read every buffer placed in `heap`
    /// through an address held in memory, and every texture made in it or over a buffer placed
    /// in it through its resource ID, as Metal's contract makes a heap's resources resident for
    /// reading only (`residency`): the textures that live now, and each made later, from the first
    /// dispatch encoded after it is made. The heap's whole range is one used resource, ordered as
    /// the one resource its tracked buffers and textures are.
    ///
    /// A sparse heap holds no buffer: its live sparse textures are made resident, as
    /// `use_resource` with `READ` makes each.
    ///
    /// Refused, changing nothing (`Invalid`): a heap of another device.
    pub fn use_heap(&mut self, heap: &Heap) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::UseHeap(heap.clone()),
            |this| this.use_heaps_of("use_heap", &[heap]),
        )
    }

    /// `useHeaps:count:`: [`use_heap`](Self::use_heap) of each of `heaps`, refused, changing
    /// nothing, as it is for any of them.
    pub fn use_heaps(&mut self, heaps: &[&Heap]) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::UseHeaps(heaps.iter().map(|&heap| heap.clone()).collect()),
            |this| this.use_heaps_of("use_heaps", heaps),
        )
    }

    fn use_heaps_of(&mut self, call: &'static str, heaps: &[&Heap]) -> Result<(), Error> {
        for heap in heaps {
            self.core.check_device(call, heap.device())?;
        }
        for heap in heaps {
            for used in self.residency.use_heap(heap, RenderStages::empty()) {
                self.core.made_resident(used);
            }
        }
        Ok(())
    }

    /// `executeCommandsInBuffer:withRange:`: run commands `range` of `icb` here, as their
    /// records stand when the command buffer is committed, or, where an encoder made `icb` resident
    /// for writing (`use_resource` with `WRITE`), once the work before them completed, as a kernel
    /// wrote them (a host step the device announces once,
    /// `Degradation::HostReplayedIndirectCommands`). Each runs as this encoder's dispatch would,
    /// with its own pipeline state, buffers and lengths, or the encoder's pipeline state and
    /// buffers as they are now where `icb` inherits them, and the resources the encoder made
    /// resident so far, which its buffers' addresses resolve against: an address in none misses,
    /// reading zero and dropping writes, and the execution report counts it. The encoder's commands
    /// before run before them and its commands after run after them.
    ///
    /// Refused, changing nothing (`Invalid`): a buffer of another device, a range that ends
    /// past its size (`IndirectCommandBufferProblem::Range`), and, for a range that is not
    /// empty, no pipeline state set where `icb` inherits it. A buffer of render command types is
    /// `NotImplemented(IndirectRenderCommands)`: Metal's validation lets it through, and what its
    /// commands do in a compute encoder is not measured. An empty range runs nothing, whatever
    /// the encoder's pipeline state, and so does a buffer of no command types, as plain Metal's.
    /// What a command's dispatch would be refused refuses the commit ([`CommandBuffer::commit`]).
    pub fn execute_commands_in_buffer(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.execute_range(icb, range, None)
    }

    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) with a check of the
    /// commands it runs (a runtime extension; Metal has none), which looks for what `needs` names
    /// ([`IndirectCommandCheckNeeds`]; a compute command draws no points): where the commit records
    /// the execution from the CPU's records, or a host step from the records as the GPU and the CPU
    /// left them (a route that may write `icb`'s records met it,
    /// `IndirectCommandBuffer::gpu_written`, or the execution lies inside a block of command
    /// predication), it first calls `check` with each command it will record, in order,
    /// before it records any of them ([`IndirectCommandCheck`]). Where what the records may hold
    /// leaves the check nothing to see, neither calls it, and an execution a host step would record
    /// may run on the GPU alone, as an unchecked one may; a route encoded after the commit that
    /// gives the check something to see before the batch is submitted sends it to the host step.
    /// Refused as [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) refuses.
    pub fn execute_commands_in_buffer_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        self.execute_range(icb, range, Some((check, needs)))
    }

    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer), with `check` for its
    /// host step where one is given, and what it looks for.
    fn execute_range(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::Execute(icb.clone(), range.clone()),
            |this| {
                const CALL: &str = indirect_execute::CALL;
                this.core.check_device(CALL, icb.device())?;
                if icb.is_render() {
                    return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
                }
                let range = icb.command_range(CALL, range.clone())?;
                // An empty range runs nothing before the inherited pipeline state is looked at, as
                // plain Metal's with none set.
                if range.is_empty() {
                    return Ok(());
                }
                let pipeline = this.inherited_pipeline(CALL, icb)?;
                // A buffer of no command types runs nothing, as plain Metal's.
                if icb.has_no_command_types() {
                    return Ok(());
                }
                let fixed = u64::from(range.start)..u64::from(range.end);
                this.defer_execution(icb, ExecutionRange::Fixed(range), pipeline, check)?;
                this.worked();
                this.capture(|| Event::Execute(icb.clone(), fixed));
                Ok(())
            },
        )
    }

    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`: run the commands of `icb`
    /// that the `MTLIndirectCommandBufferExecutionRange` at `offset` of `range_buffer`
    /// (`location`, then `length`, each a `u32`) names as the execution runs, after what the
    /// encoder's commands before it wrote there, as Metal runs them: from the location, clamped at
    /// the buffer's size, so a location at the size runs nothing and a length past it runs to its
    /// end. A location past the size, or a range past `2^32 - 1`, which Metal's GPU faults on, runs
    /// nothing, and the command buffer's execution report states `ReplayFaults::RANGE`. The
    /// commands run as [`execute_commands_in_buffer`](Self::execute_commands_in_buffer)'s that a
    /// function may write do: by a host step once the work before them completed, which the device
    /// announces once (`Degradation::HostReplayedIndirectCommands`; the encoder copies the range
    /// where it is encoded, behind a barrier from everything before).
    ///
    /// A range plain Metal takes and the runtime has no exact answer for is its own gap, not
    /// misuse: 8 bytes that run past `range_buffer`, which Metal reads from whatever follows it
    /// (`NotImplemented(ExecutionRangePastBuffer)`), and an `offset` that is not a multiple of 4
    /// bytes, which Metal's GPU faults the command buffer over where the range read there runs a
    /// command (`NotImplemented(ExecutionRangeOffFour)`; both carry the measurement).
    ///
    /// Refused, changing nothing (`Invalid`): a buffer of another device, and no pipeline state
    /// set where `icb` inherits it. A buffer of render command types is refused as
    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) refuses it. A buffer of
    /// no command types runs nothing, its range unread, as plain Metal's. What a command's dispatch
    /// would be refused fails the command buffer.
    pub fn execute_commands_in_buffer_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Error> {
        self.execute_indirect_range(icb, range_buffer, offset, None)
    }

    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect) with a
    /// check of the commands its host step runs, which always records it, looking for what `needs`
    /// names: the step first calls `check` with each command of the range it read that it will
    /// record, in order, before it records any of them, as
    /// [`execute_commands_in_buffer_checked`](Self::execute_commands_in_buffer_checked)'s does
    /// ([`IndirectCommandCheck`]), unless what the records may hold then leaves the check nothing
    /// to see ([`IndirectCommandCheckNeeds`]). Refused as
    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect) refuses.
    pub fn execute_commands_in_buffer_indirect_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        self.execute_indirect_range(icb, range_buffer, offset, Some((check, needs)))
    }

    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect), with
    /// `check` for its host step where one is given, and what it looks for.
    fn execute_indirect_range(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::ExecuteIndirect(icb.clone(), range_buffer.clone(), offset),
            |this| {
                const CALL: &str = indirect_execute::CALL_INDIRECT;
                this.core.check_device(CALL, icb.device())?;
                this.core.check_device(CALL, range_buffer.device())?;
                if icb.is_render() {
                    return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
                }
                indirect_execute::check_range_buffer(range_buffer.length(), offset)?;
                let pipeline = this.inherited_pipeline(CALL, icb)?;
                // A buffer of no command types runs nothing, as plain Metal's; its range is not
                // read.
                if icb.has_no_command_types() {
                    return Ok(());
                }
                let copy = indirect_dispatch::copy_arguments(
                    &mut this.core,
                    range_buffer,
                    offset,
                    indirect_execute::RANGE_BYTES,
                )?;
                this.defer_execution(icb, ExecutionRange::Indirect(copy), pipeline, check)?;
                this.worked();
                this.capture(|| Event::ExecuteIndirect(icb.clone(), range_buffer.clone(), offset));
                Ok(())
            },
        )
    }

    /// The encoder's pipeline state where `icb` inherits it (refused `NoPipelineState` where
    /// none is set), else `None`.
    fn inherited_pipeline(
        &self,
        call: &'static str,
        icb: &IndirectCommandBuffer,
    ) -> Result<Option<ComputePipelineState>, Error> {
        match (&self.pipeline, icb.descriptor().inherit_pipeline_state) {
            (_, false) => Ok(None),
            (Some(pipeline), true) => Ok(Some(pipeline.clone())),
            (None, true) => Err(invalid(call, Problem::NoPipelineState)),
        }
    }

    /// Leave the execution of `range` of `icb` to record later (`indirect_execute.rs`), with
    /// the encoder's pipeline state `pipeline` where `icb` inherits it, its bound buffers where
    /// `icb` inherits them, and what it made resident so far, and `check` for its host step.
    fn defer_execution(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: ExecutionRange,
        pipeline: Option<ComputePipelineState>,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.sync_heaps();
        // Its commands run the states their records name, where it inherits none.
        if !icb.descriptor().inherit_pipeline_state {
            self.core.routes.executed(icb);
        }
        let table = if icb.descriptor().inherit_buffers {
            self.table.clone()
        } else {
            ArgumentTable::default()
        };
        // Inside a block of command predication: the host step's check of the counter, and the
        // gate the GPU-alone path's preprocess of a fixed range runs from (`gpu_execute.rs`), one
        // invocation per command where the counter is 0.
        let (gate, further, preprocess_gate) = {
            let device = self.core.device().clone();
            let mut state = self.core.command_buffer().state();
            let (predication, uploads) = state.predication_and_uploads();
            let gate = predication.host_gate();
            let (further, preprocess_gate) = match (&gate, &range) {
                (Some(_), ExecutionRange::Fixed(range)) if !range.is_empty() => {
                    let count = [range.len() as u32, 1, 1];
                    let (further, mut gates) = predication.gates(uploads, &device, &[count])?;
                    (further, gates.pop())
                }
                _ => (None, None),
            };
            (gate, further, preprocess_gate)
        };
        if let Some(step) = further {
            predication::record_further(&mut self.core, &step)?;
        }
        self.core.defer(Deferred::Execute(Arc::new(Execute {
            icb: icb.clone(),
            range,
            pipeline,
            table,
            ordered: self.dispatch_type == DispatchType::Serial,
            gate,
            preprocess_gate,
            check: check.map(|(check, needs)| (Mutex::new(check), needs)),
            unchecked: Mutex::new(None),
            read_at: AtomicU64::new(indirect_execute::NOT_READ),
        })))?;
        // Its commands may name any pipeline state.
        self.command_encoded(true);
        Ok(())
    }

    /// Make resident the textures made since the encoder's last command in the heaps it used
    /// (`Residency::sync_heaps`), held until the command buffer completes: a command reaches
    /// every texture its heaps hold as it is encoded.
    fn sync_heaps(&mut self) {
        self.residency.sync_heaps();
    }

    /// Once a command was encoded, note the heaps the encoder used; and where it reads texture
    /// handles from memory (`reads`), the textures it reaches whose blocks a carrier holds.
    fn command_encoded(&mut self, reads: bool) {
        // The command, encoded, used the heaps of what the encoder made resident: noted once a
        // change of it, as an earlier command noted those it held then.
        let generation = self.residency.generation();
        if self.heaps_noted != Some(generation) {
            self.core.touch_heaps(self.residency.heaps());
            self.heaps_noted = Some(generation);
        }
        if !reads {
            return;
        }
        // A texture the kernel's handles reach whose blocks a carrier holds reads as the neutral:
        // among those made resident, as the encoder keeps the first (`compute/live.rs`).
        let bound = self.table.textures.values();
        let resident = (self.live.unreadable).of(self.core.device(), &self.residency);
        degrade_unreadable(
            self.core.device(),
            resident.into_iter().chain(bound),
            "a dispatch",
        );
    }

    /// The encoder's core.
    pub(super) fn core(&mut self) -> &mut EncoderCore {
        &mut self.core
    }

    /// The encoder's state now, which a predicated loop's copies replay a command with.
    pub(super) fn encoder_state(&self) -> EncoderState {
        EncoderState {
            pipeline: self.pipeline.clone(),
            table: self.table.clone(),
            threadgroup_lengths: self.threadgroup_lengths.clone(),
            imageblock: self.imageblock,
            stage_in_region: self.stage_in_region,
            indirect_stage_in: self.indirect_stage_in.clone(),
        }
    }

    /// Set the encoder's state back to `state`: what the table it replaces binds is held until
    /// the command buffer completes (`ArgumentTable::let_go_all`).
    pub(super) fn restore_state(&mut self, state: EncoderState) {
        self.table.let_go_all(&mut self.core);
        self.pipeline = state.pipeline;
        self.table = state.table;
        self.threadgroup_lengths = state.threadgroup_lengths;
        self.imageblock = state.imageblock;
        self.stage_in_region = state.stage_in_region;
        self.indirect_stage_in = state.indirect_stage_in;
    }

    /// The bodies of the predicated loops the encoder has open.
    pub(super) fn captures(&mut self) -> &mut Captures {
        &mut self.captures
    }

    /// What the encoder made resident so far, which a predicated loop carried to its unit's next
    /// encoder replays its commands with (`predication.rs`).
    pub(super) fn residency_now(&self) -> Residency {
        self.residency.clone()
    }

    /// Make resident here what `residency`, an earlier encoder's of the unit, made resident: a
    /// predicated loop's copy replays that encoder's commands here (`predication.rs`).
    pub(super) fn carry_residency(&mut self, residency: &Residency) {
        self.residency.carry(residency);
    }

    /// `endEncoding`. What it binds is held until the command buffer completes, as Metal holds
    /// what an encoder was handed.
    pub fn end_encoding(mut self) -> Result<(), Error> {
        // Its unit of command predication takes what it encoded (`predication.rs`).
        self.end_predication()?;
        self.table.let_go_all(&mut self.core);
        // Its bind state goes on to the device's next compute encoder (`compute/live.rs`).
        self.core.device().shared().live_dispatches.give(self.live);
        // Its log, which its command buffer keeps where a rebind may need it (`replay.rs`).
        let replay = self.log.map(|log| Replay::Compute {
            dispatch_type: self.dispatch_type,
            log,
        });
        // What it made resident joins its command buffer's (`residency.rs`), and is its own
        // (`CommandBuffer::ended_encoders`).
        self.core
            .end_with_replay(EncoderKind::Compute, Some(self.residency), replay)
    }

    /// The threadgroup size `threads` of a dispatch in `call` with the pipeline state set, or
    /// `None` where plain Metal runs nothing for it (an M4; its API validation asserts): a size,
    /// the axes' wrapping 64-bit product, of 0 or above Metal's limit for the state (Apple's 1024,
    /// or the function's or the descriptor's lower one). A size within Metal's limit but above the
    /// state's `max_total_threads_per_threadgroup`, where this device's limit is lower, Metal runs:
    /// it is the device's gap, which the binder refuses typed (`Bind`, `Unsupported`) where the
    /// dispatch is planned, never a dispatch run as nothing. Metal runs the axes modulo `2^32`
    /// where the size passes (`2^62 + 1` by 4 ran 4 threads), and nothing where those axes hold
    /// other threads than the size counts (an M4 under macOS 27.0.1, both dispatch forms: `2^64 -
    /// 1` by `2^64 - 1`, `2^64 - 2` by `2^64 - 2`, and `2^64 - 1` by 1 by `2^64 - 1`, of sizes 1, 4
    /// and 1). The two counts agree modulo `2^32`, so such axes hold at least
    /// `2^32` threads, past any state's limit: a threadgroup the GPU cannot run.
    ///
    /// Refused: no pipeline state set (plain Metal faults).
    fn threadgroup(&self, call: &'static str, threads: Size) -> Result<Option<[u32; 3]>, Error> {
        let Some(pipeline) = &self.pipeline else {
            return Err(invalid(call, Problem::NoPipelineState));
        };
        let total = threads.wrapping_volume();
        let limit = pipeline.metal_max_total_threads_per_threadgroup();
        if total == 0 || total > u64::from(limit) {
            return Ok(None);
        }
        let axes = threads.truncated();
        if axes.iter().map(|&axis| u64::from(axis)).product::<u64>() != total {
            return Ok(None);
        }
        // A size other than the state's required one runs as given, as plain Metal runs it
        // (Metal's API validation stops it).
        Ok(Some(axes))
    }

    /// Whether the kernel statically uses a texture slot that holds a memoryless texture, a
    /// texture buffer's too: plain Metal fails the command buffer of such a dispatch, of no
    /// threadgroups or of an empty threadgroup too, the encoder's run of compute encoders running
    /// nothing (`CommandBufferError::InvalidResource`, an M3 Max under macOS 27.0.1), so the
    /// encoder fails it and the dispatch records nothing (`EncoderCore::fail_memoryless_use`).
    fn fails_on_memoryless(&mut self) -> bool {
        let uses = (self.pipeline.as_ref())
            .is_some_and(|state| self.table.uses_memoryless(state.interface(), Stage::Kernel));
        if uses {
            self.core.fail_memoryless_use();
        }
        uses
    }

    /// Plan, bind, and record `dispatch` with the encoder's tables (the module's steps 2 to 4),
    /// with every texture the encoder's heaps hold resident.
    fn dispatch(&mut self, call: &'static str, dispatch: Dispatch) -> Result<(), Error> {
        self.sync_heaps();
        let reads = self
            .pipeline
            .as_ref()
            .is_some_and(ComputePipelineState::reads_texture_handles);
        self.dispatch_resident(call, dispatch)?;
        self.command_encoded(reads);
        self.dispatches += 1;
        Ok(())
    }

    /// [`dispatch`](Self::dispatch) with the heaps' textures resident.
    fn dispatch_resident(&mut self, call: &'static str, dispatch: Dispatch) -> Result<(), Error> {
        let Some(pipeline) = &self.pipeline else {
            return Err(invalid(call, Problem::NoPipelineState));
        };
        if self.indirect_stage_in.is_some() {
            let pipeline = pipeline.clone();
            return self.defer_dispatch(call, pipeline, HostDispatch::Direct(dispatch));
        }
        let tables = DispatchTables {
            pipeline,
            table: &self.table,
            threadgroup_lengths: &self.threadgroup_lengths,
            imageblock: self.imageblock,
            stage_in_region: self.stage_in_region,
            residency: &self.residency,
            forms: BufferForms::Bound,
        };
        record(
            &mut self.core,
            &tables,
            call,
            Counts::Direct(dispatch),
            Some((self.dispatches, &mut self.live)),
        )
        .map(drop)
    }
}

/// What a dispatch binds and runs with: a compute encoder's tables, or an indirect command's
/// (`indirect_execute.rs`).
pub(super) struct DispatchTables<'a> {
    pub(super) pipeline: &'a ComputePipelineState,
    /// The buffers, textures and samplers, with each buffer's attribute stride where it is not
    /// its layout's own (`setBuffer:offset:attributeStride:atIndex:`, an indirect command's
    /// `setKernelBuffer:offset:attributeStride:atIndex:`).
    pub(super) table: &'a ArgumentTable,
    pub(super) threadgroup_lengths: &'a BTreeMap<u32, u32>,
    pub(super) imageblock: ImageblockSize,
    pub(super) stage_in_region: Option<StageInRegion>,
    pub(super) residency: &'a Residency,
    /// The buffer forms its variant takes: the large-buffer form for an indirect command the
    /// GPU may write.
    pub(super) forms: BufferForms,
}

impl DispatchTables<'_> {
    /// The address-mode maps the dispatch binds its runtime samplers with (`address_map.rs`).
    fn sampler_maps(&self) -> Result<SamplerMaps, Error> {
        address_map::sampler_maps(
            self.pipeline.base_layout(),
            &[(Stage::Kernel, self.table)],
            Some(self.residency),
        )
        .map_err(Error::NotImplemented)
    }

    /// The tables as a bind state, with the device's argument samplers and the live
    /// visible-function tables the pipeline state made. The imageblock size is the plan's only for
    /// a kernel with an explicit imageblock; one without plans with none, so its shape is the one
    /// the executor computes for it.
    fn bind_state(
        &self,
        argument_samplers: &[SamplerState],
        function_tables: &[VisibleFunctionTable],
    ) -> BindState {
        // A buffer where the kernel reads a table reads as a null table.
        let views = self.pipeline.at_bind_views();
        let mut table =
            (self.table).stage_table_reading(Stage::Kernel, self.pipeline.interface(), views);
        table.threadgroup_lengths = self.threadgroup_lengths.clone();
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        state.view_base_alignment = views.alignment();
        if self.pipeline.imageblock_cell_bytes().is_some() {
            state.imageblock_size = self.imageblock;
        }
        state.stage_in_region = self.stage_in_region;
        state.used_resources = self.residency.used_resources();
        state.used_in_token_order = true;
        state.resident_apart = true;
        state.argument_samplers = heap_sets::listed(argument_samplers);
        state.function_tables = function_tables
            .iter()
            .map(VisibleFunctionTable::listed)
            .collect();
        state.heap_placement = HeapPlacement::ResourceId;
        state
    }
}

/// Make, on `device`, the variants a dispatch of `tables` in their buffer forms may run on,
/// whatever its threadgroup size (`ComputePipelineState::prepare`): in the large-buffer form
/// whatever buffers it binds. And the device's neutrals their plans may bind
/// (`binding::make_neutrals`).
pub(super) fn prepare_variants(
    device: &Device,
    tables: &DispatchTables<'_>,
) -> Result<Vec<Arc<Variant>>, Error> {
    let mut state = tables.bind_state(
        &device.argument_samplers(),
        &tables.pipeline.function_tables(),
    );
    tables.sampler_maps()?.apply(&mut state);
    let prepared = tables.pipeline.prepare(&state.shape(), tables.forms)?;
    let mut variants = Vec::with_capacity(prepared.len());
    for (variant, plan) in prepared {
        binding::make_neutrals(device, &plan)?;
        variants.push(variant);
    }
    Ok(variants)
}

/// Plan, bind, and record `dispatch` with `tables` into `core` (the module's steps 2 to 4), for
/// Metal's `call`: an execution's or a host step's, which waits for its pipelines' makings.
pub(super) fn record_dispatch(
    core: &mut EncoderCore,
    tables: &DispatchTables<'_>,
    call: &'static str,
    dispatch: Dispatch,
) -> Result<(), Error> {
    record(core, tables, call, Counts::Direct(dispatch), None).map(drop)
}

/// Where a recorded dispatch's group counts come from.
enum Counts<'a> {
    /// The dispatch's own, each region's (`vkCmdDispatch`).
    Direct(Dispatch),
    /// Metal's indirect arguments, which the flat or folded variant's preprocess reads on the
    /// device (`indirect_dispatch.rs`).
    Flat(FlatCall<'a>),
}

/// The variant an indirect dispatch's GPU path runs on, its plan for `shape`, and the
/// preprocess's description (`indirect_dispatch.rs`): the kernel's flat variant, else, for the
/// threadgroups form, its folded variant of the call's threadgroup size. `None` where neither
/// takes the dispatch, which then takes its host step.
fn settle_indirect(
    pipeline: &ComputePipelineState,
    call: FlatCall<'_>,
    shape: &BindShape,
) -> Result<Option<(Arc<Variant>, Arc<BindingPlan>, FlatIndirect)>, Error> {
    if let Some(indirect) = call.indirect(pipeline) {
        let local = indirect.threads_per_threadgroup;
        let flat = flat_dispatch(local);
        if let Some((variant, plan)) = pipeline.settle_flat(shape, local, &flat)? {
            return Ok(Some((variant, plan, indirect)));
        }
    }
    let Some(indirect) = call.folded(pipeline) else {
        return Ok(None);
    };
    let local = indirect.threads_per_threadgroup;
    Ok(pipeline
        .settle_folded(shape, local)?
        .map(|(variant, plan)| (variant, plan, indirect)))
}

/// The flat variant (`VariantKey::flat_grid`) a direct `dispatchThreads:` runs on, with its plan
/// for `shape`: one Vulkan dispatch of the call's own threadgroup size, every invocation taking
/// the grid thread it counts to and one past the grid's last returning before the body, in place
/// of the interior and ragged-edge regions a split plans — Metal's own one dispatch, on one
/// `VkPipeline`, with no barrier between regions and no `Degradation::SplitDispatch`.
///
/// `None`, so the dispatch keeps its exact split, where its grid is already a whole number of
/// threadgroups on every axis (the split is one region, which the flat form cannot better), where
/// the kernel observes its threadgroup partition or the device reaches no memory by device address
/// (the compile layer's refusal, which the state remembers), where the flat form cannot cover the
/// grid within the device's `maxComputeWorkGroupCount`, or in the large-buffer form, whose key
/// `settle_flat` does not take.
fn flat_direct(
    pipeline: &ComputePipelineState,
    shape: &BindShape,
    dispatch: &Dispatch,
    forms: BufferForms,
) -> Result<Option<(Arc<Variant>, Arc<BindingPlan>)>, Error> {
    let Dispatch::Threads {
        threads,
        threads_per_group,
    } = *dispatch
    else {
        return Ok(None);
    };
    if forms != BufferForms::Bound {
        return Ok(None);
    }
    // The split's regions per axis: its interior where whole threadgroups fill it, and an edge of
    // the remainder. More than one region in all is what the flat form replaces with one.
    let pieces = |i: usize| -> u32 {
        let (full, rem) = match threads_per_group[i] {
            0 => return 0,
            l => (threads[i] / l, threads[i] % l),
        };
        u32::from(full > 0) + u32::from(rem > 0)
    };
    if (0..3).map(pieces).product::<u32>() <= 1 {
        return Ok(None);
    }
    match pipeline.settle_flat(shape, threads_per_group, dispatch) {
        Ok(settled) => Ok(settled),
        // A grid the flat form cannot cover within the device's group-count limits: the split
        // covers it exactly, region by region.
        Err(Error::Bind(BindError::Unsupported(_))) | Err(Error::Unsupported(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Whether a dispatch of `objects`'s variant records in a `VkCommandBuffer` of its own, its
/// pipelines made: where its layout bakes samplers, which the device's driver loses after a copy
/// in the same `VkCommandBuffer` (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`).
fn apart_samplers(device: &Device, objects: &VariantObjects) -> bool {
    objects.bakes_samplers()
        && device
            .properties()
            .driver_defects
            .contains(&RuntimeDriverDefect::ImmutableSamplersLostAfterCopies)
}

/// What a dispatch settles before it binds ([`settle_dispatch`], [`settle_live`]): the device's
/// sampler states it lists (less those at IDs no set-1 sampler array of the variant holds,
/// `unlisted` saying so: `heap_sets::fit_samplers`), the visible-function tables its state made,
/// its runtime samplers' address-mode maps, and what it planned: the state's shape, and the variant
/// it runs on with its plan, for `planned.dispatch`. Its bind state is apart: the encoder's kept one
/// (`compute/live.rs`), or one of its own.
pub(super) struct SettledDispatch {
    argument_samplers: Vec<SamplerState>,
    function_tables: Vec<VisibleFunctionTable>,
    unlisted: Option<(u64, u64)>,
    sampler_maps: SamplerMaps,
    planned: Arc<Planned>,
}

/// A flat indirect call and its preprocess's description (`indirect_dispatch.rs`).
type Flat<'a> = Option<(FlatCall<'a>, FlatIndirect)>;

/// What a dispatch of `tables` that keeps no bind state (an indirect command's, a host step's),
/// of `direct` group counts or of a flat indirect call (`gpu`), settles ([`SettledDispatch`])
/// with `function_tables`, its state's, and `samplers`, the device's sampler states, and the bind
/// state it settled, made whole: the flat call's preprocess description beside it, and `None` where
/// the GPU path does not take an indirect dispatch (`indirect_dispatch.rs`). A direct dispatch
/// whose bind state has the shape one its pipeline state planned lately for the same dispatch and
/// buffer forms, compared in place (`ComputePipelineState::planned_of`), takes what that one
/// planned ([`Planned`]); one that plans keeps what it planned for the next.
fn settle_dispatch<'a>(
    tables: &DispatchTables<'_>,
    (direct, gpu): (Option<Dispatch>, Option<FlatCall<'a>>),
    function_tables: Vec<VisibleFunctionTable>,
    samplers: Vec<SamplerState>,
) -> Result<Option<(BindState, SettledDispatch, Flat<'a>)>, Error> {
    let pipeline = tables.pipeline;
    // The device's sampler states, less those at IDs no set-1 sampler array of the variant
    // holds (each array has the variant's heap capacity of elements): a degradation where
    // the variant has sampler arrays, and nothing reads them where it has none.
    let mut argument_samplers = samplers;
    let mut unlisted = None;
    // Its runtime samplers' address modes as the textures they sample map them.
    let sampler_maps = tables.sampler_maps()?;
    let (state, planned, as_before, flat) = loop {
        let mut state = tables.bind_state(&argument_samplers, &function_tables);
        sampler_maps.apply(&mut state);
        let last = direct.and_then(|dispatch| pipeline.planned_of(&state, &dispatch, tables.forms));
        let (planned, as_before, flat) = match last {
            Some(last) => (last, true, None),
            None => {
                let shape = Arc::new(state.shape());
                match plan_now(pipeline, tables.forms, shape, (direct, gpu))? {
                    Some((planned, flat)) => (planned, false, flat),
                    None => return Ok(None),
                }
            }
        };
        let set = planned.variant.pipeline().layout.sets.get(1);
        if heap_sets::fit_samplers(set, &mut argument_samplers, &mut unlisted) {
            break (state, planned, as_before, flat);
        }
    };
    planned_as_before(pipeline, &planned, as_before, direct);
    let settled = SettledDispatch {
        argument_samplers,
        function_tables,
        unlisted,
        sampler_maps,
        planned,
    };
    Ok(Some((state, settled, flat)))
}

/// [`settle_dispatch`] of a compute encoder's own dispatch, through the encoder's kept bind state
/// `live`, brought in place (`LiveDispatch::refresh`), with `samplers`, the device's sampler
/// states at `changed` changes of their slots: the state's shape kept where nothing brought in may
/// have changed it, else found among those the pipeline state planned for lately or made
/// (`LiveDispatch::shape`), and what it planned for that shape, the very one looked for first
/// (`ComputePipelineState::planned_for`).
fn settle_live<'a>(
    live: &mut LiveDispatch,
    tables: &DispatchTables<'_>,
    (direct, gpu): (Option<Dispatch>, Option<FlatCall<'a>>),
    function_tables: Vec<VisibleFunctionTable>,
    (changed, samplers): (u64, Vec<SamplerState>),
) -> Result<Option<(SettledDispatch, Flat<'a>)>, Error> {
    let pipeline = tables.pipeline;
    let counts = pipeline.device().shared().counts.thread();
    let mut argument_samplers = samplers;
    let mut unlisted = None;
    let sampler_maps = tables.sampler_maps()?;
    // Whether the states were trimmed to those a variant's set-1 sampler arrays hold.
    let mut trimmed = false;
    let (planned, as_before, flat) = loop {
        let listing = (changed, &argument_samplers[..], trimmed);
        live.refresh(tables, &sampler_maps, listing, &function_tables, counts);
        let shape = live.shape(counts, |state| pipeline.planned_shape(state));
        let last =
            direct.and_then(|dispatch| pipeline.planned_for(&shape, &dispatch, tables.forms));
        let (planned, as_before, flat) = match last {
            Some(last) => (last, true, None),
            None => match plan_now(pipeline, tables.forms, shape, (direct, gpu))? {
                Some((planned, flat)) => (planned, false, flat),
                None => return Ok(None),
            },
        };
        let set = planned.variant.pipeline().layout.sets.get(1);
        if heap_sets::fit_samplers(set, &mut argument_samplers, &mut unlisted) {
            break (planned, as_before, flat);
        }
        trimmed = true;
    };
    planned_as_before(pipeline, &planned, as_before, direct);
    let settled = SettledDispatch {
        argument_samplers,
        function_tables,
        unlisted,
        sampler_maps,
        planned,
    };
    Ok(Some((settled, flat)))
}

/// What a dispatch of a bind state of `shape` in `forms` plans now, of `direct` group counts or of
/// a flat indirect call (`gpu`): the direct dispatch's flat variant where it takes one, else
/// `ComputePipelineState::settle`'s; the indirect one's flat or folded variant, with its preprocess
/// description, or `None` where neither takes it, which then takes its host step.
fn plan_now<'a>(
    pipeline: &ComputePipelineState,
    forms: BufferForms,
    shape: Arc<BindShape>,
    (direct, gpu): (Option<Dispatch>, Option<FlatCall<'a>>),
) -> Result<Option<(Arc<Planned>, Flat<'a>)>, Error> {
    match (direct, gpu) {
        (Some(dispatch), _) => {
            let flat = flat_direct(pipeline, &shape, &dispatch, forms)?;
            let (variant, plan) = match flat {
                Some(flat) => flat,
                None => pipeline.settle(&shape, &dispatch, forms)?,
            };
            let planned = Planned {
                shape,
                dispatch,
                forms,
                variant,
                plan,
                made: OnceLock::new(),
            };
            Ok(Some((Arc::new(planned), None)))
        }
        (None, Some(gpu)) => {
            let Some((variant, plan, indirect)) = settle_indirect(pipeline, gpu, &shape)? else {
                return Ok(None);
            };
            let planned = Planned {
                shape,
                dispatch: flat_dispatch(indirect.threads_per_threadgroup),
                forms,
                variant,
                plan,
                made: OnceLock::new(),
            };
            Ok(Some((Arc::new(planned), Some((gpu, indirect)))))
        }
        (None, None) => Err(internal("a dispatch of no counts")),
    }
}

/// Count a direct dispatch that took what its pipeline state planned before (`as_before`), or
/// keep what one planned now for the next; a flat dispatch's variant and plan are the flat path's,
/// none a direct one takes.
fn planned_as_before(
    pipeline: &ComputePipelineState,
    planned: &Arc<Planned>,
    as_before: bool,
    direct: Option<Dispatch>,
) {
    match (as_before, direct) {
        (true, _) => (pipeline.device().shared().counts)
            .thread()
            .dispatch_planned_as_before(),
        (false, Some(_)) => pipeline.keep_planned(Arc::clone(planned)),
        (false, None) => {}
    }
}

/// What a compute encoder's last dispatch that settled settled ([`SettledDispatch`]), its
/// commands, and the `VkPipeline`s its regions found made, kept with what they are a function of
/// beside the encoder's kept bind state, which that dispatch brought in place and binds
/// (`compute/live.rs`): a later dispatch of the encoder of the same inputs settles, plans and
/// instantiates nothing again, brings no state in place and looks no pipeline up
/// (`WorkCounts::dispatches_settled_as_before`). Exact:
///
/// - **The bind state** is a function of the pipeline state (its interface, view base alignment
///   and imageblock among it), of what the table binds (its stamp: every call that may change it
///   draws a new one, `argument_table::Stamp`), of what each bound object's binding reads of it
///   (a texture's and a sampler's facts, fixed as it is made; a buffer's length, base and
///   addresses, which only a no-copy buffer's move changes: `buffer::host_cells_moved`), of the
///   threadgroup lengths, imageblock size and stage-in region set, of what the encoder made
///   resident (its generation, which every change of it draws anew, `Residency::generation`;
///   what a used resource states of it beyond that, its buffers' addresses, which only a no-copy
///   buffer's move changes, and whether each indirect command buffer holds its ID's slot, which a
///   buffer made later with the ID takes: `Residency::holds_slots_as`), of the device's live
///   sampler states (`Device::sampler_slots_changed`), of the state's visible-function tables
///   (none: a dispatch of a state that made one takes no memo), and of the sampler maps, which
///   are the table's and the residency's. The kept state is this entry's: a dispatch that brings
///   it in place keeps its own entry or none (`LiveDispatch::let_go`).
/// - **The variant and plan** are `ComputePipelineState::settle`'s of the shape, the dispatch and
///   the buffer forms; **the commands** are the plan's instantiation for the state and the
///   dispatch, their regions keyed by the variant for the device (`Variant::keyed_regions`).
/// - **A region's `VkPipeline`** found made is its variant's for the region's key, which its
///   making never changes; regions not all made are looked up again at the next dispatch.
/// - **A dispatch that settles again** (another table, another residency) takes the variant, the
///   plan and the made `VkPipeline`s of one of its pipeline state's latest direct dispatches that
///   planned, where its bind state has that one's shape ([`Planned`], [`settle_live`]): so does one
///   whose buffers' offsets alone moved, and an encoder's first dispatch of the state.
#[derive(Default)]
pub(super) struct DispatchMemo(Option<KeptDispatch>);

/// A [`DispatchMemo`]'s entry and what it was settled under.
pub(super) struct KeptDispatch {
    /// The pipeline state, held so that no other state is this one while the entry is kept.
    pipeline: ComputePipelineState,
    stamp: Stamp,
    threadgroup_lengths: BTreeMap<u32, u32>,
    imageblock: ImageblockSize,
    stage_in_region: Option<StageInRegion>,
    forms: BufferForms,
    /// The residency's generation (`Residency::generation`).
    residency: u64,
    /// `buffer::host_cells_moved` and `Device::sampler_slots_changed` as it was settled.
    counted: (u64, u64),
    settled: SettledDispatch,
    commands: OnceLock<DispatchCommands>,
    /// Each region's `VkPipeline`, where all were found made.
    made: OnceLock<Vec<Arc<RawPipeline>>>,
}

impl DispatchMemo {
    /// Whether the entry is of a dispatch of `tables` and `dispatch` at `counted` moves and
    /// sampler slot changes, `state` the kept bind state it settled.
    fn holds(
        &self,
        state: &BindState,
        tables: &DispatchTables<'_>,
        dispatch: &Dispatch,
        counted: (u64, u64),
    ) -> bool {
        self.0.as_ref().is_some_and(|kept| {
            kept.pipeline == *tables.pipeline
                && kept.stamp == tables.table.stamp()
                && kept.settled.planned.dispatch == *dispatch
                && kept.counted == counted
                && kept.forms == tables.forms
                && kept.imageblock == tables.imageblock
                && kept.stage_in_region == tables.stage_in_region
                && kept.threadgroup_lengths == *tables.threadgroup_lengths
                && kept.residency == tables.residency.generation()
                && (tables.residency).holds_slots_as(&state.used_resources)
        })
    }

    /// Keep `settled`, a dispatch of `tables` at `counted` moves and sampler slot changes, in
    /// place of the entry.
    fn keep(&mut self, tables: &DispatchTables<'_>, counted: (u64, u64), settled: SettledDispatch) {
        self.0 = Some(KeptDispatch {
            pipeline: tables.pipeline.clone(),
            stamp: tables.table.stamp(),
            threadgroup_lengths: tables.threadgroup_lengths.clone(),
            imageblock: tables.imageblock,
            stage_in_region: tables.stage_in_region,
            forms: tables.forms,
            residency: tables.residency.generation(),
            counted,
            settled,
            commands: OnceLock::new(),
            made: OnceLock::new(),
        });
    }

    /// The entry, where one is kept.
    fn entry(&self) -> Option<&KeptDispatch> {
        self.0.as_ref()
    }
}

/// Plan, bind, and record the dispatch of `counts` with `tables` into `core` (the module's steps
/// 2 to 4), for Metal's `call`: `Ok(false)`, recording nothing, where the GPU path does not take
/// an indirect dispatch (`indirect_dispatch.rs`); a direct one is always recorded or refused.
/// An encoder's own dispatch (`own`: its order among the encoder's dispatches, and the encoder's
/// kept bind state) inside a block of command predication runs from the gates its unit's step
/// writes (`predication.rs`), binds the kept state brought in place (`compute/live.rs`), and a
/// direct one takes what its last dispatch settled where it has the same inputs
/// ([`DispatchMemo`]). And one whose `VkPipeline`s are not made yet hands their making to the
/// device's pool and records later (`pending_dispatch.rs`): an execution's or a host step's
/// dispatch waits for them.
fn record(
    core: &mut EncoderCore,
    tables: &DispatchTables<'_>,
    call: &'static str,
    counts: Counts<'_>,
    own: Option<(usize, &mut LiveDispatch)>,
) -> Result<bool, Error> {
    let pipeline = tables.pipeline;
    let (direct, gpu) = match counts {
        Counts::Direct(dispatch) => (Some(dispatch), None),
        Counts::Flat(gpu) => (None, Some(gpu)),
    };
    // The live visible-function tables the pipeline state made: the encoder's, or an indirect
    // command's own state's.
    let function_tables = pipeline.function_tables();
    let (own, mut live) = match own {
        Some((own, live)) => (Some(own), Some(live)),
        None => (None, None),
    };
    // What a dispatch that takes no memo entry settled, and the bind state one that keeps none
    // made.
    let mut settled_here = None;
    let mut state_here = None;
    let mut flat = None;
    match live.as_deref_mut() {
        Some(live) => {
            // A direct dispatch, of a state that made no table, takes the memo.
            let memo = direct.filter(|_| function_tables.is_empty());
            let moved = super::buffer::host_cells_moved();
            let held = memo.is_some_and(|dispatch| {
                let counted = (moved, core.device().sampler_slots_changed());
                live.memo.holds(&live.state, tables, &dispatch, counted)
            });
            if held {
                core.counts().dispatch_settled_as_before();
            } else {
                // The sampler states, with the count of changes they are live at.
                let (changed, samplers) = core.device().argument_samplers_at();
                let settling = (changed, samplers);
                let settled = settle_live(live, tables, (direct, gpu), function_tables, settling);
                // The kept state is no longer what the memo's entry settled.
                let Some((settled, flat_call)) = settled.inspect_err(|_| live.let_go())? else {
                    live.let_go();
                    return Ok(false);
                };
                match memo {
                    Some(_) => live.memo.keep(tables, (moved, changed), settled),
                    None => {
                        live.let_go();
                        settled_here = Some(settled);
                        flat = flat_call;
                    }
                }
            }
        }
        None => {
            let samplers = core.device().argument_samplers();
            let Some((state, settled, flat_call)) =
                settle_dispatch(tables, (direct, gpu), function_tables, samplers)?
            else {
                return Ok(false);
            };
            state_here = Some(state);
            settled_here = Some(settled);
            flat = flat_call;
        }
    }
    let (kept, kept_state, resident_shadows) = match live {
        Some(live) => {
            let (state, memo, shadows) = live.parts();
            (memo.entry(), Some(state), Some(shadows))
        }
        None => (None, None, None),
    };
    let settled = match (kept, &settled_here) {
        (Some(kept), _) => &kept.settled,
        (None, Some(settled)) => settled,
        (None, None) => return Err(internal("a dispatch settled nowhere")),
    };
    let state = match (&state_here, kept_state) {
        (Some(state), _) => state,
        (None, Some(state)) => state,
        (None, None) => return Err(internal("a dispatch's bind state nowhere")),
    };
    let SettledDispatch {
        argument_samplers,
        function_tables,
        unlisted,
        sampler_maps,
        planned,
    } = settled;
    let Planned {
        shape,
        variant,
        plan,
        dispatch,
        ..
    } = &**planned;
    let (unlisted, dispatch) = (*unlisted, *dispatch);
    let interface = &variant.pipeline().interface;
    core.device().degrade_argument_pixel_samplers(
        std::slice::from_ref(&variant.pipeline().stage),
        argument_samplers,
    );
    tables
        .table
        .check_function_tables(call, Stage::Kernel, interface)?;
    tables.table.check_offsets(call, Stage::Kernel, interface)?;
    let kernel = pipeline.function_name();
    // The commands, their regions keyed (`Variant::keyed_regions`): those the memo's entry holds,
    // where it holds them.
    let mut instantiated = None;
    let commands = match kept.and_then(|kept| kept.commands.get()) {
        Some(commands) => commands,
        None => {
            let mut commands = plan
                .instantiate_dispatch(state, &dispatch)
                .map_err(Error::Bind)?;
            variant.keyed_regions(core.device(), &mut commands);
            match kept {
                Some(kept) => kept.commands.get_or_init(|| commands),
                None => &*instantiated.insert(commands),
            }
        }
    };
    // A `[[stage_in]]` attribute's 32-bit displacement past 4 GiB of its buffer
    // (`fetch_reach.rs`): the grid of a direct dispatch, any where GPU memory states it.
    fetch_reach::check_stage_in_reach(
        pipeline,
        interface,
        tables.table,
        tables.stage_in_region,
        direct.as_ref().and_then(fetch_reach::grid),
    )?;
    // A direct dispatch reads no indirect arguments; a flat one is one region, whose group
    // counts its preprocess writes (`indirect_dispatch.rs`). A host step reads an indirect
    // dispatch's arguments first.
    let indirect = commands
        .regions
        .iter()
        .filter(|region| region.indirect.is_some())
        .count();
    let expected = match flat {
        None => 0,
        Some(_) => 1,
    };
    if commands.indirect_clamp.is_some()
        || indirect != expected
        || (flat.is_some() && commands.regions.len() != 1)
    {
        return Err(internal(
            "a dispatch's commands read indirect arguments it does not state",
        ));
    }
    if commands.regions.is_empty() {
        core.executed(|| {
            Execution::new(
                ExecutedPipeline::Compute(Arc::clone(variant.shared_pipeline())),
                Arc::clone(shape),
                Vec::new(),
            )
        });
        return Ok(true);
    }
    // Set 1, the heap: its writes at the elements of their resource IDs, and the device's texture
    // and sampler tables at the handle tables.
    let compiled = variant.pipeline();
    let heap = heap_sets::heap_elements(&compiled.layout, plan, state)?;
    // What the variant's layout binds of the device's tables, worked out once (`heap_sets.rs`).
    let heap_tables = (variant.objects()).heap_tables(&compiled.layout, &compiled.dependencies);
    let texture_table = heap_tables.texture_table(core.device())?;
    let sampler_table = heap_tables.sampler_table(core.device())?;
    let stages = [(Stage::Kernel, tables.table)];
    let mut bound_tables = binding::Tables::new(&stages)
        .with_residency(tables.residency)
        .with_texture_table(texture_table.as_ref())
        .with_sampler_table(sampler_table.as_ref())
        .with_sampler_maps(sampler_maps.clone());
    // The flat dispatch's preprocess: its scratch holds the region record the variant reads.
    let preprocess = match &flat {
        Some((_, indirect)) => {
            let preprocess = FlatPreprocess::new(core, indirect)?;
            bound_tables.region_record = Some(preprocess.record);
            Some(preprocess)
        }
        None => None,
    };
    // A direct dispatch on a flat variant reads the record the plan states, in the command's own
    // upload (16-aligned, as the variant reads it): the grid in threads is the call's, so nothing
    // on the device writes it.
    if let Some(bytes) = commands.region_record {
        let device = core.device().clone();
        let record = {
            let mut state = core.command_buffer().state();
            state
                .uploads
                .upload_aligned(&device, &bytes, bytes.len() as u64, 16)?
        };
        let Some(address) = record.device_address() else {
            return Err(internal(
                "a flat dispatch's region record without a device address",
            ));
        };
        bound_tables.region_record = Some(address);
    }
    // The listed function tables: set 1's elements, or the snapshot a variant without the
    // function-table array reads in set 0, which has no heap of its own.
    bound_tables = bound_tables.with_function_tables(function_tables);
    if let Some(heap) = &heap {
        bound_tables = bound_tables.with_heap(argument_samplers, heap);
    }
    let pipeline_stage = &variant.pipeline().stage;
    // A flat dispatch's indirect arguments are its preprocess's command, in the command's own
    // upload, which the preprocess writes between barriers: nothing else orders against it. Its
    // read of Metal's arguments is the command's use instead (`FlatCall::reads`).
    let command_uses: Vec<_> = commands
        .uses
        .iter()
        .filter(|resource| flat.is_none() || resource.resource != UseTarget::Indirect)
        .copied()
        .collect();
    let command = binding::Command {
        descriptor_bytes: &commands.descriptor_bytes,
        heap_tables: &[],
        descriptor_ranges: &commands.descriptor_ranges,
        root_tables: commands
            .root_table
            .iter()
            .map(|table| (metal2vulkan::vk::ShaderStageFlags::COMPUTE, table))
            .collect(),
        // A kernel's artifact states no proof that it stores nothing.
        root_writes: true,
        push_holes: &commands.push_holes,
        uses: &command_uses,
        blobs: pipeline_stage
            .constant_blob
            .iter()
            .map(|blob| (Stage::Kernel, blob))
            .collect(),
        mesh: None,
        tessellation: None,
    };
    let mut uses = binding::command_uses(call, plan, &command, &bound_tables, None, None)?;
    if let Some((call, _)) = &flat {
        uses.push(call.reads());
    }
    // A dispatch that names a detached buffer records nothing: its submission fails.
    let named = uses
        .iter()
        .map(|used| used.key)
        .filter(|key| !uses.resident().contains(key));
    if core.skips(named) {
        tables.table.retain(core);
        return Ok(true);
    }
    let device = core.device().clone();
    // Inside a block of command predication, each command runs from its gate: the flat path's
    // preprocess (whose command then stays zero), else each region's dispatch.
    let gates = match own.is_some() {
        true => {
            let counts: Vec<[u32; 3]> = match &preprocess {
                Some(_) => vec![[1, 1, 1]],
                None => commands
                    .regions
                    .iter()
                    .map(|region| region.group_count)
                    .collect(),
            };
            let (further, gates) = {
                let mut state = core.command_buffer().state();
                let (predication, uploads) = state.predication_and_uploads();
                predication.gates(uploads, &device, &counts)?
            };
            if let Some(step) = further {
                predication::record_further(core, &step)?;
            }
            gates
        }
        false => Vec::new(),
    };
    let mut writes = Vec::new();
    let bound = {
        let mut state = core.command_buffer().state();
        binding::bind(
            call,
            &device,
            &mut state,
            plan,
            &command,
            &bound_tables,
            (None, &mut writes),
        )?
    };
    let mut push = Vec::with_capacity(commands.regions.len());
    for region in &commands.regions {
        let writes = binding::push_writes(region.push_constants.clone(), &bound.holes)?;
        push.push((region.group_count, writes));
    }
    let objects = std::sync::Arc::clone(variant.objects());
    // Set 1 the device's (written below, once nothing refuses).
    let (heap_writes, mut writes): (Vec<_>, Vec<_>) =
        writes.into_iter().partition(|write| write.set > 0);
    // The dispatch binds the variant: the kinds its making announces at every occurrence.
    let formats: BoundFormats<'_> = &|slot| bound_tables.bound_format(slot);
    // An encoder's own dispatch waits for no making: where one of its regions' pipelines is not
    // made, it records later (`pending_dispatch.rs`). An encoder encoded again (`replay.rs`)
    // waits for none either, and leaves the same segment: the making is its submission's to wait
    // for, as the first encoding left it, never the rebind's call on the program's thread.
    // Where the driver loses a layout's baked samplers after a copy in the same
    // `VkCommandBuffer` (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`), a variant that
    // bakes them records in one of its own, as one being made does.
    // Those the memo's entry holds, else those found made for what the dispatch planned, of its
    // regions' keys ([`Planned`]).
    let kept_made = kept
        .and_then(|kept| kept.made.get())
        .or_else(|| planned.made_for(commands))
        .filter(|_| !apart_samplers(&device, &objects));
    // Inside a predicated loop's laying, it waits for them too, so the laying, which its check
    // may submit again (`command_buffer/rerun.rs`), is not split for it; but where its variant
    // records in one of its own.
    let laying = core.command_buffer().state().predication_mut().laying();
    let waits = laying && !apart_samplers(&device, &objects);
    let (pipelines, pending) = match (own.filter(|_| !waits), kept_made) {
        (Some(_), Some(made)) => {
            if let Some(kept) = kept.filter(|kept| kept.made.get().is_none()) {
                drop(kept.made.set(made.clone()));
            }
            (Cow::Borrowed(made.as_slice()), None)
        }
        (Some(dispatch), None) => {
            let begun: Vec<_> = commands
                .regions
                .iter()
                .map(|region| {
                    let pipeline = pipeline.begin_region_pipeline(variant, region, formats);
                    (RegionKey::of(region), pipeline)
                })
                .collect();
            let made: Option<Vec<_>> = begun
                .iter()
                .map(|(_, pipeline)| match pipeline {
                    RegionPipeline::Made(made) => Some(Arc::clone(made)),
                    RegionPipeline::Making(_) => None,
                })
                .collect();
            match made {
                Some(made) if !apart_samplers(&device, &objects) => {
                    // Kept with the entry, which the next dispatch of its inputs takes, and with
                    // what it planned, which the next dispatch of its shape takes.
                    if let Some(kept) = kept {
                        drop(kept.made.set(made.clone()));
                    }
                    // Not a key's residual form, which serves only while its own is made.
                    let own = (begun.iter().zip(&made)).all(|((key, _), made)| {
                        planned.made.get().is_none() && variant.made_as(key, made)
                    });
                    if own {
                        let keys = begun.iter().map(|(key, _)| key.clone()).collect();
                        drop(planned.made.set((keys, made.clone())));
                    }
                    (Cow::Owned(made), None)
                }
                Some(_) => {
                    device
                        .shared()
                        .counts
                        .thread()
                        .baked_sampler_dispatch_apart();
                    (Cow::Owned(Vec::new()), Some((dispatch, begun)))
                }
                None => (Cow::Owned(Vec::new()), Some((dispatch, begun))),
            }
        }
        (None, _) => {
            let mut made = Vec::with_capacity(commands.regions.len());
            for region in &commands.regions {
                made.push(variant.region_pipeline(region, formats)?);
            }
            (Cow::Owned(made), None)
        }
    };
    // Set 0: pushed, or the encoder's last set where it holds the same writes, else one from the
    // command buffer's arena, written (`EncoderCore::set_zero`), with its dynamic offsets.
    let mut offsets = Vec::new();
    let ((set, written), pushed) = match objects.pushes_set_zero() {
        true => (
            (None, None),
            binding::SetWrite::pushed(&writes, objects.immutable_pushes()),
        ),
        false => (
            core.set_zero(&objects, (&mut writes, &mut offsets))?,
            Vec::new(),
        ),
    };
    // The resident textures' shadows as the encoder keeps them (`compute/live.rs`).
    let textures = tables.table.textures.values();
    let shadows = match resident_shadows {
        Some(kept) => shadow::shadow_uses_of(textures, kept.of(&device, tables.residency), &uses),
        None => shadow::shadow_uses(textures.chain(tables.residency.textures()), &uses),
    };
    let copies = core.plan_shadows(&shadows, &uses)?;
    let heap_set = match heap {
        Some(_) => Some(
            heap_sets::bind(
                &device,
                &objects,
                &mut core.command_buffer().state().heaps,
                heap_writes,
                &uses,
            )?
            .set,
        ),
        None => None,
    };
    // A dispatch that records later goes on in a new `VkCommandBuffer`, made now.
    let pending = match pending {
        Some((dispatch, begun)) => Some((dispatch, begun, core.continuation()?)),
        None => None,
    };
    // Recording starts here: nothing below can be refused (but a `VkCommandBuffer` that does not
    // end, which abandons the command buffer).
    if let Some(written) = written {
        core.keep_set_zero(written, &writes);
    }
    core.shadow_copies(&copies);
    if !pushed.is_empty() {
        device.shared().counts.thread().dispatch_pushing_set_zero();
    }
    let recording = Recording {
        layout: objects.layout(),
        set,
        offsets,
        pushed,
        heap: heap_set,
        preprocess,
        gates,
        regions: push,
    };
    let stages = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let regions = &commands.regions;
    match pending {
        Some((dispatch, begun, next)) => {
            // Tracked here, as a command recorded here is; its execution recorded here, its
            // creation times set as its segment records.
            let creation = core.records_executions().then(|| Arc::new(OnceLock::new()));
            if let Some(barrier) = core.command_later(stages, &uses, uses.resident()) {
                let encoder = core.command_buffer().state().open_encoder_place();
                let pending = PendingDispatch {
                    variant: Arc::clone(variant),
                    pipelines: begun,
                    recording,
                    barrier,
                    creation: creation.clone(),
                    state: pipeline.clone(),
                    encoder,
                    dispatch,
                };
                core.pend(next, pending)?;
            }
            core.shadows_used(&shadows, &uses, &[]);
            if let Some(creation) = creation {
                core.executed(|| {
                    Execution::made_later(
                        ExecutedPipeline::Compute(Arc::clone(variant.shared_pipeline())),
                        Arc::clone(shape),
                        creation,
                    )
                });
            }
        }
        None => {
            let made = &*pipelines;
            core.command_reaching(stages, &uses, uses.resident(), |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass;
                // every object recorded is live until the command buffer completes (retained
                // below), and the pipelines are the variant's, one per region.
                unsafe { recording.record(raw, command_buffer, made) }
            });
            core.shadows_used(&shadows, &uses, &[]);
            core.executed(|| {
                Execution::new(
                    ExecutedPipeline::Compute(Arc::clone(variant.shared_pipeline())),
                    Arc::clone(shape),
                    pipelines.iter().map(|made| made.created_in).collect(),
                )
            });
        }
    }
    objects.degrade_bound_formats(
        std::slice::from_ref(&variant.pipeline().stage),
        Some(formats),
    );
    // Each specialization the dispatch's regions take, once.
    for (n, region) in regions.iter().enumerate() {
        if regions[..n]
            .iter()
            .all(|earlier| earlier.specialization != region.specialization)
        {
            device.degrade_bound_values(
                std::slice::from_ref(&variant.pipeline().stage),
                &region.specialization,
                &|k| plan.specialized_value(k),
                Some(formats),
            );
        }
    }
    device.degrade_cross_class(&plan.diagnostics, formats);
    device.degrade_astc_storage(&plan.diagnostics, formats);
    objects.announce_bound(
        (
            std::slice::from_ref(&variant.pipeline().stage),
            CompiledPipeline::Kernel(variant.pipeline()),
        ),
        regions
            .iter()
            .map(|region| (&region.specialization, Some(formats))),
    );
    // Held by reference, each once while the encoder held it lately (`retained.rs`): a
    // dispatch of the objects of the one before it holds none again.
    core.retain_shared(pipeline.held());
    core.retain_shared(objects);
    for made in pipelines.iter() {
        core.retain_shared(made.clone());
    }
    tables.table.retain(core);
    for sampler in bound.samplers {
        core.retain_shared(sampler);
    }
    if let Some((call, _)) = &flat {
        core.retain_shared(call.buffer.held());
    }
    // The tables the heap's function-table array may name: their IDs stay theirs until the
    // command buffer completes.
    if heap.is_some() && !function_tables.is_empty() {
        core.retain(function_tables.clone());
    }
    let emulated = variant.emulates_simdgroups();
    degrade_dispatch(&device, kernel, &dispatch, regions.len(), emulated);
    if let Some(unlisted) = unlisted {
        heap_sets::degrade_unlisted(
            &device,
            unlisted,
            &format!("kernel `{kernel}`'s dispatches"),
        );
        // It runs with the states its sampler arrays hold.
        let mut held = core.command_buffer().state();
        held.unlisted_argument_sampler(&device);
    }
    Ok(true)
}
