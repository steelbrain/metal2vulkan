//! Encoders encoded again at a rebind: Metal resolves what a command buffer's commands name as it
//! first submits it, so a command buffer committed before a rebind of a buffer one of its commands
//! named, and first submitted after it, runs that command over the buffer's new pages (the
//! consumer's rebind probe on the M3 Max, table 2 cases 1, 2 and 5). The runtime records each
//! command into its `VkCommandBuffer` as it is encoded, with the `VkBuffer` the buffer lay in then;
//! so a compute, blit or render encoder logs its calls while the device has a buffer over host
//! pages, and the rebind encodes the encoders that named the buffer again from their logs before
//! the command buffer's first submission (`command_buffer/reencode.rs`).
//!
//! - **The log.** Each public call of a compute, blit or render encoder, with its arguments owned (a
//!   handle's clone, held once for the calls that name it in turn; bytes copied into the log's own
//!   arena), in order, before the call runs:
//!   encoding it again re-runs each call, so the commands come out as the program encoded them,
//!   over where their buffers lie now. A call refused at its first encoding is noted so
//!   ([`CallLog::refuse`]: the encoder's `logged`, a render encoder's call made at once): encoded
//!   again, it is refused again and the calls after it run, as they ran after its refusal. A call
//!   that answers otherwise than at its first encoding leaves the encoder as it was encoded first
//!   ([`Replay::run`] fails, and the encoder is not encoded again). The log starts at the
//!   encoder's open where the device has a buffer over host pages (`Rebinds::any`); else nothing
//!   is logged. It costs one push per call; a slice of buffers, offsets, textures or bytes goes
//!   into one of the log's arenas, so the common calls allocate nothing of their own.
//! - **What is kept.** The command buffer keeps an ended encoder's log only where a command of
//!   it named a buffer over host pages and the encoder can be encoded again: it left no segment
//!   for later (an indirect stage-in region) but an indirect command buffer's execution, which
//!   the encoding again records at once on the GPU alone, as its commit did, or refuses where it
//!   no longer runs there (`EncoderCore::finish_replay`); and it fails nothing as it commits. A
//!   compute encoder of a command buffer that used command predication keeps its log, its Ifs,
//!   Elses and Ends among its calls, where its unit laid no loop and left no work out and, of its
//!   unit's predication, nothing reached in over its open and nothing of its own reaches out over
//!   its end (`Predication::alone`): encoded again, it steps and gates in its unit begun anew, as
//!   it did (`Predication::replaying`). Any other compute encoder of such a command buffer keeps
//!   none (a command of it encoded again after the commit would run outside its block, or leave a
//!   command of its unit's reading a gate nothing writes): it fails
//!   `CommandBufferError::EncodedUseRebound` at a rebind as before, as does
//!   any other encoder (a resource state or acceleration structure encoder's), which logs
//!   nothing. (Changed:) a
//!   render encoder logs as these do, with its pass's descriptor (`render/replay.rs`).
//! - **Its memory.** No fixed cap (the coordinator's ruling of 2026-10-11): each entry holds its
//!   call's own arguments, so a log grows with its encoder's recording, as Metal's command buffer
//!   holds what the same calls encode. The command buffer keeps the logs its commit kept until it
//!   completed: they go in its release, after its completed handlers, with what it retained, on its
//!   queue's handler thread, before its waiters return (`CommandBufferShared::complete`). Metal
//!   holds what an encoder was given until its command buffer completed, a binding replaced before
//!   any command among it (a no-copy buffer's deallocator ran only after the completion, never at
//!   the end of encoding nor while the committed command buffer waited: an M3 Max under macOS
//!   27.0.1, probe `dealloc-probe`). And no other thread may hold such a handle past the
//!   completion: a no-copy buffer's deallocator would run where it went, on the scheduler's thread
//!   for one, whose own hold of the command buffer keeps the deallocator's wait for it from
//!   returning (`scheduler::Spent`). A log let go of in the release, or at its encoder's end
//!   where it is not kept, is cleared (its calls and handles go then) and its room kept for the
//!   device's next encoders ([`SpareLogs`]), so the encoding thread's logs reuse their memory
//!   rather than allocate what another thread frees.
//! - **Notices.** A call encoded again announces no kind its first encoding announced: each kind
//!   at most once (`Device::degrade`), as the program's call already met it.

use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::depth_stencil::DepthStencilState;
use super::device::lock;
use super::encoder::Replayed;
use super::error::{internal, Error};
use super::fence::Fence;
use super::function_table::VisibleFunctionTable;
use super::heap::Heap;
use super::host_ram::copy::HostRamCall;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::pipeline::{ComputePipelineState, RenderPipelineState};
use super::render::{RenderLog, RenderRoom};
use super::render_pass::RenderPassDescriptor;
use super::render_values::ClearColor;
use super::residency::Residency;
use super::resource::ResourceRef;
use super::sampler::SamplerState;
use super::texture::Texture;
use super::used_resources::EncoderKind;
use super::values::{Origin, Region, Size};
use super::{BlitCommandEncoder, ComputeCommandEncoder};
use crate::copy_plan::BlitOption;
use metal2vulkan::metal::{BarrierScope, DispatchType, ResourceUsage};
use std::cell::Cell;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// The handles of one kind a log holds, each once among the latest it took (`Handles::hold`).
struct Handles<T> {
    held: Vec<T>,
}

impl<T> Default for Handles<T> {
    fn default() -> Handles<T> {
        Handles { held: Vec::new() }
    }
}

/// How many of a kind's latest handles a log looks among for the one a call names before it holds
/// that one again: calls name what the calls just before them named (a buffer bound again at
/// another offset, a pipeline state or sampler set again), and a look past these few would cost
/// more than the hold it saves.
const RECENT: usize = 8;

impl<T: Clone + PartialEq> Handles<T> {
    /// Where `handle` (the same object, by identity) is held: among the latest [`RECENT`], or
    /// held now, and whether it was held now.
    fn hold(&mut self, handle: &T) -> (Held, bool) {
        let recent = self.held.len().saturating_sub(RECENT);
        let at = match self.held[recent..].iter().rposition(|held| held == handle) {
            Some(at) => (recent + at, false),
            None => {
                self.held.push(handle.clone());
                (self.held.len() - 1, true)
            }
        };
        // A log holds fewer handles than it has calls, each of 32 bytes or more.
        (Held(at.0 as u32), at.1)
    }

    fn get(&self, held: Held) -> &T {
        &self.held[held.0 as usize]
    }
}

impl<T> Handles<T> {
    /// The bytes its list has room for.
    fn room(&self) -> usize {
        self.held.capacity() * std::mem::size_of::<T>()
    }

    /// Let go of every handle, keeping the list's room.
    fn clear(&mut self) {
        self.held.clear();
    }
}

/// A handle a log holds, by its place among those of its kind (`CallLog::hold_buffer` and the
/// like): a call that names a handle again shares the log's one hold of it, so a call takes no
/// reference of its own, whose count the thread that drops the log would share with the encoding
/// thread's.
#[derive(Clone, Copy)]
pub(crate) struct Held(u32);

/// Where a log stood ([`CallLog::mark`]): its lists' lengths and its size.
#[derive(Clone, Copy)]
pub(super) struct LogMark {
    calls: usize,
    bytes: usize,
    buffers: usize,
    words: usize,
    textures: usize,
    held: [usize; 5],
    size: usize,
    refused: usize,
}

/// A run of elements in one of a log's arenas.
#[derive(Clone, Copy)]
pub(super) struct Span {
    start: usize,
    len: usize,
}

impl Span {
    fn range(self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

/// A resource a call named, owned.
#[derive(Clone)]
pub(super) enum Resource {
    Buffer(Buffer),
    Texture(Texture),
    IndirectCommandBuffer(IndirectCommandBuffer),
}

impl Resource {
    fn of(resource: ResourceRef<'_>) -> Resource {
        match resource {
            ResourceRef::Buffer(buffer) => Resource::Buffer(buffer.clone()),
            ResourceRef::Texture(texture) => Resource::Texture(texture.clone()),
            ResourceRef::IndirectCommandBuffer(icb) => Resource::IndirectCommandBuffer(icb.clone()),
        }
    }

    pub(crate) fn as_ref(&self) -> ResourceRef<'_> {
        match self {
            Resource::Buffer(buffer) => ResourceRef::Buffer(buffer),
            Resource::Texture(texture) => ResourceRef::Texture(texture),
            Resource::IndirectCommandBuffer(icb) => ResourceRef::IndirectCommandBuffer(icb),
        }
    }

    /// `resources`, owned.
    pub(super) fn all(resources: &[ResourceRef<'_>]) -> Box<[Resource]> {
        resources
            .iter()
            .map(|&resource| Resource::of(resource))
            .collect()
    }

    /// One of `resource`, owned.
    pub(super) fn one(resource: ResourceRef<'_>) -> Resource {
        Resource::of(resource)
    }
}

/// An encoder's calls, in order, with the arenas their slices lie in.
pub(super) struct CallLog<C> {
    pub(crate) calls: Vec<C>,
    bytes: Vec<u8>,
    buffers: Vec<Option<Buffer>>,
    words: Vec<u64>,
    textures: Vec<Option<Texture>>,
    /// The handles its calls name one at a time, each held once among the latest of its kind.
    held_buffers: Handles<Buffer>,
    held_textures: Handles<Texture>,
    held_samplers: Handles<SamplerState>,
    held_render_states: Handles<RenderPipelineState>,
    held_depth_stencils: Handles<DepthStencilState>,
    /// What it holds, counted as it grows: each call's size, and its arenas' and boxes'.
    size: usize,
    /// Calls of the encoder's own public calls run now, which the call that makes them logged
    /// (`BlitCommandEncoder::unlogged`).
    nested: usize,
    /// The places among its calls of those refused at their first encoding, in order
    /// ([`refuse`](Self::refuse)).
    refused: Vec<u32>,
}

impl<C> CallLog<C> {
    /// An empty log, which allocates nothing until its first call.
    pub(super) fn new() -> CallLog<C> {
        CallLog {
            calls: Vec::new(),
            bytes: Vec::new(),
            buffers: Vec::new(),
            words: Vec::new(),
            textures: Vec::new(),
            held_buffers: Handles::default(),
            held_textures: Handles::default(),
            held_samplers: Handles::default(),
            held_render_states: Handles::default(),
            held_depth_stencils: Handles::default(),
            size: 0,
            nested: 0,
            refused: Vec::new(),
        }
    }

    /// `held`'s place in its kind's arena, its size counted where it was held now.
    fn counted<T>(&mut self, (held, now): (Held, bool)) -> Held {
        if now {
            self.size += std::mem::size_of::<T>();
        }
        held
    }

    /// `buffer`, held once ([`Held`]).
    pub(super) fn hold_buffer(&mut self, buffer: &Buffer) -> Held {
        let held = self.held_buffers.hold(buffer);
        self.counted::<Buffer>(held)
    }

    /// `texture`, held once ([`Held`]).
    pub(super) fn hold_texture(&mut self, texture: &Texture) -> Held {
        let held = self.held_textures.hold(texture);
        self.counted::<Texture>(held)
    }

    /// `sampler`, held once ([`Held`]).
    pub(super) fn hold_sampler(&mut self, sampler: &SamplerState) -> Held {
        let held = self.held_samplers.hold(sampler);
        self.counted::<SamplerState>(held)
    }

    /// `state`, held once ([`Held`]).
    pub(super) fn hold_render_state(&mut self, state: &RenderPipelineState) -> Held {
        let held = self.held_render_states.hold(state);
        self.counted::<RenderPipelineState>(held)
    }

    /// `state`, held once ([`Held`]).
    pub(super) fn hold_depth_stencil(&mut self, state: &DepthStencilState) -> Held {
        let held = self.held_depth_stencils.hold(state);
        self.counted::<DepthStencilState>(held)
    }

    /// The buffer `held` holds ([`CallLog::hold_buffer`]).
    pub(crate) fn buffer(&self, held: Held) -> &Buffer {
        self.held_buffers.get(held)
    }

    /// The texture `held` holds ([`CallLog::hold_texture`]).
    pub(crate) fn texture(&self, held: Held) -> &Texture {
        self.held_textures.get(held)
    }

    /// The sampler `held` holds ([`CallLog::hold_sampler`]).
    pub(crate) fn sampler(&self, held: Held) -> &SamplerState {
        self.held_samplers.get(held)
    }

    /// The render pipeline state `held` holds ([`CallLog::hold_render_state`]).
    pub(crate) fn render_state(&self, held: Held) -> &RenderPipelineState {
        self.held_render_states.get(held)
    }

    /// The depth and stencil state `held` holds ([`CallLog::hold_depth_stencil`]).
    pub(crate) fn depth_stencil(&self, held: Held) -> &DepthStencilState {
        self.held_depth_stencils.get(held)
    }

    /// Whether the encoder runs a public call of its own now, which logs nothing.
    pub(super) fn muted(&self) -> bool {
        self.nested > 0
    }

    /// Enter (`true`) or leave a public call of the encoder's own, which logs nothing.
    pub(super) fn mute(&mut self, enter: bool) {
        match enter {
            true => self.nested += 1,
            false => self.nested -= 1,
        }
    }

    /// Log `call`, which holds `extra` bytes of its own besides its size (a box's): its place
    /// among the calls.
    pub(super) fn push(&mut self, call: C, extra: usize) -> usize {
        self.calls.push(call);
        self.size += std::mem::size_of::<C>() + extra;
        self.calls.len() - 1
    }

    /// The call at `at`, logged by a public call that was refused, was refused at its first
    /// encoding ([`again`](Self::again)).
    pub(super) fn refuse(&mut self, at: usize) {
        // Fewer than the log's calls, each of 24 bytes or more.
        self.refused.push(at as u32);
        self.size += std::mem::size_of::<u32>();
    }

    /// The call at `at`, encoded again with the answer `ran`, answered as at its first encoding:
    /// a call refused then is refused again and the calls after it run, as they ran after its
    /// refusal; a refusal of a call that ran then stands, as does a call that runs now that was
    /// refused then (`Replay::run` fails: the encoding again differs from the first).
    pub(crate) fn again(&self, at: usize, ran: Result<(), Error>) -> Result<(), Error> {
        let refused = self.refused.binary_search(&(at as u32)).is_ok();
        match (ran, refused) {
            (Ok(()), false) | (Err(_), true) => Ok(()),
            (Ok(()), true) => Err(internal(
                "a call refused at its first encoding ran as it was encoded again",
            )),
            (Err(error), false) => Err(error),
        }
    }

    /// The bytes it holds.
    pub(super) fn bytes(&self) -> usize {
        self.size
    }

    /// `bytes`, copied into the log's arena.
    pub(super) fn byte_span(&mut self, bytes: &[u8]) -> Span {
        self.size += bytes.len();
        span(&mut self.bytes, bytes.iter().copied())
    }

    /// `buffers`, held in the log's arena.
    pub(super) fn buffer_span(&mut self, buffers: &[Option<&Buffer>]) -> Span {
        self.size += std::mem::size_of_val(buffers);
        span(
            &mut self.buffers,
            buffers.iter().map(|buffer| buffer.cloned()),
        )
    }

    /// `words`, copied into the log's arena.
    pub(super) fn word_span(&mut self, words: &[u64]) -> Span {
        self.size += std::mem::size_of_val(words);
        span(&mut self.words, words.iter().copied())
    }

    /// `textures`, held in the log's arena.
    pub(super) fn texture_span(&mut self, textures: &[Option<&Texture>]) -> Span {
        self.size += std::mem::size_of_val(textures);
        span(
            &mut self.textures,
            textures.iter().map(|texture| texture.cloned()),
        )
    }

    pub(crate) fn bytes_of(&self, span: Span) -> &[u8] {
        &self.bytes[span.range()]
    }

    pub(crate) fn buffers_of(&self, span: Span) -> Vec<Option<&Buffer>> {
        self.buffers[span.range()]
            .iter()
            .map(Option::as_ref)
            .collect()
    }

    /// `span`'s buffers as the log holds them.
    pub(crate) fn buffer_slice(&self, span: Span) -> &[Option<Buffer>] {
        &self.buffers[span.range()]
    }

    /// `span`'s textures as the log holds them.
    pub(crate) fn texture_slice(&self, span: Span) -> &[Option<Texture>] {
        &self.textures[span.range()]
    }

    pub(crate) fn words_of(&self, span: Span) -> &[u64] {
        &self.words[span.range()]
    }

    pub(crate) fn textures_of(&self, span: Span) -> Vec<Option<&Texture>> {
        self.textures[span.range()]
            .iter()
            .map(Option::as_ref)
            .collect()
    }

    /// Where the log stands now: [`truncate`](Self::truncate) takes it back there.
    pub(super) fn mark(&self) -> LogMark {
        LogMark {
            calls: self.calls.len(),
            bytes: self.bytes.len(),
            buffers: self.buffers.len(),
            words: self.words.len(),
            textures: self.textures.len(),
            held: [
                self.held_buffers.held.len(),
                self.held_textures.held.len(),
                self.held_samplers.held.len(),
                self.held_render_states.held.len(),
                self.held_depth_stencils.held.len(),
            ],
            size: self.size,
            refused: self.refused.len(),
        }
    }

    /// Take the log back to `mark`: what was logged since, a call refused by its checks
    /// (`render/deferral.rs`), let go of.
    pub(super) fn truncate(&mut self, mark: LogMark) {
        self.calls.truncate(mark.calls);
        self.bytes.truncate(mark.bytes);
        self.buffers.truncate(mark.buffers);
        self.words.truncate(mark.words);
        self.textures.truncate(mark.textures);
        let [buffers, textures, samplers, states, depth_stencils] = mark.held;
        self.held_buffers.held.truncate(buffers);
        self.held_textures.held.truncate(textures);
        self.held_samplers.held.truncate(samplers);
        self.held_render_states.held.truncate(states);
        self.held_depth_stencils.held.truncate(depth_stencils);
        self.refused.truncate(mark.refused);
        self.size = mark.size;
    }

    /// Let go of its calls and handles, keeping its lists' room: a deferred render encoder's
    /// calls once they are lowered (`render/deferral.rs`).
    pub(super) fn clear(&mut self) {
        self.calls.clear();
        self.bytes.clear();
        self.buffers.clear();
        self.words.clear();
        self.textures.clear();
        self.held_buffers.clear();
        self.held_textures.clear();
        self.held_samplers.clear();
        self.held_render_states.clear();
        self.held_depth_stencils.clear();
        self.refused.clear();
        self.size = 0;
        self.nested = 0;
    }

    /// The log emptied for another encoder, its calls and handles let go of and its lists' room
    /// kept: `None` where that room is more than [`SPARE_LOG_BYTES`].
    pub(super) fn cleared(mut self) -> Option<CallLog<C>> {
        let room = self.calls.capacity() * std::mem::size_of::<C>()
            + self.bytes.capacity()
            + self.buffers.capacity() * std::mem::size_of::<Option<Buffer>>()
            + self.words.capacity() * std::mem::size_of::<u64>()
            + self.textures.capacity() * std::mem::size_of::<Option<Texture>>()
            + self.held_buffers.room()
            + self.held_textures.room()
            + self.held_samplers.room()
            + self.held_render_states.room()
            + self.held_depth_stencils.room()
            + self.refused.capacity() * std::mem::size_of::<u32>();
        if room > SPARE_LOG_BYTES {
            return None;
        }
        self.clear();
        Some(self)
    }
}

/// How many cleared logs of each kind of encoder the device keeps ([`SpareLogs`]).
const SPARE_LOGS: usize = 16;

/// The most room a cleared log keeps for a next encoder, in bytes; a log with more goes.
pub(super) const SPARE_LOG_BYTES: usize = 64 * 1024;

/// Cleared logs the device keeps for its next encoders (the module docs): at most [`SPARE_LOGS`]
/// of each kind, each with at most [`SPARE_LOG_BYTES`] of room, and no call or handle. An
/// encoder that logs takes one, or a new log where none is kept; its log comes back as its
/// encoder ends where its command buffer does not keep it, and else in the command buffer's
/// release, on its queue's handler thread.
#[derive(Default)]
pub(crate) struct SpareLogs {
    compute: Spares<CallLog<ComputeCall>>,
    blit: Spares<CallLog<BlitCall>>,
    render: Spares<RenderRoom>,
}

/// One kind's [`SpareLogs`].
struct Spares<T>(Mutex<Vec<T>>);

impl<T> Default for Spares<T> {
    fn default() -> Spares<T> {
        Spares(Mutex::new(Vec::new()))
    }
}

impl<T> Spares<T> {
    fn take(&self) -> Option<T> {
        lock(&self.0).pop()
    }

    /// Keep `spare` where fewer than [`SPARE_LOGS`] are kept; else it goes, after the lock.
    fn give(&self, spare: T) {
        let mut spares = lock(&self.0);
        if spares.len() < SPARE_LOGS {
            spares.push(spare);
        }
    }
}

impl SpareLogs {
    /// A compute encoder's log: a kept one, or a new one.
    pub(super) fn compute(&self) -> CallLog<ComputeCall> {
        self.compute.take().unwrap_or_else(CallLog::new)
    }

    /// A blit encoder's log: a kept one, or a new one.
    pub(super) fn blit(&self) -> CallLog<BlitCall> {
        self.blit.take().unwrap_or_else(CallLog::new)
    }

    /// The log of a pass `descriptor` opens, over a rasterization-rate map of `screen` pixels
    /// where it has one: in a kept one's room, or a new one.
    pub(super) fn render(
        &self,
        descriptor: &RenderPassDescriptor,
        screen: Option<[u32; 2]>,
    ) -> RenderLog {
        RenderLog::new(descriptor, screen, self.render.take())
    }

    /// Clear `replay` and keep it for a next encoder, where its room is small enough
    /// ([`CallLog::cleared`]): its calls and handles go now.
    pub(super) fn give(&self, replay: Replay) {
        match replay {
            Replay::Compute { log, .. } => {
                if let Some(log) = log.cleared() {
                    self.compute.give(log);
                }
            }
            Replay::Blit(log) => {
                if let Some(log) = log.cleared() {
                    self.blit.give(log);
                }
            }
            Replay::Render(log) => {
                if let Some(room) = log.cleared() {
                    self.render.give(room);
                }
            }
        }
    }

    /// [`give`](Self::give) `replay` where this is its last hold; else the hold goes.
    pub(super) fn give_last(&self, replay: Arc<Replay>) {
        if let Some(replay) = Arc::into_inner(replay) {
            self.give(replay);
        }
    }
}

/// Append `items` to `arena`: their span.
fn span<T>(arena: &mut Vec<T>, items: impl ExactSizeIterator<Item = T>) -> Span {
    let start = arena.len();
    let len = items.len();
    arena.extend(items);
    Span { start, len }
}

/// One public call of a compute encoder, its arguments owned (the method of each variant's name).
pub(super) enum ComputeCall {
    PipelineState(ComputePipelineState),
    Buffer(Option<Buffer>, u64, u32),
    Buffers(Span, Span, u32),
    BufferOffset(u64, u32),
    BufferWithStride(Option<Buffer>, u64, u64, u32),
    BuffersWithStrides(Span, Span, Span, u32),
    BufferOffsetWithStride(u64, u64, u32),
    Bytes(Span, u32),
    BytesWithStride(Span, u64, u32),
    FunctionTable(Option<VisibleFunctionTable>, u32),
    FunctionTables(Box<[Option<VisibleFunctionTable>]>, u32),
    Texture(Option<Texture>, u32),
    Textures(Span, u32),
    Sampler(Option<SamplerState>, u32),
    Samplers(Box<[Option<SamplerState>]>, u32),
    SamplerWithLodClamps(Option<SamplerState>, f32, f32, u32),
    SamplersWithLodClamps(Box<SamplersWithLodClamps>),
    ThreadgroupMemoryLength(u32, u32),
    ImageblockSize(u64, u64),
    StageInRegion(Region),
    StageInRegionIndirect(Buffer, u64),
    Threadgroups(Size, Size),
    Threads(Size, Size),
    ThreadgroupsIndirect(Buffer, u64, Size),
    ThreadsIndirect(Buffer, u64),
    BarrierWithScope(BarrierScope),
    BarrierWithResources(Box<[Resource]>),
    UseResource(Resource, ResourceUsage),
    UseResources(Box<[Resource]>, ResourceUsage),
    UseHeap(Heap),
    UseHeaps(Box<[Heap]>),
    Execute(IndirectCommandBuffer, Range<u64>),
    ExecuteIndirect(IndirectCommandBuffer, Buffer, u64),
    UpdateFence(Fence),
    WaitForFence(Fence),
    /// `encode_start_if`, `encode_start_else` and `encode_end_if` (with the answer its first
    /// encoding gave) of command predication: a loop's record stops the log (`predication.rs`).
    StartIf(Option<Buffer>, u64, u64, u32),
    StartElse,
    EndIf(bool),
}

impl ComputeCall {
    /// The bytes it holds in boxes of its own, besides its size.
    pub(super) fn extra(&self) -> usize {
        match self {
            ComputeCall::FunctionTables(tables, _) => std::mem::size_of_val(&**tables),
            ComputeCall::Samplers(samplers, _) => std::mem::size_of_val(&**samplers),
            ComputeCall::SamplersWithLodClamps(call) => {
                std::mem::size_of::<SamplersWithLodClamps>()
                    + std::mem::size_of_val(&*call.samplers)
                    + std::mem::size_of_val(&*call.mins)
                    + std::mem::size_of_val(&*call.maxes)
            }
            ComputeCall::BarrierWithResources(resources)
            | ComputeCall::UseResources(resources, _) => std::mem::size_of_val(&**resources),
            ComputeCall::UseHeaps(heaps) => std::mem::size_of_val(&**heaps),
            _ => 0,
        }
    }
}

/// `set_sampler_states_with_lod_clamps`'s arguments.
pub(super) struct SamplersWithLodClamps {
    pub(super) samplers: Box<[Option<SamplerState>]>,
    pub(super) mins: Box<[f32]>,
    pub(super) maxes: Box<[f32]>,
    pub(super) first: u32,
}

/// One public call of a blit encoder, its arguments owned (the method of each variant's name; a
/// color fill the device does not clear as the byte fill of the bytes it fills).
pub(super) enum BlitCall {
    CopyBufferToBuffer(Buffer, u64, Buffer, u64, u64),
    FillBuffer(Buffer, Range<u64>, u8),
    FillBufferPattern4(Buffer, Range<u64>, u32),
    SynchronizeResource(Resource),
    FillTexture(Box<(Texture, u64, u64, Region)>, Span),
    FillTextureColor(Box<(Texture, u64, u64, Region, ClearColor)>),
    CopyBufferToTexture(Box<BufferToTexture>),
    CopyTextureToBuffer(Box<TextureToBuffer>),
    CopyTextureToTexture(Box<TextureToTexture>),
    CopyTextureSlices(Box<(Texture, u64, u64, Texture, u64, u64, u64, u64)>),
    CopyTexture(Texture, Texture),
    GenerateMipmaps(Texture),
    SynchronizeTexture(Texture, u64, u64),
    OptimizeForGpu(Texture),
    OptimizeForGpuSliceLevel(Texture, u64, u64),
    OptimizeForCpu(Texture),
    OptimizeForCpuSliceLevel(Texture, u64, u64),
    CopyIndirectCommandBuffer(
        Box<(
            IndirectCommandBuffer,
            Range<u64>,
            IndirectCommandBuffer,
            u64,
        )>,
    ),
    ResetCommandsInBuffer(IndirectCommandBuffer, Range<u64>),
    OptimizeIndirectCommandBuffer(IndirectCommandBuffer, Range<u64>),
    UpdateFence(Fence),
    WaitForFence(Fence),
    InvalidateCompressedTexture(Texture),
    InvalidateCompressedTextureSlice(Texture, u64, u64),
    /// `copy_from_host_ram` or `copy_to_host_ram` (`host_ram/copy.rs`).
    HostRamCopy(Arc<HostRamCall>),
}

impl BlitCall {
    /// The bytes it holds in boxes of its own, besides its size.
    pub(super) fn extra(&self) -> usize {
        match self {
            BlitCall::FillTexture(call, _) => std::mem::size_of_val(&**call),
            BlitCall::FillTextureColor(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyBufferToTexture(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureToBuffer(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureToTexture(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyTextureSlices(call) => std::mem::size_of_val(&**call),
            BlitCall::CopyIndirectCommandBuffer(call) => std::mem::size_of_val(&**call),
            BlitCall::HostRamCopy(call) => std::mem::size_of_val(&**call) + call.extra(),
            _ => 0,
        }
    }
}

/// `copy_from_buffer_to_texture`'s arguments.
pub(super) struct BufferToTexture {
    pub(super) source: Buffer,
    pub(super) source_offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
    pub(super) size: Size,
    pub(super) destination: Texture,
    pub(super) slice: u64,
    pub(super) level: u64,
    pub(super) origin: Origin,
    pub(super) options: BlitOption,
}

/// `copy_from_texture_to_buffer`'s arguments.
pub(super) struct TextureToBuffer {
    pub(super) source: Texture,
    pub(super) slice: u64,
    pub(super) level: u64,
    pub(super) origin: Origin,
    pub(super) size: Size,
    pub(super) destination: Buffer,
    pub(super) destination_offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
    pub(super) options: BlitOption,
}

/// `copy_from_texture_to_texture`'s arguments.
pub(super) struct TextureToTexture {
    pub(super) source: Texture,
    pub(super) source_slice: u64,
    pub(super) source_level: u64,
    pub(super) source_origin: Origin,
    pub(super) source_size: Size,
    pub(super) destination: Texture,
    pub(super) destination_slice: u64,
    pub(super) destination_level: u64,
    pub(super) destination_origin: Origin,
}

/// An ended encoder's log, which encodes it again (the module docs).
pub(super) enum Replay {
    Compute {
        dispatch_type: DispatchType,
        log: CallLog<ComputeCall>,
    },
    Blit(CallLog<BlitCall>),
    Render(Box<RenderLog>),
}

impl Replay {
    /// The bytes its log holds.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Replay::Compute { log, .. } => log.bytes(),
            Replay::Blit(log) => log.bytes(),
            Replay::Render(log) => log.bytes(),
        }
    }

    /// The kind of its encoder.
    pub(super) fn kind(&self) -> EncoderKind {
        match self {
            Replay::Compute { .. } => EncoderKind::Compute,
            Replay::Blit(_) => EncoderKind::Blit,
            Replay::Render(_) => EncoderKind::Render,
        }
    }

    /// Encode its encoder again into `command_buffer`, committed and not yet first submitted,
    /// starting with `residency` as the encoder started (`CommandBuffer::inherited_residency`,
    /// of the encoders ended before it): what it recorded, and (a compute encoder's) what it
    /// made resident. Its calls announce no kind announced before (the module docs). Refused
    /// where a call answers otherwise than at its first encoding ([`CallLog::again`]), or where
    /// it records otherwise than its first encoding did (`EncoderCore::finish_replay`).
    pub(super) fn run(
        &self,
        command_buffer: &CommandBuffer,
        residency: Residency,
    ) -> Result<(Replayed, Option<Residency>), Error> {
        replaying(|| match self {
            Replay::Compute { dispatch_type, log } => {
                let mut encoder =
                    ComputeCommandEncoder::replaying(command_buffer, *dispatch_type, residency)?;
                for (at, call) in log.calls.iter().enumerate() {
                    log.again(at, compute(&mut encoder, log, call))?;
                }
                encoder.finish_replay()
            }
            Replay::Blit(log) => {
                let mut encoder = BlitCommandEncoder::replaying(command_buffer)?;
                for (at, call) in log.calls.iter().enumerate() {
                    log.again(at, blit(&mut encoder, log, call))?;
                }
                Ok((encoder.finish_replay()?, None))
            }
            Replay::Render(log) => log.run(command_buffer, residency),
        })
    }
}

thread_local! {
    /// The encoders this thread is encoding again now ([`replaying`]).
    static REPLAYING: Cell<usize> = const { Cell::new(0) };
}

/// Run `f`, which encodes an encoder again, or records again work its commit recorded
/// (`command_buffer/reencode.rs`), marked ([`is_replaying`]).
pub(crate) fn replaying<T>(f: impl FnOnce() -> T) -> T {
    /// Ends the mark as `f` returns or unwinds.
    struct Mark;
    impl Drop for Mark {
        fn drop(&mut self) {
            REPLAYING.with(|count| count.set(count.get() - 1));
        }
    }
    REPLAYING.with(|count| count.set(count.get() + 1));
    let _mark = Mark;
    f()
}

/// Whether this thread is encoding an encoder again now, or recording again work a commit
/// recorded: its notices were announced as the program encoded it, or as it committed
/// (`Device::degrade`).
pub(crate) fn is_replaying() -> bool {
    REPLAYING.with(Cell::get) > 0
}

/// Run `call` again on `encoder`, its slices in `log`'s arenas.
fn compute(
    encoder: &mut ComputeCommandEncoder,
    log: &CallLog<ComputeCall>,
    call: &ComputeCall,
) -> Result<(), Error> {
    use ComputeCall as C;
    match call {
        C::PipelineState(state) => encoder.set_compute_pipeline_state(state),
        C::Buffer(buffer, offset, index) => encoder.set_buffer(buffer.as_ref(), *offset, *index),
        C::Buffers(buffers, offsets, first) => {
            let buffers = log.buffers_of(*buffers);
            encoder.set_buffers(&buffers, log.words_of(*offsets), *first)
        }
        C::BufferOffset(offset, index) => encoder.set_buffer_offset(*offset, *index),
        C::BufferWithStride(buffer, offset, stride, index) => {
            encoder.set_buffer_with_stride(buffer.as_ref(), *offset, *stride, *index)
        }
        C::BuffersWithStrides(buffers, offsets, strides, first) => {
            let buffers = log.buffers_of(*buffers);
            let (offsets, strides) = (log.words_of(*offsets), log.words_of(*strides));
            encoder.set_buffers_with_strides(&buffers, offsets, strides, *first)
        }
        C::BufferOffsetWithStride(offset, stride, index) => {
            encoder.set_buffer_offset_with_stride(*offset, *stride, *index)
        }
        C::Bytes(bytes, index) => encoder.set_bytes(log.bytes_of(*bytes), *index),
        C::BytesWithStride(bytes, stride, index) => {
            encoder.set_bytes_with_stride(log.bytes_of(*bytes), *stride, *index)
        }
        C::FunctionTable(table, index) => {
            encoder.set_visible_function_table(table.as_ref(), *index)
        }
        C::FunctionTables(tables, first) => {
            let tables: Vec<_> = tables.iter().map(Option::as_ref).collect();
            encoder.set_visible_function_tables(&tables, *first)
        }
        C::Texture(texture, index) => encoder.set_texture(texture.as_ref(), *index),
        C::Textures(textures, first) => encoder.set_textures(&log.textures_of(*textures), *first),
        C::Sampler(sampler, index) => encoder.set_sampler_state(sampler.as_ref(), *index),
        C::Samplers(samplers, first) => {
            let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
            encoder.set_sampler_states(&samplers, *first)
        }
        C::SamplerWithLodClamps(sampler, min, max, index) => {
            encoder.set_sampler_state_with_lod_clamps(sampler.as_ref(), *min, *max, *index)
        }
        C::SamplersWithLodClamps(call) => {
            let samplers: Vec<_> = call.samplers.iter().map(Option::as_ref).collect();
            encoder.set_sampler_states_with_lod_clamps(
                &samplers,
                &call.mins,
                &call.maxes,
                call.first,
            )
        }
        C::ThreadgroupMemoryLength(length, index) => {
            encoder.set_threadgroup_memory_length(*length, *index)
        }
        C::ImageblockSize(width, height) => encoder.set_imageblock_size(*width, *height),
        C::StageInRegion(region) => encoder.set_stage_in_region(*region),
        C::StageInRegionIndirect(buffer, offset) => {
            encoder.set_stage_in_region_indirect(buffer, *offset)
        }
        C::Threadgroups(groups, threads) => encoder.dispatch_threadgroups(*groups, *threads),
        C::Threads(threads, per_group) => encoder.dispatch_threads(*threads, *per_group),
        C::ThreadgroupsIndirect(buffer, offset, threads) => {
            encoder.dispatch_threadgroups_indirect(buffer, *offset, *threads)
        }
        C::ThreadsIndirect(buffer, offset) => encoder.dispatch_threads_indirect(buffer, *offset),
        C::BarrierWithScope(scope) => {
            encoder.memory_barrier_with_scope(*scope);
            Ok(())
        }
        C::BarrierWithResources(resources) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.memory_barrier_with_resources(&resources)
        }
        C::UseResource(resource, usage) => encoder.use_resource(resource.as_ref(), *usage),
        C::UseResources(resources, usage) => {
            let resources: Vec<_> = resources.iter().map(Resource::as_ref).collect();
            encoder.use_resources(&resources, *usage)
        }
        C::UseHeap(heap) => encoder.use_heap(heap),
        C::UseHeaps(heaps) => {
            let heaps: Vec<&Heap> = heaps.iter().collect();
            encoder.use_heaps(&heaps)
        }
        C::Execute(icb, range) => encoder.execute_commands_in_buffer(icb, range.clone()),
        C::ExecuteIndirect(icb, buffer, offset) => {
            encoder.execute_commands_in_buffer_indirect(icb, buffer, *offset)
        }
        C::UpdateFence(fence) => encoder.update_fence(fence),
        C::WaitForFence(fence) => encoder.wait_for_fence(fence),
        C::StartIf(buffer, offset, comparison, reference) => {
            encoder.encode_start_if(buffer.as_ref(), *offset, *comparison, *reference)
        }
        C::StartElse => encoder.encode_start_else(),
        C::EndIf(answer) => match encoder.encode_end_if()? == *answer {
            true => Ok(()),
            false => Err(internal(
                "an encodeEndIf answered otherwise as it was encoded again",
            )),
        },
    }
}

/// Run `call` again on `encoder`, its slices in `log`'s arenas.
fn blit(
    encoder: &mut BlitCommandEncoder,
    log: &CallLog<BlitCall>,
    call: &BlitCall,
) -> Result<(), Error> {
    use BlitCall as B;
    match call {
        B::CopyBufferToBuffer(source, source_offset, destination, destination_offset, size) => {
            encoder.copy_from_buffer_to_buffer(
                source,
                *source_offset,
                destination,
                *destination_offset,
                *size,
            )
        }
        B::FillBuffer(buffer, range, value) => encoder.fill_buffer(buffer, range.clone(), *value),
        B::FillBufferPattern4(buffer, range, pattern) => {
            encoder.fill_buffer_pattern4(buffer, range.clone(), *pattern)
        }
        B::SynchronizeResource(resource) => encoder.synchronize_resource(resource.as_ref()),
        B::FillTexture(call, pattern) => {
            let (texture, level, slice, region) = &**call;
            encoder.fill_texture(texture, *level, *slice, *region, log.bytes_of(*pattern))
        }
        B::FillTextureColor(call) => {
            let (texture, level, slice, region, color) = &**call;
            encoder.fill_texture_color(texture, *level, *slice, *region, *color, None)
        }
        B::CopyBufferToTexture(call) => encoder.copy_from_buffer_to_texture(
            &call.source,
            call.source_offset,
            call.bytes_per_row,
            call.bytes_per_image,
            call.size,
            &call.destination,
            call.slice,
            call.level,
            call.origin,
            call.options,
        ),
        B::CopyTextureToBuffer(call) => encoder.copy_from_texture_to_buffer(
            &call.source,
            call.slice,
            call.level,
            call.origin,
            call.size,
            &call.destination,
            call.destination_offset,
            call.bytes_per_row,
            call.bytes_per_image,
            call.options,
        ),
        B::CopyTextureToTexture(call) => encoder.copy_from_texture_to_texture(
            &call.source,
            call.source_slice,
            call.source_level,
            call.source_origin,
            call.source_size,
            &call.destination,
            call.destination_slice,
            call.destination_level,
            call.destination_origin,
        ),
        B::CopyTextureSlices(call) => {
            let (source, source_slice, source_level, destination, slice, level, slices, levels) =
                &**call;
            encoder.copy_texture_slices(
                source,
                *source_slice,
                *source_level,
                destination,
                *slice,
                *level,
                *slices,
                *levels,
            )
        }
        B::CopyTexture(source, destination) => encoder.copy_texture(source, destination),
        B::GenerateMipmaps(texture) => encoder.generate_mipmaps(texture),
        B::SynchronizeTexture(texture, slice, level) => {
            encoder.synchronize_texture(texture, *slice, *level)
        }
        B::OptimizeForGpu(texture) => encoder.optimize_contents_for_gpu_access(texture),
        B::OptimizeForGpuSliceLevel(texture, slice, level) => {
            encoder.optimize_contents_for_gpu_access_slice_level(texture, *slice, *level)
        }
        B::OptimizeForCpu(texture) => encoder.optimize_contents_for_cpu_access(texture),
        B::OptimizeForCpuSliceLevel(texture, slice, level) => {
            encoder.optimize_contents_for_cpu_access_slice_level(texture, *slice, *level)
        }
        B::CopyIndirectCommandBuffer(call) => {
            let (source, range, destination, index) = &**call;
            encoder.copy_indirect_command_buffer(source, range.clone(), destination, *index)
        }
        B::ResetCommandsInBuffer(icb, range) => {
            encoder.reset_commands_in_buffer(icb, range.clone())
        }
        B::OptimizeIndirectCommandBuffer(icb, range) => {
            encoder.optimize_indirect_command_buffer(icb, range.clone())
        }
        B::UpdateFence(fence) => encoder.update_fence(fence),
        B::WaitForFence(fence) => encoder.wait_for_fence(fence),
        B::InvalidateCompressedTexture(texture) => encoder.invalidate_compressed_texture(texture),
        B::InvalidateCompressedTextureSlice(texture, slice, level) => {
            encoder.invalidate_compressed_texture_slice(texture, *slice, *level)
        }
        B::HostRamCopy(call) => encoder.host_ram_copy(call),
    }
}
