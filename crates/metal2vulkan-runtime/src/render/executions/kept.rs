//! An execution of render commands, recorded chunk by chunk so a later one runs the chunks no
//! write reached again (`kept_commands.rs`): of the records the CPU encoded, at `commit`; and of
//! those the GPU may write, at the host step, where the records hold the bytes a kept chunk was
//! recorded from.
//!
//! - **Which executions.** One `commit` records from the CPU's records (no host step, no GPU
//!   form), of a range in the call, of a buffer whose command types hold a draw of primitives
//!   (`DRAW` or `DRAW_INDEXED`: a mesh or patch draw is never plain, so a buffer of those alone
//!   has no chunk to keep), with no check of the caller's that may see something among its
//!   commands (one whose needs rule out what the buffer's records may hold calls nothing at
//!   `commit` either, `RenderExecution::check_sees`), no visibility result counted, no shadowed
//!   texture, no tile threadgroup memory argument set or reached, no rasterization rate map, no
//!   memoryless attachment or stand-in, no record barrier the buffer may hold, no write before
//!   the fragments and no indirect read of the encoder's draws before the call, no fragment write
//!   of the pass noted before it, no inherited table that binds anything, in a command buffer
//!   with no executed-commands handler that is not encoded again after its commit;
//!   no chunk of its range last recorded otherwise than plainly; and a chunk of its range run
//!   before, no write reaching it since (`kept_commands.rs`, **From the second execution**): a
//!   range's first execution records as one that keeps nothing.
//! - **At the host step.** One a host step records from the records as the GPU and the CPU left
//!   them, of a buffer a route that may write its records met (`IndirectCommandBuffer::gpu_written`),
//!   under the same rules but three: it inherits the encoder's pipeline state (no record names
//!   one); its inherited tables bind buffers alone (no bytes, texture, sampler or visible-function
//!   table), which its key holds; and a record barrier the buffer may hold makes the chunk it lies
//!   in not plain, as below, rather than ruling the execution out. Its chunks are its own, and
//!   a step keeps them only where its records hold, chunk by chunk, the bytes a kept recording of
//!   the chunk was recorded from or else those the last step read, compared whole
//!   (`RenderExecution::steady`, `IndirectCommandBuffer::holds_written`,
//!   `WorkCounts::host_step_records_compared` and `host_step_compare_nanos`), once the header
//!   holds no fault. Else the step records the execution inline, as one that keeps nothing, from
//!   each chunk's records copied for the next step's comparison
//!   (`IndirectCommandBuffer::written_bytes_into`, [`StepReads`]), forgetting a recording whose
//!   records changed (`WorkCounts::host_step_chunks_recorded_again`): records the GPU writes
//!   otherwise at each step are read once, as an execution that keeps nothing reads them. Where
//!   they hold, the step records apart and keeps each
//!   chunk it keeps no recording of, with those bytes, and runs a kept one again where its key is
//!   equal (below), else records it again and keeps that (`host_step_chunks_recorded_again`). No
//!   write forgets such a chunk, as the comparison decides. Counted:
//!   `WorkCounts::host_step_chunks_kept`.
//! - **The key.** A chunk is recorded from its records and from the inputs [`RenderKey`] holds:
//!   the inherited pipeline state, the instance's compatible pass and shape, the pass's extent,
//!   layers, samples and sample positions, the attachments that are among the resources made
//!   resident, the encoder's state, scissor rectangles, render target array length and
//!   amplification at the call, whether the pass holds tile imageblock data, the buffers of the
//!   inherited tables, by key, with their offsets and attribute strides, and the resources made
//!   resident (`Residency::kept_key`, one list an execution, compared whole once a list its kept
//!   chunks hold: `kept_commands::ResidentMet`). A kept chunk runs again only under an equal key,
//!   where every address its commands named resolves as it did and no no-copy buffer moved
//!   (`KeptChunk::runs_again`).
//! - **Each chunk** runs its kept recording again, or records apart: its draws into a secondary of
//!   its own, begun against the instance's compatible pass, through an encoder of the call's
//!   state over a core of its own (`EncoderCore::capture`), which the pass's core then takes
//!   (`EncoderCore::absorb`); from a context of its own where an execution ran it before, no
//!   write reaching it since, else from its command buffer's own, kept by none. A host step's
//!   chunks draw in the large-buffer form (`BufferForms::Large`), as its other draws do. Its draws are
//!   plain where each one is (`RenderLowering::replay_plain`) and they left the pass's storage
//!   and layout as they found them in one run: such a chunk's secondary joins the others in the
//!   execution's first run, and is kept where it was recorded from a context of its own, its
//!   recording did nothing a later command buffer cannot do again (`Captured::plain`) and it
//!   reached nothing its command buffer alone holds (`CommandBuffer::own_reached`). Plain draws
//!   write no memory, so no draw of a later chunk depends on them but through the attachments,
//!   which the chunks' order in their subpass keeps.
//! - **A chunk that is not plain** is no more apart: its encoder records the rest of the range
//!   too, as the execution's would, its runs after the plain chunks' (its first sharing their
//!   run, as an execution's first run shares its instance with the encoder's draws before the
//!   call, which its encoder's state takes for granted), and the chunk is noted, so later
//!   executions of it record as before until a write reaches it; a host step forgets what the
//!   chunk kept instead, and records it apart again at its next step.

use super::{
    Asked, Command, Draw, ExecutionRange, Found, PassStorage, PointIndices, Recorded,
    RenderExecution, Rendered, Run, Visibility,
};
use crate::amplification::Amplification;
use crate::argument_table::ArgumentTable;
use crate::binding::BindMemo;
use crate::command_buffer::CommandBuffer;
use crate::encoder::EncoderCore;
use crate::error::{internal, vulkan, Error};
use crate::indirect_command_buffer::{IndirectCommandBuffer, Reader, WrittenRecords};
use crate::kept_commands::{
    self, Kept, KeptChunk, KeptContext, KeptEffects, ResidentKey, ResidentMet, Resolution,
    StepChunk,
};
use crate::live_bind::LiveBind;
use crate::pipeline::{BufferForms, RenderPipelineState};
use crate::render::{memoryless, RenderLowering};
use crate::render_pass::PassLayout;
use crate::render_state::{EncoderState, ScissorRects};
use crate::render_target::{PassShape, RenderPassObject};
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::sample_positions::Sixteenths;
use metal2vulkan::bind::indirect_commands::{BufferStage, CommandRecord};
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::IndirectCommandType;
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What a chunk of an execution's draws is recorded from beside its records (the module docs).
#[derive(Clone)]
pub(super) struct RenderKey {
    pipeline: Option<RenderPipelineState>,
    /// Held, so no other pass takes its `VkRenderPass`'s handle while the key lives: a key of the
    /// same handle names the same pass, which the device keeps for the shape (each encoder holds
    /// a pass object of its own over it).
    compatible: Arc<RenderPassObject>,
    shape: PassShape,
    extent: [u32; 2],
    layers: u32,
    samples: u32,
    sample_positions: Vec<Sixteenths>,
    attached: Vec<ResourceKey>,
    state: EncoderState,
    scissors: ScissorRects,
    render_target_array_length: u32,
    amplification: Amplification,
    tile_imageblock: bool,
    /// The inherited tables' buffers (vertex, fragment, object, mesh), none where the buffer does
    /// not inherit them.
    tables: [TableBuffers; 4],
    resident: ResidentKey,
}

/// An inherited table's buffers as a [`RenderKey`] holds them: each slot's buffer by its key,
/// with its bind offset and per-bind attribute stride, in slot order.
type TableBuffers = Vec<(u32, ResourceKey, u64, Option<u32>)>;

/// `table`'s buffers as a key holds them; `None` where it binds more than buffers (bytes, a
/// texture, a sampler, a visible-function table), which no kept chunk is recorded under.
fn table_buffers(table: &ArgumentTable) -> Option<TableBuffers> {
    let buffers_alone = table.bytes.is_empty()
        && table.textures.is_empty()
        && table.samplers.is_empty()
        && table.function_tables.is_empty()
        && table.null_tables.is_empty();
    buffers_alone.then(|| {
        (table.buffers.iter())
            .map(|(&index, (buffer, offset))| {
                let stride = table.attribute_strides.get(&index).copied();
                (index, buffer.identity().key, *offset, stride)
            })
            .collect()
    })
}

impl PartialEq for RenderKey {
    fn eq(&self, other: &RenderKey) -> bool {
        self.pipeline == other.pipeline
            && self.compatible.raw == other.compatible.raw
            && self.shape == other.shape
            && self.extent == other.extent
            && self.layers == other.layers
            && self.samples == other.samples
            && self.sample_positions == other.sample_positions
            && self.attached == other.attached
            && self.state.same_values(&other.state)
            && self.scissors == other.scissors
            && self.render_target_array_length == other.render_target_array_length
            && self.amplification == other.amplification
            && self.tile_imageblock == other.tile_imageblock
            && self.tables == other.tables
            && self.resident == other.resident
    }
}

/// An execution that keeps its commands (the module docs): its key, what is kept of its range's
/// chunks in order, with the bytes of each one's records a host step read last (`None` for one
/// it keeps a recording of, and at `commit`), and the CPU's writes of the buffer as they were
/// looked up (`None` for a host step's, which compares the records instead).
pub(super) struct Keeping {
    key: RenderKey,
    kept: Vec<(Kept, Option<Box<[u8]>>)>,
    read_at: Option<u64>,
}

/// The bind state, bind memo and residency a chunk recorded apart starts from (`live_bind.rs`,
/// `binding::BindMemo`): those the chunk before it left, the first chunk's the device's spares'
/// and a copy of the command buffer's residency. An execution's chunks share one residency, so a
/// chunk's first draw lists its used resources again only where the one before it did not list
/// them (a render encoder's first draw lists the residency's whole: thousands of resources), and
/// the residency is copied once an execution, as an execution recorded inline copies it once.
type Carried = (Arc<Mutex<LiveBind>>, BindMemo, Residency);

/// A chunk recorded apart: its encoder after its commands, the addresses they named and what
/// they resolved to, the no-copy buffers' moves before it read its records, whether a command
/// of it named a released state, and for a host step the bytes of its records.
struct Apart {
    encoder: RenderLowering,
    resolved: Vec<Resolution>,
    moved: u64,
    released: bool,
    records: Option<Box<[u8]>>,
}

/// Whether an execution keeps its commands ([`RenderExecution::keeping`]).
pub(super) enum Keeps {
    /// It does: its key and its chunks.
    Kept(Box<Keeping>),
    /// It records inline, as one that keeps nothing, at a host step that keeps its chunks from
    /// its records read as [`StepReads`] says, where it gives some.
    Inline(Option<StepReads>),
}

/// What a host step that records inline reads its records into, chunk by chunk
/// ([`RenderExecution::steady`]): the copy the next step compares them with, which the step's
/// recording reads too, so records that change at each step are read once.
pub(super) struct StepReads(Vec<StepRead>);

/// A chunk's part of [`StepReads`].
struct StepRead {
    chunk: u32,
    covered: Range<u32>,
    from: ReadFrom,
}

/// Where a chunk of [`StepReads`] reads its records.
enum ReadFrom {
    /// Copied from the records' storage, into the bytes the last step read where it has them.
    Copy(Option<Box<[u8]>>),
    /// The bytes its kept recording was recorded from, which the records hold still.
    Kept(Arc<KeptChunk>),
}

/// A host step's copies of its chunks' records ([`StepReads::read`]), for the next step's
/// comparison once it recorded from them.
pub(super) struct ReadRecords(Vec<(u32, Box<[u8]>)>);

impl StepReads {
    /// The commands it reads.
    pub(super) fn range(&self) -> Range<u32> {
        let start = self.0.first().map_or(0, |read| read.covered.start);
        start..self.0.last().map_or(start, |read| read.covered.end)
    }

    /// The records of its commands, in order, as the GPU and the CPU left them, read once: each
    /// chunk's copied (`IndirectCommandBuffer::written_bytes_into`), else its kept recording's,
    /// and decoded from those bytes (`IndirectCommandBuffer::written_records_in`); with the
    /// copies.
    pub(super) fn read(
        self,
        icb: &IndirectCommandBuffer,
    ) -> Result<(WrittenRecords, ReadRecords), Error> {
        let range = self.range();
        let read: Vec<(u32, ReadBytes)> = (self.0.into_iter())
            .map(|read| {
                let bytes = match read.from {
                    ReadFrom::Copy(into) => {
                        ReadBytes::Copied(icb.written_bytes_into(read.covered, into))
                    }
                    ReadFrom::Kept(kept) => ReadBytes::Kept(kept),
                };
                (read.chunk, bytes)
            })
            .collect();
        let bytes: Vec<&[u8]> = read.iter().map(|(_, bytes)| bytes.bytes()).collect();
        let written = icb.written_records_in(range, &bytes)?;
        let copies = read.into_iter().filter_map(|(chunk, bytes)| match bytes {
            ReadBytes::Copied(copy) => Some((chunk, copy)),
            ReadBytes::Kept(_) => None,
        });
        Ok((written, ReadRecords(copies.collect())))
    }
}

/// A chunk's records as [`StepReads::read`] read them.
enum ReadBytes {
    /// Copied from their storage.
    Copied(Box<[u8]>),
    /// Its kept recording's, which they hold still.
    Kept(Arc<KeptChunk>),
}

impl ReadBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            ReadBytes::Copied(copy) => copy,
            // A kept recording a step reads from holds them (`RenderExecution::steady`); one that
            // held none would read as no record, which the decode refuses.
            ReadBytes::Kept(kept) => kept.records().unwrap_or_default(),
        }
    }
}

impl ReadRecords {
    /// Keep each copy as the bytes its chunk's records were last read as
    /// (`IndirectCommandBuffer::step_read`).
    pub(super) fn keep(self, icb: &IndirectCommandBuffer) {
        for (chunk, bytes) in self.0 {
            icb.step_read(chunk, bytes);
        }
    }
}

impl RenderExecution {
    /// Whether the execution keeps its commands (the module docs), recorded by `command_buffer`
    /// against `residency` in a pass `rendered` lays out, at `commit` or at the host step
    /// (`step`): its key and its chunks.
    pub(super) fn keeping(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
        rendered: &Rendered,
        step: bool,
    ) -> Keeps {
        let snapshot = &self.snapshot;
        let layout = &snapshot.layout;
        let ExecutionRange::Fixed(range) = &self.range else {
            return Keeps::Inline(None);
        };
        let inherited = self.icb.descriptor().inherit_buffers;
        let tables = [
            &snapshot.vertex,
            &snapshot.fragment,
            &snapshot.object,
            &snapshot.mesh,
        ];
        let binds_nothing = tables.iter().all(|table| table.binds_nothing());
        // The inherited tables' buffers, which the key holds; none where the buffer does not
        // inherit them.
        let [vertex, fragment, object, mesh] = tables.map(|table| match inherited {
            true => table_buffers(table),
            false => Some(Vec::new()),
        });
        let colors = layout.colors.iter().flatten();
        let plain_colors = colors
            .clone()
            .all(|color| color.stand_in.is_none() && color.planes.is_none());
        let memoryless_target = colors.clone().any(|color| memoryless(&color.target))
            || (layout.depth_stencil.as_ref())
                .is_some_and(|ds| ds.stand_in.is_some() || memoryless(&ds.target));
        // A mesh or patch draw is never plain (`RenderLowering::replay_plain`), so a buffer whose
        // commands can be no draw of primitives keeps no chunk: recording one apart would only
        // find it not plain.
        let primitives = IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        // A host step's: of a buffer the GPU may write, whose commands inherit the pipeline
        // state, where a barrier makes its chunk not plain; `commit`'s: of the CPU's records alone.
        let keeps = !range.is_empty()
            && self.icb.descriptor().command_types.intersects(primitives)
            && self.icb.gpu_written() == step
            && (!step || self.pipeline.is_some())
            && !self.check_sees_now()
            && self.lent.is_none()
            && layout.visibility.is_none()
            && self.shadows.is_empty()
            && (step || !self.icb.may_hold_barriers())
            && !snapshot.tile_memory.set()
            && !snapshot.tile_memory_reached
            && snapshot.rate_map.is_none()
            && snapshot.geometry_writes.is_empty()
            && snapshot.indirect_reads.is_empty()
            && !snapshot.pass_start.fragment_wrote()
            && (step || !inherited || binds_nothing)
            && rendered.backed.is_empty()
            && plain_colors
            && !memoryless_target
            && !crate::replay::is_replaying()
            && !command_buffer.reports_executed();
        let (true, [Some(vertex), Some(fragment), Some(object), Some(mesh)]) =
            (keeps, [vertex, fragment, object, mesh])
        else {
            return Keeps::Inline(None);
        };
        let tables = [vertex, fragment, object, mesh];
        let (kept, read_at) = match step {
            // Where the header holds no fault: one that does runs no command.
            true if self.icb.written_header_fault().is_some() => return Keeps::Inline(None),
            true => {
                let counts = &command_buffer.device().shared().counts;
                match self.steady(range.clone(), counts) {
                    Ok(kept) => (kept, None),
                    Err(reads) => return Keeps::Inline(Some(reads)),
                }
            }
            false => {
                let Some((kept, read_at)) = self.icb.kept_chunks(range.clone()) else {
                    return Keeps::Inline(None);
                };
                let kept = kept.into_iter().map(|kept| (kept, None)).collect();
                (kept, Some(read_at))
            }
        };
        let resident = ResidentKey::new(residency.kept_key());
        // An attachment is among them as a texture made resident, or in a heap or over a buffer
        // made resident.
        let keys: BTreeSet<ResourceKey> = resident.keys().iter().copied().collect();
        let attached = colors
            .map(|color| &color.target.texture)
            .chain(layout.depth_stencil.as_ref().map(|ds| &ds.target.texture))
            .filter(|texture| {
                keys.contains(&texture.use_key())
                    || texture.bytes_key().is_some_and(|key| keys.contains(&key))
            })
            .map(|texture| texture.use_key())
            .collect();
        let key = RenderKey {
            pipeline: self.pipeline.clone(),
            compatible: Arc::clone(&snapshot.compatible),
            shape: snapshot.shape.clone(),
            extent: layout.extent,
            layers: layout.layers,
            samples: layout.samples,
            sample_positions: layout.sample_positions.clone(),
            attached,
            state: snapshot.state.clone(),
            scissors: snapshot.scissors.clone(),
            render_target_array_length: snapshot.render_target_array_length,
            amplification: snapshot.amplification.clone(),
            tile_imageblock: snapshot.tile_imageblock,
            tables,
            resident,
        };
        Keeps::Kept(Box::new(Keeping { key, kept, read_at }))
    }

    /// What a host step that keeps its commands finds of each chunk of commands `range`, where
    /// the records hold, chunk by chunk, the bytes a kept recording of the chunk was recorded from
    /// or else the bytes of them the last step read, compared whole
    /// (`IndirectCommandBuffer::holds_written`, `WorkCounts::host_step_records_compared`): what is
    /// kept of each, with those bytes it read. Else, the records changing (or read by no step
    /// before), what the execution reads its records into as it records inline, as one that
    /// keeps nothing ([`StepReads`]): each chunk's into the bytes it last read, for the next
    /// step's comparison, but where a kept recording's records hold still; and a recording whose
    /// records changed is forgotten (`WorkCounts::host_step_chunks_recorded_again`). So a step
    /// records apart and keeps only what two steps read alike, and records whose bytes change at
    /// each step are read once, into the copy the next step compares with.
    fn steady(
        &self,
        range: Range<u32>,
        counts: &crate::work_counts::Counters,
    ) -> Result<Vec<(Kept, Option<Box<[u8]>>)>, StepReads> {
        let found = self.icb.step_kept_chunks(range.clone());
        let compared = Instant::now();
        let mut records = 0;
        let holds: Vec<bool> = (kept_commands::chunks(range.clone()).zip(&found))
            .map(|((_, covered), found)| {
                let bytes = match &found.kept {
                    Kept::Recording(kept) => kept.records(),
                    Kept::Again | Kept::First => found.read.as_deref(),
                };
                bytes.is_some_and(|bytes| {
                    records += u64::from(covered.end - covered.start);
                    self.icb.holds_written(covered.clone(), bytes)
                })
            })
            .collect();
        counts.host_step_records_compared(records, compared.elapsed());
        if holds.iter().all(|&holds| holds) {
            let found = found.into_iter().map(|found| (found.kept, found.read));
            return Ok(found.collect());
        }
        let chunks = kept_commands::chunks(range).zip(found).zip(holds);
        let reads = chunks.map(|(((chunk, covered), StepChunk { kept, read }), holds)| {
            let from = match kept {
                Kept::Recording(kept) if holds && kept.records().is_some() => ReadFrom::Kept(kept),
                Kept::Recording(_) => {
                    counts.host_step_chunk_recorded_again();
                    self.icb.keep_step_chunk(chunk, None);
                    ReadFrom::Copy(read)
                }
                Kept::Again | Kept::First => ReadFrom::Copy(read),
            };
            StepRead {
                chunk,
                covered,
                from,
            }
        });
        Err(StepReads(reads.collect()))
    }

    /// Record the commands chunk by chunk through `core`, the pass's (the module docs), into
    /// secondaries begun against the execution's instance, over the pass's `storage`, in a pass
    /// `rendered` lays out, with `residency`, `command_buffer`'s: `record`'s, where the execution
    /// keeps its commands (`keeping`).
    pub(super) fn record_kept(
        &self,
        mut core: EncoderCore,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        keeping: Keeping,
        found: &mut Found,
        (storage, rendered): (PassStorage, &mut Rendered),
    ) -> Result<Recorded, Error> {
        let device = command_buffer.device().clone();
        let (range, faults) = self.range.commands(self.icb.size())?;
        found.faults = ReplayFaults(found.faults.0 | faults.0);
        let Keeping {
            mut key,
            kept,
            read_at,
        } = keeping;
        // Counted before the records are read: a write meanwhile records it again. A host step's
        // records are compared instead.
        let step = read_at.is_none();
        if let Some(read_at) = read_at {
            self.read_at.store(read_at, Ordering::Release);
        }
        let layout = rendered.backing(&self.snapshot.layout)?;
        let shared = device.shared();
        let counts = &shared.counts;
        let carried = (shared.live_binds.take(), shared.bind_memos.take());
        let mut carried = Some((carried.0, carried.1, residency.clone()));
        let mut met = ResidentMet::default();
        let mut draws = Vec::new();
        let mut rest = None;
        for ((chunk, covered), (kept, read)) in kept_commands::chunks(range.clone()).zip(kept) {
            let (kept, keeps) = match kept {
                Kept::Recording(kept) => (Some(kept), true),
                Kept::Again => (None, true),
                Kept::First => (None, false),
            };
            let had = kept.is_some();
            // A host step's records hold the bytes it was recorded from (`steady`).
            let again = kept.and_then(|kept| {
                let resident = &kept.key::<RenderKey>()?.resident;
                if !met.same(&mut key.resident, resident, counts) {
                    return None;
                }
                let held = kept.runs_again(command_buffer, (&key, &covered), residency)?;
                Some((kept, held))
            });
            if let Some((kept, held)) = again {
                core.run_kept(&kept.effects, held)?;
                found.misses = found.misses.wrapping_add(kept.misses);
                found.faults = ReplayFaults(found.faults.0 | kept.faults.0);
                draws.push(kept.raw);
                match step {
                    true => counts.host_step_chunk_kept(),
                    false => counts.indirect_chunk_kept(u64::from(covered.end - covered.start)),
                }
                core.retain_shared(kept);
                continue;
            }
            if step && had {
                counts.host_step_chunk_recorded_again();
            }
            // A chunk's first execution records from its command buffer's own.
            let context = match keeps {
                true => Some(KeptContext::new(command_buffer)?),
                false => None,
            };
            if let Some(context) = &context {
                core.retain_shared(Arc::clone(context) as _);
                command_buffer.lend_kept(Some(context));
            }
            let reached = command_buffer.own_reached();
            let mut chunk_found = Found::default();
            let start = match carried.take() {
                Some(carried) => carried,
                None => (Arc::default(), BindMemo::default(), residency.clone()),
            };
            // A host step's from the bytes of the records it last read, as they stand now
            // (`steady`), else from a copy of them.
            let records =
                step.then(|| read.unwrap_or_else(|| self.icb.written_bytes(covered.clone())));
            let apart = self.record_apart(
                (command_buffer, residency),
                (covered.clone(), records),
                (&storage, &layout, start),
                &mut chunk_found,
            );
            if context.is_some() {
                command_buffer.lend_kept(None);
            }
            let Apart {
                encoder,
                resolved,
                moved,
                released,
                records,
            } = apart?;
            let plain = encoder.replay_plain
                && encoder.earlier.is_empty()
                && same_storage(&storage, &encoder)
                && same_layout(&layout, &encoder.layout);
            if !plain {
                self.keep_chunk(chunk, None, read_at);
                rest = Some((encoder, covered.end, chunk_found));
                break;
            }
            let raw = encoder.current.draws;
            let (capture, back) = self.end_apart(encoder, &device)?;
            carried = Some(back);
            let captured = capture.finish_capture()?;
            let held = captured.retained.all_shared();
            let keeps = captured.plain() && !released && command_buffer.own_reached() == reached;
            if let (true, Some(held), Some(context)) = (keeps, held, context) {
                let effects = KeptEffects::new(
                    captured.summary.clone(),
                    (captured.keys.clone(), captured.taken.named().clone()),
                    &held,
                    captured.executions.clone(),
                );
                let kept = KeptChunk::new(
                    (context, raw),
                    covered,
                    Box::new(key.clone()),
                    (resolved, moved),
                    effects,
                    (chunk_found.misses, chunk_found.faults),
                );
                let kept = match records {
                    Some(records) => kept.recorded_from(records),
                    None => kept,
                };
                self.keep_chunk(chunk, Some(Arc::new(kept)), read_at);
            }
            core.absorb(captured)?;
            found.misses = found.misses.wrapping_add(chunk_found.misses);
            found.faults = ReplayFaults(found.faults.0 | chunk_found.faults.0);
            draws.push(raw);
        }
        if let Some((live, memo, _)) = carried {
            shared.live_binds.give(live);
            shared.bind_memos.give(memo);
        }
        let Some((mut encoder, from, chunk_found)) = rest else {
            let run = Run {
                draws,
                other: None,
                shadow: false,
                steps: None,
                framed: Vec::new(),
                before: Vec::new(),
            };
            return Ok(Recorded {
                core,
                runs: vec![run],
                storage,
                layout,
            });
        };
        // The rest of the range, from the chunk that is not plain on, through its encoder, as the
        // execution's own records them.
        found.misses = found.misses.wrapping_add(chunk_found.misses);
        found.faults = ReplayFaults(found.faults.0 | chunk_found.faults.0);
        // A chunk at a time, as an execution recorded inline reads its range.
        for piece in kept_commands::pieces(from..range.end, false) {
            if step {
                let (commands, written) = self.written_commands(&device, piece, None)?;
                found.faults = ReplayFaults(found.faults.0 | written.0);
                self.replay(
                    &mut encoder,
                    commands.into_iter().flatten(),
                    residency,
                    found,
                )?;
                continue;
            }
            let (commands, released) = self.icb.render_records(piece, Reader::Commit)?;
            self.icb.degrade_released(&released);
            self.replay(&mut encoder, commands, residency, found)?;
        }
        let tail = self.finish(encoder, found, rendered, &device)?;
        let Recorded {
            core: capture,
            mut runs,
            storage,
            layout,
        } = tail;
        core.absorb(capture.finish_capture()?)?;
        let first = runs
            .first_mut()
            .ok_or_else(|| internal("an execution's rest of no runs"))?;
        draws.append(&mut first.draws);
        first.draws = draws;
        Ok(Recorded {
            core,
            runs,
            storage,
            layout,
        })
    }

    /// Record commands `covered` apart (the module docs): through an encoder of the call's state
    /// over a core of its own into a secondary of its own, begun against the instance's
    /// compatible pass, over the pass's `storage`, in `layout`, the call's, from the bind state
    /// and residency `carried` holds; what their draws found noted into `found`. From the CPU's
    /// records, or for a host step from `records`, the bytes of the records as the GPU and the CPU
    /// left them, which it keeps.
    fn record_apart(
        &self,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        (covered, records): (Range<u32>, Option<Box<[u8]>>),
        (storage, layout, carried): (&PassStorage, &PassLayout, Carried),
        found: &mut Found,
    ) -> Result<Apart, Error> {
        let snapshot = &self.snapshot;
        let device = command_buffer.device();
        let moved = crate::buffer::host_cells_moved();
        let (commands, released, records, forms) = match records {
            Some(bytes) => {
                let written = self.icb.decode_written(&bytes);
                let (commands, faults, released) =
                    self.named_commands(device, covered.start, written);
                found.faults = ReplayFaults(found.faults.0 | faults.0);
                let commands: Vec<Command> = commands.into_iter().flatten().collect();
                (commands, released, Some(bytes), BufferForms::Large)
            }
            None => {
                let (commands, released) = self.icb.render_records(covered, Reader::Commit)?;
                self.icb.degrade_released(&released);
                (commands, !released.is_empty(), None, BufferForms::Bound)
            }
        };
        let resolved = self.resolutions(&commands, residency);
        let draws = command_buffer.begin_draws(&snapshot.compatible)?;
        let raw = command_buffer.device().shared().vk();
        let visibility = Visibility::borrowing(None, None, raw, draws);
        let (live, bind_memo, carried) = carried;
        let mut encoder = self.replay_encoder(
            EncoderCore::capture(command_buffer),
            (draws, visibility),
            (storage.clone(), layout.clone(), carried),
            forms,
        );
        (encoder.live, encoder.bind_memo) = (live, bind_memo);
        self.replay(&mut encoder, commands, residency, found)?;
        Ok(Apart {
            encoder,
            resolved,
            moved,
            released,
            records,
        })
    }

    /// Keep `kept`, chunk `chunk`'s recording, or (`None`) note that it was not plain: among the
    /// CPU's records' chunks as the CPU's writes `read_at` left them, or (`None`) the host step's.
    fn keep_chunk(&self, chunk: u32, kept: Option<Arc<KeptChunk>>, read_at: Option<u64>) {
        match read_at {
            Some(read_at) => self.icb.keep_chunk(chunk, kept, read_at),
            None => self.icb.keep_step_chunk(chunk, kept),
        }
    }

    /// End a plain chunk's encoder: its secondary ended. Its core back, and the bind state and
    /// residency the next chunk starts from, its encoder ended (`LiveBind::end_encoder`).
    fn end_apart(
        &self,
        encoder: RenderLowering,
        device: &crate::Device,
    ) -> Result<(EncoderCore, Carried), Error> {
        let raw = device.shared().vk();
        let RenderLowering {
            core,
            current,
            mut visibility,
            mut live,
            bind_memo,
            residency,
            ..
        } = encoder;
        visibility.end(raw, current.draws);
        visibility.close_spares(raw, current.draws);
        // SAFETY: only this recording records the secondary command buffer.
        unsafe { core.command_buffer().end_raw(current.draws) }.map_err(vulkan)?;
        // No draw holds it once its encoder ended; one that did leaves the next chunk a new one.
        match Arc::get_mut(&mut live).map(Mutex::get_mut) {
            Some(Ok(state)) => state.end_encoder(),
            Some(Err(_)) | None => live = Arc::default(),
        }
        Ok((core, (live, bind_memo, residency)))
    }

    /// The addresses `commands` name (their buffers', where the buffer does not inherit them,
    /// and their draws'), resolved against `residency` (`kept_commands::resolutions`).
    fn resolutions(&self, commands: &[Command], residency: &Residency) -> Vec<Resolution> {
        let descriptor = self.icb.descriptor();
        let stages = [
            (BufferStage::Vertex, descriptor.max_vertex_buffer_bind_count),
            (
                BufferStage::Fragment,
                descriptor.max_fragment_buffer_bind_count,
            ),
            (BufferStage::Object, descriptor.max_object_buffer_bind_count),
            (BufferStage::Mesh, descriptor.max_mesh_buffer_bind_count),
        ];
        let addresses = commands.iter().flat_map(|(record, _)| {
            let bound = stages
                .iter()
                .filter(|_| !descriptor.inherit_buffers)
                .flat_map(|&(stage, count)| {
                    let bindings = record.bindings.buffers(stage);
                    (0..count).zip(bindings).filter_map(|(_, binding)| *binding)
                })
                .filter(|binding| binding.names_buffer())
                .map(|binding| binding.address);
            bound
                .chain(addresses(record, &self.icb))
                .collect::<Vec<_>>()
        });
        kept_commands::resolutions(addresses, residency)
    }
}

/// The addresses a command's draw names: its index buffer's, and a patch draw's buffers'.
fn addresses(
    record: &CommandRecord,
    icb: &crate::indirect_command_buffer::IndirectCommandBuffer,
) -> Vec<u64> {
    let Asked::Draw(draw) = Asked::of(&record.action, icb) else {
        return Vec::new();
    };
    match draw {
        Draw::Primitives(..) | Draw::Mesh(..) => Vec::new(),
        Draw::Indexed(_, address) => vec![address],
        Draw::Patches {
            patch_indices,
            control_point_indices,
            factors,
            ..
        } => {
            let points = match control_point_indices {
                PointIndices::Buffer(address) => Some(address),
                PointIndices::Own | PointIndices::Zero => None,
            };
            patch_indices
                .into_iter()
                .chain(points)
                .chain([factors.buffer])
                .collect()
        }
    }
}

/// Whether a chunk's `encoder` left the pass's storage as `storage` held it: it made none. (No
/// draw of an execution that keeps its commands reaches the tile threadgroup memory.)
fn same_storage(storage: &PassStorage, encoder: &RenderLowering) -> bool {
    storage.imageblock.is_some() == encoder.imageblock.is_some()
        && storage.shadow.is_some() == encoder.shadow.is_some()
}

/// Whether a chunk's draws left the pass's layout as `before` held it: no stand-in or sample
/// planes made.
fn same_layout(before: &PassLayout, after: &PassLayout) -> bool {
    let colors = |layout: &PassLayout| {
        (layout.colors.iter())
            .map(|color| {
                color
                    .as_ref()
                    .map(|c| (c.stand_in.is_some(), c.planes.is_some()))
            })
            .collect::<Vec<_>>()
    };
    let depth =
        |layout: &PassLayout| (layout.depth_stencil.as_ref()).map(|ds| ds.stand_in.is_some());
    colors(before) == colors(after) && depth(before) == depth(after)
}
