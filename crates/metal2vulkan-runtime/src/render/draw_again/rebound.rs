//! A render encoder's kept draw drawn again after calls that bound other buffers, offsets, textures
//! or sampler states at slots of its tables (`WorkCounts::draws_rebound`): its every other step is
//! the kept draw's (`render/draw_again.rs`), so only what those slots decide is worked out again.
//!
//! - **Which draw.** The kept draw again ([`RenderLowering::draw_again`]: the same call and
//!   arguments, under the same `Under`, into the same command buffer) after calls that changed only
//!   buffer, texture and sampler slots of its tables, each table's changes since the kept draw read
//!   it still listed (`argument_table::Changes`, at most [`SLOTS`] slots), where the kept draw kept
//!   what this takes ([`KeptStorage`]: a draw whose set-0 writes hold no keeper, of no use beyond
//!   its plan's and its draw's own, and of a state that made no visible-function table). The kept
//!   bind state takes the changed slots (`ArgumentTable::refresh_changed`), and the draw goes on
//!   only where its shape stays, and its runtime samplers' address modes (`address_map.rs`) with
//!   it: its variant, plan, pipeline and draw state are then the kept draw's, functions of the
//!   shape and of what no call changed.
//! - **What it works out again.** Its commands, rebound at those slots
//!   (`BindingPlan::rebind_draw_slots_into`: their descriptor ranges, push fields and vertex
//!   buffers' offsets); set 0's writes of them with their dynamic offsets (`binding::rebind_into`:
//!   a texture's view and a sampler state's `VkSampler` as every step makes them, through its
//!   encoder's memo), and the set its encoder's last set or its command buffer's holds, else a new
//!   one (`EncoderCore::set_zero_of`); its push bytes and vertex buffers; and its uses, whole
//!   (`binding::command_uses_into`). And the checks a draw of changed tables takes: a pulled
//!   attribute's reach, the stages' function tables, the attachments' reads. Each is the function
//!   every step of the draw runs, on the same inputs.
//! - **Where it stops.** At a changed shape or runtime samplers' address modes, a plan that reads a
//!   changed slot some other way (a texture buffer, a YCbCr texture's planes, an argument sampler),
//!   a refusal of any of those steps, a pass whose fragments wrote memory (a stage before them may
//!   read a copy of it), a changed texture where its functions use a memoryless texture, the plan
//!   samples or writes a color attachment's texture, or an extended-range texture binds through a
//!   shadow, a shadowed texture among its uses, or a barrier its fragments need after an earlier
//!   draw's geometry writes: the draw takes every step, which answers as it would have. Only the
//!   attachments' reads' refusal is answered here, the last of its checks, which may have
//!   announced.
//! - **What it records and notes.** Its uses noted as any draw notes them (`note_reaching`, the
//!   unordered writes, the pass's reads, the instance's gathered uses, indirect reads and geometry
//!   writes); the state commands that changed and its draw, recorded through the kept draw's
//!   recording against what its command buffer holds (`render/bound.rs`), which compares none of
//!   the dynamic states a value there holds where nothing recorded through the record since the
//!   kept draw, or the draw rebound before, set them (`Kept::settled`): it holds them, of the
//!   values this draw sets, as every compare would answer; its tables' new objects
//!   and its new samplers retained; and the rest as a draw again: kept for recording again (but
//!   where it met a notice met at each occurrence, as its every step forgets it then), its
//!   execution record, its counts into its thread's shard of the device's (`shards.rs`).
//! - **Checked in builds with debug assertions.** As a draw again is: the first rebound draw of a
//!   kept draw and every second after it work out the rebind, then take every step instead, which
//!   must keep the draw, with the same commands, set-0 writes, dynamic offsets, push bytes, vertex
//!   buffers and uses.

use ash::vk;
use metal2vulkan::bind::DrawCall;
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};
use std::sync::Arc;

use super::{Again, RenderLowering};
use crate::argument_table::Stamp;
use crate::binding::{self, SetWrite, Write};
use crate::draw_source::{DrawSource, IndexBind};
use crate::error::Error;
use crate::live_bind::{LiveBind, Reshape};
use crate::pipeline::{RenderStateKind, Settled};
use crate::render::{Assembly, Bound, DrawBuffers, DrawState, Recording};
use crate::texture::Texture;

/// The most slots a draw rebound takes changed: a draw after more takes every step.
const SLOTS: usize = 8;

/// What a draw rebound takes of the kept draw (the module docs) but its variant and plan, which
/// its bind state's kept settle holds (`LiveBind::settled`): in the storage the bind state keeps
/// from one draw and one encoder to the next (`LiveBind::kept`), read only while the kept draw is
/// kept for one ([`Kept::rebinds`](super::Kept)). It holds no object alive: its writes hold no
/// keeper, and its recording names handles alone.
#[derive(Default)]
pub(crate) struct KeptStorage {
    /// Its set-0 writes as its set holds them (`binding::dynamic_writes`).
    writes: Vec<Write>,
    /// Its dynamic states.
    dynamic: Vec<vk::DynamicState>,
    /// Those of them no value of a record holds (`Bound::holds_value`), which every draw sets.
    unheld: Vec<vk::DynamicState>,
    /// The rest of its recording.
    recording: Option<KeptRecording>,
}

/// A kept draw's recording but for what a draw rebound works out again (its set, dynamic
/// offsets, push bytes and vertex buffers) and its dynamic states (`KeptStorage::dynamic`).
pub(in crate::render) struct KeptRecording {
    pub(in crate::render) layout: vk::PipelineLayout,
    pub(in crate::render) shape: u64,
    pub(in crate::render) pipeline: vk::Pipeline,
    pub(in crate::render) index: Option<IndexBind>,
    pub(in crate::render) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(in crate::render) primitive_restart: bool,
    pub(in crate::render) state: DrawState,
    pub(in crate::render) draw: metal2vulkan::vk::DrawCommand,
}

impl KeptStorage {
    /// Keep, over the storage it holds, what a draw rebound takes of the draw just kept: its set-0
    /// `writes`, each of which holds no keeper, its `dynamic` states and the rest of its
    /// `recording`.
    pub(in crate::render) fn keep(
        &mut self,
        writes: &[Write],
        dynamic: &[vk::DynamicState],
        recording: KeptRecording,
    ) {
        self.writes.clear();
        self.writes.extend(writes.iter().filter_map(Write::unheld));
        self.dynamic.clear();
        self.dynamic.extend_from_slice(dynamic);
        self.unheld.clear();
        let unheld = dynamic.iter().filter(|&&state| !Bound::holds_value(state));
        self.unheld.extend(unheld);
        self.recording = Some(recording);
    }

    /// The kept draw's set-0 writes.
    #[cfg(debug_assertions)]
    pub(in crate::render) fn writes(&self) -> &[Write] {
        &self.writes
    }
}

/// Whether `live`'s kept settle holds the variant and plan the kept draw ran: where the draw's
/// shape is the kept one and the state it draws ran it (`LiveBind::settles_kept`).
fn settles_kept(encoder: &RenderLowering, live: &LiveBind) -> bool {
    match encoder.pipeline.as_ref().map(|pipeline| pipeline.kind()) {
        Some(RenderStateKind::Graphics(pipeline)) => live.settles_kept(pipeline, encoder.forms),
        _ => false,
    }
}

/// The buffer, texture and sampler slots calls changed since the kept draw read its tables: at
/// most [`SLOTS`].
struct Changed {
    slots: [SlotRef; SLOTS],
    count: usize,
}

impl Changed {
    /// None yet.
    fn new() -> Changed {
        let any = SlotRef {
            stage: Stage::Vertex,
            class: SlotClass::Buffer,
            index: 0,
        };
        Changed {
            slots: [any; SLOTS],
            count: 0,
        }
    }

    /// Add `slot`: whether there was room.
    fn add(&mut self, slot: SlotRef) -> bool {
        if self.list().contains(&slot) {
            return true;
        }
        let Some(free) = self.slots.get_mut(self.count) else {
            return false;
        };
        *free = slot;
        self.count += 1;
        true
    }

    fn list(&self) -> &[SlotRef] {
        &self.slots[..self.count]
    }

    /// Whether one is a texture or sampler slot.
    fn images(&self) -> bool {
        self.list()
            .iter()
            .any(|slot| slot.class != SlotClass::Buffer)
    }
}

impl RenderLowering {
    /// The kept draw again after calls that bound other buffers, offsets, textures or samplers in
    /// its tables, whose stamps are now `stamps` (the module docs): recorded, refused, or
    /// [`Again::No`] where it takes every step (its check left in `checked` where it is checked).
    pub(super) fn draw_rebound(
        &mut self,
        call: &'static str,
        (assembly, draw): (Assembly, &DrawCall),
        buffers: &DrawBuffers<'_>,
        stamps: [Stamp; 4],
        checked: &mut Option<super::Checked>,
    ) -> Again {
        // A notice met at each occurrence forgets the draw once recorded, as its every step does.
        let watch = crate::degradation::EachWatch::start();
        let live = Arc::clone(&self.live);
        let mut held = crate::device::lock(&live);
        let live = &mut *held;
        let Some(changed) = self.bring_in(live, stamps) else {
            return Again::No;
        };
        if !self.rebind(call, live, &changed, (draw, buffers)) {
            return Again::No;
        }
        #[cfg(debug_assertions)]
        {
            let hits = self.again.hits;
            self.again.hits += 1;
            if hits.is_multiple_of(2) {
                let rebound = super::Expected::Rebound(super::Rebound::of(live));
                drop(held);
                *checked = Some(super::Checked::of(self, rebound));
                return Again::Check;
            }
        }
        #[cfg(not(debug_assertions))]
        let _: &mut Option<super::Checked> = checked;
        // The attachments' reads: its last check, refused here, as it may have announced.
        let attached =
            (self.core.device()).degrade_attachment_reads(call, &self.layout, &live.uses);
        if let Err(refused) = attached {
            return Again::Refused(refused);
        }
        let counted = (stamps, changed.images());
        let recorded = self.record_rebound(live, (call, assembly, draw), buffers, counted);
        if let Err(refused) = recorded {
            return Again::Refused(refused);
        }
        if watch.met() {
            self.forget_again();
        }
        Again::Drawn
    }

    /// Bring into the kept draw's bind state the slots calls changed since in each table whose
    /// stamp is no longer the kept draw's, each table so read counted read in part
    /// (`WorkCounts::stage_tables_read_in_part`): the changed slots; `None` where one is not a
    /// buffer, texture or sampler slot, a table keeps no list of them, or the shape changed (then
    /// the kept state makes its shape again, as any draw's does).
    fn bring_in(&self, live: &mut LiveBind, stamps: [Stamp; 4]) -> Option<Changed> {
        let RenderStateKind::Graphics(pipeline) = self.pipeline.as_ref()?.kind() else {
            return None;
        };
        let kept = self.again.kept.as_ref()?;
        let interface = &pipeline.base().interface;
        let views = pipeline.at_bind_views();
        let moved = crate::buffer::host_cells_moved();
        let maps = live.read_from.maps().clone();
        if !live.read_from.reads(interface, &maps, moved) {
            return None;
        }
        let tables = [&self.vertex, &self.fragment, &self.object, &self.mesh];
        let stages = [Stage::Vertex, Stage::Fragment, Stage::Object, Stage::Mesh];
        // Each changed table: read last as the kept draw read it, its slots that tell the view
        // base alignment apart as the state holds them, its changes since listed, all of buffer,
        // texture and sampler slots.
        let mut changed = Changed::new();
        for (at, (stage, table)) in stages.into_iter().zip(tables).enumerate() {
            let under = kept.stamps[at];
            if stamps[at] == under {
                continue;
            }
            let state = live.state.stages.get(&stage)?;
            let read = live.read_from.read_under(at) == Some(under);
            if !read || !views.holds(stage, state.at_bind_views) {
                return None;
            }
            for (class, index) in table.changed_since(under)? {
                let bound = matches!(
                    class,
                    SlotClass::Buffer | SlotClass::Texture | SlotClass::Sampler
                );
                if !bound
                    || !changed.add(SlotRef {
                        stage,
                        class,
                        index,
                    })
                {
                    return None;
                }
            }
        }
        // A texture or sampler changed: no texture its functions use memoryless (a draw of one
        // fails its command buffer, its every step's first check), and the runtime samplers'
        // address modes as the textures now map them the kept draw's (its every step settles and
        // binds under them).
        if changed.images() {
            if self.uses_memoryless(interface) {
                return None;
            }
            let stages = [
                (Stage::Vertex, &self.vertex),
                (Stage::Fragment, &self.fragment),
                (Stage::Object, &self.object),
                (Stage::Mesh, &self.mesh),
            ];
            let layout = &pipeline.base().layout;
            let now = crate::address_map::sampler_maps(layout, &stages, Some(&self.residency));
            if now.ok()? != maps {
                return None;
            }
        }
        // Their slots brought in. What one brought in changed stays told where a later table
        // takes them no more (`Reshape`), as the draw's every step reads that one again.
        let mut reshape = Reshape::default();
        let mut read = 0;
        for (at, (stage, table)) in stages.into_iter().zip(tables).enumerate() {
            let under = kept.stamps[at];
            if stamps[at] == under {
                continue;
            }
            let from = (live.state.stages.entry(stage).or_default(), under);
            if !table.refresh_changed(from, (stage, interface), (&maps, views), &mut reshape) {
                live.reshaped(reshape);
                return None;
            }
            read += 1;
        }
        live.read_from
            .read(stamps, (interface, || pipeline.held()), &maps, moved);
        // Read now, whatever the draw does next: a draw that takes every step from here finds
        // them read, as the draw rebound does.
        (self.core.device().shared().counts).tables_read_in_part(read);
        let reshaped = reshape.may_reshape();
        live.reshaped(reshape);
        (!reshaped).then_some(changed)
    }

    /// Work out again in `live` what the slots `changed` decide of the kept draw, `draw` of
    /// `buffers`: its commands, set-0 writes and dynamic offsets, push bytes, vertex buffers and
    /// uses, with the checks a draw's steps take of them (the module docs). Whether it did; else
    /// the draw takes every step, which writes them all again.
    fn rebind(
        &mut self,
        call: &'static str,
        live: &mut LiveBind,
        changed: &Changed,
        (draw, buffers): (&DrawCall, &DrawBuffers<'_>),
    ) -> bool {
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_ref())
            .flatten();
        let Some(Settled { variant, plan, .. }) = kept.map(|here| here.settled()) else {
            return false;
        };
        let slots = changed.list();
        if plan.rebind_draw_slots_into(&live.state, slots, &mut live.commands) != Ok(true) {
            return false;
        }
        let compiled = variant.pipeline();
        let interface = &compiled.interface;
        let checked = self
            .check_vertex_reach(&compiled.vertex_input.pulled, draw)
            .is_ok()
            && (self.vertex)
                .check_function_tables(call, Stage::Vertex, interface)
                .is_ok()
            && (self.fragment)
                .check_function_tables(call, Stage::Fragment, interface)
                .is_ok();
        // The pass's fragment writes, a copy of which a stage before the fragments may read
        // (`render/pass_start_reads.rs`): none, so each stage reads its encoder's table.
        if !checked || self.pass_start.fragments_wrote() {
            return false;
        }
        // A changed texture: none its plan samples or writes is a color attachment's (a draw
        // that reads one reads its copy, or a snapshot, or records apart), and no extended-range
        // texture is bound through a shadow (`LiveBind`'s state), as for the kept draw.
        let images = changed.images();
        if images
            && (self.samples_attachment(plan, Texture::attached_in)
                || self.storage_colors(plan) != 0
                || !live.state.extended_shadows.is_empty())
        {
            return false;
        }
        let stages = [
            (Stage::Vertex, &self.vertex),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, &self.object),
            (Stage::Mesh, &self.mesh),
        ];
        let mut tables = binding::Tables::new(&stages)
            .with_pass(&self.layout, plan.views)
            .with_residency(&self.residency)
            .with_texture_table(None)
            .with_sampler_table(None)
            .with_function_tables(&[]);
        // Its runtime samplers' address modes: the kept draw's, as `bring_in` found them.
        if images {
            tables = tables.with_sampler_maps(live.read_from.maps().clone());
        }
        let commands = &live.commands;
        let command = binding::Command {
            descriptor_bytes: &commands.descriptor_bytes,
            descriptor_ranges: &commands.descriptor_ranges,
            root_tables: Vec::new(),
            push_holes: &commands.push_holes,
            uses: &commands.uses,
            blobs: Vec::new(),
            mesh: None,
            tessellation: None,
        };
        let objects = variant.objects();
        let rebound = {
            let device = self.core.device();
            let writes = &mut live.kept.writes[..];
            let set_zero = (writes, objects.dynamic_bindings(), &mut live.offsets[..]);
            let samplers = (Some(&mut self.bind_memo), &mut live.samplers);
            binding::rebind_into(
                call,
                device,
                (&**plan, &command, &tables),
                slots,
                set_zero,
                samplers,
            )
        };
        let index = buffers.index.as_ref().map(DrawSource::metal);
        let uses = (plan, &command, &tables);
        // Its uses: where only buffer slots changed, the kept draw's with those slots' patched in
        // place, else made again whole, those of what the encoder made resident kept
        // (`binding::command_uses_into`).
        let patched = !images
            && binding::patch_buffer_uses((&**plan, &command, &tables), slots, &mut live.uses)
                == Ok(true);
        let rebound = rebound.is_ok_and(|done| done)
            && (patched
                || binding::command_uses_into(call, uses, (index, None), &mut live.uses).is_ok())
            && binding::push_writes_into(&commands.push_constants, &[], &mut live.pushes).is_ok();
        if !rebound {
            return false;
        }
        let made = live.uses.made() as u64;
        if !patched && made > 0 {
            self.core.device().shared().counts.resident_uses_made(made);
        }
        // Its fixed-function vertex buffers at the changed slots, in the plan's order.
        for (bind, at) in plan.vertex_buffers.iter().zip(&mut live.vertex_buffers) {
            let slot = SlotRef {
                stage: Stage::Vertex,
                class: SlotClass::Buffer,
                index: bind.buffer_index,
            };
            if !slots.contains(&slot) {
                continue;
            }
            let offset =
                (commands.buffer_offsets.iter()).find(|(binding, _)| *binding == bind.binding);
            let space = match (
                self.vertex.buffers.get(&bind.buffer_index),
                self.vertex.bytes.get(&bind.buffer_index),
            ) {
                (Some((buffer, _)), _) => buffer.space(),
                (None, Some(bytes)) => (bytes.region.raw(), bytes.region.offset()),
                (None, None) => return false,
            };
            let (Some(&(_, offset)), (raw, start)) = (offset, space) else {
                return false;
            };
            at.1 = raw;
            at.2 = start + offset;
        }
        // No shadowed texture among its uses, and no barrier its fragments need after an earlier
        // draw's geometry writes.
        let textures = (self.vertex.textures.values())
            .chain(self.fragment.textures.values())
            .chain(self.object.textures.values())
            .chain(self.mesh.textures.values())
            .chain(self.residency.textures());
        // Nor a detached buffer, a draw of which records nothing (`EncoderCore::skips`).
        let uses = &live.uses;
        let named = (uses.iter().map(|used| used.key)).filter(|key| !uses.resident().contains(key));
        crate::shadow::shadow_uses(textures, uses).is_empty()
            && self.current.geometry_writes.barrier_before(uses).is_none()
            && !self.core.would_skip(named)
    }

    /// Record the draw rebound in `live` (the module docs): its set 0, its uses noted, its state
    /// commands and draw, and the rest a draw again takes. A refusal is set 0's or the noting's.
    fn record_rebound(
        &mut self,
        live: &mut LiveBind,
        (call, assembly, draw): (&'static str, Assembly, &DrawCall),
        buffers: &DrawBuffers<'_>,
        (stamps, images): ([Stamp; 4], bool),
    ) -> Result<(), Error> {
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_ref())
            .flatten();
        let unsettled = || crate::error::internal("a draw rebound unsettled");
        let objects = kept.ok_or_else(unsettled)?.settled().variant.objects();
        let (set, written) = match objects.pushes_set_zero() {
            true => {
                let immutable = objects.immutable_pushes();
                SetWrite::push_into(&live.kept.writes, immutable, &mut live.pushed);
                (None, None)
            }
            false => self.core.set_zero_of(objects, &mut live.kept.writes)?,
        };
        let pushed: &[SetWrite] = match objects.pushes_set_zero() {
            true => &live.pushed,
            false => &[],
        };
        let uses = &live.uses;
        let stages = (uses.iter()).fold(vk::PipelineStageFlags2::empty(), |all, used| {
            all | used.read.stages | used.write.stages
        });
        (self.core).note_reaching(stages, (uses, uses.resident()), Some(&mut live.noted))?;
        self.unordered.add(uses);
        let prefix = uses.prefix();
        match prefix {
            None => {
                self.pass_start.announce(self.core.device(), uses, false);
                self.pass_start.add(uses);
            }
            Some(prefix) => (self.pass_start).draw(self.core.device(), (uses, prefix), false),
        }
        match prefix {
            None => self.gathering().add(&[], uses),
            Some(prefix) => {
                if self.gathering().add_draw(&[], (uses, prefix)) {
                    (self.core.device().shared().counts).resident_uses_gathered(prefix.len as u64);
                }
            }
        }
        self.current.indirect_reads.add(uses);
        self.current.geometry_writes.add(uses);
        let owned = (live.kept.recording.as_ref())
            .ok_or_else(|| crate::error::internal("a draw rebound without its recording"))?;
        let shared = self.core.device().shared();
        let settled = (self.again.kept.as_ref()).and_then(|kept| kept.settled);
        let bound = match self.current.inline {
            true => &mut self.held_bound,
            false => &mut self.current.bound,
        };
        let recording = Recording {
            raw: shared.vk(),
            command_buffer: self.current.draws,
            layout: owned.layout,
            shape: owned.shape,
            pipeline: owned.pipeline,
            sets: set.as_slice(),
            offsets: &live.offsets,
            pushed,
            push_descriptor: shared.push_descriptor,
            pushes_after_pipeline_binds: shared.pushes_after_pipeline_binds,
            heap: None,
            push: &live.pushes,
            vertex_buffers: &live.vertex_buffers,
            index: owned.index,
            indirect: None,
            dynamic: &live.kept.dynamic,
            // Nothing recorded through the record since the kept draw, or the draw rebound
            // before, set the pipeline and dynamic states this draw sets (`Kept::settled`).
            settled: (settled == Some(bound.records())).then_some(&live.kept.unheld[..]),
            topology: owned.topology,
            primitive_restart: owned.primitive_restart,
            state: &owned.state,
            extended: shared.extended_dynamic_state,
            mesh: shared.mesh_draws.as_ref(),
            draw: owned.draw,
            pair: None,
        };
        // SAFETY: as the kept draw's recording: the secondary command buffer (or the pass's,
        // inline) records the kept draw's instance, and only this encoder records it; every
        // object recorded is live until the command buffer completes (the kept draw's retained,
        // and the tables' retained below); `bound` is the record of what its draws bound there.
        let recorded = unsafe { recording.record(bound) };
        let settled = bound.records();
        if let Some(written) = written {
            self.core.keep_set_zero(written, &live.kept.writes);
        }
        let kept = (self.again.kept.as_mut())
            .ok_or_else(|| crate::error::internal("a draw rebound without its kept draw"))?;
        kept.stamps = stamps;
        kept.settled = Some(settled);
        // Whether its uses read an image: what its drawing again notes (`Kept::reads_images`).
        if images {
            kept.reads_images = live.uses.iter().any(super::reads_image);
        }
        let counts = &self.core.device().shared().counts;
        let pushed = (recorded.states, recorded.writes_pushed);
        counts.draw_rebound((kept.pushes_set_zero, kept.last_vertex), pushed);
        recorded.count_equal_shapes(counts);
        self.current.recorded = true;
        let again = kept.again;
        let (reads, rasterizes, depth_writes) = (kept.reads, kept.rasterizes, kept.depth_writes);
        let execution = (kept.execution.as_ref()).map(|execution| {
            let shape = Arc::clone(&execution.shape);
            (execution.pipeline.clone(), shape, execution.created.clone())
        });
        if let Some((pipeline, shape, created)) = execution {
            self.core
                .executed(|| crate::execution::Execution::new(pipeline, shape, created));
        }
        self.keep_draw(again, (call, assembly, *draw), buffers.index.as_ref());
        self.vertex.retain_exclusive(&mut self.core);
        self.fragment.retain_exclusive(&mut self.core);
        self.object.retain_exclusive(&mut self.core);
        self.mesh.retain_exclusive(&mut self.core);
        self.last_retained
            .retain_samplers(&mut self.core, &mut live.samplers);
        let under = self.under();
        if let Some(kept) = self.again.kept.as_mut() {
            kept.under = under;
        }
        self.draw_calls += 1;
        self.command_encoded(reads);
        self.called = true;
        self.drew |= rasterizes;
        self.drew_depth_writes |= depth_writes;
        Ok(())
    }
}
