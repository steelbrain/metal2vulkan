//! A render encoder's kept draw drawn again after calls that bound other buffers, offsets, textures
//! or sampler states at slots of its tables (`WorkCounts::draws_rebound`): its every other step is
//! the kept draw's (`render/draw_again.rs`), so only what those slots decide is worked out again.
//!
//! - **Which draw.** The kept draw again ([`RenderLowering::draw_again`]: the same call and
//!   arguments, through the same pipeline state, under the same `Under`, into the same command
//!   buffer) after calls that changed only buffer, texture and sampler slots of its tables, after
//!   draws of other pipeline states (`Kept::between`, `render/draw_again.rs`), or after calls that
//!   set the encoder's render state, what it decides derived again as the kept draw's
//!   (`Kept::stated`, `restated.rs`), each table's changes since the kept draw read
//!   it still listed (`argument_table::Changes`, at most [`SLOTS`] slots), where the kept draw kept
//!   what this takes ([`KeptStorage`]: a draw whose set-0 writes hold no keeper, of no use beyond
//!   its plan's and its draw's own, and of a state that made no visible-function table). The kept
//!   bind state takes the changed slots (`ArgumentTable::refresh_changed`), and the draw goes on
//!   only where its shape stays, and its runtime samplers' address modes (`address_map.rs`) with
//!   it: its variant, plan, pipeline and draw state are then the kept draw's, functions of the
//!   shape and of what no call changed.
//! - **What it works out again.** Its commands, rebound at those slots
//!   (`BindingPlan::rebind_draw_slots_into`: their descriptor ranges, push fields and vertex
//!   buffers' offsets), by the plan's rebinds of them (`BindingPlan::slot_rebinds_into`), made
//!   once for the kept draw, in lists the state keeps from one settle and encoder to the next
//!   (`LiveBind::rebinds`), and kept with it while the draws change the same slots
//!   (`SettledHere::rebinds`);
//!   set 0's writes of them with their dynamic offsets (`binding::rebind_into`:
//!   a texture's view and a sampler state's `VkSampler` as every step makes them, through its
//!   encoder's memo), and the set its encoder's last set or its command buffer's holds, else a new
//!   one (`EncoderCore::set_zero_of`); its push bytes and vertex buffers; and its uses, whole
//!   (`binding::command_uses_into`), a texture of a color attachment of its pass bound through its
//!   copy (`texture/attached_reads.rs`); and its uses of shadowed textures
//!   (`RenderLowering::draw_shadows`). And the checks a draw of changed tables takes: a pulled
//!   attribute's reach, the stages' function tables, the attachments' reads. Each is the function
//!   every step of the draw runs, on the same inputs.
//! - **Where it stops.** At a changed shape or runtime samplers' address modes, a plan that reads a
//!   changed slot some other way (a texture buffer, a YCbCr texture's planes, an argument sampler),
//!   a refusal of any of those steps, a pass whose fragments wrote memory (a stage before them may
//!   read a copy of it), a changed texture where its functions use a memoryless texture, the plan
//!   writes a color attachment's texture, or an extended-range texture binds through a shadow, a
//!   read of a color its pass clears behind its load actions, uses of shadowed textures other than
//!   the kept draw's (another shadow, or one used otherwise: written, or its reads bounded
//!   elsewhere, as a scissor restated bounds a read at the fragments' own pixels), or a barrier
//!   its fragments need after an earlier draw's geometry writes: the draw takes every step, which
//!   answers as it would have. Only the
//!   attachments' reads' refusal is answered here, the last of its checks, which may have
//!   announced.
//! - **What it records and notes.** Its uses noted as any draw notes them (`note_reaching`, the
//!   pass's reads, the instance's gathered uses), of the kept draw's access types: none writes or
//!   reads indirect arguments (`draw_again::unwritten`), so the unordered writes, indirect reads and
//!   geometry writes take none of them, and in a pass whose fragments wrote nothing the pass's
//!   reads announce nothing of them; only uses that begin with those of what the encoder made
//!   resident are walked there, as those trackers take them by their identity. The state commands
//!   that changed and its draw, recorded through the kept draw's
//!   recording against what its command buffer holds (`render/bound.rs`), which compares none of
//!   the dynamic states a value there holds where nothing recorded through the record since the
//!   kept draw, or the draw rebound before, set them (`Kept::settled`): it holds them, of the
//!   values this draw sets, as every compare would answer; its tables' new objects
//!   and its new samplers retained; and the rest as a draw again: kept for recording again (but
//!   where it met a notice met at each occurrence, as its every step forgets it then), its
//!   execution record, its counts into its encoder's tally of the device's (`shards.rs`).
//! - **A draw moved.** Where every call since the kept draw moved a buffer slot's offset (or
//!   attribute stride) alone, the slot binding the buffer it bound (`argument_table::Changed`), the
//!   draw is moved (`WorkCounts::draws_moved`): its tables bind the kept draw's objects, so it
//!   looks at no stage's function tables or detached buffer again, nor at shadowed textures where
//!   the kept draw used none (where it used some, its reads' bounds may follow a scissor
//!   restated, so it works them out as a draw rebound does); and where its
//!   commands' descriptor ranges are the kept draw's, it keeps the kept draw's set-0 writes and
//!   dynamic offsets (`binding::rebind_into` writes a buffer's descriptor of its buffer and range
//!   alone): where nothing recorded through the record since the kept draw, the record holds that
//!   set 0 as pushed or bound, so the draw pushes or binds none. One whose uses were patched in
//!   place, none of them of what the encoder made resident, is narrowed: its uses differ from the
//!   kept draw's only at the moved slots' uses (`binding::patch_buffer_uses`), which alone it
//!   checks against the attachments and the geometry writes, adds to the pass's reads and gathers
//!   (`InstanceShadows::add_moved`), and
//!   compares and records where the last command noted is the kept draw (`EncoderCore::note_moved`);
//!   a moved use inside the bytes of the use noted, or gathered, at its place before, as a vertex
//!   buffer's offset moving along it reads, is noted and gathered already (`Use::covers`,
//!   `noted_apart.rs`). One whose moved uses each lie inside the uses they replace in its bind
//!   state, the last draw's (as the patch finds them), is inside (`WorkCounts::draws_moved_inside`):
//!   each is in the encoder's part, the pass's reads and the instance's gathered uses already,
//!   inside the last draw's or one covering it, so it notes, adds and gathers none of them,
//!   comparing none.
//! - **Checked in builds with debug assertions.** As a draw again is: the first rebound draw of a
//!   kept draw and every second after it work out the rebind, then take every step instead, which
//!   must keep the draw, with the same commands, set-0 writes, dynamic offsets, push bytes, vertex
//!   buffers and uses. What a draw moved skips is asserted where it skips it: its function tables
//!   pass, it meets no shadow where its kept draw met none and no detached buffer, it pushes the
//!   kept draw's set-0 writes, a set
//!   0 it pushes or binds none of is the record's, its uses are unwritten in a pass whose
//!   fragments wrote nothing, and of a draw narrowed, its other uses share no attachment memory,
//!   meet no geometry write the moved ones do not, add nothing to the pass's reads, are noted and
//!   touched as the last command's, and are the instance's last gathered uses; of a draw inside,
//!   its moved uses lie inside the noted ones and add nothing to the pass's reads or the
//!   instance's gathered uses.

use ash::vk;
use metal2vulkan::bind::{DescriptorRange, DrawCall};
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
use crate::resource::ResourceKey;
use crate::shadow::{ShadowUse, UsedAs};
use crate::texture::Texture;
use crate::tracker::Use;

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
    /// Its uses of shadowed textures, as it gathered them into its instance
    /// (`shadow::ShadowUse::used_as`): a draw rebound goes on only where it uses the same.
    shadows: Vec<UsedAs>,
    /// A draw moved's storage, kept likewise: its commands' descriptor ranges before it worked
    /// them out, and the positions and uses of the moved slots' uses (the module docs).
    ranges: Vec<DescriptorRange>,
    moved_at: Vec<usize>,
    moved: Vec<Use<ResourceKey>>,
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
    /// `writes`, each of which holds no keeper, its `dynamic` states, the rest of its
    /// `recording` and its uses of shadowed textures, `shadows`.
    pub(in crate::render) fn keep(
        &mut self,
        writes: &[Write],
        dynamic: &[vk::DynamicState],
        recording: KeptRecording,
        shadows: &[ShadowUse],
    ) {
        self.shadows.clear();
        self.shadows.extend(shadows.iter().map(ShadowUse::used_as));
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

    /// The kept draw's recording, where it is kept.
    #[cfg(debug_assertions)]
    pub(in crate::render) fn recording(&self) -> Option<&KeptRecording> {
        self.recording.as_ref()
    }

    /// The kept draw's recording, which a draw restated records with its new draw state
    /// (`restated.rs`).
    pub(super) fn recording_mut(&mut self) -> Option<&mut KeptRecording> {
        self.recording.as_mut()
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
/// most [`SLOTS`]; of the vertex stage's buffer slots below 64, those every one of those calls
/// changed by the offset alone, a bit each; and whether every one of those calls moved a buffer
/// slot's offset alone.
struct Changed {
    slots: [SlotRef; SLOTS],
    count: usize,
    /// The vertex buffer slots some call changed by the offset alone, and those some call changed
    /// otherwise.
    offsets: u64,
    whole: u64,
    /// Whether every call moved the offset of a buffer slot alone, keeping the buffer bound there
    /// (`argument_table::Changed::offset`): the draw is moved (the module docs).
    moved: bool,
    /// Whether one of the slots is a texture or sampler slot.
    images: bool,
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
            offsets: 0,
            whole: 0,
            moved: true,
            images: false,
        }
    }

    /// Note a call's change of `slot`, by its offset alone where `offset` (a buffer slot's).
    fn note(&mut self, slot: SlotRef, offset: bool) {
        self.moved &= offset && slot.class == SlotClass::Buffer;
        if slot.stage != Stage::Vertex || slot.class != SlotClass::Buffer {
            return;
        }
        let bit = 1u64.checked_shl(slot.index).unwrap_or(0);
        match offset {
            true => self.offsets |= bit,
            false => self.whole |= bit,
        }
    }

    /// Whether every call since changed vertex buffer slot `index` by its offset alone: the slot
    /// binds the buffer the kept draw bound there.
    fn moved_alone(&self, index: u32) -> bool {
        let bit = 1u64.checked_shl(index).unwrap_or(0);
        self.offsets & !self.whole & bit != 0
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
        self.images |= slot.class != SlotClass::Buffer;
        true
    }

    fn list(&self) -> &[SlotRef] {
        &self.slots[..self.count]
    }

    /// Whether one is a texture or sampler slot.
    fn images(&self) -> bool {
        self.images
    }
}

/// What a draw rebound worked out of its rebinding (`RenderLowering::rebind`): whether it is moved
/// ([`Changed::moved`]), whether its set-0 writes and dynamic offsets are then the kept draw's
/// (its descriptor ranges are), and whether its uses differ from the kept draw's only at the
/// moved slots' uses, listed in its bind state's kept storage (`KeptStorage::moved`): the draw is
/// narrowed; and whether, narrowed, each of those uses lies inside the bytes of the last draw's at
/// its place: the draw is inside (the module docs).
#[derive(Clone, Copy)]
struct Rebinding {
    moved: bool,
    writes_kept: bool,
    narrowed: bool,
    inside: bool,
    /// Whether it reads the texture of a color attachment of its pass through its copy, and of
    /// that, at its fragments' own pixels alone, which it counts once recorded.
    through_copies: (bool, bool),
}

impl RenderLowering {
    /// The kept draw again after calls that bound other buffers, offsets, textures or samplers in
    /// its tables, whose stamps are now `stamps` (the module docs), with the encoder's bind state
    /// `live`: recorded, refused, or [`Again::No`] where it takes every step (its check left in
    /// `checked` where it is checked).
    pub(super) fn draw_rebound(
        &mut self,
        live: &mut LiveBind,
        call: &'static str,
        (assembly, draw): (Assembly, &DrawCall),
        buffers: &DrawBuffers<'_>,
        stamps: [Stamp; 4],
        checked: &mut Option<super::Checked>,
    ) -> Again {
        // A notice met at each occurrence forgets the draw once recorded, as its every step does.
        let watch = crate::degradation::EachWatch::start();
        // After a call set the encoder's render state, what it decides derived again
        // (`restated.rs`).
        let restated =
            (self.again.kept.as_ref()).is_some_and(|kept| kept.stated != self.again.stated);
        if restated && !self.restate_kept(live, call, assembly, draw) {
            return self.rebind_stopped();
        }
        let Some(changed) = self.bring_in(live, stamps) else {
            return self.rebind_stopped();
        };
        let Some(rebinding) = self.rebind(call, live, &changed, (draw, buffers)) else {
            return self.rebind_stopped();
        };
        #[cfg(debug_assertions)]
        {
            let hits = self.again.hits;
            self.again.hits += 1;
            if hits.is_multiple_of(2) {
                let rebound = super::Expected::Rebound(super::Rebound::of(live));
                *checked = Some(super::Checked::of(self, rebound));
                return Again::Check;
            }
        }
        #[cfg(not(debug_assertions))]
        let _: &mut Option<super::Checked> = checked;
        // The attachments' reads: its last check, refused here, as it may have announced.
        let core = &self.core;
        let version = self.layout.version();
        let (memo, layout) = (&mut self.attached, &*self.layout);
        let mut shares = |uses: &[_]| memo.shares((layout, version), uses);
        // Of a draw narrowed, the moved slots' uses alone: every other use is the kept draw's,
        // whose check found them sharing none (it was drawn), against the same attachments.
        let attached = match (rebinding.narrowed, live.uses.prefix()) {
            (true, _) => {
                let moved = (&live.kept.moved[..], &live.uses[..]);
                core.degrade_moved_attachment_reads(call, shares, moved)
            }
            (false, None) => core.degrade_attachment_reads(call, shares, &live.uses),
            (false, Some(prefix)) => {
                let shares = (&mut shares, Some(version));
                core.degrade_resident_attachment_reads(call, shares, (&mut live.uses, prefix))
            }
        };
        if let Err(refused) = attached {
            return Again::Refused(refused);
        }
        let counted = (stamps, changed.images());
        let recorded =
            self.record_rebound(live, (call, assembly, draw), buffers, (counted, rebinding));
        if let Err(refused) = recorded {
            return Again::Refused(refused);
        }
        let counts = self.core.counts();
        if rebinding.through_copies.0 {
            counts.attachment_read_through_copy();
        }
        if rebinding.through_copies.1 {
            counts.own_pixel_read();
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
        // Read with the maps they were read with, which only a changed texture or sampler may
        // change (below).
        if !live.read_from.reads_through(interface, moved) {
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
            for (class, index, offset) in table.changed_since(under)? {
                let bound = matches!(
                    class,
                    SlotClass::Buffer | SlotClass::Texture | SlotClass::Sampler
                );
                let slot = SlotRef {
                    stage,
                    class,
                    index,
                };
                if !bound || !changed.add(slot) {
                    return None;
                }
                changed.note(slot, offset);
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
            if now.ok()? != *live.read_from.maps() {
                return None;
            }
        }
        // Their slots brought in. What one brought in changed stays told where a later table
        // takes them no more (`Reshape`), as the draw's every step reads that one again.
        let mut reshape = Reshape::default();
        let mut read = 0;
        let counts = self.core.counts();
        for (at, (stage, table)) in stages.into_iter().zip(tables).enumerate() {
            let under = kept.stamps[at];
            if stamps[at] == under {
                continue;
            }
            // The state holds the stage, as each changed table's (above).
            let Some(state) = live.state.stages.get_mut(&stage) else {
                live.reshaped(reshape);
                return None;
            };
            let maps = live.read_from.maps();
            let from = (state, under);
            let reading = (stage, interface);
            if !table.refresh_changed(from, reading, (maps, views), &mut reshape, counts) {
                live.reshaped(reshape);
                return None;
            }
            read += 1;
        }
        live.read_from
            .read_again(stamps, (interface, || pipeline.held()), moved);
        // Read now, whatever the draw does next: a draw that takes every step from here finds
        // them read, as the draw rebound does.
        counts.tables_read_in_part(read);
        let reshaped = reshape.may_reshape();
        live.reshaped(reshape);
        (!reshaped).then_some(changed)
    }

    /// Work out again in `live` what the slots `changed` decide of the kept draw, `draw` of
    /// `buffers`: its commands, set-0 writes and dynamic offsets, push bytes, vertex buffers and
    /// uses, with the checks a draw's steps take of them (the module docs). What it worked out
    /// where it did ([`Rebinding`]); else `None`, and the draw takes every step, which writes them
    /// all again.
    fn rebind(
        &mut self,
        call: &'static str,
        live: &mut LiveBind,
        changed: &Changed,
        (draw, buffers): (&DrawCall, &DrawBuffers<'_>),
    ) -> Option<Rebinding> {
        let slots = changed.list();
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_mut())
            .flatten();
        let (settled, rebinds) = kept?.rebinds(slots, &mut live.rebinds);
        let Settled {
            variant,
            plan,
            facts,
            ..
        } = settled;
        // A draw moved: its commands' descriptor ranges as the kept draw left them, which tell
        // whether its set-0 writes are the kept draw's (below). Not one after draws of other
        // pipeline states (`Kept::between`): what it skips looking at again, and what it notes
        // apart, it takes from the encoder's last draw.
        let between = (self.again.kept.as_ref()).is_some_and(|kept| kept.between);
        let moved = changed.moved && !between;
        if moved {
            live.kept.ranges.clear();
            (live.kept.ranges).extend_from_slice(&live.commands.descriptor_ranges);
        }
        // What the slots decide of its commands, by the plan's rebinds of them, made once for the
        // kept draw (`BindingPlan::rebind_draw_slots_into`).
        if plan.rebind_draw_slots_with(rebinds?, &live.state, &mut live.commands) != Ok(true) {
            return None;
        }
        let compiled = variant.pipeline();
        let interface = &compiled.interface;
        // A moved offset binds the buffer the kept draw bound there, never a visible-function
        // table (`argument_table::Changed::offset`): the tables' function tables are those the
        // kept draw's checks passed.
        let tables_checked = || {
            (self.vertex)
                .check_function_tables(call, Stage::Vertex, interface)
                .is_ok()
                && (self.fragment)
                    .check_function_tables(call, Stage::Fragment, interface)
                    .is_ok()
        };
        debug_assert!(
            !moved || tables_checked(),
            "a draw moved binds a function table its kept draw's checks did not pass"
        );
        let checked = self
            .check_vertex_reach(&compiled.vertex_input.pulled, draw)
            .is_ok()
            && (moved || tables_checked());
        // The pass's fragment writes, a copy of which a stage before the fragments may read
        // (`render/pass_start_reads.rs`): none, so each stage reads its encoder's table.
        if !checked || self.pass_start.fragments_wrote() {
            return None;
        }
        // A draw that samples the texture of a color attachment of its pass reads the texture's
        // copy (`texture/attached_reads.rs`), which its tables bind and its shadows hold (below),
        // as every step's do; one that reads a color the pass clears behind its load actions
        // keeps nothing (`render.rs`), so its every step decides it. A changed texture: none its
        // plan writes is a color attachment's (a draw that does records apart), and no
        // extended-range texture is bound through a shadow (`LiveBind`'s state), as for the kept
        // draw.
        let attached = self.samples_attachment(facts, Texture::attached_in);
        if attached && self.loads_ordered && self.samples_attachment(facts, Texture::cleared_in) {
            return None;
        }
        let images = changed.images();
        if images && (self.storage_colors(facts) != 0 || !live.state.extended_shadows.is_empty()) {
            return None;
        }
        let stages = [
            (Stage::Vertex, &self.vertex),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, &self.object),
            (Stage::Mesh, &self.mesh),
        ];
        let commands = &live.commands;
        let command = binding::Command {
            descriptor_bytes: &commands.descriptor_bytes,
            heap_tables: &[],
            descriptor_ranges: &commands.descriptor_ranges,
            root_tables: Vec::new(),
            root_writes: false,
            push_holes: &commands.push_holes,
            uses: &commands.uses,
            blobs: Vec::new(),
            mesh: None,
            tessellation: None,
        };
        // Set 0's writes and dynamic offsets: a draw moved whose descriptor ranges are the kept
        // draw's binds the same buffers over the same ranges (`binding::rebind_into` writes a
        // buffer's descriptor of its buffer and range alone), so it keeps them.
        let writes_kept = moved && commands.descriptor_ranges == live.kept.ranges;
        // Its uses: where only buffer slots changed, the kept draw's with those slots' patched in
        // place, which reads the stages' tables alone, else made again whole, those of what the
        // encoder made resident kept (`binding::command_uses_into`).
        let patch = (&**plan, &command, &stages[..]);
        let moved_uses = (&mut live.kept.moved_at, &mut live.kept.moved);
        let inside = match images {
            true => None,
            false => binding::patch_buffer_uses(patch, slots, &mut live.uses, moved_uses)
                .ok()
                .flatten(),
        };
        let patched = inside.is_some();
        // The tables a step that works the draw out again reads (its set-0 writes where they are
        // not the kept draw's, its uses where they were not patched), made only for one.
        let mut rebound = true;
        if !writes_kept || !patched {
            let mut tables = binding::Tables::new(&stages)
                .with_pass(&self.layout, plan.views)
                .with_residency(&self.residency);
            if attached {
                tables = tables.with_attachments(&self.layout);
            }
            // Its runtime samplers' address modes: the kept draw's, as `bring_in` found them.
            if images {
                tables = tables.with_sampler_maps(live.read_from.maps().clone());
            }
            if !writes_kept {
                let device = self.core.device();
                let objects = variant.objects();
                let writes = &mut live.kept.writes[..];
                let set_zero = (writes, objects.dynamic_bindings(), &mut live.offsets[..]);
                let samplers = (Some(&mut self.bind_memo), &mut live.samplers);
                let plan = (&**plan, &command, &tables);
                let rebind = binding::rebind_into(call, device, plan, slots, set_zero, samplers);
                rebound = rebind.is_ok_and(|done| done);
            }
            if rebound && !patched {
                let index = buffers.index.as_ref().map(DrawSource::metal);
                let uses = (plan, &command, &tables);
                rebound =
                    binding::command_uses_into(call, uses, (index, None), &mut live.uses).is_ok();
            }
        }
        let rebound = rebound
            && binding::push_writes_into(&commands.push_constants, &[], &mut live.pushes).is_ok();
        if !rebound {
            return None;
        }
        let made = live.uses.made() as u64;
        if !patched {
            let counts = self.core.counts();
            if made > 0 {
                counts.resident_uses_made(made);
            }
            counts.resident_keys_settled(live.uses.settled() as u64);
        }
        // A draw moved whose uses were patched, none of them of what the encoder made resident
        // (whose trackers take them by their identity): the moved slots' uses, as the patch
        // listed them, every use that differs from the kept draw's, each of the kept use's key and
        // access types.
        let narrowed = moved && patched && live.uses.prefix().is_none();
        // Of a draw narrowed, whether each moved use lies inside the bytes of the one it replaced,
        // the last draw's at its place: the draw is inside (the module docs).
        let inside = narrowed && inside == Some(true);
        // Its fixed-function vertex buffers at the changed slots, in the plan's order. A slot
        // whose offset alone moved binds the buffer the kept draw bound there, whose space starts
        // where it did, as no no-copy buffer moved since (`bring_in`): it is not looked at again
        // (`LiveBind::vertex_starts`).
        let kept = (live.vertex_buffers.iter_mut()).zip(&mut live.vertex_starts);
        if plan.vertex_buffers.len() != kept.len() {
            return None;
        }
        for (bind, (at, kept_start)) in plan.vertex_buffers.iter().zip(kept) {
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
                changed.moved_alone(bind.buffer_index),
                self.vertex.buffers.get(&bind.buffer_index),
                self.vertex.bytes.get(&bind.buffer_index),
            ) {
                (true, Some(_), _) => (at.1, *kept_start),
                (_, Some((buffer, _)), _) => buffer.space(),
                (_, None, Some(bytes)) => (bytes.region.raw(), bytes.region.offset()),
                (_, None, None) => return None,
            };
            let (Some(&(_, offset)), (raw, start)) = (offset, space) else {
                return None;
            };
            at.1 = raw;
            at.2 = start + offset;
            *kept_start = start;
        }
        // Its shadowed textures the kept draw's, each used as it used it, gathered into the
        // instance already (`render/draw_again.rs`), worked out as every step works them out
        // (`RenderLowering::draw_shadows`); no barrier its fragments need after an earlier
        // draw's geometry writes, nor a detached buffer, a draw of which records nothing
        // (`EncoderCore::skips`). A draw moved binds the kept draw's textures, and its uses name
        // the kept draw's keys, which met no detached buffer, as no no-copy buffer moved since
        // (`Under`): it looks at none again, nor at shadows where the kept draw used none (its
        // uses of shadows are of their images, not of the buffers whose offsets moved, but its
        // reads' bounds follow a scissor restated). Of a draw narrowed, only the moved slots'
        // uses meet the geometry writes anew: every other use is the kept draw's, which met none
        // of them, and no draw added any since.
        let resident = live.shadows.of(self.core.device(), &self.residency);
        let uses = &live.uses;
        let reading = attached.then(|| (&**plan, compiled));
        let worked_out = !(moved && live.kept.shadows.is_empty());
        let (shadows, own_pixel) = match worked_out {
            true => self.draw_shadows(resident, uses, reading),
            false => (Vec::new(), false),
        };
        debug_assert!(
            worked_out || self.draw_shadows(resident, uses, reading).0.is_empty(),
            "a draw moved met a shadow its kept draw did not"
        );
        if !(shadows.iter().map(ShadowUse::used_as)).eq(live.kept.shadows.iter().copied()) {
            return None;
        }
        let named =
            || (uses.iter().map(|used| used.key)).filter(|key| !uses.resident().contains(key));
        debug_assert!(
            !moved || !self.core.would_skip(named()),
            "a draw moved met a detached buffer its kept draw did not"
        );
        let counts = self.core.counts();
        let unbarriered = match (narrowed, uses.prefix()) {
            (true, _) => {
                let writes = &self.current.geometry_writes;
                let unbarriered = writes.barrier_before(&live.kept.moved).is_none();
                debug_assert!(
                    unbarriered == writes.barrier_before(uses).is_none(),
                    "a draw narrowed's other uses meet a geometry write"
                );
                unbarriered
            }
            (false, None) => self.current.geometry_writes.barrier_before(uses).is_none(),
            (false, Some(prefix)) => {
                let (barrier, walked) =
                    (self.current.geometry_writes).barrier_before_draw((uses, prefix));
                counts.resident_uses_geometry_checked(walked as u64);
                barrier.is_none()
            }
        };
        let rebinding = Rebinding {
            moved,
            writes_kept,
            narrowed,
            inside,
            through_copies: (attached, own_pixel),
        };
        (unbarriered && (moved || !self.core.would_skip(named()))).then_some(rebinding)
    }

    /// Record the draw rebound in `live` (the module docs): its set 0, its uses noted, its state
    /// commands and draw, and the rest a draw again takes. A refusal is set 0's or the noting's.
    fn record_rebound(
        &mut self,
        live: &mut LiveBind,
        (call, assembly, draw): (&'static str, Assembly, &DrawCall),
        buffers: &DrawBuffers<'_>,
        ((stamps, images), rebinding): (([Stamp; 4], bool), Rebinding),
    ) -> Result<(), Error> {
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_ref())
            .flatten();
        let unsettled = || crate::error::internal("a draw rebound unsettled");
        let objects = kept.ok_or_else(unsettled)?.settled().variant.objects();
        let narrowed = rebinding.narrowed;
        // Nothing recorded through the record since the kept draw, or the draw rebound before,
        // set the pipeline and dynamic states this draw sets (`Kept::settled`), nor bound or
        // pushed another set 0.
        let records = match self.current.inline {
            true => self.held_bound.records(),
            false => self.current.bound.records(),
        };
        let settled = (self.again.kept.as_ref()).and_then(|kept| kept.settled) == Some(records);
        // A draw restated (`restated.rs`) compares every dynamic state it sets with the record:
        // its values may be others.
        let stated = self.again.stated;
        let restated = (self.again.kept.as_ref()).is_some_and(|kept| kept.stated != stated);
        let (set, written) = match (objects.pushes_set_zero(), rebinding.writes_kept) {
            // Its set-0 writes the kept draw's (`rebind`): it pushes what the kept draw pushed.
            (true, true) => {
                #[cfg(debug_assertions)]
                {
                    let pushed = SetWrite::pushed(&live.kept.writes, objects.immutable_pushes());
                    let same = (pushed.iter().zip(&live.pushed)).all(|(a, b)| a.same(b));
                    assert!(
                        same && pushed.len() == live.pushed.len(),
                        "a draw moved pushes other writes than its kept draw"
                    );
                }
                (None, None)
            }
            (true, false) => {
                let immutable = objects.immutable_pushes();
                SetWrite::push_into(&live.kept.writes, immutable, &mut live.pushed);
                (None, None)
            }
            // Its set-0 writes the kept draw's (`rebind`), whose set and dynamic offsets the record
            // holds as bound: the set the encoder's last command took (`binding::LastSetZero`),
            // reused, which nothing binds again.
            (false, true) if settled && objects.set_zero_layout().is_some() => {
                #[cfg(debug_assertions)]
                {
                    let layout = objects.set_zero_layout();
                    let writes = &live.kept.writes;
                    let set = layout.and_then(|layout| self.core.last_set_zero(layout, writes));
                    let bound = match self.current.inline {
                        true => &self.held_bound,
                        false => &self.current.bound,
                    };
                    let recorded = live.kept.recording.as_ref().map(|owned| owned.layout);
                    let holds =
                        |(set, layout)| bound.holds_sets(layout, 0, (&[set], &live.offsets));
                    assert!(
                        set.zip(recorded).is_some_and(holds),
                        "a draw moved binds a set 0 its record does not hold"
                    );
                }
                self.core.counts().set_zero_set_reused();
                (None, None)
            }
            (false, _) => self.core.set_zero_of(objects, &mut live.kept.writes)?,
        };
        let pushed: &[SetWrite] = match objects.pushes_set_zero() {
            true => &live.pushed,
            false => &[],
        };
        let uses = &live.uses;
        let prefix = uses.prefix();
        // Its uses are of the kept draw's access types, its plan's: none writes or reads
        // indirect arguments (`draw_again::unwritten`), so none is an unordered write, an
        // indirect read or a geometry write, and in a pass whose fragments wrote nothing
        // (`rebind`) none meets what the pass's reads announce. Their noting adds them alone.
        debug_assert!(
            super::unwritten(uses).0 && !self.pass_start.fragments_wrote(),
            "a draw rebound writes, or follows a fragment write of its pass"
        );
        // Noted: a draw narrowed's moved slots' uses alone where the last command noted is the
        // kept draw (`EncoderCore::note_moved`), else the uses as any draw notes them.
        let noted = match narrowed {
            true => {
                let moved = (&**uses, uses.resident(), &live.kept.moved_at[..]);
                (self.core).note_moved(moved, rebinding.inside, &mut live.noted)?
            }
            false => false,
        };
        match (noted, uses.reaching_prefix()) {
            (true, _) => {}
            (false, None) => {
                let stages = (uses.iter()).fold(vk::PipelineStageFlags2::empty(), |all, used| {
                    all | used.read.stages | used.write.stages
                });
                (self.core).note_reaching(
                    stages,
                    (uses, uses.resident()),
                    Some(&mut live.noted),
                )?;
            }
            (false, Some(prefix)) => {
                let reaching = (&**uses, uses.resident(), prefix);
                (self.core).note_draw(reaching, &mut live.noted)?;
            }
        }
        let resident = prefix.map_or(0, |prefix| prefix.len as u64);
        if let Some(prefix) = prefix {
            if self.unordered.add_draw((uses, prefix)) {
                self.core.counts().resident_uses_unordered_writes(resident);
            }
        }
        // The pass's reads, and the instance's gathered uses: of a draw narrowed, the moved slots'
        // uses alone, every other use being the kept draw's, which they hold; of a draw inside,
        // none, each moved use lying inside the last draw's at its place, which they hold.
        match (narrowed, prefix) {
            (true, _) if rebinding.inside => {
                #[cfg(debug_assertions)]
                {
                    let held = super::digest(self.pass_start.held());
                    self.pass_start.add(&live.kept.moved);
                    assert!(
                        held == super::digest(self.pass_start.held()),
                        "a draw inside's moved uses add to the pass's reads"
                    );
                }
            }
            (true, _) => {
                self.pass_start.add(&live.kept.moved);
                #[cfg(debug_assertions)]
                {
                    let held = super::digest(self.pass_start.held());
                    self.pass_start.add(uses);
                    assert!(
                        held == super::digest(self.pass_start.held()),
                        "a draw narrowed's other uses add to the pass's reads"
                    );
                }
            }
            (false, None) => self.pass_start.add(uses),
            (false, Some(prefix)) => {
                (self.pass_start).draw(self.core.device(), (uses, prefix), false);
            }
        }
        match (narrowed, prefix) {
            (true, _) if rebinding.inside => {
                #[cfg(debug_assertions)]
                {
                    let gathering = self.gathering();
                    let held = super::digest(gathering.gathered());
                    assert!(
                        gathering.add_moved(uses, &live.kept.moved_at)
                            && held == super::digest(gathering.gathered()),
                        "a draw inside's moved uses add to the instance's gathered uses"
                    );
                }
            }
            (true, _) => {
                let gathering = self.gathering();
                if !gathering.add_moved(uses, &live.kept.moved_at) {
                    gathering.add(&[], uses);
                }
            }
            (false, None) => self.gathering().add(&[], uses),
            (false, Some(prefix)) => {
                if self.gathering().add_draw(&[], (uses, prefix)) {
                    self.core.counts().resident_uses_gathered(prefix.len as u64);
                }
            }
        }
        if let Some(prefix) = prefix {
            if self.current.indirect_reads.add_draw((uses, prefix)) {
                self.core.counts().resident_uses_indirect_reads(resident);
            }
            if self.current.geometry_writes.add_draw((uses, prefix)) {
                self.core.counts().resident_uses_geometry_writes(resident);
            }
        }
        let owned = (live.kept.recording.as_ref())
            .ok_or_else(|| crate::error::internal("a draw rebound without its recording"))?;
        let shared = self.core.generation();
        let bound = match self.current.inline {
            true => &mut self.held_bound,
            false => &mut self.current.bound,
        };
        // Where it pushes the kept draw's set-0 writes (`rebind`) into a record settled since the
        // kept draw (above), the record holds them as pushed, so its push compares none.
        debug_assert_eq!(
            records,
            bound.records(),
            "a draw rebound recorded through its record before its draw"
        );
        let pushed = match settled && rebinding.writes_kept && !pushed.is_empty() {
            true => {
                #[cfg(debug_assertions)]
                assert!(
                    (bound.pushed_set(owned.layout, owned.shape, pushed).0).is_empty(),
                    "a draw moved pushes writes its record does not hold"
                );
                &[]
            }
            false => pushed,
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
            settled: (settled && !restated).then_some(&live.kept.unheld[..]),
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
        kept.between = false;
        kept.stated = stated;
        // Whether its uses read an image: what its drawing again notes (`Kept::reads_images`).
        if images {
            kept.reads_images = live.uses.iter().any(super::reads_image);
        }
        let counts = self.core.counts();
        let pushed = (recorded.states, recorded.writes_pushed);
        counts.draw_rebound((kept.pushes_set_zero, kept.last_vertex), pushed);
        if rebinding.moved {
            counts.draw_moved();
        }
        if restated {
            counts.draw_restated();
        }
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
        // The pixels a draw restated may write, through its scissors and viewports now: those of
        // a draw rebound alone are its kept draw's, added already.
        if restated && rasterizes {
            let selects = match self.pipeline.as_ref().map(|pipeline| pipeline.kind()) {
                Some(RenderStateKind::Graphics(state)) => {
                    state.base().graphics.writes_viewport_index
                }
                _ => true,
            };
            self.add_drawn(selects)?;
        }
        Ok(())
    }
}
