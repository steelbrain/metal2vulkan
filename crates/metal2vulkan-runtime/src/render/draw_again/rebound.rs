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
//!   its plan's and its draw's own, and of a state that made no visible-function table); or the
//!   kept draw of other arguments alone, of something to draw (below), after such calls or none.
//!   The kept
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
//!   (`RenderLowering::draw_shadows`), which it gathers into its instance with its uses, as every
//!   step gathers them (`shadow::join`), where they are not the kept draw's, each used as it used
//!   it, which the instance gathered already: another shadow, as a changed slot binds another
//!   texture's copy, or one read within other bounds, as a scissor restated bounds a read at the
//!   fragments' own pixels; the next draw rebound compares its own with them. And the checks a
//!   draw of changed tables takes: a pulled
//!   attribute's reach, the stages' function tables, the attachments' reads. Each is the function
//!   every step of the draw runs, on the same inputs.
//! - **Where it stops.** At a changed shape or runtime samplers' address modes, a plan that reads a
//!   changed slot some other way (a texture buffer, a YCbCr texture's planes, an argument sampler),
//!   a refusal of any of those steps, a pass whose fragments wrote memory (a stage before them may
//!   read a copy of it), a changed texture where its functions use a memoryless texture, the plan
//!   writes a color attachment's texture, or an extended-range texture binds through a shadow, a
//!   read of a color its pass clears behind its load actions or of a color's copy after a barrier
//!   inside its instance (every step takes the split there, or weighs whether it reads what the
//!   instance drew: `render.rs`; a kept draw that read one so is let go at the barrier), uses of
//!   shadowed textures where the kept draw used none, or none where it used some (whether it uses
//!   some decides whether its instance may record it again, `Kept::again`, which a draw rebound
//!   takes as the kept draw's), or a
//!   barrier its fragments need after an earlier draw's geometry writes: the draw takes every step,
//!   which answers as it would have. Only the
//!   attachments' reads' refusal is answered here, the last of its checks, which may have
//!   announced.
//! - **Of other arguments.** A draw that differs from the kept draw only in its arguments (its
//!   call of the same primitive type, with indices or without, its counts and starts, its index
//!   buffer, index type or index offset, each index buffer read in place, its instance count, base
//!   vertex or base instance: `UnlikeKept::arguments_alone`) works out again each step that reads
//!   them, from them and the kept draw's state: its route, which reads them only through whether
//!   the draw may make a primitive, whether it is a strip over indices the encoder's call
//!   restarts at another index than Vulkan's (`render/restart.rs`) and whether it runs a
//!   de-indexed variant (`primitive_id_indices`), so the draw goes on only where each is the kept
//!   draw's; its commands' Vulkan draw,
//!   index overrun, base vertex and instance push fields and index buffer use
//!   (`BindingPlan::redraw_apart_into`), which stops where the plan reads them some other way (a
//!   de-indexed draw, a stage's argument or root table, the vertices per primitive of a draw with
//!   indices after one without) or a fixed-function vertex buffer's fetch
//!   window moves with them (another vertex range without indices, another instance range of a
//!   per-instance buffer); its pulled attributes' reach (`check_vertex_reach`); its uses, the
//!   index buffer's patched in place as a moved slot's is (`binding::patch_command_uses`), so a
//!   draw moved and narrowed checks, notes and gathers it with the moved slots' uses, or made
//!   again whole where the draw reads indices and the kept draw none, or the other way, or
//!   another index buffer; and its index buffer's bind (`IndexBind::of`: the index source is read
//!   in place, so its bound range is the one every step binds). One of another index buffer than
//!   the kept draw's, or of one where the kept draw read none, is no draw moved: its uses meet a
//!   buffer the kept draw's did not, so it looks at the detached buffers and shadows again as a
//!   draw rebound does, and its encoder holds that buffer as every step's does. It then records
//!   that bind where its record holds another, with its new draw, and the kept draw takes its
//!   call, arguments, index source and draw (`WorkCounts::draws_rebound_arguments`). Where its
//!   rebind stops, at any step, the draw takes every step, counted by where
//!   (`WorkCounts::draws_rebind_stopped_arguments_*`: its route, its plan's reads of its slots, a
//!   fetch window, its attributes' reach, or another).
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
//!   restated, so it works them out and gathers them as a draw rebound does); and where its
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
//! - **Carried.** The kept draw of the command buffer's last render encoder, carried into this
//!   one (`render/carried.rs`), is rebound as one after draws of other states and calls that set
//!   the render state, through no record: its tables' stamps are the encoder's first ones, so it
//!   brings in every slot the encoder's calls bound since its opening, whatever table its bind
//!   state was read from. It stops where that state holds a slot none of them bound (of texture
//!   and sampler slots it holds only those its pipeline state reads: `argument_table::ReadSlots`),
//!   or a threadgroup memory length or acceleration structure
//!   (`WorkCounts::draws_carry_stopped_stale`: a new encoder inherits no binding), where it reads
//!   a texture an attachment of its pass renders through a slot narrowed to levels apart from the
//!   attached ones (`draws_carry_stopped_attached`), where it would read the pass's colors
//!   otherwise than its kept draw at every step (below, `draws_carry_stopped_snapshot`), or at any
//!   other stop (`draws_carry_stopped_other`).
//!   Its uses are made again whole, and so are its uses of shadowed textures, a texture of its
//!   pass's color attachment it reads through the texture's copy among them: its kept draw's were
//!   gathered into the ended encoder's instance, so it gathers its own into its instance, as
//!   every step gathers them, where it uses some exactly where its kept draw used some (and stops
//!   otherwise, as any draw rebound does), and keeps them for the encoder's next draw rebound to
//!   compare with. It begins its
//!   pass's first instance as every step's first draw does (`RenderLowering::inline_begin`), and
//!   records every state command it sets into that instance's command buffer, whose record holds
//!   none (`WorkCounts::draws_rebound_carried`). It retains into its encoder what every step
//!   retains beyond its tables' objects: its pipeline state, its variant's objects and its
//!   `VkPipeline`, which its encoder's later draws then retain no more, as after every step's
//!   (`LastRetained`); one carried across a commit (`Kept::across`), whose command buffer holds
//!   none of them, the call's index buffer too.
//! - **Re-homed.** A kept draw kept into a later render pass instance of its pass by a draw that
//!   kept nothing only for recording into it (`render/draw_again.rs`, `Kept::rehomed`) is
//!   rebound as one after draws of other states and calls that set the render state, through no
//!   record: its tables' stamps are its own, so it brings in what its encoder's calls changed
//!   since, as any draw rebound does; its uses are made again whole, and its uses of shadowed
//!   textures are gathered with them into its instance, whose copies then take them, as every
//!   step gathers them there.
//! - **Into an instance recorded inline after a split** (`render/inline_splits.rs`), which keeps
//!   its draws' recordings in its room until its copies are known: its recording is kept there
//!   as every step's is, and recorded at its group's end through the pass's record, which
//!   compares every state it sets; the kept draw is then recorded through no record
//!   (`Kept::settled`).
//! - **A draw that reads through no snapshot** reads its pass's colors through none again, as
//!   every step would ([`RenderLowering::reads_otherwise`]): every step reads none of them in
//!   place (`RenderLowering::in_place_wanted`), asks to read none through snapshots, or reads the
//!   copy taken before the pass all the same, a check's or the area rule's answer
//!   (`RenderLowering::snapshot_wanted_or` and `weigh_snapshot`, which count and take nothing),
//!   and takes no split. The rebind works out the copy's reads as every step does (below). Else
//!   it takes every step (of a carried draw `draws_carry_stopped_snapshot`): a texture the calls
//!   bound may be an attachment of the pass, and a carried draw's next pass may read its colors
//!   otherwise, through a snapshot whatever the area over a buffer's bytes among them
//!   (`render/snapshot.rs`).
//! - **A reader through snapshots** (`render/snapshot.rs`, `render/draw_again.rs`) works out again
//!   that it reads through them as every step would ([`RenderLowering::read_again`]): its draw
//!   reads no color in place, asks to read the same slots of the same colors, pinning none
//!   (`RenderLowering::snapshot_wanted_or`, which counts nothing); its plan reads through them
//!   (`snapshot_holds`); its instance can hold those colors in the feedback loop layout
//!   (`snapshot_apart`), a carried draw's unbegun one beginning in its secondary with them there;
//!   and its form, weighed as every step weighs it, is the snapshot. Else it takes every step
//!   (`WorkCounts::reading_draws_snapshotted_stopped`, and of a carried draw
//!   `draws_carry_stopped_snapshot`), which decides its form anew. Its tables bind each slot's
//!   texture in the feedback loop layout and its snapshot, made or kept with that texture
//!   (`binding::rebind_into` writes a changed slot's snapshot beside it); every step reads no
//!   copy of its colors and checks no read of them as an attachment's, nor does it. Its uses are
//!   made again whole, its snapshots' after them, never moved or narrowed, and its snapshots'
//!   writes are unordered writes as every step's. It records its two draws, each its phase's
//!   first, through its kept recording, the drawing one inside its query where it is measured,
//!   takes its identity, count and query as every step takes them, and forgets the kept draws set
//!   aside that read its colors (`RenderLowering::forget_aside_readers`).
//! - **Checked in builds with debug assertions.** As a draw again is: the first rebound draw of a
//!   kept draw and every second after it work out the rebind, then take every step instead, which
//!   must keep the draw, with the same commands, set-0 writes, dynamic offsets, push bytes, vertex
//!   buffers, uses and uses of shadowed textures, and record it with the same draw state, Vulkan
//!   draw and index buffer bind;
//!   of other arguments, the draw kept is the kept draw with them, its index source and draw.
//!   What a draw moved skips is asserted where it skips it: its function tables pass, it meets no
//!   shadow where its kept draw met none and no detached buffer, it pushes the kept draw's set-0
//!   writes, a set 0 it pushes or binds none of is the record's, its uses are unwritten in a pass
//!   whose fragments wrote nothing, and of a draw narrowed, its other uses share no attachment
//!   memory, meet no geometry write the moved ones do not, add nothing to the pass's reads, are
//!   noted and touched as the last command's, and are the instance's last gathered uses; of a draw
//!   inside, its moved uses lie inside the noted ones and add nothing to the pass's reads or the
//!   instance's gathered uses. Of carried draws, the first of a command buffer and every second
//!   after it take every step instead, counted apart (`DrawAgain::carries`), the draw they keep
//!   recorded into the instance they began.

use ash::vk;
use metal2vulkan::bind::{BindShape, DescriptorRange, DrawCall};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};
use std::sync::Arc;

use super::{Again, RenderLowering};
use crate::argument_table::Stamp;
use crate::binding::{self, NarrowedSlots, SetWrite, SnapshotViews, Write};
use crate::draw_source::{DrawSource, IndexBind};
use crate::error::Error;
use crate::live_bind::{LiveBind, Reshape};
use crate::pipeline::GraphicsState as DrawingState;
use crate::pipeline::{RenderStateKind, Settled};
use crate::render::bound::Recorded;
use crate::render::snapshot::{self, KeptReader, NotWanted, Phase, Weighed};
use crate::render::{line_width, Assembly, Bound, DrawBuffers, DrawSettings, DrawState, Recording};
use crate::render_target::Feedback;
use crate::resource::ResourceKey;
use crate::shadow::{ShadowUse, UsedAs};
use crate::texture::Texture;
use crate::tracker::Use;
use crate::work_counts::{ArgumentsStop, CarryStop, ReshapedBy};

/// The most slots a draw rebound takes changed, together over its stages' tables: as many as a
/// table's change log keeps of one call (`argument_table`'s `CHANGES_KEPT`), so a draw after a
/// material's textures and samplers bound anew is rebound with them. A draw after more takes
/// every step.
const SLOTS: usize = 64;

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
    /// (`shadow::ShadowUse::used_as`), or those of the last draw rebound that gathered its own: a
    /// draw rebound that uses the same gathers nothing.
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
        self.keep_shadows(shadows);
        self.writes.clear();
        self.writes.extend(writes.iter().filter_map(Write::unheld));
        self.dynamic.clear();
        self.dynamic.extend_from_slice(dynamic);
        self.unheld.clear();
        let unheld = dynamic.iter().filter(|&&state| !Bound::holds_value(state));
        self.unheld.extend(unheld);
        self.recording = Some(recording);
    }

    /// Keep `shadows` as the kept draw's uses of shadowed textures: a draw rebound's that gathered
    /// them into its instance ([`Rebinding::gathers`]), against which the encoder's next draw
    /// rebound compares its own.
    fn keep_shadows(&mut self, shadows: &[ShadowUse]) {
        self.shadows.clear();
        self.shadows.extend(shadows.iter().map(ShadowUse::used_as));
    }

    /// The kept draw's set-0 writes.
    #[cfg(debug_assertions)]
    pub(in crate::render) fn writes(&self) -> &[Write] {
        &self.writes
    }

    /// The kept draw's uses of shadowed textures.
    #[cfg(debug_assertions)]
    pub(in crate::render) fn shadows(&self) -> &[UsedAs] {
        &self.shadows
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
    /// The bind state's shape, where the slots reshaped it and it settled as the kept draw did
    /// ([`RenderLowering::bring_in`]): the kept draw's execution record takes it.
    shape: Option<Arc<BindShape>>,
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
            shape: None,
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
struct Rebinding {
    moved: bool,
    writes_kept: bool,
    narrowed: bool,
    inside: bool,
    /// Whether it reads the texture of a color attachment of its pass through its copy, and of
    /// that, at its fragments' own pixels alone, which it counts once recorded.
    through_copies: (bool, bool),
    /// Whether it reads another index buffer than the kept draw, which its encoder holds once it
    /// is recorded, as every step's does.
    reindexed: bool,
    /// Its uses of shadowed textures (`RenderLowering::draw_shadows`), and whether it gathers them
    /// into its instance: a carried draw's, whose kept draw gathered its own into the ended
    /// encoder's instance (`render/carried.rs`), and any other's that are not the kept draw's,
    /// each used as it used it (`KeptStorage::shadows`), which its instance gathered already.
    shadows: Vec<ShadowUse>,
    gathers: bool,
    /// Of a draw that reads through snapshots, its form weighed again, its colors and its
    /// snapshots' uses, the last of its uses ([`ReadAgain`]).
    reader: Option<(Weighed, Feedback, usize)>,
}

/// What a kept draw that reads through snapshots works out again as it is rebound (the module
/// docs): its form weighed as every step weighs it, its colors, and its slots' snapshots, made or
/// kept with their textures, with their views, which its tables bind.
struct ReadAgain {
    weighed: Weighed,
    feedback: Feedback,
    views: SnapshotViews,
    texels: Vec<Texture>,
}

impl RenderLowering {
    /// The kept draw again after calls that bound other buffers, offsets, textures or samplers in
    /// its tables, whose stamps are now `stamps`, or of other arguments where `reargued` (the
    /// module docs), with the encoder's bind state `live`: recorded, refused, or [`Again::No`]
    /// where it takes every step (its check left in `checked` where it is checked).
    pub(super) fn draw_rebound(
        &mut self,
        live: &mut LiveBind,
        call: &'static str,
        (assembly, draw, reargued): (Assembly, &DrawCall, bool),
        buffers: &DrawBuffers<'_>,
        stamps: [Stamp; 4],
        checked: &mut Option<super::Checked>,
    ) -> Again {
        // A notice met at each occurrence forgets the draw once recorded, as its every step does.
        let watch = crate::degradation::EachWatch::start();
        // Carried in from the command buffer's last render encoder (`render/carried.rs`).
        let carried = (self.again.kept.as_ref()).is_some_and(|kept| kept.carried);
        // A draw of other arguments, or a carried one, whose rebind stops is counted by where.
        let stopped = |encoder: &Self, (why, carry): (ArgumentsStop, CarryStop)| {
            if reargued {
                encoder.core.counts().draw_rebind_stopped_arguments(why);
            }
            if carried {
                encoder.core.counts().draw_carry_stopped(carry);
            }
            encoder.rebind_stopped()
        };
        // Of other arguments, routed as the kept draw.
        if reargued && !self.routed_as_kept(assembly, draw) {
            return stopped(self, (ArgumentsStop::Route, CarryStop::Other));
        }
        // Of another index buffer than the kept draw's, or one where it read none.
        let reindexed = reargued && self.reindexed(buffers);
        // After a call set the encoder's render state, what it decides derived again, or its
        // scissors alone after a set of its scissor rectangles alone (`restated.rs`).
        let stated = (self.again.kept.as_ref()).map(|kept| kept.stated);
        if let Some(stated) = stated.filter(|&stated| stated != self.again.stated) {
            if !self.restate(live, stated, (call, assembly, draw)) {
                return stopped(self, (ArgumentsStop::Restated, CarryStop::Other));
            }
        }
        // A carried draw: every slot its bind state holds bound, the encoder's calls bound
        // (`render/carried.rs`).
        if carried && self.carried_stale(live) {
            return stopped(self, (ArgumentsStop::Stale, CarryStop::Stale));
        }
        let changed = match self.bring_in(live, stamps, carried) {
            Ok(changed) => changed,
            Err(why) => return stopped(self, (why, CarryStop::Other)),
        };
        // Reshaped and settled as the kept draw: its execution record names the new shape, as
        // every step's does.
        let kept = self.again.kept.as_mut();
        if let (Some(shape), Some(execution)) =
            (&changed.shape, kept.and_then(|k| k.execution.as_mut()))
        {
            execution.shape = Arc::clone(shape);
        }
        let mut stop = (ArgumentsStop::Other, CarryStop::Other);
        let drawn = (draw, buffers, (reargued, reindexed, carried));
        let Some(rebinding) = self.rebind(call, live, &changed, drawn, &mut stop) else {
            return stopped(self, stop);
        };
        if reargued {
            self.reargue_kept(live, call, draw, buffers);
        }
        #[cfg(debug_assertions)]
        {
            // The first and every second draw rebound since the draw was kept, or carried in since
            // the command buffer's first render encoder, or a commit's (`render/carried.rs`), takes
            // every step.
            let count = match carried {
                true => &mut self.again.carries,
                false => &mut self.again.hits,
            };
            let hits = *count;
            *count += 1;
            if hits.is_multiple_of(2) {
                // Its reads through copies as the rebind worked them out, which it records
                // (`record_rebound`) and every step must work out too: a carried draw's pass may
                // render a texture its kept draw's did not.
                if let Some(kept) = self.again.kept.as_mut() {
                    kept.through_copies = rebinding.through_copies;
                }
                // Its uses of shadowed textures as the rebind worked them out, which every step
                // keeps as its own.
                live.kept.keep_shadows(&rebinding.shadows);
                let rebound = super::Expected::Rebound(super::Rebound::of(live));
                *checked = Some(super::Checked::of(self, rebound));
                return Again::Check;
            }
        }
        #[cfg(not(debug_assertions))]
        let _: &mut Option<super::Checked> = checked;
        // A reader through snapshots: its identity, count and query taken, as every step takes
        // them, and its colors.
        let (query, feedback) = match &rebinding.reader {
            Some((weighed, feedback, _)) => match self.take_weighed(weighed.clone()) {
                Ok(query) => (query, *feedback),
                Err(refused) => return Again::Refused(refused),
            },
            None => (None, Feedback::NONE),
        };
        // The attachments' reads: its last check, refused here, as it may have announced. A
        // reader through snapshots reads its colors through them, and every step checks none of
        // its reads so (`render/snapshot.rs`).
        let core = &self.core;
        let version = self.layout.version();
        let (memo, layout) = (&mut self.attached, &*self.layout);
        let mut shares = |uses: &[_]| memo.shares((layout, version), uses);
        // Of a draw narrowed, the moved slots' uses alone: every other use is the kept draw's,
        // whose check found them sharing none (it was drawn), against the same attachments.
        let attached = match (rebinding.narrowed, live.uses.prefix()) {
            _ if rebinding.reader.is_some() => Ok(()),
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
        // A carried draw begins its pass's first instance (`render/carried.rs`), its colors read
        // through snapshots in the feedback loop layout from its begin.
        if carried {
            if let Err(refused) = self.begin_carried(feedback) {
                return Again::Refused(refused);
            }
        }
        self.current.feedback.0 |= feedback.0;
        let counted = (stamps, changed.images());
        let recorded = self.record_rebound(
            live,
            (call, assembly, draw),
            buffers,
            (counted, &rebinding, query),
        );
        if let Err(refused) = recorded {
            return Again::Refused(refused);
        }
        // A reader through snapshots counted as every step counts it, and the kept draws set
        // aside that read its colors forgotten (`render/draw_again.rs`).
        if rebinding.reader.is_some() {
            self.core.counts().reading_draw_again();
            let shared = &self.core.device().shared().counts;
            shared.reading_draw_snapshotted();
            if query.is_some() {
                shared.coverage_query();
            }
            if let Err(refused) = self.forget_aside_readers(live) {
                return Again::Refused(refused);
            }
        }
        let counts = self.core.counts();
        if rebinding.through_copies.0 {
            counts.attachment_read_through_copy();
        }
        if rebinding.through_copies.1 {
            counts.own_pixel_read();
        }
        if reargued {
            counts.draw_rebound_arguments();
        }
        if watch.met() {
            self.forget_again();
        }
        Again::Drawn
    }

    /// Whether the bind state `live` holds a slot bound that no call of the encoder bound since
    /// its opening: of the kept draw carried into it, whose tables' stamps are the encoder's
    /// tables' first ones (`render/carried.rs`). A new encoder inherits no binding, so such a slot
    /// is stale; and a threadgroup memory length or an acceleration structure, which no stage
    /// table's change lists, is told stale too. A table whose changes since its opening it no
    /// longer lists is not told here: [`bring_in`](Self::bring_in) stops at it.
    fn carried_stale(&self, live: &LiveBind) -> bool {
        let Some(kept) = self.again.kept.as_ref() else {
            return true;
        };
        let tables = [&self.vertex, &self.fragment, &self.object, &self.mesh];
        let stages = [Stage::Vertex, Stage::Fragment, Stage::Object, Stage::Mesh];
        for (at, (stage, table)) in stages.into_iter().zip(tables).enumerate() {
            let Some(state) = live.state.stages.get(&stage) else {
                continue;
            };
            if !state.threadgroup_lengths.is_empty() || !state.acceleration_structures.is_empty() {
                return true;
            }
            let mut bound = (state
                .buffers
                .keys()
                .map(|&index| (SlotClass::Buffer, index)))
            .chain(
                state
                    .textures
                    .keys()
                    .map(|&index| (SlotClass::Texture, index)),
            )
            .chain(
                state
                    .samplers
                    .keys()
                    .map(|&index| (SlotClass::Sampler, index)),
            );
            // A table no call changed binds nothing.
            let opened = kept.stamps[at];
            if table.stamp() == opened {
                if bound.next().is_some() {
                    return true;
                }
                continue;
            }
            let Some(since) = table.changed_since(opened) else {
                continue;
            };
            let stale = bound.any(|slot| {
                let mut changed = since.clone();
                !changed.any(|(class, index, _)| (class, index) == slot)
            });
            if stale {
                return true;
            }
        }
        false
    }

    /// Begin the pass's first instance for the kept draw carried into the encoder, as every
    /// step's first draw that records begins it (`render/inline_first.rs`): inline where the pass
    /// opened so and its instance allows it, else its secondary; a reader's, of its `feedback`
    /// colors read through snapshots, in its secondary. Refused only where an object cannot be
    /// made, as every step's is.
    fn begin_carried(&mut self, feedback: Feedback) -> Result<(), Error> {
        let next = super::super::Next::Current;
        let begin = self.inline_begin(&next, None, (true, true, false), feedback)?;
        if let Some(begin) = begin {
            self.begin_inline(begin);
        }
        Ok(())
    }

    /// Whether `draw` of `assembly`, of other arguments than the kept draw's alone, is routed as
    /// the kept draw: a draw's route (`RenderLowering::routed_draw`) reads its arguments only
    /// through whether it may draw a primitive (`line_width::draws_primitives`, and the provoking
    /// vertex's draw of none, its converse), through how far its vertices run where it is drawn
    /// as a list (`PrivateForm`), which no kept draw is, through whether it is a strip over
    /// indices the encoder's call restarts at another index than Vulkan's (`render/restart.rs`)
    /// and whether it runs a de-indexed variant (`primitive_id_indices`), and through the
    /// fixed-function layouts it pulls past 4 GiB (`render/vertex_reach.rs`), each of which must
    /// be the kept draw's.
    fn routed_as_kept(&self, assembly: Assembly, draw: &DrawCall) -> bool {
        let primitives = |draw: &DrawCall| line_width::draws_primitives(assembly, draw);
        let Some(kept) = self.again.kept.as_ref() else {
            return false;
        };
        let (far, deindexed) = match self.pipeline.as_ref().map(|p| p.kind()) {
            Some(RenderStateKind::Graphics(state)) => {
                let base = state.base();
                let far = self.far_vertex_buffers(base, draw).pulled
                    == self.far_vertex_buffers(base, &kept.draw).pulled;
                let deindexed = |draw: &DrawCall| super::super::primitive_id_indices(state, draw);
                (far, deindexed(draw) == deindexed(&kept.draw))
            }
            _ => (true, true),
        };
        let restarted = |draw: &DrawCall| self.restarted_strip(assembly, draw);
        primitives(&kept.draw) == primitives(draw)
            && far
            && deindexed
            && restarted(draw) == restarted(&kept.draw)
    }

    /// Whether the index source of a draw of `buffers`, read in place, is of another index buffer
    /// than the kept draw's, or the kept draw read none, or the other way.
    fn reindexed(&self, buffers: &DrawBuffers<'_>) -> bool {
        let index = buffers.index.as_ref().and_then(super::KeptIndex::of);
        let kept = self.again.kept.as_ref().and_then(|kept| kept.index);
        kept.map(super::KeptIndex::buffer) != index.map(super::KeptIndex::buffer)
    }

    /// Take into the kept draw, rebound with the arguments of `draw` of `buffers` by `call`, those
    /// arguments: its call, index source and Vulkan draw, and its kept recording's Vulkan draw and
    /// index buffer's bind, as every step binds it (`IndexBind::of`), which it records.
    fn reargue_kept(
        &mut self,
        live: &mut LiveBind,
        call: &'static str,
        draw: &DrawCall,
        buffers: &DrawBuffers<'_>,
    ) {
        let command = live.commands.draw;
        let sized = self.core.device().shared().sized_index_bind;
        if let Some(recording) = live.kept.recording_mut() {
            recording.draw = command;
            let index = buffers.index.as_ref();
            recording.index = index.and_then(|source| IndexBind::of(source, command, sized));
        }
        if let Some(kept) = self.again.kept.as_mut() {
            kept.call = call;
            kept.draw = *draw;
            kept.index = buffers.index.as_ref().and_then(super::KeptIndex::of);
            kept.command = command;
        }
    }

    /// Bring into the kept draw's bind state the slots calls changed since in each table whose
    /// stamp is no longer the kept draw's, each table so read counted read in part
    /// (`WorkCounts::stage_tables_read_in_part`): the changed slots; else where it stopped, where
    /// one is not a buffer, texture or sampler slot, a table keeps no list of them, or the shape
    /// changed (then the kept state makes its shape again, as any draw's does). A `carried`
    /// draw's stamps are its encoder's tables' first ones, under which its state holds the slots
    /// the calls since bound, each a slot of the state ([`carried_stale`](Self::carried_stale)),
    /// whatever table the state was read from.
    fn bring_in(
        &self,
        live: &mut LiveBind,
        stamps: [Stamp; 4],
        carried: bool,
    ) -> Result<Changed, ArgumentsStop> {
        let pipeline = self.pipeline.as_ref().map(|state| state.kind());
        let (Some(RenderStateKind::Graphics(pipeline)), Some(kept)) =
            (pipeline, self.again.kept.as_ref())
        else {
            return Err(ArgumentsStop::Unsettled);
        };
        let interface = &pipeline.base().interface;
        let views = pipeline.at_bind_views();
        let moved = crate::buffer::host_cells_moved();
        // Read with the maps they were read with, which only a changed texture or sampler may
        // change (below).
        if !live.read_from.reads_through(interface, moved) {
            return Err(ArgumentsStop::Unread);
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
            let state = live.state.stages.get(&stage);
            let read = carried || live.read_from.read_under(at) == Some(under);
            if !read || !state.is_some_and(|state| views.holds(stage, state.at_bind_views)) {
                return Err(ArgumentsStop::Unread);
            }
            let since = table.changed_since(under).ok_or(ArgumentsStop::Unlisted)?;
            for (class, index, offset) in since {
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
                    return Err(ArgumentsStop::Slots);
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
                return Err(ArgumentsStop::Images);
            }
            let stages = [
                (Stage::Vertex, &self.vertex),
                (Stage::Fragment, &self.fragment),
                (Stage::Object, &self.object),
                (Stage::Mesh, &self.mesh),
            ];
            let layout = &pipeline.base().layout;
            let now = crate::address_map::sampler_maps(layout, &stages, Some(&self.residency));
            if now.ok().as_ref() != Some(live.read_from.maps()) {
                return Err(ArgumentsStop::Images);
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
                return Err(ArgumentsStop::Unread);
            };
            let maps = live.read_from.maps();
            let from = (state, under);
            let reading = (stage, interface);
            if !table.refresh_changed(from, reading, (maps, views), &mut reshape, counts) {
                live.reshaped(reshape);
                counts.draw_rebind_reshaped(ReshapedBy::Views);
                return Err(ArgumentsStop::Reshaped);
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
        // Slots that may have reshaped the state: its shape found or made, and settled, as every
        // step finds and settles it (`RenderLowering::settle_draw`); the draw goes on where that
        // settles to the kept draw's variant and plan, which are all a rebind works out of.
        if reshaped {
            let shape = live.shape(counts, |state, key| pipeline.kept_shape(state, key, counts));
            let settled = pipeline.settle_shape(&shape, self.forms);
            let same =
                settled.is_ok_and(|settled| live.settle_again(pipeline, self.forms, settled));
            if !same {
                if let Some(by) = reshape.reshaped_by() {
                    counts.draw_rebind_reshaped(by);
                }
                return Err(ArgumentsStop::Reshaped);
            }
            changed.shape = Some(shape);
        }
        Ok(changed)
    }

    /// Work out again in `live` what the slots `changed` decide of the kept draw, `draw` of
    /// `buffers`, of other arguments where `reargued` (of another index buffer where `reindexed`,
    /// which is no draw moved: the module docs): its commands, set-0 writes and dynamic
    /// offsets, push bytes, vertex buffers and uses, with the checks a draw's steps take of them
    /// (the module docs). What it worked out where it did ([`Rebinding`]); else `None`, and the
    /// draw takes every step, which writes them all again, where it stopped left in `stop` where
    /// it was a step the arguments' count names (else it holds what it held).
    fn rebind(
        &mut self,
        call: &'static str,
        live: &mut LiveBind,
        changed: &Changed,
        (draw, buffers, (reargued, reindexed, carried)): (
            &DrawCall,
            &DrawBuffers<'_>,
            (bool, bool, bool),
        ),
        stop: &mut (ArgumentsStop, CarryStop),
    ) -> Option<Rebinding> {
        let slots = changed.list();
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_mut())
            .flatten();
        let Some(kept) = kept else {
            stop.0 = ArgumentsStop::Unsettled;
            return None;
        };
        let (settled, rebinds) = kept.rebinds(slots, &mut live.rebinds);
        // A reader through snapshots reads through them again as every step would; any other
        // draw reads its pass's colors through no snapshot, in place or after the split, as
        // every step would not either: else it takes every step, which decides its form anew
        // (`render/snapshot.rs`).
        let kept = self.again.kept.as_ref()?;
        let read_again = match &kept.reader {
            Some(reader) => match self.read_again(reader, &kept.settings, settled) {
                Some(read_again) => Some(read_again),
                None => {
                    self.core.counts().draw_reader_stopped();
                    stop.1 = CarryStop::Snapshot;
                    return None;
                }
            },
            None if self.reads_otherwise(&kept.settings) => {
                stop.1 = CarryStop::Snapshot;
                return None;
            }
            None => None,
        };
        let reads = read_again.is_some();
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
        // Nor one of another index buffer, whose uses meet a buffer the kept draw's did not.
        let between = (self.again.kept.as_ref()).is_some_and(|kept| kept.between);
        // Nor a reader's, whose snapshots' uses follow its plan's (`ReadAgain`).
        let moved = changed.moved && !between && !reindexed && !reads;
        if moved {
            live.kept.ranges.clear();
            (live.kept.ranges).extend_from_slice(&live.commands.descriptor_ranges);
        }
        // What the slots decide of its commands, by the plan's rebinds of them, made once for the
        // kept draw (`BindingPlan::rebind_draw_slots_into`).
        let rebound = match rebinds {
            Some(rebinds) => plan.rebind_draw_slots_with(rebinds, &live.state, &mut live.commands),
            None => Ok(false),
        };
        if rebound != Ok(true) {
            stop.0 = ArgumentsStop::Tables;
            return None;
        }
        // A draw of other arguments: what they decide of its commands, its Vulkan draw, push
        // fields and own uses (`BindingPlan::redraw_apart_into`); an error is every step's. Its
        // plan reads them no other way where a rebind of its slots goes on (above), and it is no
        // de-indexed, mesh or patch draw's (a kept draw's): where it answers `Ok(false)`, a
        // fixed-function vertex buffer's fetch window moves.
        if reargued {
            let from = (self.again.kept.as_ref()).map(|kept| kept.draw)?;
            match plan.redraw_apart_into(&live.state, (&from, draw), &mut live.commands) {
                Ok(true) => {}
                Ok(false) => {
                    if !plan.vertex_buffers.is_empty() {
                        stop.0 = ArgumentsStop::FetchWindow;
                    }
                    return None;
                }
                Err(_) => return None,
            }
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
        let reached = (self.check_vertex_reach(&compiled.vertex_input.pulled, draw)).is_ok();
        if !reached {
            stop.0 = ArgumentsStop::Reach;
            return None;
        }
        // The pass's fragment writes, a copy of which a stage before the fragments may read
        // (`render/pass_start_reads.rs`): none, so each stage reads its encoder's table.
        if !(moved || tables_checked()) || self.pass_start.fragments_wrote() {
            stop.0 = ArgumentsStop::Checks;
            return None;
        }
        // A draw that samples the texture of a color attachment of its pass reads the texture's
        // copy (`texture/attached_reads.rs`), which its tables bind and its shadows hold (below),
        // as every step's do; one that reads a color the pass clears behind its load actions
        // keeps nothing (`render.rs`), so its every step decides it, and so does one after a
        // barrier inside its instance, whose every step takes the split there or weighs whether
        // it reads what the instance drew (a kept draw that read a copy so is let go at the
        // barrier, `RenderLowering::keep_across_barrier`, but one whose changed slot binds the
        // attachment's texture now is not). A changed texture: none its
        // plan writes is a color attachment's (a draw that does records apart), and no
        // extended-range texture is bound through a shadow (`LiveBind`'s state), as for the kept
        // draw.
        let attached = !reads && self.samples_attachment(facts, Texture::attached_among);
        let images = changed.images();
        if (attached
            && self.loads_ordered
            && self.samples_attachment(facts, Texture::cleared_among))
            || (attached && (self.current.barriers || self.barrier))
            || (images
                && (self.storage_colors(facts) != 0 || !live.state.extended_shadows.is_empty()))
        {
            stop.0 = ArgumentsStop::Checks;
            return None;
        }
        let stages = [
            (Stage::Vertex, &self.vertex),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, &self.object),
            (Stage::Mesh, &self.mesh),
        ];
        // Its slots narrowed to the levels before an attached one, worked out as every step
        // works them out (`render/attached_levels.rs`): the kept draw's, else it takes every step.
        // A carried draw's kept draw narrowed none (`render/carried.rs`).
        let narrowed = match attached {
            true => {
                let device = self.core.device();
                let feedback = self.current.feedback;
                self.attached_slots(device, (&**plan, compiled), &stages, (feedback, false))
            }
            false => NarrowedSlots::default(),
        };
        let kept_narrowed = (self.again.kept.as_ref()).map(|kept| &kept.narrowed[..]);
        if !kept_narrowed.is_some_and(|kept| narrowed.listed().eq(kept.iter().copied())) {
            if carried {
                stop.1 = CarryStop::Attached;
            }
            stop.0 = ArgumentsStop::Narrowed;
            return None;
        }
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
            fan_out: None,
        };
        // Set 0's writes and dynamic offsets: a draw moved whose descriptor ranges are the kept
        // draw's binds the same buffers over the same ranges (`binding::rebind_into` writes a
        // buffer's descriptor of its buffer and range alone), so it keeps them.
        let writes_kept = moved && commands.descriptor_ranges == live.kept.ranges;
        // Its uses: where only buffer slots changed, the kept draw's with those slots' patched in
        // place, which reads the stages' tables alone, else made again whole, those of what the
        // encoder made resident kept (`binding::command_uses_into`); a carried draw's whole, as its
        // kept draw's are another encoder's (`render/carried.rs`), and a re-homed one's, as its
        // kept draw's were gathered into another instance (`Kept::rehomed`).
        // A draw of other arguments patches its index buffer's use too, as a moved slot's.
        let rehomed = (self.again.kept.as_ref()).is_some_and(|kept| kept.rehomed);
        let patch = (&**plan, &command, &stages[..]);
        let inside = match images || carried || rehomed || reads {
            true => None,
            false => {
                let moved_uses = (&mut live.kept.moved_at, &mut live.kept.moved);
                let slots_inside =
                    binding::patch_buffer_uses(patch, slots, &mut live.uses, moved_uses);
                match (slots_inside.ok().flatten(), reargued) {
                    (Some(inside), true) => {
                        let index = buffers.index.as_ref().map(DrawSource::metal);
                        let moved_uses = (&mut live.kept.moved_at, &mut live.kept.moved);
                        let own = (&**plan, &command);
                        let patched =
                            binding::patch_command_uses(own, index, &mut live.uses, moved_uses);
                        patched.ok().flatten().map(|own| inside && own)
                    }
                    (inside, _) => inside,
                }
            }
        };
        let patched = inside.is_some();
        // The tables a step that works the draw out again reads (its set-0 writes where they are
        // not the kept draw's, its uses where they were not patched), made only for one.
        let mut rebound = true;
        if !writes_kept || !patched {
            let mut untwinned = binding::Tables::new(&stages)
                .with_pass(&self.layout, plan.views)
                .with_residency(&self.residency);
            if attached {
                untwinned = untwinned
                    .with_attachments(&self.layout)
                    .with_narrowed(&narrowed);
            }
            if let Some(read_again) = &read_again {
                untwinned = untwinned.with_snapshots(&read_again.views);
            }
            // The slots it only reads through read twins, as every step binds them
            // (`read_twin.rs`).
            let cb = self.core.command_buffer();
            let roots = !command.root_tables.is_empty();
            let twins = facts
                .twins()
                .slots(cb, (&**plan, command.uses, roots), &untwinned);
            let mut tables = match &twins {
                Some(twins) => untwinned.with_twins(twins),
                None => untwinned,
            };
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
        // A reader's snapshots' uses after its plan's, as every step adds them.
        if let (true, Some(read_again)) = (rebound, &read_again) {
            (live.uses).extend(read_again.texels.iter().map(Texture::snapshot_use));
        }
        let rebound = rebound
            && binding::push_writes_into(&commands.push_constants, &[], &mut live.pushes).is_ok();
        // A narrowed slot whose levels a write of its own meets takes its copy at every step.
        if !rebound || (narrowed.len() > 0 && narrowed.written_by(&live.uses)) {
            stop.0 = ArgumentsStop::Worked;
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
        stop.0 = ArgumentsStop::VertexBuffers;
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
        // Its shadowed textures, worked out as every step works them out
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
        // It uses some exactly where the kept draw did: a draw that uses some is never kept for
        // recording again (`Kept::again`), which a draw rebound takes as the kept draw's.
        stop.0 = ArgumentsStop::Shadows;
        if shadows.is_empty() != live.kept.shadows.is_empty() {
            return None;
        }
        // Where they are the kept draw's, each used as it used it, its instance gathered them
        // already (`render/draw_again.rs`); else (another shadow, or a read's bounds a scissor
        // restated moved) it gathers them, as every step gathers them. So does a carried draw,
        // whose kept draw gathered its own into the ended encoder's instance
        // (`render/carried.rs`), and a re-homed one, whose kept draw gathered them into the
        // instance before the split (`Kept::rehomed`).
        let kept_uses =
            (shadows.iter().map(ShadowUse::used_as)).eq(live.kept.shadows.iter().copied());
        let gathers = carried || rehomed || !kept_uses;
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
        if !unbarriered {
            stop.0 = ArgumentsStop::Barrier;
            return None;
        }
        if !moved && self.core.would_skip(named()) {
            stop.0 = ArgumentsStop::Detached;
            return None;
        }
        let reader = read_again.map(|read_again| {
            let ReadAgain {
                weighed,
                feedback,
                texels,
                ..
            } = read_again;
            (weighed, feedback, texels.len())
        });
        Some(Rebinding {
            moved,
            writes_kept,
            narrowed,
            inside,
            through_copies: (attached, own_pixel),
            reindexed,
            shadows,
            gathers,
            reader,
        })
    }

    /// Whether the draw of the kept draw that reads through no snapshot, under `settings`, would
    /// read its pass's colors otherwise than a rebind records it at every step (the module docs,
    /// `render/snapshot.rs`), asked as every step asks it, with nothing counted or taken: in
    /// place, through snapshots it weighs within, or after the split, which a check or the area
    /// rule takes. Not where it reads no color so, or reads the copy taken before the pass, a
    /// check's or its weighing's (`texture/attached_reads.rs`), which a rebind works out as every
    /// step does.
    pub(super) fn reads_otherwise(&self, settings: &DrawSettings) -> bool {
        let pipeline: &DrawingState = match self.pipeline.as_ref().map(|p| p.kind()) {
            Some(RenderStateKind::Graphics(pipeline)) => pipeline,
            _ => return true,
        };
        if self.in_place_wanted(pipeline, &[], settings).is_some() {
            return true;
        }
        match self.snapshot_wanted_or(pipeline, &[]) {
            Err(NotWanted::Quiet) => false,
            Err(NotWanted::Check { split, .. }) => split,
            Err(NotWanted::Unreadable(_)) => true,
            Ok(wanted) => {
                let weighed = self.weigh_snapshot(pipeline, wanted.feedback, wanted.leading);
                wanted.split_otherwise || weighed.within
            }
        }
    }

    /// What the kept draw that reads through snapshots as `reader`, under `settings`, works out
    /// again to read through them as every step would with its settle `settled` (the module
    /// docs, `render/snapshot.rs`): its draw reads no color in place, asks to read the same slots
    /// of the same colors, pinning none; the plan reads through them; its instance holds them in
    /// the feedback loop layout, or a carried draw's yet unbegun instance will; and its form,
    /// weighed again, is the snapshot. Its slots' snapshots made or kept with their textures, as
    /// every step makes them. `None` where any of those differs, or a snapshot's view is refused,
    /// as every step's would be: the draw takes every step.
    fn read_again(
        &self,
        reader: &KeptReader,
        settings: &DrawSettings,
        settled: &Settled,
    ) -> Option<ReadAgain> {
        let pipeline: &DrawingState = match self.pipeline.as_ref().map(|p| p.kind()) {
            Some(RenderStateKind::Graphics(pipeline)) => pipeline,
            _ => return None,
        };
        self.core.device().shared().color_write_masks?;
        // A kept draw renders no color through a view of another format (`Kept`).
        if self.in_place_wanted(pipeline, &[], settings).is_some() {
            return None;
        }
        let wanted = self.snapshot_wanted_or(pipeline, &[]).ok()?;
        if !reader.reads_as(&wanted)
            || self.snapshot_holds(settled, &wanted).is_err()
            || self.snapshot_apart(&wanted)
        {
            return None;
        }
        let weighed = self.weigh_snapshot(pipeline, wanted.feedback, wanted.leading);
        if !weighed.within {
            return None;
        }
        let mut views = Vec::with_capacity(wanted.slots.len());
        let mut texels = Vec::with_capacity(wanted.slots.len());
        for &slot in &wanted.slots {
            let texture = self.fragment.textures.get(&slot)?;
            let form = snapshot::form(&settled.plan, slot)?;
            let made = texture
                .snapshot_texels(self.core.command_buffer(), form)
                .ok()?;
            views.push((slot, made.snapshot_view(form).ok()?));
            texels.push(made);
        }
        Some(ReadAgain {
            weighed,
            feedback: wanted.feedback,
            views: SnapshotViews {
                slots: wanted.slots,
                views,
                feedback: true,
            },
            texels,
        })
    }

    /// Record the draw rebound in `live` (the module docs): its set 0, its uses noted, its state
    /// commands and draw, and the rest a draw again takes. A refusal is set 0's or the noting's.
    fn record_rebound(
        &mut self,
        live: &mut LiveBind,
        (call, assembly, draw): (&'static str, Assembly, &DrawCall),
        buffers: &DrawBuffers<'_>,
        ((stamps, images), rebinding, query): (
            ([Stamp; 4], bool),
            &Rebinding,
            Option<(vk::QueryPool, u32)>,
        ),
    ) -> Result<(), Error> {
        let kept = settles_kept(self, live)
            .then_some(live.settled.as_ref())
            .flatten();
        let unsettled = || crate::error::internal("a draw rebound unsettled");
        let objects = kept.ok_or_else(unsettled)?.settled().variant.objects();
        // Of a draw carried in (`render/carried.rs`), what every step retains beyond its tables'
        // objects and index buffer: its pipeline state, its variant's objects and the `VkPipeline`
        // its recording names. The command buffer of one carried across a commit holds none of
        // them; the encoder of one carried from its command buffer's last render encoder retained
        // none of them, and its later draws retain only what no earlier draw of it retained
        // (`LastRetained`), as every step's does.
        let kept_made = (self.again.kept.as_ref()).filter(|kept| kept.carried);
        let across = (self.again.kept.as_ref()).is_some_and(|kept| kept.across);
        let carried_in = match (kept_made, self.pipeline.as_ref().map(|state| state.kind())) {
            (None, _) => None,
            (Some(kept), Some(RenderStateKind::Graphics(state))) => {
                let made = Some(Arc::clone(&kept.made.0));
                Some((state.clone(), Arc::clone(objects), made))
            }
            (Some(_), _) => {
                return Err(crate::error::internal(
                    "a draw carried without its pipeline state",
                ))
            }
        };
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
        // A reader's snapshots' uses, its last, write its snapshots, images the pass's reads take
        // nothing of (`render/pass_start_reads.rs`), and are unordered writes as every step's.
        let snapshots = rebinding
            .reader
            .as_ref()
            .map_or(0, |&(_, _, snapshots)| snapshots);
        let planned = uses.len() - snapshots;
        debug_assert!(
            super::unwritten(&uses[..planned]).0 && !self.pass_start.fragments_wrote(),
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
        match prefix {
            Some(prefix) => {
                if self.unordered.add_draw((uses, prefix)) {
                    self.core.counts().resident_uses_unordered_writes(resident);
                }
            }
            None => self.unordered.add(&uses[planned..]),
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
        let gathers = match rebinding.gathers {
            true => &rebinding.shadows[..],
            false => &[],
        };
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
            // Its shadows with them where it gathers them (`Rebinding::gathers`), as every step's.
            (false, None) => self.gathering().add(gathers, uses),
            (false, Some(prefix)) => {
                if self.gathering().add_draw(gathers, (uses, prefix)) {
                    self.core.counts().resident_uses_gathered(prefix.len as u64);
                }
            }
        }
        // A draw narrowed's apart from its uses, as `InstanceShadows::add` joins them.
        if narrowed {
            self.gathering().join(gathers);
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
        let mut recording = Recording {
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
            multi: shared.multi_draws.as_ref(),
            draw: owned.draw,
            pair: None,
        };
        // A reader's two draws, each its phase's first, the drawing one inside its query, as
        // every step records them (`render/snapshot.rs`).
        let reader = (self.again.kept.as_ref()).and_then(|kept| kept.reader.as_ref());
        let pairs = match (reader, shared.color_write_masks) {
            (Some(reader), Some(masks)) => Some(reader.pairs(masks)),
            (Some(_), None) => {
                return Err(crate::error::internal(
                    "a reader rebound without the write mask command",
                ))
            }
            (None, _) => None,
        };
        let mut recorded = Recorded::default();
        let raw = shared.vk();
        let cb = self.current.draws;
        // An instance recorded inline after a split keeps its draws' recordings in its room until
        // its copies are known, recorded then through the pass's record (`render/inline_splits.rs`):
        // this one's too, through no record of its own.
        let room = self.current.kept.is_some();
        match (pairs, &mut self.current.kept) {
            (None, Some(held)) => held.draw(&recording, owned.pipeline),
            (None, None) => {
                // SAFETY: as the kept draw's recording: the secondary command buffer (or the
                // pass's, inline) records the kept draw's instance, and only this encoder
                // records it; every object recorded is live until the command buffer completes
                // (the kept draw's retained, and the tables' retained below); `bound` is the
                // record of what its draws bound there.
                recorded += unsafe { recording.record(bound) };
            }
            // A reader through snapshots never records into such an instance (its admission),
            // nor is it kept into one (`RenderLowering::rehome_aside`).
            (Some(_), Some(_)) => {
                return Err(crate::error::internal(
                    "a reader rebound into an instance recorded inline after a split",
                ))
            }
            (Some(pairs), None) => {
                for pair in pairs {
                    let counted = query.filter(|_| pair.phase == Phase::Drawing);
                    // SAFETY: as above; the pass is one `render_target` made, of no views, and
                    // the query the pass's own, reset on the host and begun nowhere else, with no
                    // other occlusion query active (`RenderLowering::snapshot_wanted`).
                    unsafe {
                        if let Some((pool, query)) = counted {
                            raw.cmd_begin_query(cb, pool, query, vk::QueryControlFlags::PRECISE);
                        }
                        recording.pair = Some(pair);
                        recorded += recording.record(bound);
                        if let Some((pool, query)) = counted {
                            raw.cmd_end_query(cb, pool, query);
                        }
                    }
                }
            }
        }
        let settled = bound.records();
        if let Some(written) = written {
            self.core.keep_set_zero(written, &live.kept.writes);
        }
        let kept = (self.again.kept.as_mut())
            .ok_or_else(|| crate::error::internal("a draw rebound without its kept draw"))?;
        kept.stamps = stamps;
        kept.settled = (!room).then_some(settled);
        kept.between = false;
        kept.stated = stated;
        kept.rehomed = false;
        // A carried draw is the encoder's own from here, recorded into the instance it began, with
        // its reads through copies; the shadows' uses a draw gathered (a carried draw's among
        // them) are those the next draw rebound compares.
        let carried = kept.carried;
        kept.carried = false;
        kept.across = false;
        kept.command_buffer = self.current.draws;
        kept.through_copies = rebinding.through_copies;
        if rebinding.gathers {
            live.kept.keep_shadows(&rebinding.shadows);
        }
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
        if carried {
            counts.draw_rebound_carried();
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
        // Its index buffer where it is another than the kept draw's, or the draw carried across.
        let reindexed = rebinding.reindexed || across;
        if let (true, Some(source)) = (reindexed, buffers.index.as_ref()) {
            self.core.retain_shared(source.buffer.held());
        }
        if let Some(retained) = carried_in {
            self.last_retained.retain_draw(&mut self.core, retained);
        }
        self.last_retained
            .retain_samplers(&mut self.core, &mut live.samplers);
        let under = self.kept_under();
        if let Some(kept) = self.again.kept.as_mut() {
            kept.under = under;
        }
        self.draw_calls += 1;
        self.last_draw_empty = false;
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
