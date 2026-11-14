//! A render encoder's last direct draw, kept so that the next draw of the same call with the same
//! arguments, which no call of the encoder came between, records its Vulkan draw alone
//! (`WorkCounts::draws_as_before`).
//!
//! - **What is kept.** Of its last `draw_primitives` or `draw_indexed_primitives` call, the
//!   encoder keeps ([`Kept`]) the call, what it assembled, its draw (the binding layer's, indices
//!   and offsets included) and its index source; what it read of the state that changes with no
//!   call of the encoder ([`Under`]); the command buffer it recorded into and the Vulkan draw
//!   command it recorded; and what its other steps did that a draw again repeats (its per-draw
//!   counts, its execution record, whether its instance keeps it for recording again).
//! - **Only a plain draw is kept.** One that recorded one Vulkan draw into the current instance as
//!   it stood, or into its pass's first instance begun inline at it (`render/inline_first.rs`:
//!   the next draw finds the instance begun, its record holding the states the kept draw set),
//!   an instance recorded inline after a split among them, whose room keeps its draws'
//!   recordings until its copies are known (`render/inline_splits.rs`): a kept draw's next draw
//!   there is rebound (`rebound.rs`), its recording kept in the room as every step's is, never a
//!   Vulkan draw recorded ahead of the room's,
//!   through a pipeline made ahead, needing nothing before or around it of its own: no
//!   split, stand-in, snapshot (but a reader's, below), copy, index bounds or rewrite pass, mesh or
//!   tessellation work, rate map, color groups, extended-range color or its shadow, set 1 or
//!   handle table, root table or push hole (but a reader's phase), barrier in its instance, read
//!   of a color its pass clears behind its load actions (a copy cleared to the clear,
//!   `texture/attached_reads.rs`), read of input attachments ordered between draws alone beside a
//!   color write (the next such draw records after a barrier: `render.rs`'s
//!   `InterlockedStorage`), and no write among its uses (but a reader's of its snapshots), which
//!   each change what the next draw meets or are work of the draw's own. Its uses are noted as the
//!   next draw would note them (`NotedApart`). A draw that met a notice its device announces or
//!   refuses at every occurrence (`EachWatch`) keeps nothing: its next draw must meet it again.
//! - **Shadowed textures gathered once.** A kept draw may use shadowed textures, a texture of its
//!   pass's color attachment it reads through the texture's copy among them
//!   (`texture/attached_reads.rs`): their copies go around its instance, which gathered them as
//!   the draw took every step (`render/shadows.rs`), so a draw that uses the same shadowed
//!   textures as the same uses into the same instance gathers nothing new (`shadow::join` of an
//!   equal use changes nothing). A draw again uses them so, through the kept draw's bindings; a
//!   draw rebound works out its own (`render.rs`'s `draw_shadows`, as every step does) and goes
//!   on only where they are the kept draw's, each the same shadow used as it was
//!   (`shadow::ShadowUse::used_as`), its reads' bounds included (`rebound.rs`). Each counts the
//!   kept draw's reads through copies (`WorkCounts::attachment_reads_through_copies`,
//!   `own_pixel_reads`) as it did.
//! - **Forgotten at every call but a bind, a pipeline state set, a render state set, a fence, a
//!   use or a barrier inside the pass.** Every call of the encoder but the two direct draws, the
//!   vertex and fragment buffer,
//!   bytes, texture and sampler state binds, `set_render_pipeline_state`, the sets of its render
//!   state (below) and the calls kept across (below) forgets it as it logs itself
//!   (`RenderLowering::log`, which every call does: at the call, on the device's lowering thread,
//!   and encoded again), refused or not, whatever it changes, with every kept draw set aside
//!   (below), counted by the call's kind (`WorkCounts::kept_draws_let_go_*`); so does a draw that
//!   keeps nothing (a draw again, a draw rebound and a draw kept are the only draws that leave
//!   them), but a reader through snapshots that writes nothing else, and a draw that keeps
//!   nothing only for a split, which re-homes them (below). A bind changes its
//!   stage table's slots alone (and a bytes bind takes room in its command buffer's uploads), and a
//!   table's stamp tells each change apart (`argument_table::Stamp`): the kept draw keeps the four
//!   tables' stamps as it read them, and a draw whose tables kept them is the kept draw again; one
//!   after binds is drawn again rebound where it can be (`rebound.rs`), else takes every step. So
//!   the encoder's state a draw resolves from is the kept draw's, but for the tables' slots the
//!   stamps tell apart, what the kept draw itself left and what changes with no call of the
//!   encoder: the latter is [`Under`], compared at each draw: the count of no-copy buffers' moves
//!   (a bound buffer's binding, a detach), the residency's generation once the draw made the heaps'
//!   new textures resident, whether the device has buffers over host pages and its host aliases'
//!   version, the encoder's parts and held images (what noting reads, `noted_apart.rs`), with the
//!   command buffer the current instance records into, and the count of the device's sampler IDs'
//!   slots' changes (a sampler state made or gone, which every step of a draw lists again:
//!   `sampler::ArgumentSamplers`). Another thread may change the moves, the sampler slots and the
//!   buffers over host pages with their aliases' version while a draw takes its steps (the device's
//!   in-place check makes and lets go buffers over host pages on a thread of its own): a draw is
//!   kept under them as its steps began, which they read ([`Outside`]), and its uses' noting counts
//!   only where it was noted under them, so a change made meanwhile makes the next draw take every
//!   step.
//! - **Kept across a fence, a use or a barrier inside the pass.** A call that changes nothing a
//!   kept draw read or recorded keeps every kept draw as it was (`RenderLowering::log_across`,
//!   `WorkCounts::kept_draws_kept_across_*`): `update_fence`, which only notes the update for the
//!   encoder's end (`tracker::EncoderUses::update_fence`); `wait_for_fence` where the encoder used
//!   no shadowed texture, which notes the wait, all of the encoder's work waiting, and carries no
//!   shadowed texture's image on (`shadow::ShadowSync::not_carried`), which no kept draw of the
//!   encoder read (one carried in works its own shadows out again as it is rebound, `rebound.rs`);
//!   where the encoder used one, its later commands take its copies again, which a draw again would
//!   not, so the wait forgets; and `use_resource`, `use_resources`, `use_heap` and `use_heaps` but
//!   of a memoryless texture (which fails the command buffer, and forgets): a use that makes a
//!   resource resident or widens a usage, stages or a heap's textures gives the residency a new
//!   generation (`residency.rs`), which [`Under`] holds, so the next draw of each kept draw takes
//!   every step, as every draw that reaches the residency must list it again; one that changes
//!   nothing makes nothing resident and holds nothing more, so the next draw is the kept draw's
//!   again. A use for writing of an indirect command buffer marks its write routes
//!   (`write_routes.rs`) as at any use, which no draw reads. And `memory_barrier_with_scope` over
//!   the render targets alone where it records its barrier inside the current instance or nothing
//!   (`fence.rs`; one that puts the pass's colors apart there splits the pass, and forgets): a
//!   barrier changes no state its command buffer holds, so each kept draw's next draw records into
//!   it as before, drawn again or rebound, but for the kept draws that read the pass's colors
//!   other than through snapshots of their own (through their textures' copies, at levels apart
//!   from the attached ones, or by a use of their memory, as past a reader below), whose every
//!   step after the barrier takes a split or a copy again: those it forgets
//!   (`RenderLowering::keep_across_barrier`). A reader through snapshots reads behind its own
//!   `B1` with the barrier or without, and weighs its form again at its next draw as below.
//! - **One per pipeline state.** A pipeline state set keeps the kept draw: at the next draw
//!   through another state, it goes aside with its state's bind state and the storage its draws
//!   wrote (`LiveBind::select`, `RenderLowering::select_kept`), and the draw starts from its own
//!   state's, with the draw kept beside it, where one is under the encoder's epoch (which every
//!   forgetting draws afresh: `DrawAgain`). Such a draw is between (`Kept::between`): draws of
//!   other states recorded since, each a draw kept or drawn again, or a reader through snapshots
//!   that wrote none of what it reads (below), into the same instance, whose steps only add to
//!   what the encoder holds (below). Its next draw is rebound whatever its
//!   tables' stamps (`rebound.rs`), the changes of its tables since it read them brought in, its
//!   uses noted, added to the pass's reads and gathered whole, and every state command its
//!   record does not hold recorded; it takes no shortcut of a draw moved, which reads what the
//!   encoder's last draw left. Each state's bind state is its own, though states of one function
//!   pair share an interface, so the draw kept of one is never the other's.
//! - **A reader through snapshots** (`render/snapshot.rs`) is kept as a plain draw is
//!   (`snapshot::KeptReader`), but for its two Vulkan draws, its phase's
//!   push hole and its snapshots' writes, where it reads no color in place and pins none of its
//!   colors to their images (the pass's end decides from those). Its draw again records both
//!   draws and their barriers as its first recording did, but for the state commands, which its
//!   command buffer holds: `B1`, the snapshot draw's masks 0 and phase 1 and its draw, then `B2`,
//!   the drawing draw's own masks and phase 0 and its draw, inside its query where it is measured,
//!   every snapshot draw of the Metal draw before every drawing draw. `B1` orders every color
//!   write before it, so it reads what the draws before it wrote, as every step's reader does.
//!   It weighs its form again as every step weighs it (`RenderLowering::weigh_snapshot`): its
//!   identity among the pass's readers and what its texture learned it covered change with no
//!   call (`render/coverage.rs`). Where it would take another form it takes every step
//!   (`WorkCounts::reading_draws_snapshotted_stopped`); else it takes its identity, count and
//!   query as every step takes them (`WorkCounts::reading_draws_snapshotted_again`). A draw
//!   rebound of it works out again that it reads through snapshots as every step would
//!   (`rebound.rs`). A kept draw that reads through none, drawn again or rebound, takes every
//!   step where every step would read its pass's colors through snapshots, in place or after the
//!   split (`RenderLowering::reads_otherwise`): one that read the copy taken before the pass by
//!   the area rule weighs its form again too.
//! - **Past a reader through snapshots.** Where a reader records into its instance as it stood,
//!   of the encoder's own colors, as a direct draw, and no use of it but its snapshots' writes,
//!   what it writes is its pass's colors and its snapshots, kept or not: it forgets only the kept
//!   draws set aside that read them (`RenderLowering::forget_aside_readers`; one that keeps
//!   nothing forgets the last direct draw's too), each that reads a color's texture through its
//!   copy (`Kept::through_copies`) or at levels apart from the attached ones
//!   (`render/attached_levels.rs`), or whose uses meet a color's memory
//!   (`render_target::shares_attachment`), but a kept reader through snapshots of its own, whose
//!   next draw takes its snapshots again behind its own `B1`. Only readers bind snapshots. The
//!   others stay kept under the epoch, between, so the next draw of each is rebound
//!   (`rebound.rs`), recording every state command its instance's record does not hold, as after
//!   a draw kept of another state. The reader's other steps add to what the encoder holds as a
//!   kept draw's do; its barriers are inside the subpass, by region; and its snapshots are images,
//!   so the pass's fragments still wrote no buffer bytes a stage before them reads
//!   (`render/pass_start_reads.rs`). `tests::snapshot_reader_keeps` checks it.
//! - **Of other arguments.** A draw that differs from the kept draw only in its arguments (its
//!   call of the same primitive type, with indices or without, its vertex or index counts and
//!   starts, its index buffer, index type or index buffer's offset, each read in place, its
//!   instance count, base vertex or base instance: `UnlikeKept::arguments_alone`), of something to
//!   draw, is rebound with them (`rebound.rs`), whatever its tables' stamps: the steps that read
//!   them are worked out again, the rest are the kept draw's. Every other draw unlike the kept
//!   draw takes every step (`WorkCounts::draws_unlike_kept`, by what differed).
//! - **Restated by a render state set.** A call that sets the encoder's render state alone (its
//!   viewports, scissor rectangles, cull mode, winding, depth/stencil state, stencil references,
//!   depth bias, depth clip mode, fill mode or blend color: `RenderLowering::log_restating`)
//!   keeps every kept draw, restated: the encoder's state stamp is drawn afresh (`DrawAgain`), and
//!   a kept draw whose stamp is not the encoder's derives again, at its next draw, everything the
//!   state decides of it (`restated.rs`). It is then rebound (`rebound.rs`), recording its new
//!   draw state's dynamic states against what its record holds, where what it derived leaves
//!   its pipeline, plan, shape and push bytes the kept draw's; else it takes every step. Where the
//!   encoder set its scissor rectangles alone since the kept draw's stamp ([`Stated`]), which
//!   nothing else the state decides of a draw reads, a kept draw whose route meets every condition
//!   of a restate ([`Kept::restates_by_scissors`]) takes the new scissors alone into its draw
//!   state and derives nothing else again (`WorkCounts::draws_restated_scissors`).
//! - **Re-homed past a split.** A draw that keeps nothing only because it recorded into a later
//!   render pass instance of its pass than the one it found (a split, not a continuation or a
//!   stand-in's arrival: the read of a color attachment's texture after a barrier among them),
//!   of the shape, views and formats of the one before, its other steps those of a plain draw,
//!   wrote nothing but its pass's colors (and their copies the split takes): it keeps the kept
//!   draws set aside with the states of other pipeline states into its instance
//!   (`RenderLowering::rehome_aside`, `WorkCounts::kept_draws_rehomed`), but those that read
//!   its pass's colors other than through snapshots of their own (as past a reader below), the
//!   readers through snapshots, the draws carried in and not drawn since, and those that read
//!   what changes with no call otherwise than it did, which it lets go. Each is kept there as a
//!   carried draw is kept in the instance its encoder opened with ([`Kept::rehomed`]): after
//!   draws of other states, recorded through no record, under another render state, so its
//!   next draw is rebound (`rebound.rs`), deriving again what the render state decides and
//!   adding the pixels it may write to the instance's drawn area (`restated.rs`), its uses
//!   noted, added to the pass's reads and gathered whole into the new instance with its
//!   shadowed textures' (which the copies before it then take), where it uses those exactly as
//!   the kept draw did, and every state command it sets recorded against what that instance's
//!   record holds. Its tables' stamps are its own still: its encoder's tables went on.
//! - **Carried to the next encoder.** An encoder that ends with a kept draw of its pass's first
//!   instance carries it, with its bind state and settles and the kept draws set aside with the
//!   states of up to seven other pipeline states (`live_bind::SET_ASIDE`), into its command
//!   buffer's next render encoder where nothing comes between, or past its commit into a render
//!   encoder of a later command buffer of its queue, where that encoder's pass is its pass's like
//!   (`render/carried.rs`): each draw is kept there as one after draws of other states, recorded
//!   through no record, under another render state ([`Kept::carried`]), in the instance that
//!   encoder opened with, so its state's first draw there is rebound (`rebound.rs`), bringing in
//!   every slot that encoder's calls bound and gathering its own shadowed textures' uses (a
//!   texture of that pass's color attachment it reads through the texture's copy among them)
//!   into that instance, as every step gathers them.
//! - **What a kept draw left holds for the next.** Each of its steps adds to what the encoder
//!   holds, and adding the same again changes nothing: its drawn area and coverage (rectangles
//!   and boxes, `clear_reach::DrawnArea`, `linear_attachment::Coverage`), the pass's reads
//!   (`PassStartReads::add`, which takes an equal range once) and the instance's gathered uses
//!   (`InstanceShadows::add`, which skips the uses of the last draw and joins its shadowed
//!   textures' as nothing new), its writes (none), its noted
//!   uses, its retains (each a hold of an object held already: `LastRetained`, a table's stamp,
//!   an index buffer the kept draw held), the shape memo's entries and the bind state (found as
//!   the kept draw left them), and the pass's first instance, which the kept draw decided
//!   (`render/inline_first.rs`). A notice the kept draw met is announced once per device, which it
//!   was. Its Vulkan state commands are those its instance's record holds (`render/bound.rs`): a
//!   draw again records no state command.
//! - **What a draw again does.** It logs itself, makes resident the heaps' new textures, notes its
//!   uses' reads of fresh textures (`fresh_unwritten.rs`), records the kept Vulkan draw command
//!   into the kept command buffer, keeps itself for recording again where its instance keeps draws
//!   (`render/reroute.rs`), adds its execution record where the device keeps them, and counts
//!   itself drawn. It takes no lock and changes no count other encoders share: its counts, and
//!   those the kept draw counted per draw, go into its encoder's tally of the device's
//!   (`shards.rs`).
//! - **Checked in builds with debug assertions.** The first draw again of a kept draw, and every
//!   second one after it, takes every step instead, and checks that it announced nothing, recorded
//!   no state command, kept an equal draw, retained only its index buffer again, and left the
//!   encoder's drawn area, coverage, reads, gathered uses, noted uses and instance as they were;
//!   where another thread changed what it reads meanwhile ([`Outside`]), its steps read another
//!   state than the kept draw's, and it checks nothing.

mod rebound;
mod restated;

use ash::vk;
use metal2vulkan::bind::{BindShape, DrawCall};
use std::sync::Arc;
use std::time::Duration;

pub(super) use rebound::KeptRecording;
pub(crate) use rebound::KeptStorage;

use super::{Assembly, DrawBuffers, DrawSettings, RenderLowering};
use crate::argument_table::Stamp;
use crate::degradation::EachWatch;
use crate::draw_source::DrawSource;
use crate::error::Error;
use crate::execution::{ExecutedPipeline, Execution};
use crate::live_bind::LiveBind;
use crate::noted_apart::NotedInto;
use crate::pipeline::{RawPipeline, RenderPipelineState, RenderStateKind, WeakRenderPipelineState};
use crate::render_pass::PassLayout;
use crate::render_target::AttachedMemo;
use crate::resource::ResourceKey;
use crate::tracker::{Use, UseRange};
use crate::work_counts::{ArgumentsStop, LoggedCall, UnlikeKept};
use metal2vulkan::metal::RenderStages;

/// The encoder's last direct draw, where it is kept (the module docs).
#[derive(Default)]
pub(super) struct DrawAgain {
    kept: Option<Kept>,
    /// What the encoder's kept draws were kept under (a fresh stamp, its default): drawn afresh
    /// as every kept draw is forgotten, the one of the last direct draw and those its bind state
    /// set aside with the states of other pipeline states (`LiveBind::select`), so a draw set
    /// aside is kept still where it holds this.
    epoch: Stamp,
    /// The encoder's render state's stamps (fresh stamps, their default): drawn afresh at each
    /// call that sets it ([`restate_again`](RenderLowering::restate_again)), so a kept draw that
    /// holds others is restated (the module docs).
    stated: Stated,
    /// What another thread may change of what the encoder's current draw reads, as its steps
    /// began ([`Outside`], [`RenderLowering::draw_began`]).
    began: Outside,
    /// Draws again since the draw was kept: in builds with debug assertions, the first and every
    /// second one after it take every step instead (the module docs).
    #[cfg(debug_assertions)]
    hits: u64,
    /// Kept draws carried in and rebound since the command buffer's first render encoder: in
    /// builds with debug assertions, the first and every second one after it take every step
    /// instead (`render/carried.rs`).
    #[cfg(debug_assertions)]
    carries: u64,
    /// The version of the instance the encoder opened with (`Versioned::version` of its shape),
    /// where it started from kept draws carried in (`render/carried.rs`): while the current
    /// instance is that one, as it was, it is the pass's first, which a kept draw carried in
    /// records into, whichever draw began it. `None` where none was carried in.
    opened: Option<u64>,
}

/// The encoder's render state's stamps ([`DrawAgain`]): one drawn afresh at each call that sets
/// it, the other at each such call but a set of its scissor rectangles. A kept draw holds both as
/// it was kept or last rebound ([`Kept::stated`]): where the first moved since and the second did
/// not, the encoder set its scissor rectangles alone, which no part of a draw's route, settings,
/// bind state or draw state but the draw state's scissors reads (`restated.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stated {
    set: Stamp,
    beyond_scissors: Stamp,
}

impl Stated {
    /// Whether, of a kept draw that holds these stamps, the encoder whose stamps are `now` set its
    /// scissor rectangles alone since (and set them).
    pub(super) fn scissors_alone(self, now: Stated) -> bool {
        self.set != now.set && self.beyond_scissors == now.beyond_scissors
    }
}

/// What a kept draw was, what it read, and what a draw again repeats of it (the module docs).
#[derive(PartialEq)]
pub(crate) struct Kept {
    /// The pipeline state it drew, and the encoder's epoch it was kept under ([`DrawAgain`]).
    pub(super) pipeline: WeakRenderPipelineState,
    pub(super) epoch: Stamp,
    /// The encoder's render state's stamps as it was kept or last rebound ([`DrawAgain`]), and the
    /// settings it drew under then: a draw of it under other stamps derives them again
    /// (`restated.rs`).
    pub(super) stated: Stated,
    pub(super) settings: Arc<DrawSettings>,
    /// Whether its route meets every condition a restate asks of it (`restated.rs`): no
    /// depth/stencil state failing every fragment, no hidden depth or stencil given, no strip
    /// listed, a provoking vertex drawn as it is or natively. None of them reads the scissor
    /// rectangles, so a restate by them alone ([`Stated::scissors_alone`]) takes them alone.
    pub(super) restates_by_scissors: bool,
    /// Whether another draw recorded since it, or since the last draw again of it: it was set
    /// aside with its pipeline state's bind state while draws of other states recorded
    /// (`LiveBind::select`). Its next draw is then rebound whatever its tables' stamps
    /// (`rebound.rs`), recording every state command its record does not hold.
    pub(super) between: bool,
    pub(super) call: &'static str,
    pub(super) assembly: Assembly,
    pub(super) draw: DrawCall,
    pub(super) index: Option<KeptIndex>,
    pub(super) under: Under,
    pub(super) command_buffer: vk::CommandBuffer,
    pub(super) command: metal2vulkan::vk::DrawCommand,
    /// Whether its instance may record it again (`RenderLowering::keep_draw`).
    pub(super) again: bool,
    /// What it counted per draw beyond its own draw: a push of set 0, a `VkPipeline` of the last
    /// provoking vertex.
    pub(super) pushes_set_zero: bool,
    pub(super) last_vertex: bool,
    /// Whether it read the texture of a color attachment of its pass through its copy, and of
    /// that, at its fragments' own pixels alone (`WorkCounts::attachment_reads_through_copies`,
    /// `own_pixel_reads`), which a draw again counts as it did.
    pub(super) through_copies: (bool, bool),
    /// Its sampled slots narrowed to the levels of their texture apart from the attached ones,
    /// each with its levels (`render/attached_levels.rs`): a draw rebound is rebound only where it
    /// narrows the same.
    pub(super) narrowed: Vec<(
        metal2vulkan::pipeline::SlotRef,
        crate::binding::NarrowedLevels,
    )>,
    /// Whether one of its uses reads an image other than as an attachment: its noting looks
    /// for fresh textures among them (`EncoderCore::note_again`).
    pub(super) reads_images: bool,
    /// Its execution record's parts, where the device keeps records.
    pub(super) execution: Option<KeptExecution>,
    /// What `RenderLowering::draw` takes of it: the stages whose functions read texture handles
    /// from memory, and whether it rasterizes and writes depth.
    pub(super) reads: RenderStages,
    pub(super) rasterizes: bool,
    pub(super) depth_writes: bool,
    /// Whether it recorded no state command (`Recorded::draw_alone`), as a draw again does.
    pub(super) draw_alone: bool,
    /// The stamps of its vertex, fragment, object and mesh tables as it read them: a draw whose
    /// tables kept them binds as it did.
    pub(super) stamps: [Stamp; 4],
    /// Whether its encoder's bind state keeps what a draw rebound takes of it (`LiveBind::kept`,
    /// `rebound.rs`).
    pub(super) rebinds: bool,
    /// The count of its instance's record's recordings just after its recording, or the last
    /// draw rebound after it, recorded through it (`Bound::records`): while the record's count is
    /// this, the record holds the pipeline and dynamic states it set, which a draw rebound then
    /// compares with none. `None` where it recorded through none.
    pub(super) settled: Option<u64>,
    /// Whether its command buffer's last render encoder carried it into this one, which has not
    /// drawn it since (`render/carried.rs`): its tables' stamps are the encoder's tables' first
    /// ones, and its next draw is rebound as a carried draw is.
    pub(super) carried: bool,
    /// Whether it was carried in from the last render encoder of a command buffer committed on
    /// the queue (`render/carried.rs`), and not drawn since: its next draw retains into this
    /// encoder's command buffer what every step retains beyond the tables' objects, which that
    /// command buffer may hold no longer (`rebound.rs`).
    pub(super) across: bool,
    /// Whether it was set aside when a draw that kept nothing only for recording into a later
    /// render pass instance of its pass began that instance, and was kept into it
    /// ([`rehomed_into`](Kept::rehomed_into), the module docs), and has not been drawn since:
    /// its next draw is rebound, gathering its uses whole into that instance and its shadowed
    /// textures' uses with them (`rebound.rs`).
    pub(super) rehomed: bool,
    /// The `VkPipeline` it recorded through, held while it is kept: its recording names it by
    /// handle, and a draw carried across retains it (`render/carried.rs`).
    pub(super) made: KeptPipeline,
    /// What it repeats of its two draws, where it reads through snapshots
    /// (`render/snapshot.rs`; the module docs).
    pub(super) reader: Option<super::snapshot::KeptReader>,
}

/// A kept draw's `VkPipeline` ([`Kept::made`]): the same where it is the same object.
pub(crate) struct KeptPipeline(pub(super) Arc<RawPipeline>);

impl PartialEq for KeptPipeline {
    fn eq(&self, other: &KeptPipeline) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Kept {
    /// Carry this draw into a render encoder of epoch `epoch` whose tables' stamps are `stamps`
    /// at its opening, whose `Under` is `under` there and whose first instance records into
    /// `draws` (`render/carried.rs`): of a later command buffer than the one it was kept in where
    /// `across` (a draw set aside, carried again before it is drawn, stays across).
    pub(super) fn carried_into(
        &mut self,
        epoch: Stamp,
        stamps: [Stamp; 4],
        under: Under,
        (draws, across): (vk::CommandBuffer, bool),
    ) {
        self.across |= across;
        self.epoch = epoch;
        // Stamps no render state had: its first draw derives again what the encoder's render
        // state decides (`restated.rs`).
        self.stated = Stated::default();
        self.between = true;
        self.stamps = stamps;
        self.under = under;
        self.command_buffer = draws;
        self.settled = None;
        self.carried = true;
    }

    /// Whether what the draw read that changes with no call is as it was, but what it was noted
    /// into: `moved` no-copy buffers' moves, a residency of `residency`'s generation, and the
    /// device's sampler IDs' slots changed `samplers` times (`render/carried.rs`).
    pub(super) fn carried_under(&self, moved: u64, residency: u64, samplers: u64) -> bool {
        let under = &self.under;
        (under.moved, under.residency, under.samplers) == (moved, residency, samplers)
    }

    /// Keep this draw, set aside, into the render pass instance a draw that kept nothing only for
    /// recording into it began after a split, of the shape of the one before (the module docs):
    /// recording into `draws` through no record of that instance's, under a render state of
    /// another stamp (so its next draw derives again what that state decides, adding the pixels
    /// it may write to the instance's drawn area: `restated.rs`), kept under `under`, what the
    /// draw that began it was kept under. Where what it read that changes with no call is not as
    /// `under` holds it, but what it was noted into (which its next draw, rebound, notes its uses
    /// into whole), it is not kept: `false`.
    pub(super) fn rehomed_into(&mut self, under: Under, draws: vk::CommandBuffer) -> bool {
        let read = |under: &Under| {
            let into = under.into;
            (
                under.moved,
                under.residency,
                under.samplers,
                into.rebinds,
                into.aliases,
            )
        };
        if read(&self.under) != read(&under) {
            return false;
        }
        self.stated = Stated::default();
        self.between = true;
        self.under = under;
        self.command_buffer = draws;
        self.settled = None;
        self.rehomed = true;
        true
    }

    /// Whether the draw read a texture an attachment of its pass renders through a sampled slot
    /// narrowed to levels apart from the attached ones (`render/attached_levels.rs`). One that
    /// reads it through its copy carries: the next encoder's draw rebound gathers its own copies
    /// (`rebound.rs`).
    fn reads_attached(&self) -> bool {
        !self.narrowed.is_empty()
    }

    /// Whether it reads its pass's colors (`memo` over the pass's `layout` of `version`, its
    /// state's last draw's `uses` its own) other than through snapshots of its own: through their
    /// textures' copies, at levels apart from the attached ones, or by a use of their memory
    /// (`render_target::shares_attachment`). A reader through snapshots reads them behind its
    /// own barrier, which its next draw takes again (the module docs).
    fn reads_colors(
        &self,
        (memo, layout, version): (&mut AttachedMemo, &PassLayout, u64),
        uses: &[Use<ResourceKey>],
    ) -> Result<bool, Error> {
        if self.reader.is_some() {
            return Ok(false);
        }
        let attached = self.through_copies.0 || self.reads_attached();
        Ok(attached || memo.shares((layout, version), uses)?)
    }

    /// Whether the draw, of its own, may be carried into the next render encoder
    /// (`render/carried.rs`): it rebinds, with no sampled slot narrowed, under a depth/stencil
    /// state that does not fail every fragment.
    pub(super) fn carries(&self) -> bool {
        self.carries_but_for_reads() && !self.reads_attached()
    }

    /// Whether the draw would carry of its own ([`carries`](Self::carries)) but for its reads of
    /// a texture an attachment of its pass renders (`WorkCounts::kept_draws_uncarried_attached`).
    pub(super) fn carries_but_attached(&self) -> bool {
        self.carries_but_for_reads() && self.reads_attached()
    }

    /// Whether the draw rebinds under a depth/stencil state that does not fail every fragment.
    /// One that fails every fragment holds that state (`DrawSettings::failing_state`), the
    /// program's object, and takes every step once restated (`restated.rs`), as a carried draw
    /// is: it does not carry, so a carry names the program's objects by key and handle alone.
    fn carries_but_for_reads(&self) -> bool {
        self.rebinds && self.settings.failing_state.is_none()
    }

    /// Whether the draw, set aside with its pipeline state's bind state, is carried with `last`,
    /// its encoder's last kept draw (`render/carried.rs`): kept under its epoch, carrying of its
    /// own ([`carries`](Self::carries)), and having read what changes with no call as `last`
    /// read it, but what it was noted into, so the next encoder's check of `last`'s holds for it.
    pub(super) fn carries_with(&self, last: &Kept) -> bool {
        self.read_as(last) && self.carries()
    }

    /// Whether the draw, set aside, would be carried with `last` ([`carries_with`]) but for its
    /// reads of a texture an attachment of its pass renders.
    ///
    /// [`carries_with`]: Self::carries_with
    pub(super) fn carries_with_but_attached(&self, last: &Kept) -> bool {
        self.read_as(last) && self.carries_but_attached()
    }

    /// Whether the draw was kept under `last`'s epoch, having read what changes with no call as
    /// `last` read it, but what it was noted into.
    fn read_as(&self, last: &Kept) -> bool {
        let under = &last.under;
        self.epoch == last.epoch && self.carried_under(under.moved, under.residency, under.samplers)
    }

    /// The parts of the draw that are not `other`'s, by name: what the check of a draw again
    /// that kept another draw names ([`Checked`]).
    #[cfg(debug_assertions)]
    fn differences(&self, other: &Kept) -> Vec<&'static str> {
        let Kept {
            pipeline,
            epoch,
            stated,
            settings,
            restates_by_scissors,
            between,
            call,
            assembly,
            draw,
            index,
            under,
            command_buffer,
            command,
            again,
            pushes_set_zero,
            last_vertex,
            through_copies,
            narrowed,
            reads_images,
            execution,
            reads,
            rasterizes,
            depth_writes,
            draw_alone,
            stamps,
            rebinds,
            settled,
            carried,
            across,
            rehomed,
            made,
            reader,
        } = self;
        let [executed, shape, created] = match (execution, &other.execution) {
            (Some(execution), Some(other)) => execution.same_parts(other),
            (execution, other) => [execution.is_none() && other.is_none(); 3],
        };
        [
            ("pipeline", *pipeline == other.pipeline),
            ("epoch", *epoch == other.epoch),
            ("stated", *stated == other.stated),
            ("settings", *settings == other.settings),
            (
                "restates_by_scissors",
                *restates_by_scissors == other.restates_by_scissors,
            ),
            ("between", *between == other.between),
            ("call", *call == other.call),
            ("assembly", *assembly == other.assembly),
            ("draw", *draw == other.draw),
            ("index", *index == other.index),
            ("under", *under == other.under),
            ("command_buffer", *command_buffer == other.command_buffer),
            ("command", *command == other.command),
            ("again", *again == other.again),
            ("pushes_set_zero", *pushes_set_zero == other.pushes_set_zero),
            ("last_vertex", *last_vertex == other.last_vertex),
            ("through_copies", *through_copies == other.through_copies),
            ("narrowed", *narrowed == other.narrowed),
            ("reads_images", *reads_images == other.reads_images),
            ("execution's pipeline", executed),
            ("execution's shape", shape),
            ("execution's creation times", created),
            ("reads", *reads == other.reads),
            ("rasterizes", *rasterizes == other.rasterizes),
            ("depth_writes", *depth_writes == other.depth_writes),
            ("draw_alone", *draw_alone == other.draw_alone),
            ("stamps", *stamps == other.stamps),
            ("rebinds", *rebinds == other.rebinds),
            ("settled", *settled == other.settled),
            ("carried", *carried == other.carried),
            ("across", *across == other.across),
            ("rehomed", *rehomed == other.rehomed),
            ("made", *made == other.made),
            ("reader", *reader == other.reader),
        ]
        .into_iter()
        .filter_map(|(part, same)| (!same).then_some(part))
        .collect()
    }
}

/// An index source as a kept draw read it in place: the Metal buffer, by key, and offset, and
/// the `VkBuffer` and offset it read there (`DrawSource`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct KeptIndex {
    key: ResourceKey,
    offset: u64,
    raw: vk::Buffer,
    at: u64,
}

impl KeptIndex {
    /// The Metal buffer, by key, and the `VkBuffer` it is read from, its offsets aside.
    fn buffer(self) -> (ResourceKey, vk::Buffer) {
        (self.key, self.raw)
    }

    /// `source`'s, where it is read in place; `None` for a copy, which each draw takes.
    pub(super) fn of(source: &DrawSource<'_>) -> Option<KeptIndex> {
        source.in_place().then(|| KeptIndex {
            key: source.buffer.identity().key,
            offset: source.offset,
            raw: source.raw,
            at: source.at,
        })
    }
}

/// How `draw` differs from `kept`, a kept draw's call: in its counts and starts, in its index
/// buffer's offset, and in its index buffer or index type.
fn apart(kept: &DrawCall, draw: &DrawCall) -> (bool, bool, bool) {
    match (kept, draw) {
        (
            DrawCall::Indexed {
                index_type: kept_type,
                index_buffer: kept_buffer,
                index_count: kept_count,
                index_offset: kept_offset,
                instance_count: kept_instances,
                base_vertex: kept_base,
                base_instance: kept_first,
            },
            DrawCall::Indexed {
                index_type,
                index_buffer,
                index_count,
                index_offset,
                instance_count,
                base_vertex,
                base_instance,
            },
        ) => (
            (kept_count, kept_instances, kept_base, kept_first)
                != (index_count, instance_count, base_vertex, base_instance),
            kept_offset != index_offset,
            kept_type != index_type || kept_buffer != index_buffer,
        ),
        (kept, draw) => (kept != draw, false, false),
    }
}

/// An execution record's parts (`Execution::new`): the same where its pipeline and shape are
/// the same objects.
pub(super) struct KeptExecution {
    pub(super) pipeline: ExecutedPipeline,
    pub(super) shape: Arc<BindShape>,
    pub(super) created: Vec<Duration>,
}

impl KeptExecution {
    /// Whether its pipeline, its shape and its creation times are `other`'s, each.
    fn same_parts(&self, other: &KeptExecution) -> [bool; 3] {
        let pipeline = match (&self.pipeline, &other.pipeline) {
            (ExecutedPipeline::Render(a), ExecutedPipeline::Render(b)) => Arc::ptr_eq(a, b),
            (ExecutedPipeline::Compute(a), ExecutedPipeline::Compute(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        let shape = Arc::ptr_eq(&self.shape, &other.shape);
        [pipeline, shape, self.created == other.created]
    }
}

impl PartialEq for KeptExecution {
    fn eq(&self, other: &KeptExecution) -> bool {
        self.same_parts(other).into_iter().all(|same| same)
    }
}

/// What a kept draw read that changes with no call of the encoder (the module docs): the no-copy
/// buffers' moves, the residency's generation, what its uses were noted into, and the count of
/// the device's sampler IDs' slots' changes, at which its bind state listed the live sampler
/// states (`LiveBind::listed_at`: a state made or gone since, every step lists them again).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Under {
    moved: u64,
    residency: u64,
    into: NotedInto,
    samplers: u64,
}

impl Under {
    /// What its uses were noted into.
    pub(super) fn noted_into(&self) -> NotedInto {
        self.into
    }
}

/// What of an [`Under`] another thread may change while a draw takes its steps: the no-copy
/// buffers' moves, the count of the device's sampler IDs' slots' changes, and whether the device
/// has buffers over host pages and the host aliases' version, which a buffer over host pages made
/// or let go on any thread moves (the device's in-place check makes and lets go its own,
/// `in_place_check.rs`). A draw that takes its steps is kept under these as they were when its
/// steps began, which they read ([`RenderLowering::kept_under`]): a change made meanwhile makes
/// the next draw take every step.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Outside {
    moved: u64,
    rebinds: bool,
    aliases: u64,
    samplers: u64,
}

/// Whether a draw is its encoder's kept draw again ([`RenderLowering::draw_again`]).
pub(super) enum Again {
    /// It is, and was recorded.
    Drawn,
    /// It is, and takes every step instead, which the [`Checked`] it left checks (builds with
    /// debug assertions).
    #[cfg(debug_assertions)]
    Check,
    /// It is not.
    No,
    /// It was rebound, and refused as every step of it would refuse it.
    Refused(Error),
}

impl DrawAgain {
    /// Keep `kept`, the draw just recorded.
    pub(super) fn keep(&mut self, kept: Kept) {
        self.kept = Some(kept);
        #[cfg(debug_assertions)]
        {
            self.hits = 0;
        }
    }
}

/// Whether `used` reads an image other than as an attachment: what noting a draw's reads of
/// fresh textures looks at (`EncoderCore::observe_fresh`).
/// Whether `uses` write nothing and read nothing as indirect arguments, and, where they do not,
/// whether one of them reads an image other than as an attachment ([`reads_image`]): what a
/// draw kept for drawing again asks of its uses.
pub(super) fn unwritten(uses: &[Use<ResourceKey>]) -> (bool, bool) {
    let mut reads_images = false;
    let unwritten = uses.iter().all(|used| {
        reads_images |= reads_image(used);
        used.write.is_empty()
            && !(used.read.stages).intersects(vk::PipelineStageFlags2::DRAW_INDIRECT)
    });
    (unwritten, unwritten && reads_images)
}

pub(super) fn reads_image(used: &Use<ResourceKey>) -> bool {
    let attachment =
        vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
    matches!(used.range, UseRange::Image { .. }) && !(used.read.access & !attachment).is_empty()
}

impl RenderLowering {
    /// What a kept draw reads that changes with no call of the encoder (the module docs).
    pub(super) fn under(&self) -> Under {
        Under {
            moved: crate::buffer::host_cells_moved(),
            residency: self.residency.generation(),
            into: self.core.noted_into(),
            samplers: self.core.generation().sampler_slots_changed(),
        }
    }

    /// What of [`under`](Self::under) another thread may change ([`Outside`]), now.
    fn outside(&self) -> Outside {
        let Under {
            moved,
            into,
            samplers,
            ..
        } = self.under();
        Outside {
            moved,
            rebinds: into.rebinds,
            aliases: into.aliases,
            samplers,
        }
    }

    /// A draw begins its steps: note what another thread may change of what it reads
    /// ([`Outside`]), which a draw it keeps is kept under ([`kept_under`](Self::kept_under)).
    pub(super) fn draw_began(&mut self) {
        self.again.began = self.outside();
    }

    /// What a draw that took its steps is kept under: what changes with no call of the encoder
    /// as [`under`](Self::under) answers now, but what another thread may change as the draw's
    /// steps began ([`Outside`]), which they read; so one made meanwhile differs from the next
    /// draw's, which then takes every step.
    pub(super) fn kept_under(&self) -> Under {
        let Outside {
            moved,
            rebinds,
            aliases,
            samplers,
        } = self.again.began;
        let mut under = self.under();
        (under.moved, under.samplers) = (moved, samplers);
        (under.into.rebinds, under.into.aliases) = (rebinds, aliases);
        under
    }

    /// Forget every kept draw, the last direct draw's and those set aside with other pipeline
    /// states' bind states (a call came, a draw that kept nothing, or the encoder's end).
    pub(super) fn forget_again(&mut self) {
        self.again.kept = None;
        self.again.epoch = Stamp::fresh();
    }

    /// Forget every kept draw at a call of kind `by` the encoder logged (`RenderLowering::log`),
    /// the last direct draw's counted where there is one (`WorkCounts::kept_draws_let_go_*`).
    pub(super) fn let_go_again(&mut self, by: LoggedCall) {
        if self.again.kept.is_some() && !self.replaying && !self.rerouting {
            self.core.counts().kept_draw_let_go(by);
        }
        self.forget_again();
    }

    /// Keep every kept draw across a call of kind `by` the encoder logged that changes nothing
    /// they read or recorded (`RenderLowering::log_across`, the module docs), the last direct
    /// draw's counted where there is one (`WorkCounts::kept_draws_kept_across_*`).
    pub(super) fn keep_again_across(&self, by: LoggedCall) {
        if self.again.kept.is_some() && !self.replaying && !self.rerouting {
            self.core.counts().kept_draw_kept_across(by);
        }
    }

    /// Forget every kept draw set aside with other pipeline states' bind states (`live`) that
    /// reads what a draw that read through snapshots, keeping nothing, wrote
    /// ([`forget_aside_readers`](Self::forget_aside_readers)), and the last direct draw's.
    pub(super) fn forget_readers(&mut self, live: &mut LiveBind) -> Result<(), Error> {
        self.again.kept = None;
        self.forget_aside_readers(live)
    }

    /// Forget the kept draws set aside with other pipeline states' bind states (`live`) that read
    /// what a draw that read through snapshots wrote, kept or not: its pass's colors, read
    /// through their textures' copies, at levels apart from the attached ones, or by a use of
    /// their memory (`render_target::shares_attachment`), but where a kept draw reads them through
    /// snapshots of its own, which its next draw takes again behind its own barrier (the module
    /// docs). The others stay kept under the encoder's epoch.
    pub(super) fn forget_aside_readers(&mut self, live: &mut LiveBind) -> Result<(), Error> {
        let (memo, version) = (&mut self.attached, self.layout.version());
        let layout: &PassLayout = &self.layout;
        live.forget_kept_where(|kept, uses| kept.reads_colors((&mut *memo, layout, version), uses))
    }

    /// Keep the kept draws set aside with other pipeline states' bind states (`live`) into the
    /// render pass instance the draw just drawn began after a split, keeping nothing only for
    /// recording into it (the module docs): each that reads its pass's colors other than through
    /// snapshots of its own (whose next draw's steps read them past the split), a reader through
    /// snapshots, one carried in and not drawn since, and one that read what changes with no call
    /// otherwise than that draw ([`Kept::rehomed_into`]) let go, the others kept into it
    /// (`WorkCounts::kept_draws_rehomed`). The last direct draw's is that draw's, which kept
    /// nothing.
    pub(super) fn rehome_aside(&mut self, live: &mut LiveBind) -> Result<(), Error> {
        let (under, draws) = (self.kept_under(), self.current.draws);
        let (memo, version) = (&mut self.attached, self.layout.version());
        let layout: &PassLayout = &self.layout;
        let rehomed = live.rehome_kept(|kept, uses| -> Result<bool, Error> {
            if kept.reader.is_some()
                || kept.carried
                || kept.reads_colors((&mut *memo, layout, version), uses)?
            {
                return Ok(false);
            }
            Ok(kept.rehomed_into(under, draws))
        })?;
        self.again.kept = None;
        self.core.counts().kept_draws_rehomed(rehomed);
        Ok(())
    }

    /// Keep every kept draw across a memory barrier over the render targets alone that records
    /// nothing but its barrier inside the current instance, or nothing (`fence.rs`), but those
    /// whose next draw's steps it changes: each that reads its pass's colors other than through
    /// snapshots of its own ([`Kept::reads_colors`]), which every step after the barrier reads
    /// past a split, or through a copy taken again (`render.rs`'s attached reads). A reader
    /// through snapshots orders its reads behind its own `B1`, as every step's does after the
    /// barrier, and the barrier changes no state its command buffer holds: the next draw of each
    /// kept draw is drawn again or rebound (the module docs). The last direct draw's kept draw is
    /// counted kept across or let go (`WorkCounts::kept_draws_kept_across_barrier`,
    /// `kept_draws_let_go_barrier`). Where whether one reads them cannot be told (an attachment
    /// whose use the device refuses), it is let go: its next draw takes every step, which answers.
    pub(crate) fn keep_across_barrier(&mut self, live: &mut LiveBind) {
        let (memo, version) = (&mut self.attached, self.layout.version());
        let layout: &PassLayout = &self.layout;
        let mut reads_colors = |kept: &Kept, uses: &[Use<ResourceKey>]| {
            (kept.reads_colors((&mut *memo, layout, version), uses)).unwrap_or(true)
        };
        let forgets = |kept: &Kept, uses: &crate::binding::CommandUses| {
            Ok::<_, std::convert::Infallible>(reads_colors(kept, uses))
        };
        let Ok(()) = live.forget_kept_where(forgets);
        let last = (self.again.kept.as_ref()).map(|kept| reads_colors(kept, &live.uses));
        match last {
            Some(true) => {
                if !self.replaying && !self.rerouting {
                    self.core.counts().kept_draw_let_go(LoggedCall::Barrier);
                }
                self.again.kept = None;
            }
            Some(false) => self.keep_again_across(LoggedCall::Barrier),
            None => {}
        }
    }

    /// The epoch a draw kept now is kept under ([`DrawAgain`]).
    pub(super) fn again_epoch(&self) -> Stamp {
        self.again.epoch
    }

    /// A call set the encoder's render state, its scissor rectangles alone where `scissors`:
    /// every kept draw is restated (the module docs).
    pub(super) fn restate_again(&mut self, scissors: bool) {
        self.again.stated.set = Stamp::fresh();
        if !scissors {
            self.again.stated.beyond_scissors = Stamp::fresh();
        }
    }

    /// The encoder's render state's stamps, which a draw kept now holds ([`DrawAgain`]).
    pub(super) fn again_stated(&self) -> Stated {
        self.again.stated
    }

    /// The last direct draw's kept draw, where it is kept.
    pub(super) fn again_kept(&self) -> Option<&Kept> {
        self.again.kept.as_ref()
    }

    /// Let go of the last direct draw's kept draw alone, which the draw about to take every step
    /// keeps itself in place of where it can: those set aside stay kept.
    pub(super) fn drop_kept(&mut self) {
        self.again.kept = None;
    }

    /// The last direct draw's kept draw, taken: an ended encoder's, carried into its command
    /// buffer's next render encoder (`render/carried.rs`).
    pub(super) fn take_kept(&mut self) -> Option<Kept> {
        self.again.kept.take()
    }

    /// Keep `kept`, a draw carried in from the command buffer's last render encoder
    /// (`render/carried.rs`), as the last direct draw's, in the instance the encoder opened with.
    pub(super) fn keep_carried(&mut self, kept: Kept) {
        self.again.kept = Some(kept);
        self.again.opened = Some(self.current.shape.version());
    }

    /// The kept draws carried in since the command buffer's first render encoder
    /// ([`DrawAgain`]), which a carried draw carries on (`render/carried.rs`).
    #[cfg(debug_assertions)]
    pub(super) fn carries(&self) -> u64 {
        self.again.carries
    }

    /// Take `carries` kept draws carried in since the command buffer's first render encoder: a
    /// carried draw's.
    #[cfg(debug_assertions)]
    pub(super) fn set_carries(&mut self, carries: u64) {
        self.again.carries = carries;
    }

    /// Bring in the kept draw of the pipeline state the encoder draws through, where its last draw
    /// drew through another: the last draw's kept draw set aside with the bind state `live` holds
    /// for that state, and the bind state of this one selected (`LiveBind::select`), with the
    /// draw kept beside it where one is, kept under the encoder's epoch still. Such a draw drew
    /// before draws of other states recorded: it is [`Kept::between`]. One carried in from the
    /// command buffer's last render encoder (`render/carried.rs`) is kept while the current
    /// instance is the one the encoder opened with, the pass's first, which the draws since
    /// recorded into or began (each a draw kept, drawn again or rebound, under the epoch): it
    /// records into that instance's command buffer as it is now.
    pub(super) fn select_kept(&mut self, live: &mut LiveBind) {
        let Some(RenderStateKind::Graphics(pipeline)) = self.pipeline.as_ref().map(|p| p.kind())
        else {
            return;
        };
        if live.draws(pipeline) {
            return;
        }
        live.set_aside_kept(self.again.kept.take());
        live.select(pipeline);
        let epoch = self.again.epoch;
        let opened = self.again.opened == Some(self.current.shape.version());
        self.again.kept = (live.take_kept()).filter(|kept| {
            kept.epoch == epoch && kept.pipeline.is(pipeline) && (opened || !kept.carried)
        });
        if let Some(kept) = self.again.kept.as_mut() {
            kept.between = true;
            if kept.carried {
                kept.command_buffer = self.current.draws;
            }
        }
    }

    /// A draw of the kept draw again after calls that takes every step: counted
    /// (`WorkCounts::draws_rebind_stopped`).
    pub(super) fn rebind_stopped(&self) -> Again {
        self.core.counts().draw_rebind_stopped();
        Again::No
    }

    /// What the kept draw `reader`, which reads through snapshots, weighs its form by now
    /// (`RenderLowering::weigh_snapshot`), where it takes the snapshot again on a device of the
    /// write mask command; else `None`, and the draw takes every step.
    fn weigh_again(
        &self,
        reader: &super::snapshot::KeptReader,
    ) -> Option<super::snapshot::Weighed> {
        let Some(RenderStateKind::Graphics(pipeline)) = self.pipeline.as_ref().map(|p| p.kind())
        else {
            return None;
        };
        self.core.device().shared().color_write_masks?;
        // A kept reader never leads its instance (`KeptReader::reads_as`).
        let weighed = self.weigh_snapshot(pipeline, reader.feedback, false);
        weighed.within.then_some(weighed)
    }

    /// The stamps of the encoder's vertex, fragment, object and mesh tables.
    pub(super) fn table_stamps(&self) -> [Stamp; 4] {
        [&self.vertex, &self.fragment, &self.object, &self.mesh].map(|table| table.stamp())
    }

    /// Whether the draw of `call` assembling `assembly`, `draw` with `buffers`, a draw of nothing
    /// where `empty`, is the kept draw again (the module docs), or the kept draw of other
    /// arguments, which is rebound (`rebound.rs`): then, but where it is checked (its check left
    /// in `checked`), it is recorded, with the encoder's bind state `live`. Call it once the draw
    /// made the heaps' new textures resident.
    pub(super) fn draw_again(
        &mut self,
        live: &mut LiveBind,
        (call, assembly): (&'static str, Assembly),
        (draw, empty): (&DrawCall, bool),
        buffers: &DrawBuffers<'_>,
        checked: &mut Option<Checked>,
    ) -> Again {
        let Some(kept) = self.again.kept.as_ref() else {
            if !self.replaying && !self.rerouting {
                self.core.counts().draw_none_kept();
            }
            return Again::No;
        };
        let index = buffers.index.as_ref().map(KeptIndex::of);
        let drawn_by = |state: &RenderPipelineState| kept.pipeline.names(state);
        let same = buffers.indirect.is_none()
            && buffers.patch_indices.is_none()
            && buffers.control_point_indices.is_none()
            && self.pipeline.as_ref().is_some_and(drawn_by)
            && kept.call == call
            && kept.assembly == assembly
            && kept.draw == *draw
            && kept.index == index.flatten()
            && index.is_none_or(|index| index.is_some())
            && kept.command_buffer == self.current.draws
            && kept.under == self.under();
        // Of other arguments alone, of something to draw: rebound with them.
        let mut reargued = false;
        if !same {
            if buffers.indirect.is_none() {
                let (counts, index_offset, index_buffer) = apart(&kept.draw, draw);
                let copied = matches!(index, Some(None));
                let index = index.flatten();
                let index_buffer = index_buffer
                    || copied
                    || kept.index.map(KeptIndex::buffer) != index.map(KeptIndex::buffer);
                let why = UnlikeKept {
                    pipeline: !self.pipeline.as_ref().is_some_and(drawn_by),
                    call: kept.call != call,
                    assembly: kept.assembly != assembly,
                    counts,
                    index_buffer,
                    index_copied: copied,
                    index_offset: !index_buffer && (index_offset || kept.index != index),
                    tessellation: buffers.patch_indices.is_some()
                        || buffers.control_point_indices.is_some(),
                    command_buffer: kept.command_buffer != self.current.draws,
                    under: kept.under != self.under(),
                };
                reargued = why.arguments_alone() && !empty;
                if !reargued {
                    self.core.counts().draw_unlike_kept(why);
                }
            }
            if !reargued {
                return Again::No;
            }
        }
        let stamps = self.table_stamps();
        // In an instance recorded inline after a split, a draw again records its recording into
        // the instance's room, as every step's (`render/inline_splits.rs`): rebound, through no
        // Vulkan draw of its own recorded ahead of the room's.
        let room = self.current.kept.is_some();
        if reargued
            || kept.stamps != stamps
            || kept.between
            || kept.stated != self.again.stated
            || room
        {
            return match kept.rebinds {
                true => {
                    let drawn = (assembly, draw, reargued);
                    self.draw_rebound(live, call, drawn, buffers, stamps, checked)
                }
                false => {
                    if reargued {
                        let why = ArgumentsStop::Unkept;
                        self.core.counts().draw_rebind_stopped_arguments(why);
                    }
                    self.rebind_stopped()
                }
            };
        }
        // A reader through snapshots weighs its form again, as every step does: its identity
        // among the pass's readers and what its texture learned it covered change with no call
        // (`render/coverage.rs`). Where it would take another form, it takes every step; so does
        // any other draw that would read its pass's colors through snapshots now, in place or
        // after the split, as a rebound one does (`rebound.rs`).
        let weighed = match &kept.reader {
            Some(reader) => match self.weigh_again(reader) {
                Some(weighed) => Some(weighed),
                None => {
                    self.core.counts().draw_reader_stopped();
                    return Again::No;
                }
            },
            None if self.reads_otherwise(&kept.settings) => return Again::No,
            None => None,
        };
        #[cfg(debug_assertions)]
        {
            let hits = self.again.hits;
            self.again.hits += 1;
            if hits.is_multiple_of(2) {
                let held = Held::of(self, live);
                *checked = Some(Checked::of(self, Expected::Plain(held)));
                return Again::Check;
            }
        }
        // Its identity, count and query taken, as every step takes them.
        let query = match weighed.map(|weighed| self.take_weighed(weighed)) {
            Some(Ok(query)) => query,
            Some(Err(refused)) => return Again::Refused(refused),
            None => None,
        };
        let Some(kept) = self.again.kept.as_ref() else {
            return Again::No;
        };
        let counts = self.core.counts();
        counts.draw_drawn_again((kept.pushes_set_zero, kept.last_vertex));
        if kept.reader.is_some() {
            counts.reading_draw_again();
            let shared = &self.core.device().shared().counts;
            shared.reading_draw_snapshotted();
            if query.is_some() {
                shared.coverage_query();
            }
        }
        if kept.through_copies.0 {
            counts.attachment_read_through_copy();
        }
        if kept.through_copies.1 {
            counts.own_pixel_read();
        }
        // Its noting: its reads of fresh textures, where any texture is fresh, and what the
        // encoder made resident since (nothing: a use that made anything resident gave the
        // residency a new generation, which `Under` compares).
        match kept.reads_images && self.core.observes_fresh() {
            true => self.core.note_again(live.noted.uses()),
            false => self.core.note_again(&[]),
        }
        // SAFETY: the kept draw recorded this command into this command buffer, which records
        // the same render pass instance, with every handle it binds live until the command
        // buffer completes (retained as the kept draw recorded); the command buffer's state is
        // the kept draw's, as no call came and no draw recorded since (`render/bound.rs`).
        unsafe {
            let shared = self.core.device().shared();
            let raw = shared.vk();
            match (&kept.reader, shared.color_write_masks) {
                // Its snapshot's two draws and their barriers, the drawing one inside its query
                // (`render/snapshot.rs`): the device has the write mask command, as the kept
                // draw's reads through snapshots took it (`weigh_again`).
                (Some(reader), Some(masks)) => {
                    reader.record_again(raw, masks, kept.command_buffer, kept.command, query);
                }
                // A kept draw is a whole draw's (`DrawStates::Whole`), of no regions.
                _ => super::record_draw_command(
                    raw,
                    kept.command_buffer,
                    kept.command,
                    None,
                    (None, None),
                ),
            }
        }
        if let Some(execution) = &kept.execution {
            let pipeline = execution.pipeline.clone();
            let shape = Arc::clone(&execution.shape);
            let created = execution.created.clone();
            self.core
                .executed(|| Execution::new(pipeline, shape, created));
        }
        let (again, reads) = (kept.again, kept.reads);
        let (rasterizes, depth_writes) = (kept.rasterizes, kept.depth_writes);
        self.keep_draw(again, (call, assembly, *draw), buffers.index.as_ref());
        self.draw_calls += 1;
        self.last_draw_empty = false;
        self.command_encoded(reads);
        self.called = true;
        self.drew |= rasterizes;
        self.drew_depth_writes |= depth_writes;
        Again::Drawn
    }

    /// Finish keeping the draw just taken step by step: forget it where `watch` met a notice the
    /// device announces or refuses at every occurrence, else note what `RenderLowering::draw`
    /// takes of it (`reads`, and whether it rasterized and wrote depth); and where it was
    /// `checked`, a kept draw again that took every step, check it (the module docs).
    pub(super) fn kept_again(
        &mut self,
        live: &LiveBind,
        watch: &EachWatch,
        (reads, rasterizes, depth_writes): (RenderStages, bool, bool),
        checked: Option<Checked>,
    ) {
        if let Some(kept) = self.again.kept.as_mut() {
            kept.reads = reads;
            kept.rasterizes = rasterizes;
            kept.depth_writes = depth_writes;
        }
        #[cfg(debug_assertions)]
        if let Some(checked) = checked {
            checked.check(self, live, watch.met());
        }
        #[cfg(not(debug_assertions))]
        let _: (&LiveBind, Option<Checked>) = (live, checked);
        if watch.met() {
            self.forget_again();
        }
    }
}

/// What a draw again that takes every step checks of it (the module docs): the kept draw, the
/// messages delivered on its thread by then, and what its every step must leave or work out.
#[cfg(debug_assertions)]
pub(super) struct Checked {
    kept: Kept,
    hits: u64,
    announced: u64,
    expected: Expected,
}

/// What every step of a draw again must leave, or of a draw rebound work out.
#[cfg(debug_assertions)]
pub(super) enum Expected {
    /// What the encoder held before the draw again, which it must hold after it.
    Plain(Held),
    /// What the draw rebound worked out of its rebound slots (`rebound.rs`).
    Rebound(Rebound),
}

/// The digest of `part`'s `Debug` form, streamed into a hasher: what a check compares of a part
/// of the encoder's state, which it neither copies nor allocates for (builds with debug
/// assertions; the frame's allocations are counted in them too).
#[cfg(debug_assertions)]
fn digest(part: impl std::fmt::Debug) -> u64 {
    use std::fmt::Write as _;
    use std::hash::Hasher as _;
    /// A `fmt::Write` into a hasher.
    struct Hashing(std::hash::DefaultHasher);
    impl std::fmt::Write for Hashing {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0.write(text.as_bytes());
            Ok(())
        }
    }
    let mut hashing = Hashing(std::hash::DefaultHasher::new());
    write!(hashing, "{part:?}").unwrap_or_else(|_| unreachable!("a part's Debug form failed"));
    hashing.0.finish()
}

/// What a draw rebound worked out (`rebound.rs`), which its every step must work out alike: its
/// commands, set-0 writes, dynamic offsets, push bytes, vertex buffers, uses and resident keys,
/// and the Vulkan draw and index buffer bind its recording records (a draw of other arguments'
/// worked out again), by digest ([`digest`]), and the draw state its recording records with (a
/// draw restated's derived again, `restated.rs`).
#[cfg(debug_assertions)]
pub(super) struct Rebound([u64; 8], Option<super::DrawState>);

#[cfg(debug_assertions)]
impl Rebound {
    /// The parts' names, in their order.
    const PARTS: [&str; 8] = [
        "commands",
        "set-0 writes",
        "dynamic offsets",
        "push bytes",
        "vertex buffers",
        "uses",
        "resident keys",
        "Vulkan draw and index buffer bind",
    ];

    /// What `live` holds of a draw rebound.
    pub(super) fn of(live: &crate::live_bind::LiveBind) -> Rebound {
        /// Set writes, each without its keeper (`binding::SetWrite::same`'s parts), and push
        /// writes, their types and stages by their raw values (ash's print nothing).
        struct Writes<'a>(&'a [crate::binding::Write]);
        struct Pushes<'a>(&'a [crate::binding::PushWrite]);
        impl std::fmt::Debug for Writes<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let each = (self.0.iter()).map(|write| {
                    let ty = write.ty.as_raw();
                    (write.set, write.binding, write.element, ty, write.info)
                });
                f.debug_list().entries(each).finish()
            }
        }
        impl std::fmt::Debug for Pushes<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let each = (self.0.iter()).map(|(stages, at, bytes)| (stages.as_raw(), at, bytes));
                f.debug_list().entries(each).finish()
            }
        }
        Rebound(
            [
                digest(&live.commands),
                digest(Writes(live.kept.writes())),
                digest(&live.offsets),
                digest(Pushes(&live.pushes)),
                digest(&live.vertex_buffers),
                digest(&*live.uses),
                digest(live.uses.resident()),
                digest((live.kept.recording()).map(|recording| {
                    // By their raw values: ash's handles and enums print nothing.
                    let index = (recording.index).map(|index| {
                        let size = index.sized.map(|(_, size)| size);
                        let buffer = vk::Handle::as_raw(index.buffer);
                        (buffer, index.offset, index.index_type.as_raw(), size)
                    });
                    (recording.draw, index)
                })),
            ],
            (live.kept.recording()).map(|recording| recording.state.clone()),
        )
    }

    /// Assert that every step of the draw worked out what it did, in `live`.
    fn assert_as(&self, live: &crate::live_bind::LiveBind) {
        let now = Rebound::of(live);
        for (part, (now, then)) in Self::PARTS.iter().zip(now.0.iter().zip(&self.0)) {
            assert!(
                now == then,
                "a draw rebound worked out other {part} than its every step"
            );
        }
        assert!(
            now.1 == self.1,
            "a draw rebound records with another draw state than its every step"
        );
    }
}

/// No check in builds without debug assertions: no draw again takes every step.
#[cfg(not(debug_assertions))]
pub(super) enum Checked {}

/// What a draw again leaves as it was (the module docs): its drawn areas, coverage, pass's reads,
/// gathered and noted uses and the pass's inputs read and extended colors written by digest
/// ([`digest`]), and its instance's shape and holds.
#[cfg(debug_assertions)]
pub(super) struct Held {
    parts: [u64; 7],
    instance: (bool, bool, bool, usize, usize, super::InterlockedStorage),
    holds: (usize, usize),
}

#[cfg(debug_assertions)]
impl Held {
    /// The digested parts' names, in their order.
    const PARTS: [&str; 7] = [
        "drawn area",
        "instance's drawn area",
        "coverage",
        "pass's reads",
        "gathered uses",
        "noted uses",
        "pass",
    ];

    fn of(encoder: &RenderLowering, live: &LiveBind) -> Held {
        let current = &encoder.current;
        let pass = (
            &encoder.inputs_read,
            &encoder.extended_written,
            encoder.reaches_unnamed,
        );
        Held {
            parts: [
                digest(&encoder.drawn),
                digest(&current.drawn),
                digest(&encoder.coverage),
                digest(encoder.pass_start.held()),
                digest(current.shadows.gathered()),
                digest(&live.noted),
                digest(pass),
            ],
            instance: (
                current.tile_memory,
                current.barriers,
                current.imageblock,
                current.pieces.len(),
                current.before.len(),
                current.interlocked_writes,
            ),
            holds: encoder.core.retained_holds(),
        }
    }

    /// Assert that `self`, held after a draw again took every step, is `before`, but for its
    /// holds: one more index buffer's where it reads one (`index`), and one more execution
    /// record where the device keeps them (`executed`).
    fn assert_as(&self, before: &Held, (index, executed): (bool, bool)) {
        for (part, (now, then)) in Self::PARTS.iter().zip(self.parts.iter().zip(&before.parts)) {
            assert!(now == then, "a draw again changed its encoder's {part}");
        }
        assert!(
            self.instance == before.instance,
            "a draw again changed its encoder's instance"
        );
        let holds = (
            before.holds.0 + usize::from(index),
            before.holds.1 + usize::from(executed),
        );
        assert_eq!(
            self.holds, holds,
            "a draw again held what its kept draw did not"
        );
    }
}

#[cfg(debug_assertions)]
impl Checked {
    /// The check of the draw again about to take every step, which must leave or work out
    /// `expected`.
    pub(super) fn of(encoder: &mut RenderLowering, expected: Expected) -> Checked {
        let kept = encoder.again.kept.take();
        let kept = kept.unwrap_or_else(|| unreachable!("a draw again without its kept draw"));
        Checked {
            kept,
            hits: encoder.again.hits,
            announced: crate::degradation::announced_here(),
            expected,
        }
    }

    /// Check the draw that took every step, which `met` a notice met at each occurrence or not,
    /// against the kept one (the module docs).
    fn check(self, encoder: &mut RenderLowering, live: &LiveBind, met: bool) {
        let Checked {
            kept,
            hits,
            announced,
            expected,
        } = self;
        // What another thread may change moved while the draw took every step ([`Outside`]): its
        // steps read the state after the change, which the kept draw never read, so it may note
        // its uses anew or keep nothing (kept, it is kept under the state its steps began in),
        // as a draw again after the change takes every step. Nothing is compared.
        if encoder.outside() != encoder.again.began {
            encoder.again.hits = hits;
            return;
        }
        let rebound = matches!(expected, Expected::Rebound(_));
        match &expected {
            // Its uses differ from the kept draw's: what noting them adds is every step's own.
            Expected::Rebound(rebound) => rebound.assert_as(live),
            Expected::Plain(held) => {
                assert_eq!(
                    crate::degradation::announced_here(),
                    announced,
                    "a draw again announced a notice"
                );
                assert!(!met, "a draw again met a notice met at each occurrence");
                let executed = kept.execution.is_some();
                Held::of(encoder, live).assert_as(held, (kept.index.is_some(), executed));
            }
        }
        let again = encoder.again.kept.as_mut();
        let again = again.unwrap_or_else(|| panic!("a draw again kept no draw"));
        let (draw_alone, stamps, under, settled, between, stated) = (
            again.draw_alone,
            again.stamps,
            again.under,
            again.settled,
            again.between,
            again.stated,
        );
        // A carried draw's every step began the pass's first instance, which its kept draw named
        // before any did, and keeps a draw carried no more, nor across (`render/carried.rs`).
        let (command_buffer, carried, across) = (again.command_buffer, again.carried, again.across);
        // A draw re-homed is kept into its instance as its every step keeps itself there.
        let rehomed = again.rehomed;
        assert!(
            draw_alone || rebound,
            "a draw again recorded a state command"
        );
        assert!(
            again.rebinds || !rebound,
            "a draw rebound whose every step keeps nothing to rebind"
        );
        // A draw rebound reads its tables under their stamps now, its uses noted into the state
        // now; and every step recorded through its instance's record since (`Kept::settled`),
        // the encoder's last draw (`Kept::between`), under the encoder's render state now
        // (`Kept::stated`).
        (again.draw_alone, again.stamps, again.under) = (kept.draw_alone, kept.stamps, kept.under);
        (again.settled, again.between, again.stated) = (kept.settled, kept.between, kept.stated);
        if kept.carried {
            (again.command_buffer, again.carried) = (kept.command_buffer, kept.carried);
            again.across = kept.across;
        }
        again.rehomed = kept.rehomed;
        // The shapes' values where only their objects may differ.
        let shapes = match (&again.execution, &kept.execution) {
            (Some(again), Some(kept)) => Some(*again.shape == *kept.shape),
            _ => None,
        };
        assert!(
            *again == kept,
            "a draw again kept another draw: its {:?} differ (rebound {rebound}, carried {}, \
             execution shapes equal in value {shapes:?})",
            again.differences(&kept),
            kept.carried
        );
        (again.draw_alone, again.stamps, again.under) = (draw_alone, stamps, under);
        (again.settled, again.between, again.stated) = (settled, between, stated);
        (again.command_buffer, again.carried) = (command_buffer, carried);
        again.across = across;
        again.rehomed = rehomed;
        encoder.again.hits = hits;
    }
}
