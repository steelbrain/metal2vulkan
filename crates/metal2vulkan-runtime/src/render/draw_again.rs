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
//!   it stood, through a pipeline made ahead, needing nothing before or around it of its own: no
//!   split, stand-in, snapshot, copy, index bounds or rewrite pass, mesh or tessellation work, rate
//!   map, color groups, extended-range color or its shadow, set 1 or handle table, root table or
//!   push hole, barrier in its instance, read of a color its pass clears behind its load actions
//!   (a copy cleared to the clear, `texture/attached_reads.rs`), and no write among its uses, which
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
//! - **Forgotten at every call but a bind, a pipeline state set or a render state set.** Every
//!   call of the encoder but the two direct draws, the vertex and fragment buffer, bytes, texture
//!   and sampler state binds, `set_render_pipeline_state` and the sets of its render state (below)
//!   forgets it as it logs itself (`RenderLowering::log`,
//!   which every call does: at the call, on the device's lowering thread, and encoded again),
//!   refused or not, whatever it changes, with every kept draw set aside (below); so does a draw
//!   that keeps nothing (a draw again, a draw rebound and a draw kept are the only draws that
//!   leave them). A bind changes its
//!   stage table's slots alone (and a bytes bind takes room in its command buffer's uploads), and
//!   a table's stamp tells each change apart (`argument_table::Stamp`): the kept draw keeps the
//!   four tables' stamps as it read them, and a draw whose tables kept them is the kept draw
//!   again; one after binds is drawn again rebound where it can be (`rebound.rs`), else takes
//!   every step. So the encoder's state a draw resolves from is the kept draw's, but for the
//!   tables' slots the stamps tell apart, what the kept draw itself left and what changes with no
//!   call of the encoder: the latter is [`Under`], compared at each draw: the count of no-copy
//!   buffers' moves (a bound buffer's binding, a detach), the residency's generation once the
//!   draw made the heaps' new textures resident, whether the device has buffers over host pages
//!   and its host aliases' version, the encoder's parts and held images (what noting reads,
//!   `noted_apart.rs`), with the command buffer the current instance records into, and the count
//!   of the device's sampler IDs' slots' changes (a sampler state made or gone, which every step
//!   of a draw lists again: `sampler::ArgumentSamplers`).
//! - **One per pipeline state.** A pipeline state set keeps the kept draw: at the next draw
//!   through another state, it goes aside with its state's bind state and the storage its draws
//!   wrote (`LiveBind::select`, `RenderLowering::select_kept`), and the draw starts from its own
//!   state's, with the draw kept beside it, where one is under the encoder's epoch (which every
//!   forgetting draws afresh: `DrawAgain`). Such a draw is between (`Kept::between`): draws of
//!   other states recorded since, each a draw kept or drawn again, into the same instance, whose
//!   steps only add to what the encoder holds (below). Its next draw is rebound whatever its
//!   tables' stamps (`rebound.rs`), the changes of its tables since it read them brought in, its
//!   uses noted, added to the pass's reads and gathered whole, and every state command its
//!   record does not hold recorded; it takes no shortcut of a draw moved, which reads what the
//!   encoder's last draw left. Each state's bind state is its own, though states of one function
//!   pair share an interface, so the draw kept of one is never the other's.
//! - **Restated by a render state set.** A call that sets the encoder's render state alone (its
//!   viewports, scissor rectangles, cull mode, winding, depth/stencil state, stencil references,
//!   depth bias, depth clip mode, fill mode or blend color: `RenderLowering::log_restating`)
//!   keeps every kept draw, restated: the encoder's state stamp is drawn afresh (`DrawAgain`), and
//!   a kept draw whose stamp is not the encoder's derives again, at its next draw, everything the
//!   state decides of it (`restated.rs`). It is then rebound (`rebound.rs`), recording its new
//!   draw state's dynamic states against what its record holds, where what it derived leaves
//!   its pipeline, plan, shape and push bytes the kept draw's; else it takes every step.
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
//!   encoder's drawn area, coverage, reads, gathered uses, noted uses and instance as they were.

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
use crate::pipeline::{RenderPipelineState, RenderStateKind, WeakRenderPipelineState};
use crate::resource::ResourceKey;
use crate::tracker::{Use, UseRange};
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
    /// The encoder's render state's stamp (a fresh stamp, its default): drawn afresh at each call
    /// that sets it ([`restate_again`](RenderLowering::restate_again)), so a kept draw that holds
    /// another is restated (the module docs).
    stated: Stamp,
    /// Draws again since the draw was kept: in builds with debug assertions, the first and every
    /// second one after it take every step instead (the module docs).
    #[cfg(debug_assertions)]
    hits: u64,
}

/// What a kept draw was, what it read, and what a draw again repeats of it (the module docs).
#[derive(PartialEq)]
pub(crate) struct Kept {
    /// The pipeline state it drew, and the encoder's epoch it was kept under ([`DrawAgain`]).
    pub(super) pipeline: WeakRenderPipelineState,
    pub(super) epoch: Stamp,
    /// The encoder's render state's stamp as it was kept or last rebound ([`DrawAgain`]), and the
    /// settings it drew under then: a draw of it under another stamp derives them again
    /// (`restated.rs`).
    pub(super) stated: Stamp,
    pub(super) settings: Arc<DrawSettings>,
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

/// An execution record's parts (`Execution::new`): the same where its pipeline and shape are
/// the same objects.
pub(super) struct KeptExecution {
    pub(super) pipeline: ExecutedPipeline,
    pub(super) shape: Arc<BindShape>,
    pub(super) created: Vec<Duration>,
}

impl PartialEq for KeptExecution {
    fn eq(&self, other: &KeptExecution) -> bool {
        let pipeline = match (&self.pipeline, &other.pipeline) {
            (ExecutedPipeline::Render(a), ExecutedPipeline::Render(b)) => Arc::ptr_eq(a, b),
            (ExecutedPipeline::Compute(a), ExecutedPipeline::Compute(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        pipeline && Arc::ptr_eq(&self.shape, &other.shape) && self.created == other.created
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

    /// Forget every kept draw, the last direct draw's and those set aside with other pipeline
    /// states' bind states (a call came, a draw that kept nothing, or the encoder's end).
    pub(super) fn forget_again(&mut self) {
        self.again.kept = None;
        self.again.epoch = Stamp::fresh();
    }

    /// The epoch a draw kept now is kept under ([`DrawAgain`]).
    pub(super) fn again_epoch(&self) -> Stamp {
        self.again.epoch
    }

    /// A call set the encoder's render state: every kept draw is restated (the module docs).
    pub(super) fn restate_again(&mut self) {
        self.again.stated = Stamp::fresh();
    }

    /// The encoder's render state's stamp, which a draw kept now holds ([`DrawAgain`]).
    pub(super) fn again_stated(&self) -> Stamp {
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

    /// Bring in the kept draw of the pipeline state the encoder draws through, where its last draw
    /// drew through another: the last draw's kept draw set aside with the bind state `live` holds
    /// for that state, and the bind state of this one selected (`LiveBind::select`), with the
    /// draw kept beside it where one is, kept under the encoder's epoch still. Such a draw drew
    /// before draws of other states recorded: it is [`Kept::between`].
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
        self.again.kept =
            (live.take_kept()).filter(|kept| kept.epoch == epoch && kept.pipeline.is(pipeline));
        if let Some(kept) = self.again.kept.as_mut() {
            kept.between = true;
        }
    }

    /// A draw of the kept draw again after calls that takes every step: counted
    /// (`WorkCounts::draws_rebind_stopped`).
    pub(super) fn rebind_stopped(&self) -> Again {
        self.core.counts().draw_rebind_stopped();
        Again::No
    }

    /// The stamps of the encoder's vertex, fragment, object and mesh tables.
    pub(super) fn table_stamps(&self) -> [Stamp; 4] {
        [&self.vertex, &self.fragment, &self.object, &self.mesh].map(|table| table.stamp())
    }

    /// Whether the draw of `call` assembling `assembly`, `draw` with `buffers`, is the kept draw
    /// again (the module docs): then, but where it is checked (its check left in `checked`), it
    /// is recorded, with the encoder's bind state `live`. Call it once the draw made the heaps'
    /// new textures resident.
    pub(super) fn draw_again(
        &mut self,
        live: &mut LiveBind,
        call: &'static str,
        assembly: Assembly,
        draw: &DrawCall,
        buffers: &DrawBuffers<'_>,
        checked: &mut Option<Checked>,
    ) -> Again {
        let Some(kept) = self.again.kept.as_ref() else {
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
        if !same {
            if buffers.indirect.is_none() {
                self.core.counts().draw_unlike_kept();
            }
            return Again::No;
        }
        let stamps = self.table_stamps();
        if kept.stamps != stamps || kept.between || kept.stated != self.again.stated {
            return match kept.rebinds {
                true => self.draw_rebound(live, call, (assembly, draw), buffers, stamps, checked),
                false => self.rebind_stopped(),
            };
        }
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
        let counts = self.core.counts();
        counts.draw_drawn_again((kept.pushes_set_zero, kept.last_vertex));
        if kept.through_copies.0 {
            counts.attachment_read_through_copy();
        }
        if kept.through_copies.1 {
            counts.own_pixel_read();
        }
        // Its noting: its reads of fresh textures, where any texture is fresh, and what the
        // encoder made resident since (nothing: no call came).
        match kept.reads_images && self.core.observes_fresh() {
            true => self.core.note_again(live.noted.uses()),
            false => self.core.note_again(&[]),
        }
        // SAFETY: the kept draw recorded this command into this command buffer, which records
        // the same render pass instance, with every handle it binds live until the command
        // buffer completes (retained as the kept draw recorded); the command buffer's state is
        // the kept draw's, as no call came and no draw recorded since (`render/bound.rs`).
        unsafe {
            let raw = self.core.device().shared().vk();
            super::record_draw_command(raw, kept.command_buffer, kept.command, None, None);
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
/// by digest ([`digest`]), and the draw state its recording records with (a draw restated's
/// derived again, `restated.rs`).
#[cfg(debug_assertions)]
pub(super) struct Rebound([u64; 7], Option<super::DrawState>);

#[cfg(debug_assertions)]
impl Rebound {
    /// The parts' names, in their order.
    const PARTS: [&str; 7] = [
        "commands",
        "set-0 writes",
        "dynamic offsets",
        "push bytes",
        "vertex buffers",
        "uses",
        "resident keys",
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
        assert!(*again == kept, "a draw again kept another draw");
        (again.draw_alone, again.stamps, again.under) = (draw_alone, stamps, under);
        (again.settled, again.between, again.stated) = (settled, between, stated);
        encoder.again.hits = hits;
    }
}
