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
//!   it stood, through a pipeline made ahead, needing nothing before or around it: no split,
//!   stand-in, shadow, snapshot, copy, index bounds or rewrite pass, mesh or tessellation work, rate
//!   map, color groups, extended-range color, set 1 or handle table, root table or push hole,
//!   barrier in its instance, and no write among its uses, which each change what the next draw
//!   meets or are work of the draw's own. Its uses are noted as the next draw would note them
//!   (`NotedApart`). A draw that met a notice its device announces or refuses at every occurrence
//!   (`EachWatch`) keeps nothing: its next draw must meet it again.
//! - **Forgotten at every call but a bind.** Every call of the encoder but the two direct draws
//!   and the vertex and fragment buffer, bytes, texture and sampler state binds forgets it as it
//!   logs itself (`RenderLowering::log`, which every call does: at the call, on the device's
//!   lowering thread, and encoded again), refused or not, whatever it changes. A bind changes its
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
//! - **What a kept draw left holds for the next.** Each of its steps adds to what the encoder
//!   holds, and adding the same again changes nothing: its drawn area and coverage (rectangles
//!   and boxes, `clear_reach::DrawnArea`, `linear_attachment::Coverage`), the pass's reads
//!   (`PassStartReads::add`, which takes an equal range once) and the instance's gathered uses
//!   (`InstanceShadows::add`, which skips the uses of the last draw), its writes (none), its noted
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
//!   itself drawn. It takes no lock and changes no count other threads share: its counts, and
//!   those the kept draw counted per draw, are the encoder's until its next call or its end
//!   ([`Counted`]).
//! - **Checked in builds with debug assertions.** The first draw again of a kept draw, and every
//!   second one after it, takes every step instead, and checks that it announced nothing, recorded
//!   no state command, kept an equal draw, retained only its index buffer again, and left the
//!   encoder's drawn area, coverage, reads, gathered uses, noted uses and instance as they were.

mod rebound;

use ash::vk;
use metal2vulkan::bind::{BindShape, DrawCall};
use std::sync::Arc;
use std::time::Duration;

pub(super) use rebound::KeptRecording;
pub(crate) use rebound::KeptStorage;

use super::bound::Recorded;
use super::{Assembly, DrawBuffers, RenderLowering};
use crate::argument_table::Stamp;
use crate::degradation::EachWatch;
use crate::draw_source::DrawSource;
use crate::error::Error;
use crate::execution::{ExecutedPipeline, Execution};
use crate::noted_apart::NotedInto;
use crate::resource::ResourceKey;
use crate::tracker::{Use, UseRange};
use metal2vulkan::metal::RenderStages;

/// The encoder's last direct draw, where it is kept, and the counts of its draws again (the
/// module docs).
#[derive(Default)]
pub(super) struct DrawAgain {
    kept: Option<Kept>,
    counted: Counted,
    /// Draws again since the draw was kept: in builds with debug assertions, the first and every
    /// second one after it take every step instead (the module docs).
    #[cfg(debug_assertions)]
    hits: u64,
}

/// The counts of an encoder's draws again not yet added to its device's
/// (`WorkCounts::draws_as_before`, and what the kept draws counted per draw).
#[derive(Default)]
struct Counted {
    draws: u64,
    pushing_set_zero: u64,
    last_vertex: u64,
    /// Of `draws`, those rebound (`rebound.rs`), the stage tables they read in part, and the state
    /// commands and pushed writes they recorded.
    rebound: u64,
    tables: u64,
    recorded: Recorded,
}

/// What a kept draw was, what it read, and what a draw again repeats of it (the module docs).
#[derive(PartialEq)]
pub(super) struct Kept {
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
    /// It is, and takes every step instead, which [`Checked`] checks (builds with debug
    /// assertions).
    #[cfg(debug_assertions)]
    Check(Box<Checked>),
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
            samplers: self.core.device().sampler_slots_changed(),
        }
    }

    /// Forget the kept draw (a call came, another draw, or the encoder's end), adding its draws
    /// again's counts to the device's (the module docs).
    pub(super) fn forget_again(&mut self) {
        self.again.kept = None;
        let Counted {
            draws,
            pushing_set_zero,
            last_vertex,
            rebound,
            tables,
            recorded,
        } = self.again.counted;
        if draws > 0 {
            let counts = &self.core.device().shared().counts;
            counts.draws_drawn_again(draws - rebound, pushing_set_zero, last_vertex);
            if rebound > 0 {
                counts.draws_rebound(rebound, tables);
                recorded.count(counts);
            }
            self.again.counted = Counted::default();
        }
    }

    /// The stamps of the encoder's vertex, fragment, object and mesh tables.
    pub(super) fn table_stamps(&self) -> [Stamp; 4] {
        [&self.vertex, &self.fragment, &self.object, &self.mesh].map(|table| table.stamp())
    }

    /// Whether the draw of `call` assembling `assembly`, `draw` with `buffers`, is the kept draw
    /// again (the module docs): then, but where it is checked, it is recorded. Call it once the
    /// draw made the heaps' new textures resident.
    pub(super) fn draw_again(
        &mut self,
        call: &'static str,
        assembly: Assembly,
        draw: &DrawCall,
        buffers: &DrawBuffers<'_>,
    ) -> Again {
        let Some(kept) = self.again.kept.as_ref() else {
            return Again::No;
        };
        let index = buffers.index.as_ref().map(KeptIndex::of);
        let same = buffers.indirect.is_none()
            && buffers.patch_indices.is_none()
            && buffers.control_point_indices.is_none()
            && kept.call == call
            && kept.assembly == assembly
            && kept.draw == *draw
            && kept.index == index.flatten()
            && index.is_none_or(|index| index.is_some())
            && kept.command_buffer == self.current.draws
            && kept.under == self.under();
        if !same {
            return Again::No;
        }
        let stamps = self.table_stamps();
        if kept.stamps != stamps {
            return match kept.rebinds {
                true => self.draw_rebound(call, (assembly, draw), buffers, stamps),
                false => Again::No,
            };
        }
        #[cfg(debug_assertions)]
        {
            let hits = self.again.hits;
            self.again.hits += 1;
            if hits.is_multiple_of(2) {
                return Again::Check(Box::new(Checked::of(self, None)));
            }
        }
        let counted = &mut self.again.counted;
        counted.draws += 1;
        counted.pushing_set_zero += u64::from(kept.pushes_set_zero);
        counted.last_vertex += u64::from(kept.last_vertex);
        // Its noting: its reads of fresh textures, where any texture is fresh, and what the
        // encoder made resident since (nothing: no call came).
        match kept.reads_images && self.core.observes_fresh() {
            true => self.core.note_again(super::lock(&self.live).noted.uses()),
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
        watch: &EachWatch,
        (reads, rasterizes, depth_writes): (RenderStages, bool, bool),
        checked: Option<Box<Checked>>,
    ) {
        if let Some(kept) = self.again.kept.as_mut() {
            kept.reads = reads;
            kept.rasterizes = rasterizes;
            kept.depth_writes = depth_writes;
        }
        #[cfg(debug_assertions)]
        if let Some(checked) = checked {
            checked.check(self, watch.met());
        }
        #[cfg(not(debug_assertions))]
        let _: Option<Box<Checked>> = checked;
        if watch.met() {
            self.forget_again();
        }
    }
}

/// What a draw again that takes every step checks of it (the module docs): the kept draw, what
/// the encoder held before the draw, and the messages delivered on its thread by then.
#[cfg(debug_assertions)]
pub(super) struct Checked {
    kept: Kept,
    hits: u64,
    held: Held,
    announced: u64,
    /// What it worked out of its rebound slots, where it is a draw rebound (`rebound.rs`).
    rebound: Option<Rebound>,
}

/// What a draw rebound worked out (`rebound.rs`), which its every step must work out alike.
#[cfg(debug_assertions)]
pub(super) struct Rebound {
    commands: metal2vulkan::bind::DrawCommands,
    writes: Vec<(u32, crate::binding::SetWrite)>,
    offsets: Vec<u32>,
    pushes: Vec<crate::binding::PushWrite>,
    vertex_buffers: Vec<(u32, vk::Buffer, u64, u64)>,
    uses: Vec<Use<ResourceKey>>,
    resident: std::collections::BTreeSet<ResourceKey>,
}

#[cfg(debug_assertions)]
impl Rebound {
    /// What `live` and `binds` hold of a draw rebound.
    pub(super) fn of(live: &crate::live_bind::LiveBind) -> Rebound {
        Rebound {
            commands: live.commands.clone(),
            writes: Self::writes(live.kept.writes()),
            offsets: live.offsets.clone(),
            pushes: live.pushes.clone(),
            vertex_buffers: live.vertex_buffers.clone(),
            uses: live.uses.to_vec(),
            resident: live.uses.resident().clone(),
        }
    }

    fn writes(writes: &[crate::binding::Write]) -> Vec<(u32, crate::binding::SetWrite)> {
        (writes.iter())
            .map(|write| (write.set, crate::binding::SetWrite::of(write)))
            .collect()
    }

    /// Assert that every step of the draw worked out what it did, in `live` and `binds`.
    fn assert_as(&self, live: &crate::live_bind::LiveBind) {
        let writes = Self::writes(live.kept.writes());
        let same_writes = writes.len() == self.writes.len()
            && (writes.iter().zip(&self.writes))
                .all(|((set, write), (kept_set, kept))| set == kept_set && write.same(kept));
        let parts = [
            ("commands", live.commands == self.commands),
            ("set-0 writes", same_writes),
            ("dynamic offsets", live.offsets == self.offsets),
            ("push bytes", live.pushes == self.pushes),
            ("vertex buffers", live.vertex_buffers == self.vertex_buffers),
            ("uses", *live.uses == self.uses[..]),
            ("resident keys", *live.uses.resident() == self.resident),
        ];
        for (part, same) in parts {
            assert!(
                same,
                "a draw rebound worked out other {part} than its every step"
            );
        }
    }
}

/// No check in builds without debug assertions: no draw again takes every step.
#[cfg(not(debug_assertions))]
pub(super) enum Checked {}

/// What a draw again leaves as it was (the module docs).
#[cfg(debug_assertions)]
struct Held {
    drawn: crate::clear_reach::DrawnArea,
    instance_drawn: crate::clear_reach::DrawnArea,
    coverage: crate::linear_attachment::Coverage,
    pass_start: super::pass_start_reads::HeldReads,
    gathered: super::shadows::HeldGathered,
    noted: crate::noted_apart::NotedApart,
    instance: (bool, bool, bool, usize, usize, super::InterlockedStorage),
    holds: (usize, usize),
    pass: (
        std::collections::BTreeSet<u32>,
        std::collections::BTreeSet<u32>,
        bool,
    ),
}

#[cfg(debug_assertions)]
impl Held {
    fn of(encoder: &RenderLowering) -> Held {
        let current = &encoder.current;
        Held {
            drawn: encoder.drawn.clone(),
            instance_drawn: current.drawn.clone(),
            coverage: encoder.coverage.clone(),
            pass_start: encoder.pass_start.held(),
            gathered: current.shadows.gathered(),
            noted: super::lock(&encoder.live).noted.clone(),
            instance: (
                current.tile_memory,
                current.barriers,
                current.imageblock,
                current.pieces.len(),
                current.before.len(),
                current.interlocked_writes,
            ),
            holds: encoder.core.retained_holds(),
            pass: (
                encoder.inputs_read.clone(),
                encoder.extended_written.clone(),
                encoder.reaches_unnamed,
            ),
        }
    }

    /// Assert that `self`, held after a draw again took every step, is `before`, but for its
    /// holds: one more index buffer's where it reads one (`index`), and one more execution
    /// record where the device keeps them (`executed`).
    fn assert_as(&self, before: &Held, (index, executed): (bool, bool)) {
        let parts = [
            ("drawn area", self.drawn == before.drawn),
            (
                "instance's drawn area",
                self.instance_drawn == before.instance_drawn,
            ),
            ("coverage", self.coverage == before.coverage),
            ("pass's reads", self.pass_start == before.pass_start),
            ("gathered uses", self.gathered == before.gathered),
            ("noted uses", self.noted == before.noted),
            ("instance", self.instance == before.instance),
            ("pass", self.pass == before.pass),
        ];
        for (part, same) in parts {
            assert!(same, "a draw again changed its encoder's {part}");
        }
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
    /// The check of the draw again about to take every step, `rebound` where it is rebound.
    pub(super) fn of(encoder: &mut RenderLowering, rebound: Option<Rebound>) -> Checked {
        let kept = encoder.again.kept.take();
        let kept = kept.unwrap_or_else(|| unreachable!("a draw again without its kept draw"));
        Checked {
            kept,
            hits: encoder.again.hits,
            held: Held::of(encoder),
            announced: crate::degradation::announced_here(),
            rebound,
        }
    }

    /// Check the draw that took every step, which `met` a notice met at each occurrence or not,
    /// against the kept one (the module docs).
    fn check(self, encoder: &mut RenderLowering, met: bool) {
        let Checked {
            kept,
            hits,
            held,
            announced,
            rebound,
        } = self;
        if let Some(rebound) = &rebound {
            // Its uses differ from the kept draw's: what noting them adds is every step's own.
            rebound.assert_as(&crate::device::lock(&encoder.live));
        } else {
            assert_eq!(
                crate::degradation::announced_here(),
                announced,
                "a draw again announced a notice"
            );
            assert!(!met, "a draw again met a notice met at each occurrence");
            let executed = kept.execution.is_some();
            Held::of(encoder).assert_as(&held, (kept.index.is_some(), executed));
        }
        let again = encoder.again.kept.as_mut();
        let again = again.unwrap_or_else(|| panic!("a draw again kept no draw"));
        let (draw_alone, stamps, under) = (again.draw_alone, again.stamps, again.under);
        assert!(
            draw_alone || rebound.is_some(),
            "a draw again recorded a state command"
        );
        assert!(
            again.rebinds || rebound.is_none(),
            "a draw rebound whose every step keeps nothing to rebind"
        );
        // A draw rebound reads its tables under their stamps now, its uses noted into the state
        // now.
        (again.draw_alone, again.stamps, again.under) = (kept.draw_alone, kept.stamps, kept.under);
        assert!(*again == kept, "a draw again kept another draw");
        (again.draw_alone, again.stamps, again.under) = (draw_alone, stamps, under);
        encoder.again.hits = hits;
    }
}
