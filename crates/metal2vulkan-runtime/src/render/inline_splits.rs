//! A plain pass's later render pass instances recorded inline too: after a split of a pass whose
//! first instance records inline (`render/inline_first.rs`), each new instance of its shape, or
//! of its shape with colors read and written in the pixel interlock, records into the same
//! primary command buffer, its render pass instance begun after the one before it ends, in place
//! of a secondary command buffer the pass executes at its end.
//!
//! A split instance costs a secondary command buffer and its state recorded again in it on every
//! driver measured: on ms01's Iris Xe (ANV) the encoding thread's CPU of a split read is about
//! 61 to 87 us in a secondary against 5 to 9 us inline, and its GPU time 3.7 to 5.4 us more. In
//! one primary command buffer the draws keep what the draws before them bound: a pipeline binding
//! "affects subsequent commands that interact with the given pipeline type in the command buffer
//! until a different pipeline of the same type is bound to the bind point", dynamic state
//! "persists for the lifetime of the command buffer, or until modified by another dynamic state
//! setting command, or made invalid by binding a pipeline in which that state is statically
//! specified", and descriptor sets and vertex and index buffers stay bound likewise
//! (`render/bound.rs`), across the render pass instances, the copies and the barriers between
//! them, which bind nothing.
//!
//! - **What goes between two instances** is what the pass's end records there for them: a barrier
//!   from everything before to everything after, the shadowed textures' copies into the later
//!   instance, each followed by such a barrier (`render/shadows.rs`), and its draws' steps before
//!   it (`Before`). The copies depend on every draw of the later instance (and of the instances
//!   that continue it, `Instance::continues`), which is known only once the next instance
//!   begins or the pass ends. So the later instance's commands (its draws, and the barriers
//!   between them) are kept as they come (`Kept`), and recorded into the primary command buffer
//!   when its group ends ([`RenderCommandEncoder::flush_inline`]): there the end of the
//!   instance before it, the barrier, the copies, the steps, the render pass instance's begin,
//!   and its commands, in their order. The encoder's commands meanwhile record into the command
//!   buffer before the pass, as for the first instance. Every command keeps its place in
//!   submission order: where the pass's end would record it.
//! - **The copies** are the ones the pass's end plans, one instance at a time and in order,
//!   through the same shadowed textures' record (`EncoderCore::plan_shadows`): the instance
//!   before's first (the pass's first instance's go before the pass, into the command buffer
//!   before it), as each instance's group ends. An instance's writes of the colors the pass
//!   loads are bounded by its draws' scissors as the end bounds them, the pass's drawn area and
//!   reach taken as far as the pass has gone ([`RenderCommandEncoder::plan_one`]): the next copy
//!   takes the texels inside them alone. Where the end would find them unbounded by its time, it
//!   copies more, never fewer. Where neither the instance's draws nor any command of the encoder
//!   before them used a shadowed texture, nothing is planned: there is nothing to copy, as the
//!   end finds where no instance used one.
//! - **The kept draws' room** is one the device keeps from an earlier such instance whose pass
//!   ended ([`SpareKept`]): each draw's recording is written over one left there, so an instance
//!   of no more draws than an earlier one's allocates nothing for them.
//! - **Admitted** at a split, the new instance's first draw recording: the pass's first instance
//!   recorded inline, every instance since did, and the instance before is no continuation; the
//!   pass has no attachment that is a shadowed texture (`shadows::attachments_shadowed`), so the
//!   copies before an instance depend on its own draws alone; and the new instance is of the
//!   first instance's shape but for the colors its draws read and write as storage images in the
//!   pixel interlock (`PassShape::storage`, which its render pass leaves out of its subpass, as
//!   the end's would), and allows what the first allowed (`inline_first::Inline`): its pipeline
//!   made, no view of another format, multiview, shader-side test, depth stand-in, tile
//!   dispatch, execution of render commands, pending pipeline, imageblock or visibility query in
//!   the pass so far, no stand-in arriving with it but memoryless colors' backings (below), and
//!   no color its draw reads through a snapshot (`render/snapshot.rs`), whose feedback loop
//!   layout the later render pass would lack. Elsewhere the instance records as before, in a
//!   secondary, and so does every later one.
//! - **Render passes and framebuffers** ([`Later`]): one of each per shape and set of
//!   attachments, made at the first instance that needs it. Two attachment references "are
//!   compatible if they have matching format and sample count, or are both
//!   `VK_ATTACHMENT_UNUSED`", so a render pass that leaves a color out for the interlock is not
//!   compatible with the first instance's, whose framebuffer it cannot begin in: it takes a
//!   framebuffer of its own, as every instance does once memoryless colors' backings stand in for
//!   their transient images. Such a framebuffer holds a placeholder in the slot of each color left
//!   out (`storage_placeholder.rs`), as the end's does, never a view of the image the draws write
//!   as a storage image.
//! - **A memoryless color's backing** arrives at the pass's first split (`next_instance`), and
//!   the first instance never wrote the color's transient image (`render/inline_first.rs`): the
//!   backing is cleared to the first instance's clear right before the instance it arrives with,
//!   between barriers from everything before and to everything after, in place of after the first
//!   instance, as the end clears it where that instance alone records inline. Every instance
//!   from there loads and stores it.
//! - **Counted**: `WorkCounts::inline_split_instances`.

use super::bound::{Bound, Recorded};
use super::pending::OwnedRecording;
use super::{inline_first, shadows, Clears, Instance, Next, Recording, RenderLowering, ViewSetup};
use crate::attachment_stand_in::Cause;
use crate::clear_reach;
use crate::color_list::ColorList;
use crate::device::lock;
use crate::encoder::EncoderCore;
use crate::error::Error;
use crate::render_pass::PassLayout;
use crate::render_target::{
    self, Feedback, FramebufferObject, PassShape, Position, RenderPassObject,
};
use crate::render_values::LoadAction;
use crate::shadow::ShadowCopy;
use crate::tracker::{Barrier, LegacyBarrier};
use ash::vk;
use std::sync::Mutex;

/// The commands of an instance recorded inline after a split (the module docs), kept until the
/// instance's copies are known.
pub(super) struct Kept {
    /// Its draws and commands, in a room the device kept where it had one ([`SpareKept`]).
    room: KeptRoom,
    /// How many of the room's recordings are its draws': the first, in order.
    draws: usize,
    /// The render pass and framebuffer its render pass instance begins in, of a [`Later`] the
    /// pass holds.
    objects: (vk::RenderPass, vk::Framebuffer),
    /// The clears of the memoryless colors' backings that arrived with it, before it.
    clears: Option<Clears>,
}

/// Where a [`Kept`] keeps its draws and commands, which the device keeps once its pass ended
/// ([`SpareKept`]).
#[derive(Default)]
struct KeptRoom {
    /// Draws' recordings, each of its made pipeline: those of the instance's draws first, in
    /// order, then the room an earlier instance's left, whose handles nothing records.
    recordings: Vec<(OwnedRecording, vk::Pipeline)>,
    /// Its commands, in order: each draw's place among them, and the barriers.
    commands: Vec<KeptCommand>,
}

/// The most rooms a device keeps for its next instances recorded inline after a split.
const SPARE_ROOMS: usize = 8;

/// The most draws' recordings a room the device keeps holds, and four times as many commands:
/// a room with more goes.
const SPARE_ROOM_DRAWS: usize = 64;

/// The rooms of the instances recorded inline after a split whose passes ended, which the device
/// keeps for the next such instances, so that one of as many draws as an earlier one's takes no
/// allocation for its recordings: at most [`SPARE_ROOMS`], each of at most [`SPARE_ROOM_DRAWS`]
/// draws' room, the latest given taken first.
#[derive(Default)]
pub(crate) struct SpareKept(Mutex<Vec<KeptRoom>>);

impl SpareKept {
    /// A room for a new instance's draws and commands: the latest one given back, else an empty
    /// one.
    fn take(&self) -> KeptRoom {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep the room of `kept`, an instance whose pass ended and whose commands are recorded,
    /// emptied of its commands, where it is small enough and fewer than [`SPARE_ROOMS`] are kept;
    /// else it goes.
    pub(super) fn give(&self, kept: Kept) {
        let Kept { mut room, .. } = kept;
        let small = room.recordings.len() <= SPARE_ROOM_DRAWS
            && room.commands.capacity() <= 4 * SPARE_ROOM_DRAWS;
        if !small {
            return;
        }
        room.commands.clear();
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_ROOMS {
            spare.push(room);
        }
    }
}

/// One of [`Kept`]'s commands.
enum KeptCommand {
    /// The next of its draws.
    Draw,
    /// A memory barrier between its draws over the render targets, of the view mask
    /// (`render_target::between_draws`).
    Barrier(u32),
    /// A barrier before a draw that reaches interlocked storage a draw before it wrote, of the
    /// view mask (`render_target::between_storage_draws`).
    StorageBarrier(u32),
    /// A barrier before a draw that writes, in the stages, what a draw before it read as its
    /// indirect arguments, of the view mask (`render_target::after_indirect_reads`).
    IndirectBarrier(u32, vk::PipelineStageFlags),
    /// A barrier before a draw whose fragment shader touches what a draw before it wrote before
    /// its fragments, of the view mask (`render_target::after_geometry_writes`).
    GeometryBarrier(u32, LegacyBarrier),
}

impl Kept {
    /// Keep a draw's `recording`, of its made `pipeline`: over the room an earlier instance's
    /// recording left where there is one.
    pub(super) fn draw(&mut self, recording: &Recording, pipeline: vk::Pipeline) {
        match self.room.recordings.get_mut(self.draws) {
            Some((owned, made)) => {
                recording.own_into(owned);
                *made = pipeline;
            }
            None => (self.room.recordings).push((recording.owned(), pipeline)),
        }
        self.draws += 1;
        self.room.commands.push(KeptCommand::Draw);
    }

    /// Keep a memory barrier between draws over the render targets.
    pub(super) fn barrier(&mut self, view_mask: u32) {
        self.room.commands.push(KeptCommand::Barrier(view_mask));
    }

    /// Keep a barrier before a draw that reaches interlocked storage.
    pub(super) fn storage_barrier(&mut self, view_mask: u32) {
        self.room
            .commands
            .push(KeptCommand::StorageBarrier(view_mask));
    }

    /// Keep a barrier before a draw that writes, in `stages`, what a draw before it read as its
    /// indirect arguments.
    pub(super) fn indirect_barrier(&mut self, view_mask: u32, stages: vk::PipelineStageFlags) {
        self.room
            .commands
            .push(KeptCommand::IndirectBarrier(view_mask, stages));
    }

    /// Keep a barrier before a draw whose fragment shader touches what a draw before it wrote
    /// before its fragments.
    pub(super) fn geometry_barrier(&mut self, view_mask: u32, barrier: LegacyBarrier) {
        self.room
            .commands
            .push(KeptCommand::GeometryBarrier(view_mask, barrier));
    }
}

/// A render pass and framebuffer inline instances after a split begin in: a render pass of
/// their shape that loads and stores every attachment, and a framebuffer of the pass's
/// attachments as they stand, made at the first such instance; the first instance's framebuffer
/// serves those of its shape and attachments.
pub(super) struct Later {
    /// The shape of the instances it serves.
    shape: PassShape,
    /// The backings' arrivals before them (`inline_first::Begun::stood`).
    stood: u32,
    pass: RenderPassObject,
    /// `None`: the first instance's.
    framebuffer: Option<FramebufferObject>,
}

/// The copies of the instance a split ends, planned as the split's draw is admitted
/// ([`RenderCommandEncoder::inline_split`]), recorded once its new instance is installed
/// ([`RenderCommandEncoder::begin_inline_split`]); and the new instance's objects and clears
/// ([`Kept`]).
pub(super) struct SplitBegin {
    copies: Vec<ShadowCopy>,
    objects: (vk::RenderPass, vk::Framebuffer),
    clears: Option<Clears>,
}

impl RenderLowering {
    /// Whether the draw about to record into `next` begins a new instance that records inline
    /// after the current one (the module docs; `records`: the draw records, `made` its pipeline,
    /// `stood` a stand-in arriving with it); where it does, the current
    /// instance's copies planned, which the draw records once installed. `None` where it does
    /// not, or where the plan is refused (the pass's end plans it again, and refuses it there).
    pub(super) fn inline_split(
        &mut self,
        next: &Next,
        stood: Option<&PassLayout>,
        (records, made): (bool, bool),
    ) -> Option<SplitBegin> {
        let inline_first::Inline::Begun(begun) = &self.inline else {
            return None;
        };
        let Next::New {
            instance,
            split: true,
        } = next
        else {
            return None;
        };
        // The memoryless colors' backings arriving with it, which no instance before rendered:
        // nothing else may arrive.
        let arriving = match stood {
            Some(stood) => arriving_backings(&self.layout, stood)?,
            None => ColorList::new(),
        };
        let admitted = begun.splits
            && records
            && made
            && self.current.inline
            && !self.current.continues
            && !self.replaying
            && !self.rerouting
            && self.pending.is_empty()
            && self.executions.is_empty()
            && self.dispatches.is_empty()
            && !self.dispatched
            && self.imageblock.is_none()
            && !self.tile_imageblock
            && self.shadow.is_none()
            && self.depth_stand_in.is_none()
            && !self.visibility.active()
            && instance.pieces.is_empty()
            && same_but_storage(&instance.shape, &begun.shape)
            && instance.reformats.is_empty()
            && instance.views == ViewSetup::default()
            && !instance.shadow
            && !instance.shader_tests
            && instance.feedback == Feedback::NONE;
        if !admitted {
            return None;
        }
        // Its render pass and framebuffer: the instances' of its shape since the last backings
        // arrived, else made now, of the pass's attachments as they stand from it on.
        let stood_now = begun.stood + u32::from(stood.is_some());
        let found = (begun.later.iter())
            .find(|later| later.shape == *instance.shape && later.stood == stood_now)
            .map(|later| {
                let framebuffer = later.framebuffer.as_ref().unwrap_or(&begun.framebuffer);
                (later.pass.raw, framebuffer.raw)
            });
        let layout = stood.unwrap_or(&self.layout);
        let made_later = match found {
            Some(_) => None,
            None => {
                let at = Position {
                    first: false,
                    last: false,
                };
                let (none, general) = (Feedback::NONE, vk::ImageLayout::GENERAL);
                let device = self.core.device().clone();
                let (shape, views) = (&instance.shape, instance.views);
                let pass =
                    render_target::final_pass(&device, layout, shape, at, none, general).ok()?;
                // The first instance's framebuffer is compatible with a render pass of its
                // shape, of the same attachments.
                let framebuffer = match **shape == begun.shape && stood_now == 0 {
                    true => None,
                    false => Some(
                        render_target::framebuffer(&device, &pass, layout, shape, views, none)
                            .ok()?,
                    ),
                };
                Some(Later {
                    shape: PassShape::clone(shape),
                    stood: stood_now,
                    pass,
                    framebuffer,
                })
            }
        };
        let backings = (arriving.iter()).filter_map(|&index| {
            let color = layout.colors.get(index)?.as_ref()?;
            Some((&color.target, color.clear))
        });
        let clears = Clears::plan(backings).ok()?;
        let copies = self.plan_one(None).ok()?;
        let inline_first::Inline::Begun(begun) = &mut self.inline else {
            return None;
        };
        let objects = match (found, made_later) {
            (Some(objects), _) => objects,
            (None, Some(later)) => {
                let framebuffer = later.framebuffer.as_ref().unwrap_or(&begun.framebuffer);
                let objects = (later.pass.raw, framebuffer.raw);
                begun.later.push(later);
                objects
            }
            (None, None) => return None,
        };
        begun.stood = stood_now;
        Some(SplitBegin {
            copies,
            objects,
            clears,
        })
    }

    /// Begin the instance a split's draw installed inline after the one before
    /// ([`inline_split`](Self::inline_split)): the instance before's copies recorded (before the
    /// pass for the pass's first instance; else with its kept commands, `flush_inline`'s), and
    /// the new instance's commands kept from here on. It has no secondary: the draw began none
    /// for it (`Instance::draws`).
    pub(super) fn begin_inline_split(&mut self, begin: SplitBegin) {
        let Some(before) = self.earlier.len().checked_sub(1) else {
            return;
        };
        let SplitBegin {
            copies,
            objects,
            clears,
        } = begin;
        self.record_flushed(before, &copies);
        let device = self.core.device().clone();
        self.current.inline = true;
        self.current.kept = Some(Kept {
            room: device.shared().kept_rooms.take(),
            draws: 0,
            objects,
            clears,
        });
        self.current.rerecord = None;
        device.shared().counts.inline_split_instance();
    }

    /// Record the last instance recorded inline after a split whose commands are still kept,
    /// where one is (the module docs): its copies planned, then recorded with its commands into
    /// the pass's command buffer. As the pass ends, before its own plan. Refused as the plan is.
    pub(super) fn flush_inline(&mut self) -> Result<(), Error> {
        let waiting = |instance: &Instance| instance.kept.is_some() && !instance.planned;
        if waiting(&self.current) {
            let copies = self.plan_one(None)?;
            let bound = &mut self.held_bound;
            record_kept(&mut self.core, bound, &self.current, &copies, &self.layout);
            self.current.planned = true;
            return Ok(());
        }
        if let Some(at) = self.earlier.iter().position(waiting) {
            let copies = self.plan_one(Some(at))?;
            self.record_flushed(at, &copies);
        }
        Ok(())
    }

    /// The copies before the head of a group, the current instance (`at` `None`) or the earlier
    /// one at `at`, through the encoder's shadowed textures, as the pass's end plans them for it
    /// (`shadows::plan`). Its writes of the colors the pass loads are bounded by its draws'
    /// scissors where the end bounds them: an instance after a split (not the pass's first,
    /// whose copies the steps before the pass follow) that no instance continues (whose writes
    /// the end bounds no more), of draws alone (no reformat or piece), in a pass whose draws so
    /// far reach the tiles they touch (`clear_reach::PassReach::Draws`, which no later draw
    /// changes but to reaching none, and where no color is cleared, so no reach clear writes
    /// one) and whose drawn area bounds them so far. The bound holds whatever the pass does
    /// after: the instance's draws write those colors inside their scissors alone. Where the end
    /// would find the pass's area unbounded by then (an execution after, or past the rectangles
    /// `clear_reach::DrawnArea` keeps), it copies more, never fewer.
    fn plan_one(&mut self, at: Option<usize>) -> Result<Vec<ShadowCopy>, Error> {
        let instance = match at {
            Some(at) => &self.earlier[at],
            None => &self.current,
        };
        // No copy, and no use to note, where neither its draws nor any command of the encoder
        // before them used a shadowed texture (the pass's attachments are none: the module docs),
        // as the pass's end plans nothing where no instance did.
        if instance.shadows.is_empty() && !self.core.used_shadows() {
            return Ok(Vec::new());
        }
        let alone = at.is_none()
            && instance.kept.is_some()
            && instance.reformats.is_empty()
            && instance.pieces.is_empty();
        let bound = alone.then(|| instance.drawn.bounds()).flatten();
        let reach = clear_reach::PassReach::of(&self.layout, self.called, self.drew);
        let area = match reach {
            clear_reach::PassReach::Draws => self.drawn.bounds(),
            clear_reach::PassReach::Nothing | clear_reach::PassReach::Every => None,
        };
        let loaded = ColorList::of((self.layout.colors.iter()).map(|color| {
            color
                .as_ref()
                .is_some_and(|color| color.load == LoadAction::Load)
        }))?;
        let attached = render_target::attachment_uses(&self.layout)?;
        let planned = shadows::plan(
            &mut self.core,
            std::iter::once((&instance.shadows, bound)),
            (&self.layout, &attached),
            (true, area.map(|area| (area, &loaded[..]))),
            area.is_some().then_some(&self.drawn),
        )?;
        Ok(planned.into_iter().next().unwrap_or_default())
    }

    /// Record the copies `copies` planned for the earlier instance at `at`, and mark it
    /// planned: before the pass for the pass's first instance, recorded inline as its draws
    /// came; else between the instance before it and it, with its kept commands.
    fn record_flushed(&mut self, at: usize, copies: &[ShadowCopy]) {
        if self.earlier[at].kept.is_none() {
            // The pass's first instance: its copies go before the pass, into the command buffer
            // before the pass's, as the pass's end records them before its first instance.
            self.core.shadow_copies(copies);
        } else {
            let (bound, instance) = (&mut self.held_bound, &self.earlier[at]);
            record_kept(&mut self.core, bound, instance, copies, &self.layout);
        }
        self.earlier[at].planned = true;
    }
}

/// Whether `shape`, a new instance's, is `first`'s, the pass's first instance's, but for the
/// attachments its draws' subpass leaves out for the pixel interlock (`PassShape::storage`),
/// which its own render pass leaves out too.
fn same_but_storage(shape: &PassShape, first: &PassShape) -> bool {
    let storage = first.storage.clone();
    PassShape {
        storage,
        ..shape.clone()
    } == *first
}

/// The indices of the memoryless colors whose backings arrive in `stood`, the pass's layout from
/// a split on where it was `layout`, each loaded `Clear` (`inline_first::Inline` gives each the
/// clear its first instance loaded): `None` where anything else arrives (a scratch, sample planes,
/// a hidden depth and stencil), or where such a color is loaded otherwise. A split brings every
/// memoryless color's backing (`next_instance`), at the pass's first split.
fn arriving_backings(layout: &PassLayout, stood: &PassLayout) -> Option<ColorList<usize>> {
    let unchanged_depth = layout.depth_stencil.is_none() && stood.depth_stencil.is_none();
    if !unchanged_depth || layout.colors.len() != stood.colors.len() {
        return None;
    }
    let mut arriving = ColorList::new();
    for (index, (was, now)) in layout.colors.iter().zip(&stood.colors).enumerate() {
        let (was, now) = match (was, now) {
            (Some(was), Some(now)) => (was, now),
            (None, None) => continue,
            _ => return None,
        };
        if was.planes.is_some() != now.planes.is_some() {
            return None;
        }
        match (&was.stand_in, &now.stand_in) {
            (None, None) => {}
            (None, Some(stand_in))
                if stand_in.cause == Cause::Backing
                    && now.memoryless
                    && now.load == LoadAction::Clear =>
            {
                arriving.push(index).ok()?;
            }
            _ => return None,
        }
    }
    Some(arriving)
}

/// Record `instance`, kept (the module docs), into the pass's command buffer after the render
/// pass instance before it: that one's end, a barrier from everything before, `copies` (one run
/// of them, `EncoderCore::held_shadow_copies`, followed by a barrier), the steps its draws take
/// before it, its render pass instance's begin
/// in `objects` (a render pass compatible with its draws' pipelines, and the pass's
/// framebuffer), and its commands, recorded through `bound`, the record of what the command
/// buffer bound.
fn record_kept(
    core: &mut EncoderCore,
    bound: &mut Bound,
    instance: &Instance,
    copies: &[ShadowCopy],
    layout: &PassLayout,
) {
    let (Some(held), Some(kept)) = (core.held_pass(), instance.kept.as_ref()) else {
        return;
    };
    let (pass, framebuffer) = kept.objects;
    let device = core.device().clone();
    let shared = device.shared();
    let raw = shared.vk();
    // SAFETY: the held command buffer is recording inside the render pass instance before this
    // one, and only this encoder records it.
    unsafe {
        raw.cmd_end_render_pass(held);
        Barrier::everything().record(raw, held);
    }
    core.held_shadow_copies(copies);
    if let Some(clears) = &kept.clears {
        // SAFETY: the held command buffer is recording outside a render pass instance, and only
        // this encoder records it; the backings are live until it completes (the layout's
        // textures, which the encoder retains), rest in `GENERAL` and take transfer writes
        // (`attachment_stand_in::stand_in`, of a format `inline_first::opens_inline` found
        // clearable by transfer).
        unsafe { clears.record_in(raw, held) };
    }
    let [width, height] = layout.extent;
    let clear_values = render_target::clear_values(layout);
    let info = vk::RenderPassBeginInfo::default()
        .render_pass(pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        })
        .clear_values(clear_values.as_slice());
    let mut recorded = Recorded::default();
    // SAFETY: the held command buffer is recording outside a render pass instance, and only this
    // encoder records it; the steps' objects are live until it completes (retained at their
    // draws); the pass and framebuffer are live until it completes (the encoder holds them to its
    // end, which retains them), the pass compatible with every kept draw's pipeline (made for
    // the instance's shape, `render_target::final_pass`) and the framebuffer made for the pass's
    // first instance, of the same attachments; every image rests in `GENERAL`; every kept
    // draw's objects are live until the command buffer completes (retained at the draw), and
    // `bound` is the record of what the draws recorded into it bound, which the copies and
    // barriers recorded since (transfer commands) leave bound.
    unsafe {
        for step in &instance.before {
            step.record_in(raw, held);
        }
        raw.cmd_begin_render_pass(held, &info, vk::SubpassContents::INLINE);
        let begun = crate::work_counts::PassBegun::InlineSplit;
        shared.counts.render_pass_begun(begun);
        let mut draws = kept.room.recordings[..kept.draws].iter();
        for command in &kept.room.commands {
            match command {
                KeptCommand::Draw => {
                    let Some((owned, pipeline)) = draws.next() else {
                        break;
                    };
                    recorded += Recording {
                        raw,
                        command_buffer: held,
                        layout: owned.layout,
                        shape: owned.shape,
                        pipeline: *pipeline,
                        sets: &owned.sets,
                        offsets: &owned.offsets,
                        pushed: &owned.pushed,
                        push_descriptor: shared.push_descriptor,
                        pushes_after_pipeline_binds: shared.pushes_after_pipeline_binds,
                        heap: owned.heap,
                        push: &owned.push,
                        vertex_buffers: &owned.vertex_buffers,
                        index: owned.index,
                        indirect: owned.indirect,
                        dynamic: &owned.dynamic,
                        settled: None,
                        topology: owned.topology,
                        primitive_restart: owned.primitive_restart,
                        state: &owned.state,
                        extended: owned.extended,
                        mesh: shared.mesh_draws.as_ref(),
                        draw: owned.draw,
                        pair: owned.pair.clone(),
                    }
                    .record(bound);
                }
                KeptCommand::Barrier(view_mask) => {
                    render_target::between_draws(raw, held, *view_mask)
                }
                KeptCommand::StorageBarrier(view_mask) => {
                    render_target::between_storage_draws(raw, held, *view_mask)
                }
                KeptCommand::IndirectBarrier(view_mask, stages) => {
                    render_target::after_indirect_reads(raw, held, *view_mask, *stages)
                }
                KeptCommand::GeometryBarrier(view_mask, barrier) => {
                    render_target::after_geometry_writes(raw, held, *view_mask, *barrier)
                }
            }
        }
    }
    recorded.count(&shared.counts);
}
