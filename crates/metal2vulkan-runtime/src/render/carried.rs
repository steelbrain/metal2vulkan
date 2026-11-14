//! A render encoder's kept draws carried into the next render encoder of its command buffer, or,
//! past its commit, into a render encoder of a later command buffer of its queue
//! (`WorkCounts::kept_draws_carried`, `kept_draws_carried_across`): that encoder starts from the
//! ended encoder's bind state, its settles and its kept draws (`render/draw_again.rs`), so that its
//! first draw of each pipeline state carried is rebound (`render/draw_again/rebound.rs`,
//! `WorkCounts::draws_rebound_carried`) instead of taking every step.
//!
//! - **What carries.** Where an encoder ends with a kept draw that rebinds (`Kept::rebinds`),
//!   recorded into its pass's first instance (no split, no executed commands or tile dispatches,
//!   no stand-in, shadow or reformat of its own, the instance's views the default) through a bind
//!   state whose settle it holds (`LiveBind::settled`), with no sampled slot narrowed to levels
//!   apart from those an attachment of its pass renders (`render/attached_levels.rs`; else it is
//!   let go, `WorkCounts::kept_draws_uncarried_attached`), under a depth/stencil state that does
//!   not fail every fragment (whose draw takes every step once restated, `restated.rs`), in a pass
//!   of no color groups, rasterization-rate map, custom sample positions or replay, its command
//!   buffer holds the bind state, with the settle, and the kept draw ([`Carried`]) in place of
//!   giving the state to its device's spares (`live_bind::SpareLiveBinds`). With it go the kept
//!   draws its bind state set aside with the states of other pipeline states (`LiveBind::select`)
//!   that are kept still (under the encoder's epoch), carry as it does of their own
//!   (`Kept::carries`) through a state that holds its settle, and read what changes with no call
//!   as it did (`Kept::carries_with`), so the next encoder's check of it holds for each
//!   (`LiveBind::carry_aside`, `WorkCounts::kept_draws_carried_aside`); the rest it lets go. A
//!   bind state sets aside at most `live_bind::SET_ASIDE` (7) states, so a carry holds at most
//!   eight kept draws: an encoder whose draws switch between up to eight states finds each one's
//!   first draw carried (`WorkCounts::draws_rebound_carried`). A carried state the next encoder
//!   never draws is work its opening did for nothing: the carries an opening takes
//!   (`kept_draws_carried` and `kept_draws_carried_aside`) less the carried states drawn
//!   (`draws_rebound_carried` and `draws_carry_stopped`). The shadows'
//!   copies it listed and the heap parts it kept go as they go at any end
//!   (`LiveBind::end_carried`): they are the residency's, which is the encoder's. The carry names
//!   the program's objects by key and handle alone (the bind state's values, the kept draws'
//!   index buffers, their pipeline states held weakly), and holds only the runtime's own: the
//!   settles' variants, the kept draws' `VkPipeline`s (`Kept::made`) and settings. An encoder
//!   that ends carrying none counts the first reason ([`Uncarried`],
//!   `WorkCounts::encoders_uncarried_*`): no draw, no kept draw, or which of the above did not
//!   hold.
//! - **Let go.** Any call of the command buffer but the commit that settles its last render
//!   encoder's end (`CommandBuffer::settle_end`: another encoder's opening, an event, a
//!   semaphore) lets it go to the device's spares, counted
//!   (`WorkCounts::kept_draws_carry_dropped_between`): only the opening of a render encoder takes
//!   it. That encoder takes it ([`carried_fits`]) where its pass is the ended one's in everything the
//!   kept draw's settle and recording read: the same [`PassShape`] (formats, samples,
//!   resolves, input attachments, view mask) and layers (the layout's, and the descriptor's
//!   `renderTargetArrayLength`, 0 or those, which the draw's bind shape holds), no memoryless
//!   color, visibility buffer, custom sample positions, extended-range color, rasterization-rate
//!   map, color groups or encoding of an ended encoder again (`kept_draws_carry_dropped_layout`);
//!   and where what the kept draw read that changes with no call is as it was: no no-copy buffer
//!   moved, the device's sampler IDs' slots as they were, and its residency of the generation the
//!   ended encoder's had (`kept_draws_carry_dropped_under`). Else it lets it go, counted, and
//!   starts from a spare, as any encoder.
//! - **Across a commit.** The commit hands it to its queue's place (`queue.rs`), which holds one:
//!   the one it held, which no render encoder took, is let go
//!   (`WorkCounts::kept_draws_carry_dropped_commit`). A render encoder of a later command buffer
//!   of the queue that holds no carry of its own takes it, as above. Its kept draws are across
//!   (`Kept::across`): the command buffer they were recorded in may have completed and let go of
//!   what it retained, so the first draw of each, rebound, retains into its own command buffer
//!   what every step retains beyond its tables' objects, which its encoder's tables retain at each
//!   draw: its pipeline state, its variant's objects and its `VkPipeline`
//!   (`LastRetained::retain_draw`), and its index buffer (`rebound.rs`). A key names one object
//!   for good (`ResourceKey`) and a pipeline state is held weakly, so a draw of an index buffer,
//!   texture or pipeline state the program released and made again is no draw of a carried one's:
//!   the carry keeps none of them alive, and a draw never takes one gone. The runtime's objects it
//!   holds are bounded by the one carry a place holds, until the next commit that carries or the
//!   place goes.
//! - **Taken.** The encoder starts from the carried bind state, and keeps the carried draws under
//!   its own epoch ([`Kept::carried`]), each beside its state as the ended encoder left it: after
//!   draws of other states (`Kept::between`), recorded through no record (`Kept::settled`), under
//!   a render state of another stamp (`Kept::stated`), so its state's first draw derives again
//!   what the encoder's render state decides (`restated.rs`) and records every state command it
//!   sets into the instance's command buffer. That instance is the one the encoder opened with,
//!   the pass's first: a carried draw is kept only while it is (`RenderLowering::select_kept`),
//!   and records into its command buffer as it stands, which a draw of another state before it
//!   may have begun, each such draw kept, drawn again or rebound under the epoch, as every draw
//!   between a kept draw and its next is. Its tables' stamps are the encoder's tables' first
//!   ones: a new encoder inherits no binding, so its first draw brings in every slot its calls
//!   bound since the opening (`argument_table::Changes`), and its rebind stops where the bind
//!   state holds a slot bound that none of them bound (`WorkCounts::draws_carry_stopped_stale`):
//!   the state is the ended encoder's there, not this one's. A texture or sampler slot its
//!   pipeline state does not read is no part of that state (`argument_table::ReadSlots`), so one
//!   the new encoder leaves unbound stops nothing. Its uses are made again whole, and so are its
//!   uses of shadowed textures (`RenderLowering::draw_shadows`), a texture of a color attachment
//!   of its pass it reads through the texture's copy among them (`texture/attached_reads.rs`):
//!   the kept draw gathered its own into the ended encoder's instance, so it gathers these into
//!   its instance, whose copies in its pass's plan then takes (`render/shadows.rs`), as every
//!   step gathers them. It goes on only where it uses some exactly where the kept draw used some:
//!   whether a draw may be recorded again into another instance follows from that
//!   (`Kept::again`), which it takes of the kept draw. It stops where it reads such a texture
//!   through a slot narrowed to levels apart from the attached ones, which the kept draw did not
//!   (`draws_carry_stopped_attached`). It begins its pass's first instance as every step's draw
//!   begins it (`render/inline_first.rs`), where no draw before it began it. What it compares at
//!   a draw again is the encoder's: the command buffer its instance records into, and what it
//!   read that changes with no call (`Under`), as of the opening.
//! - **A reader through snapshots** (`render/snapshot.rs`) carries as any kept draw: same pass
//!   shape and layers, residency, sampler slots and no-copy moves, the feedback loop layout being
//!   no part of the shape (pipelines and passes are compatible whatever their colors' layouts).
//!   What its snapshots need is of the slots the next encoder binds after its opening: its first
//!   draw there, rebound (`rebound.rs`), works out again that it reads through snapshots of the
//!   same slots and colors, pinning none, through the kept plan, and that the snapshot is its
//!   form; it begins the pass's first instance in its secondary, with its colors in the feedback
//!   loop layout from its begin, where no draw before it began that instance, else records into
//!   that instance where it holds them so; and it binds the snapshots of the textures that
//!   encoder bound. Where any of those does not hold, that draw takes every step
//!   (`WorkCounts::draws_carry_stopped_snapshot`); the carry's other draws are rebound still. A
//!   kept draw that reads through no snapshot stops so where its first draw there would read the
//!   new pass's colors through snapshots, in place or after the split (`rebound.rs`).
//! - **None kept.** An encoder that drew but ended with no kept draw carries nothing, and counts
//!   what a kept draw of it would have come to (`WorkCounts::encoders_none_kept_next_*`): where
//!   its end is not one a kept draw carries from (`carry_ends`), the first reason, a split or
//!   another; else its command buffer holds its pass's shape and layers and what changes with no
//!   call as it ended ([`NoneKept`]), until the next call. Another call than a render encoder's
//!   opening counts it as one that came between, the commit as the command buffer's last, and an
//!   opening by the checks a carry meets ([`carried_fits`]): another pass, or a turn of a
//!   color's texture to its other face, what changed with no call, or a fit. It changes nothing
//!   recorded: a count alone.
//! - **Checked in builds with debug assertions.** As a draw rebound is: the first carried draw and
//!   every second one after it takes every step instead, which must keep the same draw, with the
//!   same commands, writes, offsets, push bytes, vertex buffers, uses and draw state; the count of
//!   carried draws carries with the draw (`DrawAgain::carries`), across a commit too.

use super::draw_again::Kept;
use super::{Lowering, Opening, RenderLowering, ViewSetup};
use crate::device::Device;
use crate::live_bind::LiveBind;
use crate::render_target::PassShape;
use crate::work_counts::{CarryDropped, NoneKeptNext, Uncarried};

/// A render encoder's bind state, its settle and its kept draw, carried by its command buffer for
/// the next render encoder of it (the module docs).
pub(crate) struct Carried {
    live: Box<LiveBind>,
    kept: Kept,
    /// The ended encoder's pass's shape and layers (the layout's, and the descriptor's
    /// `renderTargetArrayLength` as the encoder keeps it), and its residency's generation.
    shape: PassShape,
    layers: Layers,
    residency: u64,
    /// Whether its command buffer was committed, and its queue holds it for a later command
    /// buffer's render encoder ([`across`](Carried::across)).
    across: bool,
    /// The kept draws carried in since the first render encoder of the command buffer, or of the
    /// first of the command buffers of its queue it was carried across (`DrawAgain::carries`).
    #[cfg(debug_assertions)]
    carries: u64,
}

impl Carried {
    /// Let it go: its bind state to `device`'s spares, as any ended encoder's.
    pub(crate) fn let_go(self: Box<Self>, device: &Device) {
        device.shared().live_binds.give(self.live);
    }

    /// It, as its command buffer commits, for its queue to hold for the first render encoder of
    /// a later command buffer of it (the module docs). Its count of draws carried in goes on
    /// (`DrawAgain::carries`), so the draws carried across are checked every second one too.
    pub(crate) fn across(mut self: Box<Self>) -> Box<Self> {
        self.across = true;
        self
    }
}

/// The end of a render encoder that drew but kept no draw, held by its command buffer for its
/// next render encoder's opening to count what a kept draw of it would have come to (the module
/// docs): its pass's shape and layers, and what changes with no call as it ended.
pub(crate) struct NoneKept {
    shape: PassShape,
    layers: Layers,
    /// The no-copy buffers' moves, its residency's generation and its device's sampler IDs'
    /// slots' changes, at its end.
    under: (u64, u64, u64),
}

impl NoneKept {
    /// What a kept draw of it carried into the render encoder `opening` opens would have come
    /// to, its device's sampler IDs' slots changed `samplers` times: the checks [`carried_fits`]
    /// makes.
    fn next(&self, opening: &Opening, samplers: u64) -> NoneKeptNext {
        if !same_pass(&self.shape, self.layers, opening) {
            return NoneKeptNext::Layout;
        }
        let moved = crate::buffer::host_cells_moved();
        match self.under == (moved, opening.residency.generation(), samplers) {
            true => NoneKeptNext::Fits,
            false => NoneKeptNext::Under,
        }
    }
}

/// What a render encoder's opening takes from its command buffer's last render encoder's end:
/// the kept draw it carried ([`Carried`]), and its end where it kept none ([`NoneKept`]).
#[derive(Default)]
pub(super) struct Handed {
    pub(super) carried: Option<Box<Carried>>,
    pub(super) none_kept: Option<Box<NoneKept>>,
}

impl Handed {
    /// It, where the opening turned a color's texture to its other face (`render/face_swap.rs`):
    /// nothing, the kept draw carried let go, as it bound the face presented before, and an end
    /// that kept none counted as one of another pass.
    pub(super) fn turned(self, device: &Device) -> Handed {
        if let Some(carried) = self.carried {
            carried.let_go(device);
        }
        if self.none_kept.is_some() {
            let counts = &device.shared().counts;
            counts.encoder_none_kept_next(NoneKeptNext::Layout);
        }
        Handed::default()
    }
}

/// What a render encoder starts from: the bind state and kept draw its command buffer's last
/// render encoder carried, where it fits the encoder's pass ([`fitting`]), else the bind state of
/// its device's last ended render encoder (`live_bind::SpareLiveBinds`).
pub(super) enum Start {
    Carried(Box<Carried>),
    Spare(Box<LiveBind>),
}

/// A pass's layers, as a kept draw's settle and recording read them: its layout's, and its
/// descriptor's `renderTargetArrayLength` as its encoder keeps it (0, or the layout's), which the
/// draw's bind shape holds (`BindShape::render_target_array_length`): a pass of one layer reads 0
/// or 1 there.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Layers {
    layout: u32,
    array_length: u32,
}

impl Layers {
    /// The layers `lowering`'s pass draws.
    fn of(lowering: &RenderLowering) -> Layers {
        Layers {
            layout: lowering.layout.layers,
            array_length: lowering.render_target_array_length,
        }
    }
}

/// Whether the render encoder `opening` opens draws a pass of `shape` and `layers` in everything
/// a kept draw's settle and recording read (the module docs).
fn same_pass(shape: &PassShape, layers: Layers, opening: &Opening) -> bool {
    let layout = &opening.layout;
    let opened = Layers {
        layout: layout.layers,
        array_length: opening.render_target_array_length,
    };
    opening.shape == *shape
        && opened == layers
        && opening.rate_map.is_none()
        && opening.grouped.is_none()
        && layout.visibility.is_none()
        && layout.sample_positions.is_empty()
        && !opening.extended.any()
        && (layout.colors.iter().flatten()).all(|color| !color.memoryless)
}

/// Whether `carried` fits the render encoder `opening` opens, its device's sampler IDs' slots
/// changed `samplers` times (the module docs); else why it is let go.
fn carried_fits(carried: &Carried, opening: &Opening, samplers: u64) -> Result<(), CarryDropped> {
    if !same_pass(&carried.shape, carried.layers, opening) {
        return Err(CarryDropped::Layout);
    }
    let moved = crate::buffer::host_cells_moved();
    if !carried
        .kept
        .carried_under(moved, opening.residency.generation(), samplers)
        || carried.residency != opening.residency.generation()
    {
        return Err(CarryDropped::Under);
    }
    Ok(())
}

impl RenderLowering {
    /// The kept draw this encoder, ending with its bind state `live`, carries into its command
    /// buffer's next render encoder, taken from it where it does, with those set aside with the
    /// states of other pipeline states that carry with it, every other set aside let go (the
    /// module docs).
    fn carried_out(&mut self, live: &mut LiveBind) -> Option<Kept> {
        let counts = &self.core.device().shared().counts;
        let Some(kept) = self.again_kept() else {
            counts.encoder_uncarried(match self.draw_calls {
                0 => Uncarried::Drawless,
                _ => Uncarried::NoneKept,
            });
            if self.draw_calls > 0 && self.last_draw_empty {
                counts.encoder_last_draw_empty();
            }
            if self.draw_calls > 0 {
                self.hold_none_kept(live);
            }
            return None;
        };
        if let Err(why) = self.carry_ends(live) {
            counts.encoder_uncarried(why);
            return None;
        }
        if !kept.carries() {
            match (kept.carries_but_attached(), kept.rebinds) {
                (true, _) => counts.kept_draws_uncarried_attached(1),
                (false, false) => counts.encoder_uncarried(Uncarried::Unrebound),
                (false, true) => counts.encoder_uncarried(Uncarried::FailingState),
            }
            return None;
        }
        let mut attached = 0;
        live.carry_aside(|aside, settled| {
            attached += u64::from(settled && aside.carries_with_but_attached(kept));
            settled && aside.carries_with(kept)
        });
        counts.kept_draws_uncarried_attached(attached);
        self.take_kept()
    }

    /// Of the encoder, with its bind state `live`, drawn but ending with no kept draw: where its
    /// end is not one a kept draw carries from, the first reason counted; else its end held by its
    /// command buffer for its next render encoder's opening ([`NoneKept`], the module docs).
    fn hold_none_kept(&self, live: &LiveBind) {
        let counts = &self.core.device().shared().counts;
        match self.carry_ends(live) {
            Err(Uncarried::Split) => counts.encoder_none_kept_next(NoneKeptNext::Split),
            Err(_) => counts.encoder_none_kept_next(NoneKeptNext::Ends),
            Ok(()) => {
                let none_kept = Box::new(NoneKept {
                    shape: (*self.current.shape).clone(),
                    layers: Layers::of(self),
                    under: (
                        crate::buffer::host_cells_moved(),
                        self.residency.generation(),
                        self.core.generation().sampler_slots_changed(),
                    ),
                });
                self.core.command_buffer().hold_none_kept(none_kept);
            }
        }
    }

    /// Whether the encoder ends in a state its kept draw carries from (the module docs): its
    /// bind state holds the settle of its last draw's state, and its pass recorded no step a
    /// carried draw's first rebind does not repeat; else the first reason it does not
    /// (`WorkCounts::encoders_uncarried_*`).
    fn carry_ends(&self, live: &LiveBind) -> Result<(), Uncarried> {
        let current = &self.current;
        let checks = [
            (live.settled.is_some(), Uncarried::Unsettled),
            (
                self.earlier.is_empty() && current.pieces.is_empty(),
                Uncarried::Split,
            ),
            // A kept draw's `VkPipeline` holds its pass's custom sample positions
            // (`render/draw_shape.rs`'s `PipelineInputs`), which an opening of the same shape
            // may not set.
            (
                self.grouped.is_none()
                    && self.rate_map.is_none()
                    && self.layout.sample_positions.is_empty(),
                Uncarried::Pass,
            ),
            (!self.replaying && !self.rerouting, Uncarried::Replay),
            (
                self.executions.is_empty()
                    && self.pending.is_empty()
                    && self.dispatches.is_empty()
                    && !self.dispatched,
                Uncarried::Commands,
            ),
            (
                self.shadow.is_none()
                    && self.depth_stand_in.is_none()
                    && !current.shadow
                    && !current.shader_tests,
                Uncarried::ShaderTests,
            ),
            (
                self.imageblock.is_none() && !self.tile_imageblock,
                Uncarried::Imageblock,
            ),
            (!self.visibility.active(), Uncarried::Visibility),
            (
                current.reformats.is_empty()
                    && current.views == ViewSetup::default()
                    && current.shape.storage.is_empty(),
                Uncarried::Instance,
            ),
        ];
        match checks.into_iter().find(|&(holds, _)| !holds) {
            Some((_, why)) => Err(why),
            None => Ok(()),
        }
    }

    /// Start from `kept`, the kept draw the command buffer's last render encoder carried, or
    /// the last render encoder of a command buffer committed on its queue where `across`, and
    /// those `live`, its bind state, set aside with it, of `carries` kept draws carried in before
    /// it (the module docs).
    fn carried_in(
        &mut self,
        mut kept: Kept,
        live: &mut LiveBind,
        across: bool,
        #[cfg(debug_assertions)] carries: u64,
    ) {
        let (stamps, under) = (self.table_stamps(), self.under());
        let (epoch, draws) = (self.again_epoch(), (self.current.draws, across));
        kept.carried_into(epoch, stamps, under, draws);
        self.keep_carried(kept);
        let aside = live.carried_aside(|kept| kept.carried_into(epoch, stamps, under, draws));
        #[cfg(debug_assertions)]
        self.set_carries(carries);
        let counts = &self.core.device().shared().counts;
        counts.kept_draw_carried(across);
        counts.kept_draws_carried_aside(aside);
    }
}

impl Lowering {
    /// The lowering at [`RenderCommandEncoder::end_encoding`](super::RenderCommandEncoder), its
    /// bind state and kept draw carried into its command buffer's next render encoder where they
    /// carry (the module docs), else its bind state given to its device, as at every end.
    pub(super) fn ending_carried(self) -> Box<RenderLowering> {
        let Lowering {
            mut lowering,
            mut live,
        } = self;
        let Some(kept) = lowering.carried_out(&mut live) else {
            lowering.core.device().shared().live_binds.give(live);
            return lowering;
        };
        live.end_carried();
        let carried = Box::new(Carried {
            live,
            kept,
            shape: (*lowering.current.shape).clone(),
            layers: Layers::of(&lowering),
            residency: lowering.residency.generation(),
            across: false,
            #[cfg(debug_assertions)]
            carries: lowering.carries(),
        });
        let command_buffer = lowering.core.command_buffer();
        command_buffer.carry(carried);
        lowering
    }

    /// The lowering of an encoder, `lowering`, starting from `start` (the module docs): a
    /// carried bind state, and its kept draw, or a spare one.
    pub(super) fn starting_from(mut lowering: Box<RenderLowering>, start: Start) -> Lowering {
        let Carried {
            mut live,
            kept,
            across,
            #[cfg(debug_assertions)]
            carries,
            ..
        } = match start {
            Start::Carried(carried) => *carried,
            Start::Spare(live) => return Lowering { lowering, live },
        };
        #[cfg(debug_assertions)]
        lowering.carried_in(kept, &mut live, across, carries);
        #[cfg(not(debug_assertions))]
        lowering.carried_in(kept, &mut live, across);
        Lowering { lowering, live }
    }
}

/// The kept draw `handed` carried where it fits the encoder `opening` opens (`carried_fits`), its
/// device's sampler IDs' slots changed `samplers` times, else `None`, let go to `device` and
/// counted; and where `handed` holds an end that kept none, what a kept draw of it would have come
/// to, counted ([`NoneKept`]).
pub(super) fn fitting(
    handed: Handed,
    opening: &Opening,
    (device, samplers): (&Device, u64),
) -> Option<Box<Carried>> {
    if let Some(none_kept) = handed.none_kept {
        let next = none_kept.next(opening, samplers);
        device.shared().counts.encoder_none_kept_next(next);
    }
    let carried = handed.carried?;
    match carried_fits(&carried, opening, samplers) {
        Ok(()) => Some(carried),
        Err(why) => {
            device.shared().counts.kept_draw_carry_dropped(why);
            carried.let_go(device);
            None
        }
    }
}
