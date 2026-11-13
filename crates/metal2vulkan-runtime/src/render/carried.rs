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
//!   of no color groups, rasterization-rate map or replay, its command
//!   buffer holds the bind state, with the settle, and the kept draw ([`Carried`]) in place of
//!   giving the state to its device's spares (`live_bind::SpareLiveBinds`). With it go the kept
//!   draws its bind state set aside with the states of other pipeline states (`LiveBind::select`)
//!   that are kept still (under the encoder's epoch), carry as it does of their own
//!   (`Kept::carries`) through a state that holds its settle, and read what changes with no call
//!   as it did (`Kept::carries_with`), so the next encoder's check of it holds for each: those of
//!   the [`CARRIED_STATES`] states drawn most lately, its own among them
//!   (`LiveBind::carry_aside`, `WorkCounts::kept_draws_carried_aside`); the rest it lets go
//!   (`WorkCounts::kept_draws_carry_capped`, those past the bound that would carry). The shadows'
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
//!   resolves, input attachments, view mask) and layers, no memoryless color, visibility buffer,
//!   custom sample positions, extended-range color, rasterization-rate map, color groups or
//!   encoding of an ended encoder again (`kept_draws_carry_dropped_layout`); and where what the
//!   kept draw read that changes with no call is as it was: no no-copy buffer moved, the device's
//!   sampler IDs' slots as they were, and its residency of the generation the ended encoder's
//!   had (`kept_draws_carry_dropped_under`). Else it lets it go, counted, and starts from a
//!   spare, as any encoder.
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
//! - **Checked in builds with debug assertions.** As a draw rebound is: the first carried draw and
//!   every second one after it takes every step instead, which must keep the same draw, with the
//!   same commands, writes, offsets, push bytes, vertex buffers, uses and draw state; the count of
//!   carried draws carries with the draw (`DrawAgain::carries`), across a commit too.

use super::draw_again::Kept;
use super::{Lowering, Opening, RenderLowering, ViewSetup};
use crate::device::Device;
use crate::live_bind::LiveBind;
use crate::render_target::PassShape;
use crate::work_counts::{CarryDropped, Uncarried};

/// The most pipeline states whose kept draws a render encoder carries into its command buffer's
/// next render encoder: its last draw's and those of the three drawn most lately before it, of the
/// states its bind state sets aside (`LiveBind::select`), so an encoder whose draws switch between
/// up to four states finds each one's first draw carried (a window compositor's pass draws
/// through three). The kept draw of a state drawn before those, which would carry, is let go
/// (`WorkCounts::kept_draws_carry_capped`): an opening converts at most this many.
const CARRIED_STATES: usize = 4;

/// A render encoder's bind state, its settle and its kept draw, carried by its command buffer for
/// the next render encoder of it (the module docs).
pub(crate) struct Carried {
    live: Box<LiveBind>,
    kept: Kept,
    /// The ended encoder's pass's shape and layers, and its residency's generation.
    shape: PassShape,
    layers: u32,
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

/// What a render encoder starts from: the bind state and kept draw its command buffer's last
/// render encoder carried, where it fits the encoder's pass ([`fitting`]), else the bind state of
/// its device's last ended render encoder (`live_bind::SpareLiveBinds`).
pub(super) enum Start {
    Carried(Box<Carried>),
    Spare(Box<LiveBind>),
}

/// Whether `carried` fits the render encoder `opening` opens, its device's sampler IDs' slots
/// changed `samplers` times (the module docs); else why it is let go.
fn carried_fits(carried: &Carried, opening: &Opening, samplers: u64) -> Result<(), CarryDropped> {
    let layout = &opening.layout;
    let colors = layout.colors.iter().flatten();
    let same_pass = opening.shape == carried.shape
        && layout.layers == carried.layers
        && opening.rate_map.is_none()
        && opening.grouped.is_none()
        && layout.visibility.is_none()
        && layout.sample_positions.is_empty()
        && !opening.extended.any()
        && colors.clone().all(|color| !color.memoryless);
    if !same_pass {
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
    /// states of the [`CARRIED_STATES`] pipeline states drawn most lately that carry with it,
    /// every other set aside let go (the module docs).
    fn carried_out(&mut self, live: &mut LiveBind) -> Option<Kept> {
        let counts = &self.core.device().shared().counts;
        let Some(kept) = self.again_kept() else {
            counts.encoder_uncarried(match self.draw_calls {
                0 => Uncarried::Drawless,
                _ => Uncarried::NoneKept,
            });
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
        let capped = live.carry_aside(CARRIED_STATES - 1, |aside, settled| {
            attached += u64::from(settled && aside.carries_with_but_attached(kept));
            settled && aside.carries_with(kept)
        });
        counts.kept_draws_carry_capped(capped);
        counts.kept_draws_uncarried_attached(attached);
        self.take_kept()
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
            (
                self.grouped.is_none() && self.rate_map.is_none(),
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
            layers: lowering.layout.layers,
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

/// `carried` where it fits the encoder `opening` opens (`carried_fits`), its device's sampler
/// IDs' slots changed `samplers` times, else `None`, let go to `device` and counted.
pub(super) fn fitting(
    carried: Option<Box<Carried>>,
    opening: &Opening,
    (device, samplers): (&Device, u64),
) -> Option<Box<Carried>> {
    let carried = carried?;
    match carried_fits(&carried, opening, samplers) {
        Ok(()) => Some(carried),
        Err(why) => {
            device.shared().counts.kept_draw_carry_dropped(why);
            carried.let_go(device);
            None
        }
    }
}
