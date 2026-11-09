//! A render encoder's kept draw carried into the next render encoder of its command buffer
//! (`WorkCounts::kept_draws_carried`): that encoder starts from the ended encoder's bind state, its
//! settle and its kept draw (`render/draw_again.rs`), so that its first draw of the same pipeline
//! state is rebound (`render/draw_again/rebound.rs`, `WorkCounts::draws_rebound_carried`) instead
//! of taking every step.
//!
//! - **What carries.** Where an encoder ends with a kept draw that rebinds (`Kept::rebinds`),
//!   recorded into its pass's first instance (no split, no executed commands or tile dispatches,
//!   no stand-in, shadow or reformat of its own, the instance's views the default) through a bind
//!   state whose settle it holds (`LiveBind::settled`), that read no texture an attachment of its
//!   pass renders, in a pass of no color groups, rasterization-rate map or replay, its command
//!   buffer holds the bind state, with the settle, and the kept draw ([`Carried`]) in place of
//!   giving the state to its device's spares (`live_bind::SpareLiveBinds`). The shadows' copies it
//!   listed and the heap parts it kept go as they go at any end (`LiveBind::end_carried`): they
//!   are the residency's, which is the encoder's. The settle holds its variant's objects, which
//!   the command buffer holds already, so the carry keeps nothing alive past the command buffer.
//! - **Let go.** Any call of the command buffer that settles its last render encoder's end
//!   (`CommandBuffer::settle_end`: another encoder's opening, an event, a semaphore, the commit)
//!   lets it go to the device's spares, counted where it comes between
//!   (`WorkCounts::kept_draws_carry_dropped_between`): only the opening of a render encoder takes
//!   it. That encoder takes it ([`carried_fits`]) where its pass is the ended one's in everything
//!   the kept draw's settle and recording read: the same [`PassShape`] (formats, samples,
//!   resolves, input attachments, view mask) and layers, no memoryless color, visibility buffer,
//!   custom sample positions, extended-range color, rasterization-rate map, color groups or
//!   encoding of an ended encoder again (`kept_draws_carry_dropped_layout`); and where what the
//!   kept draw read that changes with no call is as it was: no no-copy buffer moved, the device's
//!   sampler IDs' slots as they were, and its residency of the generation the ended encoder's
//!   had (`kept_draws_carry_dropped_under`). Else it lets it go, counted, and starts from a
//!   spare, as any encoder.
//! - **Taken.** The encoder starts from the carried bind state, and keeps the carried draw under
//!   its own epoch ([`Kept::carried`]): after draws of no other state (`Kept::between`), recorded
//!   through no record (`Kept::settled`), under a render state of another stamp (`Kept::stated`),
//!   so its first draw of it derives again what the encoder's render state decides
//!   (`restated.rs`) and records every state command it sets into the encoder's command buffer,
//!   which holds none. Its tables' stamps are the encoder's tables' first ones: a new encoder
//!   inherits no binding, so its first draw brings in every slot its calls bound since the
//!   opening (`argument_table::Changes`), and its rebind stops where the bind state holds a slot
//!   bound that none of them bound (`WorkCounts::draws_carry_stopped_stale`): the state is the
//!   ended encoder's there, not this one's. Its uses are made again whole, and it stops where it
//!   samples a texture an attachment of its pass renders (`draws_carry_stopped_attached`). It
//!   begins its pass's first instance as every step's first draw begins it
//!   (`render/inline_first.rs`). What it compares at a draw again is the encoder's: the command
//!   buffer its instance records into, and what it read that changes with no call (`Under`), as
//!   of the opening.
//! - **Checked in builds with debug assertions.** As a draw rebound is: the first carried draw of a
//!   command buffer's encoders and every second one after it takes every step instead, which must
//!   keep the same draw, with the same commands, writes, offsets, push bytes, vertex buffers, uses
//!   and draw state; the count of carried draws carries with the draw (`DrawAgain::carries`).

use super::draw_again::Kept;
use super::{Lowering, Opening, RenderLowering, ViewSetup};
use crate::device::Device;
use crate::live_bind::LiveBind;
use crate::render_target::PassShape;
use crate::work_counts::CarryDropped;

/// A render encoder's bind state, its settle and its kept draw, carried by its command buffer for
/// the next render encoder of it (the module docs).
pub(crate) struct Carried {
    live: Box<LiveBind>,
    kept: Kept,
    /// The ended encoder's pass's shape and layers, and its residency's generation.
    shape: PassShape,
    layers: u32,
    residency: u64,
    /// The kept draws carried in since the command buffer's first render encoder
    /// (`DrawAgain::carries`).
    #[cfg(debug_assertions)]
    carries: u64,
}

impl Carried {
    /// Let it go: its bind state to `device`'s spares, as any ended encoder's.
    pub(crate) fn let_go(self: Box<Self>, device: &Device) {
        device.shared().live_binds.give(self.live);
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
    /// buffer's next render encoder, taken from it where it does (the module docs).
    fn carried_out(&mut self, live: &LiveBind) -> Option<Kept> {
        let kept = self.again_kept()?;
        let carries = kept.rebinds
            && kept.through_copies == (false, false)
            && kept.narrowed.is_empty()
            && live.settled.is_some()
            && self.earlier.is_empty()
            && self.grouped.is_none()
            && self.rate_map.is_none()
            && !self.replaying
            && !self.rerouting
            && self.executions.is_empty()
            && self.pending.is_empty()
            && self.dispatches.is_empty()
            && !self.dispatched
            && self.shadow.is_none()
            && self.depth_stand_in.is_none()
            && self.imageblock.is_none()
            && !self.tile_imageblock
            && !self.visibility.active()
            && self.current.pieces.is_empty()
            && self.current.reformats.is_empty()
            && self.current.views == ViewSetup::default()
            && !self.current.shadow
            && !self.current.shader_tests
            && self.current.shape.storage.is_empty();
        carries.then(|| self.take_kept()).flatten()
    }

    /// Start from `kept`, the kept draw the command buffer's last render encoder carried, of
    /// `carries` kept draws carried in before it (the module docs).
    fn carried_in(&mut self, kept: Kept, #[cfg(debug_assertions)] carries: u64) {
        let (stamps, under, draws) = (self.table_stamps(), self.under(), self.current.draws);
        let epoch = self.again_epoch();
        self.keep_carried(kept.carried_into(epoch, stamps, under, draws));
        #[cfg(debug_assertions)]
        self.set_carries(carries);
        self.core.device().shared().counts.kept_draw_carried();
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
        let Some(kept) = lowering.carried_out(&live) else {
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
        let carried = match start {
            Start::Carried(carried) => *carried,
            Start::Spare(live) => return Lowering { lowering, live },
        };
        #[cfg(debug_assertions)]
        lowering.carried_in(carried.kept, carried.carries);
        #[cfg(not(debug_assertions))]
        lowering.carried_in(carried.kept);
        Lowering {
            lowering,
            live: carried.live,
        }
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
