//! A compute encoder's bind state, kept between its dispatches: each dispatch that does not take
//! what the last one settled (`DispatchMemo`) brings the state the encoder's last such dispatch
//! bound to its own in place, and makes its bind shape again only where a change may have changed
//! it, as a render encoder's draw does (`live_bind.rs`).
//!
//! - **Brought in place.** The kernel's slots are derived again from the encoder's table and
//!   compared with the kept ones (`ArgumentTable::refresh_reading`), at each dispatch whose table
//!   may bring in another: a table read last through the same pipeline state's interface and
//!   samplers' maps, that no call changed since (its stamp, `argument_table::Stamp`), whose slots
//!   that tell the device's view base alignment apart the state still holds as they were read
//!   (`argument_table::AtBindViews`), and that no rebind or detach of a no-copy buffer may have
//!   changed (`buffer::host_cells_moved`), is not read again ([`ReadFrom`]; checked whole in debug
//!   builds). Where only calls changed it since it was so read, and the table keeps the slots they
//!   changed (`ArgumentTable::refresh_changed`), only those slots are brought in. Every other field
//!   is computed as before and compared ([`Reshape::field`]); the stage-in region, which no shape
//!   reads, changes no shape ([`Reshape::unshaped`]). The new value is always stored, so the state
//!   is the dispatch's exactly. While the same slots stay bound, nothing is allocated. The kinds
//!   of values a dispatch brought in changed ([`Reshape::moved`]) let go of the commands kept
//!   below where they may read them.
//! - **What the encoder made resident, listed once a change.** The state's used resources are
//!   listed again (`Residency::used_resources`, `WorkCounts::dispatch_used_resources_listed`) only
//!   where the residency changed its generation since they were listed (`Residency::generation`:
//!   each change of what it holds draws a new one), a no-copy buffer moved since (their
//!   addresses), or an indirect command buffer made resident no longer holds its ID's slot as they
//!   state (`Residency::holds_slots_as`): what they are a function of, but for those, is the
//!   residency's contents, which a generation names. Checked whole in debug builds.
//! - **The shape found again only where it may have changed.** Where no change brought in may have
//!   changed the shape, it is the kept one (`WorkCounts::dispatch_shapes_made` and
//!   `dispatch_shapes_found` stay); after any other change, it is one the pipeline state planned
//!   for lately where it has it (`ComputePipelineState::planned_shape`, compared in place), else it
//!   is made again, whole. The pipeline state's plan of a kept shape is found by identity first
//!   (`ComputePipelineState::planned_for`), so a dispatch whose offsets alone moved compares no
//!   shape at all, however many resources the encoder made resident.
//! - **Kept from one encoder to the next.** An encoder that ends leaves its state, with its shape,
//!   to its command buffer's next compute encoder (**Parked in its command buffer**, below), and
//!   the commit gives it to its device ([`SpareLiveDispatches`]), whose next compute encoder opened
//!   starts from it, as a dispatch starts from the one before it: exact for the reason a dispatch's
//!   is, every field of the state a dispatch binds being derived again at each dispatch that
//!   settles. The memo of what its last dispatch settled goes as the device takes it
//!   ([`LiveDispatch::let_go`]): it holds the pipeline state, sampler states and pipelines, where the
//!   state holds values alone (its bindings, its used resources and argument samplers as a bind
//!   state lists them, its table's pipeline state held weakly), so a kept one holds nothing alive.
//!   The lists of its resident textures' shadows and carriers go at each end
//!   ([`ResidentShadows`], [`ResidentUnreadable`]).
//! - **Commands kept while the state binds what they read.** The commands its last direct dispatch
//!   that instantiated made are kept, with what that dispatch planned held weakly
//!   (`compute::Instantiated`), and go on with the state to the next encoder: a later dispatch of
//!   what the same dispatch planned takes them while no change since moved a kind of value they
//!   may read, so dispatches that bind two textures in turn, one encoder each, instantiate once.
//! - **Views, sampler descriptions and host pages kept for its binds.** The views, the sampler
//!   descriptions and where the buffers over host pages lie that its dispatches' binds made or
//!   found are kept (`binding::BindMemo`), by everything each is a function of, and go on with the
//!   state to the next encoder: a dispatch that binds what one before it bound works none out
//!   again. The memo holds its textures, sampler states and host cells weakly, so a kept one holds
//!   nothing alive.
//! - **Parked in its command buffer.** A compute encoder's end parks its state, with what its last
//!   dispatch settled, and the box of its core in its command buffer ([`Parked`]); the command
//!   buffer's next compute encoder opens on them rather than on the device's spares
//!   (`WorkCounts::parked_compute_taken`), and the commit gives them back to the device, the memo let
//!   go. The memo's entry so goes on, and a dispatch whose new table, read, brings in the very state
//!   the entry settled takes it ([`LiveDispatch::settled_across`],
//!   `WorkCounts::dispatches_settled_across_encoders`): one encoder for each dispatch of the same
//!   bindings settles once. Exact: the entry is a function of the bind state and of the inputs
//!   `DispatchMemo` compares beside its table's stamp (`compute::DispatchMemo`), and the table's
//!   stamp stands for the table-derived part of that state alone; bringing the state in place
//!   notes every value it changed ([`Reshape::moved`]), so where it changed none, and the samplers'
//!   maps (the table's and the residency's) are the entry's, the state is the one it settled. A
//!   parked entry holds nothing its command buffer does not: its pipeline state, variant and
//!   `VkPipeline`s the command buffer's dispatch held until it completes. The device's sampler
//!   states the entry listed are let go as it parks, as are the resident textures' lists, and the
//!   dispatch that takes it lists them again: those the device lists at the count of sampler slot
//!   changes the entry compares, so the very states it listed.
//! - **Resident textures looked at once a change.** A dispatch's look at the shadows of the
//!   textures the encoder made resident (`shadow::shadow_uses_of`), and a reading of texture
//!   handles' look for one a carrier holds (`pvrtc::degrade_unreadable`), read the lists kept here,
//!   made again only where they may have changed.

use metal2vulkan::bind::{BindShape, BindState, Dispatch, HeapPlacement, ImageblockSize};
use metal2vulkan::pipeline::Stage;
use std::sync::{Arc, Mutex};

use super::{DispatchMemo, DispatchTables, Instantiated};
use crate::address_map::SamplerMaps;
use crate::argument_table::Stamp;
use crate::binding::{BindMemo, CommandUses};
use crate::device::{lock, Device};
use crate::encoder::{CoreBox, EncoderCore};
use crate::error::Error;
use crate::function_table::VisibleFunctionTable;
use crate::heap_sets;
use crate::live_bind::{Reshape, ValueKinds};
use crate::pipeline::{ComputePipelineState, WeakComputePipelineState};
use crate::pvrtc::ResidentUnreadable;
use crate::sampler::ArgumentSamplers;
use crate::shadow::ResidentShadows;
use crate::work_counts::Counting;

/// The most bind states a device keeps for its compute encoders to start from: one per encoder
/// open at once beyond it starts empty, as every encoder did before.
const SPARE_LIVE_DISPATCHES: usize = 4;

/// A compute encoder's bind state as its last dispatch that settled bound it, its shape while no
/// change may have changed it, and what that dispatch settled (the module docs).
#[derive(Default)]
pub(in crate::compute) struct LiveDispatch {
    /// The state. Each dispatch that settles brings it to its own through [`Reshape`]
    /// ([`refresh`](Self::refresh)).
    pub(in crate::compute) state: BindState,
    /// The state's shape, `None` where a change may have changed it.
    shape: Option<Arc<BindShape>>,
    /// The count of the device's sampler IDs' slots' changes (`DeviceShared::sampler_slots_changed`)
    /// at which the state's argument samplers are every live sampler state, `None` where they are
    /// not (none listed yet, or those a variant's set-1 arrays hold alone).
    listed_at: Option<u64>,
    /// What the state's kernel table was read from ([`ReadFrom`]).
    read_from: ReadFrom,
    /// What the dispatch that last brought the state in place settled, where it is kept for the
    /// next dispatch of the same inputs (`DispatchMemo`): of this state, so a dispatch that takes
    /// it binds this state.
    pub(in crate::compute) memo: DispatchMemo,
    /// The shadows' copies of the textures the encoder made resident.
    pub(in crate::compute) shadows: ResidentShadows,
    /// The first texture the encoder made resident whose bytes a carrier holds for copies alone.
    pub(in crate::compute) unreadable: ResidentUnreadable,
    /// The last dispatch's uses (`binding::command_uses_into`), whose storage the next one's
    /// take, as a draw's take the last draw's (`LiveBind::uses`).
    uses: CommandUses,
    /// The views, sampler descriptions and host pages its dispatches' binds made or found
    /// (`binding::BindMemo`).
    bind: BindMemo,
    /// The commands its last direct dispatch that instantiated made, while the state binds what
    /// they read (`Instantiated`).
    instantiated: Instantiated,
}

/// What a kept state's kernel table was read from at the dispatch that read it last
/// ([`LiveDispatch::refresh`]): the encoder's table by its stamp, the pipeline state whose
/// interface it was read through, the samplers' address-mode maps, and the count of no-copy
/// buffers' moves (`buffer::host_cells_moved`); and the generation of the residency the state's
/// used resources were listed of (`Residency::generation`). A table that keeps its stamp, read
/// through the same state with the same maps while no no-copy buffer moved, brings in what it
/// brought in then where the state's slots that tell the view base alignment apart are still those
/// it read under (`AtBindViews::holds`): a slot's binding is a function of the object bound there,
/// of the table's other entries, of the function's reading of the slot (the interface), of the
/// maps, and of those slots.
#[derive(Default)]
struct ReadFrom {
    /// The table's stamp.
    table: Option<Stamp>,
    /// The pipeline state it was read through, held weakly: while it is held, no other state lies
    /// where that one lay, so a state it names is the very one.
    pipeline: Option<WeakComputePipelineState>,
    /// The maps it was read with.
    maps: SamplerMaps,
    /// The count of no-copy buffers' moves as it was read.
    moved: u64,
    /// The generation of the residency the state's used resources were listed of, at `moved`
    /// moves; `None` before any were.
    resident: Option<u64>,
}

impl ReadFrom {
    /// Whether the table was read through `pipeline` with `maps` at `moved` moves: then a table
    /// that keeps its stamp brings in what it brought in.
    fn reads(&self, pipeline: &ComputePipelineState, maps: &SamplerMaps, moved: u64) -> bool {
        self.moved == moved
            && (self.pipeline.as_ref()).is_some_and(|read| read.names(pipeline))
            && self.maps == *maps
    }
}

impl LiveDispatch {
    /// Bring the state to the one a dispatch of `tables` binds, in place (the module docs): the
    /// kernel's table read as the pipeline state's interface reads it (a buffer where the kernel
    /// reads a visible-function table reads as a null table: `ArgumentTable::refresh_reading`),
    /// its samplers bound as `maps` binds them, its threadgroup memory lengths, the imageblock size
    /// for a kernel with an explicit imageblock (one without plans with none, so its shape is the
    /// one the executor computes for it), the stage-in region, the resources the encoder made
    /// resident, the device's sampler states listed (those `samplers` took, which it takes where
    /// the kept listing is not of every live state at `changed` changes of their slots:
    /// `sampler::ArgumentSamplers`), and the live tables the pipeline state made. Each change that
    /// may change the shape is noted, and the shape kept only where none may have. The kinds of
    /// values it changed.
    pub(in crate::compute) fn refresh(
        &mut self,
        tables: &DispatchTables<'_>,
        maps: &SamplerMaps,
        (changed, samplers): (u64, &mut ArgumentSamplers),
        function_tables: &[VisibleFunctionTable],
        counts: Counting<'_>,
    ) -> ValueKinds {
        let mut reshape = Reshape::default();
        let pipeline = tables.pipeline;
        let views = pipeline.at_bind_views();
        let state = &mut self.state;
        reshape.field(&mut state.view_base_alignment, views.alignment());
        state.uniform_views_at_residual = views.at_residual();
        let moved = crate::buffer::host_cells_moved();
        let same = self.read_from.reads(pipeline, maps, moved);
        // The state's used resources are what the encoder made resident still where the residency
        // kept its generation since they were listed of it (`Residency::generation`), no no-copy
        // buffer moved since (their addresses), and each indirect command buffer made resident
        // holds its ID's slot as they state (`Residency::holds_slots_as`).
        let residency = tables.residency;
        let generation = residency.generation();
        let resident = self.read_from.moved == moved
            && self.read_from.resident == Some(generation)
            && residency.holds_slots_as(&state.used_resources);
        let table = tables.table;
        let read = (Stage::Kernel, pipeline.interface());
        let kept = state.stages.entry(Stage::Kernel).or_default();
        // Its slots that tell the view base alignment apart are as the state holds them
        // (`AtBindViews::holds`): another table's offsets may have left some out since.
        let views_held = views.holds(Stage::Kernel, kept);
        let read_under = self.read_from.table.filter(|_| same && views_held);
        if read_under == Some(table.stamp()) {
            #[cfg(debug_assertions)]
            table.check_reading(kept, read, (maps, views));
        } else {
            // Only the slots calls changed since it was read, where the table keeps them.
            let in_part = read_under.is_some_and(|stamp| {
                let kept = (&mut *kept, stamp);
                table.refresh_changed(kept, read, (maps, views), &mut reshape, counts)
            });
            if !in_part {
                table.refresh_reading(kept, read, (maps, views), &mut reshape);
            }
        }
        reshape.field_from(&mut kept.threadgroup_lengths, tables.threadgroup_lengths);
        self.read_from.table = Some(table.stamp());
        if !(self.read_from.pipeline.as_ref()).is_some_and(|read| read.names(pipeline)) {
            self.read_from.pipeline = Some(pipeline.downgrade());
        }
        if self.read_from.maps != *maps {
            self.read_from.maps = maps.clone();
        }
        self.read_from.moved = moved;
        let imageblock = match pipeline.imageblock_cell_bytes() {
            Some(_) => tables.imageblock,
            None => ImageblockSize::default(),
        };
        reshape.field(&mut state.imageblock_size, imageblock);
        // No shape reads it (`BindState::stage_in_region`).
        reshape.unshaped(&mut state.stage_in_region, tables.stage_in_region);
        debug_assert!(
            !resident || state.used_resources == residency.used_resources(),
            "a dispatch's kept used resources that are not its residency's"
        );
        if !resident {
            counts.dispatch_used_resources_listed();
            reshape.field(&mut state.used_resources, residency.used_resources());
        }
        self.read_from.resident = Some(generation);
        // Listed again only where the device's sampler states changed since: not where the
        // listing kept is of every live state at the count of their slots' changes that holds;
        // else they are taken and compared state by state.
        let current = samplers.states().is_none() && self.listed_at == Some(changed);
        if !current {
            let states = samplers.take(tables.pipeline.generation());
            if !heap_sets::lists(&state.argument_samplers, states) {
                reshape.field(&mut state.argument_samplers, heap_sets::listed(states));
            }
            self.listed_at = samplers.listed_at();
        }
        let listed = function_tables
            .iter()
            .map(VisibleFunctionTable::listed)
            .collect();
        reshape.field(&mut state.function_tables, listed);
        // Values no shape reads (`BindState::heap_placement`, `used_in_token_order`,
        // `resident_apart`): the residency lists its used resources in token order, and root tables
        // the commit points elsewhere keep them out, as a dispatch's whole state has them.
        reshape.unshaped(&mut state.heap_placement, HeapPlacement::ResourceId);
        reshape.unshaped(&mut state.used_in_token_order, true);
        reshape.unshaped(&mut state.resident_apart, true);
        #[cfg(debug_assertions)]
        self.check_whole(tables, maps, samplers.states(), function_tables);
        if reshape.may_reshape() {
            self.shape = None;
        }
        self.instantiated.moved(reshape.moved());
        reshape.moved()
    }

    /// Whether the memo's entry, which an earlier compute encoder of the command buffer settled
    /// and parked (the module docs, **Parked in its command buffer**), serves a direct dispatch of
    /// `tables` and `dispatch` at `counted` moves and sampler slot changes: it holds in every
    /// input but its table's stamp (`DispatchMemo::holds_but_stamp`), and the table, read now with
    /// its samplers' maps and the device's sampler states `samplers` takes where it needs them
    /// ([`refresh`](Self::refresh)), brings in the very state it settled, no value of any kind
    /// changed, with the samplers' maps it was settled with. Then it goes on under the table's stamp ([`Across::Taken`]). Where the
    /// entry holds but for the stamp, the state is brought in place either way, and a dispatch that
    /// takes it not settles from the state as it lies ([`Across::Refreshed`]).
    pub(in crate::compute) fn settled_across(
        &mut self,
        tables: &DispatchTables<'_>,
        (dispatch, counted): (&Dispatch, (u64, u64)),
        samplers: &mut ArgumentSamplers,
        function_tables: &[VisibleFunctionTable],
        counts: Counting<'_>,
    ) -> Result<Across, Error> {
        if !(self.memo).holds_but_stamp(&self.state, tables, dispatch, counted) {
            return Ok(Across::Missed);
        }
        let maps = tables.sampler_maps()?;
        let listing = (counted.1, &mut *samplers);
        let moved = self.refresh(tables, &maps, listing, function_tables, counts);
        let same = moved == ValueKinds::default()
            && (self.memo.entry()).is_some_and(|kept| kept.settled.sampler_maps == maps);
        if !same {
            return Ok(Across::Refreshed(maps));
        }
        debug_assert!(
            (self.memo.entry())
                .is_some_and(|kept| *kept.settled.planned.shape == self.state.shape()),
            "a parked dispatch memo taken for a state of another shape"
        );
        let device = tables.pipeline.generation();
        self.memo.restamp(tables.table.stamp(), (samplers, device));
        Ok(Across::Taken)
    }

    /// That the state brought in place is the one a dispatch of `tables` makes whole
    /// (`DispatchTables::bind_state` with `maps` applied), field by field (debug builds): the
    /// kernel's table as `ArgumentTable::check_reading` compares it (a float's sign told apart,
    /// every NaN alike), the argument samplers as `samplers` lists them where the dispatch took
    /// the device's (else the kept listing is of every live state at the count that holds), and
    /// every other field equal.
    #[cfg(debug_assertions)]
    fn check_whole(
        &self,
        tables: &DispatchTables<'_>,
        maps: &SamplerMaps,
        samplers: Option<&[crate::sampler::SamplerState]>,
        function_tables: &[VisibleFunctionTable],
    ) {
        let mut whole = tables.bind_state(samplers.unwrap_or_default(), function_tables);
        maps.apply(&mut whole);
        let state = &self.state;
        let kept = &state.stages[&Stage::Kernel];
        let read = (Stage::Kernel, tables.pipeline.interface());
        let views = tables.pipeline.at_bind_views();
        tables.table.check_reading(kept, read, (maps, views));
        const DIFFERS: &str = "a dispatch's bind state brought in place that is not the one made";
        assert!(
            kept.threadgroup_lengths == whole.stages[&Stage::Kernel].threadgroup_lengths,
            "{DIFFERS}: its threadgroup lengths"
        );
        assert!(
            samplers.is_none_or(|samplers| heap_sets::lists(&state.argument_samplers, samplers)),
            "{DIFFERS}: its argument samplers"
        );
        // Every other field equal, the table and the listing compared above, as their values
        // print (a NaN LOD clamp equals none).
        let mut rest = state.clone();
        for of in [&mut whole, &mut rest] {
            of.stages.clear();
            of.argument_samplers.clear();
        }
        assert!(whole == rest, "{DIFFERS}");
    }

    /// The state's shape: the kept one, else the one `found` finds of the state, else made again
    /// (counted in `counts`, found or made).
    pub(in crate::compute) fn shape(
        &mut self,
        counts: Counting<'_>,
        found: impl FnOnce(&BindState) -> Option<Arc<BindShape>>,
    ) -> Arc<BindShape> {
        if let Some(shape) = &self.shape {
            debug_assert!(
                **shape == self.state.shape(),
                "a kept bind shape that is not its state's"
            );
            return Arc::clone(shape);
        }
        let shape = match found(&self.state) {
            Some(shape) => {
                debug_assert!(
                    *shape == self.state.shape(),
                    "a bind shape found that is not its state's"
                );
                counts.dispatch_shape_found();
                shape
            }
            None => {
                counts.dispatch_shape_made();
                Arc::new(self.state.shape())
            }
        };
        self.shape = Some(Arc::clone(&shape));
        shape
    }

    /// Let go of what its last dispatch settled (`DispatchMemo`), which holds the objects it
    /// bound: the state alone, which holds values, is kept for the next encoder.
    pub(in crate::compute) fn let_go(&mut self) {
        self.memo = DispatchMemo::default();
    }

    /// Its state and memo, read, beside the shadows of its resident textures, the last
    /// dispatch's uses, its bind memo and its instantiated commands, kept.
    pub(in crate::compute) fn parts(
        &mut self,
    ) -> (
        &BindState,
        &DispatchMemo,
        &mut ResidentShadows,
        (&mut CommandUses, &mut BindMemo, &mut Instantiated),
    ) {
        let kept = (&mut self.uses, &mut self.bind, &mut self.instantiated);
        (&self.state, &self.memo, &mut self.shadows, kept)
    }
}

/// A device's bind states that ended compute encoders left, which the compute encoders it opens
/// next start from (the module docs): at most [`SPARE_LIVE_DISPATCHES`], the latest given taken
/// first. Each lies in a box an encoder takes and gives back whole, so its open and end allocate
/// none, and move a pointer rather than the state.
#[derive(Default)]
pub(crate) struct SpareLiveDispatches(Mutex<Vec<LiveRoom>>);

/// A kept bind state in the box the next compute encoder takes whole.
struct LiveRoom(Box<LiveDispatch>);

impl SpareLiveDispatches {
    /// The state a compute encoder opening starts from: the latest an encoder gave back, else an
    /// empty one in a box of its own.
    pub(in crate::compute) fn take(&self) -> Box<LiveDispatch> {
        let room = lock(&self.0).pop();
        room.map_or_else(Box::default, |LiveRoom(live)| live)
    }

    /// Keep `live`, an ended compute encoder's state, for the next encoder to start from, its
    /// memo let go ([`LiveDispatch::let_go`]) and its resident textures' lists emptied (its bind
    /// memo kept), where fewer than [`SPARE_LIVE_DISPATCHES`] are kept; else it goes.
    pub(in crate::compute) fn give(&self, mut live: Box<LiveDispatch>) {
        live.let_go();
        live.shadows.let_go();
        live.unreadable.let_go();
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_LIVE_DISPATCHES {
            spare.push(LiveRoom(live));
        }
    }
}

/// What [`LiveDispatch::settled_across`] found of a dispatch.
pub(in crate::compute) enum Across {
    /// The memo's entry serves it, under its table's stamp now.
    Taken,
    /// The entry holds but for the stamp, and the state was brought in place with these maps of
    /// the table's samplers and the device's sampler states untrimmed; a value moved, or the maps
    /// are not the entry's: the dispatch settles from the state as it lies, which a refresh of the
    /// same inputs leaves as it is (`compute::settle_live`).
    Refreshed(SamplerMaps),
    /// The entry does not hold but for the stamp: nothing was done.
    Missed,
}

/// What a compute encoder's end parks in its command buffer for the command buffer's next compute
/// encoder, in place of the device's spares (the module docs, **Parked in its command buffer**):
/// the box of its core, emptied, and its bind state with what its last dispatch settled.
pub(crate) struct Parked {
    room: CoreBox,
    live: Box<LiveDispatch>,
}

impl Parked {
    /// `room` and `live`, an ended compute encoder's: the lists of its resident textures' shadows
    /// and carriers emptied, as the device's spares empty them, and the sampler states its memo's
    /// entry listed let go (the device's live ones, which its command buffer holds none of:
    /// `DispatchMemo::unlist_samplers`).
    pub(in crate::compute) fn new(room: CoreBox, mut live: Box<LiveDispatch>) -> Parked {
        live.shadows.let_go();
        live.unreadable.let_go();
        live.memo.unlist_samplers();
        Parked { room, live }
    }

    /// The core's box and the bind state, for the compute encoder that opens on them.
    pub(in crate::compute) fn into_parts(self) -> (CoreBox, Box<LiveDispatch>) {
        (self.room, self.live)
    }

    /// Give them back to `device`'s spares as their command buffer commits: the box to its
    /// cores', the state to its bind states', its memo let go ([`SpareLiveDispatches::give`]).
    pub(crate) fn give_back(self, device: &Device) {
        EncoderCore::give_room(device, self.room);
        device.shared().live_dispatches.give(self.live);
    }
}
