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
//!   reads, is stored uncompared. The new value is always stored, so the state is the dispatch's
//!   exactly. While the same slots stay bound, nothing is allocated.
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
//! - **Kept from one encoder to the next.** An encoder that ends gives its state, with its shape,
//!   to its device ([`SpareLiveDispatches`]), and the next compute encoder opened starts from it,
//!   as a dispatch starts from the one before it: exact for the reason a dispatch's is, every field
//!   of the state a dispatch binds being derived again at each dispatch that settles. The memo of
//!   what its last dispatch settled goes first ([`LiveDispatch::let_go`]): it holds the pipeline
//!   state, sampler states and pipelines, where the state holds values alone (its bindings, its used
//!   resources and argument samplers as a bind state lists them, its table's pipeline state held
//!   weakly), so a kept one holds nothing alive.

use metal2vulkan::bind::{BindShape, BindState, HeapPlacement, ImageblockSize};
use metal2vulkan::pipeline::Stage;
use std::sync::{Arc, Mutex};

use super::{DispatchMemo, DispatchTables};
use crate::address_map::SamplerMaps;
use crate::argument_table::Stamp;
use crate::device::lock;
use crate::function_table::VisibleFunctionTable;
use crate::heap_sets;
use crate::live_bind::Reshape;
use crate::pipeline::{ComputePipelineState, WeakComputePipelineState};
use crate::sampler::SamplerState;
use crate::work_counts::Counters;

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
    /// The count of the device's sampler IDs' slots' changes (`Device::sampler_slots_changed`)
    /// at which the state's argument samplers are every live sampler state, `None` where they are
    /// not (none listed yet, or those a variant's set-1 arrays hold alone).
    listed_at: Option<u64>,
    /// What the state's kernel table was read from ([`ReadFrom`]).
    read_from: ReadFrom,
    /// What the dispatch that last brought the state in place settled, where it is kept for the
    /// next dispatch of the same inputs (`DispatchMemo`): of this state, so a dispatch that takes
    /// it binds this state.
    pub(in crate::compute) memo: DispatchMemo,
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
    /// resident, `samplers` listed (the device's sampler states at `changed` changes of their
    /// slots, those a variant's set-1 arrays hold alone where `trimmed`), and the live tables the
    /// pipeline state made. Each change that may change the shape is noted, and the shape kept only
    /// where none may have.
    pub(in crate::compute) fn refresh(
        &mut self,
        tables: &DispatchTables<'_>,
        maps: &SamplerMaps,
        (changed, samplers, trimmed): (u64, &[SamplerState], bool),
        function_tables: &[VisibleFunctionTable],
        counts: &Counters,
    ) {
        let mut reshape = Reshape::default();
        let pipeline = tables.pipeline;
        let views = pipeline.at_bind_views();
        let state = &mut self.state;
        reshape.field(&mut state.view_base_alignment, views.alignment());
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
        let views_held = views.holds(Stage::Kernel, kept.at_bind_views);
        let read_under = self.read_from.table.filter(|_| same && views_held);
        if read_under == Some(table.stamp()) {
            #[cfg(debug_assertions)]
            table.check_reading(kept, read, (maps, views));
        } else {
            // Only the slots calls changed since it was read, where the table keeps them.
            let in_part = read_under.is_some_and(|stamp| {
                table.refresh_changed((&mut *kept, stamp), read, (maps, views), &mut reshape)
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
        state.stage_in_region = tables.stage_in_region;
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
        // listing kept is of every live state at the count of their slots' changes they were taken
        // at; else they are compared state by state.
        if trimmed || self.listed_at != Some(changed) {
            if !heap_sets::lists(&state.argument_samplers, samplers) {
                reshape.field(&mut state.argument_samplers, heap_sets::listed(samplers));
            }
            self.listed_at = (!trimmed).then_some(changed);
        }
        let listed = function_tables
            .iter()
            .map(VisibleFunctionTable::listed)
            .collect();
        reshape.field(&mut state.function_tables, listed);
        // A value no shape reads (`BindState::heap_placement`).
        state.heap_placement = HeapPlacement::ResourceId;
        #[cfg(debug_assertions)]
        self.check_whole(tables, maps, samplers, function_tables);
        if reshape.may_reshape() {
            self.shape = None;
        }
    }

    /// That the state brought in place is the one a dispatch of `tables` makes whole
    /// (`DispatchTables::bind_state` with `maps` applied), field by field (debug builds): the
    /// kernel's table as `ArgumentTable::check_reading` compares it (a float's sign told apart,
    /// every NaN alike), and every other field equal.
    #[cfg(debug_assertions)]
    fn check_whole(
        &self,
        tables: &DispatchTables<'_>,
        maps: &SamplerMaps,
        samplers: &[SamplerState],
        function_tables: &[VisibleFunctionTable],
    ) {
        let mut whole = tables.bind_state(samplers, function_tables);
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
            heap_sets::lists(&state.argument_samplers, samplers),
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
        counts: &Counters,
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
}

/// A device's bind states that ended compute encoders left, which the compute encoders it opens
/// next start from (the module docs): at most [`SPARE_LIVE_DISPATCHES`], the latest given taken
/// first. An encoder holds its own in place, so its open and end allocate none.
#[derive(Default)]
pub(crate) struct SpareLiveDispatches(Mutex<Vec<LiveDispatch>>);

impl SpareLiveDispatches {
    /// The state a compute encoder opening starts from: the latest an encoder gave back, else an
    /// empty one.
    pub(in crate::compute) fn take(&self) -> LiveDispatch {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep `live`, an ended compute encoder's state, for the next encoder to start from, its
    /// memo let go ([`LiveDispatch::let_go`]), where fewer than [`SPARE_LIVE_DISPATCHES`] are
    /// kept; else it goes.
    pub(in crate::compute) fn give(&self, mut live: LiveDispatch) {
        live.let_go();
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_LIVE_DISPATCHES {
            spare.push(live);
        }
    }
}
