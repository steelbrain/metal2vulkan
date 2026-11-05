//! A draw's heap part, kept by its render encoder's bind state from one draw to the next while
//! what it is made of holds (`live_bind.rs`): its plan's heap tables
//! (`BindingPlan::heap_tables_into`), where its heap writes go (`BindingPlan::heap_elements`),
//! and the plan's writes a draw makes past the heap's entries once a draw checked each entry
//! (`binding::PastHeap`). Each grows with the textures the encoder made resident and those its
//! slots bind that a handle names, so a draw that worked them out paid that count at each draw,
//! whatever it bound; a draw that keeps them costs what it binds
//! (`WorkCounts::draw_heap_parts_made`).
//!
//! - **What it is made of.** The plan (held weakly, compared by address), the capacity of its
//!   variant's heap arrays, and the state's heap inputs: its used resources, the residency's
//!   listing, named by the residency's generation (`residency.rs`; a texture's identity, facts
//!   and resource ID, which the heap part reads of it, no call changes); its argument samplers,
//!   the device's live sampler states at the count of their slots' changes the state listed them
//!   at (`LiveBind::listed_at`), else compared state by state; its function tables, compared;
//!   what it binds at the plan's heap slots (`BindingPlan::heap_slots`), compared; and its heap
//!   placement and order of the used resources. The tables and the elements are functions of
//!   those alone (`BindingPlan::heap_tables_into`). An entry's check is a function of its texture
//!   and the plan's view: a texture's image, record and decoded twin are fixed at its creation,
//!   and a texture a slot binds, retained by its command buffer, keeps its resource ID meanwhile.
//! - **Several, one a plan.** It keeps the parts of the last [`KEPT_PARTS`] plans its draws
//!   drew through, the one drawn through longest ago given up first: draws that switch between
//!   pipeline states, or whose primitive types take plans of their own, each keep their own.
//! - **In kept recordings.** A recording of an execution's draws kept for a later host step
//!   (`render/executions/kept.rs`) holds the tables and writes of its draws' parts: its key holds
//!   what those parts are made of that may differ from step to step, the pipeline state (the
//!   plan, the capacity, the function tables), the resources made resident (the used resources
//!   and their order) and the count of the sampler IDs' slots' changes (the argument samplers); a
//!   host step's tables bind no texture, so no heap slot of its draws binds one, and a replayed
//!   draw whose tables bind one there is not kept ([`KeptHeap::binds_slot_textures`]).
//! - **Checked in builds with debug assertions.** A draw that keeps one works the tables and
//!   elements out again and compares them, and lists its writes past the heap's entries again
//!   and compares them (`binding::bind_into`).
//! - **Let go as the encoder ends** (`SpareLiveBinds::give`): the next encoder works out its own
//!   at its first draw, into the storage of those let go of, which hold nothing else (no texture,
//!   sampler or plan: what a part holds is plain data, and its plan is held weakly and let go), so
//!   a part worked out allocates no more than a draw that worked out its heap part did
//!   (`runtime_lane::frame_costs`). The encoders of the chunks of one execution recorded apart
//!   keep them from one chunk to the next (`LiveBind::end_chunk`), as one encoder keeps them
//!   from one draw to the next: a host step that records an execution's chunks apart works out
//!   each part once, as one that records them inline does.

use std::sync::{Arc, Weak};

use metal2vulkan::bind::{
    ArgumentFunctionTable, ArgumentSampler, BindState, BindingPlan, BoundTexture, DescriptorBytes,
    HeapElements, HeapPlacement,
};
use metal2vulkan::pipeline::{PipelineLayout, SlotRef};

use crate::binding::PastHeap;
use crate::error::{internal, Error};
use crate::heap_sets;
use crate::work_counts::Counters;

/// The most heap parts a [`KeptHeap`] keeps: one for each of the plans of a few pipeline states
/// drawn in turn, or of one state's draws of each primitive type (Metal's five,
/// `MTLPrimitiveType`).
const KEPT_PARTS: usize = 8;

/// The heap parts a render encoder's draws worked out lately (the module docs), the latest
/// drawn through last.
#[derive(Default)]
pub(crate) struct KeptHeap {
    parts: Vec<HeapPart>,
    /// The parts let go of as an encoder ended, emptied, the storage of those worked out later.
    spare: Vec<HeapPart>,
    /// The heap tables a kept part's check works out, in builds with debug assertions: kept, so
    /// the check allocates no more than a draw that worked them out did (`runtime_lane::
    /// frame_costs`).
    #[cfg(debug_assertions)]
    tables: Vec<DescriptorBytes>,
}

/// One draw's heap part, and what it is made of.
#[derive(Default)]
struct HeapPart {
    of: HeapOf,
    tables: Vec<DescriptorBytes>,
    /// Where its heap writes go, where `heap`: kept allocated whatever plan the part is worked
    /// out for next, so a part worked out again allocates none of its own.
    elements: Arc<HeapElements>,
    /// Whether its layout has a heap, so its draws write `elements`.
    heap: bool,
    past: PastHeap,
}

/// What a [`HeapPart`] is made of (the module docs).
#[derive(Default)]
struct HeapOf {
    plan: Weak<BindingPlan>,
    capacity: Option<u32>,
    generation: u64,
    listed_at: Option<u64>,
    samplers: Vec<ArgumentSampler>,
    function_tables: Vec<ArgumentFunctionTable>,
    slots: Vec<(SlotRef, Option<BoundTexture>)>,
    placement: HeapPlacement,
    in_token_order: bool,
}

/// What a draw's heap part is worked out from: its plan, its variant's layout, the generation of
/// its encoder's residency, the count of the device's sampler IDs' slots' changes its state listed
/// its argument samplers at (`LiveBind::listed_at`), and its state.
pub(crate) struct HeapInputs<'a> {
    pub(crate) plan: &'a Arc<BindingPlan>,
    pub(crate) layout: &'a PipelineLayout,
    pub(crate) generation: u64,
    pub(crate) listed_at: Option<u64>,
    pub(crate) state: &'a BindState,
}

/// What `state` binds at texture slot `slot`.
fn bound_at(state: &BindState, slot: SlotRef) -> Option<BoundTexture> {
    let table = state.stages.get(&slot.stage)?;
    table.textures.get(&slot.index).copied()
}

impl HeapOf {
    /// Make it what a part worked out from `inputs` is made of, in its own storage.
    fn fill(&mut self, inputs: &HeapInputs<'_>, capacity: Option<u32>) {
        let state = inputs.state;
        self.plan = Arc::downgrade(inputs.plan);
        self.capacity = capacity;
        self.generation = inputs.generation;
        self.listed_at = inputs.listed_at;
        self.samplers.clone_from(&state.argument_samplers);
        self.function_tables.clone_from(&state.function_tables);
        self.slots.clear();
        let slots = inputs.plan.heap_slots();
        (self.slots).extend(slots.map(|slot| (slot, bound_at(state, slot))));
        self.placement = state.heap_placement;
        self.in_token_order = state.used_in_token_order;
    }

    /// Whether a part made of it is the one `inputs` work out, the layout's heap arrays holding
    /// `capacity` elements.
    fn holds(&self, inputs: &HeapInputs<'_>, capacity: Option<u32>) -> bool {
        let state = inputs.state;
        let samplers = match (self.listed_at, inputs.listed_at) {
            (Some(was), Some(now)) if was == now => true,
            _ => heap_sets::same_listing(&self.samplers, &state.argument_samplers),
        };
        self.plan.as_ptr() == Arc::as_ptr(inputs.plan)
            && self.capacity == capacity
            && self.generation == inputs.generation
            && self.placement == state.heap_placement
            && self.in_token_order == state.used_in_token_order
            && samplers
            && self.function_tables == state.function_tables
            && (self.slots.iter()).all(|&(slot, bound)| bound_at(state, slot) == bound)
    }
}

impl KeptHeap {
    /// Bring in the heap part of a draw of `inputs`: the one kept for them, else one worked out
    /// (counted in `counts`, where the plan has a heap part) and kept in place of the one drawn
    /// through longest ago. Where its heap writes go: `None` for a layout without a heap.
    pub(crate) fn bring_in(
        &mut self,
        counts: &Counters,
        inputs: HeapInputs<'_>,
    ) -> Result<Option<Arc<HeapElements>>, Error> {
        let capacity = heap_sets::capacity(inputs.layout);
        let kept = self
            .parts
            .iter()
            .position(|part| part.of.holds(&inputs, capacity));
        let part = match kept {
            Some(at) => {
                let part = self.parts.remove(at);
                #[cfg(debug_assertions)]
                part.check(&inputs, &mut self.tables)?;
                part
            }
            None => {
                // Worked out into the storage of the one drawn through longest ago, where it keeps
                // as many as it may, else of one let go of.
                let mut part = match self.parts.len() == KEPT_PARTS {
                    true => self.parts.remove(0),
                    false => self.spare.pop().unwrap_or_default(),
                };
                part.fill(&inputs, capacity)?;
                if !part.tables.is_empty() || part.heap {
                    counts.draw_heap_part_made();
                }
                part
            }
        };
        let elements = part.heap.then(|| Arc::clone(&part.elements));
        self.parts.push(part);
        Ok(elements)
    }

    /// Whether the part the last draw brought in reads a texture its tables bind at a heap slot
    /// (`BindingPlan::heap_slots`).
    pub(crate) fn binds_slot_textures(&self) -> bool {
        (self.parts.last())
            .is_some_and(|part| part.of.slots.iter().any(|(_, bound)| bound.is_some()))
    }

    /// The heap tables of the part the last draw brought in, and the writes past its heap's
    /// entries listed of it.
    pub(crate) fn brought_in(&mut self) -> Result<(&[DescriptorBytes], &mut PastHeap), Error> {
        let part =
            (self.parts.last_mut()).ok_or_else(|| internal("a draw's heap part not brought in"))?;
        Ok((&part.tables, &mut part.past))
    }

    /// Let go of every part, as the encoder ends, keeping its storage (the module docs).
    pub(crate) fn let_go(&mut self) {
        let spare = &mut self.spare;
        for mut part in self.parts.drain(..) {
            part.empty();
            spare.push(part);
        }
    }
}

impl HeapPart {
    /// Make it the part `inputs` work out, in its own storage.
    fn fill(&mut self, inputs: &HeapInputs<'_>, capacity: Option<u32>) -> Result<(), Error> {
        let (plan, state) = (inputs.plan, inputs.state);
        plan.heap_tables_into(state, &mut self.tables)
            .map_err(Error::Bind)?;
        let elements = heap_sets::heap_elements(inputs.layout, plan, state)?;
        // Into the elements' own allocation where no draw holds them still.
        self.heap = elements.is_some();
        if let Some(made) = elements {
            match Arc::get_mut(&mut self.elements) {
                Some(held) => *held = made,
                None => self.elements = Arc::new(made),
            }
        }
        self.of.fill(inputs, capacity);
        self.past.forget();
        Ok(())
    }

    /// Hold nothing but its storage: a part let go of (the module docs).
    fn empty(&mut self) {
        self.of.plan = Weak::new();
        self.tables.clear();
        self.past.forget();
    }

    /// Check that it is the part `inputs` work out (the module docs), its tables worked out
    /// into `tables`.
    #[cfg(debug_assertions)]
    fn check(
        &self,
        inputs: &HeapInputs<'_>,
        tables: &mut Vec<DescriptorBytes>,
    ) -> Result<(), Error> {
        let (plan, state) = (inputs.plan, inputs.state);
        plan.heap_tables_into(state, tables).map_err(Error::Bind)?;
        let elements = heap_sets::heap_elements(inputs.layout, plan, state)?;
        assert!(
            *tables == self.tables && elements.as_ref() == self.heap.then_some(&*self.elements),
            "a draw's kept heap part that is not its own"
        );
        KEPT_HEAP_CHECKS.with(|checks| checks.set(checks.get() + 1));
        Ok(())
    }
}

#[cfg(debug_assertions)]
thread_local! {
    /// The draws whose kept heap part this thread checked against one worked out whole
    /// ([`HeapPart::check`]), in builds with debug assertions.
    static KEPT_HEAP_CHECKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many draws' kept heap parts this thread checked so far ([`KeptHeap::bring_in`]).
#[cfg(all(test, debug_assertions, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn kept_heap_checks() -> u64 {
    KEPT_HEAP_CHECKS.with(std::cell::Cell::get)
}
