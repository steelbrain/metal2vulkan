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
//! - **Checked in builds with debug assertions.** A draw that keeps one works the tables and
//!   elements out again and compares them, and lists its writes past the heap's entries again
//!   and compares them (`binding::bind_into`).
//! - **Let go as the encoder ends** (`SpareLiveBinds::give`), with what it holds: the next encoder
//!   works out its own at its first draw.

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
pub(crate) struct KeptHeap(Vec<HeapPart>);

/// One draw's heap part, and what it is made of.
struct HeapPart {
    of: HeapOf,
    tables: Vec<DescriptorBytes>,
    elements: Option<Arc<HeapElements>>,
    past: PastHeap,
}

/// What a [`HeapPart`] is made of (the module docs).
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
    /// What a part worked out from `inputs` is made of.
    fn of(inputs: &HeapInputs<'_>, capacity: Option<u32>) -> HeapOf {
        let state = inputs.state;
        let slots = inputs.plan.heap_slots();
        HeapOf {
            plan: Arc::downgrade(inputs.plan),
            capacity,
            generation: inputs.generation,
            listed_at: inputs.listed_at,
            samplers: state.argument_samplers.clone(),
            function_tables: state.function_tables.clone(),
            slots: slots.map(|slot| (slot, bound_at(state, slot))).collect(),
            placement: state.heap_placement,
            in_token_order: state.used_in_token_order,
        }
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
            .0
            .iter()
            .position(|part| part.of.holds(&inputs, capacity));
        let part = match kept {
            Some(at) => {
                let part = self.0.remove(at);
                #[cfg(debug_assertions)]
                part.check(&inputs)?;
                part
            }
            None => {
                let mut tables = Vec::new();
                let (plan, state) = (inputs.plan, inputs.state);
                plan.heap_tables_into(state, &mut tables)
                    .map_err(Error::Bind)?;
                let elements = heap_sets::heap_elements(inputs.layout, plan, state)?;
                if !tables.is_empty() || elements.is_some() {
                    counts.draw_heap_part_made();
                }
                if self.0.len() == KEPT_PARTS {
                    self.0.remove(0);
                }
                HeapPart {
                    of: HeapOf::of(&inputs, capacity),
                    tables,
                    elements: elements.map(Arc::new),
                    past: PastHeap::default(),
                }
            }
        };
        let elements = part.elements.clone();
        self.0.push(part);
        Ok(elements)
    }

    /// The heap tables of the part the last draw brought in, and the writes past its heap's
    /// entries listed of it.
    pub(crate) fn brought_in(&mut self) -> Result<(&[DescriptorBytes], &mut PastHeap), Error> {
        let part =
            (self.0.last_mut()).ok_or_else(|| internal("a draw's heap part not brought in"))?;
        Ok((&part.tables, &mut part.past))
    }

    /// Let go of every part, as the encoder ends.
    pub(crate) fn let_go(&mut self) {
        self.0.clear();
    }
}

impl HeapPart {
    /// Check that it is the part `inputs` work out (the module docs).
    #[cfg(debug_assertions)]
    fn check(&self, inputs: &HeapInputs<'_>) -> Result<(), Error> {
        let (plan, state) = (inputs.plan, inputs.state);
        let mut tables = Vec::new();
        plan.heap_tables_into(state, &mut tables)
            .map_err(Error::Bind)?;
        let elements = heap_sets::heap_elements(inputs.layout, plan, state)?;
        assert!(
            tables == self.tables && elements.as_ref() == self.elements.as_deref(),
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
