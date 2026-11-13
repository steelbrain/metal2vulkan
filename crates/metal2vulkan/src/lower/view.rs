//! The structurizer's view of a specialized entry and its plan.
//!
//! The graph is the entry body's CFG; terminator shapes come from the AIR terminators; the
//! block-crossing values, phis, and tangled instructions come from the `Plan`, so their demotion
//! types are the lowered representations.

use crate::air::error::Location;
use crate::air::ir::{BlockId, Body, InstKind, Unroll, ValueId};
use crate::air::Entry;
use crate::graph::Graph;
use crate::lower::select::Plan;
use crate::spirv::ids::DemoteTy;
use crate::spirv::ir::Unrolling;
use crate::structurize::{CfgView, TermShape};

/// An entry body plus its plan, as the structurizer's `CfgView`.
pub struct EntryView<'a> {
    entry: &'a Entry,
    body: &'a Body,
    plan: &'a Plan,
}

impl<'a> EntryView<'a> {
    /// The view of `body` (the entry function's) under `plan`.
    pub fn new(entry: &'a Entry, body: &'a Body, plan: &'a Plan) -> EntryView<'a> {
        EntryView { entry, body, plan }
    }

    fn id(&self, v: crate::air::ir::Value) -> ValueId {
        // A value outside the index names no plan entry; `u32::MAX` makes the structurizer's
        // consumer (the emitter) report it as an invariant failure instead of panicking.
        self.plan.values.id(v).unwrap_or(ValueId::new(u32::MAX))
    }
}

impl Graph for EntryView<'_> {
    type Node = BlockId;

    fn node_count(&self) -> usize {
        self.body.node_count()
    }

    fn entry(&self) -> BlockId {
        self.body.entry()
    }

    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.body.successors(node)
    }
}

impl CfgView for EntryView<'_> {
    fn terminator(&self, b: BlockId) -> TermShape {
        match self.body.terminator(b).map(|t| &t.kind) {
            Some(InstKind::Br { .. }) => TermShape::Br,
            Some(InstKind::CondBr { c, .. }) => TermShape::CondBr(self.id(*c)),
            Some(InstKind::Switch { v, cases, .. }) => {
                TermShape::Switch(self.id(*v), cases.iter().map(|(l, _)| *l).collect())
            }
            Some(InstKind::Ret(v)) => TermShape::Return(v.map(|v| self.id(v))),
            // A verified body ends every block in a terminator; anything else is unreachable.
            _ => TermShape::Unreachable,
        }
    }

    fn has_tangled(&self, b: BlockId) -> bool {
        self.body.blocks.get(b).is_some_and(|block| {
            block.inst_ids().any(|i| {
                self.plan
                    .values
                    .inst(i)
                    .is_some_and(|v| self.plan.tangled.contains(&v))
            })
        })
    }

    fn cross_block_values(&self) -> &[(ValueId, DemoteTy)] {
        &self.plan.cross
    }

    fn phis(&self) -> &[(ValueId, DemoteTy)] {
        &self.plan.phis
    }

    fn location(&self, b: BlockId) -> Location {
        Location::function(self.entry.ep.name.clone(), Some(b.raw()), None)
    }

    fn unrolling(&self, b: BlockId) -> Unrolling {
        let hints = self.body.terminator(b).and_then(|t| t.kind.loop_hints());
        match hints.map(|h| h.unroll) {
            Some(Unroll::Full | Unroll::Enable) => Unrolling::Unroll,
            Some(Unroll::Count(n)) if n > 0 => Unrolling::PartialCount(n),
            Some(Unroll::Disable) => Unrolling::DontUnroll,
            _ => Unrolling::Driver,
        }
    }
}
