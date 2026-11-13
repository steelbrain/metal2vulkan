//! Structurization (DTS P0–P9).
//!
//! The structurizer reads an entry's control-flow graph through [`CfgView`] and never touches the
//! AIR instructions: blocks are opaque `Code(BlockId)` units, terminators are [`TermShape`]s, and
//! the only per-block fact it needs besides the edges is whether the block holds a tangled
//! (convergent) instruction. Demotion types come from the Plan.
//!
//! # Passes
//!
//! [`structurize`] runs P0–P7 and returns a [`tree::Structured`]; [`prepare`] runs the CFG-level
//! passes P0–P2 alone and returns a [`Prepared`] working graph:
//!
//! - **P0** ([`prune`]): copy the reachable blocks into a [`cfg::StructCfg`], fold `CondBr(c,t,t)`
//!   to `Br(t)`, group switch literals by target.
//! - **P1** ([`irreducible`]): multi-entry dispatch until the graph is reducible,
//!   refusing tangled irreducible regions (`IrreducibleConvergent`), nesting deeper than
//!   `Limit::IrreducibleDepth`, and output above `Limit::OutputBlocks`.
//! - **P2** ([`loops`]): the loop nest of the reducible graph, its natural loops.
//! - **P3** is retired: no loop-exit tail joins its loop. AGX runs a tail
//!   after the loop it leaves, with every invocation that left through it, and so does P5's exit
//!   dispatch.
//!
//! Then, over [`Prepared`]:
//!
//! - **P4** ([`demote`]): one `Function` variable per block-crossing value and per phi (the
//!   demotion plan; P8 emits the loads and stores).
//! - **P5** ([`normalize`]): canonical loops: back edges become `Continue`, exits
//!   `SetExit; Break` with one exit dispatch per loop.
//! - **P6** ([`regions`]): Ramsey dominator-tree placement of every acyclic region, innermost
//!   loop first, with `Block`/`Jump` for forward jumps.
//! - **P7** ([`pending`]): tail jumps fall out; other jumps go through one `pending` variable.
//!
//! The output size is checked against `Limit::OutputBlocks` (`4(n+e)+1024` for the pruned
//! graph) before anything is emitted. P8/P9 (emit and promote) consume [`tree::Structured`].
//!
//! # Budget
//!
//! Every pass takes the per-attempt [`Meter`]: work is ticked at every input-scaled loop, every
//! input-scaled container is charged before it is reserved, and a pass that fails leaves the
//! ledger as it found it. [`Prepared::release`] returns what the result keeps charged.

// Panic-lint policy for the new core: product code in this tree may not unwrap or index unchecked;
// tests are exempt.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::indexing_slicing))]

pub mod cfg;
pub mod demote;
pub mod emit;
pub mod irreducible;
pub mod loops;
pub mod normalize;
pub mod pending;
pub mod promote;
pub mod prune;
pub mod regions;
// Test support for the crate's tests.
#[cfg(test)]
pub mod testkit;
pub mod tree;

#[cfg(test)]
mod structurizer_properties;

use crate::air::ir::{BlockId, ValueId};
use crate::budget::{output_blocks_cap, Meter};
use crate::error::{Error, Limit, Location};
use crate::graph::Graph;
use crate::spirv::ids::DemoteTy;
use crate::spirv::ir::Unrolling;

/// The shape of a block's terminator. Its targets are the block's [`Graph::successors`], in that
/// order, so the shape carries only what the edges do not:
///
/// - `Br`: one successor.
/// - `CondBr(c)`: two successors, taken when `c` is true, then false.
/// - `Switch(sel, literals)`: the first successor is the default; successor `i + 1` is taken when
///   `sel == literals[i]` (the zero-extended bit pattern, as in `air::ir::InstKind::Switch`).
/// - `Return(v)` and `Unreachable`: no successors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermShape {
    Br,
    CondBr(ValueId),
    Switch(ValueId, Box<[u64]>),
    Return(Option<ValueId>),
    Unreachable,
}

/// What the structurizer reads about an entry's control flow.
///
/// Contract: the [`Graph`] contract, plus: [`CfgView::terminator`] agrees with
/// [`Graph::successors`] as documented on [`TermShape`], and a switch's literals are distinct.
/// A view that breaks it gets `Error::Internal(Invariant)`, never a panic.
pub trait CfgView: Graph<Node = BlockId> {
    /// The terminator shape of block `b`.
    fn terminator(&self, b: BlockId) -> TermShape;
    /// Whether block `b` holds a tangled instruction (subgroup/quad op, derivative, implicit-LOD
    /// sample, barrier).
    fn has_tangled(&self, b: BlockId) -> bool;
    /// The values that cross blocks, with the Plan's demotion type. Read by P4.
    fn cross_block_values(&self) -> &[(ValueId, DemoteTy)];
    /// The phis, each with the demotion type of its incoming variable. Read by P4.
    fn phis(&self) -> &[(ValueId, DemoteTy)];
    /// Where block `b` is, for typed refusals that name a block.
    fn location(&self, b: BlockId) -> Location;
    /// What block `b`'s terminator asks of the driver's unrolling of the loop it closes (the
    /// AIR's `llvm.loop.unroll.*` on a latch): nothing by default.
    fn unrolling(&self, _b: BlockId) -> Unrolling {
        Unrolling::Driver
    }
}

/// The result of P0–P2: a reducible working graph and its loop nest.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub cfg: cfg::StructCfg,
    pub loops: loops::Loops,
    /// `Limit::OutputBlocks` for this function: `4(n+e)+1024` of the pruned graph.
    pub output_cap: u64,
}

impl Prepared {
    /// Return the ledger bytes this result keeps charged.
    pub fn release(self, meter: &mut Meter) {
        meter.release(self.cfg.ledger_bytes());
        meter.release(self.loops.ledger_bytes());
    }
}

/// Run P0–P2 on `view`. On error nothing stays charged to `meter`'s ledger.
pub fn prepare(view: &impl CfgView, meter: &mut Meter) -> Result<Prepared, Error> {
    let mut sc = prune::prune(view, meter)?;
    let output_cap = output_blocks_cap(sc.len() as u64, sc.edge_count() as u64);
    let prepared = irreducible::resolve_capped(&mut sc, view, meter, output_cap)
        .and_then(|()| loops::LoopNest::compute(&sc, meter))
        .and_then(|nest| nest.finish(&sc, meter));
    match prepared {
        Ok(loops) => Ok(Prepared {
            cfg: sc,
            loops,
            output_cap,
        }),
        Err(e) => {
            meter.release(sc.ledger_bytes());
            Err(e)
        }
    }
}

/// Structurize `view` (P0–P7): the structured tree with its variables. On error
/// nothing stays charged to `meter`'s ledger.
pub fn structurize(view: &impl CfgView, meter: &mut Meter) -> Result<tree::Structured, Error> {
    let prepared = prepare(view, meter)?;
    let result = structure_prepared(view, &prepared, meter);
    prepared.release(meter);
    result
}

/// P4–P7 over a prepared graph.
fn structure_prepared(
    view: &impl CfgView,
    prepared: &Prepared,
    meter: &mut Meter,
) -> Result<tree::Structured, Error> {
    let (sc, loops) = (&prepared.cfg, &prepared.loops);
    let charged = loops.loops().len() * std::mem::size_of::<Unrolling>();
    meter.charge(charged)?;
    let result = loop_unrolling(view, sc, loops, meter)
        .and_then(|unrolling| structure_loops(view, prepared, &unrolling, meter));
    meter.release(charged);
    result
}

/// What each loop's `OpLoopMerge` asks of the driver's unrolling, by loop: the request
/// ([`CfgView::unrolling`]) of the first of its own blocks (not a nested loop's) whose original
/// block's terminator makes one, as a latch's `!llvm.loop` does. The caller charges its bytes.
fn loop_unrolling(
    view: &impl CfgView,
    sc: &cfg::StructCfg,
    loops: &loops::Loops,
    meter: &mut Meter,
) -> Result<Vec<Unrolling>, Error> {
    let mut out = Vec::with_capacity(loops.loops().len());
    for l in loops.loops() {
        let mut asked = Unrolling::Driver;
        for node in loops.own_blocks(l) {
            meter.tick(1)?;
            if let cfg::Content::Code(b) = sc.block(node)?.content {
                asked = view.unrolling(b);
                if asked != Unrolling::Driver {
                    break;
                }
            }
        }
        out.push(asked);
    }
    Ok(out)
}

/// P4–P7 over a prepared graph whose loops ask `unrolling`.
fn structure_loops(
    view: &impl CfgView,
    prepared: &Prepared,
    unrolling: &[Unrolling],
    meter: &mut Meter,
) -> Result<tree::Structured, Error> {
    let (sc, loops) = (&prepared.cfg, &prepared.loops);
    let demoted = demote::demote(view, meter)?;
    let first_selector = cfg::to_u32(sc.selector_count())?;
    let lowered =
        normalize::Normalizer::new(loops, sc.len(), first_selector, meter).and_then(|mut norm| {
            let tree = regions::structure(sc, loops, unrolling, &mut norm, meter).and_then(|p6| {
                let p7 = pending::place(&p6.tree, p6.labels, meter);
                meter.release(p6.tree.ledger_bytes());
                p7
            });
            let selectors = norm.selectors();
            norm.release(meter);
            tree.map(|p7| (p7, selectors))
        });
    let (p7, selectors) = match lowered {
        Ok(done) => done,
        Err(e) => {
            meter.release(demoted.ledger_bytes);
            return Err(e);
        }
    };
    let pending = p7.uses_pending.then_some(cfg::SelVar::new(selectors));
    let result = selectors
        .checked_add(u32::from(pending.is_some()))
        .filter(|&s| s != u32::MAX)
        .ok_or_else(|| cfg::invariant("selector count exceeds u32"))
        .and_then(|selectors| {
            meter
                .bound(
                    Limit::OutputBlocks,
                    p7.tree.lowered_blocks(),
                    prepared.output_cap,
                )
                .map(|()| selectors)
                .map_err(Error::from)
        });
    match result {
        Ok(selectors) => Ok(tree::Structured {
            tree: p7.tree,
            demoted: demoted.demoted,
            phis: demoted.phis,
            selectors,
            pending,
            ledger_bytes: demoted.ledger_bytes,
        }),
        Err(e) => {
            meter.release(p7.tree.ledger_bytes());
            meter.release(demoted.ledger_bytes);
            Err(e)
        }
    }
}
