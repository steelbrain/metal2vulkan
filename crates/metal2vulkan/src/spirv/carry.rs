//! Stores of an undefined value at the top of each loop trip, which end the structurizer variables
//! whose values no later trip reads.
//!
//! P4 gives every block-crossing value and every phi of the source a `Function` variable
//! (`structurize::demote`): a demoted value's variable is stored where the value is defined, a
//! phi's at the end of each block that enters the phi. P5 to P7 add exit dispatches and pending
//! guards, whose paths lead from a loop's back edge to a read of such a variable (after the loop,
//! or later in its body) without passing its store. No execution takes such a path to read the
//! value an earlier trip left, but a driver's compiler cannot know that: it keeps the variable's
//! value around the back edge, a phi at the loop's header live through the whole loop.
//!
//! Where a loop's body holds every store of such a variable, no trip reads the value an earlier
//! trip left in it. In the source, the value's definition dominates each of its uses and the
//! loop's header dominates the definition, so every path from the header to a use passes the
//! definition again; a phi's variable is read only at its block, which is entered from one of the
//! blocks that store it. [`end_dead_carries`] stores an undefined value into the variable at the
//! top of each trip, so the driver keeps nothing of it around the back edge.
//!
//! The variables it ends are the ones P8 marks [`LocalVar::trip_local`]: the demoted values' and
//! the phis' of a walk that is not predicated. The predicated walk runs a loop while any
//! invocation of the workgroup is live in it, so an invocation that has left the loop still runs
//! the next trip's top, where such a store would end the value it reads after the loop.
//!
//! A store is written for a marked variable `v` of a function and a loop of it when all of the
//! following hold, read from the function's blocks, their merges and terminators, and the accesses
//! of `v`:
//!
//! - **`v` is written and read whole.** Every use of `v`'s pointer is a load through it or a store
//!   through it: no access chain is made from it, and no other instruction, phi or terminator
//!   takes it (a call, an atomic, a copy, a store of the pointer itself). A variable written by
//!   parts (an arena, an array written element by element, what a lowering keeps per thread
//!   across a barrier) holds what earlier trips wrote into it, which such a store would wipe; it
//!   is left as it is, marked or not.
//! - **The loop's body holds every whole store of `v`.** The body is the header and the blocks
//!   that reach its continue target without passing the header. A whole store is an `OpStore`
//!   through `v`'s own pointer, or `v`'s initializer, which counts as a store in the function's
//!   first block.
//! - **`v` is live into the body's first block**, the block the header branches to (the header
//!   holds only its merge and its branch): a load through `v` is reachable from the block's top
//!   before a whole store.
//! - **The first block does not read `v` before it stores it.**
//!
//! The stores go at the top of the first block: variable after variable in the function's order
//! of variables, and each variable's loops in the function's block order. In one block, a store
//! written later stands before the earlier ones.

use std::collections::BTreeMap;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ids::{BlockId, FuncId, TypeId, Value};
use crate::spirv::ir::{
    Block, Function, Inst, InstNode, LocalVar, MemAccess, Merge, Terminator, TypedModule, ValueDef,
    ValueOrigin,
};

/// No block, or no variable: the sentinel of the per-block tables.
const NONE: u32 = u32::MAX;

/// The bytes one entry of a `BTreeMap<Value, u32>` is charged.
const MAP_ENTRY: usize = 32;

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Write the stores of the module docs into every function of `m` that has a marked variable.
/// Returns the stores written.
pub fn end_dead_carries(m: &mut TypedModule, meter: &mut Meter) -> Result<u32, Error> {
    let mut written = 0u32;
    for f in 0..m.functions.len() {
        let f = FuncId::new(u32::try_from(f).map_err(|_| invariant("function ids"))?);
        let ends = ends(m, f, meter)?;
        // The module keeps what it gains: two instructions and a value for each store.
        let bytes = ends
            .len()
            .saturating_mul(2 * std::mem::size_of::<InstNode>() + std::mem::size_of::<ValueDef>());
        meter.charge(bytes)?;
        for (ptr, ty, first) in ends {
            let undef =
                Value::new(u32::try_from(m.values.len()).map_err(|_| invariant("value ids"))?);
            m.values.push(ValueDef {
                ty,
                origin: ValueOrigin::Inst(first),
            });
            let block = m
                .blocks
                .get_mut(first.index())
                .ok_or_else(|| invariant("a loop's first block"))?;
            block.insts.insert(
                0,
                InstNode {
                    result: None,
                    inst: Inst::Store {
                        ptr,
                        value: undef,
                        access: MemAccess::default(),
                    },
                },
            );
            block.insts.insert(
                0,
                InstNode {
                    result: Some(undef),
                    inst: Inst::Undef,
                },
            );
            written = written.saturating_add(1);
        }
    }
    Ok(written)
}

/// The number of (variable, loop) pairs of `m` that the rule of the module docs selects: each a
/// marked variable whose value the driver would keep around the loop's back edge though no trip
/// reads it. None is left once [`end_dead_carries`] has run.
#[cfg(test)]
pub(crate) fn dead_carries(m: &TypedModule) -> usize {
    let mut meter = Meter::new(&crate::budget::Budget::standard());
    (0..m.functions.len())
        .map(|f| {
            let f = FuncId::new(u32::try_from(f).expect("function ids"));
            ends(m, f, &mut meter).expect("dead carries").len()
        })
        .sum()
}

/// The stores function `f` of `m` takes, in the order the module docs give: for each, the
/// variable's pointer, its pointee and the loop's first block.
fn ends(
    m: &TypedModule,
    f: FuncId,
    meter: &mut Meter,
) -> Result<Vec<(Value, TypeId, BlockId)>, Error> {
    let func = m
        .functions
        .get(f.index())
        .ok_or_else(|| invariant("function"))?;
    let marked: Vec<&LocalVar> = func
        .vars
        .iter()
        .filter_map(|v| m.vars.get(v.index()))
        .filter(|v| v.trip_local)
        .collect();
    if marked.is_empty() {
        return Ok(Vec::new());
    }
    let mut held = 0usize;
    let r = select(m, func, &marked, meter, &mut held);
    meter.release(held);
    r
}

/// Charge `bytes` to `meter`, held until [`ends`] returns.
fn hold(meter: &mut Meter, held: &mut usize, bytes: usize) -> Result<(), Error> {
    meter.charge(bytes)?;
    *held = held.saturating_add(bytes);
    Ok(())
}

#[cfg(test)]
mod tests;

fn tick(meter: &mut Meter, units: usize) -> Result<(), Error> {
    meter.tick(u32::try_from(units).unwrap_or(u32::MAX))?;
    Ok(())
}

/// What one marked variable does in one block: its first access there is a read (else a whole
/// store).
#[derive(Clone, Copy)]
struct Event {
    block: u32,
    read_first: bool,
}

/// One marked variable's accesses, in block order.
#[derive(Default)]
struct Accesses {
    /// Each block that reads or wholly stores the variable, once.
    events: Vec<Event>,
    /// Each block that wholly stores it, once.
    stores: Vec<u32>,
}

impl Accesses {
    fn note(&mut self, block: u32, read: bool) {
        if self.events.last().map(|e| e.block) != Some(block) {
            self.events.push(Event {
                block,
                read_first: read,
            });
        }
        if !read && self.stores.last() != Some(&block) {
            self.stores.push(block);
        }
    }
}

fn block_of<'m>(m: &'m TypedModule, func: &Function, i: usize) -> Result<&'m Block, Error> {
    func.blocks
        .get(i)
        .and_then(|b| m.blocks.get(b.index()))
        .ok_or_else(|| invariant("block"))
}

/// [`ends`]'s selection over `func`'s blocks for its marked variables `marked`.
fn select(
    m: &TypedModule,
    func: &Function,
    marked: &[&LocalVar],
    meter: &mut Meter,
    held: &mut usize,
) -> Result<Vec<(Value, TypeId, BlockId)>, Error> {
    let n = func.blocks.len();
    if u32::try_from(n).is_err() {
        return Err(invariant("block count"));
    }
    let u32_of = |i: usize| u32::try_from(i).unwrap_or(NONE);
    // What the tables below hold, charged before they are built.
    let (mut edges, mut insts) = (0usize, 0usize);
    for i in 0..n {
        let b = block_of(m, func, i)?;
        b.term.for_each_target(|_| edges += 1);
        insts += b.insts.len();
    }
    tick(meter, n.saturating_add(insts))?;
    let vec = std::mem::size_of::<Vec<u32>>();
    let bytes = m
        .blocks
        .len()
        .saturating_mul(4)
        .saturating_add(n.saturating_mul(1 + 3 * 4 + 2 * vec))
        .saturating_add(edges.saturating_mul(2 * 4))
        .saturating_add(insts.saturating_mul(std::mem::size_of::<Event>() + 4))
        .saturating_add(marked.len().saturating_mul(MAP_ENTRY + 2 * vec + 1));
    hold(meter, held, bytes)?;

    // Local block numbers, in layout order.
    let mut local = vec![NONE; m.blocks.len()];
    for (i, b) in func.blocks.iter().enumerate() {
        if let Some(x) = local.get_mut(b.index()) {
            *x = u32_of(i);
        }
    }
    let at = |b: BlockId| {
        local
            .get(b.index())
            .copied()
            .filter(|&x| x != NONE)
            .map(|x| x as usize)
    };
    // The blocks reachable from the first, and each one's reachable predecessors.
    let mut reach = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    if n > 0 {
        stack.push(0);
    }
    while let Some(i) = stack.pop() {
        match reach.get_mut(i) {
            Some(r) if !*r => *r = true,
            _ => continue,
        }
        block_of(m, func, i)?.term.for_each_target(|t| {
            if let Some(j) = at(t) {
                stack.push(j);
            }
        });
    }
    let reached = |i: usize| reach.get(i).copied().unwrap_or(false);
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in (0..n).filter(|&i| reached(i)) {
        block_of(m, func, i)?.term.for_each_target(|t| {
            if let Some(p) = at(t).and_then(|j| preds.get_mut(j)) {
                p.push(u32_of(i));
            }
        });
    }
    let preds_of = |i: usize| preds.get(i).map(Vec::as_slice).unwrap_or_default();

    // The loops, by header in layout order: each one's first block and the loops each block is in.
    let mut firsts: Vec<usize> = Vec::new();
    let mut loops_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen = vec![NONE; n];
    for i in (0..n).filter(|&i| reached(i)) {
        let b = block_of(m, func, i)?;
        let Some(Merge::Loop { cont, .. }) = b.merge else {
            continue;
        };
        let (Some(c), Terminator::Branch(first)) = (at(cont), &b.term) else {
            continue;
        };
        let Some(first) = at(*first) else {
            continue;
        };
        if !reached(c) {
            continue;
        }
        let l = u32_of(firsts.len());
        let mut body = vec![i];
        if let Some(s) = seen.get_mut(i) {
            *s = l;
        }
        stack.clear();
        stack.push(c);
        while let Some(x) = stack.pop() {
            match seen.get_mut(x) {
                Some(s) if *s != l => *s = l,
                _ => continue,
            }
            body.push(x);
            let p = preds_of(x);
            tick(meter, 1 + p.len())?;
            stack.extend(p.iter().map(|&p| p as usize));
        }
        hold(meter, held, body.len().saturating_mul(4))?;
        for x in body {
            if let Some(of) = loops_of.get_mut(x) {
                of.push(l);
            }
        }
        firsts.push(first);
    }
    if firsts.is_empty() {
        return Ok(Vec::new());
    }
    let in_loop = |b: u32, l: u32| {
        loops_of
            .get(b as usize)
            .is_some_and(|of| of.binary_search(&l).is_ok())
    };

    // The marked variables' pointers, and the variables some use of whose pointer is neither a
    // load nor a store through it (written or read by parts, or the pointer passed on).
    let own: BTreeMap<Value, u32> = marked
        .iter()
        .enumerate()
        .map(|(k, v)| (v.value, u32_of(k)))
        .collect();
    let mut parts = vec![false; marked.len()];
    let mut by_parts = |v: Value| {
        if let Some(x) = own.get(&v).and_then(|&k| parts.get_mut(k as usize)) {
            *x = true;
        }
    };
    tick(meter, n.saturating_add(insts))?;
    for i in 0..n {
        let b = block_of(m, func, i)?;
        for phi in &b.phis {
            for (v, _) in &phi.incoming {
                by_parts(*v);
            }
        }
        for node in &b.insts {
            match &node.inst {
                Inst::Load { .. } => {}
                Inst::Store { value, .. } => by_parts(*value),
                other => other.for_each_value(&mut by_parts),
            }
        }
        b.term.for_each_value(&mut by_parts);
    }

    // Each marked variable's reads and whole stores, block by block.
    let mut acc: Vec<Accesses> = marked
        .iter()
        .map(|v| {
            let mut a = Accesses::default();
            if v.init.is_some() {
                a.note(0, false);
            }
            a
        })
        .collect();
    for i in (0..n).filter(|&i| reached(i)) {
        let bi = u32_of(i);
        let mut note = |k: u32, read: bool| {
            if let Some(a) = acc.get_mut(k as usize) {
                a.note(bi, read);
            }
        };
        for node in &block_of(m, func, i)?.insts {
            match &node.inst {
                Inst::Load { ptr, .. } => {
                    if let Some(&k) = own.get(ptr) {
                        note(k, true);
                    }
                }
                Inst::Store { ptr, .. } => {
                    if let Some(&k) = own.get(ptr) {
                        note(k, false);
                    }
                }
                _ => {}
            }
        }
    }

    // Each variable's live blocks, and the loops it is ended at.
    let mut live = vec![NONE; n];
    let mut kills = vec![NONE; n];
    let mut reads = vec![NONE; n];
    let stamped = |table: &[u32], b: usize, k: u32| table.get(b) == Some(&k);
    let mut out = Vec::new();
    for (k, (v, a)) in marked.iter().zip(&acc).enumerate() {
        let Some(&s0) = a.stores.first() else {
            continue;
        };
        if parts.get(k).copied().unwrap_or(true) {
            continue;
        }
        let k = u32_of(k);
        stack.clear();
        for e in &a.events {
            let b = e.block as usize;
            let table = if e.read_first { &mut reads } else { &mut kills };
            if let Some(x) = table.get_mut(b) {
                *x = k;
            }
            if e.read_first && reached(b) {
                if let Some(x) = live.get_mut(b) {
                    *x = k;
                }
                stack.push(b);
            }
        }
        while let Some(b) = stack.pop() {
            let p = preds_of(b);
            tick(meter, 1 + p.len())?;
            for &p in p {
                let p = p as usize;
                if stamped(&live, p, k) || stamped(&kills, p, k) {
                    continue;
                }
                if let Some(x) = live.get_mut(p) {
                    *x = k;
                }
                stack.push(p);
            }
        }
        for &l in loops_of
            .get(s0 as usize)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(&first) = firsts.get(l as usize) else {
                continue;
            };
            if a.stores.iter().all(|&s| in_loop(s, l))
                && stamped(&live, first, k)
                && !stamped(&reads, first, k)
            {
                let b = func
                    .blocks
                    .get(first)
                    .copied()
                    .ok_or_else(|| invariant("block"))?;
                out.push((v.value, v.pointee, b));
            }
        }
    }
    Ok(out)
}
