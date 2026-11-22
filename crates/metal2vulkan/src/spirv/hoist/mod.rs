//! Loads of uniform views that every trip of a loop reads alike, moved out of the loop.
//!
//! A kernel that reads its parameters from a `constant` buffer inside a loop often reads them again
//! on every trip, inside the branches of the loop body, because nothing told its compiler the
//! address cannot be read out of bounds. A driver's compiler keeps such a load where it stands
//! too: it does not move a load out of a branch. On a 1x1 convolution kernel whose outer loop
//! read 13 such words on each of its trips, the same module with those loads in the loop's
//! preheader ran 13% faster at 120 trips and 3.8% at 24 on an RX 7900 XTX (RADV), and 3.6% faster
//! at 120 trips on an RTX 2080 (bare dispatches, device-local buffers, alternating runs).
//!
//! [`hoist_uniform_loads`] moves a load out of a loop, into the loop's preheader (the block that
//! enters it), when all of the following hold. Each is read from the module's structure: storage
//! classes, the view's declaration, operand definitions, the loop table and the control flow
//! graph.
//!
//! - **A uniform view.** The load reads, through a chain of access indices, a module-scope
//!   `Uniform` variable that is a descriptor resource whose type is a `Block` structure:
//!   `memory::views` declares one only for a read-only buffer slot, and no shader writes a
//!   `Uniform` block. The load is neither volatile nor `MakePointerVisible`.
//! - **The same address on every trip.** Every operand of its address is defined outside the loop,
//!   or by an instruction inside it that computes its value from its operands alone (arithmetic,
//!   conversions, selects, composites; [`pure`]), or by another such load, whose operands are in
//!   turn the same on every trip. Those instructions move with it, in their order.
//! - **No write that could change the bytes.** No block from which the loop's header can be
//!   reached (its body, the code before it, and the rest of an enclosing loop through that loop's
//!   back edge) writes buffer or image memory: a store or atomic through a `StorageBuffer`,
//!   `PhysicalStorageBuffer`, `Uniform` or `Image` pointer, an image write, or a call to a function
//!   that makes one. Another slot's storage view may cover the same memory, and another
//!   invocation's write that a barrier inside the loop orders before a read would be such a
//!   write too. A loop in a function the entry calls moves loads only where the module makes no
//!   such write at all.
//! - **Safe to read early.** A load in the loop's header, or in a block the header leads to
//!   unconditionally and alone ([`entry_run`]), runs whenever the loop is entered, so it moves on
//!   every profile. A load elsewhere in the loop runs on some trips or paths only; it
//!   moves only where the profile's bounds rest on the device's robust buffer access
//!   ([`Robustness::Rba`] or [`Robustness::Rba2`]), under which a read
//!   past the bound range reads defined bytes rather than faulting, and a uniform view carries no
//!   in-shader check that moving it could pass by.
//!
//! The loops are taken innermost first, so a load leaves every loop it is the same in. The moved
//! values are the same values, computed earlier: nothing a dispatch computes changes.

#[cfg(test)]
mod tests;

use crate::budget::Meter;
use crate::device::profile::Robustness;
use crate::error::Error;
use crate::spirv::ids::{BlockId, Value};
use crate::spirv::ir::{
    GlobalKind, Inst, InstNode, LoopMeta, StorageClass, Terminator, Type, TypedModule, ValueOrigin,
};

/// How deep an address's chain of moved instructions may go: a deeper one stays (a bound on the
/// walk, which no measured address reaches).
const MAX_DEPTH: u32 = 64;

/// Move the loads of uniform views that are the same on every trip of their loop out of it, under
/// `containment` (the module docs). Returns the moves made: a load counts once for each loop it
/// leaves.
pub fn hoist_uniform_loads(
    m: &mut TypedModule,
    robustness: Robustness,
    meter: &mut Meter,
) -> Result<u32, Error> {
    if m.loops.is_empty() {
        return Ok(0);
    }
    let speculate = matches!(robustness, Robustness::Rba | Robustness::Rba2);
    let writers = buffer_writers(m, meter)?;
    let any_writer = writers.iter().any(|&w| w);
    let blocks = m.blocks.len();
    let values = m.values.len();
    let bytes = blocks
        .saturating_mul(std::mem::size_of::<u32>() * 2 + std::mem::size_of::<Vec<BlockId>>())
        .saturating_add(values.saturating_mul(std::mem::size_of::<u32>() * 3 + 1));
    meter.charge(bytes)?;
    let mut scratch = Scratch {
        body: vec![0; blocks],
        reach: vec![0; blocks],
        seen: vec![0; values],
        movable: vec![false; values],
        stamp: 0,
        preds: preds(m),
        pos: vec![u32::MAX; values],
    };
    for b in 0..blocks {
        scratch.place(m, BlockId::new(u32::try_from(b).unwrap_or(u32::MAX)));
    }
    let r = hoist_loops(m, (speculate, any_writer), &writers, &mut scratch, meter);
    meter.release(bytes);
    let moved = r?;
    if moved > 0 {
        m.rederive()?;
    }
    Ok(moved)
}

/// The per-block and per-value marks one loop at a time reads, each stamped with the loop's turn
/// so a new loop needs no clearing.
struct Scratch {
    body: Vec<u32>,
    reach: Vec<u32>,
    seen: Vec<u32>,
    movable: Vec<bool>,
    stamp: u32,
    preds: Vec<Vec<BlockId>>,
    /// The index of each instruction result in its block, indexed by [`Value`].
    pos: Vec<u32>,
}

impl Scratch {
    fn marked(marks: &[u32], b: BlockId, stamp: u32) -> bool {
        marks.get(b.index()).is_some_and(|&s| s == stamp)
    }

    fn in_body(&self, b: BlockId) -> bool {
        Scratch::marked(&self.body, b, self.stamp)
    }

    /// Record the place of every instruction result of `b`.
    fn place(&mut self, m: &TypedModule, b: BlockId) {
        let Some(block) = m.blocks.get(b.index()) else {
            return;
        };
        for (i, n) in block.insts.iter().enumerate() {
            if let Some(slot) = n.result.and_then(|r| self.pos.get_mut(r.index())) {
                *slot = u32::try_from(i).unwrap_or(u32::MAX);
            }
        }
    }

    /// The instruction that defines `v`.
    fn node_of<'m>(&self, m: &'m TypedModule, v: Value) -> Option<&'m InstNode> {
        let ValueOrigin::Inst(b) = m.values.get(v.index())?.origin else {
            return None;
        };
        let at = usize::try_from(*self.pos.get(v.index())?).ok()?;
        m.blocks
            .get(b.index())?
            .insts
            .get(at)
            .filter(|n| n.result == Some(v))
    }
}

/// Every loop of `m` whose preheader enters it, innermost first, each moved as the module docs
/// say. `(speculate, any_writer)`: whether a load outside the header may move, and whether the
/// module writes buffer memory anywhere.
fn hoist_loops(
    m: &mut TypedModule,
    (speculate, any_writer): (bool, bool),
    writers: &[bool],
    s: &mut Scratch,
    meter: &mut Meter,
) -> Result<u32, Error> {
    let entry = m.entry.func;
    let mut loops: Vec<(usize, LoopMeta)> = Vec::new();
    for &lm in &m.loops {
        let Some(header) = m.blocks.get(lm.header.index()) else {
            continue;
        };
        let enters = m
            .blocks
            .get(lm.preheader.index())
            .is_some_and(|p| p.func == header.func && p.term == Terminator::Branch(lm.header));
        if !enters || (header.func != entry && any_writer) {
            continue;
        }
        s.stamp = s.stamp.wrapping_add(1);
        let size = mark_body(m, &lm, s, meter)?;
        loops.push((size, lm));
    }
    // A loop nested in another is smaller than it: innermost first.
    loops.sort_by_key(|&(size, lm)| (size, lm.header));
    let mut moved = 0u32;
    for (_, lm) in loops {
        s.stamp = s.stamp.wrapping_add(1);
        mark_body(m, &lm, s, meter)?;
        if writes_before(m, &lm, writers, s, meter)? {
            continue;
        }
        moved = moved.saturating_add(hoist_loop(m, &lm, speculate, s, meter)?);
    }
    Ok(moved)
}

/// Whether `f` and the functions it calls write buffer or image memory, indexed by
/// [`FuncId`](crate::air::ir::FuncId).
fn buffer_writers(m: &TypedModule, meter: &mut Meter) -> Result<Vec<bool>, Error> {
    let mut writes = vec![false; m.functions.len()];
    // Calls form no cycle: a pass per call depth reaches the fixed point.
    loop {
        let mut changed = false;
        for b in &m.blocks {
            meter.tick(u32::try_from(b.insts.len()).unwrap_or(u32::MAX))?;
            if writes.get(b.func.index()).copied().unwrap_or(true) {
                continue;
            }
            if b.insts.iter().any(|n| writes_memory(m, &n.inst, &writes)) {
                if let Some(w) = writes.get_mut(b.func.index()) {
                    *w = true;
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(writes);
        }
    }
}

/// Whether `inst` writes buffer or image memory, a call through `writers`.
fn writes_memory(m: &TypedModule, inst: &Inst, writers: &[bool]) -> bool {
    let reaches = |ptr: Value| {
        pointer_class(m, ptr).is_none_or(|c| {
            matches!(
                c,
                StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::Uniform
                    | StorageClass::Image
            )
        })
    };
    match inst {
        Inst::Store { ptr, .. }
        | Inst::AtomicStore { ptr, .. }
        | Inst::Atomic { ptr, .. }
        | Inst::AtomicCompareExchange { ptr, .. } => reaches(*ptr),
        Inst::ImageWrite { .. } => true,
        Inst::Call { func, .. } => writers.get(func.index()).copied().unwrap_or(true),
        _ => false,
    }
}

/// The storage class of the pointer `v`.
fn pointer_class(m: &TypedModule, v: Value) -> Option<StorageClass> {
    let def = m.values.get(v.index())?;
    match m.types.get(def.ty).ok()? {
        Type::Pointer { class, .. } => Some(*class),
        _ => None,
    }
}

/// The blocks `term` may branch to.
fn targets(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Branch(t) => vec![*t],
        Terminator::CondBranch { t, f, .. } => vec![*t, *f],
        Terminator::Switch { default, cases, .. } => std::iter::once(*default)
            .chain(cases.iter().map(|&(_, c)| c))
            .collect(),
        _ => Vec::new(),
    }
}

/// Every block's predecessors, indexed by [`BlockId`].
fn preds(m: &TypedModule) -> Vec<Vec<BlockId>> {
    let mut preds = vec![Vec::new(); m.blocks.len()];
    for (i, b) in m.blocks.iter().enumerate() {
        let from = BlockId::new(u32::try_from(i).unwrap_or(u32::MAX));
        for t in targets(&b.term) {
            if let Some(p) = preds.get_mut(t.index()) {
                p.push(from);
            }
        }
    }
    preds
}

/// Mark the blocks of `lm`'s loop with the current stamp: those its header reaches without passing
/// its merge, less any whose predecessors are not all among them (the header's excepted), so
/// every marked block is reached only through the header. Returns their count.
fn mark_body(
    m: &TypedModule,
    lm: &LoopMeta,
    s: &mut Scratch,
    meter: &mut Meter,
) -> Result<usize, Error> {
    let stamp = s.stamp;
    let mut stack = vec![lm.header];
    let mut body = Vec::new();
    while let Some(b) = stack.pop() {
        if b == lm.merge || Scratch::marked(&s.body, b, stamp) {
            continue;
        }
        let Some(block) = m.blocks.get(b.index()) else {
            continue;
        };
        meter.tick(1)?;
        if let Some(mark) = s.body.get_mut(b.index()) {
            *mark = stamp;
        }
        body.push(b);
        stack.extend(targets(&block.term));
    }
    // Unmark any block entered from outside, until none is left.
    loop {
        let mut dropped = false;
        for &b in &body {
            if b == lm.header || !Scratch::marked(&s.body, b, stamp) {
                continue;
            }
            let outside = s
                .preds
                .get(b.index())
                .is_some_and(|ps| ps.iter().any(|&p| !Scratch::marked(&s.body, p, stamp)));
            if outside {
                if let Some(mark) = s.body.get_mut(b.index()) {
                    *mark = 0;
                }
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }
    Ok(body
        .iter()
        .filter(|&&b| Scratch::marked(&s.body, b, stamp))
        .count())
}

/// Whether a block from which `lm`'s header can be reached, or a block of its body, writes buffer
/// or image memory.
fn writes_before(
    m: &TypedModule,
    lm: &LoopMeta,
    writers: &[bool],
    s: &mut Scratch,
    meter: &mut Meter,
) -> Result<bool, Error> {
    let stamp = s.stamp;
    let writes = |b: BlockId| {
        m.blocks
            .get(b.index())
            .is_none_or(|bl| bl.insts.iter().any(|n| writes_memory(m, &n.inst, writers)))
    };
    let mut stack = vec![lm.header];
    while let Some(b) = stack.pop() {
        if Scratch::marked(&s.reach, b, stamp) {
            continue;
        }
        if let Some(mark) = s.reach.get_mut(b.index()) {
            *mark = stamp;
        }
        let n = m.blocks.get(b.index()).map_or(0, |bl| bl.insts.len());
        meter.tick(u32::try_from(n).unwrap_or(u32::MAX))?;
        if writes(b) {
            return Ok(true);
        }
        if let Some(ps) = s.preds.get(b.index()) {
            stack.extend(ps.iter().copied());
        }
    }
    let func = m.blocks.get(lm.header.index()).map(|b| b.func);
    let body: Vec<BlockId> = func
        .and_then(|f| m.functions.get(f.index()))
        .map(|f| f.blocks.iter().copied().filter(|&b| s.in_body(b)).collect())
        .unwrap_or_default();
    Ok(body.into_iter().any(writes))
}

/// Move `lm`'s movable loads and the instructions their addresses need into its preheader.
/// Returns the loads moved.
fn hoist_loop(
    m: &mut TypedModule,
    lm: &LoopMeta,
    speculate: bool,
    s: &mut Scratch,
    meter: &mut Meter,
) -> Result<u32, Error> {
    let Some(func) = m.blocks.get(lm.header.index()).map(|b| b.func) else {
        return Ok(0);
    };
    let layout: Vec<BlockId> = m
        .functions
        .get(func.index())
        .map(|f| f.blocks.iter().copied().filter(|&b| s.in_body(b)).collect())
        .unwrap_or_default();
    let run = entry_run(m, lm, s);
    // Every movable load, then every instruction their addresses need, in layout order.
    let mut take: Vec<(usize, usize)> = Vec::new();
    let mut loads = 0u32;
    for (at, &b) in layout.iter().enumerate() {
        let Some(block) = m.blocks.get(b.index()) else {
            continue;
        };
        meter.tick(u32::try_from(block.insts.len()).unwrap_or(u32::MAX))?;
        let early = speculate || run.contains(&b);
        for (i, node) in block.insts.iter().enumerate() {
            let (Some(v), Inst::Load { .. } | Inst::ImageFetch { .. }) = (node.result, &node.inst)
            else {
                continue;
            };
            if early && movable(m, v, s, 0) {
                loads = loads.saturating_add(1);
                take.push((at, i));
            }
        }
    }
    if loads == 0 {
        return Ok(0);
    }
    // The loads' operands defined in the body, which `movable` proved movable.
    let mut need: Vec<Value> = Vec::new();
    for &(at, i) in &take {
        if let Some(v) = layout
            .get(at)
            .and_then(|b| m.blocks.get(b.index()))
            .and_then(|bl| bl.insts.get(i))
            .and_then(|n| n.result)
        {
            need.push(v);
        }
    }
    let mut closure: Vec<Value> = Vec::new();
    while let Some(v) = need.pop() {
        if closure.contains(&v) {
            continue;
        }
        let Some(node) = s.node_of(m, v) else {
            continue;
        };
        closure.push(v);
        node.inst.for_each_value(|o| {
            if body_inst(m, o, s).is_some() {
                need.push(o);
            }
        });
    }
    // In layout order: a definition comes before its uses.
    let mut order: Vec<(usize, u32, Value)> = Vec::new();
    for &v in &closure {
        let Some(b) = body_inst(m, v, s) else {
            continue;
        };
        let at = layout.iter().position(|&l| l == b).unwrap_or(usize::MAX);
        let i = s.pos.get(v.index()).copied().unwrap_or(u32::MAX);
        order.push((at, i, v));
    }
    order.sort_unstable();
    let mut moving: Vec<InstNode> = Vec::new();
    for &(_, _, v) in &order {
        if let Some(node) = s.node_of(m, v) {
            moving.push(node.clone());
        }
    }
    for b in &layout {
        if let Some(block) = m.blocks.get_mut(b.index()) {
            block
                .insts
                .retain(|n| n.result.is_none_or(|r| !closure.contains(&r)));
        }
    }
    for &(_, _, v) in &order {
        if let Some(def) = m.values.get_mut(v.index()) {
            def.origin = ValueOrigin::Inst(lm.preheader);
        }
    }
    if let Some(pre) = m.blocks.get_mut(lm.preheader.index()) {
        pre.insts.extend(moving);
    }
    for &b in layout.iter().chain(std::iter::once(&lm.preheader)) {
        s.place(m, b);
    }
    Ok(loads)
}

/// The blocks that run whenever the loop of `lm` is entered, before any of its branches can leave
/// it: the header, then each block the one before it branches to unconditionally and alone enters.
fn entry_run(m: &TypedModule, lm: &LoopMeta, s: &Scratch) -> Vec<BlockId> {
    let mut run = vec![lm.header];
    let mut at = lm.header;
    while let Some(Terminator::Branch(next)) = m.blocks.get(at.index()).map(|b| &b.term) {
        let alone = s.preds.get(next.index()).is_some_and(|p| p == &[at]);
        if !alone || !s.in_body(*next) || run.contains(next) {
            break;
        }
        run.push(*next);
        at = *next;
    }
    run
}

/// The body block that defines `v` by an instruction, if any.
fn body_inst(m: &TypedModule, v: Value, s: &Scratch) -> Option<BlockId> {
    match m.values.get(v.index())?.origin {
        ValueOrigin::Inst(b) if s.in_body(b) => Some(b),
        _ => None,
    }
}

/// Whether `v` is the same on every trip of the loop marked in `s`: defined outside it, or by an
/// instruction of its body that may move with the loads ([`movable`]).
fn invariant(m: &TypedModule, v: Value, s: &mut Scratch, depth: u32) -> bool {
    match m.values.get(v.index()).map(|d| d.origin) {
        Some(ValueOrigin::Inst(b)) if s.in_body(b) => movable(m, v, s, depth),
        Some(ValueOrigin::Phi(b)) if s.in_body(b) => false,
        Some(_) => true,
        None => false,
    }
}

/// Whether the body instruction defining `v` may move to the preheader: a load of a uniform view,
/// a fetch of a texel form's texel, or an instruction [`pure`] admits, whose operands are all
/// [`invariant`]. Memoized per loop.
fn movable(m: &TypedModule, v: Value, s: &mut Scratch, depth: u32) -> bool {
    let stamp = s.stamp;
    if s.seen.get(v.index()).is_some_and(|&x| x == stamp) {
        return s.movable.get(v.index()).copied().unwrap_or(false);
    }
    // Marked first, as not movable: an operand cycle (through no phi, so none exists) or a walk
    // past the depth bound reads it so.
    if let Some(x) = s.seen.get_mut(v.index()) {
        *x = stamp;
    }
    if let Some(x) = s.movable.get_mut(v.index()) {
        *x = false;
    }
    if depth >= MAX_DEPTH {
        return false;
    }
    let Some(node) = s.node_of(m, v) else {
        return false;
    };
    let ok = match &node.inst {
        Inst::Load { ptr, access } => {
            !access.volatile
                && access.make_visible.is_none()
                && uniform_view(m, s, *ptr)
                && invariant(m, *ptr, s, depth + 1)
        }
        Inst::AccessChain {
            base,
            non_uniform: false,
            ..
        } if uniform_view(m, s, *base) => operands_invariant(m, &node.inst, s, depth),
        inst if pure(inst) => operands_invariant(m, inst, s, depth),
        _ => false,
    };
    if let Some(x) = s.movable.get_mut(v.index()) {
        *x = ok;
    }
    ok
}

fn operands_invariant(m: &TypedModule, inst: &Inst, s: &mut Scratch, depth: u32) -> bool {
    let mut ops = Vec::new();
    inst.for_each_value(|o| ops.push(o));
    ops.into_iter().all(|o| invariant(m, o, s, depth + 1))
}

/// Whether `inst` computes its result from its operands alone, reading no memory and with no
/// effect, and is defined for every operand value (a division by zero gives an undefined value, not
/// undefined behavior), so computing it early changes nothing.
fn pure(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Unary { .. }
            | Inst::Binary { .. }
            | Inst::Select { .. }
            | Inst::Convert { .. }
            | Inst::BitFieldInsert { .. }
            | Inst::BitFieldExtract { .. }
            | Inst::CompositeConstruct { .. }
            | Inst::CompositeExtract { .. }
            | Inst::CompositeInsert { .. }
            | Inst::VectorExtractDynamic { .. }
            | Inst::VectorInsertDynamic { .. }
            | Inst::VectorShuffle { .. }
    )
}

/// Whether the pointer `v` points into a uniform view: a chain of access indices from a
/// module-scope `Uniform` descriptor resource whose type is a `Block` structure.
fn uniform_view(m: &TypedModule, s: &Scratch, v: Value) -> bool {
    let mut v = v;
    for _ in 0..MAX_DEPTH {
        match m.values.get(v.index()).map(|d| d.origin) {
            Some(ValueOrigin::Global(g)) => {
                let Some(g) = m.globals.get(g.index()) else {
                    return false;
                };
                let block = matches!(
                    m.types.get(g.pointee),
                    Ok(Type::Struct { layout: Some(l), .. }) if l.block
                );
                return matches!(
                    (g.class, &g.kind),
                    (StorageClass::Uniform, GlobalKind::Resource(_))
                ) && block;
            }
            Some(ValueOrigin::Inst(_)) => match s.node_of(m, v).map(|n| &n.inst) {
                Some(Inst::AccessChain {
                    base,
                    non_uniform: false,
                    ..
                }) => v = *base,
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}
