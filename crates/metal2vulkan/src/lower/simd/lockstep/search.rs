//! The pair search over the accesses of one memory space: from each written block, a walk forward
//! to the accesses it reaches and a walk back to the reads that reach it, on barrier-free paths,
//! each stopping where every pair beyond it already has its barrier.
//!
//! **What a pair beyond wants.** [`Shape::place`] answers a pair `a` then `b` in blocks `x` and
//! `y` (`x != y`) with a barrier that, in three cases, depends on one side only:
//!
//! - `x` uniform across the threadgroup: right after `a`, to the end of `x` ([`after`]);
//! - otherwise `y` uniform across the threadgroup: right before `b`, from the start of `y`
//!   ([`before`]);
//! - otherwise `y` dominated by `x`'s crossing block `c` ([`crossing`]): the whole of `c`
//!   ([`whole`]), whatever `y` is.
//!
//! A pair whose answer is already a want adds nothing, whether it conflicts or not, so it is not
//! tried. Wants only grow, so a pair skipped stays answered.
//!
//! **Where a walk stops.** A dominator subtree `D(y)` that no edge leaves, with no barrier and no
//! retreating edge in it, is entered only through `y`, and every block of it is reached from `y`'s
//! end on a barrier-free path that crosses no back edge: so each block of it is reached from the
//! walk's start exactly as `y` is. The forward walk from a divergent `x` whose crossing block's
//! want is wanted, at such a `y` inside `D(c)`, has nothing left there but pairs into uniform
//! blocks (each answered by the barrier before its access): it stops expanding `y` and tries
//! those blocks' unanswered accesses directly, found by their preorder numbers inside `D(y)`.
//! The backward walk stops the same way at a post-dominator subtree that no edge enters from
//! outside, from whose divergent reading blocks every pair into the walk's block is answered,
//! and tries the uniform reading blocks inside it directly. A walk from a uniform `x` ends once
//! each of its writes has its barrier.
//!
//! So a long barrier-free body whose pairs conflict is searched in time about linear in its
//! blocks; a pair that conflicts with nothing is still tried against each access it reaches.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::ir::{BlockId, Idx};
use crate::budget::{tree_bytes, vec_bytes, Meter};
use crate::error::Error;
use crate::graph::{Cfg, Graph, Sccs};

use super::facts::{Access, Accesses, Addr};
use super::{via, Avoid, Order, Shape, Want};

const NONE: u32 = u32::MAX;

/// A block where a walk stopped expanding, beside [`via`]'s bits in its scratch state.
const STOPPED: u8 = 4;

/// The longest chain of immediate post-dominators [`Shape::place`] climbs.
const CHAIN: u32 = 64;

fn block(raw: usize) -> BlockId {
    BlockId::new(u32::try_from(raw).unwrap_or(NONE))
}

fn flag(cross: bool) -> u8 {
    if cross {
        via::CROSS
    } else {
        via::SAME
    }
}

/// The barrier right after `a`, to the end of its block `x`.
fn after(s: &Shape<'_>, x: BlockId, a: &Access) -> Want {
    Want {
        block: x.raw(),
        order: Order::Threadgroup,
        hi: s.term(x),
        lo: a.pos.saturating_add(1),
    }
}

/// The barrier right before `b`, from the start of its block `y`.
fn before(s: &Shape<'_>, y: BlockId, b: &Access) -> Want {
    Want {
        block: y.raw(),
        order: Order::Threadgroup,
        hi: b.pos,
        lo: s.first(y),
    }
}

/// The barrier anywhere in block `c`.
fn whole(s: &Shape<'_>, c: BlockId) -> Want {
    Want {
        block: c.raw(),
        order: Order::Threadgroup,
        hi: s.term(c),
        lo: s.first(c),
    }
}

/// The crossing block of divergent block `x`: the first block on its chain of immediate
/// post-dominators that the threadgroup runs together and that does not dominate `x`, within
/// the steps [`Shape::place`] takes, when no block before it on the chain lies in its dominator
/// subtree. A pair from `x` into a divergent block that it dominates then wants a threadgroup
/// barrier across all of it: no block before it on the chain stops the climb or answers.
fn crossing(
    s: &Shape<'_>,
    x: BlockId,
    chain: &mut Vec<BlockId>,
    m: &mut Meter,
) -> Result<Option<BlockId>, Error> {
    chain.clear();
    let mut z = s.pdom.ipdom(x);
    let mut steps = 0u32;
    while let Some(zb) = z {
        m.tick(1)?;
        steps += 1;
        if steps > CHAIN {
            return Ok(None);
        }
        if s.wg(zb) && !s.e.dom.dominates(zb, x) {
            let clear = chain.iter().all(|&p| !s.e.dom.dominates(zb, p));
            return Ok(clear.then_some(zb));
        }
        m.grow(chain, 1)?;
        chain.push(zb);
        z = s.pdom.ipdom(zb);
    }
    Ok(None)
}

/// A vector of `n` copies of `v`, charged.
fn filled<T: Clone>(n: usize, v: T, m: &mut Meter) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    m.reserve_exact(&mut out, n)?;
    out.resize(n, v);
    Ok(out)
}

fn get(v: &[u32], i: usize) -> u32 {
    v.get(i).copied().unwrap_or(NONE)
}

/// Preorder intervals of a forest over the blocks (the dominator or post-dominator tree): the
/// subtree of a block holds the blocks numbered in `[pre, end)`.
struct Forest {
    pre: Vec<u32>,
    end: Vec<u32>,
    /// Preorder number -> block.
    order: Vec<u32>,
}

impl Forest {
    fn new(
        n: usize,
        member: impl Fn(BlockId) -> bool,
        parent: impl Fn(BlockId) -> Option<BlockId>,
        m: &mut Meter,
    ) -> Result<Forest, Error> {
        let up = |v: usize| {
            let b = block(v);
            if member(b) {
                parent(b)
            } else {
                None
            }
        };
        // The children of each block, by counting.
        let mut start = filled(n + 1, 0u32, m)?;
        for v in 0..n {
            m.tick(1)?;
            if let Some(c) = up(v).and_then(|p| start.get_mut(p.index() + 1)) {
                *c += 1;
            }
        }
        for i in 1..start.len() {
            let prev = get(&start, i - 1);
            if let Some(c) = start.get_mut(i) {
                *c = c.saturating_add(prev);
            }
        }
        let mut next = start.clone();
        m.charge(vec_bytes(&next))?;
        let mut children = filled(get(&start, n) as usize, 0u32, m)?;
        for v in 0..n {
            if let Some(p) = up(v) {
                if let Some(at) = next.get_mut(p.index()) {
                    if let Some(c) = children.get_mut(*at as usize) {
                        *c = block(v).raw();
                    }
                    *at += 1;
                }
            }
        }
        let mut pre = filled(n, NONE, m)?;
        let mut end = filled(n, NONE, m)?;
        let mut order = Vec::new();
        m.reserve_exact(&mut order, n)?;
        let mut stack: Vec<(u32, u32)> = Vec::new();
        let mut count = 0u32;
        for r in 0..n {
            if !member(block(r)) || up(r).is_some() {
                continue;
            }
            let mut v = block(r).raw();
            loop {
                // Enter `v`.
                if let Some(p) = pre.get_mut(v as usize) {
                    *p = count;
                }
                count += 1;
                order.push(v);
                m.grow(&mut stack, 1)?;
                stack.push((v, get(&start, v as usize)));
                // Leave every finished block, then go down to the next child.
                let mut next_child = None;
                while let Some(&(top, at)) = stack.last() {
                    m.tick(1)?;
                    if at < get(&start, top as usize + 1) {
                        if let Some(t) = stack.last_mut() {
                            t.1 += 1;
                        }
                        next_child = Some(get(&children, at as usize));
                        break;
                    }
                    if let Some(e) = end.get_mut(top as usize) {
                        *e = count;
                    }
                    stack.pop();
                }
                match next_child {
                    Some(c) => v = c,
                    None => break,
                }
            }
        }
        m.release(vec_bytes(&start) + vec_bytes(&next) + vec_bytes(&children) + vec_bytes(&stack));
        Ok(Forest { pre, end, order })
    }

    fn pre(&self, b: BlockId) -> u32 {
        get(&self.pre, b.index())
    }

    fn end(&self, b: BlockId) -> u32 {
        get(&self.end, b.index())
    }

    fn at(&self, p: u32) -> Option<BlockId> {
        self.order.get(p as usize).map(|&v| BlockId::new(v))
    }

    /// Blocks of the forest, children before their parents.
    fn upward(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.order.iter().rev().map(|&v| BlockId::new(v))
    }

    fn bytes(&self) -> usize {
        vec_bytes(&self.pre) + vec_bytes(&self.end) + vec_bytes(&self.order)
    }
}

/// Counts by position, summed over ranges (a Fenwick tree).
struct Counts(Vec<i32>);

impl Counts {
    fn add(&mut self, at: u32, d: i32) {
        let mut i = at as usize + 1;
        while let Some(c) = self.0.get_mut(i) {
            *c += d;
            i += i.isolate_lowest_one();
        }
    }

    /// The sum over `[0, at)`.
    fn below(&self, at: u32) -> i32 {
        let mut i = (at as usize).min(self.0.len().saturating_sub(1));
        let mut sum = 0;
        while i > 0 {
            sum += self.0.get(i).copied().unwrap_or(0);
            i -= i.isolate_lowest_one();
        }
        sum
    }

    fn range(&self, lo: u32, hi: u32) -> i32 {
        self.below(hi) - self.below(lo)
    }
}

/// The body with the barrier blocks' out-edges cut: a cycle of it is a barrier-free loop.
struct Unbarred<'a> {
    cfg: &'a Cfg<BlockId>,
    acc: &'a Accesses,
}

impl Graph for Unbarred<'_> {
    type Node = BlockId;

    fn node_count(&self) -> usize {
        self.cfg.node_count()
    }

    fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    fn successors(&self, b: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        let open = !self.acc.has_barrier(b);
        self.cfg.successors(b).filter(move |_| open)
    }
}

/// The facts of the body the walks stop by, fixed for the search.
struct Regions {
    dom: Forest,
    pdom: Forest,
    /// By divergent block with an access: its crossing block ([`crossing`]), or `NONE`.
    crossing: Vec<u32>,
    /// By barrier-free block: it lies on a barrier-free cycle.
    cyclic: Vec<bool>,
    /// By block `y`: no edge leaves its dominator subtree, which holds no barrier and no
    /// retreating edge.
    ahead: Vec<bool>,
    /// By block `y`: its dominator subtree holds a divergent block with an access.
    ahead_divergent: Vec<bool>,
    /// By block `w`: no edge enters its post-dominator subtree from outside, and the subtree
    /// holds no barrier and no retreating edge.
    behind: Vec<bool>,
    /// By block `w`: its post-dominator subtree holds a divergent block with a read.
    behind_divergent: Vec<bool>,
    /// By block `w`: the dominator preorder interval every crossing block of the divergent
    /// reading blocks in its post-dominator subtree dominates (empty when one has none).
    behind_crossed: Vec<(u32, u32)>,
}

impl Regions {
    fn new(s: &Shape<'_>, m: &mut Meter) -> Result<Regions, Error> {
        let (e, acc) = (s.e, s.acc);
        let cfg = &e.cfg;
        let n = cfg.node_count();
        let dom = Forest::new(n, |b| e.dom.is_reachable(b), |b| e.dom.idom(b), m)?;
        let pdom = Forest::new(n, |b| s.pdom.is_reachable(b), |b| s.pdom.ipdom(b), m)?;
        let reads = |b: BlockId| acc.in_block(b).iter().any(|a| !a.write);
        let mut crossing_of = filled(n, NONE, m)?;
        let mut chain = Vec::new();
        for v in 0..n {
            let b = block(v);
            if s.wg(b) || acc.in_block(b).is_empty() {
                continue;
            }
            if let (Some(c), Some(slot)) = (crossing(s, b, &mut chain, m)?, crossing_of.get_mut(v))
            {
                *slot = c.raw();
            }
        }
        m.release(vec_bytes(&chain));
        let sccs = Sccs::compute(&Unbarred { cfg, acc }, m)?;
        let mut cyclic = filled(n, false, m)?;
        for c in 0..u32::try_from(sccs.len()).unwrap_or(NONE) {
            m.tick(1)?;
            if sccs.is_cyclic(c) {
                for b in sccs.members(c) {
                    if let Some(f) = cyclic.get_mut(b.index()) {
                        *f = true;
                    }
                }
            }
        }
        m.release(sccs.ledger_bytes());
        // Ahead: by dominator subtree, the lowest and highest numbers of its blocks' successors,
        // whether it holds a barrier or a retreating edge, and a divergent block with an access.
        let mut lo = filled(n, NONE, m)?;
        let mut hi = filled(n, 0u32, m)?;
        let mut shut = filled(n, false, m)?;
        let mut divergent = filled(n, false, m)?;
        for y in dom.upward() {
            m.tick(1)?;
            let v = y.index();
            let (mut l, mut h) = (get(&lo, v), get(&hi, v));
            let mut barred = shut.get(v).copied().unwrap_or(true) || acc.has_barrier(y);
            for t in cfg.successors(y) {
                let p = dom.pre(t);
                l = l.min(p);
                h = h.max(p);
                barred |= cfg.is_retreating(y, t);
            }
            let d = divergent.get(v).copied().unwrap_or(false)
                || (!s.wg(y) && !acc.in_block(y).is_empty());
            for (slot, val) in [(&mut lo, l), (&mut hi, h)] {
                if let Some(x) = slot.get_mut(v) {
                    *x = val;
                }
            }
            if let Some(x) = shut.get_mut(v) {
                *x = barred;
            }
            if let Some(x) = divergent.get_mut(v) {
                *x = d;
            }
            if let Some(p) = e.dom.idom(y).map(|p| p.index()) {
                let (pl, ph) = (get(&lo, p).min(l), get(&hi, p).max(h));
                if let Some(x) = lo.get_mut(p) {
                    *x = pl;
                }
                if let Some(x) = hi.get_mut(p) {
                    *x = ph;
                }
                if let Some(x) = shut.get_mut(p) {
                    *x |= barred;
                }
                if let Some(x) = divergent.get_mut(p) {
                    *x |= d;
                }
            }
        }
        let mut ahead = filled(n, false, m)?;
        for (v, slot) in ahead.iter_mut().enumerate() {
            let y = block(v);
            *slot = dom.pre(y) != NONE
                && !shut.get(v).copied().unwrap_or(true)
                && get(&lo, v) >= dom.pre(y)
                && get(&hi, v) < dom.end(y);
        }
        let ahead_divergent = divergent;
        // Behind: the same by post-dominator subtree over its blocks' predecessors, a divergent
        // block with a read, and the crossing blocks' interval.
        lo.fill(NONE);
        hi.fill(0);
        shut.fill(false);
        let mut divergent = filled(n, false, m)?;
        let mut crossed = filled(n, (0u32, NONE), m)?;
        for w in pdom.upward() {
            m.tick(1)?;
            let v = w.index();
            let (mut l, mut h) = (get(&lo, v), get(&hi, v));
            let mut barred = shut.get(v).copied().unwrap_or(true) || acc.has_barrier(w);
            for p in cfg.predecessors(w) {
                let q = pdom.pre(p);
                l = l.min(q);
                h = h.max(q);
                barred |= cfg.is_retreating(p, w);
            }
            let mut d = divergent.get(v).copied().unwrap_or(false);
            let (mut cl, mut ch) = crossed.get(v).copied().unwrap_or((NONE, 0));
            if !s.wg(w) && reads(w) {
                d = true;
                match get(&crossing_of, v) {
                    NONE => (cl, ch) = (NONE, 0),
                    c => {
                        let c = BlockId::new(c);
                        cl = cl.max(dom.pre(c));
                        ch = ch.min(dom.end(c));
                    }
                }
            }
            for (slot, val) in [(&mut lo, l), (&mut hi, h)] {
                if let Some(x) = slot.get_mut(v) {
                    *x = val;
                }
            }
            if let Some(x) = shut.get_mut(v) {
                *x = barred;
            }
            if let Some(x) = divergent.get_mut(v) {
                *x = d;
            }
            if let Some(x) = crossed.get_mut(v) {
                *x = (cl, ch);
            }
            if let Some(p) = s.pdom.ipdom(w).map(|p| p.index()) {
                let (pl, ph) = (get(&lo, p).min(l), get(&hi, p).max(h));
                if let Some(x) = lo.get_mut(p) {
                    *x = pl;
                }
                if let Some(x) = hi.get_mut(p) {
                    *x = ph;
                }
                if let Some(x) = shut.get_mut(p) {
                    *x |= barred;
                }
                if let Some(x) = divergent.get_mut(p) {
                    *x |= d;
                }
                if let Some(x) = crossed.get_mut(p) {
                    *x = (x.0.max(cl), x.1.min(ch));
                }
            }
        }
        let mut behind = filled(n, false, m)?;
        for (v, slot) in behind.iter_mut().enumerate() {
            let w = block(v);
            *slot = pdom.pre(w) != NONE
                && !shut.get(v).copied().unwrap_or(true)
                && get(&lo, v) >= pdom.pre(w)
                && get(&hi, v) < pdom.end(w);
        }
        m.release(vec_bytes(&lo) + vec_bytes(&hi) + vec_bytes(&shut));
        Ok(Regions {
            dom,
            pdom,
            crossing: crossing_of,
            cyclic,
            ahead,
            ahead_divergent,
            behind,
            behind_divergent: divergent,
            behind_crossed: crossed,
        })
    }

    fn crossing(&self, x: BlockId) -> Option<BlockId> {
        Some(get(&self.crossing, x.index()))
            .filter(|&c| c != NONE)
            .map(BlockId::new)
    }

    fn bytes(&self) -> usize {
        self.dom.bytes()
            + self.pdom.bytes()
            + vec_bytes(&self.crossing)
            + vec_bytes(&self.cyclic)
            + vec_bytes(&self.ahead)
            + vec_bytes(&self.ahead_divergent)
            + vec_bytes(&self.behind)
            + vec_bytes(&self.behind_divergent)
            + vec_bytes(&self.behind_crossed)
    }
}

/// The direction of a walk: forward from a block's end, or back from a block's start.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    Ahead,
    Behind,
}

/// The search over the accesses of one memory space, collecting into `wants`.
pub(super) struct Search<'s, 'a> {
    shape: &'s Shape<'a>,
    addrs: &'s [Addr],
    empty: Addr,
    wants: &'s mut BTreeSet<Want>,
    /// Wants inserted so far.
    added: u64,
    avoid: Avoid,
    regions: Regions,
    /// Scratch: by block, the states the current walk reached ([`via`]), and [`STOPPED`].
    seen: Vec<u8>,
    /// Dominator preorder numbers of the barrier-free uniform blocks with an access that a pair
    /// from a divergent block may still want a barrier before.
    open_targets: BTreeSet<u32>,
    /// Post-dominator preorder numbers of the barrier-free uniform blocks with a read that a pair
    /// may still want a barrier after.
    open_sources: BTreeSet<u32>,
    /// By post-dominator preorder number: the divergent reading blocks whose crossing block's
    /// want is not wanted yet (or that have none).
    unready: Counts,
    /// By crossing block: the post-dominator numbers of the divergent reading blocks it answers.
    waiting: BTreeMap<u32, Vec<u32>>,
    /// The last barrier block whose own end reaches its start, walked, and how.
    looped: Option<(BlockId, u8)>,
}

impl<'s, 'a> Search<'s, 'a> {
    pub(super) fn new(
        shape: &'s Shape<'a>,
        addrs: &'s [Addr],
        wants: &'s mut BTreeSet<Want>,
        m: &mut Meter,
    ) -> Result<Search<'s, 'a>, Error> {
        let acc = shape.acc;
        let regions = Regions::new(shape, m)?;
        let n = shape.e.cfg.node_count();
        let seen = filled(n, 0u8, m)?;
        let mut open_targets = BTreeSet::new();
        let mut open_sources = BTreeSet::new();
        let mut unready = Counts(filled(n + 1, 0i32, m)?);
        let mut waiting: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let entry = tree_bytes(1, std::mem::size_of::<u32>());
        for v in 0..n {
            m.tick(1)?;
            let b = block(v);
            let list = acc.in_block(b);
            if list.is_empty() {
                continue;
            }
            let reads = list.iter().any(|a| !a.write);
            let (dp, pp) = (regions.dom.pre(b), regions.pdom.pre(b));
            if shape.wg(b) {
                if acc.has_barrier(b) {
                    continue;
                }
                if dp != NONE {
                    m.charge(entry)?;
                    open_targets.insert(dp);
                }
                if reads && pp != NONE {
                    m.charge(entry)?;
                    open_sources.insert(pp);
                }
            } else if reads && pp != NONE {
                unready.add(pp, 1);
                if let Some(c) = regions.crossing(b) {
                    let list = waiting.entry(c.raw()).or_default();
                    if list.is_empty() {
                        m.charge(tree_bytes(1, std::mem::size_of::<(u32, Vec<u32>)>()))?;
                    }
                    m.grow(list, 1)?;
                    list.push(pp);
                }
            }
        }
        Ok(Search {
            shape,
            addrs,
            empty: Addr::default(),
            wants,
            added: 0,
            avoid: Avoid {
                memo: BTreeMap::new(),
            },
            regions,
            seen,
            open_targets,
            open_sources,
            unready,
            waiting,
            looped: None,
        })
    }

    /// Release the search's tables (the wants stay).
    pub(super) fn finish(self, m: &mut Meter) {
        let entry = tree_bytes(1, std::mem::size_of::<u32>());
        let waiting: usize = self
            .waiting
            .values()
            .map(|l| vec_bytes(l) + tree_bytes(1, std::mem::size_of::<(u32, Vec<u32>)>()))
            .sum();
        m.release(
            self.regions.bytes()
                + vec_bytes(&self.seen)
                + vec_bytes(&self.unready.0)
                + entry * (self.open_targets.len() + self.open_sources.len())
                + waiting,
        );
    }

    /// Try the pair `a` then `b` (indices into the accesses), `b` reached as `flags` say and
    /// inside `a`'s block when `inb`: where they conflict, want the barrier that orders them.
    pub(super) fn consider(
        &mut self,
        (ai, bi): (usize, usize),
        inb: bool,
        flags: u8,
        m: &mut Meter,
    ) -> Result<(), Error> {
        m.tick(1)?;
        let acc = self.shape.acc;
        let (Some(a), Some(b)) = (acc.list.get(ai), acc.list.get(bi)) else {
            return Ok(());
        };
        let fa = self.addrs.get(ai).unwrap_or(&self.empty);
        let fb = self.addrs.get(bi).unwrap_or(&self.empty);
        if !self
            .shape
            .conflict((a, fa), (b, fb), flags & via::CROSS != 0)
        {
            m.tick(u32::try_from(self.shape.lanes_work.take()).unwrap_or(u32::MAX))?;
            return Ok(());
        }
        m.tick(u32::try_from(self.shape.lanes_work.take()).unwrap_or(u32::MAX))?;
        let w = self.shape.place((a, b), inb, &mut self.avoid, m)?;
        if self.wants.insert(w) {
            m.charge(tree_bytes(1, std::mem::size_of::<Want>()))?;
            self.added += 1;
            self.answered(w, m);
        }
        Ok(())
    }

    /// Note a new want: the divergent reading blocks whose crossing block it covers are ready.
    fn answered(&mut self, w: Want, m: &mut Meter) {
        let c = BlockId::new(w.block);
        if w != whole(self.shape, c) {
            return;
        }
        if let Some(list) = self.waiting.remove(&w.block) {
            for &p in &list {
                self.unready.add(p, -1);
            }
            m.release(vec_bytes(&list) + tree_bytes(1, std::mem::size_of::<(u32, Vec<u32>)>()));
        }
    }

    /// How the end of block `x` reaches its start on barrier-free paths ([`via`]): across a back
    /// edge, or not at all.
    pub(super) fn self_flags(&mut self, x: BlockId, m: &mut Meter) -> Result<u8, Error> {
        let acc = self.shape.acc;
        if !acc.has_barrier(x) {
            let cyclic = self.regions.cyclic.get(x.index()).copied().unwrap_or(false);
            return Ok(if cyclic { via::CROSS } else { 0 });
        }
        if let Some((at, f)) = self.looped {
            if at == x {
                return Ok(f);
            }
        }
        let shape = self.shape;
        let cfg = &shape.e.cfg;
        let mut stack: Vec<BlockId> = Vec::new();
        let mut touched: Vec<BlockId> = Vec::new();
        let mut found = 0;
        m.grow(&mut stack, cfg.successors(x).len())?;
        stack.extend(cfg.successors(x));
        while let Some(y) = stack.pop() {
            m.tick(1)?;
            if y == x {
                found = via::CROSS;
                break;
            }
            let Some(seen) = self.seen.get_mut(y.index()) else {
                continue;
            };
            if *seen != 0 {
                continue;
            }
            *seen = via::SAME;
            m.grow(&mut touched, 1)?;
            touched.push(y);
            if !acc.has_barrier(y) {
                m.grow(&mut stack, cfg.successors(y).len())?;
                stack.extend(cfg.successors(y));
            }
        }
        for &y in &touched {
            if let Some(seen) = self.seen.get_mut(y.index()) {
                *seen = 0;
            }
        }
        m.release(vec_bytes(&stack) + vec_bytes(&touched));
        self.looped = Some((x, found));
        Ok(found)
    }

    /// The pairs from the writes `ends` of block `x` that no barrier follows in it, to the
    /// accesses of other blocks they reach.
    pub(super) fn ahead(&mut self, x: BlockId, ends: &[u32], m: &mut Meter) -> Result<(), Error> {
        self.walk(Dir::Ahead, x, ends, m)
    }

    /// The pairs into the writes `ends` of block `y` that no barrier precedes in it, from the
    /// reads of other blocks that reach them.
    pub(super) fn behind(&mut self, y: BlockId, ends: &[u32], m: &mut Meter) -> Result<(), Error> {
        self.walk(Dir::Behind, y, ends, m)
    }

    fn walk(&mut self, dir: Dir, from: BlockId, ends: &[u32], m: &mut Meter) -> Result<(), Error> {
        let shape = self.shape;
        let (cfg, acc) = (&shape.e.cfg, shape.acc);
        let uniform = dir == Dir::Ahead && shape.wg(from);
        // A state is a block and whether the path to it crossed a retreating edge ([`via`]).
        let mut queue: Vec<(BlockId, bool)> = Vec::new();
        let mut touched: Vec<BlockId> = Vec::new();
        let mut stopped: Vec<BlockId> = Vec::new();
        let edges = |at: BlockId, cross: bool, queue: &mut Vec<_>, m: &mut Meter| {
            match dir {
                Dir::Ahead => {
                    m.grow(queue, cfg.successors(at).len())?;
                    for to in cfg.successors(at) {
                        queue.push((to, cross || cfg.is_retreating(at, to)));
                    }
                }
                Dir::Behind => {
                    m.grow(queue, cfg.predecessors(at).len())?;
                    for p in cfg.predecessors(at) {
                        queue.push((p, cross || cfg.is_retreating(p, at)));
                    }
                }
            }
            Ok::<(), Error>(())
        };
        edges(from, false, &mut queue, m)?;
        let mut head = 0;
        while let Some(&(y, cross)) = queue.get(head) {
            head += 1;
            m.tick(1)?;
            let f = flag(cross);
            let Some(seen) = self.seen.get_mut(y.index()) else {
                continue;
            };
            if *seen & f != 0 {
                continue;
            }
            if *seen == 0 {
                m.grow(&mut touched, 1)?;
                touched.push(y);
            }
            *seen |= f;
            if y != from {
                let added = self.added;
                match dir {
                    Dir::Ahead => self.visit_ahead(from, ends, y, f, m)?,
                    Dir::Behind => self.visit_behind(y, from, ends, f, m)?,
                }
                if uniform && self.added != added && self.all_after(from, ends) {
                    break;
                }
            }
            if acc.has_barrier(y) {
                continue;
            }
            if y != from && !uniform && self.stops(dir, from, ends, y) {
                if let Some(seen) = self.seen.get_mut(y.index()) {
                    if *seen & STOPPED == 0 {
                        *seen |= STOPPED;
                        m.grow(&mut stopped, 1)?;
                        stopped.push(y);
                    }
                }
                continue;
            }
            edges(y, cross, &mut queue, m)?;
        }
        for &y in &stopped {
            let f = self.seen.get(y.index()).copied().unwrap_or(0) & (via::SAME | via::CROSS);
            match dir {
                Dir::Ahead => self.jump_ahead(from, ends, y, f, m)?,
                Dir::Behind => self.jump_behind(y, from, ends, f, m)?,
            }
        }
        for &y in &touched {
            if let Some(seen) = self.seen.get_mut(y.index()) {
                *seen = 0;
            }
        }
        m.release(vec_bytes(&queue) + vec_bytes(&touched) + vec_bytes(&stopped));
        Ok(())
    }

    /// Every write `ends` of uniform block `x` has its barrier right after it.
    fn all_after(&self, x: BlockId, ends: &[u32]) -> bool {
        let list = &self.shape.acc.list;
        ends.iter().all(|&i| {
            list.get(i as usize)
                .is_some_and(|a| self.wants.contains(&after(self.shape, x, a)))
        })
    }

    /// Divergent block `x`'s pairs into divergent blocks its crossing block dominates are
    /// answered there, at block `y`.
    fn crossed(&self, x: BlockId, y: BlockId) -> bool {
        self.regions.crossing(x).is_some_and(|c| {
            self.shape.e.dom.dominates(c, y) && self.wants.contains(&whole(self.shape, c))
        })
    }

    /// The pairs from the writes `ends` of `x` to the accesses of block `y`, reached as `f`.
    fn visit_ahead(
        &mut self,
        x: BlockId,
        ends: &[u32],
        y: BlockId,
        f: u8,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let shape = self.shape;
        let acc = shape.acc;
        let (wx, wy) = (shape.wg(x), shape.wg(y));
        if !wx && !wy && self.crossed(x, y) {
            return Ok(());
        }
        let start = acc.range.get(y.index()).map_or(0, |r| r.0 as usize);
        let first_bar = acc.first_barrier(y);
        for (k, b) in acc.in_block(y).iter().enumerate() {
            if first_bar.is_some_and(|p| b.pos >= p) {
                break;
            }
            if !wx && wy && self.wants.contains(&before(shape, y, b)) {
                continue;
            }
            for &ai in ends {
                let answered = wx
                    && acc
                        .list
                        .get(ai as usize)
                        .is_some_and(|a| self.wants.contains(&after(shape, x, a)));
                if !answered {
                    self.consider((ai as usize, start + k), false, f, m)?;
                }
            }
        }
        Ok(())
    }

    /// The pairs from the reads of block `w`, reached as `f`, to the writes `ends` of `y`.
    fn visit_behind(
        &mut self,
        w: BlockId,
        y: BlockId,
        ends: &[u32],
        f: u8,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let shape = self.shape;
        let acc = shape.acc;
        let (ww, wy) = (shape.wg(w), shape.wg(y));
        if !ww && !wy && self.crossed(w, y) {
            return Ok(());
        }
        let start = acc.range.get(w.index()).map_or(0, |r| r.0 as usize);
        let last_bar = acc.barriers.get(w.index()).and_then(|v| v.last().copied());
        for (k, a) in acc.in_block(w).iter().enumerate() {
            if a.write || last_bar.is_some_and(|p| a.pos < p) {
                continue;
            }
            if ww && self.wants.contains(&after(shape, w, a)) {
                continue;
            }
            for &bi in ends {
                let answered = !ww
                    && wy
                    && acc
                        .list
                        .get(bi as usize)
                        .is_some_and(|b| self.wants.contains(&before(shape, y, b)));
                if !answered {
                    self.consider((start + k, bi as usize), false, f, m)?;
                }
            }
        }
        Ok(())
    }

    /// Whether the walk from `from` stops expanding block `y` (the module documentation).
    fn stops(&self, dir: Dir, from: BlockId, ends: &[u32], y: BlockId) -> bool {
        let shape = self.shape;
        let r = &self.regions;
        let at = |v: &[bool]| v.get(y.index()).copied().unwrap_or(false);
        match dir {
            // `from`, divergent, is outside `D(y)`; every divergent block with an access in
            // `D(y)` has its pair from `from` answered by the crossing block's want.
            Dir::Ahead => {
                at(&r.ahead)
                    && !shape.e.dom.dominates(y, from)
                    && (!at(&r.ahead_divergent)
                        || r.crossing(from).is_some_and(|c| {
                            shape.e.dom.dominates(c, y) && self.wants.contains(&whole(shape, c))
                        }))
            }
            // Every path from `PD(y)` to `from` passes `y`; every divergent reading block in
            // `PD(y)` has its pairs into `from` answered.
            Dir::Behind => {
                let target = from;
                let list = &shape.acc.list;
                at(&r.behind)
                    && shape.pdom.is_reachable(target)
                    && !shape.pdom.post_dominates(y, target)
                    && (!at(&r.behind_divergent)
                        || (shape.wg(target)
                            && ends.iter().all(|&i| {
                                list.get(i as usize)
                                    .is_some_and(|b| self.wants.contains(&before(shape, target, b)))
                            }))
                        || (!shape.wg(target)
                            && self.unready.range(r.pdom.pre(y), r.pdom.end(y)) == 0
                            && r.behind_crossed.get(y.index()).is_some_and(|&(lo, hi)| {
                                let p = r.dom.pre(target);
                                (lo..hi).contains(&p)
                            })))
            }
        }
    }

    /// At block `y` where the forward walk from `x` stopped, reached as `f`: the pairs into the
    /// uniform blocks of `D(y)` whose barrier before an access is not wanted yet.
    fn jump_ahead(
        &mut self,
        x: BlockId,
        ends: &[u32],
        y: BlockId,
        f: u8,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let (lo, hi) = (self.regions.dom.pre(y), self.regions.dom.end(y));
        let mut open: Vec<u32> = Vec::new();
        if lo.saturating_add(1) >= hi {
            return Ok(());
        }
        for &p in self.open_targets.range(lo.saturating_add(1)..hi) {
            m.tick(1)?;
            m.grow(&mut open, 1)?;
            open.push(p);
        }
        for &p in &open {
            let Some(z) = self.regions.dom.at(p) else {
                continue;
            };
            self.visit_ahead(x, ends, z, f, m)?;
            let shape = self.shape;
            if shape
                .acc
                .in_block(z)
                .iter()
                .all(|b| self.wants.contains(&before(shape, z, b)))
            {
                self.open_targets.remove(&p);
                m.release(tree_bytes(1, std::mem::size_of::<u32>()));
            }
        }
        m.release(vec_bytes(&open));
        Ok(())
    }

    /// At block `w` where the backward walk from `y` stopped, reached as `f`: the pairs from the
    /// uniform blocks of `PD(w)` whose barrier after a read is not wanted yet.
    fn jump_behind(
        &mut self,
        w: BlockId,
        y: BlockId,
        ends: &[u32],
        f: u8,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let (lo, hi) = (self.regions.pdom.pre(w), self.regions.pdom.end(w));
        let mut open: Vec<u32> = Vec::new();
        if lo.saturating_add(1) >= hi {
            return Ok(());
        }
        for &p in self.open_sources.range(lo.saturating_add(1)..hi) {
            m.tick(1)?;
            m.grow(&mut open, 1)?;
            open.push(p);
        }
        for &p in &open {
            let Some(u) = self.regions.pdom.at(p) else {
                continue;
            };
            self.visit_behind(u, y, ends, f, m)?;
            let shape = self.shape;
            if shape
                .acc
                .in_block(u)
                .iter()
                .all(|a| a.write || self.wants.contains(&after(shape, u, a)))
            {
                self.open_sources.remove(&p);
                m.release(tree_bytes(1, std::mem::size_of::<u32>()));
            }
        }
        m.release(vec_bytes(&open));
        Ok(())
    }
}
