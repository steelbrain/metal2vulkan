//! P6: acyclic region structuring, Ramsey's dominator-tree placement.
//!
//! A region is the function body or one loop body. Its nodes are the blocks whose innermost loop
//! it is, plus one collapsed node per directly nested loop (P5); its edges come from
//! `Normalizer::classify`, so back edges are already `Continue` and exits `SetExit; Break`,
//! and the region is a DAG. Loops are structured innermost first, so a collapsed node's
//! `Loop { body }` is ready when its parent region needs it.
//!
//! Within a region, a **merge node** has two or more incoming region edges. For every node `X`,
//! bottom-up over the region's dominator tree:
//!
//! ```text
//! tree(X) = Seq[content(X), branch(X)] wrapped, for each merge child Y of X in ascending RPO,
//!           as Seq[Block(Y, previous), tree(Y)]
//! branch(X -> T) = Continue | [SetExit;] Break                (T outside the region graph)
//!                | tree(T)   if T is a dominator-tree child of X and not a merge (inline arm)
//!                | Jump(T)   otherwise (T is a merge child of an ancestor: an enclosing Block)
//! ```
//!
//! A collapsed loop `L` contributes `Loop { body(L) }` followed by its exit dispatch over its
//! exit targets. Every node is placed exactly once, by its immediate dominator (inline arm or
//! merge child), so every original block appears in exactly one `Code`. Ramsey's argument
//! ("Beyond Relooper", ICFP 2022) gives every `Jump(T)` an enclosing `Block(T)`: `T` is a merge
//! child of `idom(T)`, which dominates the jump site, and the jump site cannot lie in the tree of
//! a later merge child (that would make the edge retreating). The tree still holds
//! `Block`/`Jump`; P7 lowers them.
//!
//! Children are built before parents (reverse dominator-tree preorder), which matches the tree
//! arena's order. Per region the work is linear in its nodes and edges plus the dominator
//! computation and the sort of each node's merge children; each node and edge belongs to one
//! region, so the pass is O((n + e) log n) overall.

use super::cfg::Term;
use super::cfg::{at, at_mut, invariant, to_u32, with_scratch, Content, Node, Scratch, StructCfg};
use super::loops::{LoopIx, Loops, NONE};
use super::normalize::{Normalizer, Target};
use super::tree::{Label, Stmt, StmtId, StructTree};
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, DomTree, Graph};
use crate::spirv::ir::Unrolling;

/// P6's result: the tree (with `Block`/`Jump`) and the number of labels it uses.
#[derive(Debug)]
pub struct RegionTree {
    pub tree: StructTree,
    pub labels: u32,
}

/// P6 over the whole function. On error nothing stays charged to `meter`'s ledger.
pub(super) fn structure(
    sc: &StructCfg,
    loops: &Loops,
    unrolling: &[Unrolling],
    norm: &mut Normalizer,
    meter: &mut Meter,
) -> Result<RegionTree, Error> {
    let capacity = sc.len().saturating_mul(2).saturating_add(16);
    let mut tree = StructTree::with_capacity(meter, capacity)?;
    let built = with_scratch(meter, |meter, scratch| {
        let mut b = Builder {
            sc,
            loops,
            unrolling,
            local: scratch.filled(meter, sc.len(), NONE)?,
            body: scratch.filled(meter, loops.len(), None)?,
            items: Vec::new(),
            succ_start: Vec::new(),
            succ: Vec::new(),
            indeg: Vec::new(),
            tree_of: Vec::new(),
            label_of: Vec::new(),
            order: Vec::new(),
            merges: Vec::new(),
            parts: Vec::new(),
            arms: Vec::new(),
            labels: 0,
        };
        for l in loops.innermost_first() {
            let body = b.region(Some(l), norm, &mut tree, meter, scratch)?;
            *at_mut(&mut b.body, l.index())? = Some(body);
        }
        let root = b.region(None, norm, &mut tree, meter, scratch)?;
        Ok((root, b.labels))
    });
    match built.and_then(|(root, labels)| tree.set_root(root).map(|()| labels)) {
        Ok(labels) => Ok(RegionTree { tree, labels }),
        Err(e) => {
            meter.release(tree.ledger_bytes());
            Err(e)
        }
    }
}

/// A node of the region graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Item {
    Own(Node),
    Child(LoopIx),
}

/// One of two iterator types.
enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Iterator for Either<A, B> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Either::Left(a) => a.next(),
            Either::Right(b) => b.next(),
        }
    }
}

/// The node an item stands for (a collapsed loop by its header).
fn item_node(loops: &Loops, item: Item) -> Result<Node, Error> {
    match item {
        Item::Own(n) => Ok(n),
        Item::Child(c) => loops
            .header(c)
            .ok_or_else(|| invariant(format!("loop {c:?} out of range"))),
    }
}

/// The edge targets of an item, in branch order (a collapsed loop's are its exit targets).
fn item_targets<'a>(
    sc: &'a StructCfg,
    loops: &'a Loops,
    item: Item,
) -> Result<impl Iterator<Item = Node> + 'a, Error> {
    Ok(match item {
        Item::Own(n) => Either::Left(sc.block(n)?.term.targets()),
        Item::Child(c) => Either::Right(loops.exits(c)),
    })
}

/// The region graph over local indices, for the `graph` analyses.
struct RegionGraph<'a> {
    entry: u32,
    succ_start: &'a [u32],
    succ: &'a [u32],
}

impl RegionGraph<'_> {
    fn local_successors(&self, node: u32) -> &[u32] {
        let from = self.succ_start.get(node as usize).copied().unwrap_or(0) as usize;
        let to = self.succ_start.get(node as usize + 1).copied().unwrap_or(0) as usize;
        self.succ.get(from..to).unwrap_or(&[])
    }
}

impl Graph for RegionGraph<'_> {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.succ_start.len().saturating_sub(1)
    }

    fn entry(&self) -> u32 {
        self.entry
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.local_successors(node).iter().copied()
    }
}

/// What one region's placement reads.
struct Place<'r> {
    region: Option<LoopIx>,
    dom: &'r DomTree<u32>,
    cfg: &'r Cfg<u32>,
    norm: &'r Normalizer,
}

/// Per-pass tables; the local-index tables are refilled per region.
struct Builder<'a> {
    sc: &'a StructCfg,
    loops: &'a Loops,
    /// Loop -> what its `OpLoopMerge` asks of the driver's unrolling (none where absent).
    unrolling: &'a [Unrolling],
    /// Node -> local index in the region being built (own block, or a child loop's header).
    local: Vec<u32>,
    /// Loop -> its structured body, once built.
    body: Vec<Option<StmtId>>,
    items: Vec<Item>,
    succ_start: Vec<u32>,
    succ: Vec<u32>,
    indeg: Vec<u32>,
    tree_of: Vec<Option<StmtId>>,
    label_of: Vec<u32>,
    order: Vec<u32>,
    merges: Vec<u32>,
    parts: Vec<StmtId>,
    arms: Vec<StmtId>,
    labels: u32,
}

impl Builder<'_> {
    /// Structure one region and return its tree.
    fn region(
        &mut self,
        region: Option<LoopIx>,
        norm: &mut Normalizer,
        tree: &mut StructTree,
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<StmtId, Error> {
        let (sc, loops) = (self.sc, self.loops);
        norm.enter(loops, region, meter)?;
        self.collect_items(region, meter, scratch)?;
        let count = self.items.len();
        for i in 0..count {
            let n = item_node(loops, *at(&self.items, i)?)?;
            *at_mut(&mut self.local, n.index())? = to_u32(i)?;
        }
        let entry_node = match region {
            Some(r) => loops
                .header(r)
                .ok_or_else(|| invariant("region loop out of range"))?,
            None => sc.entry(),
        };
        let entry = *at(&self.local, entry_node.index())?;

        // Local edges (CSR) and in-degrees.
        scratch.refill(meter, &mut self.indeg, count, 0)?;
        self.succ_start.clear();
        self.succ.clear();
        scratch.push(meter, &mut self.succ_start, 0)?;
        for i in 0..count {
            let item = *at(&self.items, i)?;
            for t in item_targets(sc, loops, item)? {
                meter.tick(1)?;
                if let Target::Local(t) = norm.classify(loops, t)? {
                    let v = *at(&self.local, t.index())?;
                    scratch.push(meter, &mut self.succ, v)?;
                    *at_mut(&mut self.indeg, v as usize)? += 1;
                }
            }
            let end = to_u32(self.succ.len())?;
            scratch.push(meter, &mut self.succ_start, end)?;
        }

        let graph = RegionGraph {
            entry,
            succ_start: &self.succ_start,
            succ: &self.succ,
        };
        let cfg = Cfg::build(&graph, meter)?;
        let checked = check_region_dag(&graph, &cfg, count, meter);
        let result = checked
            .and_then(|()| DomTree::compute(&cfg, meter).map_err(Error::from))
            .and_then(|dom| {
                let place = Place {
                    region,
                    dom: &dom,
                    cfg: &cfg,
                    norm,
                };
                let placed = self.place(&place, entry, tree, meter, scratch);
                meter.release(dom.ledger_bytes());
                placed
            });
        meter.release(cfg.ledger_bytes());
        result
    }

    /// The region's items: own blocks, then directly nested loops.
    fn collect_items(
        &mut self,
        region: Option<LoopIx>,
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<(), Error> {
        let (sc, loops) = (self.sc, self.loops);
        self.items.clear();
        match region {
            Some(r) => {
                for n in loops.own_blocks(r) {
                    meter.tick(1)?;
                    scratch.push(meter, &mut self.items, Item::Own(n))?;
                }
                for c in loops.children(r) {
                    meter.tick(1)?;
                    scratch.push(meter, &mut self.items, Item::Child(c))?;
                }
            }
            None => {
                for (n, _) in sc.blocks() {
                    meter.tick(1)?;
                    if loops.innermost(n).is_none() {
                        scratch.push(meter, &mut self.items, Item::Own(n))?;
                    }
                }
                for c in loops.top_level() {
                    meter.tick(1)?;
                    scratch.push(meter, &mut self.items, Item::Child(c))?;
                }
            }
        }
        Ok(())
    }

    /// Build `tree(X)` for every region node, children first; return the entry's.
    fn place(
        &mut self,
        p: &Place<'_>,
        entry: u32,
        tree: &mut StructTree,
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<StmtId, Error> {
        let sc = self.sc;
        let count = self.items.len();
        scratch.refill(meter, &mut self.tree_of, count, None)?;
        scratch.refill(meter, &mut self.label_of, count, NONE)?;
        self.order.clear();
        for x in p.dom.preorder() {
            scratch.push(meter, &mut self.order, x)?;
        }
        for k in (0..self.order.len()).rev() {
            meter.tick(1)?;
            let x = *at(&self.order, k)?;
            let item = *at(&self.items, x as usize)?;

            // content(X), then branch(X).
            self.parts.clear();
            let content = match item {
                Item::Own(n) => match sc.block(n)?.content {
                    Content::Code(b) => Some(Stmt::Code(b)),
                    Content::SetSel { sel, value } => Some(Stmt::SetSel { sel, value }),
                    Content::Empty => None,
                },
                Item::Child(c) => {
                    let body = at(&self.body, c.index())?.ok_or_else(|| {
                        invariant(format!("loop {c:?} structured after its parent"))
                    })?;
                    let unrolling = self.unrolling.get(c.index()).copied().unwrap_or_default();
                    Some(Stmt::Loop { body, unrolling })
                }
            };
            if let Some(stmt) = content {
                let id = tree.push(meter, stmt)?;
                scratch.push(meter, &mut self.parts, id)?;
            }
            let branch = self.branch_stmt(p, item, x, tree, meter, scratch)?;
            scratch.push(meter, &mut self.parts, branch)?;
            let mut node = match self.parts.as_slice() {
                [only] => *only,
                parts => tree.seq(meter, parts)?,
            };

            // Merge children, innermost (smallest RPO) label first.
            self.merges.clear();
            for y in p.dom.children(x) {
                meter.tick(1)?;
                if *at(&self.indeg, y as usize)? >= 2 {
                    scratch.push(meter, &mut self.merges, y)?;
                }
            }
            self.merges
                .sort_unstable_by_key(|&y| p.cfg.rpo_number(y).unwrap_or(u32::MAX));
            for i in 0..self.merges.len() {
                let y = *at(&self.merges, i)?;
                let label = self.label(y)?;
                let block = tree.push(meter, Stmt::Block { label, body: node })?;
                let after = at(&self.tree_of, y as usize)?
                    .ok_or_else(|| invariant(format!("merge child {y} placed before its tree")))?;
                node = tree.seq(meter, &[block, after])?;
            }
            *at_mut(&mut self.tree_of, x as usize)? = Some(node);
        }
        at(&self.tree_of, entry as usize)?.ok_or_else(|| invariant("region entry not placed"))
    }

    /// The statement for X's terminator (a collapsed loop's is its exit dispatch).
    fn branch_stmt(
        &mut self,
        p: &Place<'_>,
        item: Item,
        x: u32,
        tree: &mut StructTree,
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<StmtId, Error> {
        let (sc, loops) = (self.sc, self.loops);
        let stmt = match item {
            Item::Own(n) => match &sc.block(n)?.term {
                Term::Br(t) => return self.branch(p, x, *t, tree, meter),
                Term::CondBr { cond, t, f } => {
                    let then = self.branch(p, x, *t, tree, meter)?;
                    let otherwise = self.branch(p, x, *f, tree, meter)?;
                    Stmt::If {
                        cond: *cond,
                        then,
                        otherwise,
                    }
                }
                Term::Switch {
                    sel,
                    default,
                    cases,
                } => {
                    let default = self.branch(p, x, *default, tree, meter)?;
                    self.arms.clear();
                    for (_, t) in cases.iter() {
                        let arm = self.branch(p, x, *t, tree, meter)?;
                        scratch.push(meter, &mut self.arms, arm)?;
                    }
                    return tree.switch(meter, *sel, default, cases, &self.arms);
                }
                Term::Dispatch { sel, targets } => {
                    self.arms.clear();
                    for t in targets.iter() {
                        let arm = self.branch(p, x, *t, tree, meter)?;
                        scratch.push(meter, &mut self.arms, arm)?;
                    }
                    return tree.dispatch(meter, *sel, &self.arms);
                }
                Term::Return(v) => Stmt::Return(*v),
                Term::Unreachable => Stmt::Unreachable,
            },
            Item::Child(c) => {
                let mut exits = loops.exits(c);
                match exits.len() {
                    0 => Stmt::Unreachable,
                    1 => {
                        let t = exits
                            .next()
                            .ok_or_else(|| invariant("exit list shorter than its length"))?;
                        return self.branch(p, x, t, tree, meter);
                    }
                    _ => {
                        let sel = p.norm.exit_sel(c).ok_or_else(|| {
                            invariant(format!("multi-exit loop {c:?} has no selector"))
                        })?;
                        self.arms.clear();
                        for t in exits {
                            let arm = self.branch(p, x, t, tree, meter)?;
                            scratch.push(meter, &mut self.arms, arm)?;
                        }
                        return tree.dispatch(meter, sel, &self.arms);
                    }
                }
            }
        };
        tree.push(meter, stmt)
    }

    /// `branch(X -> t)`.
    fn branch(
        &mut self,
        p: &Place<'_>,
        x: u32,
        t: Node,
        tree: &mut StructTree,
        meter: &mut Meter,
    ) -> Result<StmtId, Error> {
        meter.tick(1)?;
        let norm = p.norm;
        match norm.classify(self.loops, t)? {
            Target::Continue => tree.push(meter, Stmt::Continue),
            Target::Exit(j) => {
                let brk = tree.push(meter, Stmt::Break)?;
                match p.region.and_then(|r| norm.exit_sel(r)) {
                    None => Ok(brk),
                    Some(sel) => {
                        let set = tree.push(meter, Stmt::SetSel { sel, value: j })?;
                        tree.seq(meter, &[set, brk])
                    }
                }
            }
            Target::Local(t) => {
                let y = *at(&self.local, t.index())?;
                if p.dom.idom(y) == Some(x) && *at(&self.indeg, y as usize)? == 1 {
                    at(&self.tree_of, y as usize)?
                        .ok_or_else(|| invariant(format!("inline arm {y} not placed yet")))
                } else {
                    let label = self.label(y)?;
                    tree.push(meter, Stmt::Jump(label))
                }
            }
        }
    }

    /// The label of merge node `y`, assigned on first use.
    fn label(&mut self, y: u32) -> Result<Label, Error> {
        let slot = at_mut(&mut self.label_of, y as usize)?;
        if *slot == NONE {
            *slot = self.labels;
            self.labels = self
                .labels
                .checked_add(1)
                .filter(|&l| l != NONE)
                .ok_or_else(|| invariant("label count exceeds u32"))?;
        }
        Ok(Label::new(*slot))
    }
}

/// Every region node is reachable from the region entry and no region edge is retreating: the
/// region is a DAG entered at its entry (P1–P5 guarantee it; a failure is an internal bug).
fn check_region_dag(
    graph: &RegionGraph<'_>,
    cfg: &Cfg<u32>,
    count: usize,
    meter: &mut Meter,
) -> Result<(), Error> {
    if cfg.reachable_count() != count {
        return Err(invariant(format!(
            "{} of {count} region nodes are unreachable from the region entry",
            count.saturating_sub(cfg.reachable_count())
        )));
    }
    for u in 0..to_u32(count)? {
        for &v in graph.local_successors(u) {
            meter.tick(1)?;
            if cfg.is_retreating(u, v) {
                return Err(invariant(format!("region edge {u} -> {v} closes a cycle")));
            }
        }
    }
    Ok(())
}
