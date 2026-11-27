//! [`DomTree`] (Semi-NCA) and [`PostDomTree`].

use std::marker::PhantomData;

use super::{ix, metered, to_u32, Cfg, Graph, Ledger, NodeId, Work, NONE};
use crate::budget::Meter;
use crate::error::Error;

/// The dominator tree of the reachable part of a [`Cfg`], computed by Semi-NCA (Georgiadis;
/// the variant LLVM uses): semidominators by path-compressed `eval` in reverse DFS preorder, then
/// immediate dominators as nearest common ancestors in DFS preorder.
///
/// Dominance queries are O(1): each node carries its interval in a preorder walk of the
/// dominator tree, and `a` dominates `b` iff `b`'s number lies in `a`'s interval. Unreachable
/// nodes are not in the tree: they dominate nothing and nothing dominates them.
#[derive(Clone, Debug)]
pub struct DomTree<N> {
    /// Node -> immediate dominator node (`NONE` for the root and unreachable nodes).
    idom: Vec<u32>,
    /// Node -> dominator-tree preorder number (`NONE` when unreachable).
    pre: Vec<u32>,
    /// Node -> one past the last dominator-tree preorder number of its subtree.
    end: Vec<u32>,
    /// Dominator-tree preorder number -> node.
    order: Vec<u32>,
    /// Children CSR (children in ascending CFG DFS preorder).
    child_start: Vec<u32>,
    children: Vec<u32>,
    ledger_bytes: usize,
    node: PhantomData<fn() -> N>,
}

/// Count one dominator tree ([`trees_computed`]).
#[cfg(test)]
fn note_computed() {
    TREES_COMPUTED.with(|n| n.set(n.get() + 1));
}

/// Count one tree; nothing outside the crate's own tests.
#[cfg(not(test))]
fn note_computed() {}

#[cfg(test)]
std::thread_local! {
    /// The dominator trees computed on this thread, which the tests that hold a stage to one tree
    /// a body read: a specialization must compute no tree the verification of its own pruned
    /// module already built. Per thread, so tests the harness runs side by side in one process
    /// do not count each other's trees (the crate computes every tree on its caller's thread).
    static TREES_COMPUTED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// [`TREES_COMPUTED`].
#[cfg(test)]
pub(crate) fn trees_computed() -> u64 {
    TREES_COMPUTED.with(std::cell::Cell::get)
}

impl<N: NodeId> DomTree<N> {
    /// The dominators of `cfg`, rooted at its entry. O((n + e)·α) work in practice (Semi-NCA's
    /// NCA step is O(n²) only on contrived inputs); O(n) memory beyond the `Cfg`.
    /// On error nothing stays charged to `meter`'s ledger.
    pub fn compute(cfg: &Cfg<N>, meter: &mut Meter) -> Result<DomTree<N>, Error> {
        note_computed();
        metered(meter, |meter, keep, scratch| {
            Self::compute_in(cfg, meter, keep, scratch)
        })
    }

    fn compute_in(
        cfg: &Cfg<N>,
        meter: &mut Meter,
        keep: &mut Ledger,
        scratch: &mut Ledger,
    ) -> Result<DomTree<N>, Error> {
        let mut work = Work::new();

        let count = cfg.node_count();
        let pre_of = cfg.pre_raw();
        let order = cfg.order_raw();
        let r = order.len();

        // Semi-NCA over DFS preorder numbers. `ancestor` is the compressed link forest (initially
        // the DFS-tree parent); `label` is the vertex of minimum semidominator on the compressed
        // path; `idom` starts as the DFS parent.
        let mut ancestor = scratch.with_capacity::<u32>(meter, r)?;
        ancestor.extend_from_slice(cfg.dfs_parent_raw());
        if let Some(root) = ancestor.first_mut() {
            *root = 0;
        }
        let mut idom_pre = scratch.with_capacity::<u32>(meter, r)?;
        idom_pre.extend_from_slice(&ancestor);
        let mut semi = scratch.with_capacity::<u32>(meter, r)?;
        semi.extend(0..to_u32(r, "reachable count")?);
        let mut label = scratch.with_capacity::<u32>(meter, r)?;
        label.extend_from_slice(&semi);
        let mut path = scratch.with_capacity::<u32>(meter, r)?;

        for i in (1..r).rev() {
            let w = order[i];
            semi[i] = ancestor[i];
            let last_linked = to_u32(i + 1, "preorder number")?;
            for &v in cfg.pred_raw(w) {
                work.step(meter, 1)?;
                let pv = pre_of[ix(v)];
                if pv == NONE {
                    continue;
                }
                let u = eval(
                    pv,
                    last_linked,
                    &mut ancestor,
                    &mut label,
                    &semi,
                    &mut path,
                    &mut work,
                    meter,
                )?;
                semi[i] = semi[i].min(semi[ix(u)]);
            }
        }
        for i in 1..r {
            let sdom = semi[i];
            let mut candidate = idom_pre[i];
            while candidate > sdom {
                candidate = idom_pre[ix(candidate)];
                work.step(meter, 1)?;
            }
            idom_pre[i] = candidate;
            work.step(meter, 1)?;
        }

        // Node-indexed idom, children CSR, then a preorder walk for intervals.
        let mut idom = keep.filled(meter, count, NONE)?;
        let mut child_start = keep.filled(meter, count + 1, 0u32)?;
        for i in 1..r {
            let parent = order[ix(idom_pre[i])];
            idom[ix(order[i])] = parent;
            child_start[ix(parent) + 1] += 1;
        }
        for v in 0..count {
            child_start[v + 1] += child_start[v];
        }
        let mut cursor = scratch.with_capacity::<u32>(meter, count)?;
        cursor.extend_from_slice(&child_start[..count]);
        let mut children = keep.filled(meter, r.saturating_sub(1), 0u32)?;
        for &v in order.iter().skip(1) {
            let at = &mut cursor[ix(idom[ix(v)])];
            children[ix(*at)] = v;
            *at += 1;
            work.step(meter, 1)?;
        }

        let root = cfg.entry_raw();
        let mut pre = keep.filled(meter, count, NONE)?;
        let mut end = keep.filled(meter, count, NONE)?;
        let mut tree_order = keep.with_capacity::<u32>(meter, r)?;
        let mut stack = scratch.with_capacity::<(u32, u32)>(meter, r)?;
        pre[ix(root)] = 0;
        tree_order.push(root);
        stack.push((root, child_start[ix(root)]));
        while let Some(top) = stack.last_mut() {
            let (v, at) = *top;
            if at < child_start[ix(v) + 1] {
                top.1 += 1;
                let c = children[ix(at)];
                pre[ix(c)] = to_u32(tree_order.len(), "preorder number")?;
                tree_order.push(c);
                stack.push((c, child_start[ix(c)]));
            } else {
                stack.pop();
                end[ix(v)] = to_u32(tree_order.len(), "preorder number")?;
            }
            work.step(meter, 1)?;
        }

        drop((ancestor, idom_pre, semi, label, path, cursor, stack));
        work.finish(meter)?;
        Ok(DomTree {
            idom,
            pre,
            end,
            order: tree_order,
            child_start,
            children,
            ledger_bytes: keep.bytes,
            node: PhantomData,
        })
    }

    /// Whether `node` is in the tree (reachable from the entry).
    pub fn is_reachable(&self, node: N) -> bool {
        self.pre[ix(node.node_index())] != NONE
    }

    /// The immediate dominator of `node`; `None` for the root and unreachable nodes.
    pub fn idom(&self, node: N) -> Option<N> {
        let d = self.idom[ix(node.node_index())];
        (d != NONE).then(|| N::from_node_index(d))
    }

    /// Whether `a` dominates `b` (reflexive). False when either is unreachable. O(1).
    pub fn dominates(&self, a: N, b: N) -> bool {
        let a = ix(a.node_index());
        let pb = self.pre[ix(b.node_index())];
        let pa = self.pre[a];
        pa != NONE && pb != NONE && pa <= pb && pb < self.end[a]
    }

    /// Whether `a` dominates `b` and `a != b`. O(1).
    pub fn strictly_dominates(&self, a: N, b: N) -> bool {
        a != b && self.dominates(a, b)
    }

    /// The children of `node` in the tree, in ascending CFG DFS preorder.
    pub fn children(&self, node: N) -> impl ExactSizeIterator<Item = N> + '_ {
        let v = ix(node.node_index());
        self.children[ix(self.child_start[v])..ix(self.child_start[v + 1])]
            .iter()
            .map(|&c| N::from_node_index(c))
    }

    /// Reachable nodes in a preorder walk of the tree (parents before children).
    pub fn preorder(&self) -> impl ExactSizeIterator<Item = N> + '_ {
        self.order.iter().map(|&v| N::from_node_index(v))
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}

/// Semi-NCA `eval`: the vertex of minimum semidominator on the linked path from `v` towards the
/// root, compressing the path. `last_linked` is the lowest preorder number already processed;
/// vertices below it are not linked yet and are their own label.
fn eval(
    v: u32,
    last_linked: u32,
    ancestor: &mut [u32],
    label: &mut [u32],
    semi: &[u32],
    path: &mut Vec<u32>,
    work: &mut Work,
    meter: &mut Meter,
) -> Result<u32, Error> {
    if ancestor[ix(v)] < last_linked {
        return Ok(label[ix(v)]);
    }
    // Collect the linked path from `v` up to (not including) the first vertex whose ancestor is
    // unlinked. `path` has capacity r and every vertex appears on it at most once.
    path.clear();
    let mut x = v;
    loop {
        path.push(x);
        x = ancestor[ix(x)];
        work.step(meter, 1)?;
        if ancestor[ix(x)] < last_linked {
            break;
        }
    }
    // Compress: every vertex on the path now links to `x`'s ancestor, and carries the label of
    // minimum semidominator seen from the top down.
    let mut p = x;
    let mut p_label = label[ix(p)];
    while let Some(y) = path.pop() {
        ancestor[ix(y)] = ancestor[ix(p)];
        let y_label = label[ix(y)];
        if semi[ix(p_label)] < semi[ix(y_label)] {
            label[ix(y)] = p_label;
        } else {
            p_label = y_label;
        }
        p = y;
        work.step(meter, 1)?;
    }
    Ok(label[ix(v)])
}

/// The post-dominator tree: the dominator tree of the reverse graph rooted at a virtual exit
/// whose predecessors are the reachable nodes without successors.
///
/// Only nodes that are reachable from the entry **and** reach an exit are in the tree. A node
/// that cannot reach an exit (it sits in or before an infinite loop only) has no post-dominator
/// information: [`PostDomTree::is_reachable`] is false for it and it post-dominates nothing. The
/// virtual exit is not a node: a node whose immediate post-dominator would be the virtual exit
/// reports `None`.
#[derive(Clone, Debug)]
pub struct PostDomTree<N> {
    /// Dominator tree of the reverse graph; node `virtual_exit` is the virtual root.
    tree: DomTree<u32>,
    virtual_exit: u32,
    node: PhantomData<fn() -> N>,
}

/// The reverse of a `Cfg`'s reachable part plus a virtual root `n` whose successors are the exits.
struct ReverseWithExit<'a, N> {
    cfg: &'a Cfg<N>,
    exits: &'a [u32],
}

impl<N: NodeId> Graph for ReverseWithExit<'_, N> {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.cfg.node_count() + 1
    }

    fn entry(&self) -> u32 {
        // `node_count` fits u32 with room for the virtual exit: checked in `PostDomTree::compute`.
        self.cfg.node_count() as u32
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let (exits, preds): (&[u32], &[u32]) = if ix(node) == self.cfg.node_count() {
            (self.exits, &[])
        } else {
            (&[], self.cfg.pred_raw(node))
        };
        let pre = self.cfg.pre_raw();
        exits
            .iter()
            .copied()
            .chain(preds.iter().copied().filter(move |&p| pre[ix(p)] != NONE))
    }
}

impl<N: NodeId> PostDomTree<N> {
    /// The post-dominators of the reachable part of `cfg`. Same cost as [`DomTree::compute`] plus
    /// one reversed `Cfg`, which is released before returning.
    /// On error nothing stays charged to `meter`'s ledger.
    pub fn compute(cfg: &Cfg<N>, meter: &mut Meter) -> Result<PostDomTree<N>, Error> {
        metered(meter, |meter, _keep, scratch| {
            Self::compute_in(cfg, meter, scratch)
        })
    }

    /// The tree keeps its own charge (`DomTree::ledger_bytes`), so nothing goes to `keep`.
    fn compute_in(
        cfg: &Cfg<N>,
        meter: &mut Meter,
        scratch: &mut Ledger,
    ) -> Result<PostDomTree<N>, Error> {
        let virtual_exit = to_u32(cfg.node_count() + 1, "node count with the virtual exit")? - 1;
        let mut exits = scratch.with_capacity::<u32>(meter, cfg.reachable_count())?;
        let mut work = Work::new();
        for &v in cfg.order_raw() {
            if cfg.succ_raw(v).is_empty() {
                exits.push(v);
            }
            work.step(meter, 1)?;
        }
        work.finish(meter)?;
        let reverse = Cfg::<u32>::build(&ReverseWithExit { cfg, exits: &exits }, meter)?;
        let tree = DomTree::compute(&reverse, meter);
        meter.release(reverse.ledger_bytes());
        drop(reverse);
        drop(exits);
        Ok(PostDomTree {
            tree: tree?,
            virtual_exit,
            node: PhantomData,
        })
    }

    /// Whether `node` is in the tree: reachable from the entry and able to reach an exit.
    pub fn is_reachable(&self, node: N) -> bool {
        self.tree.is_reachable(node.node_index())
    }

    /// The immediate post-dominator of `node`; `None` when it is the virtual exit or `node` is
    /// not in the tree.
    pub fn ipdom(&self, node: N) -> Option<N> {
        self.tree
            .idom(node.node_index())
            .filter(|&p| p != self.virtual_exit)
            .map(N::from_node_index)
    }

    /// Whether `a` post-dominates `b` (reflexive). False when either is not in the tree. O(1).
    pub fn post_dominates(&self, a: N, b: N) -> bool {
        self.tree.dominates(a.node_index(), b.node_index())
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.tree.ledger_bytes()
    }
}
