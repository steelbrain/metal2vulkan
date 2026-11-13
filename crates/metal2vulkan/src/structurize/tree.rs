//! [`StructTree`] and [`Structured`]: the structurizer's output.
//!
//! The tree is an arena of [`Stmt`] nodes addressed by [`StmtId`]. Children are always created
//! before their parent, so ascending ids are a post-order (children first) and descending ids
//! visit every parent before its children. Nothing here recurses on the tree depth: a 10⁵-deep
//! nest of `If`s is walked with the arena order or an explicit stack, and dropping the arena drops
//! a flat vector.
//!
//! Every original block appears in exactly one [`Stmt::Code`] (the "no duplication" invariant).
//! Synthetic statements only write or read structurizer variables ([`SelVar`]): loop exit and
//! irreducible entry selectors, and the forward-jump `pending` variable.

use super::cfg::{invariant, to_u32, SelVar};
use super::demote::{DemoteVar, PhiVar};
use crate::air::ir::{BlockId, ValueId};
use crate::budget::Meter;
use crate::error::Error;
use crate::spirv::ir::Unrolling;

/// A statement of a [`StructTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StmtId(u32);

impl StmtId {
    /// The dense index, for side tables.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A forward-jump label: a merge block of a region. While the jump is pending, the
/// `pending` variable holds [`Label::pending_value`]; `0` means no jump is pending.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(u32);

impl Label {
    pub(super) const fn new(index: u32) -> Label {
        Label(index)
    }

    /// The dense index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The value `pending` holds while a jump to this label is pending (never `0`).
    pub const fn pending_value(self) -> u32 {
        self.0.saturating_add(1)
    }
}

/// One statement. `Block` and `Jump` exist only between P6 and P7: the output of
/// [`super::structurize`] never contains them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    /// Run the statements in order.
    Seq(Box<[StmtId]>),
    /// The original block's instructions, without its terminator.
    Code(BlockId),
    /// `sel = value`.
    SetSel {
        sel: SelVar,
        value: u32,
    },
    /// `if cond { then } else { otherwise }`, fresh merge.
    If {
        cond: ValueId,
        then: StmtId,
        otherwise: StmtId,
    },
    /// An original switch (literals grouped by target, no case to the default), fresh merge, no
    /// fallthrough.
    Switch {
        sel: ValueId,
        default: StmtId,
        cases: Box<[(Box<[u64]>, StmtId)]>,
    },
    /// `switch sel { i => arms[i] }` over a structurizer selector: a loop's exit dispatch or an
    /// irreducible loop's entry dispatch. Only values stored on the same path are read.
    Dispatch {
        sel: SelVar,
        arms: Box<[StmtId]>,
    },
    /// `loop { body }`: empty header, one continue block, fresh merge. The body never falls out;
    /// it ends every path in `Break`, `Continue`, `Return`, or `Unreachable`. `unrolling` is what
    /// its `OpLoopMerge` asks of the driver (`CfgView::unrolling`).
    Loop {
        body: StmtId,
        unrolling: Unrolling,
    },
    /// Leave the innermost loop.
    Break,
    /// Start the innermost loop's next iteration.
    Continue,
    /// P6 only: `block_label { body }`; a `Jump(label)` inside `body` continues after it.
    Block {
        label: Label,
        body: StmtId,
    },
    /// P6 only: jump to the end of the enclosing `Block` with this label.
    Jump(Label),
    /// `pending = label` (a non-tail forward jump).
    SetPending(Label),
    /// `pending = (pending == label) ? NONE : pending`, branch-free.
    ClearPending(Label),
    /// `if pending == NONE { body }`.
    GuardPending(StmtId),
    /// Return from the entry.
    Return(Option<ValueId>),
    Unreachable,
}

impl Stmt {
    /// The statements this one contains, in execution order (the arms of a branch in branch
    /// order: `If` then/otherwise, `Switch` default then cases).
    pub fn children(&self) -> impl Iterator<Item = StmtId> + '_ {
        let (single, list): (Option<StmtId>, &[StmtId]) = match self {
            Stmt::Seq(items) => (None, items),
            Stmt::Dispatch { arms, .. } => (None, arms),
            Stmt::If { then, .. } => (Some(*then), &[]),
            Stmt::Switch { default, .. } => (Some(*default), &[]),
            Stmt::Loop { body, .. } | Stmt::Block { body, .. } | Stmt::GuardPending(body) => {
                (Some(*body), &[])
            }
            _ => (None, &[]),
        };
        let otherwise = match self {
            Stmt::If { otherwise, .. } => Some(*otherwise),
            _ => None,
        };
        let cases: &[(Box<[u64]>, StmtId)] = match self {
            Stmt::Switch { cases, .. } => cases,
            _ => &[],
        };
        single
            .into_iter()
            .chain(otherwise)
            .chain(list.iter().copied())
            .chain(cases.iter().map(|(_, s)| *s))
    }

    /// Heap bytes this statement owns (charged to the ledger with it).
    fn heap_bytes(&self) -> usize {
        match self {
            Stmt::Seq(items) => std::mem::size_of_val::<[StmtId]>(items),
            Stmt::Dispatch { arms, .. } => std::mem::size_of_val::<[StmtId]>(arms),
            Stmt::Switch { cases, .. } => cases.iter().fold(
                std::mem::size_of_val::<[(Box<[u64]>, StmtId)]>(cases),
                |sum, (lits, _)| sum.saturating_add(std::mem::size_of_val::<[u64]>(lits)),
            ),
            _ => 0,
        }
    }
}

/// The structured tree of one entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructTree {
    stmts: Vec<Stmt>,
    root: StmtId,
    ledger_bytes: usize,
}

impl StructTree {
    /// An empty arena with room for `capacity` statements, charged first. The root is set by
    /// [`StructTree::set_root`].
    pub(super) fn with_capacity(meter: &mut Meter, capacity: usize) -> Result<StructTree, Error> {
        let mut tree = StructTree {
            stmts: Vec::new(),
            root: StmtId(0),
            ledger_bytes: 0,
        };
        tree.reserve(meter, capacity)?;
        Ok(tree)
    }

    fn reserve(&mut self, meter: &mut Meter, additional: usize) -> Result<(), Error> {
        let bytes = additional.saturating_mul(std::mem::size_of::<Stmt>());
        meter.charge(bytes)?;
        self.ledger_bytes = self.ledger_bytes.saturating_add(bytes);
        super::cfg::try_reserve(&mut self.stmts, additional, bytes)
    }

    /// Append a statement whose children are already in the arena, charging it first.
    pub(super) fn push(&mut self, meter: &mut Meter, stmt: Stmt) -> Result<StmtId, Error> {
        let id = StmtId(to_u32(self.stmts.len())?);
        if let Some(bad) = stmt.children().find(|c| c.0 >= id.0) {
            return Err(invariant(format!(
                "statement {id:?} refers to {bad:?}, which is not built yet"
            )));
        }
        if self.stmts.len() == self.stmts.capacity() {
            self.reserve(meter, self.stmts.capacity().max(16))?;
        }
        let heap = stmt.heap_bytes();
        meter.charge(heap)?;
        self.ledger_bytes = self.ledger_bytes.saturating_add(heap);
        self.stmts.push(stmt);
        Ok(id)
    }

    /// A `Seq` of `items`, charged first (the boxed list included).
    pub(super) fn seq(&mut self, meter: &mut Meter, items: &[StmtId]) -> Result<StmtId, Error> {
        let bytes = std::mem::size_of_val(items);
        meter.charge(bytes)?;
        let boxed: Box<[StmtId]> = Box::from(items);
        meter.release(bytes);
        self.push(meter, Stmt::Seq(boxed))
    }

    /// A `Dispatch` on `sel` with `arms`, charged first.
    pub(super) fn dispatch(
        &mut self,
        meter: &mut Meter,
        sel: SelVar,
        arms: &[StmtId],
    ) -> Result<StmtId, Error> {
        let bytes = std::mem::size_of_val(arms);
        meter.charge(bytes)?;
        let arms: Box<[StmtId]> = Box::from(arms);
        meter.release(bytes);
        self.push(meter, Stmt::Dispatch { sel, arms })
    }

    /// A `Switch` on `sel` with the literals of `cases` and the matching `arms`, charged first.
    pub(super) fn switch<T>(
        &mut self,
        meter: &mut Meter,
        sel: ValueId,
        default: StmtId,
        cases: &[(Box<[u64]>, T)],
        arms: &[StmtId],
    ) -> Result<StmtId, Error> {
        if cases.len() != arms.len() {
            return Err(invariant("switch arms do not match its cases"));
        }
        let bytes = cases.iter().fold(
            cases
                .len()
                .saturating_mul(std::mem::size_of::<(Box<[u64]>, StmtId)>()),
            |sum, (lits, _)| sum.saturating_add(std::mem::size_of_val::<[u64]>(lits)),
        );
        meter.charge(bytes)?;
        let cases: Box<[(Box<[u64]>, StmtId)]> = cases
            .iter()
            .zip(arms)
            .map(|((lits, _), &arm)| (lits.clone(), arm))
            .collect();
        meter.release(bytes);
        self.push(
            meter,
            Stmt::Switch {
                sel,
                default,
                cases,
            },
        )
    }

    /// Set the root statement.
    pub(super) fn set_root(&mut self, root: StmtId) -> Result<(), Error> {
        if root.index() >= self.stmts.len() {
            return Err(invariant(format!("root {root:?} out of range")));
        }
        self.root = root;
        Ok(())
    }

    /// The root statement.
    pub fn root(&self) -> StmtId {
        self.root
    }

    /// Statement `id`.
    pub fn stmt(&self, id: StmtId) -> Result<&Stmt, Error> {
        self.stmts
            .get(id.index())
            .ok_or_else(|| invariant(format!("statement {id:?} out of range")))
    }

    /// Every statement with its id, children before parents.
    pub fn stmts(&self) -> impl ExactSizeIterator<Item = (StmtId, &Stmt)> + DoubleEndedIterator {
        self.stmts
            .iter()
            .enumerate()
            .map(|(i, s)| (StmtId(i as u32), s))
    }

    /// The number of statements in the arena.
    pub(crate) fn len(&self) -> usize {
        self.stmts.len()
    }

    /// An upper bound on the SPIR-V blocks the lowering produces for this tree:
    /// the entry block, plus per construct its merge and one first block per arm (`If` 3,
    /// `GuardPending` 2, `Switch`/`Dispatch` 1 + arms), plus 4 per `Loop` (header, first body
    /// block, continue, merge). Every statement of the arena counts.
    pub fn lowered_blocks(&self) -> u64 {
        self.stmts.iter().fold(1u64, |blocks, stmt| {
            blocks.saturating_add(match stmt {
                Stmt::If { .. } => 3,
                Stmt::GuardPending(_) => 2,
                Stmt::Switch { cases, .. } => 2 + cases.len() as u64,
                Stmt::Dispatch { arms, .. } => 1 + arms.len() as u64,
                Stmt::Loop { .. } => 4,
                _ => 0,
            })
        })
    }

    /// Bytes this tree keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}

/// The structurizer's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Structured {
    pub tree: StructTree,
    /// One `Function` variable per block-crossing value (P4).
    pub demoted: Vec<DemoteVar>,
    /// One incoming variable per phi (P4).
    pub phis: Vec<PhiVar>,
    /// The number of structurizer `u32` selector variables (`SelVar` `0..selectors`), `pending`
    /// included.
    pub selectors: u32,
    /// The forward-jump variable, when some jump is not a tail jump (P7). It must hold `0` on
    /// entry.
    pub pending: Option<SelVar>,
    pub(super) ledger_bytes: usize,
}

impl Structured {
    /// Return the ledger bytes this result keeps charged.
    pub fn release(self, meter: &mut Meter) {
        meter.release(self.ledger_bytes);
        meter.release(self.tree.ledger_bytes());
    }
}
