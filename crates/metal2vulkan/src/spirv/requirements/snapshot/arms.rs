//! Which branches on a snapshot slot's fetched value keep the slot: the selections whose arms hold
//! nothing a helper invocation's path through them could pass on (the module docs of
//! [`super`]).
//!
//! A selection's arms are the blocks a walk from its header's targets reaches before its merge
//! block, and the blocks of every function a call in them reaches. The branch keeps the slot only
//! where the walk leaves the arms through the merge alone: no arm block is the merge or continue
//! target of a construct the selection lies in (a loop's break or continue, a switch's break),
//! returns, ends the invocation (`OpKill`, `OpTerminateInvocation`) or is unreachable; and where
//! no arm block or callee holds a tangled instruction (`Inst::is_tangled`: an implicit-level
//! sample or level-of-detail query, a derivative, a subgroup or quad operation), a loop, or an
//! operand an instruction's behavior is undefined for at some value that is not a constant
//! (an integer divisor, a signed one that may be `-1`, a dynamic vector component index).
//!
//! The merge then post-dominates the header, so whatever follows it runs in a helper as in a
//! fragment, and what the arms leave behind is all a helper's other path can change: the merge's
//! phis and the `Function`, `Private` and `Output` variables an arm or its callees store to. Those
//! take the branch's reach ([`Arms::control`]), and the walk judges their uses as any other.

use crate::error::Error;
use crate::spirv::ids::BlockId;
use crate::spirv::ir::{BinOp, Block, Inst, Merge, Terminator, TypedModule, ValueOrigin};

use super::{lanes_all, NONE};

/// A block declared by more than one header.
const MANY: u32 = u32::MAX - 1;

/// A header's verdict, before it is judged.
const UNJUDGED: u8 = 0;
/// A header whose arms keep the slot.
const KEPT: u8 = 1;
/// A header whose arms leave it out.
const LEFT_OUT: u8 = 2;

/// The tables of the walks over a module's selections, one entry for each block or function.
pub(super) struct Arms {
    /// The header that declares each block its merge or continue target, `NONE` for none, `MANY`
    /// for more than one.
    declared: Vec<u32>,
    /// The walk that last reached each block, and each function.
    walked: Vec<u32>,
    walked_fn: Vec<u32>,
    walks: u32,
    /// Each header's verdict.
    verdict: Vec<u8>,
    /// The reach each kept header has handed its arms.
    applied: Vec<u128>,
    /// Each block's reach of the branches it is control dependent on: the slots whose fetched
    /// values decide a kept branch whose arms hold it.
    control: Vec<u128>,
    /// The blocks a walk reached, and its open blocks and functions.
    reached: Vec<u32>,
    stack: Vec<BlockId>,
    calls: Vec<u32>,
}

impl Arms {
    /// The bytes the tables take for a module of `blocks` blocks and `functions` functions.
    pub(super) fn bytes(blocks: usize, functions: usize) -> usize {
        blocks * (4 + 4 + 1 + 16 + 16 + 4 + 4) + functions * (4 + 4)
    }

    /// The tables of module `m`.
    pub(super) fn new(m: &TypedModule) -> Arms {
        let blocks = m.blocks().len();
        let mut declared = vec![NONE; blocks];
        for (at, b) in m.blocks().iter().enumerate() {
            let targets = match b.merge {
                Some(Merge::Selection { merge }) => [Some(merge), None],
                Some(Merge::Loop { merge, cont }) => [Some(merge), Some(cont)],
                _ => [None, None],
            };
            for t in targets.into_iter().flatten() {
                if let Some(d) = declared.get_mut(t.index()) {
                    *d = if *d == NONE { at as u32 } else { MANY };
                }
            }
        }
        Arms {
            declared,
            walked: vec![0; blocks],
            walked_fn: vec![0; m.functions().len()],
            walks: 0,
            verdict: vec![UNJUDGED; blocks],
            applied: vec![0; blocks],
            control: vec![0; blocks],
            reached: Vec::with_capacity(blocks),
            stack: Vec::with_capacity(blocks),
            calls: Vec::with_capacity(m.functions().len()),
        }
    }

    /// The reach of the kept branches block `at` is control dependent on.
    pub(super) fn control(&self, at: usize) -> u128 {
        self.control.get(at).copied().unwrap_or(0)
    }

    /// Whether the branch ending block `header`, whose condition holds the fetched values of the
    /// slots of `reach`, keeps them: its merge block where it does, which takes `reach` into its
    /// phis, its arms taking it into [`Arms::control`] (and whether that grew); `None` where it
    /// leaves them out.
    pub(super) fn keeps(
        &mut self,
        m: &TypedModule,
        header: usize,
        reach: u128,
    ) -> Result<Option<(BlockId, bool)>, Error> {
        let verdict = self.verdict.get(header).copied().unwrap_or(LEFT_OUT);
        let applied = self.applied.get(header).copied().unwrap_or(0);
        let merge = match verdict {
            LEFT_OUT => return Ok(None),
            KEPT if applied | reach == applied => {
                return Ok(merge_of(m, header).map(|b| (b, false)))
            }
            _ => self.walk(m, header)?,
        };
        let Some(merge) = merge else {
            if let Some(v) = self.verdict.get_mut(header) {
                *v = LEFT_OUT;
            }
            return Ok(None);
        };
        if let Some(v) = self.verdict.get_mut(header) {
            *v = KEPT;
        }
        if let Some(a) = self.applied.get_mut(header) {
            *a |= reach;
        }
        let mut grew = false;
        for &b in &self.reached {
            if let Some(c) = self.control.get_mut(b as usize) {
                grew |= *c | reach != *c;
                *c |= reach;
            }
        }
        Ok(Some((merge, grew)))
    }

    /// Walk the arms of the selection block `header` heads, recording the blocks reached in
    /// `reached`: its merge block where the arms keep the slot, else `None`.
    fn walk(&mut self, m: &TypedModule, header: usize) -> Result<Option<BlockId>, Error> {
        let Some(merge) = merge_of(m, header) else {
            return Ok(None);
        };
        let blocks = m.blocks();
        self.walks += 1;
        let w = self.walks;
        self.reached.clear();
        self.stack.clear();
        self.calls.clear();
        if let Some(s) = self.walked.get_mut(header) {
            *s = w;
        }
        let Some(head) = blocks.get(header) else {
            return Ok(None);
        };
        targets(&head.term, |t| self.stack.push(t));
        while let Some(b) = self.stack.pop() {
            let at = b.index();
            if b == merge || (self.walked.get(at) == Some(&w) && at != header) {
                continue;
            }
            // Back at the header, or at a merge or continue target of a construct the selection
            // lies in: the arm leaves it.
            let declared = self.declared.get(at).copied().unwrap_or(MANY);
            let inner = declared == NONE || self.walked.get(declared as usize) == Some(&w);
            if at == header || !inner {
                return Ok(None);
            }
            if let Some(s) = self.walked.get_mut(at) {
                *s = w;
            }
            let Some(block) = blocks.get(at) else {
                return Ok(None);
            };
            if !self.clean(m, block, w)? {
                return Ok(None);
            }
            match block.term {
                Terminator::Branch(_)
                | Terminator::CondBranch { .. }
                | Terminator::Switch { .. } => {}
                _ => return Ok(None),
            }
            self.reached.push(at as u32);
            targets(&block.term, |t| self.stack.push(t));
        }
        // Every function a call in the arms reaches runs in them; its returns go back to them.
        while let Some(f) = self.calls.pop() {
            let Some(func) = m.functions().get(f as usize) else {
                return Ok(None);
            };
            for &b in &func.blocks {
                let Some(block) = blocks.get(b.index()) else {
                    return Ok(None);
                };
                if !self.clean(m, block, w)? {
                    return Ok(None);
                }
                match block.term {
                    Terminator::Branch(_)
                    | Terminator::CondBranch { .. }
                    | Terminator::Switch { .. }
                    | Terminator::Return
                    | Terminator::ReturnValue(_) => {}
                    _ => return Ok(None),
                }
                self.reached.push(b.index() as u32);
            }
        }
        Ok(Some(merge))
    }

    /// Whether block `block` of walk `w` heads no loop and holds no tangled instruction and no
    /// operand undefined at some value that is not a constant; each function it calls is queued.
    fn clean(&mut self, m: &TypedModule, block: &Block, w: u32) -> Result<bool, Error> {
        if matches!(block.merge, Some(Merge::Loop { .. })) {
            return Ok(false);
        }
        for node in &block.insts {
            let inst = &node.inst;
            if inst.is_tangled() {
                return Ok(false);
            }
            let defined = match inst {
                Inst::Binary { op, b, .. } => match op {
                    BinOp::UDiv | BinOp::UMod => lanes_all(m, *b, |bits, _| bits != 0)?,
                    BinOp::SDiv | BinOp::SRem | BinOp::SMod => {
                        lanes_all(m, *b, |bits, ones| bits != 0 && bits != ones)?
                    }
                    _ => true,
                },
                Inst::VectorExtractDynamic { index, .. }
                | Inst::VectorInsertDynamic { index, .. } => {
                    matches!(m.value(*index)?.origin, ValueOrigin::Const(_))
                }
                Inst::Call { func, .. } => {
                    if let Some(s) = self.walked_fn.get_mut(func.index()).filter(|s| **s != w) {
                        *s = w;
                        self.calls.push(func.index() as u32);
                    }
                    true
                }
                _ => true,
            };
            if !defined {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The merge block of the selection block `header` heads, if it heads one.
fn merge_of(m: &TypedModule, header: usize) -> Option<BlockId> {
    match m.blocks().get(header)?.merge {
        Some(Merge::Selection { merge }) => Some(merge),
        _ => None,
    }
}

/// Call `f` on each block `term` may branch to.
fn targets(term: &Terminator, mut f: impl FnMut(BlockId)) {
    match term {
        Terminator::Branch(t) => f(*t),
        Terminator::CondBranch { t, f: e, .. } => {
            f(*t);
            f(*e);
        }
        Terminator::Switch { default, cases, .. } => {
            f(*default);
            for &(_, t) in cases {
                f(t);
            }
        }
        _ => {}
    }
}
