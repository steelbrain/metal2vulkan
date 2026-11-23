//! The reconvergence of an entry whose only tangled instructions that depend on reconvergence
//! stand where their invocations reach them together wherever SPIR-V reconverges them
//! (`SimdPlan::reentry`):
//!
//! - a fragment's derivatives (`dfdx`, `dfdy` and `fwidth`), implicit-LOD samples and LOD
//!   queries: Metal leaves derivatives undefined within nonuniform control flow (MSL 6.11.1.1),
//!   and an implicit level or an LOD query is the derivative of its coordinate (MSL 6.12), so such
//!   an instruction is Metal's only where the quad reaches it together;
//! - a threadgroup barrier, which Metal requires every thread of the threadgroup to execute, the
//!   same number of times (MSL 6.10.1);
//! - a kernel's instruction in a block every invocation of the workgroup executes together, the
//!   same number of times (`uniform::Uniformity`, under the kernel's unit axes): every enclosing
//!   selection's condition and every enclosing loop's exit is the same for every invocation, and
//!   no invocation skips the block by a `break`, `continue` or `return`.
//!
//! Without a reconvergence mode SPIR-V reconverges invocations at the merge of a construct they
//! entered together and all leave by that merge (SPIR-V 2.2.5, "Uniform Control Flow"), and an
//! `OpControlBarrier` of `Workgroup` execution scope is defined only where every invocation
//! executes the same dynamic instance of it (`OpControlBarrier`; invocations resume the same
//! dynamic instances only under those reconvergence conditions, 2.2.5, "Dynamic Instance").
//! That covers every place such invocations rejoin but one: a loop's continue target.
//! Invocations that reach the continue target from different blocks (a continue from inside a
//! selection, or two latches) are not known to run the next iteration together, where Metal's run
//! it as uniform control flow; `MaximallyReconvergesKHR` reconverges them there
//! (`SPV_KHR_maximal_reconvergence`, "Reconvergence"). So the entry takes the mode the profile
//! gives a tangled entry, and its `implementation_defined` (`PrecisionNote::DeviceReconvergence`
//! where the profile has no maximal reconvergence), only where its module holds a tangled
//! instruction, itself or through a call, in the construct of a loop whose continue target two or
//! more blocks of the construct branch to ([`reentered`]); elsewhere it states no mode and its
//! reconvergence is guaranteed.

use super::{ReconvergenceMode, SimdPlan};
use crate::pipeline::artifact::Reconvergence;
use crate::spirv::error::BResult;
use crate::spirv::ids::BlockId;
use crate::spirv::ir::{Block, ExecMode, Inst, Merge, Terminator, TypedModule};

/// The entry's reconvergence once `module` is emitted, declaring the deferred mode of an entry
/// whose tangled instructions lie in a re-entered loop ([`reentered`]).
pub fn settle(simd: &SimdPlan, module: &mut TypedModule) -> BResult<Reconvergence> {
    let implementation_defined = match simd.reentry {
        Some((mode, implementation_defined)) if reentered(module) => {
            let mode = match mode {
                ReconvergenceMode::None => None,
                ReconvergenceMode::Maximal => Some(ExecMode::MaximallyReconverges),
                ReconvergenceMode::UniformControlFlow => Some(ExecMode::SubgroupUniformControlFlow),
            };
            if let Some(mode) = mode {
                module.declare_modes(&[mode])?;
            }
            implementation_defined
        }
        _ => simd.implementation_defined,
    };
    Ok(if implementation_defined {
        Reconvergence::ImplementationDefined
    } else {
        Reconvergence::Guaranteed
    })
}

/// Whether `inst` is tangled: a derivative, an implicit-LOD sample, an LOD query, a control
/// barrier or a group operation.
fn tangled(inst: &Inst) -> bool {
    match inst {
        Inst::ImageSample { form, .. } => form.implicit,
        Inst::Derivative { .. }
        | Inst::ImageQueryLod { .. }
        | Inst::ControlBarrier { .. }
        | Inst::Group { .. } => true,
        _ => false,
    }
}

/// Whether `t` branches to `to`.
fn branches_to(t: &Terminator, to: BlockId) -> bool {
    match t {
        Terminator::Branch(x) => *x == to,
        Terminator::CondBranch { t, f, .. } => *t == to || *f == to,
        Terminator::Switch { default, cases, .. } => {
            *default == to || cases.iter().any(|(_, b)| *b == to)
        }
        _ => false,
    }
}

/// Whether `module` holds a tangled instruction ([`tangled`]), itself or through a call, in the
/// construct of a loop whose continue target two or more blocks of the construct branch to. A construct is the layout interval from its header to its merge, as the module's
/// verification holds it (`spirv::verify`).
pub fn reentered(module: &TypedModule) -> bool {
    let blocks = module.blocks();
    let functions = module.functions();
    let block = |b: &BlockId| blocks.get(b.index());
    // The functions that hold a tangled instruction, themselves or through a call (a monotone
    // fixpoint).
    let mut holds = vec![false; functions.len()];
    let site = |b: &Block, holds: &[bool]| {
        b.insts.iter().any(|n| match &n.inst {
            Inst::Call { func, .. } => holds.get(func.index()).copied().unwrap_or(false),
            inst => tangled(inst),
        })
    };
    let mut changed = true;
    while changed {
        changed = false;
        for (fi, f) in functions.iter().enumerate() {
            if holds.get(fi).copied().unwrap_or(true) {
                continue;
            }
            if f.blocks.iter().filter_map(block).any(|b| site(b, &holds)) {
                if let Some(x) = holds.get_mut(fi) {
                    *x = true;
                }
                changed = true;
            }
        }
    }
    functions.iter().any(|f| {
        // Each block's site, and the sites before each layout position.
        let mut before = Vec::with_capacity(f.blocks.len() + 1);
        before.push(0usize);
        for b in &f.blocks {
            let here = block(b).is_some_and(|b| site(b, &holds));
            before.push(before.last().copied().unwrap_or(0) + usize::from(here));
        }
        f.blocks.iter().enumerate().any(|(start, h)| {
            let Some(Merge::Loop { merge, cont }) = block(h).and_then(|b| b.merge) else {
                return false;
            };
            let Some(end) = f.blocks.iter().position(|b| *b == merge) else {
                return false;
            };
            let construct = f.blocks.get(start..end).unwrap_or(&[]);
            let sites =
                before.get(end).copied().unwrap_or(0) - before.get(start).copied().unwrap_or(0);
            sites > 0
                && construct
                    .iter()
                    .filter_map(block)
                    .filter(|b| branches_to(&b.term, cont))
                    .nth(1)
                    .is_some()
        })
    })
}
