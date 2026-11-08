//! Which own-pixel texture slots a fragment module can read through a snapshot an earlier draw of
//! the same primitives took (`StageInterface::snapshot_textures`).
//!
//! A runtime may serve a draw that reads a texture its own render pass renders, at the fragment's
//! own pixel alone (`own_pixel`), by drawing the draw twice with one pipeline: first with every
//! color write mask zero, each fragment copying the texel it would read into a snapshot image, then
//! with the draw's own masks, each fragment reading the snapshot (the runtime's
//! `texture/attached_reads.rs`). A helper invocation reads nothing there: its pixel may hold no
//! fragment of the draw, so no copy wrote its snapshot texel, and the by-region barriers that order
//! a copy before the reads order it for fragments of the same pixel alone. So the slot's value in a
//! helper invocation is a constant, which is exact only where no fragment's result depends on a
//! helper's value of it, and only where the constant gives the helper no undefined behavior.
//!
//! A helper invocation's values reach other invocations' results only through derivatives and quad
//! operations ("When performing derivative or quad group operations in a fragment shader,
//! additional invocations may be spawned in order to ensure correct results", the Shaders
//! chapter's Helper Invocations), and its stores have no effect ("Stores and atomics performed by
//! helper invocations must not have any effect on memory except for the Function, Private and
//! Output storage classes"). A value that decides a branch can also move a helper off its quad's
//! path, which changes what a derivative there reads, or keep it in a loop the texture's own texel
//! would end. And where a helper's constant is an operand that an instruction's behavior is
//! undefined for, the helper has undefined behavior that Metal's read of the texel need not give
//! it: an integer division's ("Behavior is undefined if any component of Operand 2 is 0", the
//! SPIR-V specification's `OpUDiv`, `OpUMod`; `OpSDiv`, `OpSRem` and `OpSMod` also at `-1` beside
//! the minimum) and a dynamic component index's ("Behavior is undefined if Index's value is less
//! than zero or greater than or equal to the number of components in Vector",
//! `OpVectorExtractDynamic`, `OpVectorInsertDynamic`). So a slot qualifies only where no value that
//! a fetch of it reaches, through any instruction, phi, or `Function`, `Private` or `Output`
//! variable it is stored in and loaded from, is an operand of:
//!
//! - a derivative, a sample, gather or level-of-detail query of any image, a subgroup or quad
//!   operation, an atomic, an image write or texel pointer, a ray query, a mesh output count, or a
//!   store to any other storage class;
//! - a conditional branch, a switch, or the entry's returned value, but for a branch that only
//!   decides whether the invocation demotes itself (`if (c) discard_fragment();` at MSL 2.3 or
//!   later on a profile with demote): one arm holds `OpDemoteToHelperInvocation` alone and goes on
//!   to the other, where both paths meet with the same values, so a helper, which is one already,
//!   takes either path to the same state ("is not considered a flow control instruction",
//!   `OpDemoteToHelperInvocation`);
//! - the divisor of an integer division or remainder, the dividend of a signed one whose divisor
//!   is not a constant without a `-1` lane, or a dynamic vector component index.
//!
//! A call is followed into its callee: "Argument N is the object to copy to parameter N of
//! Function" (`OpFunctionCall`), so each argument's reach is its parameter's, a pointer argument's
//! memory is its parameter's (what either side stores through it the other may load), and the
//! values the callee returns reach the call's result. The callee's instructions are the module's,
//! judged as any other; a callee called from several sites takes the reach of them all. Lowering's
//! texel routines (Metal's conversion of a `half` texel, `lower::texture::half`) are such calls,
//! whose selects carry the texel to the result alone.
//!
//! A module that already holds an interlock critical section has none: the variant's entry opens
//! its own. Nor has one with a `Coherent` resource or under the Vulkan memory model: the variant
//! decorates its snapshot `Coherent`. The test is structural, never a name.
//!
//! [`snapshot_copies_alone`] checks a variant compiled to read such slots through snapshots: its
//! only texel access of each is the entry's copy.

use crate::budget::Meter;
use crate::error::Error;
use crate::pipeline::sym::ResourceKind;
use crate::spirv::ids::{BlockId, Value};
use crate::spirv::ir::{
    BinOp, ConstKind, Constant, ExecModel, GlobalKind, Inst, Phi, StorageClass, Terminator, Type,
    TypedModule, UnOp, ValueOrigin,
};

use super::own_pixel::{carried_on, holds_image, use_of, Use};

/// The most passes over the module the reach is followed for; a module whose values take more to
/// settle (a chain of loop-carried copies longer than this) has no slot.
const MOST_PASSES: u32 = 16;

/// No slot's image.
const NONE: u32 = u32::MAX;

#[cfg(test)]
mod tests;

/// The slots among `own` (fragment module `m`'s own-pixel texture slots, `own_pixel_textures`)
/// whose fetched values reach none of the uses the module docs list, sorted; empty for a module
/// of another stage, one with an interlock, or a slot at or past 128. It charges `meter` for its
/// tables while it holds them.
pub fn snapshot_textures(
    m: &TypedModule,
    own: &[u32],
    meter: &mut Meter,
) -> Result<Vec<u32>, Error> {
    if own.is_empty() || m.entry().model != ExecModel::Fragment {
        return Ok(Vec::new());
    }
    let interlocked = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .any(|n| matches!(n.inst, Inst::BeginInterlock | Inst::EndInterlock));
    // The variant decorates its snapshot `Coherent`, which the Vulkan memory model bans, beside
    // which no other resource of the function may be coherent (`lower::sync::memory_model`).
    let coherent = !m.coherent.is_empty() || super::uses_vulkan_memory_model(m.requirements());
    if interlocked || coherent {
        return Ok(Vec::new());
    }
    let own: Vec<u32> = own.iter().copied().filter(|&s| s < 128).collect();
    if own.is_empty() {
        return Ok(Vec::new());
    }
    let n = m.values().len();
    // Each value's slot image (`NONE` for none), its reach (the slots whose fetched values it
    // holds), the reach of the memory behind it where it is a variable's pointer, and the variable
    // a pointer points into; each block's function, and the reach each function returns.
    let tables = n * (4 + 16 + 16 + 4) + 4 * m.blocks().len() + 16 * m.functions().len();
    meter.charge(tables)?;
    let result = reach(m, &own, n);
    meter.release(tables);
    result
}

/// [`snapshot_textures`] once its tables are charged.
fn reach(m: &TypedModule, own: &[u32], n: usize) -> Result<Vec<u32>, Error> {
    let mut carried = vec![NONE; n];
    let mut base: Vec<u32> = (0..n as u32).collect();
    for g in m.globals() {
        if let GlobalKind::Resource(slot) = &g.kind {
            let texture = match slot.kind {
                ResourceKind::Texture
                | ResourceKind::CombinedTextureSampler(_)
                | ResourceKind::YCbCrPlanes(_) => Some(slot.index),
                ResourceKind::SeparateTexture(_) => slot.separate_texture_of().map(|(t, _)| t),
                _ => None,
            };
            if let (Some(t), Some(c)) = (texture, carried.get_mut(g.value.index())) {
                *c = t;
            }
        }
    }
    let mut mask = vec![0u128; n];
    let mut memory = vec![0u128; n];
    let mut function = vec![NONE; m.blocks().len()];
    for (i, f) in m.functions().iter().enumerate() {
        for b in &f.blocks {
            if let Some(at) = function.get_mut(b.index()) {
                *at = i as u32;
            }
        }
    }
    let mut returned = vec![0u128; m.functions().len()];
    let wanted: u128 = own.iter().fold(0, |acc, &s| acc | (1u128 << s));
    let mut bad: u128 = 0;
    let get = |table: &[u128], v: Value| table.get(v.index()).copied().unwrap_or(0);
    for _ in 0..MOST_PASSES {
        let mut changed = false;
        for (at, b) in m.blocks().iter().enumerate() {
            for phi in &b.phis {
                let reach = phi
                    .incoming
                    .iter()
                    .fold(0, |acc, (v, _)| acc | get(&mask, *v));
                changed |= join(&mut mask, phi.result, reach);
            }
            for node in &b.insts {
                let mut operands: u128 = 0;
                node.inst.for_each_value(|v| operands |= get(&mask, v));
                match &node.inst {
                    // What carries a slot's image on, and the variable a pointer points into.
                    inst if carried_on(inst).is_some() => {
                        let (Some(r), Some(from)) = (node.result, carried_on(inst)) else {
                            continue;
                        };
                        if holds_image(m, r)? {
                            let slot = carried.get(from.index()).copied().unwrap_or(NONE);
                            if let Some(c) = carried.get_mut(r.index()).filter(|c| **c != slot) {
                                *c = slot;
                                changed = true;
                            }
                        }
                        let mut reach = operands;
                        if let Inst::Load { ptr, .. } = inst {
                            reach |= get(&memory, root(&base, *ptr));
                        }
                        if let Inst::Unary {
                            op: UnOp::CopyObject,
                            a,
                        } = inst
                        {
                            changed |= point(&mut base, r, *a);
                        }
                        changed |= join(&mut mask, r, reach);
                    }
                    Inst::ImageFetch { image, .. } => {
                        let slot = carried.get(image.index()).copied().unwrap_or(NONE);
                        let source = match slot {
                            s if s < 128 && wanted & (1u128 << s) != 0 => 1u128 << s,
                            _ => 0,
                        };
                        if let Some(r) = node.result {
                            changed |= join(&mut mask, r, operands | source);
                        }
                    }
                    Inst::AccessChain { base: from, .. } => {
                        if let Some(r) = node.result {
                            changed |= point(&mut base, r, *from);
                            changed |= join(&mut mask, r, operands);
                        }
                    }
                    Inst::Store { ptr, value, .. } => {
                        let reach = get(&mask, *value) | get(&mask, *ptr);
                        if kept_class(m, *ptr)? {
                            let root = root(&base, *ptr);
                            changed |= join(&mut memory, root, reach);
                        } else {
                            bad |= reach;
                        }
                    }
                    // Each argument to its parameter, a pointer argument's memory both ways, and the
                    // callee's returned reach to the result.
                    Inst::Call { func, args } => {
                        let params = m.functions().get(func.index()).map(|f| f.params.as_slice());
                        let Some(params) = params.filter(|p| p.len() == args.len()) else {
                            bad |= operands;
                            continue;
                        };
                        for (&arg, &param) in args.iter().zip(params) {
                            let reach = get(&mask, arg);
                            changed |= join(&mut mask, param, reach);
                            if matches!(m.types().get(m.value(arg)?.ty)?, Type::Pointer { .. }) {
                                let at = root(&base, arg);
                                let (caller, callee) = (get(&memory, at), get(&memory, param));
                                changed |= join(&mut memory, param, caller);
                                changed |= join(&mut memory, at, callee);
                            }
                        }
                        if let Some(r) = node.result {
                            let back = returned.get(func.index()).copied().unwrap_or(0);
                            changed |= join(&mut mask, r, back);
                        }
                    }
                    // A divisor, and a signed dividend beside a divisor that may be `-1`.
                    Inst::Binary {
                        op:
                            op @ (BinOp::UDiv | BinOp::SDiv | BinOp::UMod | BinOp::SRem | BinOp::SMod),
                        a,
                        b,
                        ..
                    } => {
                        bad |= get(&mask, *b);
                        let unsigned = matches!(op, BinOp::UDiv | BinOp::UMod);
                        if !unsigned && !never_minus_one(m, *b)? {
                            bad |= get(&mask, *a);
                        }
                        if let Some(r) = node.result {
                            changed |= join(&mut mask, r, operands);
                        }
                    }
                    Inst::VectorExtractDynamic { index, .. }
                    | Inst::VectorInsertDynamic { index, .. } => {
                        bad |= get(&mask, *index);
                        if let Some(r) = node.result {
                            changed |= join(&mut mask, r, operands);
                        }
                    }
                    Inst::Derivative { .. }
                    | Inst::ImageSample { .. }
                    | Inst::ImageGather { .. }
                    | Inst::ImageDrefGather { .. }
                    | Inst::ImageQueryLod { .. }
                    | Inst::Group { .. }
                    | Inst::Atomic { .. }
                    | Inst::AtomicLoad { .. }
                    | Inst::AtomicStore { .. }
                    | Inst::AtomicCompareExchange { .. }
                    | Inst::ImageWrite { .. }
                    | Inst::ImageTexelPointer { .. }
                    | Inst::RayQuery(_)
                    | Inst::SetMeshOutputs { .. } => {
                        bad |= operands;
                        if let Some(r) = node.result {
                            changed |= join(&mut mask, r, operands);
                        }
                    }
                    _ => {
                        if let Some(r) = node.result {
                            changed |= join(&mut mask, r, operands);
                        }
                    }
                }
            }
            match &b.term {
                // A branch whose only effect is whether the invocation demotes itself reaches
                // nothing a helper's value can change.
                Terminator::CondBranch { t, f, .. } if demote_guard(m, at, *t, *f) => {}
                Terminator::CondBranch { cond: v, .. } | Terminator::Switch { sel: v, .. } => {
                    bad |= get(&mask, *v)
                }
                // A callee's returned value reaches its calls' results; the entry's, the stage.
                Terminator::ReturnValue(v) => {
                    let f = function.get(at).copied().unwrap_or(NONE);
                    let reach = get(&mask, *v);
                    match f {
                        f if f == m.entry().func.index() as u32 => bad |= reach,
                        f => match returned.get_mut(f as usize) {
                            Some(r) if *r | reach != *r => {
                                *r |= reach;
                                changed = true;
                            }
                            Some(_) => {}
                            None => bad |= reach,
                        },
                    }
                }
                _ => {}
            }
        }
        if !changed {
            return Ok(own
                .iter()
                .copied()
                .filter(|&s| bad & (1u128 << s) == 0)
                .collect());
        }
    }
    Ok(Vec::new())
}

/// Whether module `m`, compiled to read texture slots `slots` through snapshots
/// (`LoweringInputs::snapshot_textures`), reads each slot's texels in the entry's copy alone,
/// with the one `OpImageFetch` of the slot's image that copy makes. Every other use of the slot's
/// images is one that carries them
/// on or queries their size. A read the lowering left on the texture would race the draw's own
/// color writes, and shows here as a fetch the copy does not make. It charges `meter` for its
/// table while it holds it.
pub fn snapshot_copies_alone(
    m: &TypedModule,
    slots: &[u32],
    meter: &mut Meter,
) -> Result<bool, Error> {
    let n = m.values().len();
    let table = 4 * n;
    meter.charge(table)?;
    let result = copies_alone(m, slots, n);
    meter.release(table);
    result
}

/// [`snapshot_copies_alone`] once its table is charged.
fn copies_alone(m: &TypedModule, slots: &[u32], n: usize) -> Result<bool, Error> {
    let mut carried = vec![NONE; n];
    for g in m.globals() {
        if let GlobalKind::Resource(slot) = &g.kind {
            let texture = match slot.kind {
                ResourceKind::Texture
                | ResourceKind::CombinedTextureSampler(_)
                | ResourceKind::YCbCrPlanes(_) => Some(slot.index),
                ResourceKind::SeparateTexture(_) => slot.separate_texture_of().map(|(t, _)| t),
                _ => None,
            };
            if let (Some(t), Some(c)) = (texture, carried.get_mut(g.value.index())) {
                *c = t;
            }
        }
    }
    let mut settled = false;
    for _ in 0..MOST_PASSES {
        let mut changed = false;
        for node in m.blocks().iter().flat_map(|b| &b.insts) {
            let (Some(r), Some(from)) = (node.result, carried_on(&node.inst)) else {
                continue;
            };
            let slot = carried.get(from.index()).copied().unwrap_or(NONE);
            if slot != NONE && holds_image(m, r)? {
                if let Some(c) = carried.get_mut(r.index()).filter(|c| **c != slot) {
                    *c = slot;
                    changed = true;
                }
            }
        }
        if !changed {
            settled = true;
            break;
        }
    }
    if !settled {
        return Ok(false);
    }
    let watched = |v: Value| {
        carried
            .get(v.index())
            .copied()
            .filter(|s| slots.contains(s))
    };
    let mut fetches = vec![0u32; slots.len()];
    let mut elsewhere = false;
    for b in m.blocks() {
        let merged = b
            .phis
            .iter()
            .flat_map(|phi| phi.incoming.iter().map(|(v, _)| *v));
        let returned = match &b.term {
            Terminator::ReturnValue(v) => Some(*v),
            _ => None,
        };
        if merged.chain(returned).any(|v| watched(v).is_some()) {
            return Ok(false);
        }
        for node in &b.insts {
            node.inst.for_each_value(|v| {
                let Some(slot) = watched(v) else {
                    return;
                };
                match use_of(node, v) {
                    Use::NoTexel => {}
                    Use::Fetch { .. } => {
                        let at = slots.iter().position(|&s| s == slot);
                        if let Some(f) = at.and_then(|i| fetches.get_mut(i)) {
                            *f += 1;
                        }
                    }
                    Use::Elsewhere => elsewhere = true,
                }
            });
        }
    }
    Ok(!elsewhere && fetches.iter().all(|&f| f == 1))
}

/// Whether a conditional branch of block `from` to `t` or `f` only decides whether the invocation
/// demotes itself (`if (c) discard_fragment();` at MSL 2.3 or later on a profile with demote,
/// `lower::fragment::discard`): each target is the block both paths meet at, or a block with no
/// phi that branches on to it holding nothing or `OpDemoteToHelperInvocation` alone, at least one
/// of them the demote, and every phi of the meeting block takes the same value from both paths.
/// Both paths then meet with the same values, and a helper invocation, which is one already, is
/// no different for the demote: "Demote this fragment shader invocation to a helper invocation",
/// which "is not considered a flow control instruction" (`OpDemoteToHelperInvocation`). Neither
/// arm holds a derivative, quad or subgroup operation, loop or undefined behavior a helper's path
/// could reach.
fn demote_guard(m: &TypedModule, from: usize, t: BlockId, f: BlockId) -> bool {
    let blocks = m.blocks();
    // Where a path through `arm` meets the other: the block it reaches, the block it reaches it
    // from, and whether it demotes on the way.
    let path = |arm: BlockId| -> Option<(usize, usize, bool)> {
        let block = blocks.get(arm.index())?;
        let demotes_alone =
            (block.insts.iter()).all(|n| matches!(n.inst, Inst::DemoteToHelperInvocation));
        let forwards = block.phis.is_empty() && block.merge.is_none() && demotes_alone;
        match block.term {
            Terminator::Branch(to) if forwards => {
                Some((to.index(), arm.index(), !block.insts.is_empty()))
            }
            _ => Some((arm.index(), from, false)),
        }
    };
    let (Some((to, a, demotes_a)), Some((other, b, demotes_b))) = (path(t), path(f)) else {
        return false;
    };
    let edge = |phi: &Phi, p: usize| {
        (phi.incoming.iter())
            .find(|(_, at)| at.index() == p)
            .map(|(v, _)| *v)
    };
    to == other
        && a != b
        && (demotes_a || demotes_b)
        && (blocks.get(to))
            .is_some_and(|meet| (meet.phis.iter()).all(|phi| edge(phi, a) == edge(phi, b)))
}

/// Add `reach` to `table[v]`; whether it grew.
fn join(table: &mut [u128], v: Value, reach: u128) -> bool {
    match table.get_mut(v.index()) {
        Some(t) if *t | reach != *t => {
            *t |= reach;
            true
        }
        _ => false,
    }
}

/// Point pointer `r` into the variable `from` points into; whether that changed.
fn point(base: &mut [u32], r: Value, from: Value) -> bool {
    let to = base
        .get(from.index())
        .copied()
        .unwrap_or(from.index() as u32);
    match base.get_mut(r.index()) {
        Some(b) if *b != to => {
            *b = to;
            true
        }
        _ => false,
    }
}

/// The value of the variable pointer `ptr` points into, as far as the passes found it.
fn root(base: &[u32], ptr: Value) -> Value {
    let at = base.get(ptr.index()).copied().unwrap_or(ptr.index() as u32);
    Value::new(at)
}

/// Whether `v` is a plain integer constant none of whose lanes is `-1` (every bit of its width set).
fn never_minus_one(m: &TypedModule, v: Value) -> Result<bool, Error> {
    let ValueOrigin::Const(c) = m.value(v)?.origin else {
        return Ok(false);
    };
    let elem = match m.types().get(m.value(v)?.ty)? {
        Type::Vector { elem, .. } => *elem,
        _ => m.value(v)?.ty,
    };
    let Type::Int { width, .. } = m.types().get(elem)? else {
        return Ok(false);
    };
    let ones = match *width {
        64.. => u64::MAX,
        w => (1u64 << w) - 1,
    };
    let other = |bits: u64| bits & ones != ones;
    Ok(match &m.consts().get(c)?.kind {
        ConstKind::Plain(Constant::Scalar { bits, .. }) => other(bits.0),
        ConstKind::Plain(Constant::Composite { parts, .. }) => parts
            .iter()
            .all(|part| m.consts().scalar_bits(*part).is_some_and(|b| other(b.0))),
        ConstKind::Plain(Constant::Null { .. }) => true,
        _ => false,
    })
}

/// Whether a store through `ptr` keeps its value where the module's own loads read it alone: a
/// `Function`, `Private` or `Output` variable's.
fn kept_class(m: &TypedModule, ptr: Value) -> Result<bool, Error> {
    let class = match m.types().get(m.value(ptr)?.ty)? {
        Type::Pointer { class, .. } => *class,
        _ => return Ok(false),
    };
    let origin_ok = !matches!(m.value(ptr)?.origin, ValueOrigin::Param { .. });
    Ok(origin_ok
        && matches!(
            class,
            StorageClass::Function | StorageClass::Private | StorageClass::Output
        ))
}
