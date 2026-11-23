//! Whether a module writes memory of its own.
//!
//! A multisampled fragment function that writes memory another invocation or a later command can
//! read runs as many times per pixel as the device runs it, which its pipeline's compile records
//! (`StageInterface::writes_memory`, `PrecisionNote::DeviceFragmentRuns`). The writes that count
//! are the function's: a store, a copy, or an atomic other than a load into a storage buffer, a
//! physical storage buffer, or an image, and an image write.
//!
//! The product also writes one buffer the guest never names: the command buffer's
//! `ExecutionStatus` buffer, whose words the resident-miss count, an absent read's mark and a
//! `seq_cst` fence's order update. Those writes are the product's, wherever lowering places them,
//! so a write whose address derives from that buffer does not count. A stage module of a device
//! that addresses buffers reaches the buffer by the device address in the push block's
//! `ExecutionStatusAddress` member, which the placement order always puts in the push block
//! (`lower::push`'s `the_status_address_is_always_placed_in_the_push_block`); one of a device that
//! does not reaches it through its `Synthetic(ExecutionStatus)` storage buffer, the buffer's
//! binding, which counts as the buffer too. A
//! written pointer is traced back through access chains, `OpConvertUToPtr`, an `OpIAdd` of a
//! constant, and an `OpSelect` whose one hand is a constant (the floor's tail folds into each
//! pushed field so: `lower::push::fold_never_pushed`); it addresses the status buffer exactly when
//! the trace ends at that binding or at a load of that member. A pointer that traces anywhere else,
//! or not at all (a phi, a parameter, a selection of two values), is the function's: the guest
//! never holds the status buffer's address, so no write of the guest's is taken for the product's.
//!
//! The trace also ends at an `OpIAdd` whose one operand is a load of the address member, folded or
//! not, whatever the other: the guest does not hold the address, so no sum of the guest's has such
//! an operand.
//!
//! A tile function's color attachment (`TileAttachment`) is the product's too: its writes carry
//! Metal's color attachment write, a color output, which no other invocation or command reads but
//! through the attachment. An image write counts unless its image is a load of that binding. The
//! function's other writes count as before.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::pipeline::sym::{PushField, ResourceKind, SyntheticBinding};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConvOp, GlobalKind, Inst, StorageClass, Type, TypedModule, ValueOrigin,
};

/// Whether module `m` writes memory that another invocation or a later command can read: the
/// stage's stores and atomics of [`super::derive`] without its atomic loads, which write nothing,
/// and without the `ExecutionStatus` buffer's words, a vertex function's capture records and the
/// shader-side stencil's shadow, which are the product's (see the module docs).
pub fn writes_memory(m: &TypedModule) -> BResult<bool> {
    let status = Status::of(m)?;
    for b in m.blocks() {
        for n in &b.insts {
            let ptr = match &n.inst {
                Inst::ImageWrite { image, .. } if status.shadow(*image) => continue,
                Inst::ImageWrite { .. } => return Ok(true),
                Inst::Store { ptr, .. }
                | Inst::AtomicStore { ptr, .. }
                | Inst::Atomic { ptr, .. }
                | Inst::AtomicCompareExchange { ptr, .. } => *ptr,
                _ => continue,
            };
            let shared = matches!(
                class(m, ptr)?,
                StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::Image
            );
            if shared && !status.addresses(m, ptr)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The storage class of pointer `ptr`.
fn class(m: &TypedModule, ptr: Value) -> BResult<StorageClass> {
    match m.types().get(m.value(ptr)?.ty)? {
        Type::Pointer { class, .. } => Ok(*class),
        _ => Err(BuildError::Dangling(Dangling::Value(ptr))),
    }
}

/// How module `'m` reaches the `ExecutionStatus` buffer, with the instructions a written pointer
/// is traced through.
#[derive(Default)]
struct Status<'m> {
    /// The variables of the `Synthetic(ExecutionStatus)` binding.
    bindings: Vec<Value>,
    /// The variables of the `TileAttachment` bindings, and the loads of them.
    shadows: Vec<Value>,
    /// The push block's variable.
    block: Option<Value>,
    /// The indices of its `ExecutionStatusAddress` members, the product's address.
    members: Vec<u32>,
    /// The defining instruction of every access chain, `OpConvertUToPtr`, `OpIAdd`, `OpSelect`,
    /// and load; empty when the module reaches no status buffer.
    defs: BTreeMap<Value, &'m Inst>,
}

impl<'m> Status<'m> {
    fn of(m: &'m TypedModule) -> BResult<Self> {
        let mut s = Status::default();
        for g in m.globals() {
            match g.kind {
                GlobalKind::Resource(slot)
                    if matches!(
                        slot.kind,
                        ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus)
                    ) =>
                {
                    s.bindings.push(g.value);
                }
                GlobalKind::Resource(slot)
                    if matches!(slot.kind, ResourceKind::TileAttachment(_)) =>
                {
                    s.shadows.push(g.value);
                }
                GlobalKind::PushBlock => {
                    s.members = m
                        .push_members()
                        .iter()
                        .filter(|p| matches!(p.field, PushField::ExecutionStatusAddress))
                        .map(|p| p.index)
                        .collect();
                    s.block = Some(g.value);
                }
                _ => {}
            }
        }
        if !s.shadows.is_empty() {
            let loads: Vec<Value> = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|n| match (&n.inst, n.result) {
                    (Inst::Load { ptr, .. }, Some(r)) if s.shadows.contains(ptr) => Some(r),
                    _ => None,
                })
                .collect();
            s.shadows.extend(loads);
        }
        if s.bindings.is_empty() && s.members.is_empty() {
            return Ok(s);
        }
        for b in m.blocks() {
            for n in &b.insts {
                let traced = matches!(
                    n.inst,
                    Inst::AccessChain { .. }
                        | Inst::Convert {
                            op: ConvOp::ConvertUToPtr,
                            ..
                        }
                        | Inst::Binary {
                            op: BinOp::IAdd,
                            ..
                        }
                        | Inst::Select { .. }
                        | Inst::Load { .. }
                );
                if let (true, Some(r)) = (traced, n.result) {
                    s.defs.insert(r, &n.inst);
                }
            }
        }
        Ok(s)
    }

    /// Whether image `image` is the product's (a tile function's color attachment): a load of its
    /// binding.
    fn shadow(&self, image: Value) -> bool {
        self.shadows.contains(&image)
    }

    /// Whether pointer `ptr` addresses the `ExecutionStatus` buffer.
    fn addresses(&self, m: &TypedModule, ptr: Value) -> BResult<bool> {
        let mut v = ptr;
        // Each step moves to an operand of the value's defining instruction, which SSA defines
        // before it, and the map holds no phi, so the trace ends within the map's size.
        for _ in 0..=self.defs.len() {
            if self.bindings.contains(&v) {
                return Ok(true);
            }
            v = match self.defs.get(&v) {
                Some(Inst::AccessChain { base, .. }) => *base,
                Some(Inst::Convert {
                    op: ConvOp::ConvertUToPtr,
                    a,
                    ..
                }) => *a,
                Some(Inst::Binary {
                    op: BinOp::IAdd,
                    a,
                    b,
                    ..
                }) => match (constant(m, *a)?, constant(m, *b)?) {
                    (false, true) => *a,
                    (true, false) => *b,
                    // The product's address plus any offset.
                    (false, false) => {
                        return Ok(self.loads_member(m, *a)? || self.loads_member(m, *b)?)
                    }
                    (true, true) => return Ok(false),
                },
                Some(Inst::Select { .. }) => match self.unfolded(m, v)? {
                    Some(other) => other,
                    None => return Ok(false),
                },
                Some(Inst::Load { ptr, .. }) => return self.is_member(m, *ptr),
                _ => return Ok(false),
            };
        }
        Ok(false)
    }

    /// Whether `v` is a load of the push block's product address member, or such a load folded
    /// ([`Self::unfolded`]).
    fn loads_member(&self, m: &TypedModule, v: Value) -> BResult<bool> {
        let v = self.unfolded(m, v)?.unwrap_or(v);
        match self.defs.get(&v) {
            Some(Inst::Load { ptr, .. }) => self.is_member(m, *ptr),
            _ => Ok(false),
        }
    }

    /// The other hand of `v` where `v` is an `OpSelect` whose one hand is a constant, the form the
    /// floor's tail folds a pushed field in.
    fn unfolded(&self, m: &TypedModule, v: Value) -> BResult<Option<Value>> {
        let Some(Inst::Select { t, f, .. }) = self.defs.get(&v) else {
            return Ok(None);
        };
        Ok(match (constant(m, *t)?, constant(m, *f)?) {
            (true, false) => Some(*f),
            (false, true) => Some(*t),
            _ => None,
        })
    }

    /// Whether `ptr` points to the push block's `ExecutionStatusAddress` member.
    fn is_member(&self, m: &TypedModule, ptr: Value) -> BResult<bool> {
        let (Some(block), Some(Inst::AccessChain { base, indices, .. })) =
            (self.block, self.defs.get(&ptr))
        else {
            return Ok(false);
        };
        let [i] = indices.as_ref() else {
            return Ok(false);
        };
        Ok(*base == block
            && match m.value(*i)?.origin {
                ValueOrigin::Const(c) => m
                    .consts()
                    .scalar_bits(c)
                    .is_some_and(|b| self.members.iter().any(|&k| u64::from(k) == b.0)),
                _ => false,
            })
    }
}

/// Whether `v` is a constant.
fn constant(m: &TypedModule, v: Value) -> BResult<bool> {
    Ok(matches!(m.value(v)?.origin, ValueOrigin::Const(_)))
}
