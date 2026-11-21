//! The bits of each specialization constant a module can read.
//!
//! A binder sets some constants from words of which a module reads only part: a texture slot's
//! `SpecKey::TexelFormat` class holds its format's family, channels and border texels, of which a
//! module may test only some. Two values that agree on
//! every bit the module can read give the same program, so a binder that clears the other bits
//! makes one pipeline of them, and the driver (and Metal's own caches under MoltenVK) sees one
//! specialization.
//!
//! The analysis is a backward demanded-bits pass over the finished [`TypedModule`]. Each value
//! has a mask of the bits its readers can observe; a constant's mask is the union of what each
//! reading instruction demands of it:
//!
//! - `OpBitwiseAnd`: what its result's readers demand, restricted to the other operand's bits
//!   where that operand is a plain integer constant;
//! - `OpShiftRightLogical` by a plain constant below the width: what its result's readers demand,
//!   shifted back into place;
//! - `OpFunctionCall`: what the callee's parameter's readers demand.
//!
//! Every other reader demands every bit: any other instruction's operand, a phi's incoming
//! value, a terminator's operand, a constant operand an instruction names, a composite, workgroup
//! size or specialization-operation part, an initializer and an array length. So a use the pass
//! does not recognise keeps the whole value, and an instruction whose result nothing reads
//! demands nothing of its operands only where it computes that result and does nothing else.
//!
//! Functions are read callees first (Vulkan forbids recursion; a call cycle keeps every bit), and
//! the pass runs until no mask grows, which for a module in SPIR-V's block order is the second
//! pass. A module that does not settle within [`PASSES`] keeps every bit. The work is linear in
//! the module's instructions per pass, metered, with one mask per value charged to the ledger.

use std::collections::BTreeMap;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::artifact::SpecializedNote;
use crate::pipeline::interface::StageSpec;
use crate::pipeline::sym::SpecKey;
use crate::spirv::ids::{ConstId, Value};
use crate::spirv::ir::{BinOp, ConstKind, Constant, Decl, Inst, Type, TypedModule, ValueOrigin};

/// The most passes before the analysis gives up and keeps every bit of every constant.
const PASSES: usize = 8;

/// Every bit.
const ALL: u64 = u64::MAX;

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        format!("specialization bits: {detail}"),
    ))
}

/// Set each of `spec`'s [`StageSpec::unread_bits`]: the bits of its `size` that no instruction
/// of `m` can read (module docs) and no clause of `notes` tests, so a binder may clear them.
/// A constant `m` does not declare keeps every bit.
pub fn mark_unread(
    m: &TypedModule,
    spec: &mut [StageSpec],
    notes: &[SpecializedNote],
    meter: &mut Meter,
) -> Result<(), Error> {
    let n = m.values().len();
    let bytes = n.saturating_mul(std::mem::size_of::<u64>());
    meter.charge(bytes)?;
    let demand = demanded(m, meter);
    meter.release(bytes);
    let demand = demand?;
    let mut values: BTreeMap<SpecKey, Value> = BTreeMap::new();
    for d in m.decls() {
        let Decl::Const(c) = d else { continue };
        if let ConstKind::Spec { key, .. } = m.consts().get(*c)?.kind {
            values.insert(key, m.const_value(*c)?);
        }
    }
    for s in spec {
        let read = match (&demand, values.get(&s.key)) {
            (Some(bits), Some(v)) => bits.get(v.index()).copied().unwrap_or(ALL),
            _ => ALL,
        };
        let tested = notes
            .iter()
            .flat_map(|note| &note.when)
            .flat_map(|clause| &clause.all)
            .filter(|test| test.key == s.key)
            .fold(0u64, |acc, test| acc | u64::from(test.mask));
        let size = match s.size {
            0..=7 => (1u64 << (8 * s.size)) - 1,
            _ => ALL,
        };
        s.unread_bits = size & !(read | tested);
    }
    Ok(())
}

/// The demanded bits of every value of `m`, indexed by [`Value`], or `None` where the analysis
/// keeps every bit (a call cycle, or no settling within [`PASSES`]).
fn demanded(m: &TypedModule, meter: &mut Meter) -> Result<Option<Vec<u64>>, Error> {
    let mut bits = vec![0u64; m.values().len()];
    let full = |bits: &mut Vec<u64>, v: Value| {
        if let Some(b) = bits.get_mut(v.index()) {
            *b = ALL;
        }
    };
    // Readers that demand every bit whatever the order: module-level parts and initializers,
    // array lengths, phis, terminators, and the constants instructions name.
    meter.tick(u32::try_from(m.decls().len()).unwrap_or(u32::MAX))?;
    for d in m.decls() {
        let Decl::Const(c) = d else { continue };
        let parts: &[ConstId] = match &m.consts().get(*c)?.kind {
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts,
            ConstKind::WorkgroupSize { parts } => parts,
            ConstKind::SpecOp { operands, .. } => operands,
            ConstKind::Plain(_) | ConstKind::Spec { .. } => &[],
        };
        for p in parts {
            full(&mut bits, m.const_value(*p)?);
        }
    }
    let inits = m.globals().iter().filter_map(|g| g.init);
    for c in inits.chain(m.vars().iter().filter_map(|v| v.init)) {
        full(&mut bits, m.const_value(c)?);
    }
    meter.tick(u32::try_from(m.types().len()).unwrap_or(u32::MAX))?;
    for (_, t) in m.types().iter() {
        if let Type::Array { len, .. } = t {
            full(&mut bits, m.const_value(*len)?);
        }
    }
    for blk in m.blocks() {
        meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
        for phi in &blk.phis {
            for (v, _) in &phi.incoming {
                full(&mut bits, *v);
            }
        }
        blk.term.for_each_value(|v| full(&mut bits, v));
        for node in &blk.insts {
            let mut named = Ok(());
            node.inst.for_each_const(|c| match m.const_value(c) {
                Ok(v) => full(&mut bits, v),
                Err(e) => named = Err(e),
            });
            named?;
        }
    }
    let Some(order) = callees_first(m) else {
        return Ok(None);
    };
    for _ in 0..PASSES {
        let mut grew = false;
        for f in &order {
            let func = m
                .functions()
                .get(*f)
                .ok_or_else(|| invariant("a call of no function"))?;
            for b in func.blocks.iter().rev() {
                let blk = m
                    .blocks()
                    .get(b.index())
                    .ok_or_else(|| invariant("a function's block is missing"))?;
                meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
                for node in blk.insts.iter().rev() {
                    grew |= transfer(m, &mut bits, node.result, &node.inst);
                }
            }
        }
        if !grew {
            return Ok(Some(bits));
        }
    }
    Ok(None)
}

/// Add what `inst` (with `result`) demands of its operands to `bits`; whether a mask grew.
fn transfer(m: &TypedModule, bits: &mut [u64], result: Option<Value>, inst: &Inst) -> bool {
    let wanted = result
        .and_then(|r| bits.get(r.index()).copied())
        .unwrap_or(0);
    let mut grew = false;
    let mut add = |bits: &mut [u64], v: Value, mask: u64| {
        if let Some(b) = bits.get_mut(v.index()) {
            if *b | mask != *b {
                *b |= mask;
                grew = true;
            }
        }
    };
    match inst {
        Inst::Binary {
            op: BinOp::BitwiseAnd,
            a,
            b,
            ..
        } => {
            add(bits, *a, wanted & plain_bits(m, *b).unwrap_or(ALL));
            add(bits, *b, wanted & plain_bits(m, *a).unwrap_or(ALL));
        }
        Inst::Binary {
            op: BinOp::ShiftRightLogical,
            a,
            b,
            ..
        } => match (width_mask(m, *a), plain_bits(m, *b)) {
            (Some(width), Some(s)) if s < u64::from(width.count_ones()) => {
                add(bits, *a, (wanted << s) & width);
                add(bits, *b, ALL);
            }
            _ => {
                add(bits, *a, ALL);
                add(bits, *b, ALL);
            }
        },
        Inst::Call { func, args } => {
            let params = m.functions().get(func.index()).map(|f| f.params.as_slice());
            for (i, arg) in args.iter().enumerate() {
                let mask = params
                    .and_then(|p| p.get(i))
                    .and_then(|p| bits.get(p.index()).copied())
                    .unwrap_or(ALL);
                add(bits, *arg, mask);
            }
        }
        inst => inst.for_each_value(|v| add(bits, v, ALL)),
    }
    grew
}

/// The bits of `v` where it is a plain integer scalar constant.
fn plain_bits(m: &TypedModule, v: Value) -> Option<u64> {
    let c = match m.value(v).ok()?.origin {
        ValueOrigin::Const(c) => c,
        _ => return None,
    };
    match &m.consts().get(c).ok()?.kind {
        ConstKind::Plain(Constant::Scalar { ty, bits }) => match m.types().get(*ty).ok()? {
            Type::Int { .. } => Some(bits.0),
            _ => None,
        },
        _ => None,
    }
}

/// The mask of `v`'s width where it is an integer scalar.
fn width_mask(m: &TypedModule, v: Value) -> Option<u64> {
    let ty = m.value(v).ok()?.ty;
    match m.types().get(ty).ok()? {
        Type::Int { width: 64, .. } => Some(ALL),
        Type::Int { width, .. } => Some((1u64 << u32::from(*width)) - 1),
        _ => None,
    }
}

/// The functions of `m` in an order where every callee comes before its callers, or `None` on a
/// call cycle.
fn callees_first(m: &TypedModule) -> Option<Vec<usize>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Unread,
        Reading,
        Done,
    }
    let n = m.functions().len();
    let mut state = vec![State::Unread; n];
    let mut order = Vec::with_capacity(n);
    let callees = |f: usize| -> Vec<usize> {
        let mut out = Vec::new();
        let blocks = m
            .functions()
            .get(f)
            .map_or(&[][..], |x| x.blocks.as_slice());
        for b in blocks {
            for node in m.blocks().get(b.index()).map_or(&[][..], |x| &x.insts) {
                if let Inst::Call { func, .. } = node.inst {
                    out.push(func.index());
                }
            }
        }
        out
    };
    for root in 0..n {
        if state.get(root) != Some(&State::Unread) {
            continue;
        }
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(root, callees(root))];
        *state.get_mut(root)? = State::Reading;
        while let Some((f, pending)) = stack.last_mut() {
            let f = *f;
            match pending.pop() {
                Some(g) => match state.get(g)? {
                    State::Done => {}
                    State::Reading => return None,
                    State::Unread => {
                        *state.get_mut(g)? = State::Reading;
                        let next = callees(g);
                        stack.push((g, next));
                    }
                },
                None => {
                    *state.get_mut(f)? = State::Done;
                    order.push(f);
                    stack.pop();
                }
            }
        }
    }
    Some(order)
}
