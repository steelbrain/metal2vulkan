//! `half` arithmetic held at `float` precision between its roundings on a device without
//! `shaderFloat16`: the fast form of that capability's emulation.
//!
//! Without `shaderFloat16` a module computes a `half` `+ - *` in `float` on its operands held
//! widened and rounds the result by the device's `OpQuantizeToF16` ([`super::narrow::round`],
//! `PrecisionNote::NativeHalfQuantize`). [`plan`]
//! marks each addition, subtraction and product on `half` whose flags hold `contract` and
//! `reassoc` and whose result is read, and only by such operations: selection computes it in
//! `float` and leaves it unrounded ([`super::ArithPlan::held_at_float`]). A chain of them rounds
//! where a value leaves it, at the first operation that reads it otherwise (a store, an output, a
//! conversion, a comparison, a call, a phi, a vector move, or an operation without both flags),
//! one quantize at each exit instead of one per operation. Every value outside the chain is a
//! `half` held widened, as everywhere else. A marked operation's users note
//! `PrecisionNote::FastMathRewrites`, which records the difference: the roundings the chain skips,
//! and a value inside it past `half`'s range staying finite.
//!
//! On a device with `shaderFloat16` nothing is marked.

use std::collections::BTreeSet;

use crate::air::ir::{BinOp as AirBin, FastMath, InstId, InstKind, Type, Value};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::select::SelectCx;
use crate::spirv::builder::QuantizeForm;

/// The `half` additions, subtractions and products of `cx`'s body held at `float` precision: on a
/// profile without `shaderFloat16`, each whose flags hold
/// `contract` and `reassoc` and whose every use is such an operation; none elsewhere.
pub fn plan(cx: &SelectCx<'_>, m: &mut Meter) -> Result<BTreeSet<InstId>, Error> {
    if super::narrow::quantize_form(cx.profile) != QuantizeForm::Device {
        return Ok(BTreeSet::new());
    }
    let n = cx.body.insts.len();
    let mut chain = vec![false; n];
    for (at, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let link = match &inst.kind {
            InstKind::BinOp { op, fmf, .. } => {
                matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul)
                    && fmf.contains(FastMath::CONTRACT.union(FastMath::REASSOC))
                    && is_half(cx, at)
            }
            _ => false,
        };
        if let Some(c) = chain.get_mut(at.raw() as usize) {
            *c = link;
        }
    }
    // Whether each result is read, and whether only chain operations read it.
    let (mut read, mut inside) = (vec![false; n], vec![true; n]);
    for (at, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let user_links = chain.get(at.raw() as usize).copied().unwrap_or(false);
        inst.kind.for_each_value(|v| {
            if let Value::Inst(d) = v {
                let d = d.raw() as usize;
                if let Some(r) = read.get_mut(d) {
                    *r = true;
                }
                if !user_links {
                    if let Some(i) = inside.get_mut(d) {
                        *i = false;
                    }
                }
            }
        });
    }
    Ok(cx
        .body
        .insts
        .iter()
        .map(|(at, _)| at)
        .filter(|at| {
            let i = at.raw() as usize;
            chain.get(i).copied().unwrap_or(false)
                && read.get(i).copied().unwrap_or(false)
                && inside.get(i).copied().unwrap_or(false)
        })
        .collect())
}

/// Whether instruction `at`'s result is a `half` or a vector of them.
fn is_half(cx: &SelectCx<'_>, at: InstId) -> bool {
    let Some(ty) = cx.value_ty(Value::Inst(at)) else {
        return false;
    };
    let types = cx.entry.module.types();
    let scalar = match types.get(ty) {
        Some(Type::Vector { elem, .. }) => types.get(*elem),
        other => other,
    };
    matches!(scalar, Some(Type::Half))
}
