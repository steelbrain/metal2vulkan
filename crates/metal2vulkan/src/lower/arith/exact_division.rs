//! A constant of a form whose lanes hold bits of their own ([`per_lane`]), which `pack` reads.
//!
//! It is a file of its own so that it is among the runtime's inputs (validation's `build.rs`)
//! without the math intrinsics, which only the compile emits.

use crate::lower::arith::legal::Form;
use crate::lower::arith::ops::{elem_ty, ones, splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{ConstBits, Constant};

/// A constant of form `f` whose lane `i` is `bits[i]`: one value for every lane is a splat
/// ([`splat`]).
pub(super) fn per_lane(fb: &mut FunctionBuilder<'_>, f: Form, bits: &[u64]) -> BResult<Value> {
    match bits {
        [first, rest @ ..] if rest.iter().all(|b| b == first) => return splat(fb, f, *first),
        _ => {}
    }
    if bits.len() != usize::from(f.lanes) {
        return Err(BuildError::Operand {
            op: crate::spv::Op::ExtInst,
            rule: OperandRule::Arity,
        });
    }
    let e = elem_ty(fb, f.held)?;
    let mut parts = Vec::with_capacity(bits.len());
    for b in bits {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(b & ones(f.container())),
        })?);
    }
    let t = ty(fb, f)?;
    let c = fb.module().constant(Constant::Composite {
        ty: t,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}
