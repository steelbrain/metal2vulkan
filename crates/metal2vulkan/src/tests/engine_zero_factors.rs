//! A product by a constant `-0.0` beside a product of the same value by `+0.0`: the owned fixture
//! `sfold_precise` (its authored case `float-operations-decided-by-a-bound-constant-precise`)
//! multiplies one loaded `float4` by the function constants `+0.0` and `-0.0`. On every profile the
//! product by `-0.0` is the product by `+0.0` negated, so no product has a `-0.0` factor beside a
//! `+0.0` one, which a driver may compile as one multiply (ANV's `brw` backend, Mesa 26.2.6).
//! `llvm-as` assembles the fixture; without it the test skips visibly.

use crate::library::{FunctionConstantValue, FunctionConstantValues};
use crate::metal::DataType;
use crate::pipeline::testing::{llvm_assemble, typed_module_specialized, AssembleError};
use crate::spirv::ir::{BinOp, ConstKind, Constant, Inst, TypedModule, UnOp, ValueOrigin};
use crate::spirv::testing::committed_profiles;

/// The `f32` `-0.0`.
const NEGATIVE_ZERO: u64 = 0x8000_0000;

/// The fixture assembled, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/sfold_precise.ll");
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected sfold_precise: {e}"),
    }
}

/// The authored case's function constants: `+0.0`, `-0.0`, two NaNs, and the `half` zeros and NaN.
fn constants() -> FunctionConstantValues {
    let float = |bits: u32| FunctionConstantValue::new(DataType::Float, bits.to_le_bytes());
    let half = |bits: u16| FunctionConstantValue::new(DataType::Half, bits.to_le_bytes());
    FunctionConstantValues::defaults()
        .with_index(0, float(0))
        .with_index(1, float(0x8000_0000))
        .with_index(2, float(0xffc1_2345))
        .with_index(3, float(0x7f80_0321))
        .with_index(4, half(0))
        .with_index(5, half(0x8000))
        .with_index(6, half(0xfd21))
}

/// The `f32` products (`OpFMul`, `OpVectorTimesScalar`) of `m` with a constant factor that has a
/// `-0.0` component.
fn negative_zero_factors(m: &TypedModule) -> usize {
    let negative = |v| {
        let Ok(def) = m.value(v) else { return false };
        let ValueOrigin::Const(c) = def.origin else {
            return false;
        };
        match &m.consts().get(c).expect("constant").kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => bits.0 == NEGATIVE_ZERO,
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts.iter().any(|p| {
                m.consts()
                    .scalar_bits(*p)
                    .is_some_and(|b| b.0 == NEGATIVE_ZERO)
            }),
            _ => false,
        }
    };
    m.blocks()
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|n| match n.inst {
            Inst::Binary {
                op: BinOp::FMul | BinOp::VectorTimesScalar,
                a,
                b,
                ..
            } => negative(a) || negative(b),
            _ => false,
        })
        .count()
}

#[test]
fn a_negative_zero_factor_is_a_negated_positive_one_on_every_profile() {
    let Some(bc) = assemble() else { return };
    let values = constants();
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let (_, m) = typed_module_specialized(&bc, "sfold", &p, &values)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(negative_zero_factors(&m), 0, "{stem}: a -0.0 factor");
        let negations = (m.blocks().iter().flat_map(|b| b.insts.iter()))
            .filter(|n| {
                matches!(
                    n.inst,
                    Inst::Unary {
                        op: UnOp::FNegate,
                        ..
                    }
                )
            })
            .count();
        assert!(negations > 0, "{stem}: no negated product");
    }
}
