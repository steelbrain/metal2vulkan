//! SCCP's `fcmp` of two constants: an exact IEEE comparison, a NaN answering by the predicate's
//! order, and never of a subnormal operand, which a device that flushes subnormals compares as a
//! zero of its sign.

use super::*;
use crate::ir::FloatPredicate;

/// A kernel storing `fcmp pred (fc 0), rhs` (zero-extended) to parameter 0, specialized with
/// function constant 0, a `float`, of bits `lhs`. Whether the comparison folded, and if so the
/// stored answer.
fn compared(pred: FloatPredicate, lhs: u32, rhs: u32) -> Option<bool> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let f32t = m.float();
    let dev = m.ptr(AddrSpace::Device);
    let x = fc_global(&mut m, "x", f32t);
    let x_addr = m.global_addr(x);
    let y = Value::Const(m.constant(f32t, Const::Float(rhs.into())));
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let a = load(&mut f, f32t, x_addr);
    let c = f.push(
        i1,
        InstKind::FCmp {
            pred,
            fmf: FastMath::NONE,
            a,
            b: y,
        },
    );
    let w = f.push(
        i32t,
        InstKind::Cast {
            op: CastOp::ZExt,
            v: c,
        },
    );
    store(&mut f, w, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let decl = FunctionConstantDecl {
        type_name: DiagName::new("float"),
        ..fc_decl(0, x, f32t)
    };
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], vec![decl]);
    let value = ConstValue {
        ty: DataType::Float,
        bytes: Box::new(lhs.to_le_bytes()),
    };
    let e = specialize(module, &abi, &request(&[(0, value)]), &mut meter()).unwrap();
    let text = body_text(&e);
    if text.contains("fcmp") {
        return None;
    }
    let one = text.contains("store i32 1");
    assert!(one || text.contains("store i32 0"), "{text}");
    Some(one)
}

#[test]
fn a_comparison_of_normal_constants_folds_exactly() {
    let (one, two) = (1.0f32.to_bits(), 2.0f32.to_bits());
    assert_eq!(compared(FloatPredicate::Olt, one, two), Some(true));
    assert_eq!(compared(FloatPredicate::Oge, one, two), Some(false));
    // Zeros of either sign compare equal.
    assert_eq!(
        compared(FloatPredicate::Oeq, (-0.0f32).to_bits(), 0),
        Some(true)
    );
}

#[test]
fn a_nan_operand_answers_by_the_predicate_s_order() {
    let nan = f32::NAN.to_bits();
    assert_eq!(compared(FloatPredicate::Olt, nan, 0), Some(false));
    assert_eq!(compared(FloatPredicate::One, nan, 0), Some(false));
    assert_eq!(compared(FloatPredicate::Ult, nan, 0), Some(true));
    assert_eq!(compared(FloatPredicate::Uno, nan, 0), Some(true));
}

#[test]
fn a_subnormal_operand_is_the_device_s_to_compare() {
    // 1e-40 is subnormal in `float`: greater than zero on the host, equal to it where the device
    // flushes subnormals, so the comparison stays for the device.
    let tiny = 1e-40f32.to_bits();
    assert_eq!(compared(FloatPredicate::Ogt, tiny, 0), None);
    assert_eq!(compared(FloatPredicate::Oeq, 0, tiny), None);
    // On either side of a normal operand.
    assert_eq!(compared(FloatPredicate::Olt, tiny, 1.0f32.to_bits()), None);
    // The least normal still folds.
    let least = f32::MIN_POSITIVE.to_bits();
    assert_eq!(compared(FloatPredicate::Ogt, least, 0), Some(true));
}
