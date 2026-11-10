//! SCCP's float folds: a float operation a bound function constant decides whatever its other
//! operands hold. A NaN operand, the first in operand order, is the result: as it is beside a value
//! only the run knows, quieted with its sign and payload kept where every operand is a constant.
//! Otherwise a product under `nnan` and `nsz` with a zero operand of either sign is `+0`. No other
//! float operation folds.

use super::*;
use crate::intrinsics::{Math, MathFn, Precision};
use crate::ir::{IntFlags, Type};

/// The operation under test.
#[derive(Clone, Copy)]
enum Op {
    Bin(BinOp),
    Fma,
}

/// An operand: function constant `n`; its lane 0 alone in a vector whose other lanes are `poison`
/// (an `insertelement` into `poison` of its `extractelement`, as Metal's compiler spells an
/// operation of two splatted constants); or a value only the run knows (the buffer's).
#[derive(Clone, Copy)]
enum Operand {
    Fc(usize),
    Lane0(usize),
    Run,
}
use Operand::{Fc, Lane0, Run};

/// The value type: `half` or `float`, of `lanes` lanes (1 for a scalar).
#[derive(Clone, Copy)]
struct Shape {
    half: bool,
    lanes: u32,
}

const F32: Shape = Shape {
    half: false,
    lanes: 1,
};
const F32X4: Shape = Shape {
    half: false,
    lanes: 4,
};
const F16X4: Shape = Shape {
    half: true,
    lanes: 4,
};

/// A kernel storing `op` over `operands` to its buffer, which it also reads the run's operand
/// from, specialized with function constant `n` holding the lanes `constants[n]`. The stored
/// lanes where the operation folded (`None` for a `poison` lane), `None` where it survives.
fn folded(
    shape: Shape,
    (op, fmf): (Op, FastMath),
    operands: &[Operand],
    constants: &[&[u64]],
) -> Option<Vec<Option<u64>>> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let elem = if shape.half {
        m.ty(Type::Half)
    } else {
        m.float()
    };
    let ty = match shape.lanes {
        1 => elem,
        n => m.vector(elem, n),
    };
    let globals: Vec<_> = (0..constants.len())
        .map(|n| fc_global(&mut m, &format!("c{n}"), ty))
        .collect();
    let addrs: Vec<_> = globals.iter().map(|&g| m.global_addr(g)).collect();
    let fma = match op {
        Op::Fma => {
            let fma_ty = m.fn_ty(ty, &[ty, ty, ty]);
            Some(m.declare("air.fma", fma_ty))
        }
        Op::Bin(_) => None,
    };
    let i32t = m.int(32);
    let lane0 = m.const_int(i32t, 0);
    let poison = Value::Const(m.constant(ty, Const::Poison));
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let run = load(&mut f, ty, arg(0));
    let args: Vec<_> = operands
        .iter()
        .map(|&o| match o {
            Fc(n) => load(&mut f, ty, addrs[n]),
            Lane0(n) => {
                let vec = load(&mut f, ty, addrs[n]);
                let elt = f.push(elem, InstKind::ExtractElement { vec, idx: lane0 });
                f.push(
                    ty,
                    InstKind::InsertElement {
                        vec: poison,
                        elt,
                        idx: lane0,
                    },
                )
            }
            Run => run,
        })
        .collect();
    let v = match op {
        Op::Bin(op) => f.push(
            ty,
            InstKind::BinOp {
                op,
                flags: IntFlags::NONE,
                fmf,
                a: args[0],
                b: args[1],
            },
        ),
        Op::Fma => {
            let decl = fma.expect("the fma is declared");
            let elem = if shape.half { Elem::F16 } else { Elem::F32 };
            let op = AirIntrinsic::Math(Math {
                f: MathFn::Fma,
                precision: Precision::Precise,
                ty: NumTy {
                    elem,
                    lanes: shape.lanes,
                },
            });
            f.push(
                ty,
                InstKind::Call(Box::new(Call {
                    callee: Callee::Air { decl, op },
                    args: args.as_slice().into(),
                    fmf,
                    tail: TailKind::None,
                    attrs: None,
                    range: None,
                })),
            )
        }
    };
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let (name, data) = match (shape.half, shape.lanes) {
        (false, 1) => ("float", DataType::Float),
        (false, _) => ("float4", DataType::Float4),
        (true, 1) => ("half", DataType::Half),
        (true, _) => ("half4", DataType::Half4),
    };
    let width = if shape.half { 2 } else { 4 };
    let decls = globals
        .iter()
        .enumerate()
        .map(|(n, &g)| FunctionConstantDecl {
            type_name: DiagName::new(name),
            ..fc_decl(n as u32, g, ty)
        })
        .collect();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], decls);
    let values: Vec<_> = constants
        .iter()
        .enumerate()
        .map(|(n, lanes)| {
            assert_eq!(lanes.len(), shape.lanes as usize);
            let bytes = lanes
                .iter()
                .flat_map(|&l| l.to_le_bytes().into_iter().take(width))
                .collect();
            (n as u32, ConstValue { ty: data, bytes })
        })
        .collect();
    let e = specialize(module, &abi, &request(&values), &mut meter()).unwrap();
    let parts = e.module.parts();
    let body = parts.funcs.get(e.func).unwrap().body.as_ref().unwrap();
    let survives = body
        .insts
        .values()
        .any(|i| matches!(i.kind, InstKind::BinOp { .. } | InstKind::Call(_)));
    if survives {
        return None;
    }
    let c = body
        .insts
        .values()
        .find_map(|i| match i.kind {
            InstKind::Store {
                val: Value::Const(c),
                ..
            } => Some(c),
            _ => None,
        })
        .unwrap();
    let lane = |k| match parts.consts.get(k).unwrap().value {
        Const::Float(x) => Some(x),
        Const::Null => Some(0),
        Const::Poison => None,
        ref v => panic!("{v:?}"),
    };
    Some(match &parts.consts.get(c).unwrap().value {
        Const::Aggregate(lanes) => lanes.iter().map(|&k| lane(k)).collect(),
        Const::Null => vec![Some(0); shape.lanes as usize],
        _ => vec![lane(c)],
    })
}

/// Every lane `bits`.
fn all(lanes: u32, bits: u64) -> Option<Vec<Option<u64>>> {
    Some(vec![Some(bits); lanes as usize])
}

const FAST: FastMath = FastMath::NNAN
    .union(FastMath::NINF)
    .union(FastMath::NSZ)
    .union(FastMath::ARCP)
    .union(FastMath::CONTRACT)
    .union(FastMath::AFN)
    .union(FastMath::REASSOC);
const QNAN: u64 = 0xffc1_2345;
const SNAN: u64 = 0x7f80_0321;
const NEG_ZERO: u64 = 0x8000_0000;

fn fmul(fmf: FastMath) -> (Op, FastMath) {
    (Op::Bin(BinOp::FMul), fmf)
}

#[test]
fn a_product_by_a_bound_zero_under_nnan_and_nsz_is_positive_zero() {
    for zero in [0, NEG_ZERO] {
        for operands in [[Run, Fc(0)], [Fc(0), Run]] {
            assert_eq!(folded(F32, fmul(FAST), &operands, &[&[zero]]), all(1, 0));
        }
    }
    let nnan_nsz = FastMath::NNAN.union(FastMath::NSZ);
    assert_eq!(
        folded(F32, fmul(nnan_nsz), &[Run, Fc(0)], &[&[NEG_ZERO]]),
        all(1, 0)
    );
    // Each lane a zero of either sign.
    let zeros = [0, NEG_ZERO, 0, NEG_ZERO];
    assert_eq!(
        folded(F32X4, fmul(FAST), &[Run, Fc(0)], &[&zeros]),
        all(4, 0)
    );
    assert_eq!(
        folded(F16X4, fmul(FAST), &[Fc(0), Run], &[&[0x8000; 4]]),
        all(4, 0)
    );
}

#[test]
fn a_product_by_a_zero_needs_both_nnan_and_nsz_and_every_lane_zero() {
    for fmf in [FastMath::NONE, FastMath::NNAN, FastMath::NSZ] {
        assert_eq!(folded(F32, fmul(fmf), &[Run, Fc(0)], &[&[0]]), None);
    }
    let one = u64::from(1.0f32.to_bits());
    assert_eq!(
        folded(F32X4, fmul(FAST), &[Run, Fc(0)], &[&[0, one, 0, 0]]),
        None
    );
    // A zero beside `poison` lanes.
    assert_eq!(
        folded(F32X4, fmul(FAST), &[Run, Lane0(0)], &[&[0; 4]]),
        None
    );
    // A zero decides no other operation.
    for op in [BinOp::FAdd, BinOp::FSub, BinOp::FDiv] {
        assert_eq!(
            folded(F32, (Op::Bin(op), FAST), &[Fc(0), Run], &[&[0]]),
            None
        );
    }
    assert_eq!(
        folded(F32, (Op::Fma, FAST), &[Fc(0), Run, Run], &[&[0]]),
        None
    );
}

#[test]
fn a_nan_beside_a_run_value_is_the_result_as_it_is() {
    for op in [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv] {
        for fmf in [FAST, FastMath::NONE] {
            for operands in [[Run, Fc(0)], [Fc(0), Run]] {
                for nan in [QNAN, SNAN] {
                    let op = (Op::Bin(op), fmf);
                    assert_eq!(folded(F32, op, &operands, &[&[nan]]), all(1, nan));
                }
            }
        }
    }
    for operands in [[Fc(0), Run, Run], [Run, Run, Fc(0)]] {
        for fmf in [FAST, FastMath::NONE] {
            assert_eq!(
                folded(F32, (Op::Fma, fmf), &operands, &[&[SNAN]]),
                all(1, SNAN)
            );
        }
    }
    // Lane by lane in a vector, each lane its own NaN.
    let nans = [0x7d01, 0xfe00, 0x7c01, 0xffff];
    let fadd = (Op::Bin(BinOp::FAdd), FAST);
    assert_eq!(
        folded(F16X4, fadd, &[Run, Fc(0)], &[&nans]),
        Some(nans.map(Some).to_vec())
    );
}

#[test]
fn a_nan_among_constants_alone_is_the_first_one_quieted() {
    let quiet_snan = 0x7fc0_0321;
    for op in [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv] {
        let op = (Op::Bin(op), FAST);
        assert_eq!(
            folded(F32, op, &[Fc(0), Fc(1)], &[&[QNAN], &[SNAN]]),
            all(1, QNAN)
        );
        assert_eq!(
            folded(F32, op, &[Fc(1), Fc(0)], &[&[QNAN], &[SNAN]]),
            all(1, quiet_snan)
        );
    }
    assert_eq!(
        folded(
            F32,
            (Op::Fma, FAST),
            &[Fc(1), Fc(0), Fc(0)],
            &[&[QNAN], &[SNAN]]
        ),
        all(1, quiet_snan)
    );
    // Before a product's zero.
    for operands in [[Fc(0), Fc(1)], [Fc(1), Fc(0)]] {
        assert_eq!(
            folded(F32, fmul(FAST), &operands, &[&[0], &[SNAN]]),
            all(1, quiet_snan)
        );
    }
    // Lane 0 of each constant beside `poison` lanes, which stay `poison`.
    let fadd = (Op::Bin(BinOp::FAdd), FAST);
    let (q, s) = ([QNAN; 4], [SNAN; 4]);
    assert_eq!(
        folded(F32X4, fadd, &[Lane0(1), Lane0(0)], &[&q, &s]),
        Some(vec![Some(quiet_snan), None, None, None])
    );
}

#[test]
fn no_other_constant_decides_a_float_operation() {
    let inf = u64::from(f32::INFINITY.to_bits());
    let one = u64::from(1.0f32.to_bits());
    for c in [inf, one] {
        for op in [BinOp::FAdd, BinOp::FMul, BinOp::FDiv] {
            assert_eq!(
                folded(F32, (Op::Bin(op), FAST), &[Run, Fc(0)], &[&[c]]),
                None
            );
        }
    }
    // A vector with a lane that is no NaN.
    let mixed = [0x7e00, 0x3c00, 0x7e00, 0x7e00];
    assert_eq!(
        folded(
            F16X4,
            (Op::Bin(BinOp::FAdd), FAST),
            &[Run, Fc(0)],
            &[&mixed]
        ),
        None
    );
    // `frem` is not among the folded operations.
    assert_eq!(
        folded(F32, (Op::Bin(BinOp::FRem), FAST), &[Run, Fc(0)], &[&[QNAN]]),
        None
    );
}
