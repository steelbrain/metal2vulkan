//! A static initializer's integer vectors: lanes inserted into `undef`, compared, extended and
//! truncated, shifted and masked, each lane evaluated as its scalar is.

use super::*;
use crate::ir::ConstId;

/// The shape a stitchable library's initializer has: from the `ushort` function constant 0 (`x`)
/// it stores `lanes = <(x >> i & 1) ^ 1>` built lane by lane into `undef` (`<4 x i32>`), `bits =
/// zext <x & (1 << i) != 0>` built from scalar compares into an `undef` `<4 x i1>` (`<4 x i8>`),
/// `high = trunc(x) >> 4 & 1` (`i8`), and `vector = trunc(zext((splat(x) >> <0, 1, 2, 3> & 1) !=
/// 0))` computed on vectors throughout (`<4 x i8>`). The kernel stores, to buffers 0 to 3, the
/// lanes of each vector packed one per byte (lane `i` at bits `8 i`) and `high`. `complete`
/// false leaves lane 3 of `lanes` `undef` and masks the vector with ones. The specialization for
/// `x`.
fn initialized(x: u16, complete: bool) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let (i1, i8t, i16t, i32t, i64t) = (m.int(1), m.int(8), m.int(16), m.int(32), m.int(64));
    let (v4i1, v4i8, v4i32) = (m.vector(i1, 4), m.vector(i8t, 4), m.vector(i32t, 4));
    let dev = m.ptr(AddrSpace::Device);
    let fc = fc_global(&mut m, "x", i16t);
    let zero4 = m.constant(v4i32, Const::Null);
    let zero8 = m.constant(v4i8, Const::Null);
    let zero = m.constant(i8t, Const::Int(0));
    let lanes = m.simple_global("lanes", v4i32, AddrSpace::Constant, Some(zero4));
    let bits = m.simple_global("bits", v4i8, AddrSpace::Constant, Some(zero8));
    let high = m.simple_global("high", i8t, AddrSpace::Constant, Some(zero));
    let vector = m.simple_global("vector", v4i8, AddrSpace::Constant, Some(zero8));
    let addrs = [fc, lanes, bits, high, vector].map(|g| m.global_addr(g));
    let [fc_addr, lanes_addr, bits_addr, high_addr, vector_addr] = addrs;
    let undef4 = Value::Const(m.constant(v4i32, Const::Undef));
    let undef1 = Value::Const(m.constant(v4i1, Const::Undef));
    let (k0, k1) = (m.const_int(i32t, 0), m.const_int(i32t, 1));
    let k4 = m.const_int(i8t, 4);
    let k1b = m.const_int(i8t, 1);
    let lane = |m: &mut ModuleBuilder, i: u64| m.const_int(i64t, i);
    let idx: Vec<Value> = (0..4).map(|i| lane(&mut m, i)).collect();
    let shifts: Vec<ConstId> = (0..4)
        .map(|i| match m.const_int(i32t, i) {
            Value::Const(c) => c,
            _ => unreachable!(),
        })
        .collect();
    let shifts = Value::Const(m.constant(v4i32, Const::Aggregate(shifts.into())));
    let ones: Vec<ConstId> = (0..4)
        .map(|_| match k1 {
            Value::Const(c) => c,
            _ => unreachable!(),
        })
        .collect();
    let ones = Value::Const(m.constant(v4i32, Const::Aggregate(ones.into())));
    static_init(&mut m, "init", |f| {
        let v = load(f, i16t, fc_addr);
        let w = f.push(
            i32t,
            InstKind::Cast {
                op: CastOp::ZExt,
                v,
            },
        );
        // `lanes`, scalar steps inserted lane by lane.
        let mut acc = undef4;
        for (i, &at) in idx.iter().enumerate().take(if complete { 4 } else { 3 }) {
            let k = f.module().const_int(i32t, i as u64);
            let s = f.bin(BinOp::LShr, w, k);
            let b = f.bin(BinOp::And, s, k1);
            let inv = f.bin(BinOp::Xor, b, k1);
            acc = f.push(
                v4i32,
                InstKind::InsertElement {
                    vec: acc,
                    elt: inv,
                    idx: at,
                },
            );
        }
        // An incomplete vector takes one integer step, which its `undef` lane stops.
        let acc = if complete {
            acc
        } else {
            f.bin(BinOp::And, acc, ones)
        };
        store(f, acc, lanes_addr);
        // `bits`, scalar compares inserted into an `undef` `<4 x i1>`, then a vector `zext`.
        let mut acc = undef1;
        for (i, &at) in idx.iter().enumerate() {
            let k = f.module().const_int(i32t, 1 << i);
            let b = f.bin(BinOp::And, w, k);
            let c = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Ne,
                    a: b,
                    b: k0,
                },
            );
            acc = f.push(
                v4i1,
                InstKind::InsertElement {
                    vec: acc,
                    elt: c,
                    idx: at,
                },
            );
        }
        let z = f.push(
            v4i8,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: acc,
            },
        );
        store(f, z, bits_addr);
        // `high`.
        let t = f.push(
            i8t,
            InstKind::Cast {
                op: CastOp::Trunc,
                v,
            },
        );
        let s = f.bin(BinOp::LShr, t, k4);
        let h = f.bin(BinOp::And, s, k1b);
        store(f, h, high_addr);
        // `vector`, every step on vectors.
        let mut splat = undef4;
        for &at in &idx {
            splat = f.push(
                v4i32,
                InstKind::InsertElement {
                    vec: splat,
                    elt: w,
                    idx: at,
                },
            );
        }
        let s = f.bin(BinOp::LShr, splat, shifts);
        let b = f.bin(BinOp::And, s, ones);
        let c = f.push(
            v4i1,
            InstKind::ICmp {
                pred: IntPredicate::Ne,
                a: b,
                b: Value::Const(zero4),
            },
        );
        let wide = f.push(
            v4i32,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: c,
            },
        );
        let narrow = f.push(
            v4i8,
            InstKind::Cast {
                op: CastOp::Trunc,
                v: wide,
            },
        );
        store(f, narrow, vector_addr);
    });
    let fty = m.fn_ty(void, &[dev, dev, dev, dev]);
    let mut f = m.define("k", fty);
    // Lane `i` of the vector at `addr` (of `ty`, lanes of `bits` bits), zero-extended and shifted
    // to bits `8 i`, or-ed into one `i32`.
    let packed = |f: &mut FunctionBuilder<'_>, ty: TypeId, lane_ty: TypeId, addr: Value| {
        let v = load(f, ty, addr);
        let mut acc = k0;
        for (i, &at) in idx.iter().enumerate() {
            let e = f.push(lane_ty, InstKind::ExtractElement { vec: v, idx: at });
            let e = if lane_ty == i32t {
                e
            } else {
                f.push(
                    i32t,
                    InstKind::Cast {
                        op: CastOp::ZExt,
                        v: e,
                    },
                )
            };
            let k = f.module().const_int(i32t, 8 * i as u64);
            let e = f.bin(BinOp::Shl, e, k);
            acc = f.bin(BinOp::Or, acc, e);
        }
        acc
    };
    let a = packed(&mut f, v4i32, i32t, lanes_addr);
    store(&mut f, a, arg(0));
    let b = packed(&mut f, v4i8, i8t, bits_addr);
    store(&mut f, b, arg(1));
    let h = load(&mut f, i8t, high_addr);
    let h = f.push(
        i32t,
        InstKind::Cast {
            op: CastOp::ZExt,
            v: h,
        },
    );
    store(&mut f, h, arg(2));
    let v = packed(&mut f, v4i8, i8t, vector_addr);
    store(&mut f, v, arg(3));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(
            k,
            vec![buffer(0, 0), buffer(1, 1), buffer(2, 2), buffer(3, 3)],
        )],
        vec![fc_decl(0, fc, i16t)],
    );
    let value = ConstValue {
        ty: DataType::UShort,
        bytes: x.to_le_bytes().into(),
    };
    specialize(module, &abi, &request(&[(0, value)]), &mut meter())
}

fn stores(e: &Entry) -> Vec<String> {
    body_text(e)
        .lines()
        .filter(|l| l.contains("store"))
        .map(|l| l.trim().to_string())
        .collect()
}

#[test]
fn a_static_initializer_evaluates_each_lane_of_its_integer_vectors() {
    // `x` 22 (0b10110): its bits 0 to 3 are 0, 1, 1, 0.
    let e = initialized(22, true).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(
        stores(&e),
        [
            // `lanes` <1, 0, 0, 1>.
            "store i32 16777217, %a0, align 4",
            // `bits` <0, 1, 1, 0>.
            "store i32 65792, %a1, align 4",
            // `high`: 22 >> 4 is 1.
            "store i32 1, %a2, align 4",
            // `vector` <0, 1, 1, 0>.
            "store i32 65792, %a3, align 4",
        ]
    );
    assert!(!body_text(&e).contains("load"), "{}", body_text(&e));
}

#[test]
fn a_static_initializer_vector_with_an_undefined_lane_is_not_evaluated() {
    // Lane 3 of `lanes` is never inserted: the vector holds an `undef` lane, which the evaluator
    // does not carry through an integer step, so the module is refused as before.
    let u = match initialized(22, false) {
        Err(Error::Unsupported(u)) => u,
        other => panic!("{:?}", other.map(|_| ())),
    };
    assert_eq!(u.feature, Feature::StaticInitializer);
}
