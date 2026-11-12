//! A static initializer's integer division or remainder by zero: its result is undefined, as an
//! unsupplied function constant's value is, so the pipeline is made where Metal makes it.

use super::*;

/// A static initializer computes `q = 512 <op> d` with `d = (1 << (x & 7)) >> 2` of function
/// constant 0 (`x`, a `uint`), which is 0 for `x & 7` below 2, and `r = 64 / q`; the kernel
/// stores `q` to buffer 0 and, where `r` is 0, 7 to buffer 1. The specialization for `x`.
fn divided(op: BinOp, x: u32) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fc = fc_global(&mut m, "x", i32t);
    let zero = m.constant(i32t, Const::Int(0));
    let q = m.simple_global("q", i32t, AddrSpace::Constant, Some(zero));
    let r = m.simple_global("r", i32t, AddrSpace::Constant, Some(zero));
    let (fc_addr, q_addr, r_addr) = (m.global_addr(fc), m.global_addr(q), m.global_addr(r));
    let (one, two, seven) = (
        m.const_int(i32t, 1),
        m.const_int(i32t, 2),
        m.const_int(i32t, 7),
    );
    let (c0, c64, c512) = (
        m.const_int(i32t, 0),
        m.const_int(i32t, 64),
        m.const_int(i32t, 512),
    );
    static_init(&mut m, "init", |f| {
        let v = load(f, i32t, fc_addr);
        let low = f.bin(BinOp::And, v, seven);
        let tile = f.bin(BinOp::Shl, one, low);
        let d = f.bin(BinOp::LShr, tile, two);
        let quotient = f.bin(op, c512, d);
        store(f, quotient, q_addr);
        let iter = f.bin(BinOp::UDiv, c64, quotient);
        store(f, iter, r_addr);
    });
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    let (then, exit) = (f.block(), f.block());
    let v = load(&mut f, i32t, q_addr);
    store(&mut f, v, arg(0));
    let w = load(&mut f, i32t, r_addr);
    let none = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: w,
            b: c0,
        },
    );
    f.cond_br(none, then, exit);
    f.switch_to(then);
    store(&mut f, seven, arg(1));
    f.br(exit);
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])],
        vec![fc_decl(0, fc, i32t)],
    );
    let value = ConstValue {
        ty: DataType::UInt,
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
fn a_static_division_by_a_nonzero_divisor_folds() {
    // `x` 3: `d` is 2, `q` 256 and `r` 0.
    let e = divided(BinOp::UDiv, 3).unwrap();
    assert_eq!(
        stores(&e),
        ["store i32 256, %a0, align 4", "store i32 7, %a1, align 4"]
    );
}

#[test]
fn a_static_division_by_zero_is_undefined() {
    // `x` 1: `d` is 0. Each division and remainder by it leaves `q` undefined, and `r` from it:
    // the kernel's store of `q` reads its type's zero, and its branch on `r` takes its false
    // way, as an unsupplied constant's would. No constant is undefined, so none is read.
    for op in [BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem] {
        let e = divided(op, 1).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        assert_eq!(
            stores(&e),
            ["store i32 zeroinitializer, %a0, align 4"],
            "{op:?}"
        );
        assert!(e.undefined_reads.is_empty(), "{op:?}");
    }
}
