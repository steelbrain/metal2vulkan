//! SCCP's values of a few constants: a PHI whose entries are literal constants is one of them, and
//! what is computed from it alone folds member by member, so a branch every member decides alike
//! is decided and its other way deleted.

use super::*;

/// `kernel k(device i32* out, device i32* in)`: a `switch` on `in[0]` picks `unit` 5 (default), 7
/// (case 5) or 8 (case 29), a PHI of constants; `out[0]` is 1 where bit `unit` of `mask` is set
/// (`trunc (lshr mask, zext unit)`), 2 where not.
fn bit_test(mask: u64) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev, dev]);
    let units = [5, 7, 8].map(|u| m.const_int(i32t, u));
    let mask = m.const_int(i64t, mask);
    let (one, two) = (m.const_int(i32t, 1), m.const_int(i32t, 2));
    let mut f = m.define("k", fty);
    let entry = BlockId::new(0);
    let (five, nine, pick, set, clear, done) = (
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
    );
    let which = load(&mut f, i32t, arg(1));
    f.push_void(InstKind::Switch {
        v: which,
        default: pick,
        cases: Box::new([(5, five), (29, nine)]),
        hints: None,
    });
    for b in [five, nine] {
        f.switch_to(b);
        f.br(pick);
    }
    f.switch_to(pick);
    let unit = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(units[0], entry), (units[1], five), (units[2], nine)]),
            fmf: FastMath::NONE,
        },
    );
    let wide = f.push(
        i64t,
        InstKind::Cast {
            op: CastOp::ZExt,
            v: unit,
        },
    );
    let bits = f.bin(BinOp::LShr, mask, wide);
    let bit = f.push(
        i1,
        InstKind::Cast {
            op: CastOp::Trunc,
            v: bits,
        },
    );
    f.cond_br(bit, set, clear);
    for (b, v) in [(set, one), (clear, two)] {
        f.switch_to(b);
        store(&mut f, v, arg(0));
        f.br(done);
    }
    f.switch_to(done);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])], vec![]);
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The `i32` constants the entry stores, in order (`None` for a value the run computes).
fn stored_words(e: &Entry) -> Vec<Option<u64>> {
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    body.insts
        .values()
        .filter_map(|inst| match inst.kind {
            InstKind::Store { val, .. } => Some(match val {
                Value::Const(c) => match e.module.consts().get(c).map(|k| &k.value) {
                    Some(Const::Int(v)) => Some(*v),
                    _ => None,
                },
                _ => None,
            }),
            _ => None,
        })
        .collect()
}

/// Bits 5, 7 and 8 of 416 are set: every unit the `switch` picks takes the first way, and the
/// other way's store is deleted with its block.
#[test]
fn a_bit_test_every_picked_unit_passes_is_decided() {
    let e = bit_test(416).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(stored_words(&e), [Some(1)], "{}", body_text(&e));
    assert!(!body_text(&e).contains("br i1"), "{}", body_text(&e));
}

/// Bit 5 of 384 is clear and bits 7 and 8 are set: the units decide the test apart, and both
/// ways stay.
#[test]
fn a_bit_test_the_picked_units_decide_apart_keeps_both_ways() {
    let e = bit_test(384).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(stored_words(&e), [Some(1), Some(2)], "{}", body_text(&e));
}
