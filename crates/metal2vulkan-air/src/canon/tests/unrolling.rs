//! Loops whose trip count the constants decide are unrolled where their counter decides a
//! branch of their body, and SCCP then folds what each iteration's copy decides; the unrolled
//! body is kept only where it is no larger than the loop.

use super::*;
use crate::ir::{LoopHints, Unroll};

/// How the loop of [`counted`] is bounded.
#[derive(Clone, Copy)]
enum Bound {
    /// `i < n`, a constant.
    Trips(u64),
    /// `i < param 2's first word`, which only the run knows.
    Run,
}

/// Where the loop of [`counted`] reads its opcodes.
#[derive(Clone, Copy)]
enum Ops {
    /// Function constant 0, supplied with these lanes.
    Supplied([u32; 4]),
    /// Function constant 0, left undefined.
    Unsupplied,
    /// Parameter 2's first four words, which only the run knows.
    Run,
}

/// The operations of the default arm of [`counted`]'s switch: more than [`unroll::TRIPS`] copies
/// of one cheap case, so that a loop whose copies take the cheap cases is larger than its copies.
const HEAVY: u64 = 80;

/// The cheap opcodes the loops of these tests take.
const CHEAP: [u32; 4] = [1, 2, 2, 1];

/// A kernel whose loop runs `i` from 0 while `bound` holds, switching on lane `i & 3` of the
/// `<4 x i32>` opcodes `ops`: case 1 adds 3 to the accumulator, case 2 multiplies it by 5, any
/// other value applies [`HEAVY`] xors and adds. The accumulator starts from parameter 1's first
/// word and is stored to parameter 0. `hints` go on the latch.
fn counted(bound: Bound, ops: Ops, hints: Option<LoopHints>) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let v4 = m.vector(i32t, 4);
    let dev = m.ptr(AddrSpace::Device);
    let opsg = fc_global(&mut m, "ops", v4);
    let ops_addr = m.global_addr(opsg);
    let (zero, one, three, five, seven) = (
        m.const_int(i32t, 0),
        m.const_int(i32t, 1),
        m.const_int(i32t, 3),
        m.const_int(i32t, 5),
        m.const_int(i32t, 7),
    );
    let n = match bound {
        Bound::Trips(n) => Some(m.const_int(i32t, n)),
        Bound::Run => None,
    };
    let fty = m.fn_ty(void, &[dev, dev, dev]);
    let mut f = m.define("k", fty);
    let (header, add, mul, other, latch, exit) = (
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
    );
    let start = load(&mut f, i32t, arg(1));
    let limit = match n {
        Some(n) => n,
        None => load(&mut f, i32t, arg(2)),
    };
    let entry = BlockId::new(0);
    f.br(header);
    f.switch_to(header);
    // The PHIs' latch entries are filled in once the latch's values exist.
    let i = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(zero, entry)]),
            fmf: FastMath::NONE,
        },
    );
    let acc = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(start, entry)]),
            fmf: FastMath::NONE,
        },
    );
    let lanes = match ops {
        Ops::Run => load(&mut f, v4, arg(2)),
        _ => load(&mut f, v4, ops_addr),
    };
    let lane = f.bin(BinOp::And, i, three);
    let code = f.push(
        i32t,
        InstKind::ExtractElement {
            vec: lanes,
            idx: lane,
        },
    );
    f.push_void(InstKind::Switch {
        v: code,
        default: other,
        cases: Box::new([(1, add), (2, mul)]),
        hints: None,
    });
    f.switch_to(add);
    let a = f.bin(BinOp::Add, acc, three);
    f.br(latch);
    f.switch_to(mul);
    let b = f.bin(BinOp::Mul, acc, five);
    f.br(latch);
    f.switch_to(other);
    let mut c = acc;
    for k in 0..HEAVY {
        c = if k % 2 == 0 {
            f.bin(BinOp::Xor, c, seven)
        } else {
            let step = f.module().const_int(i32t, k);
            f.bin(BinOp::Add, c, step)
        };
    }
    f.br(latch);
    f.switch_to(latch);
    let next = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(a, add), (b, mul), (c, other)]),
            fmf: FastMath::NONE,
        },
    );
    let i1v = f.bin(BinOp::Add, i, one);
    let more = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ult,
            a: i1v,
            b: limit,
        },
    );
    f.push_void(InstKind::CondBr {
        c: more,
        t: header,
        f: exit,
        hints,
    });
    f.switch_to(exit);
    store(&mut f, next, arg(0));
    f.ret(None);
    let k = f.finish();
    // The latch entries of the header's PHIs.
    let body = m.func_mut(k).unwrap().body.as_mut().unwrap();
    for (phi, back) in [(i, i1v), (acc, next)] {
        let Value::Inst(phi) = phi else {
            unreachable!()
        };
        let InstKind::Phi { incoming, .. } = &mut body.insts.get_mut(phi).unwrap().kind else {
            unreachable!()
        };
        let mut entries = incoming.to_vec();
        entries.push((back, latch));
        *incoming = entries.into_boxed_slice();
    }
    let module = m.finish().unwrap();
    let decl = FunctionConstantDecl {
        type_name: DiagName::new("uint4"),
        ..fc_decl(0, opsg, v4)
    };
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1), buffer(2, 2)])],
        vec![decl],
    );
    let supplied = match ops {
        Ops::Supplied(lanes) => Some(lanes),
        _ => None,
    };
    let constants: Vec<(u32, ConstValue)> = supplied
        .map(|lanes| {
            let bytes: Vec<u8> = lanes.iter().flat_map(|l| l.to_le_bytes()).collect();
            (
                0,
                ConstValue {
                    ty: DataType::UInt4,
                    bytes: bytes.into(),
                },
            )
        })
        .into_iter()
        .collect();
    specialize(module, &abi, &request(&constants), &mut meter())
}

/// The operators the printed body applies, in order.
fn ops_of(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| l.split(" = ").nth(1))
        .filter_map(|r| r.split_whitespace().next())
        .filter(|op| ["add", "mul", "xor"].contains(op))
        .collect()
}

#[test]
fn a_loop_whose_counter_indexes_a_constant_vector_is_unrolled_and_folded() {
    // Six iterations over the lanes 1, 2, 2, 1, then 1, 2 again: the counter's lane decides each
    // copy's switch, so straight-line code remains, the counter's own `add`s folded away.
    let e = counted(Bound::Trips(6), Ops::Supplied(CHEAP), None).unwrap();
    let text = body_text(&e);
    assert!(!text.contains("switch"), "{text}");
    assert_eq!(
        ops_of(&text),
        ["add", "mul", "mul", "add", "add", "mul"],
        "{text}"
    );
}

#[test]
fn a_loop_past_the_ceilings_or_its_hints_stays_a_loop() {
    // More iterations than the ceiling.
    let trips = u64::from(unroll::TRIPS) + 1;
    let text = body_text(&counted(Bound::Trips(trips), Ops::Supplied(CHEAP), None).unwrap());
    assert!(text.contains("switch"), "{text}");
    // A trip count only the run knows.
    let text = body_text(&counted(Bound::Run, Ops::Supplied(CHEAP), None).unwrap());
    assert!(text.contains("switch"), "{text}");
    // `llvm.loop.unroll.disable` on the latch.
    let hints = LoopHints {
        unroll: Unroll::Disable,
        must_progress: true,
    };
    let text = body_text(&counted(Bound::Trips(4), Ops::Supplied(CHEAP), Some(hints)).unwrap());
    assert!(text.contains("switch"), "{text}");
    // At the ceiling the loop is unrolled.
    let trips = u64::from(unroll::TRIPS);
    let text = body_text(&counted(Bound::Trips(trips), Ops::Supplied(CHEAP), None).unwrap());
    assert!(!text.contains("switch"), "{text}");
    assert_eq!(ops_of(&text).len(), unroll::TRIPS as usize, "{text}");
}

#[test]
fn a_loop_whose_copies_would_be_larger_stays_a_loop() {
    // Every copy would take the heavy default arm: three of them are larger than the loop, which
    // stays, its switch with it. Undefined lanes read as zero, so an unsupplied vector is the
    // same.
    for ops in [Ops::Supplied([9, 9, 9, 9]), Ops::Unsupplied] {
        let text = body_text(&counted(Bound::Trips(3), ops, None).unwrap());
        assert!(text.contains("switch"), "{text}");
        assert_eq!(ops_of(&text).len() as u64, HEAVY + 3, "{text}");
    }
}

#[test]
fn a_loop_whose_branches_the_run_decides_stays() {
    // Opcodes only the run knows: unrolling decides nothing, and the loop stays.
    let text = body_text(&counted(Bound::Trips(3), Ops::Run, None).unwrap());
    assert!(text.contains("switch"), "{text}");
}
