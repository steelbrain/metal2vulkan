//! A private array of nested arrays is allocated as far as its accesses reach where the constants
//! bound every index of it, and keeps its declared type where one index has no bound.

use super::*;

/// How [`kernel`]'s private array is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    /// Counters of 2 and 4 runs, a constant index, and a read at constant indices.
    Counted,
    /// The outer counter runs as often as function constant 0, supplied with this value.
    Constant(u32),
    /// The read's outer index is a word of buffer 1, which only the run knows.
    Dynamic,
    /// The inner counter runs 6 times over the innermost level, of 4 elements.
    Past,
    /// The array's address is also stored to the output as an integer.
    Escapes,
    /// The array is also read whole, by an access at its own address.
    Whole,
}

/// `kernel k(device float* out, device uint* in)` with `float acc[32][32][4]`: for `i` below 2
/// (or function constant 0) and `j` below 4, `acc[i][j][2] = float(i * 16 + j)`; then `out[0] =
/// acc[1][3][2]` (its outer index `in[0]` in [`Case::Dynamic`]).
fn kernel(case: Case) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let f32t = m.float();
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let inner_ty = m.array(f32t, 4);
    let mid_ty = m.array(inner_ty, 32);
    let acc_ty = m.array(mid_ty, 32);
    let tiles = fc_global(&mut m, "tiles", i32t);
    let tiles_addr = m.global_addr(tiles);
    let (zero, one, two, three, four, six, sixteen) = (
        m.const_int(i32t, 0),
        m.const_int(i32t, 1),
        m.const_int(i32t, 2),
        m.const_int(i32t, 3),
        m.const_int(i32t, 4),
        m.const_int(i32t, 6),
        m.const_int(i32t, 16),
    );
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    let (outer, inner, latch, read) = (f.block(), f.block(), f.block(), f.block());
    let acc = f.push(
        thread,
        InstKind::Alloca {
            ty: acc_ty,
            count: one,
            align: None,
        },
    );
    let trips_i = match case {
        Case::Constant(_) => load(&mut f, i32t, tiles_addr),
        _ => two,
    };
    let trips_j = if case == Case::Past { six } else { four };
    let entry = BlockId::new(0);
    f.br(outer);
    f.switch_to(outer);
    let i = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(zero, entry)]),
            fmf: FastMath::NONE,
        },
    );
    f.br(inner);
    f.switch_to(inner);
    let j = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(zero, outer)]),
            fmf: FastMath::NONE,
        },
    );
    let row = f.bin(BinOp::Mul, i, sixteen);
    let word = f.bin(BinOp::Add, row, j);
    let value = f.push(
        f32t,
        InstKind::Cast {
            op: CastOp::UIToFP,
            v: word,
        },
    );
    let indices = match case {
        Case::Past => [zero, i, zero, j],
        _ => [zero, i, j, two],
    };
    let at = f.push(
        thread,
        InstKind::Gep {
            src_ty: acc_ty,
            base: acc,
            indices: Box::new(indices),
            inbounds: true,
        },
    );
    store(&mut f, value, at);
    let j1 = f.bin(BinOp::Add, j, one);
    let j_done = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: j1,
            b: trips_j,
        },
    );
    f.cond_br(j_done, latch, inner);
    f.switch_to(latch);
    let i1v = f.bin(BinOp::Add, i, one);
    let i_done = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: i1v,
            b: trips_i,
        },
    );
    f.cond_br(i_done, read, outer);
    f.switch_to(read);
    let outer_index = match case {
        Case::Dynamic => load(&mut f, i32t, arg(1)),
        _ => one,
    };
    let from = f.push(
        thread,
        InstKind::Gep {
            src_ty: acc_ty,
            base: acc,
            indices: Box::new([zero, outer_index, three, two]),
            inbounds: true,
        },
    );
    let y = load(&mut f, f32t, from);
    store(&mut f, y, arg(0));
    match case {
        Case::Escapes => {
            let a = f.push(
                i64t,
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    v: acc,
                },
            );
            store(&mut f, a, arg(0));
        }
        Case::Whole => {
            let whole = load(&mut f, acc_ty, acc);
            let first = f.push(
                f32t,
                InstKind::ExtractValue {
                    agg: whole,
                    indices: [1, 3, 2].into(),
                },
            );
            store(&mut f, first, arg(0));
        }
        _ => {}
    }
    f.ret(None);
    let k = f.finish();
    // The latch entries of the two counters.
    let body = m.func_mut(k).unwrap().body.as_mut().unwrap();
    for (phi, back, from) in [(i, i1v, latch), (j, j1, inner)] {
        let Value::Inst(phi) = phi else {
            unreachable!()
        };
        let InstKind::Phi { incoming, .. } = &mut body.insts.get_mut(phi).unwrap().kind else {
            unreachable!()
        };
        let mut entries = incoming.to_vec();
        entries.push((back, from));
        *incoming = entries.into_boxed_slice();
    }
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])],
        vec![fc_decl(0, tiles, i32t)],
    );
    let constants: Vec<(u32, ConstValue)> = match case {
        Case::Constant(n) => vec![(0, int(n))],
        _ => vec![(0, int(2))],
    };
    specialize(module, &abi, &request(&constants), &mut meter())
}

/// The levels of each alloca the specialized entry keeps.
fn allocas(e: &Entry) -> Vec<Vec<u64>> {
    let types = e.module.types();
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    body.insts
        .values()
        .filter_map(|inst| match inst.kind {
            InstKind::Alloca { ty, .. } => {
                let mut lens = Vec::new();
                let mut t = ty;
                while let Some(Type::Array { elem, len }) = types.get(t) {
                    lens.push(*len);
                    t = *elem;
                }
                Some(lens)
            }
            _ => None,
        })
        .collect()
}

#[test]
fn an_array_reached_by_counters_and_constants_is_allocated_as_far_as_they_reach() {
    // `i` reaches 0 and 1, `j` 0 to 3, the third level 2 (the stores) and 2 (the read): the
    // array is `float acc[2][4][3]`.
    let e = kernel(Case::Counted).unwrap();
    assert_eq!(allocas(&e), [vec![2, 4, 3]], "{}", body_text(&e));
    // The outer counter's trip count a function constant gives: 3 runs reach 3 elements.
    let e = kernel(Case::Constant(3)).unwrap();
    assert_eq!(allocas(&e), [vec![3, 4, 3]], "{}", body_text(&e));
}

#[test]
fn an_array_one_index_of_which_has_no_bound_keeps_its_declared_type() {
    // An outer index only the run knows may reach any of the 32.
    let e = kernel(Case::Dynamic).unwrap();
    assert_eq!(allocas(&e), [vec![32, 32, 4]], "{}", body_text(&e));
    // An inner index past its level's 4 elements reaches into the next row.
    let e = kernel(Case::Past).unwrap();
    assert_eq!(allocas(&e), [vec![32, 32, 4]], "{}", body_text(&e));
    // An address that becomes an integer reaches the declared layout's bytes.
    let e = kernel(Case::Escapes).unwrap();
    assert_eq!(allocas(&e), [vec![32, 32, 4]], "{}", body_text(&e));
    // A load of the whole array at its own address reads every element.
    let e = kernel(Case::Whole).unwrap();
    assert_eq!(allocas(&e), [vec![32, 32, 4]], "{}", body_text(&e));
}
