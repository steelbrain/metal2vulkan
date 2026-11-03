//! SCCP reads the integer fields of a constant global whose value is known as their constants:
//! a configuration record a static initializer fills from function constants, read through a
//! reference a function constant picks between the global and a buffer.

use super::*;

/// Whether the printed body branches on a condition.
fn branches(text: &str) -> bool {
    text.lines()
        .any(|l| l.trim_start().starts_with("br ") && l.contains(','))
}

/// `struct { i8 flag, [2 x <4 x i8>] ops, float scale }` in the constant space, filled by a
/// static initializer from function constants 1 (`flag`), 2 and 3 (the two `ops`); the kernel
/// reads it through `select(fc 0, @cfg, param 1)`. When the flag is set it switches on lane `lane`
/// of `ops[which]`, storing 10, 20 or 30 (the default), and then stores the scale's bits; else it
/// stores 40. `which` of `None` reads `ops` at an index parameter 2 holds.
fn configured(
    constants: &[(u32, ConstValue)],
    which: Option<u64>,
    lane: u64,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let f32t = m.float();
    let v4 = m.vector(i8t, 4);
    let ops = m.array(v4, 2);
    let cfg_ty = m.struct_ty(&[i8t, ops, f32t]);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let generic = fc_global(&mut m, "generic", i8t);
    let flag = fc_global(&mut m, "flag", i8t);
    let ops0 = fc_global(&mut m, "ops0", v4);
    let ops1 = fc_global(&mut m, "ops1", v4);
    let zero = m.constant(cfg_ty, Const::Null);
    let name = m.sym("cfg");
    let cfg = m.global(Global {
        name,
        value_ty: cfg_ty,
        space: AddrSpace::Constant,
        constant: false,
        init: Some(zero),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    });
    let cfg_addr = m.global_addr(cfg);
    let (generic_addr, flag_addr, ops0_addr, ops1_addr) = (
        m.global_addr(generic),
        m.global_addr(flag),
        m.global_addr(ops0),
        m.global_addr(ops1),
    );
    let (z32, one32, two32) = (
        m.const_int(i32t, 0),
        m.const_int(i32t, 1),
        m.const_int(i32t, 2),
    );
    let (z64, one64) = (m.const_int(i64t, 0), m.const_int(i64t, 1));
    let cid = |v: Value| match v {
        Value::Const(c) => c,
        _ => unreachable!("a constant"),
    };
    let field = |m: &mut ModuleBuilder, path: &[Value]| {
        let ops = std::iter::once(cfg_addr)
            .chain(path.iter().copied())
            .map(cid)
            .collect();
        let gep = ConstExpr::Gep {
            src_ty: cfg_ty,
            inbounds: true,
            ops,
        };
        Value::Const(m.constant(cptr, Const::Expr(gep)))
    };
    let ops0_field = field(&mut m, &[z64, one32, z64]);
    let ops1_field = field(&mut m, &[z64, one32, one64]);
    static_init(&mut m, "init", |f| {
        let v = load(f, i8t, flag_addr);
        store(f, v, cfg_addr);
        let a = load(f, v4, ops0_addr);
        store(f, a, ops0_field);
        let b = load(f, v4, ops1_addr);
        store(f, b, ops1_field);
    });
    let fty = m.fn_ty(void, &[dev, cptr, dev]);
    let consts: Vec<Value> = [10, 20, 30, 40]
        .iter()
        .map(|&v| m.const_int(i32t, v))
        .collect();
    let lane_v = m.const_int(i32t, lane);
    let zero8 = m.const_int(i8t, 0);
    let mut f = m.define("k", fty);
    let (on, a, b, c, scale, off, exit) = (
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
        f.block(),
    );
    let g = load(&mut f, i8t, generic_addr);
    let gc = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ne,
            a: g,
            b: zero8,
        },
    );
    let p = f.push(
        cptr,
        InstKind::Select {
            c: gc,
            t: arg(1),
            f: cfg_addr,
            fmf: FastMath::NONE,
        },
    );
    let flag_at = f.push(
        cptr,
        InstKind::Gep {
            src_ty: cfg_ty,
            base: p,
            indices: Box::new([z64, z32]),
            inbounds: true,
        },
    );
    let fl = load(&mut f, i8t, flag_at);
    let set = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ne,
            a: fl,
            b: zero8,
        },
    );
    f.cond_br(set, on, off);
    f.switch_to(on);
    let index = match which {
        Some(w) => f.module().const_int(i64t, w),
        None => load(&mut f, i64t, arg(2)),
    };
    let ops_at = f.push(
        cptr,
        InstKind::Gep {
            src_ty: cfg_ty,
            base: p,
            indices: Box::new([z64, one32, index]),
            inbounds: true,
        },
    );
    let lanes = load(&mut f, v4, ops_at);
    let x = f.push(
        i8t,
        InstKind::ExtractElement {
            vec: lanes,
            idx: lane_v,
        },
    );
    f.push_void(InstKind::Switch {
        v: x,
        default: c,
        cases: Box::new([(1, a), (2, b)]),
        hints: None,
    });
    for (block, v) in [(a, consts[0]), (b, consts[1]), (c, consts[2])] {
        f.switch_to(block);
        store(&mut f, v, arg(0));
        f.br(scale);
    }
    f.switch_to(scale);
    let scale_at = f.push(
        cptr,
        InstKind::Gep {
            src_ty: cfg_ty,
            base: p,
            indices: Box::new([z64, two32]),
            inbounds: true,
        },
    );
    let s = load(&mut f, f32t, scale_at);
    store(&mut f, s, arg(0));
    f.br(exit);
    f.switch_to(off);
    store(&mut f, consts[3], arg(0));
    f.br(exit);
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let typed = |index, global, ty, name| FunctionConstantDecl {
        type_name: DiagName::new(name),
        ..fc_decl(index, global, ty)
    };
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1), buffer(2, 2)])],
        vec![
            typed(0, generic, i8t, "bool"),
            typed(1, flag, i8t, "bool"),
            typed(2, ops0, v4, "uchar4"),
            typed(3, ops1, v4, "uchar4"),
        ],
    );
    specialize(module, &abi, &request(constants), &mut meter())
}

fn uchar4(lanes: [u8; 4]) -> ConstValue {
    ConstValue {
        ty: DataType::UChar4,
        bytes: Box::new(lanes),
    }
}

/// The constants of `configured`: the generic path off, the flag set, the two opcode vectors.
fn set(generic: u8, flag: u8) -> Vec<(u32, ConstValue)> {
    vec![
        (0, boolean(generic)),
        (1, boolean(flag)),
        (2, uchar4([2, 1, 7, 2])),
        (3, uchar4([1, 1, 2, 9])),
    ]
}

#[test]
fn an_integer_field_of_a_configuration_global_is_its_constant() {
    // Read through the select the generic flag decided, each integer field folds: the flag's
    // branch and the opcode's switch go, and only the case its lane picks is stored.
    for (which, lane, kept) in [(0, 0, "20"), (0, 1, "10"), (0, 2, "30"), (1, 2, "20")] {
        let text = body_text(&configured(&set(0, 1), Some(which), lane).unwrap());
        assert!(!text.contains("switch") && !branches(&text), "{text}");
        assert!(text.contains(&format!("store i32 {kept}")), "{text}");
        for other in ["10", "20", "30", "40"].into_iter().filter(|&o| o != kept) {
            assert!(!text.contains(&format!("store i32 {other}")), "{text}");
        }
        // The float field is the device's to read: its load stays.
        assert!(text.contains("load float"), "{text}");
    }
    // The flag clear: only the other way stays.
    let text = body_text(&configured(&set(0, 0), Some(0), 0).unwrap());
    assert!(
        text.contains("store i32 40") && !text.contains("switch"),
        "{text}"
    );
}

#[test]
fn a_field_read_the_run_decides_stays_a_load() {
    // The generic path reads the record from a buffer: nothing folds.
    let text = body_text(&configured(&set(1, 1), Some(0), 0).unwrap());
    assert!(text.contains("switch") && branches(&text), "{text}");
    // An index only the run knows: the flag folds, the opcode's switch stays.
    let text = body_text(&configured(&set(0, 1), None, 0).unwrap());
    assert!(
        text.contains("switch") && !text.contains("store i32 40"),
        "{text}"
    );
}
