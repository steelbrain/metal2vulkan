//! Aggregate loads and stores of buffer and threadgroup memory, split into their members' accesses
//! (`canon::aggregates`).

use super::*;

/// A constant buffer at `index`.
fn constant_buffer(param: u32, index: u32) -> ParamBinding {
    let mut b = buffer(param, index);
    if let ParamRole::Buffer(arg) = &mut b.role {
        arg.access = Access::Read;
        arg.space = AddrSpace::Constant;
    }
    b
}

/// How [`moved`]'s kernel moves its aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    /// `load [4 x <4 x float>]` from 16 bytes into the constant buffer, row 2 stored to `out`.
    ConstantRows,
    /// `store { i32, <2 x float>, [2 x i16] }` of the argument's fields into `out`, align 8.
    DeviceStruct,
    /// The rows of [`Move::ConstantRows`] stored whole into `out`.
    DeviceRows,
    /// A volatile load of the rows.
    Volatile,
    /// `load { ptr addrspace(1), i32 }` from the constant buffer, its `i32` stored to `out`.
    PointerStruct,
    /// `load [300 x float]` from the constant buffer, element 7 stored to `out`.
    Wide,
    /// `store [2 x i32] [7, 7]` into a threadgroup array, loaded back whole, element 1 stored to
    /// `out`.
    Threadgroup,
}

/// `kernel k(constant void *in [[buffer(0)]], device void *out [[buffer(1)]])`, moving an aggregate
/// as `how` says.
fn moved(how: Move) -> Entry {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let i16t = m.int(16);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let f32t = m.float();
    let v2 = m.vector(f32t, 2);
    let v4 = m.vector(f32t, 4);
    let rows = m.array(v4, 4);
    let halves = m.array(i16t, 2);
    let fields = m.struct_ty(&[i32t, v2, halves]);
    let cst = m.ptr(AddrSpace::Constant);
    let dev = m.ptr(AddrSpace::Device);
    let with_ptr = m.struct_ty(&[dev, i32t]);
    let wide = m.array(f32t, 300);
    let sixteen = m.const_int(i64t, 16);
    let seven = m.const_int(i32t, 7);
    let two = m.const_int(i16t, 2);
    let half = m.constant(f32t, Const::Float(0.5f32.to_bits().into()));
    let pair = Value::Const(m.constant(v2, Const::Aggregate([half, half].into())));
    let no_fields = Value::Const(m.constant(fields, Const::Undef));
    let tg_ty = m.array(i32t, 2);
    let tg_ptr = m.ptr(AddrSpace::Threadgroup);
    let tg_name = m.sym("tg");
    let tg_undef = m.constant(tg_ty, Const::Undef);
    let tg = m.global(Global {
        name: tg_name,
        value_ty: tg_ty,
        space: AddrSpace::Threadgroup,
        constant: false,
        init: Some(tg_undef),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: Align::from_bytes(4),
        section: None,
    });
    let tg_at = Value::Const(m.constant(tg_ptr, Const::GlobalAddr(tg)));
    let sevens = {
        let k = m.constant(i32t, Const::Int(7));
        Value::Const(m.constant(tg_ty, Const::Aggregate([k, k].into())))
    };
    let fty = m.fn_ty(void, &[cst, dev]);
    let mut f = m.define("k", fty);
    let load_as = |f: &mut FunctionBuilder<'_>, ty: TypeId, ptr: Value, volatile: bool| {
        f.push(
            ty,
            InstKind::Load {
                ptr,
                align: Align::from_bytes(16),
                volatile,
                atomic: None,
                range: None,
            },
        )
    };
    let store_as = |f: &mut FunctionBuilder<'_>, val: Value, align: u64| {
        f.push_void(InstKind::Store {
            val,
            ptr: arg(1),
            align: Align::from_bytes(align),
            volatile: false,
            atomic: None,
        });
    };
    let extract = |f: &mut FunctionBuilder<'_>, ty: TypeId, agg: Value, i: u32| {
        f.push(
            ty,
            InstKind::ExtractValue {
                agg,
                indices: [i].into(),
            },
        )
    };
    let at16 = f.push(
        cst,
        InstKind::Gep {
            src_ty: i8t,
            base: arg(0),
            indices: [sixteen].into(),
            inbounds: true,
        },
    );
    match how {
        Move::ConstantRows | Move::Volatile => {
            let whole = load_as(&mut f, rows, at16, how == Move::Volatile);
            let row = extract(&mut f, v4, whole, 2);
            store_as(&mut f, row, 16);
        }
        Move::DeviceRows => {
            let whole = load_as(&mut f, rows, at16, false);
            store_as(&mut f, whole, 16);
        }
        Move::DeviceStruct => {
            let a = f.push(
                fields,
                InstKind::InsertValue {
                    agg: no_fields,
                    val: seven,
                    indices: [0].into(),
                },
            );
            let b = f.push(
                fields,
                InstKind::InsertValue {
                    agg: a,
                    val: pair,
                    indices: [1].into(),
                },
            );
            let c = f.push(
                fields,
                InstKind::InsertValue {
                    agg: b,
                    val: two,
                    indices: [2, 1].into(),
                },
            );
            store_as(&mut f, c, 8);
        }
        Move::PointerStruct => {
            let whole = load_as(&mut f, with_ptr, arg(0), false);
            let x = extract(&mut f, i32t, whole, 1);
            store_as(&mut f, x, 4);
        }
        Move::Threadgroup => {
            f.push_void(InstKind::Store {
                val: sevens,
                ptr: tg_at,
                align: Align::from_bytes(4),
                volatile: false,
                atomic: None,
            });
            let whole = load_as(&mut f, tg_ty, tg_at, false);
            let x = extract(&mut f, i32t, whole, 1);
            store_as(&mut f, x, 4);
        }
        Move::Wide => {
            let whole = load_as(&mut f, wide, arg(0), false);
            let x = extract(&mut f, f32t, whole, 7);
            store_as(&mut f, x, 4);
        }
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![constant_buffer(0, 0), buffer(1, 1)])],
        Vec::new(),
    );
    specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
}

/// The `align` of each line of `text` that starts with `op`, in order.
fn aligns(text: &str, op: &str) -> Vec<u64> {
    text.lines()
        .map(str::trim)
        .filter(|l| l.contains(op))
        .filter_map(|l| l.rsplit("align ").next()?.parse().ok())
        .collect()
}

/// A whole array loaded from a buffer is its members' loads, each at its distance from the
/// access's pointer and at the access's alignment, joined by `insertvalue`.
#[test]
fn a_buffer_aggregate_load_is_its_members_loads() {
    let text = body_text(&moved(Move::ConstantRows));
    assert!(!text.contains("load %t"), "{text}");
    assert_eq!(aligns(&text, "= load <4 x float>"), [16; 4], "{text}");
    assert_eq!(text.matches("insertvalue").count(), 4, "{text}");
    for at in [16, 32, 48] {
        assert!(
            text.contains(&format!("getelementptr inbounds i8, %i0, i64 {at}")),
            "{at}: {text}"
        );
    }
}

/// A whole struct stored into a buffer is its members' stores, the members of a nested array
/// included; a member's alignment is the access's lowered to what its distance allows.
#[test]
fn a_buffer_aggregate_store_is_its_members_stores() {
    let text = body_text(&moved(Move::DeviceStruct));
    assert_eq!(text.matches("extractvalue").count(), 4, "{text}");
    assert!(text.contains("extractvalue i16 %t9 c6, 2, 1"), "{text}");
    assert_eq!(aligns(&text, "store "), [8, 8, 8, 2], "{text}");
    for at in [8, 16, 18] {
        assert!(text.contains(&format!("%a1, i64 {at}")), "{at}: {text}");
    }
    // A whole load stored whole: both sides split.
    let text = body_text(&moved(Move::DeviceRows));
    assert!(
        !text.contains("load %t") && !text.contains("store %i11"),
        "{text}"
    );
    assert_eq!(text.matches("= load <4 x float>").count(), 4, "{text}");
    assert_eq!(aligns(&text, "store "), [16; 4], "{text}");
}

/// Threadgroup memory's aggregates are split as a buffer's are.
#[test]
fn a_threadgroup_aggregate_is_split() {
    let text = body_text(&moved(Move::Threadgroup));
    assert!(
        !text.contains("load %t") && !text.contains("store c"),
        "{text}"
    );
    assert_eq!(text.matches("= load i32").count(), 2, "{text}");
    assert_eq!(text.matches("store ").count(), 3, "{text}");
}

/// A volatile access, an aggregate that holds a pointer, and one of more members than the split
/// takes stay whole.
#[test]
fn what_is_not_split_stays_whole() {
    for (how, whole) in [
        (Move::Volatile, "load volatile %t8"),
        (Move::PointerStruct, "load %t5"),
        (Move::Wide, "load %t5"),
    ] {
        let text = body_text(&moved(how));
        assert!(text.contains(whole), "{how:?}: {text}");
        assert!(!text.contains("insertvalue"), "{how:?}: {text}");
    }
}
