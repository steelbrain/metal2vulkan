//! Data-model tests: enum exhaustiveness, interners, and layout.

use std::collections::HashSet;

use super::*;
use crate::budget::{Budget, Meter};
use crate::error::{BitcodeError, InputErrorKind, VerifyError};
use crate::error::{Error, Feature, Limit};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// `ALL` lists are complete and duplicate-free. Each `position` is an exhaustive match, so a new
/// variant fails to compile here until it is listed.
macro_rules! assert_all {
    ($all:expr, $position:expr) => {{
        let all = $all;
        let positions: Vec<usize> = all.iter().map(|&v| $position(v)).collect();
        assert_eq!(positions, (0..all.len()).collect::<Vec<_>>());
    }};
}

#[test]
fn opcode_list_is_exhaustive() {
    assert_all!(Opcode::ALL, |o: Opcode| match o {
        Opcode::BinOp => 0,
        Opcode::FNeg => 1,
        Opcode::Cast => 2,
        Opcode::ICmp => 3,
        Opcode::FCmp => 4,
        Opcode::Select => 5,
        Opcode::Freeze => 6,
        Opcode::ExtractElement => 7,
        Opcode::InsertElement => 8,
        Opcode::ShuffleVector => 9,
        Opcode::ExtractValue => 10,
        Opcode::InsertValue => 11,
        Opcode::Alloca => 12,
        Opcode::Load => 13,
        Opcode::Store => 14,
        Opcode::Gep => 15,
        Opcode::AtomicRmw => 16,
        Opcode::CmpXchg => 17,
        Opcode::Fence => 18,
        Opcode::Call => 19,
        Opcode::Phi => 20,
        Opcode::Ret => 21,
        Opcode::Br => 22,
        Opcode::CondBr => 23,
        Opcode::Switch => 24,
        Opcode::Unreachable => 25,
    });
    let terminators: Vec<Opcode> = Opcode::ALL
        .into_iter()
        .filter(|o| o.is_terminator())
        .collect();
    assert_eq!(
        terminators,
        [
            Opcode::Ret,
            Opcode::Br,
            Opcode::CondBr,
            Opcode::Switch,
            Opcode::Unreachable
        ]
    );
}

#[test]
fn operator_lists_are_exhaustive_with_unique_mnemonics() {
    assert_all!(BinOp::ALL, |o: BinOp| o as usize);
    assert_all!(CastOp::ALL, |o: CastOp| o as usize);
    assert_all!(IntPredicate::ALL, |o: IntPredicate| o as usize);
    assert_all!(FloatPredicate::ALL, |o: FloatPredicate| o as usize);
    assert_all!(Ordering::ALL, |o: Ordering| o as usize);
    assert_all!(AtomicRmwOp::ALL, |o: AtomicRmwOp| o as usize);
    assert_all!(SyncScope::ALL, |o: SyncScope| o as usize);
    assert_all!(Linkage::ALL, |o: Linkage| o as usize);
    // AIR numbers no address space 5 this decoder reads, so the list is exhaustive by match and
    // ordered by LLVM number, each variant its own number.
    assert_all!(AddrSpace::ALL, |o: AddrSpace| match o {
        AddrSpace::Thread => 0,
        AddrSpace::Device => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Threadgroup => 3,
        AddrSpace::ThreadgroupImageblock => 4,
        AddrSpace::ObjectData => 5,
        AddrSpace::MeshOutput => 6,
        AddrSpace::Generic => 7,
    });
    assert!(AddrSpace::ALL.windows(2).all(|w| w[0].raw() < w[1].raw()));
    for space in AddrSpace::ALL {
        assert_eq!(AddrSpace::from_raw(space.raw()), Ok(space));
    }
    assert!(AddrSpace::from_raw(5).is_err());
    fn unique<T: Copy>(all: &[T], f: impl Fn(T) -> &'static str) {
        let names: HashSet<&str> = all.iter().map(|&v| f(v)).collect();
        assert_eq!(names.len(), all.len());
    }
    unique(&BinOp::ALL, BinOp::mnemonic);
    unique(&CastOp::ALL, CastOp::mnemonic);
    unique(&IntPredicate::ALL, IntPredicate::mnemonic);
    unique(&FloatPredicate::ALL, FloatPredicate::mnemonic);
    unique(&Ordering::ALL, Ordering::mnemonic);
    unique(&AtomicRmwOp::ALL, AtomicRmwOp::mnemonic);
    unique(&Linkage::ALL, Linkage::keyword);
    let float_ops: Vec<BinOp> = BinOp::ALL.into_iter().filter(|o| o.is_float()).collect();
    assert_eq!(
        float_ops,
        [
            BinOp::FAdd,
            BinOp::FSub,
            BinOp::FMul,
            BinOp::FDiv,
            BinOp::FRem
        ]
    );
}

#[test]
fn flag_sets_have_distinct_bits() {
    let mut seen = 0u8;
    for &(f, _) in IntFlags::ALL {
        assert_eq!(f.bits().count_ones(), 1);
        assert_eq!(seen & f.bits(), 0);
        seen |= f.bits();
    }
    let mut seen = 0u8;
    for &(f, _) in FastMath::ALL {
        assert_eq!(f.bits().count_ones(), 1);
        assert_eq!(seen & f.bits(), 0);
        seen |= f.bits();
    }
    assert!(BinOp::Add.allows(IntFlags::NUW.union(IntFlags::NSW)));
    assert!(!BinOp::Add.allows(IntFlags::EXACT));
    assert!(BinOp::Or.allows(IntFlags::DISJOINT));
    assert!(BinOp::FAdd.allows(IntFlags::NONE));
    assert!(!BinOp::FAdd.allows(IntFlags::NSW));
}

#[test]
fn operand_walks_agree() {
    let v = |n| Value::Inst(InstId::new(n));
    let mut kind = InstKind::Call(Box::new(Call {
        callee: Callee::Indirect {
            callee: v(0),
            fn_ty: TypeId::new(0),
        },
        args: Box::new([v(1), v(2)]),
        fmf: FastMath::NONE,
        tail: TailKind::None,
        attrs: None,
        range: None,
    }));
    let mut seen = Vec::new();
    kind.for_each_value(|x| seen.push(x));
    assert_eq!(seen, [v(0), v(1), v(2)]);
    kind.for_each_value_mut(|x| {
        if let Value::Inst(i) = x {
            *x = Value::Inst(InstId::new(i.raw() + 10));
        }
    });
    let mut seen = Vec::new();
    kind.for_each_value(|x| seen.push(x));
    assert_eq!(seen, [v(10), v(11), v(12)]);

    let switch = InstKind::Switch {
        v: v(0),
        default: BlockId::new(3),
        cases: Box::new([(1, BlockId::new(1)), (2, BlockId::new(1))]),
        hints: None,
    };
    assert_eq!(
        switch.successors().collect::<Vec<_>>(),
        [BlockId::new(3), BlockId::new(1), BlockId::new(1)]
    );
    assert_eq!(InstKind::Freeze(v(0)).successors().next(), None);
}

#[test]
fn align_is_power_of_two_up_to_2_pow_32() {
    assert_eq!(Align::from_bytes(16).map(Align::bytes), Some(16));
    assert_eq!(Align::from_bytes(16).map(Align::log2), Some(4));
    assert_eq!(Align::from_bytes(0), None);
    assert_eq!(Align::from_bytes(12), None);
    assert!(Align::from_bytes(1 << 32).is_some());
    assert_eq!(Align::from_bytes(1 << 33), None);
    assert_eq!(Align::from_log2(33), None);
}

#[test]
fn addr_spaces_decode_and_refuse_unknown() {
    for space in AddrSpace::ALL {
        assert_eq!(AddrSpace::from_raw(space.raw()), Ok(space));
    }
    for n in [5, 9] {
        let err = AddrSpace::from_raw(n).unwrap_err();
        assert_eq!(err.feature, Feature::AddressSpace(n));
    }
}

fn verify_kind(e: Error) -> VerifyError {
    match e {
        Error::InvalidInput(e) => match e.kind {
            InputErrorKind::Verify(v) => v,
            other => panic!("unexpected {other:?}"),
        },
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn types_intern_and_check_construction_rules() {
    let mut m = meter();
    let mut t = TypeTable::new();
    let i32t = t.intern(Type::Int(32), &mut m).unwrap();
    assert_eq!(t.intern(Type::Int(32), &mut m).unwrap(), i32t);
    let f = t.intern(Type::Float, &mut m).unwrap();
    let v4 = t.intern(Type::Vector { elem: f, len: 4 }, &mut m).unwrap();
    assert_eq!(t.lanes(v4), Some(4));
    assert_eq!(t.scalar(v4), f);
    assert_eq!(t.float_bits(v4), Some(32));
    let p = t
        .intern(
            Type::Ptr {
                space: AddrSpace::Device,
            },
            &mut m,
        )
        .unwrap();
    assert_eq!(t.ptr_space(p), Some(AddrSpace::Device));
    let s = t
        .intern(
            Type::Struct {
                fields: Box::new([i32t, v4, p]),
                packed: false,
                name: None,
            },
            &mut m,
        )
        .unwrap();
    assert!(s > v4 && s > p, "children precede parents");
    let void = t.intern(Type::Void, &mut m).unwrap();
    assert!(t
        .intern(
            Type::Function {
                ret: void,
                params: Box::new([p, i32t]),
                vararg: false,
            },
            &mut m,
        )
        .is_ok());

    let bad = [
        Type::Int(0),
        Type::Int(65),
        Type::Vector { elem: s, len: 2 },
        Type::Vector { elem: f, len: 0 },
        Type::Array { elem: void, len: 4 },
        Type::Struct {
            fields: Box::new([void]),
            packed: false,
            name: None,
        },
        Type::Array {
            elem: TypeId::new(999),
            len: 1,
        },
        Type::Function {
            ret: void,
            params: Box::new([void]),
            vararg: false,
        },
    ];
    for ty in bad {
        let before = t.len();
        let e = t.intern(ty.clone(), &mut m).unwrap_err();
        assert_eq!(verify_kind(e), VerifyError::MalformedType, "{ty:?}");
        assert_eq!(t.len(), before);
    }
}

/// Whether a type holds a pointer is kept per type at interning: a pointer, a vector of pointers,
/// and aggregates with one at any depth do; scalars and pointer-free aggregates do not. A 64-level
/// DAG of structs of two copies of the level below (`2^64` paths) answers at once.
#[test]
fn types_know_whether_they_hold_a_pointer() {
    let mut m = meter();
    let mut t = TypeTable::new();
    let i32t = t.intern(Type::Int(32), &mut m).unwrap();
    let p = t
        .intern(
            Type::Ptr {
                space: AddrSpace::Thread,
            },
            &mut m,
        )
        .unwrap();
    let pv = t.intern(Type::Vector { elem: p, len: 2 }, &mut m).unwrap();
    fn st(t: &mut TypeTable, m: &mut Meter, fields: &[TypeId]) -> TypeId {
        t.intern(
            Type::Struct {
                fields: fields.into(),
                packed: false,
                name: None,
            },
            m,
        )
        .unwrap()
    }
    let plain = st(&mut t, &mut m, &[i32t]);
    let held = st(&mut t, &mut m, &[i32t, p]);
    let (mut with, mut without) = (held, plain);
    for _ in 0..64 {
        with = st(&mut t, &mut m, &[with, with]);
        without = st(&mut t, &mut m, &[without, without]);
    }
    let arr = t
        .intern(Type::Array { elem: held, len: 3 }, &mut m)
        .unwrap();
    for (ty, holds) in [
        (i32t, false),
        (p, true),
        (pv, true),
        (plain, false),
        (held, true),
        (arr, true),
        (with, true),
        (without, false),
    ] {
        assert_eq!(t.holds_pointer(ty), holds, "{:?}", t.get(ty));
    }
}

#[test]
fn type_table_is_capped_and_charged() {
    let mut m = meter();
    let mut t = TypeTable::new();
    let before = m.ledger();
    t.intern(Type::Int(8), &mut m).unwrap();
    assert!(m.ledger() > before);
    let mut tight = Meter::with_caps(&Budget::default(), u64::MAX, 8);
    assert_eq!(
        TypeTable::new().intern(Type::Int(8), &mut tight),
        Err(Error::LimitExceeded(Limit::LedgerBytes))
    );
}

#[test]
fn constants_intern_and_require_existing_children() {
    let mut m = meter();
    let mut types = TypeTable::new();
    let i32t = types.intern(Type::Int(32), &mut m).unwrap();
    let mut c = ConstTable::new();
    let one = c
        .intern(
            Constant {
                ty: i32t,
                value: Const::Int(1),
            },
            &mut m,
        )
        .unwrap();
    assert_eq!(
        c.intern(
            Constant {
                ty: i32t,
                value: Const::Int(1)
            },
            &mut m
        )
        .unwrap(),
        one
    );
    let e = c
        .intern(
            Constant {
                ty: i32t,
                value: Const::Aggregate(Box::new([one, ConstId::new(7)])),
            },
            &mut m,
        )
        .unwrap_err();
    assert_eq!(verify_kind(e), VerifyError::MalformedConstant);
    assert_eq!(c.ty(one), Some(i32t));
}

#[test]
fn symbols_intern_bytes() {
    let mut m = meter();
    let mut s = Symbols::new();
    let a = s.intern(b"air.kernel", &mut m).unwrap();
    let b = s.intern(b"\xff\x00", &mut m).unwrap();
    assert_eq!(s.intern(b"air.kernel", &mut m).unwrap(), a);
    assert_eq!(s.as_str(a), Some("air.kernel"));
    assert_eq!(s.as_str(b), None);
    assert_eq!(s.get(b), Some(&b"\xff\x00"[..]));
    assert_eq!(s.lookup(b"missing"), None);
}

const AIR_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-\
v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

struct Fixture {
    types: TypeTable,
    m: Meter,
}

impl Fixture {
    fn new() -> Fixture {
        Fixture {
            types: TypeTable::new(),
            m: meter(),
        }
    }
    fn ty(&mut self, ty: Type) -> TypeId {
        self.types.intern(ty, &mut self.m).unwrap()
    }
    fn layouts(&mut self, dl: &DataLayout) -> TypeLayouts {
        TypeLayouts::new(dl, &self.types, &mut self.m).unwrap()
    }
}

#[test]
fn air_layout_sizes_vec3_as_16_bytes() {
    let dl = DataLayout::parse(AIR_LAYOUT).unwrap();
    assert_eq!(dl.native_ints.as_ref(), [8, 16, 32]);
    let mut fx = Fixture::new();
    let f = fx.ty(Type::Float);
    let h = fx.ty(Type::Half);
    let f3 = fx.ty(Type::Vector { elem: f, len: 3 });
    let h3 = fx.ty(Type::Vector { elem: h, len: 3 });
    let i64t = fx.ty(Type::Int(64));
    let i1 = fx.ty(Type::Int(1));
    let p = fx.ty(Type::Ptr {
        space: AddrSpace::Device,
    });
    let s = fx.ty(Type::Struct {
        fields: Box::new([f, f3, i1, i64t]),
        packed: false,
        name: None,
    });
    let packed = fx.ty(Type::Struct {
        fields: Box::new([i1, f3]),
        packed: true,
        name: None,
    });
    let arr = fx.ty(Type::Array { elem: f3, len: 5 });
    let l = fx.layouts(&dl);
    assert_eq!(l.alloc_size(f3), Some(16));
    assert_eq!(l.store_size(f3), Some(12));
    assert_eq!(l.abi_align(f3), Some(16));
    assert_eq!((l.alloc_size(h3), l.abi_align(h3)), (Some(8), Some(8)));
    assert_eq!((l.alloc_size(i64t), l.abi_align(i64t)), (Some(8), Some(8)));
    assert_eq!((l.alloc_size(i1), l.abi_align(i1)), (Some(1), Some(1)));
    assert_eq!((l.alloc_size(p), l.abi_align(p)), (Some(8), Some(8)));
    assert_eq!(l.field_offsets(s), Some(&[0, 16, 32, 40][..]));
    assert_eq!((l.alloc_size(s), l.abi_align(s)), (Some(48), Some(16)));
    assert_eq!(l.field_offsets(packed), Some(&[0, 1][..]));
    assert_eq!(
        (l.alloc_size(packed), l.abi_align(packed)),
        (Some(17), Some(1))
    );
    assert_eq!(l.alloc_size(arr), Some(80));
}

#[test]
fn layout_rules_follow_llvm_fallbacks() {
    let mut fx = Fixture::new();
    let i40 = fx.ty(Type::Int(40));
    let i2 = fx.ty(Type::Int(2));
    let i128_like = fx.ty(Type::Int(64));
    let d = fx.ty(Type::Double);
    let d3 = fx.ty(Type::Vector { elem: d, len: 3 });
    let bf = fx.ty(Type::BFloat);
    let void = fx.ty(Type::Void);
    let opaque = fx.ty(Type::OpaqueStruct { name: None });
    let big = fx.ty(Type::Array {
        elem: d,
        len: u64::MAX / 2,
    });
    let dl = DataLayout::default();
    let l = fx.layouts(&dl);
    // i40 takes the next larger entry (i64: abi 32 by default).
    assert_eq!((l.store_size(i40), l.abi_align(i40)), (Some(5), Some(4)));
    assert_eq!(l.alloc_size(i40), Some(8));
    assert_eq!((l.alloc_size(i2), l.abi_align(i2)), (Some(1), Some(1)));
    assert_eq!(l.abi_align(i128_like), Some(4));
    // <3 x double> has no entry: natural alignment of its 24-byte store size.
    assert_eq!((l.store_size(d3), l.abi_align(d3)), (Some(24), Some(32)));
    assert_eq!(l.alloc_size(d3), Some(32));
    assert_eq!(l.abi_align(bf), Some(2));
    assert_eq!(l.get(void), None);
    assert_eq!(l.get(opaque), None);
    assert_eq!(l.get(big), None, "size overflow is not a layout");

    let dl = DataLayout::parse("e-p1:32:32-a:64").unwrap();
    let mut fx = Fixture::new();
    let p1 = fx.ty(Type::Ptr {
        space: AddrSpace::Device,
    });
    let p2 = fx.ty(Type::Ptr {
        space: AddrSpace::Constant,
    });
    let i8t = fx.ty(Type::Int(8));
    let s = fx.ty(Type::Struct {
        fields: Box::new([i8t]),
        packed: false,
        name: None,
    });
    let l = fx.layouts(&dl);
    assert_eq!(l.alloc_size(p1), Some(4));
    assert_eq!(l.alloc_size(p2), Some(8), "unlisted spaces use p0");
    assert_eq!((l.alloc_size(s), l.abi_align(s)), (Some(8), Some(8)));
}

#[test]
fn data_layout_parses_every_specifier_and_rejects_the_rest() {
    let dl = DataLayout::parse(
        "E-m:e-S128-A5-P1-G2-Fi8-ni:3:4-p3:32:32:32:16-a0:0:64-i128:128-f80:128-v96:128",
    )
    .unwrap();
    assert_eq!(dl.endian, layout::Endian::Big);
    assert_eq!(dl.mangling, Some(b'e'));
    assert_eq!(dl.stack_align_bits, Some(128));
    assert_eq!(
        (dl.alloca_space, dl.program_space, dl.globals_space),
        (5, 1, 2)
    );
    assert_eq!(dl.fn_ptr_align, Some(layout::FnPtrAlign::Independent(8)));
    assert_eq!(dl.non_integral.as_ref(), [3, 4]);
    assert_eq!(dl.pointer(3).index_bits, 16);
    assert_eq!(dl.pointer(3).size_bits, 32);
    assert_eq!(DataLayout::parse("").unwrap(), DataLayout::default());

    for bad in [
        "x",
        "e-q:1",
        "p:0:64",
        "p:64:12",
        "p:64:64:32",
        "p:64:64:64:128",
        "i8:16",
        "i32:24",
        "i0:8",
        "f32",
        "i32:32:32:32",
        "m:z",
        "e-",
        "Fx8",
        "n0",
        "a1:64",
        "S12",
        "i32:32:16",
        "P-1",
    ] {
        let e = DataLayout::parse(bad).unwrap_err();
        assert!(
            matches!(
                &e,
                Error::InvalidInput(i)
                    if i.kind == InputErrorKind::Bitcode(BitcodeError::BadDataLayout)
            ),
            "{bad}: {e:?}"
        );
    }
}

#[test]
fn arenas_charge_and_look_up() {
    let mut m = meter();
    let mut a: Arena<BlockId, u64> = Arena::new();
    let before = m.ledger();
    let x = a.push(7, &mut m).unwrap();
    let y = a.push(9, &mut m).unwrap();
    // The arena is charged for the capacity it grows to: the first push reserves the 4-element
    // minimum, the second fits.
    assert_eq!(m.ledger() - before, 4 * 8);
    assert_eq!((x.raw(), y.raw()), (0, 1));
    assert_eq!(a.get(y), Some(&9));
    assert_eq!(a.get(BlockId::new(2)), None);
    assert_eq!(a.ids().collect::<Vec<_>>(), [x, y]);
    assert_eq!(format!("{a:?}"), "{bb0: 7, bb1: 9}");
}
