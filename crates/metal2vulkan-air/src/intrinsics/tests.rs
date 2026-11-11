//! Intrinsic decode: table totality, the name grammars, signature checks, and immediates.

use super::convert::FORMATS;
use super::llvm::BASES;
use super::math::{Domain, TABLE};
use super::ty::token;
use super::*;
use crate::budget::{Budget, Meter};
use crate::ir::{AddrSpace, Const, Constant, Type};

pub(super) fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// The IR type of one operand shape.
pub(super) fn operand_ty(t: &mut TypeTable, op: Operand, m: &mut Meter) -> TypeId {
    let scalar = |t: &mut TypeTable, e: Elem, m: &mut Meter| {
        let ty = match e {
            Elem::I1 => Type::Int(1),
            Elem::I8 => Type::Int(8),
            Elem::I16 => Type::Int(16),
            Elem::I32 => Type::Int(32),
            Elem::I64 => Type::Int(64),
            Elem::F16 => Type::Half,
            Elem::BF16 => Type::BFloat,
            Elem::F32 => Type::Float,
        };
        t.intern(ty, m).unwrap()
    };
    match op {
        Operand::Void => t.intern(Type::Void, m).unwrap(),
        Operand::Num(n) => {
            let elem = scalar(t, n.elem, m);
            if n.is_vector() {
                t.intern(Type::Vector { elem, len: n.lanes }, m).unwrap()
            } else {
                elem
            }
        }
        Operand::WithStatus(n) => {
            let texel = operand_ty(t, Operand::Num(n), m);
            let status = scalar(t, Elem::I8, m);
            t.intern(
                Type::Struct {
                    fields: Box::new([texel, status]),
                    packed: false,
                    name: None,
                },
                m,
            )
            .unwrap()
        }
        Operand::Ptr(space) => t.intern(Type::Ptr { space }, m).unwrap(),
        Operand::Handle(h) => t.intern(Type::Ptr { space: h.space() }, m).unwrap(),
        Operand::Imm(ImmKind::Bool) => scalar(t, Elem::I1, m),
        Operand::Imm(ImmKind::I32) => scalar(t, Elem::I32, m),
        Operand::IntersectionResult(shape) => {
            let fields: Vec<TypeId> = shape
                .fields()
                .map(|f| operand_ty(t, f.operand(), m))
                .collect();
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
    }
}

/// The function type of `sig`.
pub(super) fn fn_ty(t: &mut TypeTable, sig: &Signature) -> TypeId {
    let m = &mut meter();
    let ret = operand_ty(t, sig.ret, m);
    let params: Vec<_> = sig.params().iter().map(|&p| operand_ty(t, p, m)).collect();
    t.intern(
        Type::Function {
            ret,
            params: params.into(),
            vararg: false,
        },
        m,
    )
    .unwrap()
}

pub(super) fn sym() -> Sym {
    Sym::new(0)
}

/// Classify `name`, declared with the signature its own parse derives.
pub(super) fn classify_self(name: &str) -> DeclClass {
    let mut t = TypeTable::new();
    let Some(op) = AirIntrinsic::parse(name) else {
        return DeclClass::Unknown;
    };
    let ty = fn_ty(&mut t, &op.signature());
    classify(name.as_bytes(), sym(), ty, &t).unwrap()
}

pub(super) fn air(name: &str) -> AirIntrinsic {
    match classify_self(name) {
        DeclClass::Air(op) => op,
        other => panic!("{name}: {other:?}"),
    }
}

/// Types a family admits, for the totality sweeps.
fn num_types(float: bool) -> Vec<NumTy> {
    let elems: &[Elem] = if float {
        &[Elem::F16, Elem::BF16, Elem::F32]
    } else {
        &[Elem::I8, Elem::I16, Elem::I32, Elem::I64]
    };
    let mut out = Vec::new();
    for &e in elems {
        for lanes in [1, 2, 3, 4, 8, 16] {
            out.push(NumTy { elem: e, lanes });
        }
    }
    out
}

#[test]
fn every_math_entry_decodes_and_checks() {
    let mut decoded = 0;
    for &(stem, domain, shape) in TABLE {
        let spellings: Vec<(String, MathFn, Precision)> = match domain {
            Domain::Float(f) => ["", "fast_"]
                .iter()
                .map(|p| {
                    let precision = if p.is_empty() {
                        Precision::Precise
                    } else {
                        Precision::Fast
                    };
                    (format!("air.{p}{stem}"), f, precision)
                })
                .collect(),
            Domain::Signed(f) => Sign::ALL
                .iter()
                .map(|&s| {
                    (
                        format!("air.{stem}.{}", s.token()),
                        f(s),
                        Precision::Precise,
                    )
                })
                .collect(),
            Domain::Int(f) => vec![(format!("air.{stem}"), f, Precision::Precise)],
        };
        let float = matches!(domain, Domain::Float(_));
        for (prefix, f, precision) in spellings {
            for ty in num_types(float) {
                if f == MathFn::Dot && !ty.is_vector() {
                    continue;
                }
                let name = format!("{prefix}.{}", token(ty));
                let AirIntrinsic::Math(m) = air(&name) else {
                    panic!("{name}");
                };
                assert_eq!((m.f, m.precision, m.ty), (f, precision, ty), "{name}");
                assert_eq!(m.shape(), shape, "{name}");
                decoded += 1;
            }
        }
    }
    assert!(decoded > 1000);
}

#[test]
fn math_domains_are_exclusive() {
    for name in [
        "air.fast_abs.s.i32",  // no fast integer functions
        "air.fabs.s.f32",      // no sign on a float function
        "air.abs.i32",         // integer abs spells its sign
        "air.popcount.u.i32",  // sign-agnostic takes none
        "air.sqrt.i32",        // float function, integer type
        "air.min.s.f32",       // integer spelling, float type
        "air.dot.f32",         // dot needs a vector
        "air.clz.i1",          // booleans are not integers
        "air.fma.f64",         // no double in AIR
        "air.sqrt.v1f32",      // not a spelling
        "air.sqrt.v04f32",     // not a spelling
        "air.sqrt.f32.extra",  // trailing token
        "air.nonexistent.f32", // unknown stem
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
    // One stem, two domains, chosen by the sign token.
    assert!(matches!(
        air("air.min.f32"),
        AirIntrinsic::Math(Math { f: MathFn::Min, .. })
    ));
    assert!(matches!(
        air("air.min.u.v2i16"),
        AirIntrinsic::Math(Math {
            f: MathFn::IMin(Sign::Unsigned),
            ..
        })
    ));
}

#[test]
fn math_conversion_and_atomic_families_decode() {
    // Math, conversion, and atomic names, each with a type the fixtures use.
    for name in [
        "air.abs_diff.u.v2i16",
        "air.abs.s.v3i32",
        "air.reverse_bits.i32",
        "air.bswap.i32",
        "air.rotate.i32",
        "air.extract_bits.u.i64",
        "air.extract_bits.s.i32",
        "air.insert_bits.u.i64",
        "air.popcount.v2i64",
        "air.ctz.i16",
        "air.clz.i64",
        "air.mul_hi.u.i32",
        "air.mad_hi.s.i32",
        "air.mad_sat.u.i32",
        "air.add_sat.u.v3i8",
        "air.sub_sat.s.i16",
        "air.hadd.s.i16",
        "air.rhadd.u.i32",
        "air.dot.v3f32",
        "air.sincos.f16",
        "air.fast_ldexp.f32",
        "air.clamp.u.v4i32",
        "air.max3.u.i32",
        "air.min3.s.i32",
        // MSL's integer `median3`, as Apple's compiler spells it.
        "air.median3.s.i32",
        "air.median3.u.i32",
        "air.median3.s.i16",
        "air.median3.s.v4i32",
        "air.fmedian3.v4f16",
        "air.fast_fmax3.v2f32",
        "air.atan2.f16",
        "air.fast_tanh.v4f32",
        "air.clamp.f32",
        "air.saturate.v2f16",
        "air.powr.v2f16",
        "air.fract.v3f16",
        "air.convert.f.v16bf16.f.v16f16",
        "air.convert.u.i1.f.f32",
        "air.convert.f.v2f16.u.v2i1",
        "air.convert.s.v4i32.f.v4bf16",
        "air.pack.unorm.rgb565.v3f16",
        "air.pack.unorm4x8.srgb.v4f32",
        "air.unpack.unorm.rg11b10f.v3f32",
        "air.unpack.unorm2x16.v2f16",
        // MPS's 8-bit float tensor reads to integers.
        "air.convert.s.i16.f.f8e4m3",
        "air.convert.u.i64.f.f8e4m3fn",
        "air.convert.s.v2i32.f.v2f8e5m2",
        // MSL 4's `unpack_snorm10a2_to_float` and `pack_half_to_snorm10a2`.
        "air.unpack.snorm.rgb10a2.v4f32",
        "air.pack.snorm.rgb10a2.v4f16",
        "air.atomic.local.store.i32",
        "air.atomic.global.load.i32",
        "air.atomic.global.load.u.i32",
        "air.atomic.local.cmpxchg.weak.i32",
        "air.atomic.global.add.f32",
        "air.atomic.global.sub.f32",
        "air.atomic.local.xchg.i32",
        "air.atomic.global.max.s.i32",
        "air.atomic.global.and.u.i32",
        "air.load.system_coherent.volatile.i32.p1i32",
        "air.store.device_coherent.v4f32.p1v4f32",
    ] {
        air(name);
    }
    for name in [
        "air.convert.f8e5m2.f32.f.f32", // 8-bit float is not a destination type
        "air.convert.u.i1.f.f8e5m2",    // an 8-bit float converts to no `bool`
        "air.convert.s.i32.s.f8e5m2",   // an 8-bit float is read as a float
        "air.convert.f.f32.s.i32.sat",  // `.sat` follows a rounding
        "air.convert.f.f32.s.i32.rte.sat", // a float saturates nothing
        "air.convert.s.i32.f.f32.rtz.sat.sat",
        "air.convert.s.i32.f.f32.rtu",    // no such rounding
        "air.convert.f.f32.u.i1.rte",     // a `bool` rounds nothing
        "air.convert.s.i32.f.f8e5m2.rtz", // nor an 8-bit float
        "air.convert.s.i32.u.v2i32",      // lane counts differ
        "air.convert.f.i32.s.i32",        // kind disagrees with type
        "air.convert.s.i1.f.f32",         // booleans are `u.i1`
        "air.pack.unorm4x8.v3f32",        // lane count
        "air.atomic.global.min.i32",      // integer min spells its sign
        "air.atomic.global.and.f32",      // no float and
        "air.atomic.global.add.v2i32",    // no vector atomics
        "air.atomic.global.xchg.u.i32",   // exchange takes no sign
        "air.atomic.region.add.u.i32",    // unknown space
        "air.load.somewhat_coherent.i32.p1i32",
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
}

/// The rounding Apple's OpenCL compiler spells after a conversion's types, and `.sat` after it to
/// an integer, on every `convert_*` it emits (`convert_float4_rte(uchar4)` is
/// `air.convert.f.v4f32.u.v4i8.rte`, `convert_int_sat(float)` is
/// `air.convert.s.i32.f.f32.rtz.sat`): the conversion its types name, as Metal's compilers for
/// the x86 Macs answer it.
#[test]
fn a_rounded_or_saturated_convert_is_the_conversion_its_types_name() {
    for (suffixed, plain) in [
        (
            "air.convert.f.v4f32.u.v4i8.rte",
            "air.convert.f.v4f32.u.v4i8",
        ),
        (
            "air.convert.f.v4f32.s.v4i32.rte",
            "air.convert.f.v4f32.s.v4i32",
        ),
        ("air.convert.f.f16.s.i32.rtz", "air.convert.f.f16.s.i32"),
        ("air.convert.f.f16.f.f32.rtp", "air.convert.f.f16.f.f32"),
        ("air.convert.s.i32.f.f32.rtn", "air.convert.s.i32.f.f32"),
        ("air.convert.s.i32.f.f32.rtz.sat", "air.convert.s.i32.f.f32"),
        (
            "air.convert.u.v2i16.s.v2i64.rte.sat",
            "air.convert.u.v2i16.s.v2i64",
        ),
        ("air.convert.u.i8.u.i32.rtz", "air.convert.u.i8.u.i32"),
    ] {
        assert_eq!(air(suffixed), air(plain), "{suffixed}");
    }
}

#[test]
fn format_and_operation_tables_are_total() {
    for &(format, want, lanes, _) in FORMATS {
        for elem in [Elem::F16, Elem::F32] {
            let ty = NumTy { elem, lanes };
            for (dir, verb) in [(PackDir::Pack, "pack"), (PackDir::Unpack, "unpack")] {
                let name = format!("air.{verb}.{format}.{}", token(ty));
                assert_eq!(
                    air(&name),
                    AirIntrinsic::Pack(Pack {
                        dir,
                        format: want,
                        ty
                    }),
                    "{name}"
                );
            }
        }
    }
    for &(op, want) in super::atomic::OPS {
        for space in ["global", "local"] {
            let sign = match want {
                AtomicOp::Xchg | AtomicOp::CmpXchgWeak => "",
                _ => "s.",
            };
            let name = format!("air.atomic.{space}.{op}.{sign}i32");
            let AirIntrinsic::Atomic(a) = air(&name) else {
                panic!("{name}");
            };
            assert_eq!(a.op, want, "{name}");
        }
    }
}

/// A module's constant table with `i32`/`i1` constants for the immediate tests.
pub(super) struct Consts {
    pub(super) table: ConstTable,
    types: TypeTable,
}

impl Consts {
    pub(super) fn new() -> Consts {
        Consts {
            table: ConstTable::new(),
            types: TypeTable::new(),
        }
    }

    pub(super) fn int(&mut self, ty: Type, value: Const) -> Value {
        let m = &mut meter();
        let ty = self.types.intern(ty, m).unwrap();
        Value::Const(self.table.intern(Constant { ty, value }, m).unwrap())
    }

    pub(super) fn i32(&mut self, v: u64) -> Value {
        self.int(Type::Int(32), Const::Int(v))
    }

    pub(super) fn bool(&mut self, v: bool) -> Value {
        self.int(Type::Int(1), Const::Int(u64::from(v)))
    }
}

#[test]
fn atomics_read_their_immediates() {
    let template = DeclClass::Air(air("air.atomic.global.add.u.i32"));
    let mut c = Consts::new();
    let ptr = Value::Arg(crate::ir::ArgIdx::new(0));
    let val = Value::Arg(crate::ir::ArgIdx::new(1));
    let zero = c.int(Type::Int(32), Const::Null);
    let (device, yes) = (c.i32(2), c.bool(true));
    let args = [ptr, val, zero, device, yes];
    let Ok(Bound::Air(AirIntrinsic::Atomic(a))) = bind(template, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        (a.order, a.scope, a.volatile, a.failure),
        (MemoryOrder::Relaxed, ThreadScope::Device, true, None)
    );
    // A scope or order outside the enumerations stays unknown, never guessed.
    for (order, scope) in [(0, 5), (1, 2), (0, 99)] {
        let args = [ptr, val, c.i32(order), c.i32(scope), yes];
        assert_eq!(bind(template, &args, &c.table), Ok(Bound::Unknown));
    }
    // A non-constant immediate does not bind (decode keeps the call unbound).
    let args = [ptr, val, val, device, yes];
    assert_eq!(
        bind(template, &args, &c.table),
        Err(IntrinsicError::ImmediateNotConstant)
    );
    let undef = c.int(Type::Int(1), Const::Undef);
    let args = [ptr, val, zero, device, undef];
    assert_eq!(
        bind(template, &args, &c.table),
        Err(IntrinsicError::ImmediateNotConstant)
    );
    // Compare-exchange reads success and failure orders.
    let cas = DeclClass::Air(air("air.atomic.local.cmpxchg.weak.i32"));
    let (tg, seq) = (c.i32(1), c.i32(5));
    let args = [ptr, ptr, val, seq, zero, tg, yes];
    let Ok(Bound::Air(AirIntrinsic::Atomic(a))) = bind(cas, &args, &c.table) else {
        panic!("bind");
    };
    assert_eq!(
        (a.order, a.failure, a.scope),
        (
            MemoryOrder::SeqCst,
            Some(MemoryOrder::Relaxed),
            ThreadScope::Threadgroup
        )
    );
}

/// Metal's 64-bit `atomic_{min,max}_explicit` on a `device atomic_ulong` return nothing:
/// `air.atomic.global.{min,max}.u.i64` declared `void (ptr addrspace(1), i64, i32, i32, i1)`
/// decodes to the same operation without its result and binds the same immediates; no other atomic
/// has a result-less form, so its `void` declaration still mismatches.
#[test]
fn ulong_min_max_atomics_may_be_declared_without_their_result() {
    let mut t = TypeTable::new();
    let ptr = Operand::Ptr(AddrSpace::Device);
    let i64s = Operand::Num(NumTy::scalar(Elem::I64));
    let (i32s, flag) = (Operand::Imm(ImmKind::I32), Operand::Imm(ImmKind::Bool));
    let void = fn_ty(
        &mut t,
        &Signature::new(Operand::Void, &[ptr, i64s, i32s, i32s, flag]),
    );
    for (name, op) in [
        ("air.atomic.global.min.u.i64", AtomicOp::Min),
        ("air.atomic.global.max.u.i64", AtomicOp::Max),
    ] {
        let Some(DeclClass::Air(AirIntrinsic::Atomic(a))) =
            classify(name.as_bytes(), sym(), void, &t)
        else {
            panic!("{name}");
        };
        assert_eq!((a.op, a.fetch), (op, false), "{name}");
        assert_eq!(a.signature().ret, Operand::Void, "{name}");
        // The fetching declaration of the same name keeps its result.
        let AirIntrinsic::Atomic(fetching) = air(name) else {
            panic!("{name}");
        };
        assert_eq!(fetching, Atomic { fetch: true, ..a }, "{name}");
        // The result-less form reads the fetching form's immediates.
        let mut c = Consts::new();
        let (p, v) = (
            Value::Arg(crate::ir::ArgIdx::new(0)),
            Value::Arg(crate::ir::ArgIdx::new(1)),
        );
        let args = [p, v, c.i32(0), c.i32(2), c.bool(true)];
        let Ok(Bound::Air(AirIntrinsic::Atomic(bound))) =
            bind(DeclClass::Air(AirIntrinsic::Atomic(a)), &args, &c.table)
        else {
            panic!("{name}");
        };
        assert_eq!(
            (bound.fetch, bound.order, bound.scope, bound.volatile),
            (false, MemoryOrder::Relaxed, ThreadScope::Device, true)
        );
    }
    // Every other operation, sign, width, and memory has only its fetching form.
    let i32s_val = Operand::Num(NumTy::scalar(Elem::I32));
    let tg = Operand::Ptr(AddrSpace::Threadgroup);
    let void_i32 = fn_ty(
        &mut t,
        &Signature::new(Operand::Void, &[ptr, i32s_val, i32s, i32s, flag]),
    );
    let void_local = fn_ty(
        &mut t,
        &Signature::new(Operand::Void, &[tg, i64s, i32s, i32s, flag]),
    );
    for (name, ty) in [
        ("air.atomic.global.add.u.i64", void),
        ("air.atomic.global.min.s.i64", void),
        ("air.atomic.global.max.u.i32", void_i32),
        ("air.atomic.local.max.u.i64", void_local),
    ] {
        assert_eq!(
            classify(name.as_bytes(), sym(), ty, &t),
            Some(DeclClass::Mismatch),
            "{name}"
        );
    }
}

#[test]
fn count_zeros_reads_its_flag() {
    let template = DeclClass::Air(air("air.clz.i32"));
    let mut c = Consts::new();
    let x = Value::Arg(crate::ir::ArgIdx::new(0));
    let args = [x, c.bool(true)];
    assert_eq!(
        bind(template, &args, &c.table),
        Ok(Bound::Air(AirIntrinsic::Math(Math {
            f: MathFn::Clz { zero_undef: true },
            precision: Precision::Precise,
            ty: NumTy::scalar(Elem::I32),
        })))
    );
}

/// Metal's `frexp` and `ilogb` routines decode into the math family by their `___metal_` names,
/// `float` and `half` in every vector width, the exponent's `int32` shape matching the value's;
/// `ilogb` binds only the measured modes 0 and 3. Other `___metal_` names, other types, and a
/// mismatched exponent shape stay ordinary functions.
#[test]
fn metal_frexp_and_ilogb_routines_decode() {
    let routine = |name: &str| {
        let op = Math::parse_routine(name.strip_prefix("___metal_").expect("prefix"))?;
        let mut t = TypeTable::new();
        let ty = fn_ty(&mut t, &AirIntrinsic::Math(op).signature());
        assert!(is_intrinsic_name(name.as_bytes()), "{name}");
        match classify(name.as_bytes(), sym(), ty, &t) {
            Some(DeclClass::Air(AirIntrinsic::Math(m))) => Some(m),
            other => panic!("{name}: {other:?}"),
        }
    };
    for (name, f, elem, lanes) in [
        (
            "___metal_frexp_float_pthreadint32",
            MathFn::Frexp,
            Elem::F32,
            1,
        ),
        (
            "___metal_frexp_half_pthreadint32",
            MathFn::Frexp,
            Elem::F16,
            1,
        ),
        (
            "___metal_frexp_v3float_pthreadv3int32",
            MathFn::Frexp,
            Elem::F32,
            3,
        ),
        (
            "___metal_frexp_v16half_pthreadv16int32",
            MathFn::Frexp,
            Elem::F16,
            16,
        ),
        ("___metal_ilogb_float", MathFn::Ilogb, Elem::F32, 1),
        ("___metal_ilogb_v4half", MathFn::Ilogb, Elem::F16, 4),
    ] {
        let m = routine(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!((m.f, m.ty), (f, NumTy { elem, lanes }), "{name}");
    }
    for name in [
        "___metal_frexp_float_pthreadv2int32",
        "___metal_frexp_v2float_pthreadint32",
        "___metal_frexp_double_pthreadint32",
        "___metal_ilogb_int32",
        "___metal_ilogb_v1float",
        "___metal_ldexp_float",
    ] {
        assert!(routine(name).is_none(), "{name}");
        assert!(!is_intrinsic_name(name.as_bytes()), "{name}");
    }
    // `ilogb`'s mode: 0 and 3 bind, another stays an unknown call.
    let template = DeclClass::Air(AirIntrinsic::Math(routine("___metal_ilogb_float").unwrap()));
    let mut c = Consts::new();
    let x = Value::Arg(crate::ir::ArgIdx::new(0));
    for (mode, bound) in [(0, true), (3, true), (1, false), (2, false)] {
        let args = [x, c.i32(mode)];
        let got = bind(template, &args, &c.table).expect("bind");
        assert_eq!(matches!(got, Bound::Air(_)), bound, "mode {mode}");
    }
}

#[test]
fn declarations_must_match_their_signature() {
    let mut t = TypeTable::new();
    let m = &mut meter();
    let f32s = Operand::Num(NumTy::scalar(Elem::F32));
    let f16s = Operand::Num(NumTy::scalar(Elem::F16));
    let good = fn_ty(&mut t, &Signature::new(f32s, &[f32s, f32s]));
    let ret = fn_ty(&mut t, &Signature::new(f16s, &[f32s, f32s]));
    let arity = fn_ty(&mut t, &Signature::new(f32s, &[f32s]));
    let float = t.intern(Type::Float, m).unwrap();
    let vararg = t
        .intern(
            Type::Function {
                ret: float,
                params: [float, float].into(),
                vararg: true,
            },
            m,
        )
        .unwrap();
    let class = |ty| classify(b"air.fmax.f32", sym(), ty, &t).unwrap();
    assert!(matches!(class(good), DeclClass::Air(_)));
    assert_eq!(class(ret), DeclClass::Mismatch);
    assert_eq!(class(arity), DeclClass::Mismatch);
    // Vararg declarations never match.
    assert_eq!(class(vararg), DeclClass::Mismatch);
    // A mismatched declaration fails at its calls; an unknown one binds as unknown.
    let c = ConstTable::new();
    assert_eq!(
        bind(DeclClass::Mismatch, &[], &c),
        Err(IntrinsicError::SignatureMismatch)
    );
    assert_eq!(bind(DeclClass::Unknown, &[], &c), Ok(Bound::Unknown));
    // Ordinary functions are not intrinsics.
    assert_eq!(classify(b"helper", sym(), good, &t), None);
    assert_eq!(
        classify(b"air.\xff", sym(), good, &t),
        Some(DeclClass::Unknown)
    );
    // Wrong pointer space on an atomic, thread pointer on `sincos`.
    let atomic = AirIntrinsic::parse("air.atomic.global.add.u.i32").unwrap();
    let mut sig = atomic.signature();
    sig = Signature::new(sig.ret, &{
        let mut p = sig.params().to_vec();
        p[0] = Operand::Ptr(AddrSpace::Threadgroup);
        p
    });
    let wrong = fn_ty(&mut t, &sig);
    assert_eq!(
        classify(b"air.atomic.global.add.u.i32", sym(), wrong, &t),
        Some(DeclClass::Mismatch)
    );
}

/// Classify an `llvm.*` declaration of type `ret (params)`.
fn llvm(name: &str, ret: Operand, params: &[Operand]) -> DeclClass {
    let mut t = TypeTable::new();
    let ty = fn_ty(&mut t, &Signature::new(ret, params));
    classify(name.as_bytes(), sym(), ty, &t).unwrap()
}

/// The libm-like `llvm.*` functions Apple's math library defines `air.exp2`, `air.log2`,
/// `air.sqrt`, and the `half` `air.atan2` with decode as kept float operations of their LangRef
/// arity, checked against the declaration like the other float bases.
#[test]
fn llvm_math_library_bases_decode() {
    let f32s = Operand::Num(NumTy::scalar(Elem::F32));
    let f16s = Operand::Num(NumTy::scalar(Elem::F16));
    let v2f32 = Operand::Num(NumTy {
        elem: Elem::F32,
        lanes: 2,
    });
    for (name, op, ty, arity) in [
        ("llvm.exp2.f32", LlvmFloatOp::Exp2, f32s, 1),
        ("llvm.log2.f32", LlvmFloatOp::Log2, f32s, 1),
        ("llvm.sqrt.f32", LlvmFloatOp::Sqrt, f32s, 1),
        ("llvm.copysign.f32", LlvmFloatOp::CopySign, f32s, 2),
        ("llvm.exp2.f16", LlvmFloatOp::Exp2, f16s, 1),
        ("llvm.copysign.v2f32", LlvmFloatOp::CopySign, v2f32, 2),
    ] {
        let Operand::Num(nt) = ty else { unreachable!() };
        assert_eq!(
            llvm(name, ty, &vec![ty; arity]),
            DeclClass::Llvm(LlvmIntrinsic::Float { op, ty: nt }),
            "{name}"
        );
    }
    // A declaration that disagrees: the arity, the suffix, an integer overload.
    let i32s = Operand::Num(NumTy::scalar(Elem::I32));
    for (name, ret, params) in [
        ("llvm.copysign.f32", f32s, &[f32s][..]),
        ("llvm.exp2.f32", f32s, &[f32s, f32s][..]),
        ("llvm.sqrt.f16", f32s, &[f32s][..]),
        ("llvm.log2.i32", i32s, &[i32s][..]),
    ] {
        assert_eq!(llvm(name, ret, params), DeclClass::Mismatch, "{name}");
    }
}

/// `llvm.bitreverse` decodes as the kept integer operation of arity one on every integer width
/// and vector of one, checked against its declaration like the other integer bases.
#[test]
fn llvm_bitreverse_decodes_at_every_integer_width() {
    let num = |elem, lanes| Operand::Num(NumTy { elem, lanes });
    for (name, elem, lanes) in [
        ("llvm.bitreverse.i8", Elem::I8, 1),
        ("llvm.bitreverse.i16", Elem::I16, 1),
        ("llvm.bitreverse.i32", Elem::I32, 1),
        ("llvm.bitreverse.i64", Elem::I64, 1),
        ("llvm.bitreverse.v4i32", Elem::I32, 4),
        ("llvm.bitreverse.v2i64", Elem::I64, 2),
        ("llvm.bitreverse.v3i16", Elem::I16, 3),
    ] {
        let ty = num(elem, lanes);
        assert_eq!(
            llvm(name, ty, &[ty]),
            DeclClass::Llvm(LlvmIntrinsic::Int {
                op: LlvmIntOp::BitReverse,
                ty: NumTy { elem, lanes },
            }),
            "{name}"
        );
    }
    // A declaration that disagrees: the arity, the suffix, a float overload.
    let (i32s, i64s, f32s) = (num(Elem::I32, 1), num(Elem::I64, 1), num(Elem::F32, 1));
    for (name, ret, params) in [
        ("llvm.bitreverse.i32", i32s, &[i32s, i32s][..]),
        ("llvm.bitreverse.i64", i32s, &[i32s][..]),
        ("llvm.bitreverse.i32", i64s, &[i64s][..]),
        ("llvm.bitreverse.f32", f32s, &[f32s][..]),
    ] {
        assert_eq!(llvm(name, ret, params), DeclClass::Mismatch, "{name}");
    }
}

/// An integer operation on a scalar of a width no `NumTy` names (Metal's compiler leaves
/// `llvm.bitreverse.i2` in OpenCL C's fences) decodes as its narrow form, checked against its
/// declaration: every operand the result's width, the suffix spelling it.
#[test]
fn llvm_integer_operations_on_narrow_widths_decode() {
    let m = &mut meter();
    let mut t = TypeTable::new();
    let mut decl = |t: &mut TypeTable, ret: u8, params: &[u8]| {
        let ret = t.intern(Type::Int(ret), m).unwrap();
        let params: Vec<_> = params
            .iter()
            .map(|&w| t.intern(Type::Int(w), m).unwrap())
            .collect();
        let f = Type::Function {
            ret,
            params: params.into(),
            vararg: false,
        };
        t.intern(f, m).unwrap()
    };
    for (name, op, bits, arity) in [
        ("llvm.bitreverse.i2", LlvmIntOp::BitReverse, 2, 1),
        ("llvm.ctpop.i3", LlvmIntOp::Ctpop, 3, 1),
        ("llvm.bswap.i48", LlvmIntOp::Bswap, 48, 1),
        ("llvm.umax.i24", LlvmIntOp::UMax, 24, 2),
    ] {
        let ty = decl(&mut t, bits, &vec![bits; arity]);
        assert_eq!(
            classify(name.as_bytes(), sym(), ty, &t),
            Some(DeclClass::Llvm(LlvmIntrinsic::NarrowInt { op, bits })),
            "{name}"
        );
    }
    // A declaration that disagrees: an operand of another width, the suffix, the arity, a
    // `bswap` of a width that is no multiple of 16 bits.
    for (name, ret, params) in [
        ("llvm.bitreverse.i2", 2, &[3][..]),
        ("llvm.bitreverse.i3", 2, &[2][..]),
        ("llvm.bitreverse.i2", 2, &[2, 2][..]),
        ("llvm.bswap.i24", 24, &[24][..]),
    ] {
        let ty = decl(&mut t, ret, params);
        assert_eq!(
            classify(name.as_bytes(), sym(), ty, &t),
            Some(DeclClass::Mismatch),
            "{name}"
        );
    }
}

#[test]
fn llvm_bases_decode_against_their_declarations() {
    let f32s = Operand::Num(NumTy::scalar(Elem::F32));
    let v4f32 = Operand::Num(NumTy {
        elem: Elem::F32,
        lanes: 4,
    });
    let i32s = Operand::Num(NumTy::scalar(Elem::I32));
    let i64s = Operand::Num(NumTy::scalar(Elem::I64));
    let flag = Operand::Imm(ImmKind::Bool);
    let (p0, p1) = (
        Operand::Ptr(AddrSpace::Thread),
        Operand::Ptr(AddrSpace::Device),
    );
    let void = Operand::Void;
    assert!(matches!(
        llvm("llvm.fmuladd.v4f32", v4f32, &[v4f32, v4f32, v4f32]),
        DeclClass::Llvm(LlvmIntrinsic::Float {
            op: LlvmFloatOp::FMulAdd,
            ..
        })
    ));
    assert!(matches!(
        llvm("llvm.umin.i32", i32s, &[i32s, i32s]),
        DeclClass::Llvm(LlvmIntrinsic::Int {
            op: LlvmIntOp::UMin,
            ..
        })
    ));
    assert!(matches!(
        llvm("llvm.usub.sat.i32", i32s, &[i32s, i32s]),
        DeclClass::Llvm(LlvmIntrinsic::Int {
            op: LlvmIntOp::USubSat,
            ..
        })
    ));
    assert!(matches!(
        llvm("llvm.ctlz.i32", i32s, &[i32s, flag]),
        DeclClass::Llvm(LlvmIntrinsic::Ctlz { .. })
    ));
    // Opaque and typed pointer manglings.
    for name in ["llvm.memcpy.p0.p1.i64", "llvm.memcpy.p0i8.p1i8.i64"] {
        assert_eq!(
            llvm(name, void, &[p0, p1, i64s, flag]),
            DeclClass::Llvm(LlvmIntrinsic::Memcpy {
                dst: AddrSpace::Thread,
                src: AddrSpace::Device,
                len: Elem::I64,
                volatile: false,
            }),
            "{name}"
        );
    }
    let i8s = Operand::Num(NumTy::scalar(Elem::I8));
    assert!(matches!(
        llvm("llvm.memset.p1.i32", void, &[p1, i8s, i32s, flag]),
        DeclClass::Llvm(LlvmIntrinsic::Memset { .. })
    ));
    // The suffix must spell the declaration.
    for (name, ret, params) in [
        ("llvm.fabs.f16", f32s, &[f32s][..]),
        ("llvm.fabs", f32s, &[f32s][..]),
        ("llvm.fabs.f32.f32", f32s, &[f32s][..]),
        ("llvm.fabs.i32", i32s, &[i32s][..]),
        ("llvm.smax.i32", i32s, &[i32s][..]),
        ("llvm.memcpy.p1.p1.i64", void, &[p0, p1, i64s, flag][..]),
        ("llvm.memcpy.p0.p1.i32", void, &[p0, p1, i64s, flag][..]),
        (
            "llvm.assume.i1",
            void,
            &[Operand::Num(NumTy::scalar(Elem::I1))][..],
        ),
        ("llvm.lifetime.start.p0", i32s, &[p0][..]),
    ] {
        assert_eq!(llvm(name, ret, params), DeclClass::Mismatch, "{name}");
    }
    // Outside the closed set: other bases and overload types the model does not name.
    let mut t = TypeTable::new();
    let m = &mut meter();
    let double = t.intern(Type::Double, m).unwrap();
    let f64fn = t
        .intern(
            Type::Function {
                ret: double,
                params: [double].into(),
                vararg: false,
            },
            m,
        )
        .unwrap();
    assert_eq!(
        classify(b"llvm.fabs.f64", sym(), f64fn, &t),
        Some(DeclClass::Unknown)
    );
    assert_eq!(llvm("llvm.floor.f32", f32s, &[f32s]), DeclClass::Unknown);
    assert_eq!(llvm("llvm.fabsx.f32", f32s, &[f32s]), DeclClass::Unknown);
    assert_eq!(llvm("llvm.agx", f32s, &[f32s]), DeclClass::Unknown);
    // AGX: an operation without a modeled contract is kept unmodeled, whatever the declaration;
    // a modeled one checks its declaration.
    for name in [
        "llvm.agx3.wide.load.with.emask.global.v8i32",
        "llvm.agx3.load.with.emask.global.v1i32",
        "llvm.agx3.cluster.num",
        "llvm.agx2.edgecheck",
        "llvm.agx2.f16matmad8x8.v2f32",
        "llvm.agx2.f64matmad8x8.v2f64",
        "llvm.agx2.f32matmad16x16.v2f32",
        "llvm.agx3.igemm.v8i32.i64.i64.v8i32",
    ] {
        assert_eq!(
            llvm(name, v4f32, &[p1]),
            DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::Unmodeled(sym()))),
            "{name}"
        );
    }
    assert_eq!(
        llvm("llvm.agx3.load.with.emask.global.v4i32", v4f32, &[p1]),
        DeclClass::Mismatch
    );
    let (i16s, i32s) = (
        Operand::Num(NumTy::scalar(Elem::I16)),
        Operand::Num(NumTy::scalar(Elem::I32)),
    );
    let v4i32 = Operand::Num(NumTy {
        elem: Elem::I32,
        lanes: 4,
    });
    let p3 = Operand::Ptr(AddrSpace::Threadgroup);
    let ty = NumTy {
        elem: Elem::I32,
        lanes: 4,
    };
    assert_eq!(
        llvm(
            "llvm.agx3.load.with.emask.global.v4i32",
            v4i32,
            &[p1, i16s, i16s, i16s]
        ),
        DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::LoadEmask {
            space: AgxSpace::Global,
            ty
        }))
    );
    assert_eq!(
        llvm(
            "llvm.agx3.store.with.emask.local.v4i32",
            Operand::Void,
            &[p3, v4i32, i16s, i16s, i16s]
        ),
        DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::StoreEmask {
            space: AgxSpace::Local,
            ty
        }))
    );
    assert_eq!(
        llvm("llvm.agx3.edgecheck", i16s, &[i32s, i32s, i32s]),
        DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::EdgeCheck))
    );
    // The `agx2` cluster number and matrix multiply-adds.
    assert_eq!(
        llvm("llvm.agx2.cluster.num", i32s, &[]),
        DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::ClusterNum))
    );
    assert_eq!(
        llvm("llvm.agx2.cluster.num", i16s, &[]),
        DeclClass::Mismatch
    );
    for (name, elem, dim) in [
        ("llvm.agx2.f16matmad4x4.v2f16", Elem::F16, 4),
        ("llvm.agx2.f16matmad8x8.v2f16", Elem::F16, 8),
        ("llvm.agx2.f32matmad4x4.v2f32", Elem::F32, 4),
        ("llvm.agx2.f32matmad8x8.v2f32", Elem::F32, 8),
    ] {
        let v2 = Operand::Num(NumTy { elem, lanes: 2 });
        assert_eq!(
            llvm(name, v2, &[v2, v2, v2]),
            DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::MatMad { elem, dim })),
            "{name}"
        );
        assert_eq!(llvm(name, v2, &[v2, v2]), DeclClass::Mismatch, "{name}");
    }
}

#[test]
fn markers_are_dropped() {
    let (p0, i64s, void) = (
        Operand::Ptr(AddrSpace::Thread),
        Operand::Num(NumTy::scalar(Elem::I64)),
        Operand::Void,
    );
    for (name, params) in [
        ("llvm.lifetime.start.p0", &[p0][..]),
        ("llvm.lifetime.end.p0i8", &[i64s, p0][..]),
    ] {
        assert!(
            matches!(llvm(name, void, params), DeclClass::Drop(_)),
            "{name}"
        );
    }
    assert_eq!(
        llvm(
            "llvm.assume",
            void,
            &[Operand::Num(NumTy::scalar(Elem::I1))]
        ),
        DeclClass::Drop(Dropped::Assume)
    );
    // `llvm.experimental.noalias.scope.decl(metadata)`: no operand kind names a metadata
    // parameter, so its type is built here.
    let mut t = TypeTable::new();
    let m = &mut meter();
    let (ret, md) = (
        t.intern(Type::Void, m).unwrap(),
        t.intern(Type::Metadata, m).unwrap(),
    );
    let decl = t
        .intern(
            Type::Function {
                ret,
                params: [md].into(),
                vararg: false,
            },
            m,
        )
        .unwrap();
    assert_eq!(
        classify(b"llvm.experimental.noalias.scope.decl", sym(), decl, &t).unwrap(),
        DeclClass::Drop(Dropped::NoAliasScopeDecl)
    );
    // `lifetime.start.i64(i64)` is not a lifetime marker's shape.
    assert_eq!(
        llvm("llvm.lifetime.start.i64", void, &[i64s]),
        DeclClass::Mismatch
    );
    // `llvm.agx3.yield(i16)`, a delay with no data effect (the refinement), is dropped; its
    // declaration must be `void (i16)`, and a suffixed or other-generation name is no yield.
    let i16s = Operand::Num(NumTy::scalar(Elem::I16));
    assert_eq!(
        llvm("llvm.agx3.yield", void, &[i16s]),
        DeclClass::Drop(Dropped::AgxYield)
    );
    for (ret, params) in [(void, &[i64s][..]), (i16s, &[i16s][..]), (void, &[][..])] {
        assert_eq!(
            llvm("llvm.agx3.yield", ret, params),
            DeclClass::Mismatch,
            "{ret:?} {params:?}"
        );
    }
    for name in ["llvm.agx3.yield.i16", "llvm.agx2.yield"] {
        assert_eq!(
            llvm(name, void, &[i16s]),
            DeclClass::Llvm(LlvmIntrinsic::Agx(AgxOp::Unmodeled(sym()))),
            "{name}"
        );
    }
    let bases: Vec<&str> = BASES.iter().map(|(n, _)| *n).collect();
    assert_eq!(bases.len(), 32);
}

#[test]
fn handle_positions_are_typed_seeds() {
    let texture = HandleKind::Texture(ImageShape {
        dim: ImageDim::D2,
        array: false,
        ms: false,
        depth: false,
    });
    let f2 = Operand::Num(NumTy {
        elem: Elem::F32,
        lanes: 2,
    });
    let sig = Signature::new(
        Operand::Num(NumTy {
            elem: Elem::F32,
            lanes: 4,
        }),
        &[
            Operand::Handle(texture),
            Operand::Handle(HandleKind::Sampler),
            f2,
            Operand::Imm(ImmKind::Bool),
        ],
    );
    let handles: Vec<_> = sig.handles().collect();
    assert_eq!(handles, vec![(0, texture), (1, HandleKind::Sampler)]);
    assert_eq!(
        sig.immediates().collect::<Vec<_>>(),
        vec![(3, ImmKind::Bool)]
    );
    // Handles check as pointers in AIR's handle spaces.
    let mut t = TypeTable::new();
    let ty = fn_ty(&mut t, &sig);
    assert!(sig.matches(&t, ty));
    let swapped = Signature::new(
        sig.ret,
        &[
            Operand::Handle(HandleKind::Sampler),
            Operand::Handle(texture),
            f2,
            Operand::Imm(ImmKind::Bool),
        ],
    );
    assert!(!swapped.matches(&t, ty));
    // The families decoded so far have no handle operands.
    for name in [
        "air.atomic.global.add.u.i32",
        "air.sincos.f32",
        "air.convert.f.f32.u.i32",
    ] {
        assert_eq!(air(name).signature().handles().count(), 0, "{name}");
    }
}

#[test]
fn type_tokens_round_trip() {
    for &elem in &Elem::ALL {
        for lanes in [1, 2, 3, 4, 64] {
            let ty = NumTy { elem, lanes };
            assert_eq!(NumTy::parse(&token(ty)), Some(ty));
            let mut t = TypeTable::new();
            let id = operand_ty(&mut t, Operand::Num(ty), &mut meter());
            assert_eq!(NumTy::of(&t, id), Some(ty));
            assert!(ty.matches(&t, id));
        }
    }
    for bad in [
        "", "v", "v2", "vf32", "v0f32", "v1f32", "f64", "i128", "v2v2f32",
    ] {
        assert_eq!(NumTy::parse(bad), None, "{bad}");
    }
}

#[test]
fn dropped_family_names_only_the_markers_decode_drops() {
    use super::{dropped_family, Dropped};
    assert_eq!(
        dropped_family("llvm.lifetime.start.p0"),
        Some(Dropped::LifetimeStart)
    );
    assert_eq!(
        dropped_family("llvm.lifetime.end.p0i8"),
        Some(Dropped::LifetimeEnd)
    );
    assert_eq!(dropped_family("llvm.assume"), Some(Dropped::Assume));
    assert_eq!(
        dropped_family("llvm.experimental.noalias.scope.decl"),
        Some(Dropped::NoAliasScopeDecl)
    );
    assert_eq!(dropped_family("llvm.agx3.yield"), Some(Dropped::AgxYield));
    assert_eq!(dropped_family("llvm.agx3.yield.i16"), None);
    assert_eq!(dropped_family("llvm.agx3.igemm.v8i32.i64.i64.v8i32"), None);
    // Kept intrinsics, a base-name prefix that is not followed by `.`, and non-llvm symbols.
    assert_eq!(dropped_family("llvm.memcpy.p0.p0.i64"), None);
    assert_eq!(dropped_family("llvm.assumed"), None);
    assert_eq!(dropped_family("air.lifetime.start"), None);
}
