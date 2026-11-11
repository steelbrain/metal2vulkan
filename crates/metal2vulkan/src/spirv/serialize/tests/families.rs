//! `spirv-val` over every builder constructor family (the P8b hand-back): each test builds modules
//! that use every constructor of one family in its admitted forms, serializes them, and requires
//! `spirv-val` to accept them under Vulkan 1.2 and 1.3 with exactly the relaxations their derived
//! requirements enable. A failure is a builder, requirement, or serializer bug.

use super::{assert_spirv_valid, float, int, vec};
use crate::spirv::builder::ops::group::GroupMode;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder, ProductForm};
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, Constant, Contraction, ConvOp, ExecMode, ExecModel, GlslOp, GroupArith,
    MemAccess, QuadDirection, Scope, Semantics, ShuffleOp, StorageClass, Type, TypedModule, UnOp,
};

/// Common types of one module.
#[derive(Clone, Copy)]
pub(super) struct Tys {
    pub void: TypeId,
    pub bool_: TypeId,
    pub bvec4: TypeId,
    pub u8_: TypeId,
    pub u16_: TypeId,
    pub i16_: TypeId,
    pub u32_: TypeId,
    pub i32_: TypeId,
    pub u64_: TypeId,
    pub i64_: TypeId,
    pub f16: TypeId,
    pub f32_: TypeId,
    pub f64_: TypeId,
    pub uvec2: TypeId,
    pub uvec4: TypeId,
    pub ivec4: TypeId,
    pub vec2: TypeId,
    pub vec3: TypeId,
    pub vec4: TypeId,
    pub hvec4: TypeId,
    pub mat4: TypeId,
}

pub(super) fn tys(b: &mut ModuleBuilder) -> Tys {
    let void = b.ty(Type::Void).expect("void");
    let bool_ = b.ty(Type::Bool).expect("bool");
    let u32_ = int(b, 32, false);
    let i32_ = int(b, 32, true);
    let f32_ = float(b, 32);
    let f16 = float(b, 16);
    let f64_ = float(b, 64);
    let vec4 = vec(b, f32_, 4);
    Tys {
        void,
        bool_,
        bvec4: vec(b, bool_, 4),
        u8_: int(b, 8, false),
        u16_: int(b, 16, false),
        i16_: int(b, 16, true),
        u32_,
        i32_,
        u64_: int(b, 64, false),
        i64_: int(b, 64, true),
        f16,
        f32_,
        f64_,
        uvec2: vec(b, u32_, 2),
        uvec4: vec(b, u32_, 4),
        ivec4: vec(b, i32_, 4),
        vec2: vec(b, f32_, 2),
        vec3: vec(b, f32_, 3),
        vec4,
        hvec4: vec(b, f16, 4),
        mat4: b.ty(Type::Matrix { col: vec4, cols: 4 }).expect("mat4"),
    }
}

/// A helper function taking `params`, running `body` on them; the entry (`model`, `modes`)
/// calls it with null constants. The helper keeps every value opaque to constant folding.
pub(super) fn with_params(
    model: ExecModel,
    modes: Vec<ExecMode>,
    params: impl FnOnce(&Tys) -> Vec<TypeId>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &[Value], &Tys),
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ptys = params(&t);
    let helper = {
        let mut f = b.function(t.void).expect("helper");
        let vals: Vec<Value> = ptys.iter().map(|ty| f.param(*ty).expect("param")).collect();
        body(&mut f, &vals, &t);
        if f.current_block().is_some() {
            f.ret(None).expect("ret");
        }
        f.finish().expect("finish helper")
    };
    entry_calling(b, &t, model, modes, helper, &ptys)
}

fn entry_calling(
    mut b: ModuleBuilder,
    t: &Tys,
    model: ExecModel,
    mut modes: Vec<ExecMode>,
    helper: FuncId,
    ptys: &[TypeId],
) -> TypedModule {
    let mut f = b.function(t.void).expect("main");
    let args: Vec<Value> = ptys
        .iter()
        .map(|ty| {
            let c = f
                .module()
                .constant(Constant::Null { ty: *ty })
                .expect("null");
            f.module().const_value(c).expect("value")
        })
        .collect();
    f.call(helper, &args).expect("call");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish main");
    if model == ExecModel::GLCompute
        && !modes
            .iter()
            .any(|m| matches!(m, ExecMode::LocalSize { .. }))
    {
        modes.push(ExecMode::LocalSize { x: 8, y: 8, z: 1 });
    }
    b.entry(model, main, modes).expect("entry");
    b.finalize().expect("finalize")
}

fn compute(
    params: impl FnOnce(&Tys) -> Vec<TypeId>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &[Value], &Tys),
) -> TypedModule {
    with_params(ExecModel::GLCompute, vec![], params, body)
}

// ---- Arithmetic, bitwise, logical (ops/arith.rs) ----

#[test]
fn unary_operations() {
    let m = compute(
        |t| {
            vec![
                t.i32_, t.ivec4, t.i16_, t.i64_, t.f32_, t.vec4, t.f16, t.f64_, t.u32_, t.uvec4,
                t.u64_, t.bool_, t.bvec4,
            ]
        },
        |f, p, _| {
            let [i, iv, i16v, i64v, x, xv, h, d, u, uv, u64v, bo, bv] = p else {
                unreachable!()
            };
            for a in [i, iv, i16v, i64v] {
                f.unary(UnOp::SNegate, *a).expect("snegate");
            }
            for a in [x, xv, h, d] {
                f.unary(UnOp::FNegate, *a).expect("fnegate");
                f.unary(UnOp::IsNan, *a).expect("isnan");
                f.unary(UnOp::IsInf, *a).expect("isinf");
            }
            for a in [u, uv, u64v, i] {
                f.unary(UnOp::Not, *a).expect("not");
            }
            for a in [u, uv, i] {
                f.unary(UnOp::BitReverse, *a).expect("bitreverse");
                f.unary(UnOp::BitCount, *a).expect("bitcount");
            }
            for a in [bo, bv] {
                f.unary(UnOp::LogicalNot, *a).expect("logicalnot");
            }
            f.unary(UnOp::Any, *bv).expect("any");
            f.unary(UnOp::All, *bv).expect("all");
            for a in p {
                f.unary(UnOp::CopyObject, *a).expect("copyobject");
            }
            f.unary(UnOp::QuantizeToF16, *x).expect("quantize");
            f.unary(UnOp::QuantizeToF16, *xv).expect("quantize vec");
        },
    );
    assert_spirv_valid(m);
}

#[test]
fn binary_operations() {
    let m = compute(
        |t| {
            vec![
                t.u32_, t.i32_, t.uvec4, t.ivec4, t.u64_, t.u16_, t.f32_, t.vec4, t.f16, t.f64_,
                t.bool_, t.bvec4, t.u8_,
            ]
        },
        |f, p, _| {
            let [u, i, uv, iv, u64v, u16v, x, xv, h, d, bo, bv, u8v] = p else {
                unreachable!()
            };
            use BinOp::*;
            for (a, b) in [
                (u, u),
                (i, i),
                (uv, uv),
                (iv, iv),
                (u64v, u64v),
                (u16v, u16v),
            ] {
                for op in [
                    IAdd,
                    ISub,
                    IMul,
                    UDiv,
                    SDiv,
                    UMod,
                    SRem,
                    SMod,
                    ShiftRightLogical,
                    ShiftRightArithmetic,
                    ShiftLeftLogical,
                    BitwiseOr,
                    BitwiseXor,
                    BitwiseAnd,
                    IEqual,
                    INotEqual,
                    UGreaterThan,
                    SGreaterThan,
                    UGreaterThanEqual,
                    SGreaterThanEqual,
                    ULessThan,
                    SLessThan,
                    ULessThanEqual,
                    SLessThanEqual,
                ] {
                    let signed = *a == *i || *a == *iv;
                    let r = f.binary(op, *a, *b);
                    if signed && matches!(op, UDiv | UMod) {
                        assert!(r.is_err(), "{op:?} needs an unsigned result type");
                    } else {
                        r.unwrap_or_else(|e| panic!("{op:?}: {e}"));
                    }
                }
            }
            f.binary(ShiftLeftLogical, *u, *u8v)
                .expect("mixed-width shift");
            for (a, b) in [(u, u), (uv, uv)] {
                for op in [IAddCarry, ISubBorrow, UMulExtended] {
                    f.binary(op, *a, *b)
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
            }
            for (a, b) in [(i, i), (iv, iv)] {
                f.binary(SMulExtended, *a, *b).expect("smulextended");
            }
            for (a, b) in [(x, x), (xv, xv), (h, h), (d, d)] {
                for op in [
                    FAdd,
                    FSub,
                    FMul,
                    FDiv,
                    FRem,
                    FOrdEqual,
                    FUnordEqual,
                    FOrdNotEqual,
                    FUnordNotEqual,
                    FOrdLessThan,
                    FUnordLessThan,
                    FOrdGreaterThan,
                    FUnordGreaterThan,
                    FOrdLessThanEqual,
                    FUnordLessThanEqual,
                    FOrdGreaterThanEqual,
                    FUnordGreaterThanEqual,
                ] {
                    f.binary(op, *a, *b)
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
            }
            for (a, b) in [(bo, bo), (bv, bv)] {
                for op in [LogicalEqual, LogicalNotEqual, LogicalOr, LogicalAnd] {
                    f.binary(op, *a, *b)
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
            }
            f.binary(VectorTimesScalar, *xv, *x).expect("vts");
            f.binary(Dot, *xv, *xv).expect("dot");
        },
    );
    assert_spirv_valid(m);
}

/// `NoContraction`: every float arithmetic opcode takes it, `spirv-val` accepts the decoration, and
/// exactly the decorated results carry it; other opcodes refuse it. A multiply rounded on its own
/// takes the module's [`ProductForm`]: an `OpFMul` decorated `NoContraction` by default, or
/// `GLSL.std.450 Fma(a, b, -0.0)` decorated `NoContraction`, its addend the `-0.0` of its type,
/// where the module's SPIR-V reaches Metal through SPIRV-Cross; a product of two values no
/// constant gives is the `OpFMul` under [`ProductForm::PositiveZeroFactors`] too. An `Fma` built
/// without the decoration stays undecorated, and no other `GLSL.std.450` instruction takes it.
#[test]
fn no_contraction_decorates_float_arithmetic() {
    for form in [
        ProductForm::Multiply,
        ProductForm::NegativeZeroFma,
        ProductForm::PositiveZeroFactors,
    ] {
        no_contraction_under(form);
    }
}

fn no_contraction_under(form: ProductForm) {
    use crate::spirv::error::{BuildError, OperandRule};
    let mut decorated = Vec::new();
    let mut plain = Vec::new();
    let mut fused = Vec::new();
    let mut plain_fma = None;
    let m = compute(
        |t| vec![t.f32_, t.vec4, t.f16, t.f64_, t.u32_],
        |f, p, _| {
            f.module().product_form(form);
            let [x, xv, h, d, u] = p else { unreachable!() };
            for a in [x, xv, h, d] {
                for op in [BinOp::FAdd, BinOp::FSub, BinOp::FDiv, BinOp::FRem] {
                    let r = f
                        .binary_no_contraction(op, *a, *a)
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                    decorated.push(r);
                }
                plain.push(f.binary(BinOp::FMul, *a, *a).expect("fmul"));
                let m = f.binary_no_contraction(BinOp::FMul, *a, *a);
                fused.push(m.expect("fmul without contraction"));
            }
            let vts = f.binary_no_contraction(BinOp::VectorTimesScalar, *xv, *x);
            decorated.push(vts.expect("vts"));
            plain_fma = Some(f.glsl(GlslOp::Fma, &[*x, *x, *x]).expect("fma"));
            assert!(matches!(
                f.glsl_with(GlslOp::Sqrt, &[*x], Contraction::Forbidden),
                Err(BuildError::ExtOperand {
                    rule: OperandRule::NoContraction,
                    ..
                })
            ));
            for (op, a) in [(BinOp::IAdd, u), (BinOp::FOrdLessThan, x), (BinOp::Dot, xv)] {
                assert!(matches!(
                    f.binary_no_contraction(op, *a, *a),
                    Err(BuildError::Operand {
                        rule: OperandRule::NoContraction,
                        ..
                    })
                ));
            }
        },
    );
    let words = assert_spirv_valid(m);
    // Collect the `NoContraction` targets, the results of float arithmetic, the `Fma` results
    // with their addends, and every constant's last value word.
    let (mut targets, mut arith) = (Vec::new(), Vec::new());
    let (mut fmas, mut constants) = (Vec::new(), std::collections::BTreeMap::new());
    const GLSL_FMA: u32 = 50;
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        if opcode == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            targets.push(words[i + 1]);
        }
        if opcode == crate::spv::Op::Constant as u32 {
            constants.insert(words[i + 2], words[i + len - 1]);
        }
        if opcode == crate::spv::Op::ConstantComposite as u32 {
            if let Some(&w) = constants.get(&words[i + 3]) {
                constants.insert(words[i + 2], w);
            }
        }
        if opcode == crate::spv::Op::ExtInst as u32 && words[i + 4] == GLSL_FMA {
            fmas.push((words[i + 2], words[i + 7]));
        }
        if [
            crate::spv::Op::FAdd,
            crate::spv::Op::FSub,
            crate::spv::Op::FMul,
            crate::spv::Op::FDiv,
            crate::spv::Op::FRem,
            crate::spv::Op::VectorTimesScalar,
        ]
        .iter()
        .any(|o| *o as u32 == opcode)
        {
            arith.push(words[i + 2]);
        }
        i += len;
    }
    assert!(plain_fma.is_some());
    // Results are emitted in order, each operand's group holding its four decorated results,
    // its plain multiply, and (as an `OpFMul`) its multiply without contraction.
    let group = match form {
        ProductForm::Multiply | ProductForm::PositiveZeroFactors => 6,
        ProductForm::NegativeZeroFma => 5,
    };
    let products = match form {
        ProductForm::Multiply | ProductForm::PositiveZeroFactors => fused.len(),
        ProductForm::NegativeZeroFma => 0,
    };
    assert_eq!(
        arith.len(),
        decorated.len() + plain.len() + products,
        "{form:?}"
    );
    let plain_ids: Vec<u32> = arith
        .chunks(group)
        .filter_map(|c| c.get(4).copied())
        .collect();
    assert_eq!(plain_ids.len(), plain.len());
    assert_eq!(targets.len(), decorated.len() + fused.len(), "{form:?}");
    // Each multiply without contraction is, in the `Fma` form, one `Fma` decorated
    // `NoContraction` whose addend is `-0.0`: the sign bit alone in its last word (`half`,
    // `float`, `double`, and a `float` vector). The plain `Fma`, emitted last, is undecorated.
    let (fused_fmas, plain_fmas) = fmas.split_at(fmas.len() - 1);
    assert_eq!(fused_fmas.len(), fused.len() - products, "{form:?}");
    let signs = [0x8000_0000u32, 0x8000_0000, 0x8000, 0x8000_0000];
    for (&(r, addend), sign) in fused_fmas.iter().zip(signs) {
        assert!(targets.contains(&r), "%{r} not decorated");
        assert_eq!(constants.get(&addend), Some(&sign), "%{r}'s addend");
    }
    assert!(
        !targets.contains(&plain_fmas[0].0),
        "the plain Fma decorated"
    );
    if form != ProductForm::NegativeZeroFma {
        // No `-0.0` addend at all: each multiply is the `OpFMul` after its group's plain one.
        assert!(!constants
            .values()
            .any(|w| [0x8000_0000, 0x8000].contains(w)));
        for c in arith.chunks(group).filter(|c| c.len() == group) {
            assert!(targets.contains(&c[5]), "%{} not decorated", c[5]);
        }
    }
    for t in &targets {
        assert!(
            (arith.contains(t) && !plain_ids.contains(t)) || fused_fmas.iter().any(|f| f.0 == *t),
            "%{t} decorated"
        );
    }
}

#[test]
fn select_and_bit_fields() {
    let m = compute(
        |t| vec![t.bool_, t.bvec4, t.u32_, t.i32_, t.uvec4, t.vec4, t.mat4],
        |f, p, _| {
            let [bo, bv, u, i, uv, xv, m] = p else {
                unreachable!()
            };
            f.select(*bo, *u, *u).expect("scalar select");
            f.select(*bo, *xv, *xv).expect("scalar cond, vector values");
            f.select(*bv, *xv, *xv).expect("vector select");
            f.select(*bo, *m, *m).expect("composite select");
            f.bit_field_insert(*u, *u, *u, *u).expect("insert");
            f.bit_field_insert(*uv, *uv, *u, *u).expect("insert vec");
            f.bit_field_extract(true, *i, *u, *u).expect("sextract");
            f.bit_field_extract(false, *uv, *u, *u).expect("uextract");
        },
    );
    assert_spirv_valid(m);
}

// ---- Conversions (ops/convert.rs) ----

#[test]
fn conversions() {
    let m = compute(
        |t| {
            vec![
                t.f32_, t.u32_, t.i32_, t.u8_, t.u16_, t.u64_, t.i64_, t.f16, t.f64_, t.uvec2,
                t.vec4, t.hvec4,
            ]
        },
        |f, p, t| {
            let [x, u, i, u8v, u16v, u64v, i64v, h, d, uv2, xv, hv] = p else {
                unreachable!()
            };
            use ConvOp::*;
            f.convert(ConvertFToU, *x, t.u32_).expect("ftou");
            f.convert(ConvertFToS, *x, t.i32_).expect("ftos");
            f.convert(ConvertFToU, *h, t.u16_).expect("half ftou");
            f.convert(ConvertSToF, *i, t.f32_).expect("stof");
            f.convert(ConvertUToF, *u, t.f64_).expect("utof");
            f.convert(UConvert, *u, t.u64_).expect("widen");
            f.convert(UConvert, *u64v, t.u32_).expect("narrow");
            f.convert(UConvert, *u8v, t.u32_).expect("u8 widen");
            f.convert(UConvert, *u16v, t.u32_).expect("u16 widen");
            f.convert(SConvert, *i, t.i64_).expect("swiden");
            f.convert(SConvert, *i64v, t.i32_).expect("snarrow");
            f.convert(FConvert, *x, t.f16).expect("to half");
            f.convert(FConvert, *d, t.f32_).expect("from double");
            f.convert(FConvert, *xv, t.hvec4).expect("vector fconvert");
            f.convert(FConvert, *hv, t.vec4).expect("vector widen");
            f.convert(Bitcast, *u, t.f32_).expect("u32 as f32");
            f.convert(Bitcast, *x, t.i32_).expect("f32 as i32");
            f.convert(Bitcast, *uv2, t.u64_).expect("uvec2 as u64");
            f.convert(Bitcast, *u64v, t.uvec2).expect("u64 as uvec2");
            f.convert(Bitcast, *uv2, t.f64_).expect("uvec2 as f64");
        },
    );
    assert_spirv_valid(m);
}

#[test]
fn physical_storage_buffer_pointers() {
    let m = compute(
        |t| vec![t.u64_, t.uvec2, t.u32_],
        |f, p, t| {
            let [addr, pair, u] = p else { unreachable!() };
            let pty = f
                .module()
                .ty(Type::Pointer {
                    class: StorageClass::PhysicalStorageBuffer,
                    pointee: t.u32_,
                })
                .expect("psb pointer");
            let ptr = f
                .convert(ConvOp::ConvertUToPtr, *addr, pty)
                .expect("utoptr");
            let aligned = MemAccess {
                aligned: Some(4),
                ..MemAccess::default()
            };
            let v = f.load(ptr, aligned).expect("psb load");
            let v = f.binary(BinOp::IAdd, v, *u).expect("add");
            f.store(ptr, v, aligned).expect("psb store");
            f.convert(ConvOp::Bitcast, ptr, t.uvec2)
                .expect("ptr as uvec2");
            f.convert(ConvOp::Bitcast, *pair, pty)
                .expect("uvec2 as ptr");
            // A local holding a PSB pointer is decorated `AliasedPointer` by the serializer.
            let var = f.local_var(pty, None).expect("pointer var");
            let vp = f.var_ptr(var).expect("var");
            f.store(vp, ptr, MemAccess::default())
                .expect("store pointer");
        },
    );
    assert_spirv_valid(m);
}

/// Device places (a `Block { T }` member at the address) take aligned loads and stores and integer,
/// 64-bit, and float atomics.
#[test]
fn device_places() {
    let m = compute(
        |t| vec![t.u64_, t.u32_, t.f32_],
        |f, p, t| {
            let [addr, u, x] = p else { unreachable!() };
            let aligned = MemAccess {
                aligned: Some(4),
                ..MemAccess::default()
            };
            let word = f.device_place(*addr, t.u32_).expect("word place");
            let v = f.load(word, aligned).expect("load");
            let v = f.binary(BinOp::IAdd, v, *u).expect("add");
            f.store(word, v, aligned).expect("store");
            f.atomic(AtomicOp::Or, word, Scope::Device, Semantics::RELAXED, *u)
                .expect("atomic or");
            let wide = f.device_place(*addr, t.u64_).expect("u64 place");
            let one = super::scalar(f.module(), t.u64_, 1);
            let one = f.module().const_value(one).expect("one");
            f.atomic(AtomicOp::IAdd, wide, Scope::Device, Semantics::RELAXED, one)
                .expect("u64 atomic");
            let float = f.device_place(*addr, t.f32_).expect("f32 place");
            f.atomic(AtomicOp::FAdd, float, Scope::Device, Semantics::RELAXED, *x)
                .expect("float add");
        },
    );
    assert_spirv_valid(m);
}

// ---- Composites (ops/composite.rs) ----

#[test]
fn composites() {
    let m = compute(
        |t| vec![t.f32_, t.vec2, t.vec4, t.u32_, t.uvec4, t.mat4],
        |f, p, t| {
            let [x, v2, v4, u, uv, m] = p else {
                unreachable!()
            };
            let a = f
                .composite_construct(t.vec4, &[*x, *x, *x, *x])
                .expect("vec4");
            f.composite_construct(t.vec4, &[*v2, *x, *x])
                .expect("mixed");
            f.composite_construct(t.vec4, &[*v2, *v2]).expect("halves");
            f.composite_construct(t.mat4, &[a, a, a, *v4])
                .expect("matrix");
            let lc = super::scalar(f.module(), t.u32_, 3);
            let arr = f
                .module()
                .ty(Type::Array {
                    elem: t.u32_,
                    len: lc,
                    stride: None,
                })
                .expect("array");
            let arr_v = f
                .composite_construct(arr, &[*u, *u, *u])
                .expect("array value");
            let st = f
                .module()
                .ty(Type::Struct {
                    members: [t.u32_, t.vec4, arr].into(),
                    layout: None,
                })
                .expect("struct");
            let s = f
                .composite_construct(st, &[*u, a, arr_v])
                .expect("struct value");
            f.composite_extract(s, &[1, 2]).expect("extract nested");
            f.composite_extract(*m, &[3, 0]).expect("extract matrix");
            f.composite_insert(*x, s, &[1, 0]).expect("insert");
            f.vector_extract_dynamic(*v4, *u).expect("dyn extract");
            f.vector_insert_dynamic(*uv, *u, *u).expect("dyn insert");
            f.vector_shuffle(*v4, *v2, &[0, 5, 2]).expect("shuffle");
            f.vector_shuffle(*v2, *v2, &[3, 0]).expect("shuffle 2");
            f.undef(t.vec4).expect("undef");
            f.undef(st).expect("undef struct");
        },
    );
    assert_spirv_valid(m);
}

// ---- GLSL.std.450 (ops/glsl.rs) ----

#[test]
fn glsl_extended_instructions() {
    let m = compute(
        |t| {
            vec![
                t.f32_, t.vec4, t.vec3, t.f16, t.f64_, t.i32_, t.u32_, t.ivec4, t.vec2,
            ]
        },
        |f, p, _| {
            let [x, xv, x3, h, d, i, u, iv, x2] = p else {
                unreachable!()
            };
            use GlslOp::*;
            let floats = [x, xv];
            for a in floats {
                for op in [
                    Round,
                    RoundEven,
                    Trunc,
                    FAbs,
                    FSign,
                    Floor,
                    Ceil,
                    Fract,
                    Sqrt,
                    InverseSqrt,
                    Normalize,
                    Radians,
                    Degrees,
                    Sin,
                    Cos,
                    Tan,
                    Asin,
                    Acos,
                    Atan,
                    Sinh,
                    Cosh,
                    Tanh,
                    Asinh,
                    Acosh,
                    Atanh,
                    Exp,
                    Log,
                    Exp2,
                    Log2,
                    Length,
                    ModfStruct,
                    FrexpStruct,
                ] {
                    f.glsl(op, &[*a]).unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
                for op in [Atan2, Pow, FMin, FMax, NMin, NMax, Step, Reflect, Distance] {
                    f.glsl(op, &[*a, *a])
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
                for op in [FClamp, NClamp, FMix, SmoothStep, Fma, FaceForward] {
                    f.glsl(op, &[*a, *a, *a])
                        .unwrap_or_else(|e| panic!("{op:?}: {e}"));
                }
                f.glsl(Refract, &[*a, *a, *x]).expect("refract");
            }
            for a in [h, d] {
                for op in [FAbs, Floor, Sqrt, Fract] {
                    f.glsl(op, &[*a])
                        .unwrap_or_else(|e| panic!("{op:?} wide: {e}"));
                }
                f.glsl(Fma, &[*a, *a, *a]).expect("fma wide");
            }
            f.glsl(Ldexp, &[*x, *i]).expect("ldexp");
            f.glsl(Ldexp, &[*xv, *iv]).expect("ldexp vec");
            f.glsl(Cross, &[*x3, *x3]).expect("cross");
            for a in [i, iv] {
                f.glsl(SAbs, &[*a]).expect("sabs");
                f.glsl(SSign, &[*a]).expect("ssign");
                f.glsl(FindILsb, &[*a]).expect("findilsb");
                f.glsl(FindSMsb, &[*a]).expect("findsmsb");
                f.glsl(SMin, &[*a, *a]).expect("smin");
                f.glsl(SMax, &[*a, *a]).expect("smax");
                f.glsl(SClamp, &[*a, *a, *a]).expect("sclamp");
            }
            f.glsl(FindUMsb, &[*u]).expect("findumsb");
            f.glsl(UMin, &[*u, *u]).expect("umin");
            f.glsl(UMax, &[*u, *u]).expect("umax");
            f.glsl(UClamp, &[*u, *u, *u]).expect("uclamp");
            f.glsl(PackSnorm4x8, &[*xv]).expect("packsnorm4x8");
            f.glsl(PackUnorm4x8, &[*xv]).expect("packunorm4x8");
            f.glsl(PackSnorm2x16, &[*x2]).expect("packsnorm2x16");
            f.glsl(PackUnorm2x16, &[*x2]).expect("packunorm2x16");
            f.glsl(PackHalf2x16, &[*x2]).expect("packhalf2x16");
            for op in [
                UnpackSnorm2x16,
                UnpackUnorm2x16,
                UnpackHalf2x16,
                UnpackSnorm4x8,
                UnpackUnorm4x8,
            ] {
                f.glsl(op, &[*u]).unwrap_or_else(|e| panic!("{op:?}: {e}"));
            }
        },
    );
    assert_spirv_valid(m);
}

#[test]
fn glsl_matrix_instructions() {
    let m = compute(
        |t| vec![t.mat4],
        |f, p, _| {
            f.glsl(GlslOp::Determinant, &[p[0]]).expect("determinant");
            f.glsl(GlslOp::MatrixInverse, &[p[0]]).expect("inverse");
        },
    );
    assert_spirv_valid(m);
}

// ---- Subgroup operations (ops/group.rs) ----

#[test]
fn subgroup_operations() {
    let m = compute(
        |t| {
            vec![
                t.bool_, t.u32_, t.i32_, t.f32_, t.vec4, t.u16_, t.f16, t.u64_,
            ]
        },
        |f, p, _| {
            let [bo, u, i, x, xv, u16v, h, u64v] = p else {
                unreachable!()
            };
            f.group_elect().expect("elect");
            f.group_vote(true, *bo).expect("all");
            f.group_vote(false, *bo).expect("any");
            f.group_broadcast_first(*xv).expect("broadcast first");
            let ballot = f.group_ballot(*bo).expect("ballot");
            f.group_ballot_bit_extract(ballot, *u).expect("bit extract");
            for op in [ShuffleOp::Shuffle, ShuffleOp::Xor] {
                f.group_shuffle(op, *xv, *u).expect("shuffle");
            }
            let modes = [
                GroupMode::Reduce,
                GroupMode::InclusiveScan,
                GroupMode::ExclusiveScan,
                GroupMode::Clustered(4),
            ];
            for mode in modes {
                for (op, v) in [
                    (GroupArith::IAdd, u),
                    (GroupArith::SMin, i),
                    (GroupArith::UMin, u),
                    (GroupArith::SMax, i),
                    (GroupArith::UMax, u),
                    (GroupArith::BitwiseAnd, u),
                    (GroupArith::BitwiseOr, u),
                    (GroupArith::BitwiseXor, u),
                    (GroupArith::FAdd, x),
                    (GroupArith::FMin, x),
                    (GroupArith::FMax, x),
                    (GroupArith::LogicalAnd, bo),
                    (GroupArith::LogicalOr, bo),
                    (GroupArith::LogicalXor, bo),
                ] {
                    f.group_arith(op, mode, *v)
                        .unwrap_or_else(|e| panic!("{op:?} {mode:?}: {e}"));
                }
            }
            // Extended types.
            f.group_arith(GroupArith::IAdd, GroupMode::Reduce, *u16v)
                .expect("u16 add");
            f.group_arith(GroupArith::FAdd, GroupMode::Reduce, *h)
                .expect("f16 add");
            f.group_arith(GroupArith::IAdd, GroupMode::Reduce, *u64v)
                .expect("u64 add");
            f.group_quad_broadcast(*x, *u).expect("quad broadcast");
            for dir in [
                QuadDirection::Horizontal,
                QuadDirection::Vertical,
                QuadDirection::Diagonal,
            ] {
                f.group_quad_swap(*xv, dir).expect("quad swap");
            }
        },
    );
    assert_spirv_valid(m);
}
