//! L0 tests of the typed instruction constructors: each family derives its result types and rejects
//! ill-typed operands at construction.

use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::ops::group::GroupMode;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, EntryRule, ImageRule, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, Contraction, ConvOp, DerivOp, Dim,
    ExecMode, ExecModel, GlslOp, GroupArith, ImageOperandKind as K, ImageOperands, ImageQuery,
    ImageType, ImageUsage, Inst, IoDir, LayoutRule, MemAccess, MemberLayout, MemoryOrder,
    ModelNeed, QuadDirection, Scope, Semantics, ShuffleOp, StorageClass, StructLayout, TexelMemory,
    Type, UnOp,
};

use crate::spv::{ImageFormat, Op};

// ---- Fixture ----

fn int(b: &mut ModuleBuilder, width: u8, signed: bool) -> TypeId {
    b.ty(Type::Int { width, signed }).expect("int")
}
fn float(b: &mut ModuleBuilder, width: u8) -> TypeId {
    b.ty(Type::Float { width }).expect("float")
}
fn vec(b: &mut ModuleBuilder, elem: TypeId, len: u8) -> TypeId {
    b.ty(Type::Vector { elem, len }).expect("vector")
}
fn ptr(b: &mut ModuleBuilder, class: StorageClass, pointee: TypeId) -> TypeId {
    b.ty(Type::Pointer { class, pointee }).expect("pointer")
}
fn uconst(b: &mut ModuleBuilder, v: u64) -> crate::spirv::ids::ConstId {
    let ty = int(b, 32, false);
    b.constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })
    .expect("constant")
}

/// Common scalar and vector types of one module.
#[derive(Clone, Copy)]
struct Tys {
    void: TypeId,
    bool_: TypeId,
    u8_: TypeId,
    u16_: TypeId,
    u32_: TypeId,
    i32_: TypeId,
    u64_: TypeId,
    f16: TypeId,
    f32_: TypeId,
    f64_: TypeId,
    uvec2: TypeId,
    uvec3: TypeId,
    uvec4: TypeId,
    ivec2: TypeId,
    bvec2: TypeId,
    bvec3: TypeId,
    vec2: TypeId,
    vec3: TypeId,
    vec4: TypeId,
}

fn tys(b: &mut ModuleBuilder) -> Tys {
    let void = b.ty(Type::Void).expect("void");
    let bool_ = b.ty(Type::Bool).expect("bool");
    let u32_ = int(b, 32, false);
    let i32_ = int(b, 32, true);
    let f32_ = float(b, 32);
    Tys {
        void,
        bool_,
        u8_: int(b, 8, false),
        u16_: int(b, 16, false),
        u32_,
        i32_,
        u64_: int(b, 64, false),
        f16: float(b, 16),
        f32_,
        f64_: float(b, 64),
        uvec2: vec(b, u32_, 2),
        uvec3: vec(b, u32_, 3),
        uvec4: vec(b, u32_, 4),
        ivec2: vec(b, i32_, 2),
        bvec2: vec(b, bool_, 2),
        bvec3: vec(b, bool_, 3),
        vec2: vec(b, f32_, 2),
        vec3: vec(b, f32_, 3),
        vec4: vec(b, f32_, 4),
    }
}

/// Run `body` inside a void function of a fresh module.
fn in_fn(body: impl FnOnce(&mut FunctionBuilder<'_>, Tys)) {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut f = b.function(t.void).expect("function");
    body(&mut f, t);
}

fn p(f: &mut FunctionBuilder<'_>, ty: TypeId) -> Value {
    f.param(ty).expect("param")
}

fn ty(f: &FunctionBuilder<'_>, v: Value) -> TypeId {
    f.value_ty(v).expect("typed value")
}

fn op_err(op: Op, rule: OperandRule) -> Result<Value, BuildError> {
    Err(BuildError::Operand { op, rule })
}

fn img_err(op: Op, r: ImageRule) -> Result<Value, BuildError> {
    op_err(op, OperandRule::Image(r))
}

fn cval(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
        .expect("constant");
    f.module().const_value(c).expect("constant value")
}

fn mty(f: &mut FunctionBuilder<'_>, t: Type) -> TypeId {
    f.module().ty(t).expect("type")
}

// ---- Arithmetic ----

#[test]
fn integer_and_float_arithmetic_derive_the_first_operand_type() {
    in_fn(|f, t| {
        let (u, i, u64v) = (p(f, t.u32_), p(f, t.i32_), p(f, t.u64_));
        let (x, h) = (p(f, t.f32_), p(f, t.f16));
        let r = f.binary(BinOp::IAdd, u, i).expect("mixed signedness");
        assert_eq!(ty(f, r), t.u32_);
        let r = f.binary(BinOp::SDiv, i, u).expect("sdiv");
        assert_eq!(ty(f, r), t.i32_);
        assert_eq!(
            f.binary(BinOp::IAdd, u, u64v),
            op_err(Op::IAdd, OperandRule::Mismatch(1))
        );
        assert_eq!(
            f.binary(BinOp::IAdd, x, x),
            op_err(Op::IAdd, OperandRule::Category(0))
        );
        let r = f.binary(BinOp::FMul, x, x).expect("fmul");
        assert_eq!(ty(f, r), t.f32_);
        assert_eq!(
            f.binary(BinOp::FAdd, x, h),
            op_err(Op::FAdd, OperandRule::Mismatch(1))
        );
        let b8 = p(f, t.u8_);
        let r = f
            .binary(BinOp::ShiftLeftLogical, u, b8)
            .expect("shift widths may differ");
        assert_eq!(ty(f, r), t.u32_);
        let uv = p(f, t.uvec2);
        assert_eq!(
            f.binary(BinOp::ShiftLeftLogical, uv, u),
            op_err(Op::ShiftLeftLogical, OperandRule::Mismatch(1))
        );
        let bo = p(f, t.bool_);
        let r = f.binary(BinOp::LogicalAnd, bo, bo).expect("logical");
        assert_eq!(ty(f, r), t.bool_);
        assert_eq!(
            f.binary(BinOp::LogicalAnd, u, u),
            op_err(Op::LogicalAnd, OperandRule::Category(0))
        );
        assert_eq!(
            f.binary(BinOp::BitwiseAnd, bo, bo),
            op_err(Op::BitwiseAnd, OperandRule::Category(0))
        );
    });
}

#[test]
fn comparisons_derive_bool_of_the_component_count() {
    in_fn(|f, t| {
        let a = p(f, t.uvec3);
        let r = f.binary(BinOp::ULessThan, a, a).expect("ult");
        assert_eq!(ty(f, r), t.bvec3);
        let x = p(f, t.f32_);
        let r = f.binary(BinOp::FOrdLessThan, x, x).expect("flt");
        assert_eq!(ty(f, r), t.bool_);
        let u = p(f, t.u32_);
        assert_eq!(
            f.binary(BinOp::IEqual, a, u),
            op_err(Op::IEqual, OperandRule::Mismatch(1))
        );
        assert_eq!(
            f.binary(BinOp::FOrdEqual, u, u),
            op_err(Op::FOrdEqual, OperandRule::Category(0))
        );
    });
}

#[test]
fn extended_arithmetic_derives_a_pair_struct() {
    in_fn(|f, t| {
        let (u, i) = (p(f, t.u32_), p(f, t.i32_));
        let r = f.binary(BinOp::IAddCarry, u, u).expect("iaddcarry");
        let pair = mty(
            f,
            Type::Struct {
                members: [t.u32_, t.u32_].into(),
                layout: None,
            },
        );
        assert_eq!(ty(f, r), pair);
        assert_eq!(
            f.binary(BinOp::UMulExtended, i, i),
            op_err(Op::UMulExtended, OperandRule::Category(0))
        );
        assert_eq!(
            f.binary(BinOp::IAddCarry, u, i),
            op_err(Op::IAddCarry, OperandRule::Mismatch(1))
        );
        f.binary(BinOp::SMulExtended, i, i).expect("smulextended");
    });
}

#[test]
fn linear_algebra_derives_shapes() {
    in_fn(|f, t| {
        let (v3, s) = (p(f, t.vec3), p(f, t.f32_));
        let r = f.binary(BinOp::VectorTimesScalar, v3, s).expect("v*s");
        assert_eq!(ty(f, r), t.vec3);
        let r = f.binary(BinOp::Dot, v3, v3).expect("dot");
        assert_eq!(ty(f, r), t.f32_);
        assert_eq!(
            f.binary(BinOp::Dot, s, s),
            op_err(Op::Dot, OperandRule::Category(0))
        );
    });
}

#[test]
fn unary_operations_check_class_and_width() {
    in_fn(|f, t| {
        let (u, h, v2, bv) = (p(f, t.u16_), p(f, t.f16), p(f, t.vec2), p(f, t.bvec2));
        assert_eq!(
            f.unary(UnOp::BitCount, u),
            op_err(Op::BitCount, OperandRule::Width(0))
        );
        assert_eq!(
            f.unary(UnOp::QuantizeToF16, h),
            op_err(Op::QuantizeToF16, OperandRule::Width(0))
        );
        let r = f.unary(UnOp::IsNan, v2).expect("isnan");
        assert_eq!(ty(f, r), t.bvec2);
        let b = p(f, t.bool_);
        assert_eq!(
            f.unary(UnOp::Any, b),
            op_err(Op::Any, OperandRule::Category(0))
        );
        let r = f.unary(UnOp::All, bv).expect("all");
        assert_eq!(ty(f, r), t.bool_);
        assert_eq!(
            f.unary(UnOp::FNegate, u),
            op_err(Op::FNegate, OperandRule::Category(0))
        );
        let lp = mty(
            f,
            Type::Pointer {
                class: StorageClass::Function,
                pointee: t.u32_,
            },
        );
        let lp = p(f, lp);
        assert_eq!(
            f.unary(UnOp::CopyObject, lp),
            op_err(Op::CopyObject, OperandRule::Category(0))
        );
    });
}

#[test]
fn select_needs_a_matching_bool_condition_and_data_objects() {
    in_fn(|f, t| {
        let (c, cv, a) = (p(f, t.bool_), p(f, t.bvec2), p(f, t.uvec2));
        let r = f.select(cv, a, a).expect("vector select");
        assert_eq!(ty(f, r), t.uvec2);
        let r = f.select(c, a, a).expect("scalar condition");
        assert_eq!(ty(f, r), t.uvec2);
        let a3 = p(f, t.uvec3);
        assert_eq!(
            f.select(cv, a3, a3),
            op_err(Op::Select, OperandRule::Mismatch(0))
        );
        assert_eq!(
            f.select(c, a, a3),
            op_err(Op::Select, OperandRule::Mismatch(2))
        );
        let u = p(f, t.u32_);
        assert_eq!(
            f.select(u, a, a),
            op_err(Op::Select, OperandRule::Category(0))
        );
        let s = mty(
            f,
            Type::Struct {
                members: [t.u32_, t.f32_].into(),
                layout: None,
            },
        );
        let sv = p(f, s);
        f.select(c, sv, sv).expect("composite select (SPIR-V 1.4+)");
        let lp = mty(
            f,
            Type::Pointer {
                class: StorageClass::Private,
                pointee: t.u32_,
            },
        );
        let lp = p(f, lp);
        assert_eq!(
            f.select(c, lp, lp),
            op_err(Op::Select, OperandRule::Category(1))
        );
    });
}

#[test]
fn bit_fields_take_32_bit_bases() {
    in_fn(|f, t| {
        let (u, i, w) = (p(f, t.u32_), p(f, t.i32_), p(f, t.u64_));
        let r = f.bit_field_insert(u, u, i, i).expect("insert");
        assert_eq!(ty(f, r), t.u32_);
        assert_eq!(
            f.bit_field_insert(u, i, u, u),
            op_err(Op::BitFieldInsert, OperandRule::Mismatch(1))
        );
        assert_eq!(
            f.bit_field_extract(true, w, u, u),
            op_err(Op::BitFieldSExtract, OperandRule::Width(0))
        );
        let uv = p(f, t.uvec2);
        assert_eq!(
            f.bit_field_extract(false, u, uv, u),
            op_err(Op::BitFieldUExtract, OperandRule::Category(1))
        );
    });
}

// ---- Conversions ----

#[test]
fn numeric_conversions_check_the_target() {
    in_fn(|f, t| {
        let (x, u, v2, uv2) = (p(f, t.f32_), p(f, t.u32_), p(f, t.vec2), p(f, t.uvec2));
        let (i16_, i64_) = (int_ty(f, 16, true), int_ty(f, 64, true));
        let r = f.convert(ConvOp::ConvertFToU, x, t.u32_).expect("ftou");
        assert_eq!(ty(f, r), t.u32_);
        assert_eq!(
            f.convert(ConvOp::ConvertFToU, x, t.i32_),
            op_err(Op::ConvertFToU, OperandRule::ResultType)
        );
        f.convert(ConvOp::ConvertFToS, x, t.u32_)
            .expect("ftos, any signedness");
        assert_eq!(
            f.convert(ConvOp::ConvertFToU, v2, t.u32_),
            op_err(Op::ConvertFToU, OperandRule::ResultType)
        );
        f.convert(ConvOp::ConvertFToU, v2, t.uvec2)
            .expect("vector ftou");
        assert_eq!(
            f.convert(ConvOp::UConvert, u, t.u32_),
            op_err(Op::UConvert, OperandRule::ResultType)
        );
        f.convert(ConvOp::UConvert, u, t.u16_)
            .expect("narrowing uconvert");
        assert_eq!(
            f.convert(ConvOp::UConvert, u, i16_),
            op_err(Op::UConvert, OperandRule::ResultType)
        );
        f.convert(ConvOp::SConvert, u, i64_).expect("sconvert");
        f.convert(ConvOp::FConvert, x, t.f16).expect("fconvert");
        f.convert(ConvOp::ConvertUToF, uv2, t.vec2).expect("utof");
        assert_eq!(
            f.convert(ConvOp::ConvertSToF, x, t.f32_),
            op_err(Op::ConvertSToF, OperandRule::Category(0))
        );
    });
}

fn int_ty(f: &mut FunctionBuilder<'_>, width: u8, signed: bool) -> TypeId {
    mty(f, Type::Int { width, signed })
}

#[test]
fn pointer_conversions_and_bitcasts_preserve_width() {
    in_fn(|f, t| {
        let psb = mty(
            f,
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                pointee: t.u32_,
            },
        );
        let (pp, a64, a32) = (p(f, psb), p(f, t.u64_), p(f, t.u32_));
        let lp = mty(
            f,
            Type::Pointer {
                class: StorageClass::Function,
                pointee: t.u32_,
            },
        );
        let lpv = p(f, lp);
        f.convert(ConvOp::ConvertUToPtr, a64, psb)
            .expect("u64 to ptr");
        assert_eq!(
            f.convert(ConvOp::ConvertUToPtr, a32, psb),
            op_err(Op::ConvertUToPtr, OperandRule::Width(0))
        );
        let x = p(f, t.f32_);
        f.convert(ConvOp::Bitcast, x, t.u32_).expect("f32 as u32");
        let uv2 = p(f, t.uvec2);
        f.convert(ConvOp::Bitcast, uv2, t.u64_)
            .expect("uvec2 as u64");
        f.convert(ConvOp::Bitcast, uv2, psb).expect("uvec2 as ptr");
        f.convert(ConvOp::Bitcast, pp, t.u64_).expect("ptr as u64");
        assert_eq!(
            f.convert(ConvOp::Bitcast, x, t.u64_),
            op_err(Op::Bitcast, OperandRule::ResultType)
        );
        assert_eq!(
            f.convert(ConvOp::Bitcast, pp, t.f64_),
            op_err(Op::Bitcast, OperandRule::ResultType)
        );
        assert_eq!(
            f.convert(ConvOp::Bitcast, lpv, t.u64_),
            op_err(Op::Bitcast, OperandRule::Category(0))
        );
        let b = p(f, t.bool_);
        assert_eq!(
            f.convert(ConvOp::Bitcast, b, t.u8_),
            op_err(Op::Bitcast, OperandRule::Category(0))
        );
    });
}

// ---- Composites ----

#[test]
fn composite_construct_checks_parts_against_the_type() {
    in_fn(|f, t| {
        let (v2, x, u) = (p(f, t.vec2), p(f, t.f32_), p(f, t.u32_));
        let r = f
            .composite_construct(t.vec4, &[v2, x, x])
            .expect("vec4 from vec2, f32, f32");
        assert_eq!(ty(f, r), t.vec4);
        assert_eq!(
            f.composite_construct(t.vec4, &[v2, x]),
            op_err(Op::CompositeConstruct, OperandRule::Arity)
        );
        assert_eq!(
            f.composite_construct(t.vec4, &[v2, x, u]),
            op_err(Op::CompositeConstruct, OperandRule::Mismatch(2))
        );
        let s = mty(
            f,
            Type::Struct {
                members: [t.f32_, t.u32_].into(),
                layout: None,
            },
        );
        f.composite_construct(s, &[x, u]).expect("struct");
        assert_eq!(
            f.composite_construct(s, &[u, x]),
            op_err(Op::CompositeConstruct, OperandRule::Mismatch(0))
        );
        let three = uconst(f.module(), 3);
        let arr = mty(
            f,
            Type::Array {
                elem: t.u32_,
                len: three,
                stride: None,
            },
        );
        f.composite_construct(arr, &[u, u, u]).expect("array");
        assert_eq!(
            f.composite_construct(arr, &[u, u]),
            op_err(Op::CompositeConstruct, OperandRule::Arity)
        );
        assert_eq!(
            f.composite_construct(t.u32_, &[u]),
            op_err(Op::CompositeConstruct, OperandRule::ResultType)
        );
    });
}

#[test]
fn composite_access_walks_literal_indices() {
    in_fn(|f, t| {
        let s = mty(
            f,
            Type::Struct {
                members: [t.vec4, t.u32_].into(),
                layout: None,
            },
        );
        let (sv, x, u) = (p(f, s), p(f, t.f32_), p(f, t.u32_));
        let r = f.composite_extract(sv, &[0, 2]).expect("extract");
        assert_eq!(ty(f, r), t.f32_);
        assert_eq!(
            f.composite_extract(sv, &[0, 4]),
            op_err(Op::CompositeExtract, OperandRule::Range(2))
        );
        assert_eq!(
            f.composite_extract(sv, &[2]),
            op_err(Op::CompositeExtract, OperandRule::Range(1))
        );
        assert_eq!(
            f.composite_extract(sv, &[1, 0]),
            op_err(Op::CompositeExtract, OperandRule::Category(0))
        );
        assert_eq!(
            f.composite_extract(sv, &[]),
            op_err(Op::CompositeExtract, OperandRule::Arity)
        );
        let r = f.composite_insert(x, sv, &[0, 1]).expect("insert");
        assert_eq!(ty(f, r), s);
        assert_eq!(
            f.composite_insert(u, sv, &[0, 1]),
            op_err(Op::CompositeInsert, OperandRule::Mismatch(0))
        );
        let v = p(f, t.vec3);
        let r = f.vector_extract_dynamic(v, u).expect("dynamic extract");
        assert_eq!(ty(f, r), t.f32_);
        assert_eq!(
            f.vector_insert_dynamic(v, u, u),
            op_err(Op::VectorInsertDynamic, OperandRule::Mismatch(1))
        );
        assert_eq!(
            f.vector_extract_dynamic(x, u),
            op_err(Op::VectorExtractDynamic, OperandRule::Category(0))
        );
        let w = p(f, t.vec2);
        let r = f.vector_shuffle(v, w, &[4, 0, 1, 3]).expect("shuffle");
        assert_eq!(ty(f, r), t.vec4);
        assert_eq!(
            f.vector_shuffle(v, w, &[5, 0]),
            op_err(Op::VectorShuffle, OperandRule::Range(2))
        );
        assert_eq!(
            f.vector_shuffle(v, w, &[0]),
            op_err(Op::VectorShuffle, OperandRule::Arity)
        );
        let uv = p(f, t.uvec2);
        assert_eq!(
            f.vector_shuffle(v, uv, &[0, 1]),
            op_err(Op::VectorShuffle, OperandRule::Mismatch(1))
        );
    });
}

// ---- Memory ----

/// A `StorageBuffer` block `{ u32, vec4[] }` and its resource pointer.
fn ssbo(b: &mut ModuleBuilder, t: &Tys) -> (TypeId, crate::spirv::ids::GlobalId) {
    let rta = b
        .ty(Type::RuntimeArray {
            elem: t.vec4,
            stride: Some(16),
        })
        .expect("rta");
    let members = [t.u32_, rta];
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &members).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: members.into(),
            layout: Some(layout),
        })
        .expect("block");
    let g = b
        .resource(
            ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            },
            StorageClass::StorageBuffer,
            block,
        )
        .expect("resource");
    (block, g)
}

#[test]
fn access_chains_derive_pointers_and_check_indices() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let (_, g) = ssbo(&mut b, &t);
    let base = b.global_value(g).expect("global");
    let mut f = b.function(t.void).expect("function");
    let f = &mut f;
    let one = cval(f, t.u32_, 1);
    let two = cval(f, t.u32_, 2);
    let i = p(f, t.u32_);
    let r = f
        .access_chain(base, &[one, i])
        .expect("block member element");
    let want = mty(
        f,
        Type::Pointer {
            class: StorageClass::StorageBuffer,
            pointee: t.vec4,
        },
    );
    assert_eq!(ty(f, r), want);
    let r = f
        .access_chain(base, &[one, i, two])
        .expect("vector component");
    let want = mty(
        f,
        Type::Pointer {
            class: StorageClass::StorageBuffer,
            pointee: t.f32_,
        },
    );
    assert_eq!(ty(f, r), want);
    assert_eq!(
        f.access_chain(base, &[i]),
        op_err(Op::AccessChain, OperandRule::NotConstant(1))
    );
    assert_eq!(
        f.access_chain(base, &[two]),
        op_err(Op::AccessChain, OperandRule::Range(1))
    );
    let four = cval(f, t.u32_, 4);
    assert_eq!(
        f.access_chain(base, &[one, i, four]),
        op_err(Op::AccessChain, OperandRule::Range(3))
    );
    let zero = cval(f, t.u32_, 0);
    assert_eq!(
        f.access_chain(base, &[zero, zero]),
        op_err(Op::AccessChain, OperandRule::Category(2))
    );
    assert_eq!(
        f.access_chain(base, &[]),
        op_err(Op::AccessChain, OperandRule::Arity)
    );
    let x = p(f, t.f32_);
    assert_eq!(
        f.access_chain(base, &[one, x]),
        op_err(Op::AccessChain, OperandRule::Category(2))
    );
    // A negative signed constant indexes out of range.
    let neg = cval(f, t.i32_, 0xFFFF_FFFF);
    let arr_len = uconst(f.module(), 3);
    let arr = mty(
        f,
        Type::Array {
            elem: t.u32_,
            len: arr_len,
            stride: None,
        },
    );
    let var = f.local_var(arr, None).expect("local");
    let vp = f.var_ptr(var).expect("var pointer");
    assert_eq!(
        f.access_chain(vp, &[neg]),
        op_err(Op::AccessChain, OperandRule::Range(1))
    );
    let r = f.access_chain(vp, &[two]).expect("in-range constant");
    let want = mty(
        f,
        Type::Pointer {
            class: StorageClass::Function,
            pointee: t.u32_,
        },
    );
    assert_eq!(ty(f, r), want);
}

/// The instruction defining `v` in the function under construction.
fn def_of(f: &FunctionBuilder<'_>, v: Value) -> Inst {
    f.mb()
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .find(|n| n.result == Some(v))
        .map(|n| n.inst.clone())
        .expect("defined")
}

/// A device place is the access chain to member 0 of a one-member `Block { T }` at offset 0,
/// reached by converting the address; the block takes the pointee's rule, at most std430.
#[test]
fn device_places_are_block_members_at_offset_zero() {
    in_fn(|f, t| {
        let (a64, a32) = (p(f, t.u64_), p(f, t.u32_));
        let block_of = |f: &mut FunctionBuilder<'_>, pointee: TypeId, rule: LayoutRule| {
            let layout = StructLayout {
                rule,
                block: true,
                non_writable: false,
                members: [MemberLayout {
                    offset: 0,
                    matrix_stride: None,
                }]
                .into(),
            };
            mty(
                f,
                Type::Struct {
                    members: [pointee].into(),
                    layout: Some(layout),
                },
            )
        };
        for (pointee, rule) in [
            (t.f32_, LayoutRule::Std430),
            (t.u64_, LayoutRule::Std430),
            (t.uvec3, LayoutRule::Std430),
        ] {
            let place = f.device_place(a64, pointee).expect("place");
            let want = mty(
                f,
                Type::Pointer {
                    class: StorageClass::PhysicalStorageBuffer,
                    pointee,
                },
            );
            assert_eq!(ty(f, place), want);
            let Inst::AccessChain { base, indices, .. } = def_of(f, place) else {
                panic!("a place is an access chain");
            };
            assert_eq!(indices.len(), 1);
            assert_eq!(f.const_scalar(indices[0]).expect("index"), Some(0));
            assert_eq!(
                def_of(f, base),
                Inst::Convert {
                    op: ConvOp::ConvertUToPtr,
                    a: a64,
                    contract: Contraction::Allowed,
                }
            );
            let block = block_of(f, pointee, rule);
            let block_ptr = mty(
                f,
                Type::Pointer {
                    class: StorageClass::PhysicalStorageBuffer,
                    pointee: block,
                },
            );
            assert_eq!(ty(f, base), block_ptr);
        }
        // A scalar-layout pointee keeps its rule: `{ f32, vec3 }` at offsets 0 and 4.
        let members = [t.f32_, t.vec3];
        let layout = natural_layout(f.mb().types(), LayoutRule::Scalar, false, &members)
            .expect("scalar layout");
        let s = mty(
            f,
            Type::Struct {
                members: members.into(),
                layout: Some(layout),
            },
        );
        let place = f.device_place(a64, s).expect("scalar place");
        let Inst::AccessChain { base, .. } = def_of(f, place) else {
            panic!("a place is an access chain");
        };
        let block = block_of(f, s, LayoutRule::Scalar);
        let block_ptr = mty(
            f,
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                pointee: block,
            },
        );
        assert_eq!(ty(f, base), block_ptr);
        // The address is a 64-bit integer, as for the conversion.
        assert_eq!(
            f.device_place(a32, t.u32_),
            op_err(Op::ConvertUToPtr, OperandRule::Width(0))
        );
        // Atomics take a place.
        let place = f.device_place(a64, t.f32_).expect("place");
        let one = cval(f, t.f32_, 0x3f80_0000);
        f.atomic(
            AtomicOp::Exchange,
            place,
            Scope::Device,
            Semantics::RELAXED,
            one,
        )
        .expect("float exchange on a place");
    });
}

#[test]
fn array_length_and_texel_pointer_classes() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let (_, g) = ssbo(&mut b, &t);
    let base = b.global_value(g).expect("global");
    let mut f = b.function(t.void).expect("function");
    let f = &mut f;
    let r = f.array_length(base, 1).expect("array length");
    assert_eq!(ty(f, r), t.u32_);
    assert_eq!(
        f.array_length(base, 0),
        op_err(Op::ArrayLength, OperandRule::Range(1))
    );
    // Texel pointers never load or store (only atomics use them).
    let texel = mty(
        f,
        Type::Pointer {
            class: StorageClass::Image,
            pointee: t.u32_,
        },
    );
    let tp = p(f, texel);
    assert_eq!(
        f.load(tp, MemAccess::default()),
        op_err(Op::Load, OperandRule::Class(StorageClass::Image))
    );
    let u = p(f, t.u32_);
    assert_eq!(
        f.store(tp, u, MemAccess::default()),
        Err(BuildError::Operand {
            op: Op::Store,
            rule: OperandRule::Class(StorageClass::Image)
        })
    );
    assert_eq!(
        f.access_chain(tp, &[u]),
        op_err(Op::AccessChain, OperandRule::Class(StorageClass::Image))
    );
}

#[test]
fn calls_check_the_callee_signature() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let fp = ptr(&mut b, StorageClass::Function, t.u32_);
    let callee = {
        let mut g = b.function(t.u32_).expect("callee");
        let x = g.param(t.u32_).expect("param");
        g.param(fp).expect("pointer param");
        g.ret(Some(x)).expect("ret");
        g.finish().expect("finish")
    };
    let void_callee = {
        let mut g = b.function(t.void).expect("void callee");
        g.ret(None).expect("ret");
        g.finish().expect("finish")
    };
    let mut f = b.function(t.void).expect("caller");
    let f = &mut f;
    let u = p(f, t.u32_);
    let var = f.local_var(t.u32_, None).expect("var");
    let vp = f.var_ptr(var).expect("pointer");
    let r = f.call(callee, &[u, vp]).expect("call").expect("result");
    assert_eq!(ty(f, r), t.u32_);
    assert_eq!(f.call(void_callee, &[]), Ok(None));
    assert_eq!(
        f.call(callee, &[u]),
        Err(BuildError::Operand {
            op: Op::FunctionCall,
            rule: OperandRule::Arity
        })
    );
    assert_eq!(
        f.call(callee, &[vp, vp]),
        Err(BuildError::Operand {
            op: Op::FunctionCall,
            rule: OperandRule::Mismatch(0)
        })
    );
    // A pointer argument must be a memory object declaration, not a parameter or access chain.
    let pp = p(f, fp);
    assert_eq!(
        f.call(callee, &[u, pp]),
        Err(BuildError::Operand {
            op: Op::FunctionCall,
            rule: OperandRule::Category(1)
        })
    );
    let me = f.id();
    assert_eq!(
        f.call(me, &[]),
        Err(BuildError::Operand {
            op: Op::FunctionCall,
            rule: OperandRule::Callee
        })
    );
}

// ---- Atomics and barriers ----

fn acq_rel_uniform() -> Semantics {
    Semantics {
        order: MemoryOrder::AcquireRelease,
        uniform_memory: true,
        ..Semantics::RELAXED
    }
}

#[test]
fn atomics_need_atomic_classes_and_matching_values() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let (_, g) = ssbo(&mut b, &t);
    let base = b.global_value(g).expect("global");
    let mut f = b.function(t.void).expect("function");
    let f = &mut f;
    let zero = cval(f, t.u32_, 0);
    let counter = f.access_chain(base, &[zero]).expect("counter");
    let u = p(f, t.u32_);
    let r = f
        .atomic(
            AtomicOp::IAdd,
            counter,
            Scope::Device,
            Semantics::RELAXED,
            u,
        )
        .expect("atomic add");
    assert_eq!(ty(f, r), t.u32_);
    let i = p(f, t.i32_);
    assert_eq!(
        f.atomic(
            AtomicOp::IAdd,
            counter,
            Scope::Device,
            Semantics::RELAXED,
            i
        ),
        op_err(Op::AtomicIAdd, OperandRule::Mismatch(3))
    );
    let local = f.local_var(t.u32_, None).expect("var");
    let local = f.var_ptr(local).expect("pointer");
    assert_eq!(
        f.atomic(AtomicOp::IAdd, local, Scope::Device, Semantics::RELAXED, u),
        op_err(Op::AtomicIAdd, OperandRule::Class(StorageClass::Function))
    );
    f.atomic_load(counter, Scope::Device, Semantics::RELAXED)
        .expect("load");
    let release = Semantics {
        order: MemoryOrder::Release,
        uniform_memory: true,
        ..Semantics::RELAXED
    };
    assert_eq!(
        f.atomic_load(counter, Scope::Device, release),
        op_err(Op::AtomicLoad, OperandRule::Semantics)
    );
    let acquire_nowhere = Semantics {
        order: MemoryOrder::Acquire,
        ..Semantics::RELAXED
    };
    assert_eq!(
        f.atomic_load(counter, Scope::Device, acquire_nowhere),
        op_err(Op::AtomicLoad, OperandRule::Semantics)
    );
    f.atomic_store(counter, Scope::Workgroup, release, u)
        .expect("store");
    let acquire = Semantics {
        order: MemoryOrder::Acquire,
        uniform_memory: true,
        ..Semantics::RELAXED
    };
    assert_eq!(
        f.atomic_compare_exchange(counter, Scope::Device, Semantics::RELAXED, acquire, u, u),
        op_err(Op::AtomicCompareExchange, OperandRule::Semantics)
    );
    let r = f
        .atomic_compare_exchange(counter, Scope::Device, acq_rel_uniform(), acquire, u, u)
        .expect("cmpxchg");
    assert_eq!(ty(f, r), t.u32_);
    // One constant per distinct scope or semantics value.
    let a = f.scope_id(Scope::Device).expect("scope");
    let b2 = f.scope_id(Scope::Device).expect("scope");
    assert_eq!(a, b2);
}

/// An atomic's pointer is a place (a variable, an access chain, or a texel pointer). A converted
/// address, a pointer parameter, and a selected pointer are refused by every atomic constructor;
/// the address's device place is accepted.
#[test]
fn atomics_need_places() {
    in_fn(|f, t| {
        let psb = mty(
            f,
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                pointee: t.u32_,
            },
        );
        let (addr, u, c) = (p(f, t.u64_), p(f, t.u32_), p(f, t.bool_));
        let param = p(f, psb);
        let bare = f
            .convert(ConvOp::ConvertUToPtr, addr, psb)
            .expect("converted");
        let place = f.device_place(addr, t.u32_).expect("place");
        let selected = f.select(c, place, place).expect("select");
        let r = Semantics::RELAXED;
        for ptr in [bare, param, selected] {
            assert_eq!(
                f.atomic(AtomicOp::IAdd, ptr, Scope::Device, r, u),
                op_err(Op::AtomicIAdd, OperandRule::NotPlace)
            );
            assert_eq!(
                f.atomic_load(ptr, Scope::Device, r),
                op_err(Op::AtomicLoad, OperandRule::NotPlace)
            );
            assert_eq!(
                f.atomic_store(ptr, Scope::Device, r, u),
                Err(BuildError::Operand {
                    op: Op::AtomicStore,
                    rule: OperandRule::NotPlace
                })
            );
            assert_eq!(
                f.atomic_compare_exchange(ptr, Scope::Device, r, r, u, u),
                op_err(Op::AtomicCompareExchange, OperandRule::NotPlace)
            );
        }
        f.atomic(AtomicOp::IAdd, place, Scope::Device, r, u)
            .expect("a device place");
        f.atomic_load(place, Scope::Device, r).expect("load");
        f.atomic_store(place, Scope::Device, r, u).expect("store");
        f.atomic_compare_exchange(place, Scope::Device, r, r, u, u)
            .expect("compare exchange");
        let wg = f
            .module()
            .plain_global(StorageClass::Workgroup, t.u32_, None)
            .expect("workgroup");
        let wg = f.module().global_value(wg).expect("variable");
        f.atomic(AtomicOp::IAdd, wg, Scope::Workgroup, r, u)
            .expect("a variable");
    });
}

#[test]
fn barriers_check_scopes_and_semantics() {
    in_fn(|f, _| {
        f.control_barrier(Scope::Workgroup, Scope::Workgroup, acq_rel_uniform())
            .expect("workgroup barrier");
        f.control_barrier(Scope::Subgroup, Scope::Subgroup, Semantics::RELAXED)
            .expect("execution-only barrier");
        assert_eq!(
            f.control_barrier(Scope::Device, Scope::Device, acq_rel_uniform()),
            Err(BuildError::Operand {
                op: Op::ControlBarrier,
                rule: OperandRule::Scope
            })
        );
        let volatile = Semantics {
            volatile: true,
            ..acq_rel_uniform()
        };
        assert_eq!(
            f.control_barrier(Scope::Workgroup, Scope::Workgroup, volatile),
            Err(BuildError::Operand {
                op: Op::ControlBarrier,
                rule: OperandRule::Semantics
            })
        );
        f.memory_barrier(Scope::Device, acq_rel_uniform())
            .expect("memory barrier");
        assert_eq!(
            f.memory_barrier(Scope::Device, Semantics::RELAXED),
            Err(BuildError::Operand {
                op: Op::MemoryBarrier,
                rule: OperandRule::Semantics
            })
        );
    });
}

// ---- Images ----

fn image(
    f: &mut FunctionBuilder<'_>,
    sampled_type: TypeId,
    dim: Dim,
    usage: ImageUsage,
    edit: impl FnOnce(&mut ImageType),
) -> TypeId {
    let mut img = ImageType {
        sampled_type,
        dim,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage,
        format: ImageFormat::Unknown,
    };
    edit(&mut img);
    mty(f, Type::Image(img))
}

#[test]
fn texel_pointers_need_atomic_storage_formats() {
    in_fn(|f, t| {
        let r32 = image(f, t.u32_, Dim::D2, ImageUsage::Storage, |i| {
            i.format = ImageFormat::R32ui
        });
        let rgba = image(f, t.u32_, Dim::D2, ImageUsage::Storage, |i| {
            i.format = ImageFormat::Rgba8ui
        });
        let sampled = image(f, t.u32_, Dim::D2, ImageUsage::Sampled, |_| {});
        let pr32 = mty(
            f,
            Type::Pointer {
                class: StorageClass::UniformConstant,
                pointee: r32,
            },
        );
        let prgba = mty(
            f,
            Type::Pointer {
                class: StorageClass::UniformConstant,
                pointee: rgba,
            },
        );
        let psampled = mty(
            f,
            Type::Pointer {
                class: StorageClass::UniformConstant,
                pointee: sampled,
            },
        );
        let (a, b2, c) = (p(f, pr32), p(f, prgba), p(f, psampled));
        let coord = p(f, t.uvec2);
        let zero = cval(f, t.u32_, 0);
        let one = cval(f, t.u32_, 1);
        let tp = f
            .image_texel_pointer(a, coord, zero)
            .expect("texel pointer");
        let want = mty(
            f,
            Type::Pointer {
                class: StorageClass::Image,
                pointee: t.u32_,
            },
        );
        assert_eq!(ty(f, tp), want);
        let u = p(f, t.u32_);
        f.atomic(AtomicOp::UMax, tp, Scope::Device, Semantics::RELAXED, u)
            .expect("image atomic");
        assert_eq!(
            f.image_texel_pointer(b2, coord, zero),
            img_err(Op::ImageTexelPointer, ImageRule::ImageType)
        );
        assert_eq!(
            f.image_texel_pointer(c, coord, zero),
            img_err(Op::ImageTexelPointer, ImageRule::ImageType)
        );
        assert_eq!(
            f.image_texel_pointer(a, u, zero),
            img_err(Op::ImageTexelPointer, ImageRule::Coordinate)
        );
        assert_eq!(
            f.image_texel_pointer(a, coord, one),
            op_err(Op::ImageTexelPointer, OperandRule::Range(2))
        );
    });
}

#[test]
fn sampling_checks_forms_operands_and_block() {
    in_fn(|f, t| {
        let tex2d = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |_| {});
        let depth = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |i| i.depth = true);
        let cube = image(f, t.f32_, Dim::Cube, ImageUsage::Sampled, |_| {});
        let sampler = mty(f, Type::Sampler);
        let (img, dimg, cimg, smp) = (p(f, tex2d), p(f, depth), p(f, cube), p(f, sampler));
        let si = f.sampled_image(img, smp).expect("sampled image");
        let (uv, uvw, x) = (p(f, t.vec2), p(f, t.vec3), p(f, t.f32_));
        let none = ImageOperands::default();
        let r = f.sample(si, uv, None, false, &none).expect("implicit");
        assert_eq!(ty(f, r), t.vec4);
        let biased = ImageOperands {
            bias: Some(x),
            ..none
        };
        f.sample(si, uv, None, false, &biased).expect("bias");
        assert_eq!(
            f.sample(si, uvw, None, false, &none),
            img_err(Op::ImageSampleImplicitLod, ImageRule::Coordinate)
        );
        let lod = ImageOperands {
            lod: Some(x),
            ..none
        };
        f.sample(si, uv, None, false, &lod).expect("explicit lod");
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    bias: Some(x),
                    ..lod
                }
            ),
            img_err(Op::ImageSampleExplicitLod, ImageRule::NotAllowed(K::Bias))
        );
        let grad = ImageOperands {
            grad: Some((uv, uv)),
            ..none
        };
        f.sample(
            si,
            uv,
            None,
            false,
            &ImageOperands {
                min_lod: Some(x),
                ..grad
            },
        )
        .expect("grad with min lod");
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    grad: Some((uv, uv)),
                    ..lod
                }
            ),
            img_err(Op::ImageSampleExplicitLod, ImageRule::NotAllowed(K::Grad))
        );
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    min_lod: Some(x),
                    ..lod
                }
            ),
            img_err(Op::ImageSampleExplicitLod, ImageRule::NotAllowed(K::MinLod))
        );
        let ioff = p(f, t.ivec2);
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    const_offset: Some(ioff),
                    ..none
                }
            ),
            img_err(
                Op::ImageSampleImplicitLod,
                ImageRule::OperandType(K::ConstOffset)
            )
        );
        let one = cval(f, t.i32_, 1);
        let coff = f
            .composite_construct(t.ivec2, &[one, one])
            .expect("non-constant composite");
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    const_offset: Some(coff),
                    ..none
                }
            ),
            img_err(
                Op::ImageSampleImplicitLod,
                ImageRule::OperandType(K::ConstOffset)
            )
        );
        let c1 = f
            .module()
            .constant(Constant::Scalar {
                ty: t.i32_,
                bits: ConstBits(1),
            })
            .expect("const");
        let cc = f
            .module()
            .constant(Constant::Composite {
                ty: t.ivec2,
                parts: [c1, c1].into(),
            })
            .expect("composite const");
        let cc = f.module().const_value(cc).expect("value");
        f.sample(
            si,
            uv,
            None,
            false,
            &ImageOperands {
                const_offset: Some(cc),
                ..none
            },
        )
        .expect("constant offset");
        // A dynamic `Offset` on a sample, implicit or explicit (it needs `maintenance8`, which
        // requirement derivation records), but never beside `ConstOffset`.
        let dynamic = ImageOperands {
            offset: Some(ioff),
            ..none
        };
        f.sample(si, uv, None, false, &dynamic)
            .expect("implicit dynamic offset");
        f.sample(
            si,
            uv,
            None,
            false,
            &ImageOperands {
                offset: Some(ioff),
                ..lod
            },
        )
        .expect("explicit dynamic offset");
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    const_offset: Some(cc),
                    ..dynamic
                }
            ),
            img_err(Op::ImageSampleImplicitLod, ImageRule::NotAllowed(K::Offset))
        );
        assert_eq!(
            f.sample(
                si,
                uv,
                None,
                false,
                &ImageOperands {
                    offset: Some(uv),
                    ..none
                }
            ),
            img_err(
                Op::ImageSampleImplicitLod,
                ImageRule::OperandType(K::Offset)
            )
        );
        // Offsets are signed: an unsigned vector, dynamic or constant, is refused.
        let uoff = p(f, t.uvec2);
        let u1 = f
            .module()
            .constant(Constant::Scalar {
                ty: t.u32_,
                bits: ConstBits(1),
            })
            .expect("const");
        let ucc = f
            .module()
            .constant(Constant::Composite {
                ty: t.uvec2,
                parts: [u1, u1].into(),
            })
            .expect("composite const");
        let ucc = f.module().const_value(ucc).expect("value");
        for (o, kind) in [
            (
                ImageOperands {
                    offset: Some(uoff),
                    ..none
                },
                K::Offset,
            ),
            (
                ImageOperands {
                    const_offset: Some(ucc),
                    ..none
                },
                K::ConstOffset,
            ),
        ] {
            assert_eq!(
                f.sample(si, uv, None, false, &o),
                img_err(Op::ImageSampleImplicitLod, ImageRule::OperandType(kind))
            );
        }
        assert_eq!(
            f.sample(si, uv, Some(x), false, &none),
            img_err(Op::ImageSampleDrefImplicitLod, ImageRule::ImageType)
        );
        let dsi = f.sampled_image(dimg, smp).expect("depth sampled image");
        let r = f
            .sample(dsi, uv, Some(x), false, &lod)
            .expect("dref explicit");
        assert_eq!(ty(f, r), t.f32_);
        let r = f.sample(si, uvw, None, true, &none).expect("projective");
        assert_eq!(ty(f, r), t.vec4);
        let csi = f.sampled_image(cimg, smp).expect("cube");
        assert_eq!(
            f.sample(csi, uvw, None, true, &none),
            img_err(Op::ImageSampleProjImplicitLod, ImageRule::ImageType)
        );
        assert_eq!(
            f.sample(
                csi,
                uvw,
                None,
                false,
                &ImageOperands {
                    const_offset: Some(cc),
                    ..none
                }
            ),
            img_err(
                Op::ImageSampleImplicitLod,
                ImageRule::NotAllowed(K::ConstOffset)
            )
        );
        assert_eq!(
            f.sample(csi, uvw, None, false, &dynamic),
            img_err(Op::ImageSampleImplicitLod, ImageRule::NotAllowed(K::Offset))
        );
        // A sampled image that does not come from an `OpSampledImage` in this block.
        let sit = mty(f, Type::SampledImage { image: tex2d });
        let foreign = p(f, sit);
        assert_eq!(
            f.sample(foreign, uv, None, false, &none),
            op_err(Op::ImageSampleImplicitLod, OperandRule::SampledImageBlock)
        );
    });
}

/// An armed offset (`FunctionBuilder::arm_offset`): an explicit-level sample with it is a
/// switch-at-use of one arm a value of its dynamic lane (16), each making the sampled image again
/// and sampling with that value as `ConstOffset` (none for 0), merged by one `OpPhi`; an
/// implicit-level sample with it is refused in the switch; an offset without a dynamic lane, or of
/// another lane count, is not armed.
#[test]
fn an_armed_offset_samples_in_constant_arms() {
    use crate::spirv::builder::ops::image::ArmedOffset;
    use crate::spirv::error::ControlRule;
    in_fn(|f, t| {
        let tex2d = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |_| {});
        let sampler = mty(f, Type::Sampler);
        let (img, smp) = (p(f, tex2d), p(f, sampler));
        let (uv, x, off, sel) = (p(f, t.vec2), p(f, t.f32_), p(f, t.ivec2), p(f, t.u32_));
        let lanes = vec![None, Some(0)];
        assert!(f
            .arm_offset(
                off,
                ArmedOffset {
                    selector: sel,
                    lanes: vec![Some(1), Some(0)],
                }
            )
            .is_err());
        assert!(f
            .arm_offset(
                off,
                ArmedOffset {
                    selector: sel,
                    lanes: vec![None],
                }
            )
            .is_err());
        f.arm_offset(
            off,
            ArmedOffset {
                selector: sel,
                lanes,
            },
        )
        .expect("armed");
        let si = f.sampled_image(img, smp).expect("sampled image");
        let lod = ImageOperands {
            lod: Some(x),
            offset: Some(off),
            ..ImageOperands::default()
        };
        let r = f.sample(si, uv, None, false, &lod).expect("armed sample");
        assert_eq!(ty(f, r), t.vec4);
        let si = f.sampled_image(img, smp).expect("sampled image");
        let implicit = ImageOperands {
            offset: Some(off),
            ..ImageOperands::default()
        };
        assert_eq!(
            f.sample(si, uv, None, false, &implicit),
            Err(BuildError::Control(ControlRule::TangledInSwitchAtUse))
        );
    });
}

#[test]
fn fetch_gather_read_write_and_queries() {
    in_fn(|f, t| {
        let tex2d = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |i| {
            i.arrayed = true
        });
        let ms = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |i| {
            i.multisampled = true
        });
        let storage = image(f, t.u32_, Dim::D2, ImageUsage::Storage, |_| {});
        let scube = image(f, t.u32_, Dim::Cube, ImageUsage::Storage, |_| {});
        let tex3d = image(f, t.f32_, Dim::D3, ImageUsage::Sampled, |_| {});
        let sampler = mty(f, Type::Sampler);
        let (img, msi, st, sc, t3, smp) = (
            p(f, tex2d),
            p(f, ms),
            p(f, storage),
            p(f, scube),
            p(f, tex3d),
            p(f, sampler),
        );
        let (c2, c3, u, i, uv3) = (
            p(f, t.ivec2),
            p(f, t.uvec3),
            p(f, t.u32_),
            p(f, t.i32_),
            p(f, t.vec3),
        );
        let none = ImageOperands::default();
        let lod = ImageOperands {
            lod: Some(i),
            ..none
        };
        let r = f.fetch(img, c3, &lod).expect("fetch arrayed");
        assert_eq!(ty(f, r), t.vec4);
        assert_eq!(
            f.fetch(img, uv3, &lod),
            img_err(Op::ImageFetch, ImageRule::Coordinate)
        );
        assert_eq!(
            f.fetch(st, c2, &none),
            img_err(Op::ImageFetch, ImageRule::ImageType)
        );
        assert_eq!(
            f.fetch(msi, c2, &none),
            img_err(Op::ImageFetch, ImageRule::Missing(K::Sample))
        );
        assert_eq!(
            f.fetch(
                msi,
                c2,
                &ImageOperands {
                    sample: Some(u),
                    ..lod
                }
            ),
            img_err(Op::ImageFetch, ImageRule::NotAllowed(K::Lod))
        );
        f.fetch(
            msi,
            c2,
            &ImageOperands {
                sample: Some(u),
                ..none
            },
        )
        .expect("multisampled fetch");
        assert_eq!(
            f.fetch(
                img,
                c3,
                &ImageOperands {
                    sample: Some(u),
                    ..none
                }
            ),
            img_err(Op::ImageFetch, ImageRule::NotAllowed(K::Sample))
        );
        let si = f.sampled_image(img, smp).expect("sampled");
        let r = f.gather(si, uv3, 1, &none).expect("gather");
        assert_eq!(ty(f, r), t.vec4);
        f.gather(
            si,
            uv3,
            1,
            &ImageOperands {
                offset: Some(c2),
                ..none
            },
        )
        .expect("dynamic gather offset");
        // An unsigned dynamic gather offset is refused: MSL's `gather` takes an `int2`.
        let uoff = p(f, t.uvec2);
        assert_eq!(
            f.gather(
                si,
                uv3,
                1,
                &ImageOperands {
                    offset: Some(uoff),
                    ..none
                },
            ),
            img_err(Op::ImageGather, ImageRule::OperandType(K::Offset))
        );
        assert_eq!(
            f.gather(si, uv3, 4, &none),
            img_err(Op::ImageGather, ImageRule::Value)
        );
        let si3 = f.sampled_image(t3, smp).expect("3d");
        assert_eq!(
            f.gather(si3, uv3, 0, &none),
            img_err(Op::ImageGather, ImageRule::ImageType)
        );
        let texel = p(f, t.uvec4);
        let rd = f.read(st, c2, &none, TexelMemory::PRIVATE).expect("read");
        assert_eq!(ty(f, rd), t.uvec4);
        f.write(st, c2, texel, &none, TexelMemory::PRIVATE)
            .expect("write");
        f.write(sc, c3, texel, &none, TexelMemory::PRIVATE)
            .expect("cube write takes 3 coords");
        let v4 = p(f, t.vec4);
        assert_eq!(
            f.write(st, c2, v4, &none, TexelMemory::PRIVATE)
                .map(|_| Value::new(0)),
            img_err(Op::ImageWrite, ImageRule::Value)
        );
        assert_eq!(
            f.read(img, c3, &none, TexelMemory::PRIVATE),
            img_err(Op::ImageRead, ImageRule::ImageType)
        );
        let r = f.query(ImageQuery::SizeLod(i), img).expect("size lod");
        assert_eq!(ty(f, r), t.uvec3);
        let r = f.query(ImageQuery::Size, sc).expect("storage cube size");
        assert_eq!(ty(f, r), t.uvec2);
        assert_eq!(
            f.query(ImageQuery::Size, img),
            img_err(Op::ImageQuerySize, ImageRule::ImageType)
        );
        assert_eq!(
            f.query(ImageQuery::Levels, st),
            img_err(Op::ImageQueryLevels, ImageRule::ImageType)
        );
        let r = f.query(ImageQuery::Samples, msi).expect("samples");
        assert_eq!(ty(f, r), t.u32_);
        assert_eq!(
            f.query(ImageQuery::Samples, img),
            img_err(Op::ImageQuerySamples, ImageRule::ImageType)
        );
        let x = p(f, t.f32_);
        assert_eq!(
            f.query(ImageQuery::SizeLod(x), img),
            img_err(Op::ImageQuerySizeLod, ImageRule::OperandType(K::Lod))
        );
        let si = f.sampled_image(img, smp).expect("sampled");
        let uv = p(f, t.vec2);
        let r = f.query_lod(si, uv).expect("query lod");
        assert_eq!(ty(f, r), t.vec2);
        let h = p(f, t.f16);
        assert_eq!(
            f.derivative(DerivOp::DPdx, h),
            op_err(Op::DPdx, OperandRule::Width(0))
        );
    });
}

/// The Vulkan memory model's texel operands ([`TexelMemory`]) ride only on a storage read or write,
/// so a sampled image's fetch cannot carry them, `MakeTexel*` always comes with `NonPrivateTexel`,
/// and the opcode decides which of the two it is (a read's is `MakeTexelVisible`). Every form is
/// admitted on a storage image and recorded on the instruction; subpass data, which no invocation
/// writes, takes none.
#[test]
fn texel_operands_belong_to_storage_accesses() {
    in_fn(|f, t| {
        let storage = image(f, t.u32_, Dim::D2, ImageUsage::Storage, |_| {});
        let subpass = image(f, t.f32_, Dim::SubpassData, ImageUsage::Storage, |_| {});
        let (st, sp) = (p(f, storage), p(f, subpass));
        let c2 = p(f, t.ivec2);
        let texel = p(f, t.uvec4);
        let origin = f
            .module()
            .constant(Constant::Null { ty: t.ivec2 })
            .expect("origin");
        let origin = f.module().const_value(origin).expect("origin");
        let device = f.scope_id(Scope::Device).expect("scope");
        let none = ImageOperands::default();
        let forms = [
            TexelMemory::PRIVATE,
            TexelMemory::Private { volatile: true },
            TexelMemory::NonPrivate { volatile: false },
            TexelMemory::NonPrivate { volatile: true },
            TexelMemory::made(device, false),
            TexelMemory::made(device, true),
        ];
        for mem in forms {
            f.read(st, c2, &none, mem).expect("read");
            f.sparse_read(st, c2, &none, mem).expect("sparse read");
            f.write(st, c2, texel, &none, mem).expect("write");
        }
        let recorded: Vec<TexelMemory> = f
            .mb()
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|n| match n.inst {
                Inst::ImageRead { mem, .. } | Inst::ImageWrite { mem, .. } => Some(mem),
                _ => None,
            })
            .collect();
        let want: Vec<TexelMemory> = forms.iter().flat_map(|&m| [m; 3]).collect();
        assert_eq!(recorded, want);
        assert_eq!(TexelMemory::made(device, true).made_scope(), Some(device));
        f.read(sp, origin, &none, TexelMemory::PRIVATE)
            .expect("subpass read");
        for (mem, kind) in [
            (TexelMemory::Private { volatile: true }, K::VolatileTexel),
            (
                TexelMemory::NonPrivate { volatile: false },
                K::NonPrivateTexel,
            ),
            (TexelMemory::made(device, false), K::MakeTexelVisible),
        ] {
            assert_eq!(
                f.read(sp, origin, &none, mem),
                img_err(Op::ImageRead, ImageRule::NotAllowed(kind))
            );
        }
    });
}

// ---- Execution-model needs ----

/// Build a module whose entry runs `body`, with `model` and `modes`.
fn entry_with(
    model: ExecModel,
    modes: Vec<ExecMode>,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Tys),
) -> Result<(), BuildError> {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let mut f = b.function(t.void)?;
    body(&mut f, t);
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(model, id, modes)?;
    b.finalize().map(|_| ())
}

fn implicit_sample(f: &mut FunctionBuilder<'_>, t: Tys) {
    let tex = image(f, t.f32_, Dim::D2, ImageUsage::Sampled, |_| {});
    let sampler = mty(f, Type::Sampler);
    let slot = |kind, index| ResourceSlot {
        stage: Stage::Fragment,
        kind,
        index,
    };
    let tg = f
        .module()
        .resource(
            slot(ResourceKind::Texture, 0),
            StorageClass::UniformConstant,
            tex,
        )
        .expect("texture");
    let sg = f
        .module()
        .resource(
            slot(ResourceKind::Sampler, 0),
            StorageClass::UniformConstant,
            sampler,
        )
        .expect("sampler");
    let tp = f.module().global_value(tg).expect("texture pointer");
    let sp = f.module().global_value(sg).expect("sampler pointer");
    let img = f.load(tp, MemAccess::default()).expect("load texture");
    let smp = f.load(sp, MemAccess::default()).expect("load sampler");
    let si = f.sampled_image(img, smp).expect("sampled image");
    let uv = f.undef(t.vec2).expect("coordinate");
    f.sample(si, uv, None, false, &ImageOperands::default())
        .expect("implicit sample");
}

#[test]
fn implicit_derivatives_need_fragment_or_a_derivative_group() {
    let local = vec![ExecMode::LocalSize { x: 2, y: 2, z: 1 }];
    let deriv = |f: &mut FunctionBuilder<'_>, t: Tys| {
        let x = f.undef(t.f32_).expect("undef");
        f.derivative(DerivOp::DPdx, x).expect("derivative");
    };
    assert_eq!(
        entry_with(ExecModel::GLCompute, local.clone(), deriv),
        Err(BuildError::Entry(EntryRule::InstructionForModel(Op::DPdx)))
    );
    assert_eq!(
        entry_with(ExecModel::Vertex, vec![], deriv),
        Err(BuildError::Entry(EntryRule::InstructionForModel(Op::DPdx)))
    );
    entry_with(ExecModel::Fragment, vec![], deriv).expect("fragment derivative");
    let mut linear = local.clone();
    linear.push(ExecMode::DerivativeGroupLinear);
    entry_with(ExecModel::GLCompute, linear, deriv).expect("compute derivative group");
    let interlock = |f: &mut FunctionBuilder<'_>, _: Tys| {
        f.begin_interlock().expect("begin");
        f.end_interlock().expect("end");
    };
    assert_eq!(
        entry_with(ExecModel::Fragment, vec![], interlock),
        Err(BuildError::Entry(EntryRule::InstructionForModel(
            Op::BeginInvocationInterlockEXT
        )))
    );
    entry_with(
        ExecModel::Fragment,
        vec![ExecMode::PixelInterlockOrdered],
        interlock,
    )
    .expect("interlock with mode");
    entry_with(ExecModel::Fragment, vec![], implicit_sample).expect("fragment sampling");
    assert_eq!(
        entry_with(ExecModel::GLCompute, local, implicit_sample),
        Err(BuildError::Entry(EntryRule::InstructionForModel(
            Op::ImageSampleImplicitLod
        )))
    );
}

#[test]
fn tangled_and_model_needs_are_classified() {
    in_fn(|f, t| {
        let x = p(f, t.f32_);
        let d = f.derivative(DerivOp::DPdy, x).expect("derivative");
        let e = f.group_elect().expect("elect");
        let a = f.binary(BinOp::FAdd, x, x).expect("add");
        let insts: Vec<(Value, Inst)> = f
            .mb()
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|n| n.result.map(|r| (r, n.inst.clone())))
            .collect();
        let find = |v: Value| {
            insts
                .iter()
                .find(|(r, _)| *r == v)
                .map(|(_, i)| i.clone())
                .expect("instruction")
        };
        assert!(find(d).is_tangled());
        assert_eq!(find(d).model_need(), Some(ModelNeed::ImplicitDerivatives));
        assert!(find(e).is_tangled());
        assert_eq!(find(e).model_need(), None);
        assert!(!find(a).is_tangled());
    });
}

// ---- Subgroups ----

#[test]
fn group_operations_derive_types_and_check_operands() {
    in_fn(|f, t| {
        let (b, u, x, uv4) = (p(f, t.bool_), p(f, t.u32_), p(f, t.f32_), p(f, t.uvec4));
        let r = f.group_elect().expect("elect");
        assert_eq!(ty(f, r), t.bool_);
        let r = f.group_ballot(b).expect("ballot");
        assert_eq!(ty(f, r), t.uvec4);
        assert_eq!(
            f.group_ballot(u),
            op_err(Op::GroupNonUniformBallot, OperandRule::Category(0))
        );
        let r = f.group_ballot_bit_extract(uv4, u).expect("extract");
        assert_eq!(ty(f, r), t.bool_);
        f.group_broadcast_first(b).expect("broadcast first");
        f.group_vote(true, b).expect("all");
        assert_eq!(
            f.group_vote(false, u),
            op_err(Op::GroupNonUniformAny, OperandRule::Category(0))
        );
        f.group_shuffle(ShuffleOp::Xor, x, u).expect("shuffle xor");
        let r = f
            .group_arith(GroupArith::FAdd, GroupMode::InclusiveScan, x)
            .expect("scan");
        assert_eq!(ty(f, r), t.f32_);
        assert_eq!(
            f.group_arith(GroupArith::FAdd, GroupMode::Reduce, u),
            op_err(Op::GroupNonUniformFAdd, OperandRule::Category(0))
        );
        f.group_arith(GroupArith::UMax, GroupMode::Clustered(4), u)
            .expect("clustered");
        assert_eq!(
            f.group_arith(GroupArith::UMax, GroupMode::Clustered(3), u),
            op_err(Op::GroupNonUniformUMax, OperandRule::Range(2))
        );
        f.group_arith(GroupArith::LogicalXor, GroupMode::Reduce, b)
            .expect("logical");
        f.group_quad_broadcast(x, u).expect("quad broadcast");
        f.group_quad_swap(x, QuadDirection::Diagonal)
            .expect("quad swap");
    });
}

// ---- GLSL.std.450 ----

fn ext_err(op: GlslOp, rule: OperandRule) -> Result<Value, BuildError> {
    Err(BuildError::ExtOperand { op, rule })
}

#[test]
fn glsl_instructions_check_their_contracts() {
    in_fn(|f, t| {
        let (x, d, h, u, u16v, v2, v3) = (
            p(f, t.f32_),
            p(f, t.f64_),
            p(f, t.f16),
            p(f, t.u32_),
            p(f, t.u16_),
            p(f, t.vec2),
            p(f, t.vec3),
        );
        let r = f.glsl(GlslOp::Sin, &[h]).expect("f16 sin");
        assert_eq!(ty(f, r), t.f16);
        assert_eq!(
            f.glsl(GlslOp::Sin, &[d]),
            ext_err(GlslOp::Sin, OperandRule::Width(0))
        );
        f.glsl(GlslOp::Sqrt, &[d]).expect("f64 sqrt");
        assert_eq!(
            f.glsl(GlslOp::Sin, &[u]),
            ext_err(GlslOp::Sin, OperandRule::Category(0))
        );
        assert_eq!(
            f.glsl(GlslOp::NClamp, &[x, x, h]),
            ext_err(GlslOp::NClamp, OperandRule::Mismatch(2))
        );
        assert_eq!(
            f.glsl(GlslOp::NClamp, &[x, x]),
            ext_err(GlslOp::NClamp, OperandRule::Arity)
        );
        let r = f.glsl(GlslOp::Length, &[v3]).expect("length");
        assert_eq!(ty(f, r), t.f32_);
        let r = f.glsl(GlslOp::Cross, &[v3, v3]).expect("cross");
        assert_eq!(ty(f, r), t.vec3);
        assert_eq!(
            f.glsl(GlslOp::Cross, &[v2, v2]),
            ext_err(GlslOp::Cross, OperandRule::Category(0))
        );
        assert_eq!(
            f.glsl(GlslOp::FindUMsb, &[u16v]),
            ext_err(GlslOp::FindUMsb, OperandRule::Width(0))
        );
        f.glsl(GlslOp::UClamp, &[u, u, u]).expect("uclamp");
        let iv2 = p(f, t.ivec2);
        let r = f.glsl(GlslOp::Ldexp, &[v2, iv2]).expect("ldexp");
        assert_eq!(ty(f, r), t.vec2);
        assert_eq!(
            f.glsl(GlslOp::Ldexp, &[v2, u]),
            ext_err(GlslOp::Ldexp, OperandRule::Mismatch(1))
        );
        let r = f.glsl(GlslOp::PackHalf2x16, &[v2]).expect("pack");
        assert_eq!(ty(f, r), t.u32_);
        let r = f.glsl(GlslOp::UnpackUnorm4x8, &[u]).expect("unpack");
        assert_eq!(ty(f, r), t.vec4);
        let r = f.glsl(GlslOp::FrexpStruct, &[v2]).expect("frexp");
        let want = mty(
            f,
            Type::Struct {
                members: [t.vec2, t.ivec2].into(),
                layout: None,
            },
        );
        assert_eq!(ty(f, r), want);
        let r = f.glsl(GlslOp::Refract, &[v3, v3, x]).expect("refract");
        assert_eq!(ty(f, r), t.vec3);
        let m32 = mty(
            f,
            Type::Matrix {
                col: t.vec3,
                cols: 2,
            },
        );
        let m = p(f, m32);
        assert_eq!(
            f.glsl(GlslOp::Determinant, &[m]),
            ext_err(GlslOp::Determinant, OperandRule::Category(0))
        );
        let priv_ptr = mty(
            f,
            Type::Pointer {
                class: StorageClass::Private,
                pointee: t.vec4,
            },
        );
        let pp = p(f, priv_ptr);
        assert_eq!(
            f.glsl(GlslOp::InterpolateAtCentroid, &[pp]),
            ext_err(GlslOp::InterpolateAtCentroid, OperandRule::Category(0))
        );
    });
}

#[test]
fn interpolation_reads_fragment_inputs() {
    let run = |model: ExecModel| {
        let mut b = ModuleBuilder::new();
        let t = tys(&mut b);
        let g = b
            .io(
                crate::pipeline::sym::VaryingKey(crate::pipeline::sym::AirUserKey(
                    b"color".as_slice().into(),
                )),
                IoDir::Input,
                t.vec4,
                Default::default(),
            )
            .expect("varying");
        let gp = b.global_value(g).expect("pointer");
        let mut f = b.function(t.void)?;
        let off = f.undef(t.vec2)?;
        let r = f.glsl(GlslOp::InterpolateAtOffset, &[gp, off])?;
        assert_eq!(f.value_ty(r)?, t.vec4);
        f.ret(None)?;
        let id = f.finish()?;
        b.entry(model, id, vec![])?;
        b.finalize().map(|_| ())
    };
    run(ExecModel::Fragment).expect("fragment interpolation");
    assert_eq!(
        run(ExecModel::Vertex),
        Err(BuildError::Entry(EntryRule::InstructionForModel(
            Op::ExtInst
        )))
    );
}

#[test]
fn undef_takes_data_types() {
    in_fn(|f, t| {
        let r = f.undef(t.vec3).expect("undef");
        assert_eq!(ty(f, r), t.vec3);
        let s = mty(f, Type::Sampler);
        assert_eq!(f.undef(s), op_err(Op::Undef, OperandRule::ResultType));
    });
}

// ---- Constant folding ----

mod folding {
    use super::*;
    use crate::spirv::interp::{run, Buffer, Config, Resources};
    use crate::spirv::ir::TypedModule;

    fn slot(index: u32) -> ResourceSlot {
        ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }
    }

    /// A `u32` storage buffer at `slot(index)`.
    fn buffer(b: &mut ModuleBuilder, index: u32) -> crate::spirv::ids::GlobalId {
        let u32_ = int(b, 32, false);
        let rta = b
            .ty(Type::RuntimeArray {
                elem: u32_,
                stride: Some(4),
            })
            .expect("array");
        let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
        let block = b
            .ty(Type::Struct {
                members: [rta].into(),
                layout: Some(layout),
            })
            .expect("block");
        b.resource(slot(index), StorageClass::StorageBuffer, block)
            .expect("resource")
    }

    fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> Value {
        let c = fb.u32_const(v).expect("constant");
        fb.module().const_value(c).expect("value")
    }

    fn word(fb: &mut FunctionBuilder<'_>, buf: crate::spirv::ids::GlobalId, i: Value) -> Value {
        let base = fb.module().global_value(buf).expect("global");
        let zero = k(fb, 0);
        fb.access_chain(base, &[zero, i]).expect("chain")
    }

    /// The `width`-bit unsigned integer type.
    fn uint(fb: &mut FunctionBuilder<'_>, width: u8) -> TypeId {
        fb.module()
            .ty(Type::Int {
                width,
                signed: false,
            })
            .expect("int")
    }

    /// The `width`-bit constant `v`.
    fn kw(fb: &mut FunctionBuilder<'_>, width: u8, v: u64) -> Value {
        let ty = uint(fb, width);
        let c = fb
            .module()
            .constant(Constant::Scalar {
                ty,
                bits: ConstBits(v),
            })
            .expect("constant");
        fb.module().const_value(c).expect("value")
    }

    /// The `width`-bit value of the `u32` words `lo` and `hi` (`hi` read only at 64 bits).
    fn joined(fb: &mut FunctionBuilder<'_>, width: u8, lo: Value, hi: Value) -> Value {
        if width == 32 {
            return lo;
        }
        let t = uint(fb, width);
        let l = fb.convert(ConvOp::UConvert, lo, t).expect("convert");
        if width < 32 {
            return l;
        }
        let h = fb.convert(ConvOp::UConvert, hi, t).expect("convert");
        let s = kw(fb, 64, 32);
        let h = fb.binary(BinOp::ShiftLeftLogical, h, s).expect("shift");
        fb.binary(BinOp::BitwiseOr, h, l).expect("or")
    }

    /// `op` of the `width`-bit `x` and `y` as two `u32` words, low first (a `bool` result as 1
    /// or 0), and whether the operation itself folded.
    fn result(
        fb: &mut FunctionBuilder<'_>,
        width: u8,
        op: BinOp,
        x: Value,
        y: Value,
    ) -> ([Value; 2], bool) {
        let r = fb.binary(op, x, y).expect("binary");
        let folded = matches!(fb.origin(r), Ok(crate::spirv::ir::ValueOrigin::Const(_)));
        let bool_ = fb.module().ty(Type::Bool).expect("bool");
        let zero = k(fb, 0);
        if fb.value_ty(r).expect("type") == bool_ {
            let one = k(fb, 1);
            return ([fb.select(r, one, zero).expect("select"), zero], folded);
        }
        if width == 32 {
            return ([r, zero], folded);
        }
        let u32_ = uint(fb, 32);
        let lo = fb.convert(ConvOp::UConvert, r, u32_).expect("convert");
        if width < 32 {
            return ([lo, zero], folded);
        }
        let s = kw(fb, 64, 32);
        let hi = fb.binary(BinOp::ShiftRightLogical, r, s).expect("shift");
        let hi = fb.convert(ConvOp::UConvert, hi, u32_).expect("convert");
        ([lo, hi], folded)
    }

    /// The `words` words a kernel of `body` stores over `runs` invocations of input `input`:
    /// invocation `i` stores the words `body` returns at `out[i * len..]`.
    fn outputs(
        runs: u32,
        words: u32,
        input: &[u32],
        body: &dyn Fn(&mut FunctionBuilder<'_>, crate::spirv::ids::GlobalId, Value) -> Vec<Value>,
    ) -> Vec<u32> {
        run_kernel(&kernel(body), runs, words, input)
    }

    /// The compute module of [`outputs`]' kernel of `body`.
    fn kernel(
        body: &dyn Fn(&mut FunctionBuilder<'_>, crate::spirv::ids::GlobalId, Value) -> Vec<Value>,
    ) -> TypedModule {
        let mut b = ModuleBuilder::new();
        let out = buffer(&mut b, 0);
        let inp = buffer(&mut b, 1);
        let void = b.ty(Type::Void).expect("void");
        let u32_ = int(&mut b, 32, false);
        let v3 = vec(&mut b, u32_, 3);
        let mut fb = b.function(void).expect("function");
        let g = fb
            .module()
            .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, v3)
            .expect("builtin");
        let gp = fb.module().global_value(g).expect("global");
        let gid = fb.load(gp, MemAccess::default()).expect("load");
        let i = fb.composite_extract(gid, &[0]).expect("lane");
        let stored = body(&mut fb, inp, i);
        let len = k(&mut fb, stored.len() as u32);
        let base = fb.binary(BinOp::IMul, i, len).expect("index");
        for (j, w) in stored.iter().enumerate() {
            let j = k(&mut fb, j as u32);
            let at = fb.binary(BinOp::IAdd, base, j).expect("index");
            let p = word(&mut fb, out, at);
            fb.store(p, *w, MemAccess::default()).expect("store");
        }
        fb.ret(None).expect("ret");
        let id = fb.finish().expect("finish");
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )
        .expect("entry");
        b.finalize().expect("finalize")
    }

    /// The `words` words module `m` (a [`kernel`]) stores over `runs` invocations of `input`.
    fn run_kernel(m: &TypedModule, runs: u32, words: u32, input: &[u32]) -> Vec<u32> {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes: vec![0; 4 * words as usize],
                },
                Buffer {
                    slot: Some(slot(1)),
                    address: None,
                    bytes: input.iter().flat_map(|w| w.to_le_bytes()).collect(),
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: [runs, 1, 1],
            ..Config::default()
        };
        let o = run(m, &cfg, res).expect("run");
        o.buffers
            .first()
            .expect("output")
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// `OpUMulExtended` and `OpSMulExtended` of 64-bit lanes, which the builder emits from the
    /// operands' 32-bit halves (`mul_extended_from_halves`), give the exact 128-bit product, its
    /// low and high words, of every pair of edge operands (0, 1, 2, -1, `i64::MIN`, `i64::MAX`,
    /// `u64::MAX`, 32-bit boundaries and mixed signs), in a scalar and in each lane of a
    /// two-lane vector (whose second lane squares the first operand); and no module holds a 64-bit
    /// `MulExtended`, only the 32-bit ones of the halves.
    #[test]
    fn sixty_four_bit_extended_products_are_exact_from_halves() {
        let vals: [u64; 11] = [
            0,
            1,
            2,
            u64::MAX,
            i64::MIN as u64,
            i64::MAX as u64,
            0xffff_ffff,
            0x1_0000_0000,
            0x1_0000_0001,
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
        ];
        let pairs: Vec<(u64, u64)> = vals
            .iter()
            .flat_map(|x| vals.iter().map(move |y| (*x, *y)))
            .collect();
        let n = pairs.len() as u32;
        let input: Vec<u32> = pairs
            .iter()
            .flat_map(|(x, y)| [*x as u32, (*x >> 32) as u32, *y as u32, (*y >> 32) as u32])
            .collect();
        // The 128-bit product's low and high words, as `u32`s, low first.
        let exact = |signed: bool, x: u64, y: u64| -> [u32; 4] {
            let p = match signed {
                true => (i128::from(x as i64) * i128::from(y as i64)) as u128,
                false => u128::from(x) * u128::from(y),
            };
            [
                p as u32,
                (p >> 32) as u32,
                (p >> 64) as u32,
                (p >> 96) as u32,
            ]
        };
        for (op, signed) in [(BinOp::UMulExtended, false), (BinOp::SMulExtended, true)] {
            for lanes in [1u8, 2] {
                let m = kernel(&|fb, inp, i| {
                    let four = k(fb, 4);
                    let at = fb.binary(BinOp::IMul, i, four).expect("index");
                    let operand = |fb: &mut FunctionBuilder<'_>, d: u32| {
                        let d = k(fb, d);
                        let lo = fb.binary(BinOp::IAdd, at, d).expect("index");
                        let one = k(fb, 1);
                        let hi = fb.binary(BinOp::IAdd, lo, one).expect("index");
                        let (plo, phi) = (word(fb, inp, lo), word(fb, inp, hi));
                        let lo = fb.load(plo, MemAccess::default()).expect("lo");
                        let hi = fb.load(phi, MemAccess::default()).expect("hi");
                        joined(fb, 64, lo, hi)
                    };
                    let (x, y) = (operand(fb, 0), operand(fb, 2));
                    let elem = fb
                        .module()
                        .ty(Type::Int { width: 64, signed })
                        .expect("int");
                    let u64_ = uint(fb, 64);
                    let cast = |fb: &mut FunctionBuilder<'_>, v: Value, to: TypeId| {
                        fb.convert(ConvOp::Bitcast, v, to).expect("bitcast")
                    };
                    let (x, y) = match signed {
                        true => (cast(fb, x, elem), cast(fb, y, elem)),
                        false => (x, y),
                    };
                    let (a, b) = match lanes {
                        1 => (x, y),
                        _ => {
                            let v = fb
                                .module()
                                .ty(Type::Vector { elem, len: lanes })
                                .expect("vector");
                            let a = fb.composite_construct(v, &[x, x]).expect("a");
                            let b = fb.composite_construct(v, &[y, x]).expect("b");
                            (a, b)
                        }
                    };
                    let r = fb.binary(op, a, b).expect("product");
                    let mut words = Vec::new();
                    for lane in 0..u32::from(lanes) {
                        for half in 0..2 {
                            let w = fb.composite_extract(r, &[half]).expect("half");
                            let w = match lanes {
                                1 => w,
                                _ => fb.composite_extract(w, &[lane]).expect("lane"),
                            };
                            let w = match signed {
                                true => cast(fb, w, u64_),
                                false => w,
                            };
                            let u32_ = uint(fb, 32);
                            let lo = fb.convert(ConvOp::UConvert, w, u32_).expect("lo");
                            let s = kw(fb, 64, 32);
                            let hi = fb.binary(BinOp::ShiftRightLogical, w, s).expect("shift");
                            let hi = fb.convert(ConvOp::UConvert, hi, u32_).expect("hi");
                            words.extend([lo, hi]);
                        }
                    }
                    words
                });
                // Only 32-bit `MulExtended`s: the halves' products.
                let mut halves = 0;
                for block in &m.blocks {
                    for node in &block.insts {
                        let Inst::Binary { op: o, a, .. } = &node.inst else {
                            continue;
                        };
                        if !matches!(o, BinOp::UMulExtended | BinOp::SMulExtended) {
                            continue;
                        }
                        let ty = m.values[a.index()].ty;
                        let elem = match m.types.get(ty).expect("type") {
                            Type::Vector { elem, .. } => *elem,
                            _ => ty,
                        };
                        let width = match m.types.get(elem).expect("type") {
                            Type::Int { width, .. } => *width,
                            other => panic!("a MulExtended of {other:?}"),
                        };
                        assert_eq!(
                            width, 32,
                            "{op:?} over {lanes} lanes left a {width}-bit one"
                        );
                        halves += 1;
                    }
                }
                assert_eq!(
                    halves, 4,
                    "{op:?} over {lanes} lanes: the halves' four products"
                );
                let words = 4 * u32::from(lanes);
                let got = run_kernel(&m, n, words * n, &input);
                let expected: Vec<u32> = pairs
                    .iter()
                    .flat_map(|&(x, y)| {
                        let mut w = exact(signed, x, y).to_vec();
                        if lanes == 2 {
                            w.extend(exact(signed, x, x));
                        }
                        w
                    })
                    .collect();
                assert_eq!(got, expected, "{op:?} over {lanes} lanes, pairs {pairs:x?}");
            }
        }
    }

    /// Every integer operation the builder folds gives, on constants, the words the device (the
    /// interpreter) computes from the same operands loaded: wrapping arithmetic, bitwise
    /// operations, shifts by less than the width (logical and arithmetic), and unsigned and signed
    /// comparisons, over values at every sign and wrap boundary of 8-, 16-, 32-, and 64-bit
    /// integers; `bool` operations, a select on a constant condition, and a shift by the width
    /// (left to the device) too.
    #[test]
    fn constant_operations_fold_as_the_device_computes() {
        use BinOp as B;
        let ops = [
            B::IAdd,
            B::ISub,
            B::IMul,
            B::BitwiseAnd,
            B::BitwiseOr,
            B::BitwiseXor,
            B::ShiftLeftLogical,
            B::ShiftRightLogical,
            B::ShiftRightArithmetic,
            B::IEqual,
            B::INotEqual,
            B::UGreaterThan,
            B::UGreaterThanEqual,
            B::ULessThan,
            B::ULessThanEqual,
            B::SGreaterThan,
            B::SGreaterThanEqual,
            B::SLessThan,
            B::SLessThanEqual,
        ];
        for width in [8u8, 16, 32, 64] {
            let w = u64::from(width);
            let ones = u64::MAX >> (64 - w);
            let sign = 1u64 << (w - 1);
            let vals = [
                0,
                1,
                2,
                7,
                w - 1,
                w,
                w + 1,
                sign - 1,
                sign,
                sign + 1,
                ones - 1,
                ones,
            ];
            for op in ops {
                let shift = matches!(
                    op,
                    B::ShiftLeftLogical | B::ShiftRightLogical | B::ShiftRightArithmetic
                );
                let pairs: Vec<(u64, u64)> = vals
                    .iter()
                    .flat_map(|x| vals.iter().map(move |y| (*x, *y)))
                    .filter(|(_, y)| !shift || *y < w)
                    .collect();
                let n = pairs.len() as u32;
                let input: Vec<u32> = pairs
                    .iter()
                    .flat_map(|(x, y)| [*x as u32, (*x >> 32) as u32, *y as u32, (*y >> 32) as u32])
                    .collect();
                let device = outputs(n, 2 * n, &input, &|fb, inp, i| {
                    let four = k(fb, 4);
                    let at = fb.binary(BinOp::IMul, i, four).expect("index");
                    let operand = |fb: &mut FunctionBuilder<'_>, d: u32| {
                        let d = k(fb, d);
                        let lo = fb.binary(BinOp::IAdd, at, d).expect("index");
                        let one = k(fb, 1);
                        let hi = fb.binary(BinOp::IAdd, lo, one).expect("index");
                        let (plo, phi) = (word(fb, inp, lo), word(fb, inp, hi));
                        let lo = fb.load(plo, MemAccess::default()).expect("lo");
                        let hi = fb.load(phi, MemAccess::default()).expect("hi");
                        joined(fb, width, lo, hi)
                    };
                    let x = operand(fb, 0);
                    let y = operand(fb, 2);
                    result(fb, width, op, x, y).0.to_vec()
                });
                let folded = outputs(1, 2 * n, &[], &|fb, _, _| {
                    pairs
                        .iter()
                        .flat_map(|(x, y)| {
                            let (x, y) = (kw(fb, width, *x), kw(fb, width, *y));
                            let (words, folded) = result(fb, width, op, x, y);
                            assert!(folded, "{op:?} at {width} bits did not fold");
                            words
                        })
                        .collect()
                });
                assert!(
                    device.iter().any(|w| *w != 0),
                    "{op:?} at {width} bits: nothing stored"
                );
                assert_eq!(folded, device, "{op:?} at {width} bits over {pairs:x?}");
            }
        }
        // `bool` operations, and a select on a constant condition.
        let mut b = ModuleBuilder::new();
        let void = b.ty(Type::Void).expect("void");
        let bool_ = b.ty(Type::Bool).expect("bool");
        let mut fb = b.function(void).expect("function");
        let t = fb
            .module()
            .constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(1),
            })
            .expect("true");
        let f = fb
            .module()
            .constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(0),
            })
            .expect("false");
        let (t, f) = (
            fb.module().const_value(t).expect("t"),
            fb.module().const_value(f).expect("f"),
        );
        for (op, want) in [
            (B::LogicalAnd, [false, false, false, true]),
            (B::LogicalOr, [false, true, true, true]),
            (B::LogicalEqual, [true, false, false, true]),
            (B::LogicalNotEqual, [false, true, true, false]),
        ] {
            for (i, (x, y)) in [(f, f), (f, t), (t, f), (t, t)].into_iter().enumerate() {
                let r = fb.binary(op, x, y).expect("binary");
                assert_eq!(
                    fb.const_bool(r).expect("const"),
                    Some(want[i]),
                    "{op:?} {i}"
                );
            }
        }
        let (one, two) = (k(&mut fb, 1), k(&mut fb, 2));
        assert_eq!(fb.select(t, one, two).expect("select"), one);
        assert_eq!(fb.select(f, one, two).expect("select"), two);
        // A shift by the width or more is undefined in SPIR-V; it stays for the device.
        let width = k(&mut fb, 32);
        for op in [
            B::ShiftLeftLogical,
            B::ShiftRightLogical,
            B::ShiftRightArithmetic,
        ] {
            let r = fb.binary(op, one, width).expect("shift");
            assert!(
                matches!(fb.origin(r), Ok(crate::spirv::ir::ValueOrigin::Inst(_))),
                "{op:?} by the width folded"
            );
        }
    }
}
