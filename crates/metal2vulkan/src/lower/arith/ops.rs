//! The emission core of the LLVM instruction set on held values: integer, boolean, and float
//! operators, comparisons, and casts. Every function takes and returns SPIR-V values in their
//! lowered representation ([`Form`]) and keeps the zero-extension invariant, so the functions are
//! testable on the CPU interpreter without an AIR module (`tests`).

use crate::air::ir::{BinOp as AirBin, FloatPredicate, IntPredicate};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::spirv::builder::{FunctionBuilder, MaskForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, ConstBits, Constant, Contraction, ConvOp, GlslOp, Type, UnOp};

/// The all-ones value of `bits` bits.
pub fn ones(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// The SPIR-V element type of `h` (a wide integer's: its words').
pub fn elem_ty(fb: &mut FunctionBuilder<'_>, h: Held) -> BResult<TypeId> {
    fb.module().ty(match h {
        Held::Bool => Type::Bool,
        Held::Int { container, .. } => Type::Int {
            width: container,
            signed: false,
        },
        Held::Wide { .. } => Type::Int {
            width: 32,
            signed: false,
        },
        Held::Float(f) => Type::Float {
            width: f.container(),
        },
    })
}

/// The SPIR-V type of `f` (a wide integer's: [`super::legal::wide_demote`]).
pub fn ty(fb: &mut FunctionBuilder<'_>, f: Form) -> BResult<TypeId> {
    if f.is_wide() {
        let words = f.with(Held::Int {
            bits: 32,
            container: 32,
        });
        let w = ty(fb, words)?;
        return if f.lanes == 1 {
            fb.module().ty(Type::Vector { elem: w, len: 2 })
        } else {
            fb.module().ty(Type::Struct {
                members: vec![w, w].into(),
                layout: None,
            })
        };
    }
    let e = elem_ty(fb, f.held)?;
    if f.lanes == 1 {
        Ok(e)
    } else {
        fb.module().ty(Type::Vector {
            elem: e,
            len: f.lanes,
        })
    }
}

/// The constant of form `f` with `bits` (in the container) in every lane.
pub fn splat(fb: &mut FunctionBuilder<'_>, f: Form, bits: u64) -> BResult<Value> {
    if f.is_wide() {
        return super::wide::splat(fb, f, bits);
    }
    let e = elem_ty(fb, f.held)?;
    let bits = match f.held {
        Held::Bool => bits & 1,
        _ => bits & ones(f.container()),
    };
    let s = fb.module().constant(Constant::Scalar {
        ty: e,
        bits: ConstBits(bits),
    })?;
    let c = if f.lanes == 1 {
        s
    } else {
        let t = ty(fb, f)?;
        fb.module().constant(Constant::Composite {
            ty: t,
            parts: vec![s; usize::from(f.lanes)].into(),
        })?
    };
    fb.module().const_value(c)
}

/// The `u32` lanes of 0 and 1 an aggregate holds a member `bool` vector `v` of `lanes` lanes in
/// (`legal::as_member`).
pub fn bools_to_member(fb: &mut FunctionBuilder<'_>, v: Value, lanes: u8) -> BResult<Value> {
    let words = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    };
    let one = splat(fb, words, 1)?;
    let zero = splat(fb, words, 0)?;
    fb.select(v, one, zero)
}

/// The `bool` vector of `lanes` lanes an aggregate's member `x`, its `u32` lanes of 0 and 1,
/// holds (`legal::as_member`).
pub fn member_to_bools(fb: &mut FunctionBuilder<'_>, x: Value, lanes: u8) -> BResult<Value> {
    let words = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    };
    let zero = splat(fb, words, 0)?;
    fb.binary(BinOp::INotEqual, x, zero)
}

/// A float constant of form `f`.
pub fn fsplat(fb: &mut FunctionBuilder<'_>, f: Form, x: f64) -> BResult<Value> {
    splat(fb, f, f.const_bits(x))
}

/// `x` with the bits above its AIR width cleared (a no-op for a full-width integer): `x & (2^N -
/// 1)`, or in a 32-bit container under [`MaskForm::Extract`] `OpBitFieldUExtract(x, 0, N)`, the
/// same bits, for a driver whose compiler drops the mask after a signed clamp.
pub fn mask(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    if !f.legalized() {
        return Ok(x);
    }
    if f.container() == 32 && fb.module().mask_form() == MaskForm::Extract {
        let u32_ = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        };
        let zero = splat(fb, u32_, 0)?;
        let width = splat(fb, u32_, u64::from(f.bits()))?;
        return fb.bit_field_extract(false, x, zero, width);
    }
    let m = splat(fb, f, ones(f.bits()))?;
    fb.binary(BinOp::BitwiseAnd, x, m)
}

/// `x` (zero-extended) sign-extended from its AIR width to the whole container.
pub fn sext(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    if !f.legalized() {
        return Ok(x);
    }
    let sh = splat(fb, f, u64::from(f.container() - f.bits()))?;
    let up = fb.binary(BinOp::ShiftLeftLogical, x, sh)?;
    fb.binary(BinOp::ShiftRightArithmetic, up, sh)
}

/// An integer binary operator on two held integers of form `f`.
pub fn int_binary(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    a: Value,
    b: Value,
) -> BResult<Value> {
    let bin = |fb: &mut FunctionBuilder<'_>, o, x, y| fb.binary(o, x, y);
    match op {
        AirBin::Add => {
            let r = bin(fb, BinOp::IAdd, a, b)?;
            mask(fb, f, r)
        }
        AirBin::Sub => {
            let r = bin(fb, BinOp::ISub, a, b)?;
            mask(fb, f, r)
        }
        AirBin::Mul => {
            let r = bin(fb, BinOp::IMul, a, b)?;
            mask(fb, f, r)
        }
        AirBin::Shl => {
            let r = bin(fb, BinOp::ShiftLeftLogical, a, b)?;
            mask(fb, f, r)
        }
        AirBin::UDiv => bin(fb, BinOp::UDiv, a, b),
        AirBin::URem => bin(fb, BinOp::UMod, a, b),
        AirBin::LShr => bin(fb, BinOp::ShiftRightLogical, a, b),
        AirBin::And => bin(fb, BinOp::BitwiseAnd, a, b),
        AirBin::Or => bin(fb, BinOp::BitwiseOr, a, b),
        AirBin::Xor => bin(fb, BinOp::BitwiseXor, a, b),
        AirBin::SDiv | AirBin::SRem => {
            let (x, y) = (sext(fb, f, a)?, sext(fb, f, b)?);
            let o = if op == AirBin::SDiv {
                BinOp::SDiv
            } else {
                BinOp::SRem
            };
            let r = bin(fb, o, x, y)?;
            mask(fb, f, r)
        }
        AirBin::AShr => {
            let x = sext(fb, f, a)?;
            let r = bin(fb, BinOp::ShiftRightArithmetic, x, b)?;
            mask(fb, f, r)
        }
        AirBin::FAdd | AirBin::FSub | AirBin::FMul | AirBin::FDiv | AirBin::FRem => {
            float_binary(fb, op, f, Contraction::Forbidden, a, b)
        }
    }
}

/// A binary operator on `i1` values (scalars or vectors of `bool`). Division and shifts have one
/// non-poison divisor or amount (1, resp. 0), so they return `a`; remainders are `false`.
pub fn bool_binary(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    a: Value,
    b: Value,
) -> BResult<Value> {
    match op {
        AirBin::And | AirBin::Mul => fb.binary(BinOp::LogicalAnd, a, b),
        AirBin::Or => fb.binary(BinOp::LogicalOr, a, b),
        AirBin::Xor | AirBin::Add | AirBin::Sub => fb.binary(BinOp::LogicalNotEqual, a, b),
        AirBin::UDiv | AirBin::SDiv | AirBin::Shl | AirBin::LShr | AirBin::AShr => Ok(a),
        AirBin::URem | AirBin::SRem => splat(fb, f, 0),
        AirBin::FAdd | AirBin::FSub | AirBin::FMul | AirBin::FDiv | AirBin::FRem => {
            float_binary(fb, op, f, Contraction::Forbidden, a, b)
        }
    }
}

/// A float binary operator on floats of form `f` (LLVM `frem` is C `fmod`, the sign of the
/// dividend: `OpFRem`). A narrow form rounds the `f32` result to its format by the device's
/// rounding ([`narrow::round`]).
pub fn float_binary(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    contract: Contraction,
    a: Value,
    b: Value,
) -> BResult<Value> {
    let o = match op {
        AirBin::FAdd => BinOp::FAdd,
        AirBin::FSub => BinOp::FSub,
        AirBin::FMul => BinOp::FMul,
        AirBin::FDiv => BinOp::FDiv,
        _ => BinOp::FRem,
    };
    let r = fb.binary_with(o, a, b, contract)?;
    narrow::round(fb, f, r)
}

/// The census name of a `half` operator on the device's 16-bit arithmetic
/// ([`super::ArithStrategy::F16`]).
pub const F16: &str = "arith.float.f16";

/// An upper estimate of the SPIR-V words of one [`f16_binary`] on form `f`: the two narrowing
/// conversions, the instruction (a product: the `Fma` and its `-0.0`), its decoration, and the
/// widening, with the 16-bit types' and constant's share.
pub fn f16_binary_words(f: Form) -> u32 {
    32 + 4 * u32::from(f.lanes)
}

/// The `half` `fadd`, `fsub` or `fmul` of `a` and `b` (`half` values held in `float` lanes of
/// `lanes` lanes) by the device's 16-bit arithmetic: each operand narrowed to 16 bits, exactly
/// since it is a `half`, the instruction, which rounds the exact result once as Metal's `half`
/// arithmetic does, and its result widened, exactly. No rounding of the module's own
/// ([`narrow::round`]), so no conversion barrier and no conversion pair, which a device's compiler
/// spends most on (ANV's register allocation grows with the values the pairs add).
///
/// The instruction is Metal's where the profile takes `DriverDeviation::MetalHalfArithmetic`, and,
/// by the Vulkan spec, in an entry declaring the 16-bit float controls the module's roundings take
/// ([`narrow::controlled_modes`]): the precision table "for IEEE754 half- (16 bit) and single- (32
/// bit) precision instructions" lists `OpFAdd`, `OpFSub` and `OpFMul` as "Correctly rounded",
/// "according to IEEE 754 roundTiesToEven" under `RoundingModeRTE`, `DenormPreserve` keeps
/// denormal operands and results, and `SignedZeroInfNanPreserve` the zero signs IEEE 754 gives. That
/// is the value the `float` operation rounded once to `half` gives too ([`f16_binary`]'s
/// alternative, [`float_binary`]): `float`'s 24-bit significand is at least `2 · 11 + 2` bits, so
/// rounding its correctly rounded sum, difference or product to `half` rounds the exact result once
/// (the double rounding is innocuous). A NaN's bits are the device's, as the controlled rounding's
/// are.
///
/// The instruction is always kept apart, whatever the AIR's flags allow: a sum or difference
/// decorated `NoContraction` (SPIRV-Cross writes its `[[clang::optnone]]` `spvFAdd` or `spvFSub`),
/// a product as `Fma(a, b, -0.0)` decorated `NoContraction`, the product's value whether the
/// device fuses it or not. Undecorated, Metal's fast and relaxed compiles fold the narrowings and
/// the widening away and answer the `float` sum, difference or product of the widened operands; a
/// `NoContraction` `OpFMul`, which SPIRV-Cross writes as `fma(l, r, 0)`, answers `+0` for a `-0`
/// product. Keeping each operation apart is one of the values the flags allow, the one the widened
/// lowering gives.
pub fn f16_binary(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    lanes: u8,
    a: Value,
    b: Value,
) -> BResult<Value> {
    let float_ty = |fb: &mut FunctionBuilder<'_>, width: u8| -> BResult<TypeId> {
        let e = fb.module().ty(Type::Float { width })?;
        match lanes {
            1 => Ok(e),
            len => fb.module().ty(Type::Vector { elem: e, len }),
        }
    };
    let (h, w) = (float_ty(fb, 16)?, float_ty(fb, 32)?);
    let x = fb.convert(ConvOp::FConvert, a, h)?;
    let y = fb.convert(ConvOp::FConvert, b, h)?;
    let r = match op {
        AirBin::FAdd => fb.binary_no_contraction(BinOp::FAdd, x, y)?,
        AirBin::FSub => fb.binary_no_contraction(BinOp::FSub, x, y)?,
        AirBin::FMul => {
            let e = fb.module().ty(Type::Float { width: 16 })?;
            let s = fb.module().constant(Constant::Scalar {
                ty: e,
                bits: ConstBits(0x8000),
            })?;
            let z = match lanes {
                1 => s,
                n => fb.module().constant(Constant::Composite {
                    ty: h,
                    parts: vec![s; usize::from(n)].into(),
                })?,
            };
            let z = fb.module().const_value(z)?;
            fb.glsl_with(GlslOp::Fma, &[x, y, z], Contraction::Forbidden)?
        }
        _ => return Err(BuildError::Type(TypeRule::ScalarWidth)),
    };
    fb.convert(ConvOp::FConvert, r, w)
}

/// `icmp` on held integers of form `f`.
pub fn int_compare(
    fb: &mut FunctionBuilder<'_>,
    pred: IntPredicate,
    f: Form,
    a: Value,
    b: Value,
) -> BResult<Value> {
    use IntPredicate as P;
    let signed = matches!(pred, P::Sgt | P::Sge | P::Slt | P::Sle);
    let (a, b) = if signed {
        (sext(fb, f, a)?, sext(fb, f, b)?)
    } else {
        (a, b)
    };
    let o = match pred {
        P::Eq => BinOp::IEqual,
        P::Ne => BinOp::INotEqual,
        P::Ugt => BinOp::UGreaterThan,
        P::Uge => BinOp::UGreaterThanEqual,
        P::Ult => BinOp::ULessThan,
        P::Ule => BinOp::ULessThanEqual,
        P::Sgt => BinOp::SGreaterThan,
        P::Sge => BinOp::SGreaterThanEqual,
        P::Slt => BinOp::SLessThan,
        P::Sle => BinOp::SLessThanEqual,
    };
    fb.binary(o, a, b)
}

/// `icmp` on `i1` values: unsigned `true` is 1, signed `true` is -1.
pub fn bool_compare(
    fb: &mut FunctionBuilder<'_>,
    pred: IntPredicate,
    a: Value,
    b: Value,
) -> BResult<Value> {
    use IntPredicate as P;
    // x > y is x && !y; x >= y is x || !y (unsigned); signed swaps the roles.
    let gt = |fb: &mut FunctionBuilder<'_>, x, y| -> BResult<Value> {
        let ny = fb.unary(UnOp::LogicalNot, y)?;
        fb.binary(BinOp::LogicalAnd, x, ny)
    };
    let ge = |fb: &mut FunctionBuilder<'_>, x, y| -> BResult<Value> {
        let ny = fb.unary(UnOp::LogicalNot, y)?;
        fb.binary(BinOp::LogicalOr, x, ny)
    };
    match pred {
        P::Eq => fb.binary(BinOp::LogicalEqual, a, b),
        P::Ne => fb.binary(BinOp::LogicalNotEqual, a, b),
        P::Ugt | P::Slt => gt(fb, a, b),
        P::Uge | P::Sle => ge(fb, a, b),
        P::Ult | P::Sgt => gt(fb, b, a),
        P::Ule | P::Sge => ge(fb, b, a),
    }
}

/// `fcmp` on floats of form `f` (the result has `f`'s lane count).
///
/// `one` (ordered and not equal) is emitted as the negation of `OpFUnordEqual`, never as
/// `OpFOrdNotEqual`: the two are the same predicate on every input, but SPIRV-Cross, which
/// MoltenVK uses, renders `OpFOrdNotEqual` as MSL's `!=`, which is true for a NaN operand,
/// where `one` and Metal's answer are false. It renders `OpFUnordEqual` as `isunordered(a, b) ||
/// a == b`, which keeps the NaN case.
pub fn float_compare(
    fb: &mut FunctionBuilder<'_>,
    pred: FloatPredicate,
    f: Form,
    a: Value,
    b: Value,
) -> BResult<Value> {
    use FloatPredicate as P;
    let o = match pred {
        P::False => return splat(fb, f.with(Held::Bool), 0),
        P::True => return splat(fb, f.with(Held::Bool), 1),
        P::Ord | P::Uno => {
            let na = fb.unary(UnOp::IsNan, a)?;
            let nb = fb.unary(UnOp::IsNan, b)?;
            let any = fb.binary(BinOp::LogicalOr, na, nb)?;
            return if pred == P::Uno {
                Ok(any)
            } else {
                fb.unary(UnOp::LogicalNot, any)
            };
        }
        P::Oeq => BinOp::FOrdEqual,
        P::Ogt => BinOp::FOrdGreaterThan,
        P::Oge => BinOp::FOrdGreaterThanEqual,
        P::Olt => BinOp::FOrdLessThan,
        P::Ole => BinOp::FOrdLessThanEqual,
        P::One => {
            let ueq = fb.binary(BinOp::FUnordEqual, a, b)?;
            return fb.unary(UnOp::LogicalNot, ueq);
        }
        P::Ueq => BinOp::FUnordEqual,
        P::Ugt => BinOp::FUnordGreaterThan,
        P::Uge => BinOp::FUnordGreaterThanEqual,
        P::Ult => BinOp::FUnordLessThan,
        P::Ule => BinOp::FUnordLessThanEqual,
        P::Une => BinOp::FUnordNotEqual,
    };
    fb.binary(o, a, b)
}

/// A numeric cast (LLVM `trunc` .. `sitofp`) of `x` from form `from` to form `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastKind {
    Trunc,
    ZExt,
    SExt,
    FpResize,
    FpToUi,
    FpToSi,
    UiToFp,
    SiToFp,
}

/// Change the container of held integer `x` (zero-extended, or sign-extended when `signed`)
/// from `from` to `to`'s, without masking.
fn recontain(
    fb: &mut FunctionBuilder<'_>,
    from: Form,
    to: Form,
    x: Value,
    signed: bool,
) -> BResult<Value> {
    if from.container() == to.container() {
        return Ok(x);
    }
    let t = ty(fb, to)?;
    let op = if signed && to.container() > from.container() {
        ConvOp::SConvert
    } else {
        ConvOp::UConvert
    };
    fb.convert(op, x, t)
}

/// `select(c, t_bits, f_bits)` in form `to`.
fn pick(fb: &mut FunctionBuilder<'_>, to: Form, c: Value, t: u64, f: u64) -> BResult<Value> {
    let (tv, fv) = (splat(fb, to, t)?, splat(fb, to, f)?);
    fb.select(c, tv, fv)
}

/// Element index `index` of form `form` as `OpVectorExtractDynamic` and `OpVectorInsertDynamic`
/// take it, an integer scalar. LLVM's index may be any integer type and is read as unsigned, so
/// an `i1` index, which is lowered as a `bool`, is 1 when set and 0 otherwise; every other index
/// is already an integer.
pub fn element_index(fb: &mut FunctionBuilder<'_>, index: Value, form: Form) -> BResult<Value> {
    if !form.is_bool() {
        return Ok(index);
    }
    let word = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    pick(fb, word, index, 1, 0)
}

/// Emit cast `k` of `x`.
pub fn cast(
    fb: &mut FunctionBuilder<'_>,
    k: CastKind,
    from: Form,
    to: Form,
    x: Value,
) -> BResult<Value> {
    let int_ty = |fb: &mut FunctionBuilder<'_>, f: Form| -> BResult<TypeId> {
        // `i1` results convert through a 32-bit integer.
        let f = if f.is_bool() {
            f.with(Held::Int {
                bits: 32,
                container: 32,
            })
        } else {
            f
        };
        ty(fb, f)
    };
    let nonzero = |fb: &mut FunctionBuilder<'_>, f: Form, v: Value| -> BResult<Value> {
        let z = splat(fb, f, 0)?;
        fb.binary(BinOp::INotEqual, v, z)
    };
    let u32f = to.with(Held::Int {
        bits: 32,
        container: 32,
    });
    match k {
        CastKind::Trunc if to.is_bool() => {
            let one = splat(fb, from, 1)?;
            let low = fb.binary(BinOp::BitwiseAnd, x, one)?;
            nonzero(fb, from, low)
        }
        CastKind::Trunc => {
            let r = recontain(fb, from, to, x, false)?;
            mask(fb, to, r)
        }
        CastKind::ZExt if from.is_bool() => pick(fb, to, x, 1, 0),
        CastKind::ZExt => recontain(fb, from, to, x, false),
        CastKind::SExt if from.is_bool() => pick(fb, to, x, ones(to.bits()), 0),
        CastKind::SExt => {
            let s = sext(fb, from, x)?;
            let r = recontain(fb, from, to, s, true)?;
            mask(fb, to, r)
        }
        CastKind::FpResize => narrow::resize(fb, from, to, x),
        CastKind::FpToUi | CastKind::FpToSi => {
            let t = int_ty(fb, to)?;
            let op = if k == CastKind::FpToUi {
                ConvOp::ConvertFToU
            } else {
                ConvOp::ConvertFToS
            };
            let r = fb.convert(op, x, t)?;
            if to.is_bool() {
                nonzero(fb, u32f, r)
            } else {
                mask(fb, to, r)
            }
        }
        CastKind::UiToFp if from.is_bool() => pick(fb, to, x, to.const_bits(1.0), 0),
        CastKind::SiToFp if from.is_bool() => pick(fb, to, x, to.const_bits(-1.0), 0),
        CastKind::UiToFp | CastKind::SiToFp => int_to_float(fb, k == CastKind::SiToFp, from, to, x),
    }
}

/// Integer (`signed` source) `x` of form `from` to float form `to`. A `half` result converts
/// through `f32` and rounds once, by integer arithmetic on the `f32` bits
/// ([`narrow::round_integer`]): an integer the `f32` step rounds is at least `2^24`, which is
/// infinite as a half either way. A `bfloat` result goes through a round-to-odd `f32`.
pub fn int_to_float(
    fb: &mut FunctionBuilder<'_>,
    signed: bool,
    from: Form,
    to: Form,
    x: Value,
) -> BResult<Value> {
    let t = ty(fb, to)?;
    let x = if signed { sext(fb, from, x)? } else { x };
    if to.float_fmt() == Some(FloatFmt::BFloat) {
        let u = narrow::u32_form(from);
        let m = if signed {
            fb.glsl(crate::spirv::ir::GlslOp::SAbs, &[x])?
        } else {
            x
        };
        let r = if from.container() == 64 {
            let w = from.with(Held::Int {
                bits: 64,
                container: 64,
            });
            narrow::int64_to_f32_odd(fb, w, to, m)?
        } else {
            narrow::int_to_f32_odd(fb, u, to, m)?
        };
        let r = if signed {
            let z = splat(fb, from, 0)?;
            let neg = fb.binary(BinOp::SLessThan, x, z)?;
            let n = fb.unary(UnOp::FNegate, r)?;
            fb.select(neg, n, r)?
        } else {
            r
        };
        return narrow::round(fb, to, r);
    }
    let op = if signed {
        ConvOp::ConvertSToF
    } else {
        ConvOp::ConvertUToF
    };
    let r = fb.convert(op, x, t)?;
    narrow::round_integer(fb, to, r)
}

/// How a `bitcast` between two non-pointer forms is emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitcastPlan {
    /// Equal representations: the value itself.
    Same,
    /// Both sides hold their AIR bits exactly (no padding): one `OpBitcast`.
    Direct,
    /// Through one unsigned integer of `total` bits (at most 64): lanes are packed low first
    /// with shifts and unpacked the same way.
    Packed { total: u8 },
    /// Through `words` 32-bit words (lanes of 8, 16, 32, or 64 bits on both sides, so no lane
    /// straddles a word): lanes are packed low first into each word and unpacked the same way.
    /// Needs no 64-bit integer (a 64-bit lane moves as a pair of words).
    Words { words: u8 },
    /// Lane by lane, between two forms of the same lane count and lane width whose total is not
    /// a whole number of words (`<3 x half>` ↔ `<3 x i16>`): each lane moves through its AIR
    /// bits in its container. Needs no integer wider than a lane.
    Lanes,
}

/// Whether lanes of `f` never straddle a 32-bit word.
pub(super) fn word_lanes(f: Form) -> bool {
    matches!(f.bits(), 8 | 16 | 32 | 64)
}

/// The bitcast plan from `from` to `to`, or `None` when no plan applies (booleans, a total
/// above 64 bits with lanes that straddle words).
pub fn bitcast_plan(from: Form, to: Form) -> Option<BitcastPlan> {
    if from == to {
        return Some(BitcastPlan::Same);
    }
    if from.is_bool() || to.is_bool() {
        return None;
    }
    let total = u32::from(from.bits()) * u32::from(from.lanes);
    if total != u32::from(to.bits()) * u32::from(to.lanes) {
        return None;
    }
    if !from.padded() && !to.padded() {
        return Some(BitcastPlan::Direct);
    }
    if total > 32 && total % 32 == 0 && word_lanes(from) && word_lanes(to) {
        return u8::try_from(total / 32)
            .ok()
            .map(|words| BitcastPlan::Words { words });
    }
    // A total above one word that is no whole number of words would pack into a 64-bit
    // integer; lanes of one width move one by one instead.
    if total > 32 && from.lanes == to.lanes && from.bits() == to.bits() {
        return Some(BitcastPlan::Lanes);
    }
    u8::try_from(total)
        .ok()
        .filter(|t| *t <= 64)
        .map(|total| BitcastPlan::Packed { total })
}

/// The unsigned integer form holding `total` bits.
pub fn packed_form(total: u8) -> Form {
    Form {
        held: Held::Int {
            bits: total,
            container: if total <= 32 { 32 } else { 64 },
        },
        lanes: 1,
    }
}

/// Emit a bitcast of `x` from `from` to `to` under `plan`.
pub fn bitcast(
    fb: &mut FunctionBuilder<'_>,
    plan: BitcastPlan,
    from: Form,
    to: Form,
    x: Value,
) -> BResult<Value> {
    match plan {
        BitcastPlan::Same => Ok(x),
        BitcastPlan::Direct => {
            let t = ty(fb, to)?;
            fb.convert(ConvOp::Bitcast, x, t)
        }
        BitcastPlan::Packed { total } => {
            let p = packed_form(total);
            let w = pack(fb, from, p, x)?;
            unpack(fb, p, to, w)
        }
        BitcastPlan::Words { words } => {
            let ws = pack_words(fb, from, words, x)?;
            unpack_words(fb, to, &ws)
        }
        BitcastPlan::Lanes => {
            let bits = lanes_bits(fb, from, x)?;
            of_lanes_bits(fb, to, &bits)
        }
    }
}

/// The AIR bits of each lane of `x` (form `f`), each in an unsigned integer of its container's
/// width. A narrow float vector converts once as a vector, whose lanes are then taken; other forms
/// convert lane by lane.
fn lanes_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Vec<Value>> {
    let (whole, lf) = if f.is_narrow() && f.lanes > 1 {
        (narrow::to_bits(fb, f, x)?, None)
    } else {
        (x, Some(f.scalar_form()))
    };
    let mut out = Vec::with_capacity(usize::from(f.lanes));
    for i in 0..f.lanes {
        let e = lane(fb, f, whole, i)?;
        out.push(match lf {
            Some(lf) => lane_bits(fb, lf, e)?,
            None => e,
        });
    }
    Ok(out)
}

/// The value of form `to` whose lanes have the AIR bits `bits` (each masked, in the container's
/// integer). A narrow float vector converts once from the vector of bits; other forms convert lane
/// by lane.
fn of_lanes_bits(fb: &mut FunctionBuilder<'_>, to: Form, bits: &[Value]) -> BResult<Value> {
    if to.is_narrow() && to.lanes > 1 {
        let t = ty(fb, narrow::u32_form(to))?;
        let v = fb.composite_construct(t, bits)?;
        return narrow::from_bits(fb, to, v);
    }
    let lf = to.scalar_form();
    gather(fb, to, |fb, i| {
        let b = bits
            .get(usize::from(i))
            .copied()
            .ok_or_else(too_few_words)?;
        lane_of_bits(fb, lf, b)
    })
}

/// Lane `i` of `x` (form `f`).
fn lane(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, i: u8) -> BResult<Value> {
    if f.lanes == 1 {
        Ok(x)
    } else {
        fb.composite_extract(x, &[u32::from(i)])
    }
}

/// A lane of form `lane` as its AIR bits in an unsigned integer of its container's width
/// (a narrow float through its format bits, a `half`'s by the device's narrowing,
/// `narrow::to_bits`).
fn lane_bits(fb: &mut FunctionBuilder<'_>, lane: Form, v: Value) -> BResult<Value> {
    if lane.is_narrow() {
        narrow::to_bits(fb, lane, v)
    } else if lane.is_float() {
        let it = ty(fb, lane.int_of())?;
        fb.convert(ConvOp::Bitcast, v, it)
    } else {
        Ok(v)
    }
}

/// The lane of form `lane` whose AIR bits are `v` (masked, in the container's integer).
fn lane_of_bits(fb: &mut FunctionBuilder<'_>, lane: Form, v: Value) -> BResult<Value> {
    if lane.is_narrow() {
        narrow::from_bits(fb, lane, v)
    } else if lane.is_float() {
        let t = ty(fb, lane)?;
        fb.convert(ConvOp::Bitcast, v, t)
    } else {
        Ok(v)
    }
}

/// The lanes of `x` (form `from`) packed low first into the integer form `p`.
fn pack(fb: &mut FunctionBuilder<'_>, from: Form, p: Form, x: Value) -> BResult<Value> {
    let lf = from.scalar_form();
    let bits_form = lf.int_of().with_bits(from.bits());
    let mut acc: Option<Value> = None;
    for (i, e) in (0u8..).zip(lanes_bits(fb, from, x)?) {
        let e = recontain(fb, bits_form, p, e, false)?;
        let shifted = if i == 0 {
            e
        } else {
            let sh = splat(fb, p, u64::from(i) * u64::from(from.bits()))?;
            fb.binary(BinOp::ShiftLeftLogical, e, sh)?
        };
        acc = Some(match acc {
            None => shifted,
            Some(a) => fb.binary(BinOp::BitwiseOr, a, shifted)?,
        });
    }
    match acc {
        Some(a) => Ok(a),
        None => splat(fb, p, 0),
    }
}

/// Integer `w` (form `p`) unpacked low first into the lanes of `to`.
fn unpack(fb: &mut FunctionBuilder<'_>, p: Form, to: Form, w: Value) -> BResult<Value> {
    let lf = to.scalar_form();
    let lane_bits_n = to.bits();
    let bits_form = lf.int_of().with_bits(lane_bits_n);
    let mut bits = Vec::with_capacity(usize::from(to.lanes));
    for i in 0..to.lanes {
        let v = if i == 0 {
            w
        } else {
            let sh = splat(fb, p, u64::from(i) * u64::from(lane_bits_n))?;
            fb.binary(BinOp::ShiftRightLogical, w, sh)?
        };
        let v = recontain(fb, p, bits_form, v, false)?;
        bits.push(mask(fb, bits_form, v)?);
    }
    of_lanes_bits(fb, to, &bits)
}

/// The value of form `to` whose lane `i` is `take(i)`.
fn gather(
    fb: &mut FunctionBuilder<'_>,
    to: Form,
    mut take: impl FnMut(&mut FunctionBuilder<'_>, u8) -> BResult<Value>,
) -> BResult<Value> {
    if to.lanes == 1 {
        return take(fb, 0);
    }
    let mut parts = Vec::with_capacity(usize::from(to.lanes));
    for i in 0..to.lanes {
        parts.push(take(fb, i)?);
    }
    let t = ty(fb, to)?;
    fb.composite_construct(t, &parts)
}

/// A word plan whose lanes do not fit its words (a planning bug).
fn too_few_words() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Operand {
        op: crate::spv::Op::Bitcast,
        rule: crate::spirv::error::OperandRule::Arity,
    }
}

/// The `u32` form.
fn u32_scalar() -> Form {
    packed_form(32)
}

/// A 64-bit lane `v` (form `lane`) as its low and high words.
fn split64(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<(Value, Value)> {
    let u = u32_scalar();
    let ut = ty(fb, u)?;
    let pair = fb.module().ty(Type::Vector { elem: ut, len: 2 })?;
    let p = fb.convert(ConvOp::Bitcast, v, pair)?;
    Ok((
        fb.composite_extract(p, &[0])?,
        fb.composite_extract(p, &[1])?,
    ))
}

/// The lanes of `x` (form `from`, lanes of 8/16/32/64 bits) as `words` 32-bit words, low first.
pub(super) fn pack_words(
    fb: &mut FunctionBuilder<'_>,
    from: Form,
    words: u8,
    x: Value,
) -> BResult<Vec<Value>> {
    let u = u32_scalar();
    let n = u32::from(from.bits());
    let bits = if n == 64 {
        None
    } else {
        Some(lanes_bits(fb, from, x)?)
    };
    let mut out: Vec<Option<Value>> = vec![None; usize::from(words)];
    for i in 0..from.lanes {
        let at = u32::from(i) * n;
        let (w, off) = ((at / 32) as usize, at % 32);
        let parts: Vec<(usize, Value)> = if let Some(bits) = &bits {
            let b = bits
                .get(usize::from(i))
                .copied()
                .ok_or_else(too_few_words)?;
            let b = if off == 0 {
                b
            } else {
                let sh = splat(fb, u, u64::from(off))?;
                fb.binary(BinOp::ShiftLeftLogical, b, sh)?
            };
            vec![(w, b)]
        } else {
            let e = lane(fb, from, x, i)?;
            let (lo, hi) = split64(fb, e)?;
            vec![(w, lo), (w + 1, hi)]
        };
        for (w, b) in parts {
            let Some(slot) = out.get_mut(w) else {
                return Err(too_few_words());
            };
            *slot = Some(match *slot {
                None => b,
                Some(a) => fb.binary(BinOp::BitwiseOr, a, b)?,
            });
        }
    }
    let mut ws = Vec::with_capacity(out.len());
    for w in out {
        ws.push(match w {
            Some(w) => w,
            None => splat(fb, u, 0)?,
        });
    }
    Ok(ws)
}

/// Words `ws` (low first) unpacked into the lanes of `to` (lanes of 8/16/32/64 bits).
pub(super) fn unpack_words(fb: &mut FunctionBuilder<'_>, to: Form, ws: &[Value]) -> BResult<Value> {
    let u = u32_scalar();
    let lf = to.scalar_form();
    let n = u32::from(to.bits());
    let word = |i: usize| -> BResult<Value> { ws.get(i).copied().ok_or_else(too_few_words) };
    if n == 64 {
        return gather(fb, to, |fb, i| {
            let w = usize::from(i) * 2;
            let ut = ty(fb, u)?;
            let pair = fb.module().ty(Type::Vector { elem: ut, len: 2 })?;
            let p = fb.composite_construct(pair, &[word(w)?, word(w + 1)?])?;
            let t = ty(fb, lf)?;
            fb.convert(ConvOp::Bitcast, p, t)
        });
    }
    let mut bits = Vec::with_capacity(usize::from(to.lanes));
    for i in 0..to.lanes {
        let at = u32::from(i) * n;
        let (w, off) = ((at / 32) as usize, at % 32);
        let v = word(w)?;
        let v = if off == 0 {
            v
        } else {
            let sh = splat(fb, u, u64::from(off))?;
            fb.binary(BinOp::ShiftRightLogical, v, sh)?
        };
        bits.push(mask(fb, u.with_bits(to.bits()), v)?);
    }
    of_lanes_bits(fb, to, &bits)
}

impl Form {
    /// The scalar form of the element.
    pub fn scalar_form(self) -> Form {
        Form { lanes: 1, ..self }
    }

    /// The unsigned integer form of the element's container (a float's same-width integer).
    pub fn int_of(self) -> Form {
        let c = self.container();
        self.with(Held::Int {
            bits: c,
            container: if c <= 32 { 32 } else { 64 },
        })
    }

    /// This integer form holding `bits` in the same container.
    pub fn with_bits(self, bits: u8) -> Form {
        match self.held {
            Held::Int { container, .. } => self.with(Held::Int { bits, container }),
            _ => self,
        }
    }
}
