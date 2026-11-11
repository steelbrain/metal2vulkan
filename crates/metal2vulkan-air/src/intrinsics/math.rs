//! The math family: `air.[fast_]<function>[.<s|u>].<type>`, and OpenCL C's
//! `air.{native,half}_<function>.<type>`.
//!
//! One declarative table ([`TABLE`]) lists every function stem with its domain and operand
//! shape. A float function takes an optional `fast_` prefix and no sign token; a sign-sensitive
//! integer function requires `.s.` or `.u.`; a sign-agnostic integer function takes neither. The
//! same stem may name a float and an integer function (`min`, `max`, `clamp`): the sign token
//! decides, and the type token must agree with the domain.
//!
//! Apple's OpenCL compiler spells OpenCL C's `native_<function>` and `half_<function>` of a
//! `float` scalar or vector `air.native_<stem>.<type>` and `air.half_<stem>.<type>`, for the stems
//! [`OPENCL_STEMS`] lists (it spells `native_divide`, `native_recip`, `half_divide` and
//! `half_recip` as plain arithmetic). OpenCL C leaves their precision to the implementation, and
//! Metal's differ by device: `native_powr(100, 0.5)` answers `0x1.400002p+3` on an M3 Max (macOS
//! 27.0.1) and `0x1.4p+3` on a Radeon Pro 5300M (macOS 26.7.1). Each reads as the function at
//! [`Precision::Fast`], the device's fast form.
//!
//! Two functions reach AIR as calls to Metal runtime routines instead, declarations Apple's
//! compiler emits and its backend defines at pipeline creation: `frexp` as
//! `___metal_frexp_<T>_pthread<I>(T, thread I*)` (AIR 2.9 spells the same operation
//! `air.frexp.<T>(T, thread I*)` instead, and both read to one `Math`) and `ilogb` as
//! `___metal_ilogb_<T>(T, i32 mode)`,
//! `T` a `float` or `half` scalar or vector (`v<n>float`, `v<n>half`) and `I` its `int32` shape
//! ([`Math::parse_routine`]). They decode into this family as [`MathFn::Frexp`] and
//! [`MathFn::Ilogb`]; canonicalization replaces each call by the routine's measured answer in
//! plain AIR (`air::canon::routine`).

use super::sig::{ImmKind, Immediates, Operand, Signature};
use super::ty::{Elem, NumTy, Sign};
use crate::error::IntrinsicError;
use crate::ir::AddrSpace;

/// Whether a float function was spelled with the `fast_` prefix (or OpenCL C's `native_` or
/// `half_`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// Spelled without a prefix; always so for integer functions and the runtime routines.
    Precise,
    /// Spelled `air.fast_<stem>`: Metal's fast-math form; or `air.native_<stem>` or
    /// `air.half_<stem>`: OpenCL C's functions of the implementation's precision, read as that form
    /// (module doc).
    Fast,
}

/// The stems Apple's OpenCL compiler spells with the `native_` and `half_` prefixes, on `float`
/// types only: OpenCL C's `native_<stem>` and `half_<stem>` but `divide` and `recip`.
pub(super) const OPENCL_STEMS: &[&str] = &[
    "cos", "exp", "exp10", "exp2", "log", "log10", "log2", "powr", "rsqrt", "sin", "sqrt", "tan",
];

/// A math function. Integer functions whose result depends on signedness carry the [`Sign`] the
/// name spells; `Clz`/`Ctz` carry their `is_zero_undef` immediate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MathFn {
    // Float, one operand.
    /// `air.fabs`: the absolute value.
    Fabs,
    /// `air.sqrt`: the square root.
    Sqrt,
    /// `air.rsqrt`: the reciprocal square root, `1 / sqrt(x)`.
    Rsqrt,
    /// `air.exp`: `e` raised to `x`.
    Exp,
    /// `air.exp2`: 2 raised to `x`.
    Exp2,
    /// `air.exp10`: 10 raised to `x`.
    Exp10,
    /// `air.log`: the natural logarithm.
    Log,
    /// `air.log2`: the base-2 logarithm.
    Log2,
    /// `air.log10`: the base-10 logarithm.
    Log10,
    /// `air.sin`: the sine of `x` radians.
    Sin,
    /// `air.cos`: the cosine of `x` radians.
    Cos,
    /// `air.tan`: the tangent of `x` radians.
    Tan,
    /// `air.asin`: the arc sine, in radians.
    Asin,
    /// `air.acos`: the arc cosine, in radians.
    Acos,
    /// `air.atan`: the arc tangent, in radians.
    Atan,
    /// `air.sinh`: the hyperbolic sine.
    Sinh,
    /// `air.cosh`: the hyperbolic cosine.
    Cosh,
    /// `air.tanh`: the hyperbolic tangent.
    Tanh,
    /// `air.asinh`: the inverse hyperbolic sine.
    Asinh,
    /// `air.acosh`: the inverse hyperbolic cosine.
    Acosh,
    /// `air.atanh`: the inverse hyperbolic tangent.
    Atanh,
    /// `air.sinpi`: `sin(π x)`.
    SinPi,
    /// `air.cospi`: `cos(π x)`.
    CosPi,
    /// `air.tanpi`: `tan(π x)`.
    TanPi,
    /// `air.floor`: `x` rounded toward negative infinity.
    Floor,
    /// `air.ceil`: `x` rounded toward positive infinity.
    Ceil,
    /// `air.round`: `x` rounded to the nearest integer, halfway cases away from zero.
    Round,
    /// `air.rint`: `x` rounded to the nearest integer, halfway cases to even.
    Rint,
    /// `air.trunc`: `x` rounded toward zero.
    Trunc,
    /// `air.fract`: `x - floor(x)`, bounded below one.
    Fract,
    /// `air.sign`: Metal's float `sign`: 1 above zero, -1 below it, and +0 otherwise.
    FSign,
    /// `air.saturate`: `x` clamped to `[0, 1]`.
    Saturate,
    // Float, two operands.
    /// `air.fmin`: the lesser operand, or the other one where one is a NaN.
    FMin,
    /// `air.fmax`: the greater operand, or the other one where one is a NaN.
    FMax,
    /// `air.min` with a float type: Metal's float `min`, computed as `fmin`.
    Min,
    /// `air.max` with a float type: Metal's float `max`, computed as `fmax`.
    Max,
    /// `air.pow`: `x` raised to `y`.
    Pow,
    /// `air.powr`: `x` raised to `y`, for `x >= 0`.
    Powr,
    /// `air.fmod`: `x - y * trunc(x / y)`.
    Fmod,
    /// `air.atan2`: the arc tangent of `y / x` (operands `y`, `x`) by quadrant, in radians.
    Atan2,
    // Float, three operands.
    /// `air.fma`: `a * b + c`, rounded once.
    Fma,
    /// `air.mix`: `x + (y - x) * a`.
    Mix,
    /// `air.clamp` with a float type: `x` clamped to `[min, max]`.
    Clamp,
    /// `air.fmin3`: the least of three operands.
    FMin3,
    /// `air.fmax3`: the greatest of three operands.
    FMax3,
    /// `air.fmedian3`: the median of three operands.
    FMedian3,
    // Float, special shapes.
    /// `air.ldexp`: `x * 2^k`, `k` an `int` per lane.
    Ldexp,
    /// `air.dot`: the scalar dot product of two vectors.
    Dot,
    /// `air.sincos`: the sine, with the cosine stored through a thread pointer.
    SinCos,
    /// `___metal_frexp_*`: the mantissa, with the exponent stored through a thread pointer.
    Frexp,
    /// `___metal_ilogb_*`: the unbiased exponent as an `int`, of the measured modes 0 and 3.
    Ilogb,
    // Integer, sign-sensitive.
    /// `air.abs.<s|u>`: the integer absolute value (the operand itself when unsigned).
    Abs(Sign),
    /// `air.abs_diff.<s|u>`: `|x - y|`, without overflow.
    AbsDiff(Sign),
    /// `air.min.<s|u>`: the lesser integer.
    IMin(Sign),
    /// `air.max.<s|u>`: the greater integer.
    IMax(Sign),
    /// `air.clamp.<s|u>`: an integer clamped to `[min, max]`.
    IClamp(Sign),
    /// `air.min3.<s|u>`: the least of three integers.
    IMin3(Sign),
    /// `air.max3.<s|u>`: the greatest of three integers.
    IMax3(Sign),
    /// `air.median3.<s|u>`: the middle one of three integers.
    IMedian3(Sign),
    /// `air.mul_hi.<s|u>`: the high half of the full product.
    MulHi(Sign),
    /// `air.mad_hi.<s|u>`: `mul_hi(a, b) + c`.
    MadHi(Sign),
    /// `air.mad_sat.<s|u>`: `a * b + c`, saturated.
    MadSat(Sign),
    /// `air.add_sat.<s|u>`: `x + y`, saturated.
    AddSat(Sign),
    /// `air.sub_sat.<s|u>`: `x - y`, saturated.
    SubSat(Sign),
    /// `air.hadd.<s|u>`: `(x + y) >> 1`, without overflow in the sum.
    HAdd(Sign),
    /// `air.rhadd.<s|u>`: `(x + y + 1) >> 1`, without overflow in the sum.
    RHAdd(Sign),
    /// `air.extract_bits.<s|u>`: the `bits` bits of `x` from bit `offset`, sign- or zero-extended
    /// as the sign says.
    ExtractBits(Sign),
    /// `air.insert_bits.<s|u>`: `base` with its `bits` bits from bit `offset` replaced by the low
    /// bits of `insert`.
    InsertBits(Sign),
    // Integer, sign-agnostic.
    /// `air.popcount`: the number of set bits.
    Popcount,
    /// `air.reverse_bits`: the bits in reverse order.
    ReverseBits,
    /// `air.bswap`: the bytes in reverse order.
    Bswap,
    /// `air.rotate`: `v` rotated left by `i` bits.
    Rotate,
    /// `air.clz`: the number of leading zero bits.
    Clz {
        /// The `is_zero_undef` immediate: whether the result for a zero operand is undefined.
        zero_undef: bool,
    },
    /// `air.ctz`: the number of trailing zero bits.
    Ctz {
        /// The `is_zero_undef` immediate: whether the result for a zero operand is undefined.
        zero_undef: bool,
    },
}

/// A decoded math intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Math {
    /// The function, with the sign and immediate it carries.
    pub f: MathFn,
    /// Whether the name spelled the `fast_` prefix.
    pub precision: Precision,
    /// The type the name spells: the operand type (for `dot`, the vector operand).
    pub ty: NumTy,
}

/// The value domain of a table entry.
#[derive(Clone, Copy, Debug)]
pub(super) enum Domain {
    /// Float elements; `fast_` allowed; no sign token.
    Float(MathFn),
    /// Integer elements; a sign token is required.
    Signed(fn(Sign) -> MathFn),
    /// Integer elements; no sign token.
    Int(MathFn),
}

/// The operand shape of a table entry, with `T` the spelled type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// `T (T × n)`.
    Same(u8),
    /// `T (T, <lanes × i32>)`.
    Ldexp,
    /// `elem (T, T)`, `T` a vector.
    Dot,
    /// `T (T, ptr)`: the cosine is stored through a thread pointer.
    SinCos,
    /// `T (T, ptr)`: the exponent is stored through a thread pointer.
    Frexp,
    /// `<lanes x i32> (T, i32 immediate)`.
    Ilogb,
    /// `T (T, i1 immediate)`.
    CountZeros,
    /// `T (T, i32 offset, i32 bits)`.
    ExtractBits,
    /// `T (T base, T insert, i32 offset, i32 bits)`.
    InsertBits,
}

/// Every math stem. Order is irrelevant; a stem appears once per domain.
pub(super) const TABLE: &[(&str, Domain, Shape)] = {
    use Domain::{Float as F, Int as I, Signed as S};
    use MathFn as M;
    use Shape::*;
    &[
        ("fabs", F(M::Fabs), Same(1)),
        ("sqrt", F(M::Sqrt), Same(1)),
        ("rsqrt", F(M::Rsqrt), Same(1)),
        ("exp", F(M::Exp), Same(1)),
        ("exp2", F(M::Exp2), Same(1)),
        ("exp10", F(M::Exp10), Same(1)),
        ("log", F(M::Log), Same(1)),
        ("log2", F(M::Log2), Same(1)),
        ("log10", F(M::Log10), Same(1)),
        ("sin", F(M::Sin), Same(1)),
        ("cos", F(M::Cos), Same(1)),
        ("tan", F(M::Tan), Same(1)),
        ("asin", F(M::Asin), Same(1)),
        ("acos", F(M::Acos), Same(1)),
        ("atan", F(M::Atan), Same(1)),
        ("sinh", F(M::Sinh), Same(1)),
        ("cosh", F(M::Cosh), Same(1)),
        ("tanh", F(M::Tanh), Same(1)),
        ("asinh", F(M::Asinh), Same(1)),
        ("acosh", F(M::Acosh), Same(1)),
        ("atanh", F(M::Atanh), Same(1)),
        ("sinpi", F(M::SinPi), Same(1)),
        ("cospi", F(M::CosPi), Same(1)),
        ("tanpi", F(M::TanPi), Same(1)),
        ("floor", F(M::Floor), Same(1)),
        ("ceil", F(M::Ceil), Same(1)),
        ("round", F(M::Round), Same(1)),
        ("rint", F(M::Rint), Same(1)),
        ("trunc", F(M::Trunc), Same(1)),
        ("fract", F(M::Fract), Same(1)),
        ("sign", F(M::FSign), Same(1)),
        ("saturate", F(M::Saturate), Same(1)),
        ("fmin", F(M::FMin), Same(2)),
        ("fmax", F(M::FMax), Same(2)),
        ("min", F(M::Min), Same(2)),
        ("max", F(M::Max), Same(2)),
        ("pow", F(M::Pow), Same(2)),
        ("powr", F(M::Powr), Same(2)),
        ("fmod", F(M::Fmod), Same(2)),
        ("atan2", F(M::Atan2), Same(2)),
        ("fma", F(M::Fma), Same(3)),
        ("mix", F(M::Mix), Same(3)),
        ("clamp", F(M::Clamp), Same(3)),
        ("fmin3", F(M::FMin3), Same(3)),
        ("fmax3", F(M::FMax3), Same(3)),
        ("fmedian3", F(M::FMedian3), Same(3)),
        ("ldexp", F(M::Ldexp), Ldexp),
        ("dot", F(M::Dot), Dot),
        ("sincos", F(M::SinCos), SinCos),
        // AIR 2.9 (deployment target macOS 27) spells `frexp` as this intrinsic. Every earlier
        // emitter outlines it as the runtime routine `___metal_frexp_<T>_pthread<I>`, which
        // `parse_routine` reads to the same `Math`, so both spellings reach one expansion
        // (`canon::routine`).
        ("frexp", F(M::Frexp), Frexp),
        ("abs", S(M::Abs), Same(1)),
        ("abs_diff", S(M::AbsDiff), Same(2)),
        ("min", S(M::IMin), Same(2)),
        ("max", S(M::IMax), Same(2)),
        ("clamp", S(M::IClamp), Same(3)),
        ("min3", S(M::IMin3), Same(3)),
        ("max3", S(M::IMax3), Same(3)),
        ("median3", S(M::IMedian3), Same(3)),
        ("mul_hi", S(M::MulHi), Same(2)),
        ("mad_hi", S(M::MadHi), Same(3)),
        ("mad_sat", S(M::MadSat), Same(3)),
        ("add_sat", S(M::AddSat), Same(2)),
        ("sub_sat", S(M::SubSat), Same(2)),
        ("hadd", S(M::HAdd), Same(2)),
        ("rhadd", S(M::RHAdd), Same(2)),
        ("extract_bits", S(M::ExtractBits), ExtractBits),
        ("insert_bits", S(M::InsertBits), InsertBits),
        ("popcount", I(M::Popcount), Same(1)),
        ("reverse_bits", I(M::ReverseBits), Same(1)),
        ("bswap", I(M::Bswap), Same(1)),
        ("rotate", I(M::Rotate), Same(2)),
        ("clz", I(M::Clz { zero_undef: false }), CountZeros),
        ("ctz", I(M::Ctz { zero_undef: false }), CountZeros),
    ]
};

impl Math {
    /// Parse the part of a name after `air.`: `[fast_]stem.[s|u.]type`, or
    /// `{native,half}_stem.type` of a stem [`OPENCL_STEMS`] lists and a `float` type. `None` is not
    /// a math name (another family's, or an unknown one).
    pub(super) fn parse(rest: &str) -> Option<Math> {
        let opencl = ["native_", "half_"]
            .iter()
            .find_map(|p| rest.strip_prefix(p));
        if let Some(r) = opencl {
            let (stem, ty) = r.split_once('.')?;
            let float = NumTy::parse(ty).is_some_and(|t| t.elem == Elem::F32);
            if !(OPENCL_STEMS.contains(&stem) && float) {
                return None;
            }
            return Math::parse(&format!("fast_{r}"));
        }
        let (fast, rest) = match rest.strip_prefix("fast_") {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let mut tokens = rest.split('.');
        let stem = tokens.next()?;
        let (sign, ty) = match (tokens.next()?, tokens.next(), tokens.next()) {
            (ty, None, _) => (None, ty),
            (sign, Some(ty), None) => (Some(Sign::parse(sign)?), ty),
            _ => return None,
        };
        let ty = NumTy::parse(ty)?;
        let (f, precision) =
            TABLE
                .iter()
                .filter(|(s, _, _)| *s == stem)
                .find_map(|(_, domain, _)| match (*domain, sign) {
                    (Domain::Float(f), None) if ty.elem.is_float() => Some((
                        f,
                        if fast {
                            Precision::Fast
                        } else {
                            Precision::Precise
                        },
                    )),
                    (Domain::Signed(f), Some(s)) if !fast && ty.elem.is_int() => {
                        Some((f(s), Precision::Precise))
                    }
                    (Domain::Int(f), None) if !fast && ty.elem.is_int() => {
                        Some((f, Precision::Precise))
                    }
                    _ => None,
                })?;
        if f == MathFn::Dot && !ty.is_vector() {
            return None;
        }
        Some(Math { f, precision, ty })
    }

    /// The stem of every float function, as `air.[fast_]<stem>.<type>` spells it.
    pub fn float_stems() -> impl Iterator<Item = &'static str> {
        TABLE
            .iter()
            .filter_map(|(stem, domain, _)| matches!(domain, Domain::Float(_)).then_some(*stem))
    }

    /// Parse the part of a Metal runtime routine's name after `___metal_`:
    /// `frexp_<T>_pthread<I>` or `ilogb_<T>`, `T` one of `float`, `half`, `v<n>float`,
    /// `v<n>half` and `I` the `int32` of its lanes (`int32`, `v<n>int32`). `None` is no routine
    /// this family models.
    pub(super) fn parse_routine(rest: &str) -> Option<Math> {
        /// A routine type token: `float`, `half`, `int32`, or `v<n>` of one.
        fn ty(token: &str) -> Option<NumTy> {
            let (lanes, elem) = match token.strip_prefix('v') {
                Some(rest) => {
                    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                    let (count, elem) = rest.split_at_checked(digits)?;
                    if count.starts_with('0') {
                        return None;
                    }
                    let lanes: u32 = count.parse().ok()?;
                    (lanes >= 2).then_some((lanes, elem))?
                }
                None => (1, token),
            };
            let elem = match elem {
                "float" => Elem::F32,
                "half" => Elem::F16,
                "int32" => Elem::I32,
                _ => return None,
            };
            Some(NumTy { elem, lanes })
        }
        let (f, ty) = if let Some(rest) = rest.strip_prefix("frexp_") {
            let (t, i) = rest.split_once("_pthread")?;
            let (t, i) = (ty(t)?, ty(i)?);
            (i == t.with_elem(Elem::I32)).then_some((MathFn::Frexp, t))?
        } else {
            (MathFn::Ilogb, ty(rest.strip_prefix("ilogb_")?)?)
        };
        ty.elem.is_float().then_some(Math {
            f,
            precision: Precision::Precise,
            ty,
        })
    }

    /// The operand shape of this function.
    pub(super) fn shape(&self) -> Shape {
        match self.f {
            MathFn::Frexp => return Shape::Frexp,
            MathFn::Ilogb => return Shape::Ilogb,
            _ => {}
        }
        TABLE
            .iter()
            .find(|(_, domain, _)| match *domain {
                Domain::Float(f) | Domain::Int(f) => {
                    std::mem::discriminant(&f) == std::mem::discriminant(&self.f)
                }
                Domain::Signed(f) => Sign::ALL.iter().any(|&s| f(s) == self.f),
            })
            .map_or(Shape::Same(1), |(_, _, shape)| *shape)
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let t = Operand::Num(self.ty);
        let i32s = Operand::Num(NumTy::scalar(Elem::I32));
        match self.shape() {
            Shape::Same(1) => Signature::new(t, &[t]),
            Shape::Same(2) => Signature::new(t, &[t, t]),
            Shape::Same(_) => Signature::new(t, &[t, t, t]),
            Shape::Ldexp => Signature::new(t, &[t, Operand::Num(self.ty.with_elem(Elem::I32))]),
            Shape::Dot => Signature::new(Operand::Num(NumTy::scalar(self.ty.elem)), &[t, t]),
            Shape::SinCos | Shape::Frexp => {
                Signature::new(t, &[t, Operand::Ptr(AddrSpace::Thread)])
            }
            Shape::Ilogb => Signature::new(
                Operand::Num(self.ty.with_elem(Elem::I32)),
                &[t, Operand::Imm(ImmKind::I32)],
            ),
            Shape::CountZeros => Signature::new(t, &[t, Operand::Imm(ImmKind::Bool)]),
            Shape::ExtractBits => Signature::new(t, &[t, i32s, i32s]),
            Shape::InsertBits => Signature::new(t, &[t, t, i32s, i32s]),
        }
    }

    /// Read the call's immediates into the variant. `Ok(None)`: an `ilogb` mode other than the
    /// measured 0 (`ilogb`, `fast::ilogb`) and 3 (`precise::ilogb`), which answer alike.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<Option<Math>, IntrinsicError> {
        let f = match self.f {
            MathFn::Clz { .. } => MathFn::Clz {
                zero_undef: imm.flag(1)?,
            },
            MathFn::Ctz { .. } => MathFn::Ctz {
                zero_undef: imm.flag(1)?,
            },
            MathFn::Ilogb if !matches!(imm.get(1)?, 0 | 3) => return Ok(None),
            f => f,
        };
        Ok(Some(Math { f, ..self }))
    }
}
