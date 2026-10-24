//! Metal's NaN results.
//!
//! Metal answers one NaN per operation class, whatever NaN its operands carry, signaling or
//! quiet, of either sign and any payload, with or without fast math (the NaN-model probes): its
//! `float` arithmetic (`+ - * /`, the fused multiply-add, `sqrt`, `fmod`) and most of its `float`
//! functions answer the canonical NaN `0x7FC00000`, sign and payload dropped
//! ([`NanAnswer::Canonical`]); `asin`, `atan`, `asinh`, and `atanh` keep a NaN operand's sign
//! ([`NanAnswer::OperandSign`]); `fast::asin` gives a domain error the operand's sign
//! ([`NanAnswer::DomainSign`]); and `round`, `ldexp`, and `fast::tan` return a NaN operand itself
//! ([`NanAnswer::Operand`]). Its `half` `+`, `*`, and fused multiply-add, and its narrowing of a
//! `float` to `half` (`fptrunc` and `air.convert`), answer the canonical `half` NaN `0x7E00`, held
//! widened as `0x7FC00000`; its widening of a `half` NaN to `float`, and its `half` `fneg` and
//! `fabs`, answer the canonical NaN too ([`NanAnswer::CanonicalExact`]). Its `float` `fneg` and
//! `fabs` keep a NaN operand's payload, which the lowering moves on the bits
//! (`math::sign_on_bits`), no select needed. Vulkan states no NaN encoding, and the devices
//! measured answer their own:
//! an operand's bits, quieted or not, `0xFFC00000`, or `0x7FFFFFFF`, and for `half` `0xFE00`, the
//! operand's payload, or `0x7FFF`. Each device answers a NaN on exactly the lanes Metal does for
//! these classes, so selecting Metal's NaN where the result is a NaN ([`select`]) is exact on
//! every device.
//!
//! The select is applied where the result's bits can be observed: a result whose every use
//! drops its operands' NaN bits ([`drops_payload`]: a float comparison, or an operation that
//! itself answers the canonical NaN) needs none, so a chain of arithmetic selects once, at its
//! end, and the device may still contract an `fmul` into the `fadd` that uses it
//! ([`observed_payloads`]). Where the profile states that the device's arithmetic answers Metal's
//! canonical NaN ([`NanEncoding::MetalCanonical`], a measured device fact), the canonical class
//! of its arithmetic is left to the device; the other classes, and the canonical NaN of an
//! operation the device computes exactly, are selected on every profile.
//!
//! The NaN test is `OpIsNan` where the entry declares `SignedZeroInfNanPreserve` for 32-bit
//! floats, without which a device may assume that no operand is a NaN and fold the test; the
//! module then tests the bits instead (`(bits & 0x7FFFFFFF) > 0x7F800000`), which no float mode
//! reaches. The translation decorates no value `FPFastMathMode`. The select itself is a `float`
//! `OpSelect`, which such a device may still fold to its operand, since the condition holds
//! exactly where the operand is a NaN; where the answer is read bit for bit the module answers on
//! the bits alone instead ([`canonical_on_bits`]).

use crate::air::intrinsics::{AirIntrinsic, NumKind, Precision};
use crate::air::ir::{
    BinOp as AirBin, Body, Callee, CastOp, FastMath, FuncId, Inst, InstKind, Type, TypeId, Value,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::device::profile::NanEncoding;
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::lower::arith::divide::Reciprocal;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::FloatFn;
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, UnOp};

/// Metal's canonical `float` NaN.
pub const CANONICAL: u64 = 0x7FC0_0000;

/// Which NaN Metal answers where an operation's result is a NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NanAnswer {
    /// `0x7FC00000`: the arithmetic and most functions, and the narrowing to `half`; results the
    /// device computes by rounding arithmetic.
    Canonical,
    /// `0x7FC00000` for a result the device computes exactly, passing its operand's NaN on: the
    /// widening of a `half` to `float`, which a `half` held widened leaves no device operation at
    /// all, and `half` `fneg` and `fabs`, which a device computes on the sign alone. No device's
    /// arithmetic answers it, so no device fact leaves it to the device.
    CanonicalExact,
    /// The canonical NaN with a NaN first operand's sign, and `+` for a NaN of another origin (a
    /// domain error): precise `asin`, `atan`, `asinh`, `atanh`.
    OperandSign,
    /// The canonical NaN, `+` for a NaN first operand and with the first operand's sign for a
    /// domain error: `fast::asin`.
    DomainSign,
    /// A NaN first operand itself, bit for bit, and the canonical NaN for a NaN of another
    /// origin: `round`, `ldexp`, `fast::tan`.
    Operand,
}

impl NanAnswer {
    /// Whether the answer is left to the device under `encoding`: on a device whose arithmetic
    /// answers Metal's canonical NaN ([`NanEncoding::MetalCanonical`]) the canonical class of the
    /// arithmetic, which the device fact covers, and not [`NanAnswer::CanonicalExact`]; under
    /// native device maths ([`NanEncoding::Native`]) every answer, the device's NaN noted
    /// `PrecisionNote::NativeNanEncoding`.
    pub fn left_to(self, encoding: NanEncoding) -> bool {
        match encoding {
            NanEncoding::MetalCanonical => self == NanAnswer::Canonical,
            NanEncoding::Native => true,
            _ => false,
        }
    }

    /// Whether the answer reads the first operand.
    pub fn reads_operand(self) -> bool {
        !matches!(self, NanAnswer::Canonical | NanAnswer::CanonicalExact)
    }

    /// An upper estimate of the SPIR-V words of [`select`] on a result of form `f`: the bits tests,
    /// the selects, and the constants' share.
    pub fn words(self, f: Form) -> u32 {
        let lanes = u32::from(f.lanes);
        let base = match self {
            NanAnswer::Canonical | NanAnswer::CanonicalExact => 32,
            NanAnswer::Operand => 56,
            NanAnswer::OperandSign | NanAnswer::DomainSign => 80,
        };
        base + 4 * lanes
    }
}

/// How [`select`] tests a value for a NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanTest {
    /// `OpIsNan`: the entry declares `SignedZeroInfNanPreserve` for the width.
    IsNan,
    /// The bits: no float mode lets a device fold an integer comparison.
    Bits,
}

impl NanTest {
    /// The test for `float` lanes in an entry that declares `SignedZeroInfNanPreserve` for the
    /// widths `modes` (`ArithPlan::modes`).
    pub fn for_modes(modes: &std::collections::BTreeSet<u8>) -> NanTest {
        NanTest::for_width(modes, 32)
    }

    /// The test for lanes of float container width `width` (32 or 64) in an entry that declares
    /// `SignedZeroInfNanPreserve` for the widths `modes`.
    pub fn for_width(modes: &std::collections::BTreeSet<u8>, width: u8) -> NanTest {
        if modes.contains(&width) {
            NanTest::IsNan
        } else {
            NanTest::Bits
        }
    }
}

/// The answer Metal gives for a NaN result of float function `op` (as selected: the `fast_`
/// refinement applied, `fast` whether its symbol is the `fast_` variant) called with every
/// fast-math flag if `flagged` (`arith::every_flag`), on lanes of format `fmt`. Measured in two
/// states: the precise functions without flags and the `fast_` ones with every flag, as Metal's
/// compiler emits them without and with fast math (`measured`). A precise symbol called with every
/// flag (MSL's `precise::` functions in a fast-math compile) answers as it does without them where
/// that state is measured ([`flagged_precise`]). A call in another state (a `fast_` symbol without
/// every flag, another precise one with them) has the answer only where both states of the function
/// agree, the precise one and its `fast_` variant: Metal's `fast_fract` answers the canonical NaN
/// in both states (A1 case aed3faaa, without flags), but where the variants differ a call in the
/// other state is not measured. `None` there, and where `measured` has none. On `half` lanes the
/// answers are `half_function_answer`'s; `bfloat` and `double` have none.
pub fn function_answer(op: FloatFn, fast: bool, flagged: bool, fmt: FloatFmt) -> Option<NanAnswer> {
    match fmt {
        FloatFmt::F32 => {}
        FloatFmt::Half => return half_function_answer(op, fast),
        FloatFmt::BFloat | FloatFmt::F64 => return None,
    }
    let here = measured(op, fast)?;
    if fast == flagged || (!fast && flagged_precise(op)) {
        return Some(here);
    }
    let other = if fast {
        precise_of(op)
    } else {
        // The reciprocal a refinement divides by does not decide its NaN.
        super::refine_float(op, true, FastMath::NONE, f32_scalar(), Reciprocal::Device)
    };
    (measured(other, !fast) == Some(here)).then_some(here)
}

/// A `float` scalar.
fn f32_scalar() -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    }
}

/// The precise function a `fast_` refinement of `op` stands for (`super::refine_float`), and `op`
/// itself where it is none.
fn precise_of(op: FloatFn) -> FloatFn {
    use FloatFn as F;
    match op {
        F::FastTanh { .. } => F::Tanh,
        F::FastAsinh => F::Asinh,
        F::FastSinh => F::Sinh,
        F::FastCosh => F::Cosh,
        F::FastAcosh => F::Acosh,
        F::FastAtanh { .. } => F::Atanh,
        F::FastExp10 => F::Exp10,
        F::FastLog10 => F::Log10,
        F::FastFract { .. } => F::Fract,
        F::FastMedian3 => F::Median3,
        F::FastPow => F::Pow,
        F::FastSin => F::Sin,
        F::FastCos => F::Cos,
        F::FastSqrt => F::Sqrt,
        F::FastExp => F::Exp,
        F::FastLog => F::Log,
        F::FastLog2 => F::Log2,
        F::FastPowr => F::Powr,
        F::FastAtan2 { .. } => F::Atan2,
        F::FastTan { .. } => F::Tan,
        F::FastAsin => F::Asin,
        F::FastAcos => F::Acos,
        F::FastAtan => F::Atan,
        F::FastRsqrt => F::Rsqrt,
        F::FastSinPi => F::SinPi,
        F::FastCosPi => F::CosPi,
        F::FastTanPi => F::TanPi,
        op => op,
    }
}

/// The answer Metal gives for a NaN result of `float` function `op` (refined as in
/// [`function_answer`]) in the state measured: a precise symbol without fast-math flags
/// (`nan_functions`), a `fast_` one (`fast`) with every flag (`nan_functions_fast`). Functions
/// whose lowering answers it itself ([`FloatFn::metal_nans`]) are listed too: their result drops
/// its operands' NaN bits ([`drops_payload`]). `None` where Metal answers a number for a NaN
/// operand (`sign`, `saturate`, `fast::sin`, `fast::cos`), where the result is an operand
/// (`fmin`, `fmax`, `median3`, `copysign`, `fabs`, and precise `pow`, whose `pow(x, 1)` is `x`).
/// `sinpi`, `cospi`, and `tanpi` answer the canonical NaN in both modes for an infinity or a NaN.
/// The math-library branch's refinements of `fast_tan`, `fast_asin`, `fast_acos`,
/// `fast_atan`, and `fast_rsqrt` answer as those calls were measured: a `fast_` symbol's state
/// either way.
fn measured(op: FloatFn, fast: bool) -> Option<NanAnswer> {
    use FloatFn as F;
    use NanAnswer as A;
    Some(match op {
        F::Sqrt
        | F::FastSqrt
        | F::Rsqrt
        | F::FastRsqrt
        | F::Exp
        | F::FastExp
        | F::Exp2
        | F::Exp10
        | F::FastExp10
        | F::Log
        | F::FastLog
        | F::Log2
        | F::FastLog2
        | F::Log10
        | F::FastLog10
        | F::Acos
        | F::FastAcos
        | F::Sinh
        | F::FastSinh
        | F::Cosh
        | F::FastCosh
        | F::Tanh
        | F::FastTanh { .. }
        | F::FastAsinh
        | F::FastAtanh { .. }
        | F::Acosh
        | F::FastAcosh
        | F::Powr
        | F::FastPowr
        | F::Floor
        | F::Ceil
        | F::Trunc
        | F::Rint
        | F::Fract
        | F::FastFract { .. }
        | F::Atan2
        | F::FastAtan2 { .. }
        | F::FastPow
        | F::Dot
        | F::Mix
        | F::Fma
        | F::MulAdd
        | F::Fmod
        | F::SinPi
        | F::CosPi
        | F::TanPi
        | F::FastSinPi
        | F::FastCosPi
        | F::FastTanPi => A::Canonical,
        F::Sin | F::Cos if !fast => A::Canonical,
        F::Tan if fast => A::Operand,
        F::FastTan { .. } => A::Operand,
        F::Tan => A::Canonical,
        F::Asin if fast => A::DomainSign,
        F::FastAsin => A::DomainSign,
        F::Atan | F::Asinh | F::Atanh if fast => A::Canonical,
        F::FastAtan => A::Canonical,
        F::Asin | F::Atan | F::Asinh | F::Atanh => A::OperandSign,
        F::Round | F::Ldexp => A::Operand,
        _ => return None,
    })
}

/// The answer Metal gives for a NaN result of `half` function `op` (as in [`function_answer`]),
/// measured in both states, without flags (`nan_narrow`, `nan_half`) and with every flag
/// (`nan_narrow_fast`, `nan_half_fast`, word for word the same answers), so the flags do not
/// decide it: the canonical `half` NaN of the fused multiply-add `air.fma.f16` and `llvm.fmuladd`,
/// of `mix`, `dot`, `ldexp`, `fmod`, `sqrt`, `rsqrt`, `floor`, `ceil`, `trunc`, `rint`, `pow`,
/// `powr`, and the transcendental functions but `asin` (`atan`, `asinh`, `atanh`, and `ldexp`
/// unlike `float`'s), and of `fabs`, which the device computes exactly (class D); `asin`'s domain
/// error with the operand's sign (`0xFE00` below `-1`); and `round`'s NaN operand itself, a
/// signaling one kept. Metal has no `fast_` `half` function, so none is answered. `None` for the
/// functions that answer a number or an operand for a NaN (`fract`, `sign`, `saturate`, `fmin` and
/// its family, `median3`, `copysign`). `sinpi`, `cospi`, and `tanpi` answer the canonical `half`
/// NaN for an infinity or a NaN.
fn half_function_answer(op: FloatFn, fast: bool) -> Option<NanAnswer> {
    use FloatFn as F;
    use NanAnswer as A;
    if fast {
        return None;
    }
    Some(match op {
        F::Fma
        | F::MulAdd
        | F::Mix
        | F::Dot
        | F::Ldexp
        | F::Fmod
        | F::Sqrt
        | F::Rsqrt
        | F::Floor
        | F::Ceil
        | F::Trunc
        | F::Rint
        | F::Pow
        | F::Powr
        | F::Exp
        | F::Exp2
        | F::Exp10
        | F::Log
        | F::Log2
        | F::Log10
        | F::Sin
        | F::Cos
        | F::Tan
        | F::Acos
        | F::Atan
        | F::Atan2
        | F::Sinh
        | F::Cosh
        | F::Tanh
        | F::Asinh
        | F::Acosh
        | F::Atanh
        | F::SinPi
        | F::CosPi
        | F::TanPi => A::Canonical,
        F::Asin => A::DomainSign,
        F::Round => A::Operand,
        F::Abs => A::CanonicalExact,
        _ => return None,
    })
}

/// The answer Metal gives for a NaN result of `fneg` on lanes of format `fmt`: the canonical
/// `half` NaN for `half`, with and without fast-math flags (`nan_narrow`, `nan_narrow_fast`,
/// `nan_half`, `nan_half_fast`), which the device computes exactly, flipping the held value's
/// sign (class D). A `float` `fneg` keeps its operand's NaN with the sign flipped, which the
/// lowering moves on the bits (`math::sign_on_bits`, class E); the other formats are not
/// measured.
pub fn fneg_answer(fmt: FloatFmt) -> Option<NanAnswer> {
    (fmt == FloatFmt::Half).then_some(NanAnswer::CanonicalExact)
}

/// The answer Metal gives for a NaN result of the float binary operator `op` on lanes of format
/// `fmt`: the canonical NaN for `float` `+ - * /` (`nan_arith`, `nan_arith_fast`) and the
/// canonical `half` NaN for `half` `+ - * /` (`nan_narrow`, `nan_narrow_fast`, `nan_half`,
/// `nan_half_fast`). Each is measured without fast-math flags and with every flag, with the same
/// answer, so the operator's flags do not decide it. A division's shared routine answers it itself.
/// `frem` and the other formats are not measured.
pub fn binary_answer(op: AirBin, fmt: FloatFmt) -> Option<NanAnswer> {
    let measured = match fmt {
        FloatFmt::F32 | FloatFmt::Half => matches!(
            op,
            AirBin::FAdd | AirBin::FSub | AirBin::FMul | AirBin::FDiv
        ),
        FloatFmt::BFloat | FloatFmt::F64 => false,
    };
    measured.then_some(NanAnswer::Canonical)
}

/// The answer Metal gives for a NaN result of the float conversion `op` from format `from` to
/// `to`: the canonical `half` NaN for a `float` narrowed to `half` (`fptrunc`, `nan_narrow`), and
/// the canonical NaN, exactly computed, for a `half` widened to `float` (`fpext`,
/// `hrd_half_widen`, every NaN half). Neither conversion carries fast-math flags in either math
/// mode (`nan_narrow_fast`). The other conversions are not measured here.
pub fn cast_answer(op: CastOp, from: FloatFmt, to: FloatFmt) -> Option<NanAnswer> {
    match (op, from, to) {
        (CastOp::FPTrunc, FloatFmt::F32, FloatFmt::Half) => Some(NanAnswer::Canonical),
        (CastOp::FPExt, FloatFmt::Half, FloatFmt::F32) => Some(NanAnswer::CanonicalExact),
        _ => None,
    }
}

/// The answer Metal gives for a NaN result of the float-to-float `air.convert` from format `from`
/// to `to`, the intrinsic forms of `fpext` and `fptrunc`, answered as those are: the canonical
/// NaN, exactly computed, for a `half` widened to `float` (`float4(h)` of a `half4`:
/// `hrd_half_read`'s widened field, b383bab9; `nan_convert`, every NaN half), and the canonical
/// `half` NaN for a `float` narrowed to `half` (`half4(f)` of a `float4`: `nan_convert`, every NaN
/// lane, word for word the scalar `fptrunc`'s). Both are measured without fast math and with it
/// (`nan_convert_fast`, the same output), so the call's flags do not decide them. The other
/// formats are not measured.
pub fn convert_answer(from: FloatFmt, to: FloatFmt) -> Option<NanAnswer> {
    match (from, to) {
        (FloatFmt::Half, FloatFmt::F32) => Some(NanAnswer::CanonicalExact),
        (FloatFmt::F32, FloatFmt::Half) => Some(NanAnswer::Canonical),
        _ => None,
    }
}

/// `r` (a result of form `f`, `float` lanes) with Metal's NaN `answer` where it is a NaN; `a`
/// is the operation's first operand, which every answer but the canonical ones reads.
pub fn select(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    answer: NanAnswer,
    test: NanTest,
    r: SValue,
    a: Option<SValue>,
) -> BResult<SValue> {
    let r_nan = is_nan(fb, f, test, r)?;
    let canonical = splat(fb, f, CANONICAL)?;
    let a = match (answer.reads_operand(), a) {
        (false, _) => return fb.select(r_nan, canonical, r),
        (true, Some(a)) => a,
        (true, None) => return Err(BuildError::Type(TypeRule::FunctionSignature)),
    };
    let a_nan = is_nan(fb, f, test, a)?;
    match answer {
        NanAnswer::Canonical | NanAnswer::CanonicalExact => fb.select(r_nan, canonical, r),
        NanAnswer::Operand => {
            let r = fb.select(r_nan, canonical, r)?;
            fb.select(a_nan, a, r)
        }
        NanAnswer::OperandSign | NanAnswer::DomainSign => {
            let u = u32_form(f);
            let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
            let bits = fb.convert(ConvOp::Bitcast, a, ut)?;
            let sign_bit = splat(fb, u, 0x8000_0000)?;
            let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_bit)?;
            let zero = splat(fb, u, 0)?;
            let sign = if answer == NanAnswer::OperandSign {
                fb.select(a_nan, sign, zero)?
            } else {
                fb.select(a_nan, zero, sign)?
            };
            let quiet = splat(fb, u, CANONICAL)?;
            let nan = fb.binary(BinOp::BitwiseOr, sign, quiet)?;
            let nan = fb.convert(ConvOp::Bitcast, nan, ft)?;
            fb.select(r_nan, nan, r)
        }
    }
}

/// `x`, 32-bit `float` lanes of form `f`, with Metal's canonical NaN `0x7FC00000` where a lane is
/// a NaN, on the bits alone: the test, the select and the answer are integer operations, and the
/// one float value is the result of an `OpBitcast`, which carries no float mode.
///
/// [`select`] with [`NanAnswer::CanonicalExact`] answers the same value through a `float`
/// `OpSelect` whose condition holds exactly where its operand is a NaN: a compiler that assumes no
/// operand is a NaN may fold that select to its operand, and the module declares no
/// `SignedZeroInfNanPreserve` for it to rely on. This form leaves it nothing to fold, so a
/// widening whose answer is read bit for bit (the vertex and stage-in fetches,
/// `lower::stage_io::fetch`) uses it.
pub fn canonical_on_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: SValue) -> BResult<SValue> {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return Err(BuildError::Type(TypeRule::FunctionSignature));
    }
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let infinity = splat(fb, u, 0x7F80_0000)?;
    let m = fb.binary(BinOp::BitwiseAnd, bits, magnitude)?;
    let is_nan = fb.binary(BinOp::UGreaterThan, m, infinity)?;
    let canonical = splat(fb, u, CANONICAL)?;
    let answered = fb.select(is_nan, canonical, bits)?;
    fb.convert(ConvOp::Bitcast, answered, ft)
}

/// Whether lanes of `x` (form `f`) are NaNs, by `test`. The bits of a 64-bit container are read
/// as two words per lane ([`is_nan_wide`]), so the test needs no 64-bit integers.
pub fn is_nan(fb: &mut FunctionBuilder<'_>, f: Form, test: NanTest, x: SValue) -> BResult<SValue> {
    match test {
        NanTest::IsNan => fb.unary(UnOp::IsNan, x),
        NanTest::Bits if f.container() == 64 => is_nan_wide(fb, f, x),
        NanTest::Bits => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
            let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
            let infinity = splat(fb, u, 0x7F80_0000)?;
            let m = fb.binary(BinOp::BitwiseAnd, bits, magnitude)?;
            fb.binary(BinOp::UGreaterThan, m, infinity)
        }
    }
}

/// Whether lanes of `x`, of 64-bit container form `f`, are NaNs, by their bits: each lane's
/// `OpBitcast` to two words puts its low word first, and a NaN's high word without its sign is
/// above infinity's `0x7FF00000`, or equal to it with a nonzero low word.
fn is_nan_wide(fb: &mut FunctionBuilder<'_>, f: Form, x: SValue) -> BResult<SValue> {
    let pair = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 2,
    };
    let pt = ty(fb, pair)?;
    let (mut lo, mut hi) = (Vec::new(), Vec::new());
    for i in 0..u32::from(f.lanes) {
        let lane = if f.lanes == 1 {
            x
        } else {
            fb.composite_extract(x, &[i])?
        };
        let words = fb.convert(ConvOp::Bitcast, lane, pt)?;
        lo.push(fb.composite_extract(words, &[0])?);
        hi.push(fb.composite_extract(words, &[1])?);
    }
    let u = u32_form(f);
    let (lo, hi) = match (lo.as_slice(), hi.as_slice()) {
        ([lo], [hi]) => (*lo, *hi),
        (lo, hi) => {
            let ut = ty(fb, u)?;
            (
                fb.composite_construct(ut, lo)?,
                fb.composite_construct(ut, hi)?,
            )
        }
    };
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let infinity = splat(fb, u, 0x7FF0_0000)?;
    let zero = splat(fb, u, 0)?;
    let m = fb.binary(BinOp::BitwiseAnd, hi, magnitude)?;
    let above = fb.binary(BinOp::UGreaterThan, m, infinity)?;
    let at = fb.binary(BinOp::IEqual, m, infinity)?;
    let payload = fb.binary(BinOp::INotEqual, lo, zero)?;
    let at = fb.binary(BinOp::LogicalAnd, at, payload)?;
    fb.binary(BinOp::LogicalOr, above, at)
}

/// Whether instruction `inst` reads its float operands' NaN bits nowhere into its result: a
/// float comparison, or an operation whose every NaN result Metal answers as the canonical NaN
/// whatever its operands' NaNs, which the lowering selects (or the device answers). Its operands
/// then need no select of their own.
pub fn drops_payload(e: &Entry, func: FuncId, inst: &Inst) -> bool {
    match &inst.kind {
        InstKind::FCmp { .. } => true,
        InstKind::FNeg { v, .. } => float_fmt_of(e, func, *v).and_then(fneg_answer).is_some(),
        InstKind::BinOp { op, a, .. } => float_fmt_of(e, func, *a)
            .and_then(|fmt| binary_answer(*op, fmt))
            .is_some(),
        InstKind::Cast { op, v } => {
            match (float_fmt_of(e, func, *v), float_fmt_of_ty(e, inst.ty)) {
                (Some(from), Some(to)) => cast_answer(*op, from, to).is_some(),
                _ => false,
            }
        }
        InstKind::Call(call) => {
            if let Callee::Air {
                op: AirIntrinsic::Convert(c),
                ..
            } = &call.callee
            {
                let float = c.src_kind == NumKind::Float && c.dst_kind == NumKind::Float;
                let from = call.args.first().and_then(|x| float_fmt_of(e, func, *x));
                return match (
                    float && c.src_fp8.is_none(),
                    from,
                    float_fmt_of_ty(e, inst.ty),
                ) {
                    (true, Some(from), Some(to)) => convert_answer(from, to).is_some(),
                    _ => false,
                };
            }
            let (op, fast) = match &call.callee {
                Callee::Air {
                    op: AirIntrinsic::Math(m),
                    ..
                } => (super::air_math(m.f).ok(), m.precision == Precision::Fast),
                Callee::Llvm { op, .. } => (super::llvm_math(op).ok(), false),
                _ => (None, false),
            };
            let Some(super::Fun::Float(op)) = op else {
                return false;
            };
            let Some(first) = call.args.first() else {
                return false;
            };
            let Some(fmt) = float_fmt_of(e, func, *first) else {
                return false;
            };
            let f = Form {
                held: Held::Float(fmt),
                lanes: 1,
            };
            // `air.fast_fmod` on `float` is Metal's fused multiply-add composition, whose NaN is
            // the fused multiply-add's: canonical.
            if fast && op == FloatFn::Fmod && fmt == FloatFmt::F32 {
                return true;
            }
            // The reciprocal a refinement divides by does not decide its NaN.
            let op = super::refine_float(op, fast, call.fmf, f, Reciprocal::Device);
            matches!(
                function_answer(op, fast, super::every_flag(call.fmf), fmt),
                Some(NanAnswer::Canonical | NanAnswer::CanonicalExact)
            )
        }
        _ => false,
    }
}

/// Whether precise `float` function `op` called with every fast-math flag is measured to answer
/// the NaN it answers without them ([`measured`]'s precise state): `sin`, `cos` (and `sincos`'s
/// two results), `tan`, `asin`, `acos`, `atan`, `asinh`, `atanh` and `atan2` at NaN operands of
/// each payload and sign, and `asin`, `acos` and `atanh` at domain errors (case
/// `precise-trigonometric-functions-at-nan-operands-under-fast-math`, every lane, on the M3 Max
/// and the M4). The precise state answers `0x7FC00000` for the first, and `asin`, `atan`, `asinh`
/// and `atanh` keep a NaN operand's sign ([`NanAnswer::OperandSign`]), where their `fast_` variants
/// answer otherwise.
fn flagged_precise(op: FloatFn) -> bool {
    use FloatFn as F;
    matches!(
        op,
        F::Sin | F::Cos | F::Tan | F::Asin | F::Acos | F::Atan | F::Asinh | F::Atanh | F::Atan2
    )
}

/// The float format of `v`'s scalar or vector type, if it is one.
fn float_fmt_of(e: &Entry, func: FuncId, v: Value) -> Option<FloatFmt> {
    float_fmt_of_ty(e, e.module.value_ty(func, v)?)
}

/// The float format of scalar or vector type `t`, if it is one.
fn float_fmt_of_ty(e: &Entry, mut t: TypeId) -> Option<FloatFmt> {
    let types = e.module.types();
    if let Some(Type::Vector { elem, .. }) = types.get(t) {
        t = *elem;
    }
    match types.get(t)? {
        Type::Float => Some(FloatFmt::F32),
        Type::Half => Some(FloatFmt::Half),
        Type::BFloat => Some(FloatFmt::BFloat),
        Type::Double => Some(FloatFmt::F64),
        _ => None,
    }
}

/// Whether each instruction's result may carry its NaN's bits to an observation, by `InstId`:
/// read by an instruction or terminator that does not [`drops_payload`] (a phi, which moves the
/// value on, included). One pass over the body's instructions; the bitmap is charged to `m` and
/// released by the caller (`insts.len()` bytes).
pub fn observed_payloads(
    e: &Entry,
    func: FuncId,
    body: &Body,
    m: &mut Meter,
) -> Result<Vec<bool>, Error> {
    let n = body.insts.len();
    m.charge(n)?;
    let mut observed = vec![false; n];
    for (_, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            if drops_payload(e, func, inst) {
                continue;
            }
            inst.kind.for_each_value(|v| {
                if let Value::Inst(i) = v {
                    if let Some(o) = observed.get_mut(i.raw() as usize) {
                        *o = true;
                    }
                }
            });
        }
    }
    Ok(observed)
}

/// The profile's NaN encoding.
pub fn encoding(p: &DeviceProfile) -> NanEncoding {
    p.fields().nan_encoding
}
