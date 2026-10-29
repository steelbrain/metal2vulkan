//! Metal's NaN results.
//!
//! Metal answers one NaN per operation class, whatever NaN its operands carry, signaling or
//! quiet, of either sign and any payload, with or without fast math (the NaN-model probes): its
//! `float` arithmetic (`+ - * /`, the fused multiply-add, `sqrt`, `fmod`) and most of its `float`
//! functions answer the canonical NaN `0x7FC00000`, sign and payload dropped
//! ([`NanAnswer::Canonical`]); `asin`, `atan`, `asinh`, and `atanh` keep a NaN operand's sign
//! ([`NanAnswer::OperandSign`]); `fast::asin` gives a domain error the operand's sign
//! ([`NanAnswer::DomainSign`]); `round`, `ldexp`, and `fast::tan` return a NaN operand itself
//! ([`NanAnswer::Operand`]); and precise `pow` returns a NaN base to the exponent one itself
//! ([`NanAnswer::BaseToExponentOne`]). Its `half` `+`, `*`, and fused multiply-add, and its narrowing of a
//! `float` to `half` (`fptrunc` and `air.convert`), answer the canonical `half` NaN `0x7E00`, held
//! widened as `0x7FC00000`; its widening of a `half` NaN to `float`, and its `half` `fneg` and
//! `fabs`, answer the canonical NaN too ([`NanAnswer::CanonicalExact`]). Its `float` `fneg` and
//! `fabs` keep a NaN operand's payload, the sign flipped or cleared, and its `half` `fract` a NaN
//! operand itself ([`NanAnswer::Operand`]). Vulkan states no NaN encoding, and the devices
//! measured answer their own:
//! an operand's bits, quieted or not, `0xFFC00000`, or `0x7FFFFFFF`, and for `half` `0xFE00`, the
//! operand's payload, or `0x7FFF`. Each device answers a NaN on exactly the lanes Metal does for
//! these classes.
//!
//! The module leaves every NaN's bits to the device, and records
//! `PrecisionNote::NativeNanEncoding` where Metal answers a NaN the device is not measured to
//! answer: every answer these tables give, but the canonical class of the arithmetic on a device
//! measured to answer Metal's canonical NaN ([`NanEncoding::MetalCanonical`],
//! [`NanAnswer::device_answers`]).
//!
//! [`is_nan`] tests a value for a NaN, by `OpIsNan` where the entry declares
//! `SignedZeroInfNanPreserve` for the width, without which a device may assume that no operand is a
//! NaN and fold the test, and on the bits elsewhere (`(bits & 0x7FFFFFFF) > 0x7F800000`), which no
//! float mode reaches: the conversions and functions whose answer at a NaN is a number test so.

use crate::air::ir::{BinOp as AirBin, CastOp, FastMath};
use crate::device::profile::NanEncoding;
use crate::device::DeviceProfile;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::FloatFn;
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, UnOp};

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
    /// A NaN first operand itself, bit for bit (its sign flipped by `fneg` and cleared by `fabs`),
    /// and the canonical NaN for a NaN of another origin: `round`, `ldexp`, `fast::tan`, `float`
    /// `fneg` and `fabs`, and `half` `fract`.
    Operand,
    /// The canonical NaN, but a NaN base to the exponent one, which is the base itself, bit for
    /// bit (`pow(-NaN, 1)` is `0xffc00000`): precise `pow` on `float` (A1 case 116e052a, every
    /// lane; `fast::pow` answers the canonical NaN there).
    BaseToExponentOne,
}

impl NanAnswer {
    /// Whether the device is measured to answer this NaN under `encoding`: on a device whose
    /// arithmetic answers Metal's canonical NaN ([`NanEncoding::MetalCanonical`]) the canonical
    /// class of the arithmetic, which the device fact covers, and not
    /// [`NanAnswer::CanonicalExact`]; nothing on another ([`NanEncoding::Native`]). The device's
    /// NaN of an answer it is not measured to give is noted `PrecisionNote::NativeNanEncoding`.
    pub fn device_answers(self, encoding: NanEncoding) -> bool {
        match encoding {
            NanEncoding::MetalCanonical => self == NanAnswer::Canonical,
            NanEncoding::Native => false,
        }
    }
}

/// How [`is_nan`] tests a value for a NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanTest {
    /// `OpIsNan`: the entry declares `SignedZeroInfNanPreserve` for the width.
    IsNan,
    /// The bits: no float mode lets a device fold an integer comparison.
    Bits,
}

impl NanTest {
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
        super::refine_float(op, true, FastMath::NONE, f32_scalar())
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
        F::FastTanh => F::Tanh,
        F::FastAsinh => F::Asinh,
        F::FastSinh => F::Sinh,
        F::FastCosh => F::Cosh,
        F::FastAcosh => F::Acosh,
        F::FastAtanh => F::Atanh,
        F::FastExp10 => F::Exp10,
        F::FastLog10 => F::Log10,
        F::FastFract => F::Fract,
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
        F::FastTan => F::Tan,
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
/// whose lowering answers it itself ([`FloatFn::metal_nans`]) are listed too. `None` where Metal
/// answers a number for a NaN operand (`sign`, `saturate`, `fast::sin`, `fast::cos`), where the
/// result is an operand the device's picks or bit moves return (`fmin`, `fmax`, `median3`,
/// `copysign`). `fabs` keeps its operand's payload with the sign cleared ([`NanAnswer::Operand`]),
/// and precise `pow` a NaN base to the exponent one ([`NanAnswer::BaseToExponentOne`]).
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
        | F::FastTanh
        | F::FastAsinh
        | F::FastAtanh
        | F::Acosh
        | F::FastAcosh
        | F::Powr
        | F::FastPowr
        | F::Floor
        | F::Ceil
        | F::Trunc
        | F::Rint
        | F::Fract
        | F::FastFract
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
        F::FastTan => A::Operand,
        F::Tan => A::Canonical,
        F::Asin if fast => A::DomainSign,
        F::FastAsin => A::DomainSign,
        F::Atan | F::Asinh | F::Atanh if fast => A::Canonical,
        F::FastAtan => A::Canonical,
        F::Asin | F::Atan | F::Asinh | F::Atanh => A::OperandSign,
        F::Round | F::Ldexp | F::Abs => A::Operand,
        F::Pow if !fast => A::BaseToExponentOne,
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
/// error with the operand's sign (`0xFE00` below `-1`); and `round`'s and `fract`'s NaN operand
/// itself, a signaling one kept (`0x7c01` stays `0x7c01`). Metal has no `fast_` `half` function,
/// so none is answered. `None` for the functions that answer a number or an operand the device's
/// picks or bit moves return for a NaN (`sign`, `saturate`, `fmin` and its family, `median3`,
/// `copysign`). `sinpi`, `cospi`, and `tanpi` answer the canonical `half`
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
        F::Round | F::Fract => A::Operand,
        F::Abs => A::CanonicalExact,
        _ => return None,
    })
}

/// The answer Metal gives for a NaN result of `fneg` on lanes of format `fmt`: the canonical
/// `half` NaN for `half`, with and without fast-math flags (`nan_narrow`, `nan_narrow_fast`,
/// `nan_half`, `nan_half_fast`), which the device computes exactly, flipping the held value's
/// sign (class D). A `float` `fneg` keeps its operand's NaN with the sign flipped (class E,
/// [`NanAnswer::Operand`]); the other formats are not measured.
pub fn fneg_answer(fmt: FloatFmt) -> Option<NanAnswer> {
    match fmt {
        FloatFmt::Half => Some(NanAnswer::CanonicalExact),
        FloatFmt::F32 => Some(NanAnswer::Operand),
        FloatFmt::BFloat | FloatFmt::F64 => None,
    }
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

/// The profile's NaN encoding.
pub fn encoding(p: &DeviceProfile) -> NanEncoding {
    p.fields().nan_encoding
}
