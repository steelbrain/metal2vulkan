//! The `float` functions whose `GLSL.std.450` instruction a device may answer otherwise than
//! Metal, as the notes a call records read them, and the one of them the translation computes
//! itself: Metal's precise `fmod` ([`fmod`]).
//!
//! Vulkan's precision table bounds `Atan`, `Asin`, and `Acos` at 4096 ulp and `Sin` and `Cos` at
//! `2^-11` absolute, about what Metal's fast-math table allows, while Metal's precise functions are
//! bounded at 4 to 6 ulp (MSL Table 8.1). Every call, on `float` and on a `half` held as `float`,
//! takes the device's instruction where one is the function (`exp`, `log`, `pow`, `sinh`, `cosh`
//! and `tanh` among them, and the fast `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and `atanh`,
//! whatever formula MSL Table 8.2 gives them), and a composition of the device's instructions only
//! where none is (`exp10`, `log10`, `sinpi`, `cospi`, `tanpi`, and the fast `exp`, `log`, `powr`
//! and `atan2`, which MSL Table 8.2 defines as compositions of `exp2`, `log2`, `atan` and a
//! division), and records the device's precision where its functions are not Metal's own
//! (`PrecisionNote::NativeTranscendentals`). No routine computes Metal's answer where the device
//! has an instruction, and no select reads a special operand as Metal does where the device's
//! instruction answers it: the selects around `pow`, `powr` and `atan2` give only the C99 and MSL
//! values `GLSL.std.450` leaves undefined, by the device's comparisons. Where the device is
//! Metal's own
//! (`FloatFunctions::Metal`: MoltenVK on an Apple GPU), its instruction is Metal's function in the
//! math mode Metal compiles the module with ([`module_mode`]), so a call in that mode is Metal's
//! own answer there ([`Fun::device_answers`]), which the notes read.
//!
//! Precise `fmod` keeps its routine: the device's form, `OpFRem` (`x - y · trunc(x / y)` at the
//! device's precision), Vulkan leaves unbounded once the quotient passes `2^24`, where Metal's is
//! exact. Where the device is Metal's own, a precise module's `OpFRem` is Metal's `fmod`, and takes
//! no routine ([`Fun::device_answers`]).

pub(crate) mod fmod;

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::FloatFn;
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{splat, ty};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, Value};
use crate::spirv::ir::{BinOp, Contraction, ConvOp, Type};

/// The quiet NaN Metal answers at precise `fmod`'s domain errors ([`E::metal_nan`]).
pub(crate) const METAL_NAN: u32 = 0x7FC0_0000;

/// A math mode: Metal's precise functions (MSL Table 8.1) or its fast ones (Table 8.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    Precise,
    Fast,
}

/// The mode Metal compiles the `float` functions of an entry in on a device whose functions are
/// Metal's (MoltenVK: `MVKDevice::getMTLCompileOptions` under its default on-demand fast math):
/// precise where the entry declares a `SignedZeroInfNanPreserve` execution mode of any width
/// (`declared`, the widths [`super::ArithPlan::modes`] holds), fast otherwise.
pub fn module_mode(declared: &std::collections::BTreeSet<u8>) -> Mode {
    if declared.is_empty() {
        Mode::Fast
    } else {
        Mode::Precise
    }
}

/// A function of Metal's math library on `float` operands, as a call of a float function takes it
/// ([`uses`]): one whose answer the device's instruction may give otherwise than Metal, beyond
/// Vulkan's precision table, or a part of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fun {
    /// Metal's `atan` in a mode: within 5 ulp in both (Tables 8.1 and 8.2).
    Atan(Mode),
    /// Metal's `asin` in a mode: within 4 ulp precise, 5 fast.
    Asin(Mode),
    /// Metal's `acos` in a mode: within 4 ulp precise, 5 fast.
    Acos(Mode),
    /// The generic part of Metal's precise `atan2(y, x)` (within 6 ulp): finite nonzero normal
    /// operands, whose special values `math::atan2_special` answers around it.
    Atan2,
    /// Metal's precise `sin` (within 4 ulp).
    Sin,
    /// Metal's precise `cos` (within 4 ulp).
    Cos,
    /// Metal's precise `tan` (within 6 ulp).
    Tan,
    /// Metal's fast `sin`, which is its precise `sin` inside `[-π, π]` (Apple's `2^-13` bound, MSL
    /// Table 8.2) and `+0` past its flush point and for an infinity or NaN.
    FastSin,
    /// Metal's fast `cos`; as [`Fun::FastSin`].
    FastCos,
    /// Metal's fast `tan`, MSL Table 8.2's `sin(x) · (1 / cos(x))` of its fast `sin` and `cos`.
    FastTan,
    /// Metal's fast `sinh`, MSL Table 8.2's `(exp(x) - exp(-x)) · 0.5`.
    FastSinh,
    /// Metal's fast `cosh`, MSL Table 8.2's `(exp(x) + exp(-x)) · 0.5`.
    FastCosh,
    /// Metal's fast `tanh`, MSL Table 8.2's `(t - 1) / (t + 1)` of `t = exp(2x)`.
    FastTanh,
    /// Metal's fast `atanh`, MSL Table 8.2's `0.5 · log((1 + x) / (1 - x))`.
    FastAtanh,
    /// Metal's `sinpi`, for a precise call (`air.sinpi`) or a fast one (`air.fast_sinpi`): Metal's
    /// fast function is its precise one, and only a device whose functions are Metal's tells them
    /// apart ([`Fun::device_answers`]).
    SinPi(Mode),
    /// Metal's `cospi`, as [`Fun::SinPi`].
    CosPi(Mode),
    /// Metal's `tanpi`, as [`Fun::SinPi`].
    TanPi(Mode),
    /// Metal's precise `sqrt`: correctly rounded (MSL Tables 8.1 and 8.3).
    Sqrt,
    /// Metal's precise `rsqrt`: correctly rounded (Tables 8.1 and 8.3).
    Rsqrt,
    /// Metal's precise `fmod`: exact (0 ulp, Tables 8.1 and 8.3), the routine of [`fmod`].
    Fmod,
    /// Metal's `exp2` (within 4 ulp, Table 8.1), which is also its fast `exp2`.
    Exp2,
    /// Metal's precise `exp`: within 4 ulp.
    Exp,
    /// Metal's precise `exp10`: within 4 ulp. No `GLSL.std.450` instruction is it.
    Exp10,
    /// Metal's `log2` (within 4 ulp), also its fast `log2`, as [`Fun::Exp2`]: the function of
    /// `air.log2`, `air.fast_log2`, and the fast functions' formulas.
    Log2,
    /// Metal's precise `log`: within 4 ulp.
    Log,
    /// Metal's precise `log10`: within 4 ulp. No `GLSL.std.450` instruction is it.
    Log10,
    /// The magnitude of Metal's precise `powr(x, y)` and `pow(x, y)` (within 16 ulp): `|x|^y`,
    /// whose NaN and zero-exponent cases `math::powr_nan` and `math::pow_special` answer around
    /// it.
    Powr,
    /// Metal's precise `sinh`: within 4 ulp.
    Sinh,
    /// Metal's precise `cosh`: within 4 ulp.
    Cosh,
    /// Metal's precise `tanh`: within 5 ulp.
    Tanh,
    /// Metal's precise `asinh`: within 4 ulp.
    Asinh,
    /// Metal's precise `acosh`: within 4 ulp.
    Acosh,
    /// Metal's precise `atanh`: within 5 ulp.
    Atanh,
    /// Metal's fast `asinh`, MSL Table 8.2's `log(x + sqrt(x x + 1))`.
    FastAsinh,
    /// Metal's fast `acosh`, `log(x + sqrt(x x - 1))`.
    FastAcosh,
    /// Metal's reciprocal `1 / y`: its fast division's.
    Recip,
    /// Metal's `half` `atan` (both modes) before its rounding, of a `half` held as `float`: its own
    /// form, not its `float` function's.
    HalfAtan,
    /// The generic part of Metal's `half` `atan2(y, x)` before its rounding (finite nonzero
    /// operands; `math::atan2_special` answers the rest).
    HalfAtan2,
    /// Metal's `half` `tan` before its rounding, its own form.
    HalfTan,
    /// Metal's `half` `sin` before its rounding, its own form.
    HalfSin,
    /// Metal's `half` `cos` before its rounding, as [`Fun::HalfSin`].
    HalfCos,
    /// Metal's `half` `asinh` before its rounding, its own form.
    HalfAsinh,
    /// Metal's `half` `exp`, `log`, `powr` (and `pow`'s magnitude), `sinh`, `cosh` and `tanh`
    /// before their rounding, which the device's `float` instruction on the widened operand
    /// answers at its own precision: Metal's are its fast `float` functions rounded (`tanh` its own
    /// formula), which no instruction SPIRV-Cross emits is in every module.
    HalfExp,
    /// See [`Fun::HalfExp`].
    HalfLog,
    /// See [`Fun::HalfExp`].
    HalfPowr,
    /// See [`Fun::HalfExp`].
    HalfSinh,
    /// See [`Fun::HalfExp`].
    HalfCosh,
    /// See [`Fun::HalfExp`].
    HalfTanh,
}

impl Fun {
    /// Whether a device whose `float` functions are Metal's answers `self` with its own
    /// instruction as Metal's function in an entry whose functions Metal compiles in mode
    /// `module`: the instruction is Metal's function in that mode, and `atan2` is always the
    /// precise one (SPIRV-Cross emits `precise::atan2` whatever the flags). The precise-only
    /// functions (`sin`, `cos`, `tan`, `sqrt`, `rsqrt`, and `fmod`, whose fast calls are their own
    /// functions) are Metal's only in a precise module. Metal's fast `sin`, `cos`, `tan`, `sinh`,
    /// `cosh` and `atanh` are the device's `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh` and `Atanh` in a
    /// fast-math module only (in a precise one those are Metal's precise functions, which differ:
    /// `fast::sin` is `+0` past its flush point), and its fast `tanh` in none (SPIRV-Cross emits
    /// `precise::tanh`, below). Metal's `sinpi`, `cospi`,
    /// and `tanpi` are none of the instructions SPIRV-Cross emits (it emits no `sinpi`), so the
    /// device's `Sin` and `Cos` of `float(π) r` answer a precise call otherwise than Metal (849,
    /// 922 and 637 of case 087758cb's 1,024 lanes matched); a fast call is that composition, in
    /// either mode, within Apple's fast bound (MSL Table 8.2) and off Metal's own form
    /// ([`Fun::device_near_equivalent`]).
    ///
    /// `exp2` and `log2` are Metal's in either mode (Metal's fast ones are its precise ones), and
    /// so is `tanh` (SPIRV-Cross emits `precise::tanh` whatever the flags); `powr` and the other
    /// hyperbolic functions, `exp` and `log` only in a precise module. No `GLSL.std.450`
    /// instruction reaches Metal's `exp10` and `log10`, but Metal's own compositions of its `exp2`
    /// and `log2` do, in either mode (`math::exp10`, `math::log10`). Its reciprocal is the
    /// device's `1 / y` in a fast-math module, where Metal
    /// compiles the division with its fast reciprocal, and not in a precise one, whose quotient is
    /// correctly rounded. The device's instruction on a `half` held as `float` is its `float`
    /// function, never Metal's `half` form.
    pub fn device_answers(self, module: Mode) -> bool {
        match self {
            Fun::FastSin
            | Fun::FastCos
            | Fun::FastTan
            | Fun::FastSinh
            | Fun::FastCosh
            | Fun::FastAtanh => module == Mode::Fast,
            Fun::FastTanh => false,
            Fun::SinPi(m) | Fun::CosPi(m) | Fun::TanPi(m) => m == Mode::Fast,
            Fun::Atan(m) | Fun::Asin(m) | Fun::Acos(m) => m == module,
            Fun::Sin
            | Fun::Cos
            | Fun::Tan
            | Fun::Sqrt
            | Fun::Rsqrt
            | Fun::Fmod
            | Fun::Powr
            | Fun::Exp
            | Fun::Log
            | Fun::Sinh
            | Fun::Cosh
            | Fun::Asinh
            | Fun::Acosh
            | Fun::Atanh => module == Mode::Precise,
            // Metal's fast `asinh` and `acosh` are the device's `Asinh` and `Acosh` in a fast-math
            // module.
            Fun::Recip | Fun::FastAsinh | Fun::FastAcosh => module == Mode::Fast,
            Fun::HalfAtan
            | Fun::HalfAtan2
            | Fun::HalfTan
            | Fun::HalfSin
            | Fun::HalfCos
            | Fun::HalfAsinh
            | Fun::HalfExp
            | Fun::HalfLog
            | Fun::HalfPowr
            | Fun::HalfSinh
            | Fun::HalfCosh
            | Fun::HalfTanh => false,
            Fun::Atan2 | Fun::Exp2 | Fun::Log2 | Fun::Tanh | Fun::Exp10 | Fun::Log10 => true,
        }
    }

    /// Whether the device answers the function, where it does ([`Fun::device_answers`]), by a
    /// near-equivalent of Metal's function rather than by Metal's function itself: Metal's fast
    /// `sinpi`, `cospi`, and `tanpi`, which the device answers by its `Sin` and `Cos` of
    /// `float(π) r`, within Apple's fast bound but off Metal's own forms (case f6524207 differs at
    /// 175, 102, and 264 of its 1,024 lanes through MoltenVK). Every other function the device
    /// answers is Metal's own, word for word (`DriverDeviation::MetalFloatFunctions`).
    pub fn device_near_equivalent(self) -> bool {
        matches!(
            self,
            Fun::SinPi(Mode::Fast) | Fun::CosPi(Mode::Fast) | Fun::TanPi(Mode::Fast)
        )
    }

    /// Whether the call takes the device's instruction for the function: every function but
    /// precise `fmod`, whose routine computes Metal's exact remainder ([`fmod`]). Its
    /// precision is Vulkan's (`PrecisionNote::NativeTranscendentals`).
    pub fn native(self) -> bool {
        self != Fun::Fmod
    }

    /// Whether Metal's answer to the function is the one right value at every operand, so that
    /// the routine's answer, and the device's where it is Metal's own ([`Fun::device_answers`]), is
    /// Metal's on every device in either math mode: precise `fmod`, which Metal documents exact
    /// (MSL Table 8.1 for `float`, Table 8.3 for `half`), and which its routine computes.
    pub fn correctly_rounded(self) -> bool {
        self == Fun::Fmod
    }

    /// The function's operands.
    pub fn arity(self) -> usize {
        match self {
            Fun::Atan2 | Fun::Fmod | Fun::Powr | Fun::HalfAtan2 | Fun::HalfPowr => 2,
            _ => 1,
        }
    }
}

/// The functions a call of float function `op` on form `f` takes: its own and those of the
/// functions it composes. `float` lanes take them, and so do `half` lanes, which are held as
/// `float` and computed by a `float` function before their rounding: Metal's `half` functions are
/// the same in both math modes (the half probe, cases dc0fee8f and 3ef7c1c5; every `half` of our
/// probe), so `refine_float` refines only `float` calls, and the `float` function of a `half` one
/// is `math::half_form`'s (`exp10` and `log10` their fast formulas, which Metal's `half` ones
/// are); its `atan`, `atan2`, `tan`, `sin`, `cos`, `asinh`, `exp`, `log`, `pow`, `powr`, `sinh`,
/// `cosh` and `tanh` are Metal's own `half` forms, which the device's `float` instruction answers
/// at its precision ([`Fun::HalfAtan`], [`Fun::HalfAtan2`], [`Fun::HalfTan`], [`Fun::HalfSin`],
/// [`Fun::HalfCos`], [`Fun::HalfAsinh`], [`Fun::HalfExp`]). Another format takes none.
pub fn uses(op: FloatFn, f: Form) -> &'static [Fun] {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return &[];
    }
    if f.float_fmt() == Some(FloatFmt::Half) {
        match op {
            FloatFn::Exp => return &[Fun::HalfExp],
            FloatFn::Log => return &[Fun::HalfLog],
            FloatFn::Pow | FloatFn::Powr => return &[Fun::HalfPowr],
            FloatFn::Sinh => return &[Fun::HalfSinh],
            FloatFn::Cosh => return &[Fun::HalfCosh],
            FloatFn::Tanh => return &[Fun::HalfTanh],
            FloatFn::Atan => return &[Fun::HalfAtan],
            FloatFn::Atan2 => return &[Fun::HalfAtan2],
            FloatFn::Tan => return &[Fun::HalfTan],
            FloatFn::Sin => return &[Fun::HalfSin],
            FloatFn::Cos => return &[Fun::HalfCos],
            FloatFn::Asinh => return &[Fun::HalfAsinh],
            _ => {}
        }
    }
    match crate::lower::arith::math::half_form(op, f) {
        FloatFn::Atan => &[Fun::Atan(Mode::Precise)],
        FloatFn::FastAtan => &[Fun::Atan(Mode::Fast)],
        FloatFn::Asin => &[Fun::Asin(Mode::Precise)],
        FloatFn::FastAsin => &[Fun::Asin(Mode::Fast)],
        FloatFn::Acos => &[Fun::Acos(Mode::Precise)],
        FloatFn::FastAcos => &[Fun::Acos(Mode::Fast)],
        FloatFn::Atan2 => &[Fun::Atan2],
        // `atan(y · rcp(x))`, Metal's fast `atan` of its fast quotient.
        FloatFn::FastAtan2 { .. } => &[Fun::Atan(Mode::Fast), Fun::Recip],
        FloatFn::Sin => &[Fun::Sin],
        FloatFn::Cos => &[Fun::Cos],
        FloatFn::Tan => &[Fun::Tan],
        FloatFn::FastSin => &[Fun::FastSin],
        FloatFn::FastCos => &[Fun::FastCos],
        FloatFn::FastTan => &[Fun::FastTan],
        FloatFn::SinPi => &[Fun::SinPi(Mode::Precise)],
        FloatFn::CosPi => &[Fun::CosPi(Mode::Precise)],
        FloatFn::TanPi => &[Fun::TanPi(Mode::Precise)],
        FloatFn::FastSinPi => &[Fun::SinPi(Mode::Fast)],
        FloatFn::FastCosPi => &[Fun::CosPi(Mode::Fast)],
        FloatFn::FastTanPi => &[Fun::TanPi(Mode::Fast)],
        FloatFn::Sqrt => &[Fun::Sqrt],
        FloatFn::Rsqrt => &[Fun::Rsqrt],
        FloatFn::Fmod => &[Fun::Fmod],
        // The precise functions, the fast ones the device's instructions, and the fast ones'
        // compositions of Metal's `exp2` and `log2` (MSL Table 8.2).
        FloatFn::Exp2 => &[Fun::Exp2],
        FloatFn::Exp => &[Fun::Exp],
        FloatFn::Exp10 => &[Fun::Exp10],
        FloatFn::Log2 | FloatFn::FastLog2 => &[Fun::Log2],
        FloatFn::Log => &[Fun::Log],
        FloatFn::Log10 => &[Fun::Log10],
        FloatFn::Pow | FloatFn::Powr => &[Fun::Powr],
        FloatFn::Sinh => &[Fun::Sinh],
        FloatFn::Cosh => &[Fun::Cosh],
        FloatFn::Tanh => &[Fun::Tanh],
        FloatFn::Asinh => &[Fun::Asinh],
        FloatFn::Acosh => &[Fun::Acosh],
        FloatFn::Atanh => &[Fun::Atanh],
        FloatFn::FastSinh => &[Fun::FastSinh],
        FloatFn::FastCosh => &[Fun::FastCosh],
        FloatFn::FastTanh => &[Fun::FastTanh],
        FloatFn::FastAtanh => &[Fun::FastAtanh],
        FloatFn::FastExp | FloatFn::FastExp10 => &[Fun::Exp2],
        FloatFn::FastLog | FloatFn::FastLog10 => &[Fun::Log2],
        FloatFn::FastAsinh => &[Fun::FastAsinh],
        FloatFn::FastAcosh => &[Fun::FastAcosh],
        FloatFn::FastPow | FloatFn::FastPowr => &[Fun::Exp2, Fun::Log2],
        _ => &[],
    }
}

/// The census name of precise `fmod`'s routine: the float functions at the device's precision
/// class, whose answer it replaces.
pub const FMOD_NAME: &str = "arith.float_fn.device_transcendental";

/// An upper estimate of precise `fmod`'s routine's SPIR-V words, with its constants and types.
pub const FMOD_WORDS: u32 = 600;

/// An upper estimate of the SPIR-V words per use of a function of `arity` operands on `lanes`
/// lanes: each lane's operand extractions and call of a routine, and the result's construction,
/// which bounds the device's instruction for it too.
pub fn use_words(lanes: u8, arity: usize) -> u32 {
    let arity = u32::try_from(arity).unwrap_or(u32::MAX);
    let lanes = u32::from(lanes);
    lanes
        .saturating_mul(6u32.saturating_add(arity.saturating_mul(6)))
        .saturating_add(8)
}

/// What an entry's calls take, as emission reads them: the routine of precise `fmod` where the
/// module defines one. Every other function is the device's instruction, whichever mode Metal
/// compiles the entry in.
#[derive(Clone, Copy, Debug)]
pub struct Lib {
    fmod: Option<FuncId>,
}

impl Lib {
    /// The library of an entry whose module defines precise `fmod`'s routine as `fmod` where it
    /// takes one.
    pub fn new(fmod: Option<FuncId>) -> Lib {
        Lib { fmod }
    }

    /// A library that leaves every function to its instruction (a unit test's view of the
    /// sequences around one).
    #[cfg(test)]
    pub fn instructions() -> Lib {
        Lib::new(None)
    }

    /// Metal's precise `fmod` of `x` and `y` (`float` lanes of form `f`) by its routine, one call
    /// per lane; `None` where the module takes the device's `OpFRem` ([`Fun::device_answers`]).
    pub fn fmod(
        &self,
        fb: &mut FunctionBuilder<'_>,
        f: Form,
        x: Value,
        y: Value,
    ) -> BResult<Option<Value>> {
        match self.fmod {
            Some(func) => per_lane(fb, func, f, &[x, y]).map(Some),
            None => Ok(None),
        }
    }
}

/// `func` (a function of `float` values returning one) of `args`, lanes of form `f`: one call
/// per lane.
fn per_lane(fb: &mut FunctionBuilder<'_>, func: FuncId, f: Form, args: &[Value]) -> BResult<Value> {
    let call = |fb: &mut FunctionBuilder<'_>, xs: &[Value]| -> BResult<Value> {
        fb.call(func, xs)?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    };
    if f.lanes == 1 {
        return call(fb, args);
    }
    let mut lanes = Vec::with_capacity(usize::from(f.lanes));
    for l in 0..u32::from(f.lanes) {
        let xs = args
            .iter()
            .map(|a| fb.composite_extract(*a, &[l]))
            .collect::<BResult<Vec<_>>>()?;
        lanes.push(call(fb, &xs)?);
    }
    let t = ty(fb, f)?;
    fb.composite_construct(t, &lanes)
}

/// Define precise `fmod`'s routine: a function of two `float` values returning Metal's exact
/// remainder ([`fmod::fmod`]).
pub fn define_fmod(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = mb.ty(Type::Float { width: 32 })?;
    let mut fb = mb.routine(t)?;
    let (x, y) = (fb.param(t)?, fb.param(t)?);
    let mut e = E { fb: &mut fb };
    let v = fmod::fmod(&mut e, x, y)?;
    fb.ret(Some(v))?;
    fb.finish()
}

/// `v` as `hi + lo`, `hi` the `float` nearest `v`, `lo` the `float` nearest the rest.
pub(crate) fn split(v: f64) -> (f32, f32) {
    let hi = v as f32;
    (hi, (v - f64::from(hi)) as f32)
}

/// Metal's precise `log10` next to 1: for `x` strictly between the bits `low` and `end`,
/// `RN(f fma(p(f), f, lead))` of `f = x - 1` (exact there), `p` by Horner's rule with fused steps,
/// `f32` coefficients by their bits, lowest first: Metal's word at every `float` of its range
/// (`math::log10`).
pub(crate) struct Near {
    pub low: u32,
    pub end: u32,
    pub lead: u32,
    pub p: &'static [u32],
}

/// `log10`'s: `(495/512, 543/512)`, `lead` `log10(e)` rounded down.
pub(crate) const LOG10_NEAR: Near = Near {
    low: 0x3F77_8000,
    end: 0x3F87_C000,
    lead: 0x3EDE_5BD8,
    p: &[0xBE5E_5CE8, 0x3E14_5662, 0xBDD5_5B94],
};

/// `RN(log10(2))`, by its bits: Metal's `log10` away from 1 is its `log2` times it.
pub(crate) const LOG10_2_BITS: u32 = 0x3E9A_209B;

/// The `float` scalar.
const F32: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

/// A routine body's emission on `float` scalars: constants by their bits, and the correctly
/// rounded arithmetic, each operation `NoContraction`.
pub(crate) struct E<'f, 'm> {
    pub fb: &'f mut FunctionBuilder<'m>,
}

impl E<'_, '_> {
    /// The `float` constant `v`.
    pub fn c(&mut self, v: f32) -> BResult<Value> {
        splat(self.fb, F32, u64::from(v.to_bits()))
    }

    /// Metal's NaN at a domain error of precise `fmod` (an infinite dividend or a zero or denormal
    /// divisor, Metal's rows of case 087758cb, every such lane): the quiet NaN `0x7fc00000` by
    /// its bits, selected as a constant, so no device's NaN encoding reaches it; a NaN operand's
    /// answer stays the device's arithmetic's (`PrecisionNote::DeviceNanEncoding`).
    pub fn metal_nan(&mut self) -> BResult<Value> {
        self.c(f32::from_bits(METAL_NAN))
    }

    /// The `u32` constant `v`.
    pub fn u(&mut self, v: u32) -> BResult<Value> {
        splat(self.fb, u32_form(F32), u64::from(v))
    }

    fn op(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary_with(op, a, b, Contraction::Forbidden)
    }

    pub fn add(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FAdd, a, b)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FSub, a, b)
    }

    pub fn mul(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::FMul, a, b)
    }

    pub fn sel(&mut self, c: Value, t: Value, f: Value) -> BResult<Value> {
        self.fb.select(c, t, f)
    }

    /// The bits of `float` `a`.
    pub fn bits(&mut self, a: Value) -> BResult<Value> {
        let ut = ty(self.fb, u32_form(F32))?;
        self.fb.convert(ConvOp::Bitcast, a, ut)
    }

    /// The `float` of bits `b`.
    pub fn float(&mut self, b: Value) -> BResult<Value> {
        let ft = ty(self.fb, F32)?;
        self.fb.convert(ConvOp::Bitcast, b, ft)
    }

    pub fn iop(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.fb.binary(op, a, b)
    }

    /// The `float` of the `u32` `a`, exact where `a` has at most 24 significant bits.
    pub fn ufloat(&mut self, a: Value) -> BResult<Value> {
        let ft = ty(self.fb, F32)?;
        self.fb.convert(ConvOp::ConvertUToF, a, ft)
    }
}
