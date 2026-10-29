//! The emission core of the math intrinsics: integer functions (`air` integer math,
//! `llvm.{s,u}{min,max}`, `llvm.fsh{l,r}`, `llvm.bswap`, `llvm.ctpop`, saturating arithmetic,
//! `llvm.abs`, `llvm.ct{l,t}z`), float functions (`air` float math, `llvm.{fma,fmuladd,fabs,
//! minnum,maxnum}`), and `air.convert`.
//!
//! Every float function takes the device's instruction where `GLSL.std.450` or SPIR-V has one:
//! `FAbs`, `FSign`, `Fract`, `NMin`/`NMax`/`NClamp`, `FMix`, `OpDot`, `Sqrt`, `Exp`, `Log`, `Pow`,
//! the trigonometric and hyperbolic families. Where the instruction leaves part of Metal's
//! *definition* open, the lowering states that part and nothing more: `round` rounds halfway cases
//! away from zero from `Trunc` (`GLSL.std.450 Round` leaves the direction to the device), `fract`
//! is bounded below one (Metal's `fmin(x - floor(x), 0x1.fffffep-1)`), `median3` is
//! `min(max(c, min(a, b)), max(a, b))` of `NMin` and `NMax`, `pow` and `atan2` answer C99 Annex
//! F's special cases around `Pow` and `Atan2` (which `GLSL.std.450` leaves undefined there), and
//! float-to-integer `air.convert` saturates (NaN to 0, out-of-range to the bound, measured on
//! Apple GPUs) with the clamp applied before the conversion so no conversion is out of range.
//! Exact integer sequences stand where `GLSL.std.450` differs from Metal: `clz`/`ctz` from
//! `FindUMsb`/`FindILsb` with the zero case, `rotate` and the funnel shifts with an in-range second
//! shift (`(x >> 1) >> (N - 1 - s)`). A function no instruction is takes a composition of the
//! device's instructions: `exp10` and `log10` of `Exp2` and `Log2`, `sinpi`, `cospi` and `tanpi`
//! by an exact period reduction around `Sin` and `Cos`, `ldexp` on the bits, and the fast-math
//! formulas (MSL Table 8.2) of `fast::exp`, `fast::log`, `fast::powr` and `fast::atan2`; every
//! other `fast_` function is the device's instruction for it. The answers at special operands (a
//! zero's sign, a NaN's bits, an `f32` denormal) are the device's, recorded as precision notes at
//! selection, and so is every function whose Vulkan precision is looser than Metal's.

use crate::air::intrinsics::PackFormat;
use crate::lower::arith::exact_division::{div_exact_lanes, per_lane};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::libm::{self, Lib};
use crate::lower::arith::nan::{self, NanTest};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, mask, ones, sext, splat, ty, CastKind};
use crate::lower::arith::srgb;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp, UnOp};

/// Whether an integer function reads its operands as signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sign {
    Signed,
    Unsigned,
}

/// An integer function on held integers of one form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntFn {
    /// Signed absolute value (the unsigned one is the operand itself).
    Abs,
    AbsDiff(Sign),
    Min(Sign),
    Max(Sign),
    Clamp(Sign),
    Min3(Sign),
    Max3(Sign),
    MulHi(Sign),
    MadHi(Sign),
    /// `a * b + c` saturated from the exact (double-width) result.
    MadSat(Sign),
    AddSat(Sign),
    SubSat(Sign),
    HAdd(Sign),
    RHAdd(Sign),
    /// `OpBitFieldExtract` on 32-bit containers, shifts on 64-bit ones (Vulkan's bit-field
    /// instructions are 32-bit only).
    ExtractBits(Sign),
    /// As [`IntFn::ExtractBits`].
    InsertBits,
    Popcount,
    ReverseBits,
    Bswap,
    Rotate,
    Clz,
    Ctz,
    Fshl,
    Fshr,
}

impl IntFn {
    /// The operand count.
    pub fn arity(self) -> usize {
        match self {
            IntFn::Abs | IntFn::Popcount | IntFn::ReverseBits | IntFn::Bswap => 1,
            IntFn::Clz | IntFn::Ctz => 1,
            IntFn::AbsDiff(_)
            | IntFn::Min(_)
            | IntFn::Max(_)
            | IntFn::MulHi(_)
            | IntFn::AddSat(_)
            | IntFn::SubSat(_)
            | IntFn::HAdd(_)
            | IntFn::RHAdd(_)
            | IntFn::Rotate => 2,
            IntFn::Clamp(_)
            | IntFn::Min3(_)
            | IntFn::Max3(_)
            | IntFn::MadHi(_)
            | IntFn::MadSat(_)
            | IntFn::ExtractBits(_)
            | IntFn::Fshl
            | IntFn::Fshr => 3,
            IntFn::InsertBits => 4,
        }
    }

    /// An upper estimate of the SPIR-V words per use on `bits`-wide integers.
    pub fn words(self, bits: u8) -> u32 {
        match self {
            IntFn::Bswap => 16 + 32 * u32::from(bits / 8),
            IntFn::Clz | IntFn::Ctz | IntFn::Popcount | IntFn::ReverseBits => 128,
            _ => 96,
        }
    }
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

fn glsl(fb: &mut FunctionBuilder<'_>, op: GlslOp, args: &[Value]) -> BResult<Value> {
    fb.glsl(op, args)
}

fn arg(args: &[Value], i: usize) -> BResult<Value> {
    args.get(i)
        .copied()
        .ok_or(crate::spirv::error::BuildError::Operand {
            op: crate::spv::Op::ExtInst,
            rule: crate::spirv::error::OperandRule::Arity,
        })
}

/// The unsigned 32-bit form with `f`'s lanes.
fn u32_form(f: Form) -> Form {
    f.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

/// `ldexp(x, k)` (`x` of form `f`, `k` a held `i32` of its lanes) for every exponent, as Metal
/// answers it but for denormals, which are the device's. GLSL's `Ldexp` leaves an exponent past
/// `+128` and a product past the format's range undefined, and may flush any result of an
/// exponent below `-126`, so a `float` `ldexp(2^100, -150)` (`2^-50`) could read zero. A `float`
/// (and a `bfloat`, held in its `f32`) is computed on the container's bits and the device's
/// products ([`ldexp_bits`]; a `bfloat`'s denormals are the device's as a `float`'s); a `half`, held exactly
/// in an `f32`, is
/// scaled by `2^k` with `k` clamped to `[-64, 64]`, which keeps every product an exact normal
/// `f32` and saturates it past `half`'s range, and the caller rounds it to `half` once. A
/// `double` keeps the device's `Ldexp`: Metal has no `double` to answer it.
fn ldexp(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, k: Value) -> BResult<Value> {
    match f.float_fmt() {
        Some(FloatFmt::F32 | FloatFmt::BFloat) => ldexp_bits(fb, f, x, k),
        Some(FloatFmt::Half) => {
            let i = u32_form(f);
            let (lo, hi) = (splat(fb, i, (-64i32) as u32 as u64)?, splat(fb, i, 64)?);
            let k = glsl(fb, GlslOp::SClamp, &[k, lo, hi])?;
            let bias = splat(fb, i, 127)?;
            let e = bin(fb, BinOp::IAdd, k, bias)?;
            let at = splat(fb, i, 23)?;
            let e = bin(fb, BinOp::ShiftLeftLogical, e, at)?;
            let ft = ty(fb, f)?;
            let scale = fb.convert(ConvOp::Bitcast, e, ft)?;
            bin(fb, BinOp::FMul, x, scale)
        }
        _ => glsl(fb, GlslOp::Ldexp, &[x, k]),
    }
}

/// [`ldexp`] of an `f32` container for every exponent, its denormals the device's: an infinity
/// or a NaN is `x`, bit for bit; a zero or a denormal `x` is first scaled by `2^64` by the device's
/// product, so the device's own denormal mode reads it (preserved, or flushed to the zero of its
/// sign, which stays that zero); otherwise the exponent field `e + k` (with `k` clamped to
/// `[-512, 512]`, which saturates every answer) is an infinity of `x`'s sign from 255 up, replaces
/// `x`'s exponent field from 1 to 254, and at 0 and below is the device's product of `x` scaled
/// to the field `max(e + k + 126, 1)` and `2^-126`, which rounds a result below `2^-126` once to a
/// denormal or flushes it as the device does. Metal flushes both a denormal operand and a denormal
/// result: the device's multiply reads them so where it takes `DriverDeviation::MetalFloatDenormals`
/// (`denormal::coverage`); elsewhere they are noted `PrecisionNote::DeviceDenormals`.
fn ldexp_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, k: Value) -> BResult<Value> {
    let i = u32_form(f);
    let it = ty(fb, i)?;
    let ft = ty(fb, f)?;
    let at = splat(fb, i, 23)?;
    let field = splat(fb, i, 0xFF)?;
    let zero = splat(fb, i, 0)?;
    let exponent = |fb: &mut FunctionBuilder<'_>, bits: Value| -> BResult<Value> {
        let e = bin(fb, BinOp::ShiftRightLogical, bits, at)?;
        bin(fb, BinOp::BitwiseAnd, e, field)
    };
    let given = fb.convert(ConvOp::Bitcast, x, it)?;
    let e0 = exponent(fb, given)?;
    let (lo, hi) = (splat(fb, i, (-512i32) as u32 as u64)?, splat(fb, i, 512)?);
    let k = glsl(fb, GlslOp::SClamp, &[k, lo, hi])?;
    // A zero or denormal operand, normalized by the device's product (or flushed by it).
    let tiny = bin(fb, BinOp::IEqual, e0, zero)?;
    let up = fsplat(fb, f, 2f64.powi(64))?;
    let raised = bin(fb, BinOp::FMul, x, up)?;
    let source = fb.select(tiny, raised, x)?;
    let sixty_four = splat(fb, i, 64)?;
    let lowered = bin(fb, BinOp::ISub, k, sixty_four)?;
    let k = fb.select(tiny, lowered, k)?;
    let bits = fb.convert(ConvOp::Bitcast, source, it)?;
    let sign = splat(fb, i, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, bits, sign)?;
    let e = exponent(fb, bits)?;
    let ne = bin(fb, BinOp::IAdd, e, k)?;
    let keep = splat(fb, i, 0x807F_FFFF)?;
    let kept = bin(fb, BinOp::BitwiseAnd, bits, keep)?;
    let moved = bin(fb, BinOp::ShiftLeftLogical, ne, at)?;
    let scaled = bin(fb, BinOp::BitwiseOr, kept, moved)?;
    // A result below `2^-126`: the device's product rounds it once (or flushes it).
    let bias = splat(fb, i, 126)?;
    let low = bin(fb, BinOp::IAdd, ne, bias)?;
    let one = splat(fb, i, 1)?;
    let low = glsl(fb, GlslOp::SMax, &[low, one])?;
    let low = bin(fb, BinOp::ShiftLeftLogical, low, at)?;
    let low = bin(fb, BinOp::BitwiseOr, kept, low)?;
    let low = fb.convert(ConvOp::Bitcast, low, ft)?;
    let down = fsplat(fb, f, 2f64.powi(-126))?;
    let low = bin(fb, BinOp::FMul, low, down)?;
    let low = fb.convert(ConvOp::Bitcast, low, it)?;
    let under = bin(fb, BinOp::SLessThanEqual, ne, zero)?;
    let r = fb.select(under, low, scaled)?;
    let inf = splat(fb, i, 0x7F80_0000)?;
    let inf = bin(fb, BinOp::BitwiseOr, sign, inf)?;
    let over = bin(fb, BinOp::SGreaterThanEqual, ne, field)?;
    let r = fb.select(over, inf, r)?;
    let nothing = bin(fb, BinOp::IEqual, e, zero)?;
    let r = fb.select(nothing, bits, r)?;
    let special = bin(fb, BinOp::IEqual, e0, field)?;
    let r = fb.select(special, given, r)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The low and high 32-bit halves of a 64-bit container.
fn halves(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<(Value, Value)> {
    let t32 = ty(fb, u32_form(f))?;
    let lo = fb.convert(ConvOp::UConvert, x, t32)?;
    let k = splat(fb, f, 32)?;
    let hi = bin(fb, BinOp::ShiftRightLogical, x, k)?;
    let hi = fb.convert(ConvOp::UConvert, hi, t32)?;
    Ok((lo, hi))
}

/// A 32-bit lane result widened to `f`'s container.
fn widen(fb: &mut FunctionBuilder<'_>, f: Form, r: Value) -> BResult<Value> {
    if f.container() == 32 {
        return Ok(r);
    }
    let t = ty(fb, f)?;
    fb.convert(ConvOp::UConvert, r, t)
}

/// `(x << s) | ((y >> 1) >> (N - 1 - s))` for `s < N` (the funnel/rotate core without an
/// out-of-range shift).
fn funnel_left(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    s: Value,
) -> BResult<Value> {
    let n1 = splat(fb, f, u64::from(f.bits()) - 1)?;
    let one = splat(fb, f, 1)?;
    let hi = bin(fb, BinOp::ShiftLeftLogical, x, s)?;
    let y1 = bin(fb, BinOp::ShiftRightLogical, y, one)?;
    let rest = bin(fb, BinOp::ISub, n1, s)?;
    let lo = bin(fb, BinOp::ShiftRightLogical, y1, rest)?;
    let r = bin(fb, BinOp::BitwiseOr, hi, lo)?;
    mask(fb, f, r)
}

/// `s mod N`.
fn amount(fb: &mut FunctionBuilder<'_>, f: Form, s: Value) -> BResult<Value> {
    let n = splat(fb, f, u64::from(f.bits()))?;
    bin(fb, BinOp::UMod, s, n)
}

/// Emit integer function `op` on `args` (held integers of form `f`).
pub fn int_fn(fb: &mut FunctionBuilder<'_>, op: IntFn, f: Form, args: &[Value]) -> BResult<Value> {
    let a = arg(args, 0)?;
    let signed = |s: Sign| s == Sign::Signed;
    // The operand as its signed or unsigned value in the container.
    let ext = |fb: &mut FunctionBuilder<'_>, s: Sign, x: Value| -> BResult<Value> {
        if signed(s) {
            sext(fb, f, x)
        } else {
            Ok(x)
        }
    };
    let minmax = |fb: &mut FunctionBuilder<'_>, s: Sign, max: bool, x, y| -> BResult<Value> {
        let op = match (signed(s), max) {
            (true, true) => GlslOp::SMax,
            (true, false) => GlslOp::SMin,
            (false, true) => GlslOp::UMax,
            (false, false) => GlslOp::UMin,
        };
        glsl(fb, op, &[x, y])
    };
    let n = u64::from(f.bits());
    match op {
        IntFn::Abs => {
            let x = sext(fb, f, a)?;
            let r = glsl(fb, GlslOp::SAbs, &[x])?;
            mask(fb, f, r)
        }
        IntFn::AbsDiff(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let gt = if signed(s) {
                BinOp::SGreaterThan
            } else {
                BinOp::UGreaterThan
            };
            let c = bin(fb, gt, x, y)?;
            let d1 = bin(fb, BinOp::ISub, x, y)?;
            let d2 = bin(fb, BinOp::ISub, y, x)?;
            let r = fb.select(c, d1, d2)?;
            mask(fb, f, r)
        }
        IntFn::Min(s) | IntFn::Max(s) | IntFn::Min3(s) | IntFn::Max3(s) => {
            let max = matches!(op, IntFn::Max(_) | IntFn::Max3(_));
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let mut r = minmax(fb, s, max, x, y)?;
            if matches!(op, IntFn::Min3(_) | IntFn::Max3(_)) {
                let z = ext(fb, s, arg(args, 2)?)?;
                r = minmax(fb, s, max, r, z)?;
            }
            mask(fb, f, r)
        }
        IntFn::Clamp(s) => {
            let x = ext(fb, s, a)?;
            let (lo, hi) = (ext(fb, s, arg(args, 1)?)?, ext(fb, s, arg(args, 2)?)?);
            let r = minmax(fb, s, true, x, lo)?;
            let r = minmax(fb, s, false, r, hi)?;
            mask(fb, f, r)
        }
        IntFn::MulHi(s) | IntFn::MadHi(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let wide = if signed(s) {
                BinOp::SMulExtended
            } else {
                BinOp::UMulExtended
            };
            let p = bin(fb, wide, x, y)?;
            let hi = fb.composite_extract(p, &[1])?;
            let mut r = if f.legalized() {
                // The product's bits [N, 2N): (lo >> N) | (hi << (C - N)).
                let lo = fb.composite_extract(p, &[0])?;
                let sh = splat(fb, f, n)?;
                let lo = bin(fb, BinOp::ShiftRightLogical, lo, sh)?;
                let up = splat(fb, f, u64::from(f.container()) - n)?;
                let hi = bin(fb, BinOp::ShiftLeftLogical, hi, up)?;
                bin(fb, BinOp::BitwiseOr, lo, hi)?
            } else {
                hi
            };
            if matches!(op, IntFn::MadHi(_)) {
                r = bin(fb, BinOp::IAdd, r, arg(args, 2)?)?;
            }
            mask(fb, f, r)
        }
        IntFn::AddSat(Sign::Unsigned) => {
            let b = arg(args, 1)?;
            let s = bin(fb, BinOp::IAdd, a, b)?;
            if f.legalized() {
                // No wrap in the container: clamp to the width's maximum.
                let m = splat(fb, f, ones(f.bits()))?;
                glsl(fb, GlslOp::UMin, &[s, m])
            } else {
                let wrapped = bin(fb, BinOp::ULessThan, s, a)?;
                let m = splat(fb, f, ones(f.bits()))?;
                fb.select(wrapped, m, s)
            }
        }
        IntFn::SubSat(Sign::Unsigned) => {
            let b = arg(args, 1)?;
            let under = bin(fb, BinOp::UGreaterThan, b, a)?;
            let d = bin(fb, BinOp::ISub, a, b)?;
            let z = splat(fb, f, 0)?;
            fb.select(under, z, d)
        }
        IntFn::AddSat(Sign::Signed) | IntFn::SubSat(Sign::Signed) if f.legalized() => {
            // Each operand keyed: its zero-extended bits with bit N - 1 flipped, its value plus
            // 2^(N-1). N + 1 <= C bits: the keys' sum, or their difference plus 2^N, is the exact
            // result plus 2^N and never below zero, so clamping it unsigned to
            // [2^(N-1), 3 * 2^(N-1) - 1] and taking 2^(N-1) away leaves the saturated result's
            // key, which flipped back is the result, zero-extended. The operands sign-extended,
            // summed, clamped signed to the width's range and masked come out sign-extended on
            // the RTX 3090 (NVIDIA 615.71.09), which drops that mask.
            let half = splat(fb, f, 1 << (n - 1))?;
            let x = bin(fb, BinOp::BitwiseXor, a, half)?;
            let y = bin(fb, BinOp::BitwiseXor, arg(args, 1)?, half)?;
            let t = if matches!(op, IntFn::AddSat(_)) {
                bin(fb, BinOp::IAdd, x, y)?
            } else {
                let up = splat(fb, f, 1 << n)?;
                let x = bin(fb, BinOp::IAdd, x, up)?;
                bin(fb, BinOp::ISub, x, y)?
            };
            let hi = splat(fb, f, 3 * (1 << (n - 1)) - 1)?;
            let t = glsl(fb, GlslOp::UMax, &[t, half])?;
            let t = glsl(fb, GlslOp::UMin, &[t, hi])?;
            let k = bin(fb, BinOp::ISub, t, half)?;
            bin(fb, BinOp::BitwiseXor, k, half)
        }
        IntFn::AddSat(Sign::Signed) | IntFn::SubSat(Sign::Signed) => {
            let add = matches!(op, IntFn::AddSat(_));
            let o = if add { BinOp::IAdd } else { BinOp::ISub };
            let y = arg(args, 1)?;
            let s = bin(fb, o, a, y)?;
            // Overflow iff the result's sign differs from what the operands allow.
            let xs = bin(fb, BinOp::BitwiseXor, a, s)?;
            let t = if add {
                bin(fb, BinOp::BitwiseXor, y, s)?
            } else {
                bin(fb, BinOp::BitwiseXor, a, y)?
            };
            let both = bin(fb, BinOp::BitwiseAnd, xs, t)?;
            let z = splat(fb, f, 0)?;
            let over = bin(fb, BinOp::SLessThan, both, z)?;
            // x < 0 ? MIN : MAX == (x >>a (C-1)) ^ MAX.
            let c1 = splat(fb, f, n - 1)?;
            let sgn = bin(fb, BinOp::ShiftRightArithmetic, a, c1)?;
            let max = splat(fb, f, ones(f.bits() - 1))?;
            let sat = bin(fb, BinOp::BitwiseXor, sgn, max)?;
            fb.select(over, sat, s)
        }
        IntFn::HAdd(s) | IntFn::RHAdd(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let one = splat(fb, f, 1)?;
            let shr = if signed(s) {
                BinOp::ShiftRightArithmetic
            } else {
                BinOp::ShiftRightLogical
            };
            let hx = bin(fb, shr, x, one)?;
            let hy = bin(fb, shr, y, one)?;
            let both = if matches!(op, IntFn::HAdd(_)) {
                bin(fb, BinOp::BitwiseAnd, x, y)?
            } else {
                bin(fb, BinOp::BitwiseOr, x, y)?
            };
            let carry = bin(fb, BinOp::BitwiseAnd, both, one)?;
            let r = bin(fb, BinOp::IAdd, hx, hy)?;
            let r = bin(fb, BinOp::IAdd, r, carry)?;
            mask(fb, f, r)
        }
        IntFn::ExtractBits(s) if f.container() == 64 => {
            extract_bits64(fb, f, signed(s), a, arg(args, 1)?, arg(args, 2)?)
        }
        IntFn::ExtractBits(s) => {
            let r = fb.bit_field_extract(signed(s), a, arg(args, 1)?, arg(args, 2)?)?;
            mask(fb, f, r)
        }
        IntFn::InsertBits if f.container() == 64 => {
            insert_bits64(fb, f, a, arg(args, 1)?, arg(args, 2)?, arg(args, 3)?)
        }
        IntFn::InsertBits => fb.bit_field_insert(a, arg(args, 1)?, arg(args, 2)?, arg(args, 3)?),
        IntFn::MadSat(s) => mad_sat(fb, f, signed(s), a, arg(args, 1)?, arg(args, 2)?),
        IntFn::Popcount => {
            if f.container() == 32 {
                return fb.unary(UnOp::BitCount, a);
            }
            let (lo, hi) = halves(fb, f, a)?;
            let cl = fb.unary(UnOp::BitCount, lo)?;
            let ch = fb.unary(UnOp::BitCount, hi)?;
            let r = bin(fb, BinOp::IAdd, cl, ch)?;
            widen(fb, f, r)
        }
        IntFn::ReverseBits => {
            let full = if f.container() == 32 {
                fb.unary(UnOp::BitReverse, a)?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let rl = fb.unary(UnOp::BitReverse, lo)?;
                let rh = fb.unary(UnOp::BitReverse, hi)?;
                let rl = widen(fb, f, rl)?;
                let rh = widen(fb, f, rh)?;
                let k = splat(fb, f, 32)?;
                let rl = bin(fb, BinOp::ShiftLeftLogical, rl, k)?;
                bin(fb, BinOp::BitwiseOr, rl, rh)?
            };
            if !f.legalized() {
                return Ok(full);
            }
            let sh = splat(fb, f, u64::from(f.container()) - n)?;
            bin(fb, BinOp::ShiftRightLogical, full, sh)
        }
        IntFn::Bswap => {
            let bytes = n / 8;
            let ff = splat(fb, f, 0xFF)?;
            let mut acc: Option<Value> = None;
            for i in 0..bytes {
                let byte = if i == 0 {
                    a
                } else {
                    let k = splat(fb, f, 8 * i)?;
                    bin(fb, BinOp::ShiftRightLogical, a, k)?
                };
                let byte = bin(fb, BinOp::BitwiseAnd, byte, ff)?;
                let to = 8 * (bytes - 1 - i);
                let placed = if to == 0 {
                    byte
                } else {
                    let k = splat(fb, f, to)?;
                    bin(fb, BinOp::ShiftLeftLogical, byte, k)?
                };
                acc = Some(match acc {
                    None => placed,
                    Some(x) => bin(fb, BinOp::BitwiseOr, x, placed)?,
                });
            }
            match acc {
                Some(x) => Ok(x),
                None => Ok(a),
            }
        }
        IntFn::Rotate => {
            let s = amount(fb, f, arg(args, 1)?)?;
            funnel_left(fb, f, a, a, s)
        }
        IntFn::Fshl => {
            let s = amount(fb, f, arg(args, 2)?)?;
            funnel_left(fb, f, a, arg(args, 1)?, s)
        }
        IntFn::Fshr => {
            // (b >> s) | ((a << 1) << (N - 1 - s)).
            let b = arg(args, 1)?;
            let s = amount(fb, f, arg(args, 2)?)?;
            let n1 = splat(fb, f, n - 1)?;
            let one = splat(fb, f, 1)?;
            let lo = bin(fb, BinOp::ShiftRightLogical, b, s)?;
            let a1 = bin(fb, BinOp::ShiftLeftLogical, a, one)?;
            let rest = bin(fb, BinOp::ISub, n1, s)?;
            let hi = bin(fb, BinOp::ShiftLeftLogical, a1, rest)?;
            let r = bin(fb, BinOp::BitwiseOr, hi, lo)?;
            mask(fb, f, r)
        }
        IntFn::Clz => {
            let u = u32_form(f);
            // 31 - FindUMsb(x): 32 for zero (FindUMsb gives -1).
            let clz32 = |fb: &mut FunctionBuilder<'_>, x: Value| -> BResult<Value> {
                let m = glsl(fb, GlslOp::FindUMsb, &[x])?;
                let k = splat(fb, u, 31)?;
                bin(fb, BinOp::ISub, k, m)
            };
            let r = if f.container() == 32 {
                clz32(fb, a)?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let cl = clz32(fb, lo)?;
                let ch = clz32(fb, hi)?;
                let z = splat(fb, u, 0)?;
                let hz = bin(fb, BinOp::IEqual, hi, z)?;
                let k = splat(fb, u, 32)?;
                let low = bin(fb, BinOp::IAdd, cl, k)?;
                let r = fb.select(hz, low, ch)?;
                widen(fb, f, r)?
            };
            if !f.legalized() {
                return Ok(r);
            }
            let pad = splat(fb, f, u64::from(f.container()) - n)?;
            bin(fb, BinOp::ISub, r, pad)
        }
        IntFn::Ctz => {
            let u = u32_form(f);
            let z = splat(fb, u, 0)?;
            let r = if f.container() == 32 {
                fb.glsl(GlslOp::FindILsb, &[a])?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let ll = glsl(fb, GlslOp::FindILsb, &[lo])?;
                let lh = glsl(fb, GlslOp::FindILsb, &[hi])?;
                let k = splat(fb, u, 32)?;
                let lh = bin(fb, BinOp::IAdd, lh, k)?;
                let lnz = bin(fb, BinOp::INotEqual, lo, z)?;
                let r = fb.select(lnz, ll, lh)?;
                widen(fb, f, r)?
            };
            let fz = splat(fb, f, 0)?;
            let zero = bin(fb, BinOp::IEqual, a, fz)?;
            let w = splat(fb, f, n)?;
            fb.select(zero, w, r)
        }
    }
}

/// Scalar `u32` operand `v` of a bit-field function as a 64-bit amount with `f`'s lanes.
fn amount64(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    let s = f.scalar_form();
    let t = ty(fb, s)?;
    let v = fb.convert(ConvOp::UConvert, v, t)?;
    if f.lanes == 1 {
        return Ok(v);
    }
    let vt = ty(fb, f)?;
    fb.composite_construct(vt, &vec![v; usize::from(f.lanes)])
}

/// `extract_bits` on a 64-bit container: `(x << (64 - o - n)) >> (64 - n)` (arithmetic when
/// `signed`), 0 when `n` is 0 (Metal leaves `o + n` beyond the width undefined).
fn extract_bits64(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    signed: bool,
    x: Value,
    offset: Value,
    count: Value,
) -> BResult<Value> {
    let (o, n) = (amount64(fb, f, offset)?, amount64(fb, f, count)?);
    let k64 = splat(fb, f, 64)?;
    let z = splat(fb, f, 0)?;
    // A zero count would shift by 64: shift by 0 and select 0 instead.
    let none = bin(fb, BinOp::IEqual, n, z)?;
    let up = bin(fb, BinOp::ISub, k64, o)?;
    let up = bin(fb, BinOp::ISub, up, n)?;
    let down = bin(fb, BinOp::ISub, k64, n)?;
    let down = fb.select(none, z, down)?;
    let up = fb.select(none, z, up)?;
    let v = bin(fb, BinOp::ShiftLeftLogical, x, up)?;
    let shr = if signed {
        BinOp::ShiftRightArithmetic
    } else {
        BinOp::ShiftRightLogical
    };
    let v = bin(fb, shr, v, down)?;
    let v = fb.select(none, z, v)?;
    mask(fb, f, v)
}

/// `insert_bits` on a 64-bit container: the `n` bits at `o` from `y`, the rest from `x`.
fn insert_bits64(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    offset: Value,
    count: Value,
) -> BResult<Value> {
    let (o, n) = (amount64(fb, f, offset)?, amount64(fb, f, count)?);
    let k64 = splat(fb, f, 64)?;
    let z = splat(fb, f, 0)?;
    let all = splat(fb, f, u64::MAX)?;
    // The field mask: all ones shifted down by 64 - n (n = 0 gives no field), then up by o.
    let none = bin(fb, BinOp::IEqual, n, z)?;
    let down = bin(fb, BinOp::ISub, k64, n)?;
    let down = fb.select(none, z, down)?;
    let m = bin(fb, BinOp::ShiftRightLogical, all, down)?;
    let m = fb.select(none, z, m)?;
    let m = bin(fb, BinOp::ShiftLeftLogical, m, o)?;
    let keep = fb.unary(UnOp::Not, m)?;
    let keep = bin(fb, BinOp::BitwiseAnd, x, keep)?;
    let ins = bin(fb, BinOp::ShiftLeftLogical, y, o)?;
    let ins = bin(fb, BinOp::BitwiseAnd, ins, m)?;
    let r = bin(fb, BinOp::BitwiseOr, keep, ins)?;
    mask(fb, f, r)
}

/// `mad_sat`: `a * b + c` computed exactly as a double-width `(hi, lo)` pair of containers and
/// saturated to the AIR width's range.
fn mad_sat(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    signed: bool,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let (a, b, c) = if signed {
        (sext(fb, f, a)?, sext(fb, f, b)?, sext(fb, f, c)?)
    } else {
        (a, b, c)
    };
    let wide = if signed {
        BinOp::SMulExtended
    } else {
        BinOp::UMulExtended
    };
    let p = bin(fb, wide, a, b)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    let s = bin(fb, BinOp::IAddCarry, lo, c)?;
    let lo = fb.composite_extract(s, &[0])?;
    let carry = fb.composite_extract(s, &[1])?;
    let hi = bin(fb, BinOp::IAdd, hi, carry)?;
    let z = splat(fb, f, 0)?;
    let n = u64::from(f.bits());
    let cw = u64::from(f.container());
    if !signed {
        let max = splat(fb, f, ones(f.bits()))?;
        let over_hi = bin(fb, BinOp::INotEqual, hi, z)?;
        let over_lo = bin(fb, BinOp::UGreaterThan, lo, max)?;
        let over = bin(fb, BinOp::LogicalOr, over_hi, over_lo)?;
        return fb.select(over, max, lo);
    }
    // A negative addend adds -1 to the high word (its sign extension).
    let cneg = bin(fb, BinOp::SLessThan, c, z)?;
    let m1 = splat(fb, f, u64::MAX)?;
    let ext = fb.select(cneg, m1, z)?;
    let hi = bin(fb, BinOp::IAdd, hi, ext)?;
    // The pair is one container value when hi is lo's sign extension.
    let k = splat(fb, f, cw - 1)?;
    let lo_sign = bin(fb, BinOp::ShiftRightArithmetic, lo, k)?;
    let fits = bin(fb, BinOp::IEqual, hi, lo_sign)?;
    let min = splat(fb, f, (!0u64) << (n - 1))?;
    let max = splat(fb, f, ones(f.bits() - 1))?;
    let r = glsl(fb, GlslOp::SMax, &[lo, min])?;
    let r = glsl(fb, GlslOp::SMin, &[r, max])?;
    let neg = bin(fb, BinOp::SLessThan, hi, z)?;
    let sat = fb.select(neg, min, max)?;
    let r = fb.select(fits, r, sat)?;
    mask(fb, f, r)
}

/// A float function on floats of one form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFn {
    Abs,
    Floor,
    Ceil,
    Trunc,
    /// Round half away from zero.
    Round,
    /// Round half to even.
    Rint,
    /// Metal's `fract`: the device's `Fract`, bounded below one as Metal defines it
    /// (`fmin(x - floor(x), 0x1.fffffep-1)`, [`fract_below_one`]).
    Fract,
    /// `air.fast_fract` on `float`: the device's `Fract`, which Metal's fast function does not
    /// bound (`fast::fract(-1e-10)` is `1.0`).
    FastFract,
    /// Metal's `sign`, with and without fast math: the device's `FSign`.
    Sign,
    /// Metal's `saturate`, `clamp(x, +0, 1)`: the device's `NClamp(x, 0, 1)`.
    Saturate,
    /// `fmin` (`air.fmin`, `air.fast_fmin`, `air.min`; `llvm.minnum` where [`FloatFn::MinNum`]
    /// refines to it): the device's `NMin`.
    Min,
    /// `fmax`, as [`FloatFn::Min`] with the order reversed: the device's `NMax`.
    Max,
    /// `clamp(x, lo, hi)`: the device's `NClamp`.
    Clamp,
    /// `min3(a, b, c)`: the device's `NMin(NMin(a, b), c)`.
    Min3,
    /// `max3(a, b, c)`: the device's `NMax(NMax(a, b), c)`.
    Max3,
    /// `llvm.minnum`: LLVM's function, the device's `NMin`, on the formats other than `float`.
    /// Metal's compiler emits it for `a < b ? a : b` under fast math, with every flag; on `float`
    /// it refines to [`FloatFn::Min`] with any flags, which Metal answers alike
    /// (`nan_minmax_fast`).
    MinNum,
    /// `llvm.maxnum`, as [`FloatFn::MinNum`] for `a > b ? a : b` and [`FloatFn::Max`].
    MaxNum,
    /// Metal's `median3`: `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and `NMax`
    /// ([`median3`]).
    Median3,
    /// `air.fast_fmedian3` on `float`: [`FloatFn::Median3`]'s form, where Metal's answers `b` for
    /// two NaN first operands, which the device's picks leave to it.
    FastMedian3,
    /// `mix(x, y, t)`: the device's `FMix` on `float` and `half`, and on `bfloat` and `double`
    /// the blend `x * (1 - t) + y * t`.
    Mix,
    Ldexp,
    /// `llvm.fmuladd` on `bfloat` and `double`: fused or not, as the implementation chooses.
    /// `float` and `half` select the fused multiply-add Metal computes ([`super::fma`]).
    MulAdd,
    /// `llvm.copysign`: the first operand's magnitude with the second's sign bit, moved as bits
    /// of the `f32` container, so exact for NaNs and zeros on every device. Not on `double`.
    CopySign,
    // Device precision (`PrecisionNote::DeviceArithmetic`).
    /// Metal's precise `sqrt`: correctly rounded (MSL Table 8.1). The device's `Sqrt`, Metal's own
    /// where the device's functions are Metal's in a module compiled with precise functions
    /// ([`libm`]).
    Sqrt,
    /// `fma` on `double` (`GLSL.std.450 Fma`) and `bfloat` (`narrow::bf16_fma`). `float` and `half`
    /// select the correctly rounded [`super::fma`].
    Fma,
    /// Metal's precise `fmod`: exact (Table 8.1), `fmod(x, ±inf) = x`. On `float` and `half` its
    /// routine, but where the device's `OpFRem` is Metal's `fmod` ([`libm::Fun::device_answers`]).
    Fmod,
    /// `dot`: the device's `OpDot`.
    Dot,
    // Device precision (`PrecisionNote::DeviceTranscendentals`).
    /// Metal's precise `rsqrt`: correctly rounded (Table 8.1); as [`FloatFn::Sqrt`].
    Rsqrt,
    /// `air.fast_rsqrt` on `float`: Metal's fast `rsqrt`, within 2 ulp (Table 8.2); the device's
    /// `InverseSqrt`.
    FastRsqrt,
    /// `air.fast_sqrt` on `float`: the device's `Sqrt`, where Metal's is `x · rsqrt(x)`.
    FastSqrt,
    /// Metal's precise `exp`: within 4 ulp (MSL Table 8.1). The device's `Exp`.
    Exp,
    /// `air.fast_exp` on `float`: `exp2(x · log2(e))` as Metal composes it, the product rounded,
    /// of Metal's `exp2` ([`FloatFn::Exp2`]).
    FastExp,
    /// Metal's `exp2`, the same function in both modes: the device's instruction.
    Exp2,
    /// Metal's precise `exp10`: Metal's own composition of the device's `exp2` ([`exp10`]), as no
    /// instruction is it.
    Exp10,
    /// `air.fast_exp10` on `float`: `exp2(x · log2(10))` (MSL Table 8.2), the product rounded.
    FastExp10,
    /// Metal's precise `log`: within 4 ulp. The device's `Log`.
    Log,
    /// `air.fast_log` on `float`: `log2(x) · ln(2)` as Metal composes it, the product rounded,
    /// of Metal's `log2` ([`FloatFn::Log2`]): every lane of case f6524207.
    FastLog,
    /// Metal's `log2`, the same function in both modes; as [`FloatFn::Exp2`].
    Log2,
    /// `air.fast_log2` on `float`: Metal's `log2` ([`FloatFn::Log2`]), the device's instruction.
    FastLog2,
    /// Metal's precise `log10`; as [`FloatFn::Exp10`].
    Log10,
    /// `air.fast_log10` on `float`: `log2(x) · log10(2)` (MSL Table 8.2), the product rounded:
    /// every lane of case f6524207.
    FastLog10,
    /// Metal's precise `sin`: within 4 ulp (MSL Table 8.1). The device's instruction, Metal's own
    /// where the device's functions are Metal's in a module compiled with precise functions
    /// ([`libm`]).
    Sin,
    /// Metal's precise `cos`: within 4 ulp; as [`FloatFn::Sin`].
    Cos,
    /// `air.fast_sin` on `float`: the device's `Sin`, Metal's `fast::sin` itself where its
    /// functions are Metal's in a fast-math module. Metal answers `+0` past `float(π/2) · 2^22` and
    /// for an infinity or NaN; the device answers its `Sin` there.
    FastSin,
    /// `air.fast_cos` on `float`: the device's `Cos`; as [`FloatFn::FastSin`].
    FastCos,
    /// Metal's precise `tan`: within 6 ulp; as [`FloatFn::Sin`].
    Tan,
    /// `air.fast_tan` on `float`: the device's `Tan`, Metal's `fast::tan` itself where its
    /// functions are Metal's in a fast-math module (MSL Table 8.2 defines it as `sin(x) · (1 /
    /// cos(x))`).
    FastTan,
    /// Metal's precise `asin`: within 4 ulp (MSL Table 8.1). The device's instruction, Metal's own
    /// where the device's functions are Metal's in a module compiled with precise functions
    /// ([`libm`]).
    Asin,
    /// `air.fast_asin` on `float`: within 5 ulp (Table 8.2); as [`FloatFn::Asin`], in Metal's
    /// fast mode.
    FastAsin,
    /// Metal's precise `acos`: within 4 ulp; as [`FloatFn::Asin`].
    Acos,
    /// `air.fast_acos` on `float`: within 5 ulp; as [`FloatFn::FastAsin`].
    FastAcos,
    /// Metal's precise `atan`: within 5 ulp; as [`FloatFn::Asin`].
    Atan,
    /// `air.fast_atan` on `float`: within 5 ulp; as [`FloatFn::FastAsin`].
    FastAtan,
    /// Metal's precise `sinh`; as [`FloatFn::Exp`].
    Sinh,
    /// `air.fast_sinh` on `float`: the device's `Sinh`, Metal's `fast::sinh` itself where its
    /// functions are Metal's in a fast-math module (MSL Table 8.2 defines it as
    /// `(exp(x) - exp(-x)) · 0.5`).
    FastSinh,
    /// Metal's precise `cosh`; as [`FloatFn::Exp`].
    Cosh,
    /// `air.fast_cosh` on `float`: the device's `Cosh`; as [`FloatFn::FastSinh`].
    FastCosh,
    /// Metal's precise `tanh` (within 5 ulp): the device's instruction, Metal's own where its
    /// functions are Metal's (SPIRV-Cross emits `precise::tanh` in either mode).
    Tanh,
    /// `air.fast_tanh` on `float`: the device's `Tanh`. MSL Table 8.2 defines Metal's as the
    /// formula `(t - 1) / (t + 1)` with `t = exp(2x)`, which is `NaN` once `t` overflows, where the
    /// device's `Tanh` answers `±1`; on a device whose functions are Metal's the instruction is
    /// Metal's precise `tanh`. Narrow forms keep [`FloatFn::Tanh`]: the fast-math table is the
    /// single-precision one.
    FastTanh,
    /// Metal's precise `asinh`; as [`FloatFn::Exp`].
    Asinh,
    /// `air.fast_asinh` on `float` with every flag: Metal's fast formula
    /// `log(x + sqrt(x * x + 1))` (MSL Table 8.2), which is the canonical NaN at `-inf`
    /// (`-inf + inf`): the device's `Asinh`, Metal's own in a fast-math module where its functions
    /// are Metal's.
    FastAsinh,
    /// Metal's precise `acosh`; as [`FloatFn::Exp`].
    Acosh,
    /// `air.fast_acosh` on `float`: `log(x + sqrt(x * x - 1))`; as [`FloatFn::FastAsinh`].
    FastAcosh,
    /// Metal's precise `atanh`; as [`FloatFn::Exp`].
    Atanh,
    /// `air.fast_atanh` on `float`: the device's `Atanh`, Metal's `fast::atanh` itself where its
    /// functions are Metal's in a fast-math module (MSL Table 8.2 defines it as
    /// `0.5 · log((1 + x) / (1 - x))`).
    FastAtanh,
    /// Metal's `sinpi`: within 4 ulp (MSL Table 8.1). The exact reduction around the device's `Sin`
    /// and `Cos` (`pi_scaled`), as no instruction is it.
    SinPi,
    /// Metal's `cospi`; as [`FloatFn::SinPi`].
    CosPi,
    /// Metal's `tanpi`: within 6 ulp; as [`FloatFn::SinPi`].
    TanPi,
    /// `air.fast_sinpi` on `float`: Metal's fast `sinpi`, which is its precise one
    /// ([`FloatFn::SinPi`]), the same reduction.
    FastSinPi,
    /// `air.fast_cospi` on `float`; as [`FloatFn::FastSinPi`].
    FastCosPi,
    /// `air.fast_tanpi` on `float`; as [`FloatFn::FastSinPi`] (Table 8.2: `tan(x * pi)`).
    FastTanPi,
    Powr,
    /// `air.fast_powr` on `float`: `exp2(y · log2(x))` as Metal composes it, the product rounded.
    FastPowr,
    /// Metal `pow`: C99 Annex F's special cases that `GLSL.std.450 Pow` leaves undefined (negative
    /// bases with integral exponents, the zero, one, and NaN cases) around `Pow` of the magnitude
    /// (`pow_special`).
    Pow,
    /// `air.fast_pow` on `float`: [`FloatFn::Pow`] around the magnitude `exp2(y · log2|x|)`, as
    /// Metal composes it.
    FastPow,
    /// Metal `atan2(y, x)`: C99 Annex F's special values (`atan2_special`) around the generic
    /// value, the device's `Atan2` (always `precise::atan2` where its functions are Metal's).
    Atan2,
    /// `air.fast_atan2` on `float`: `atan(y / x)` turned by `±π` when `x` is below zero, MSL Table
    /// 8.2's formula (`fast_atan2`): `0 / 0` and `inf / inf` are NaN, where `atan2` answers C99's
    /// values, and the turn follows `y`'s comparison with zero, so `fast::atan2(-0, -1)` is `+π`.
    FastAtan2 {
        /// The quotient is `y` times `x`'s reciprocal flushed to zero beyond `2^126`, as Metal
        /// answers a call that does not carry every fast-math flag; a call that does divides
        /// over the whole range.
        reciprocal_flushed: bool,
    },
}

/// The precision class of a float function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatPrecision {
    Exact,
    DeviceArithmetic,
    DeviceTranscendental,
}

impl FloatFn {
    /// The operand count.
    pub fn arity(self) -> usize {
        use FloatFn as F;
        match self {
            F::Min
            | F::Max
            | F::MinNum
            | F::MaxNum
            | F::Ldexp
            | F::Fmod
            | F::Dot
            | F::Powr
            | F::FastPowr
            | F::Pow
            | F::FastPow
            | F::Atan2
            | F::FastAtan2 { .. }
            | F::CopySign => 2,
            F::Clamp
            | F::Min3
            | F::Max3
            | F::Median3
            | F::FastMedian3
            | F::Mix
            | F::MulAdd
            | F::Fma => 3,
            _ => 1,
        }
    }

    pub fn precision(self) -> FloatPrecision {
        use FloatFn as F;
        match self {
            F::Sqrt | F::Fma | F::Fmod | F::Dot => FloatPrecision::DeviceArithmetic,
            F::Rsqrt
            | F::FastRsqrt
            | F::FastSqrt
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
            | F::Sin
            | F::Cos
            | F::FastSin
            | F::FastCos
            | F::Tan
            | F::FastTan
            | F::Asin
            | F::FastAsin
            | F::Acos
            | F::FastAcos
            | F::Atan
            | F::FastAtan
            | F::Sinh
            | F::FastSinh
            | F::Cosh
            | F::FastCosh
            | F::Tanh
            | F::FastTanh
            | F::Asinh
            | F::FastAsinh
            | F::Acosh
            | F::FastAcosh
            | F::Atanh
            | F::FastAtanh
            | F::SinPi
            | F::CosPi
            | F::TanPi
            | F::FastSinPi
            | F::FastCosPi
            | F::FastTanPi
            | F::Powr
            | F::FastPowr
            | F::Pow
            | F::FastPow
            | F::Atan2
            | F::FastAtan2 { .. } => FloatPrecision::DeviceTranscendental,
            _ => FloatPrecision::Exact,
        }
    }

    /// Whether the lowering answers Metal at `f32` denormal operands and results itself, whatever
    /// the device's denormal mode, or records the device's answer under another note, so it records
    /// no `PrecisionNote::DeviceDenormals` of its own: `copysign`, which moves bits, and precise
    /// `fmod` on `float`, the remainder's routine, which reads a denormal divisor as a zero on the
    /// bits and computes the remainder in integers (`libm::fmod`), or on MoltenVK Metal's own;
    /// `fmin`, `fmax`, `min3`, `max3`, `clamp`, `saturate` and `median3`, the device's `NMin`,
    /// `NMax` and `NClamp`, whose comparison of a denormal is Metal's where measured
    /// (`DriverDeviation::MetalMinMax`, but in `median3`) and else `PrecisionNote::NativeMinMax`'s
    /// bound; `round`,
    /// whose every denormal operand rounds to a zero, its sign the note's
    /// (`PrecisionNote::NativeTranscendentals`); and the device's instructions for Metal's
    /// functions: `fabs`, `sign`, `fract` and `fast::sqrt` ([`FloatFn::native_instruction`]), the
    /// trigonometric, hyperbolic and arc families in both modes, `sinpi`, `cospi`, and `tanpi`,
    /// the exponentials and logarithms, and `pow`, `powr` and `atan2` with their special cases
    /// (the device's comparisons around the device's instruction), Metal's own where the device's
    /// functions are Metal's, and elsewhere recording the device's denormals with their precision
    /// (`PrecisionNote::NativeTranscendentals`). `fast::atan2` does not: its quotient is the
    /// device's division, whose operands `denormal::coverage` reads.
    pub fn metal_denormals(self) -> bool {
        use FloatFn as F;
        matches!(
            self,
            F::Abs
                | F::Round
                | F::Fmod
                | F::Sign
                | F::FastSqrt
                | F::Saturate
                | F::Fract
                | F::FastFract
                | F::Median3
                | F::FastMedian3
                | F::Min
                | F::Max
                | F::Clamp
                | F::Min3
                | F::Max3
                | F::CopySign
                | F::Atan2
                | F::Sin
                | F::Cos
                | F::Tan
                | F::FastSin
                | F::FastCos
                | F::FastTan
                | F::SinPi
                | F::CosPi
                | F::TanPi
                | F::Asin
                | F::FastAsin
                | F::Acos
                | F::FastAcos
                | F::Atan
                | F::FastAtan
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
                | F::Sinh
                | F::FastSinh
                | F::Cosh
                | F::FastCosh
                | F::Tanh
                | F::FastTanh
                | F::Asinh
                | F::FastAsinh
                | F::Acosh
                | F::FastAcosh
                | F::Atanh
                | F::FastAtanh
                | F::Pow
                | F::FastPow
                | F::Powr
                | F::FastPowr
        )
    }

    /// Whether the call takes the device's instruction for one of Metal's functions that no
    /// `libm::Fun` names, which is Metal's own function where the device's functions are Metal's
    /// (SPIRV-Cross writes Metal's `fabs`, `sign`, `fract` and `sqrt`), and elsewhere answers its
    /// special operands as the device does: `fabs` (`FAbs`), `sign` (`FSign`), `fract` and
    /// `fast::fract` (`Fract`), and `fast::sqrt` (`Sqrt`, where Metal's is `x · rsqrt(x)`).
    pub fn native_instruction(self) -> bool {
        matches!(
            self,
            FloatFn::FastSqrt | FloatFn::Abs | FloatFn::Sign | FloatFn::Fract | FloatFn::FastFract
        )
    }

    /// Whether the lowering composes the function of the device's instructions on every device,
    /// Metal's own instruction left aside: `round`, half away from zero from the device's `Trunc`
    /// (`GLSL.std.450 Round` leaves the direction to the device), whose zero is `-0` for a
    /// negative operand that rounds to it (C99's), where Metal answers `+0`
    /// (`PrecisionNote::NativeTranscendentals`).
    pub fn composed_on_every_device(self) -> bool {
        self == FloatFn::Round
    }

    /// Whether the lowering on lanes of format `fmt` records no NaN note of its own: `saturate`,
    /// which answers a number for one, `copysign`, which moves bits, `sign`, whose NaN answer is
    /// its instruction's
    /// ([`FloatFn::native_instruction`]), and the device's picks, `median3` and, on `float` and
    /// `half`, `fmin`, `fmax`, `min3`, `max3`, and `clamp`, which return an operand (two NaNs'
    /// answer is `PrecisionNote::NativeMinMax`'s where the device is not measured to pick
    /// Metal's).
    pub fn metal_nans(self, fmt: FloatFmt) -> bool {
        use FloatFn as F;
        if matches!(self, F::Min | F::Max | F::Clamp | F::Min3 | F::Max3) {
            return metal_picks(Some(fmt));
        }
        matches!(
            self,
            F::Sign | F::Saturate | F::Median3 | F::FastMedian3 | F::CopySign
        )
    }

    /// Whether the function divides by a device division besides its [`FloatFn::precision`]
    /// class (`PrecisionNote::DeviceArithmetic`): `fast_atan2`'s `y / x` (the device's division)
    /// and `fast::tanpi`'s quotient where the device's functions are Metal's.
    pub fn divides(self) -> bool {
        matches!(self, FloatFn::FastAtan2 { .. } | FloatFn::FastTanPi)
    }

    /// Whether the function is lowered on 64-bit floats: `GLSL.std.450` offers the
    /// transcendental families on 16/32-bit floats only, and `copysign` moves the bits of an
    /// `f32` container.
    pub fn allows_f64(self) -> bool {
        self.precision() != FloatPrecision::DeviceTranscendental && self != FloatFn::CopySign
    }

    /// Whether the result on values of a narrow format is again a value of the format without
    /// rounding (it is an operand, a constant of the format, or an integral value of one).
    /// `fract` is not: `x - floor(x)` of a negative non-integral `x` can need more bits than
    /// the format has (half `-0.300048828125` gives `0.699951171875`), so it rounds once, before
    /// its bound ([`fract_below_one`]).
    pub fn closed_on_narrow(self) -> bool {
        use FloatFn as F;
        matches!(
            self,
            F::Abs
                | F::Floor
                | F::Ceil
                | F::Trunc
                | F::Round
                | F::Rint
                | F::Sign
                | F::Saturate
                | F::Min
                | F::Max
                | F::MinNum
                | F::MaxNum
                | F::Clamp
                | F::Min3
                | F::Max3
                | F::Median3
                | F::FastMedian3
                | F::CopySign
        )
    }

    /// An upper estimate of the SPIR-V words per use on form `f`, including the roundings of a
    /// narrow form.
    pub fn words(self, f: Form) -> u32 {
        let half = f.float_fmt() == Some(FloatFmt::Half);
        let base = match half_form(self, f) {
            // A `half` `exp10` and `log10` by the product with the device's `exp2` and `log2`
            // (`half_form`), and a scalar's constants of the rounding.
            FloatFn::FastExp10 | FloatFn::FastLog10 if half => 128,
            // The device's `Fract`, the comparison with one and the bound's select.
            FloatFn::Fract => 64,
            // The truncation, the difference's test against one half and the step away from zero.
            FloatFn::Round => 112,
            // The device's function and the operands' tests where Metal answers NaN, the precise
            // sign's on the bits, each operand classified once.
            FloatFn::Powr => 208,
            // And the infinite product's tests and select, on the same classification
            // (`fast_powr_special`).
            FloatFn::FastPowr => 208 + INFINITE_PRODUCT_WORDS,
            // The device's remainder and, on `half`, the zero's tests and selects.
            FloatFn::Fmod if f.float_fmt() == Some(FloatFmt::Half) => 192,
            // The device's four picks.
            FloatFn::Median3 | FloatFn::FastMedian3 => 64,
            // Metal's pick: the operands' NaN tests (the bits, 6 instructions each, where the
            // entry does not preserve NaNs), the comparison, and two selects per step, and each
            // operand's flush test.
            FloatFn::Min | FloatFn::Max => 160,
            FloatFn::Clamp | FloatFn::Min3 | FloatFn::Max3 => 288,
            // `|x|`, and C99's special cases around the magnitude (`pow_special`).
            FloatFn::Pow => 344,
            // And the infinite product's tests and select (`infinite_product`).
            FloatFn::FastPow => 344 + INFINITE_PRODUCT_WORDS,
            // The special operands' tests and the selected constant's sign.
            FloatFn::Atan2 => 320,
            // The quotient, its special values and sign, and the turn.
            FloatFn::FastAtan2 { .. } => 400,
            // The period reduction, the exact axis arms, both quadrant selections, `tanpi`'s
            // quarter turns, and the finiteness guard around the device's `Sin` and `Cos`
            // (`pi_scaled`).
            FloatFn::SinPi
            | FloatFn::CosPi
            | FloatFn::TanPi
            | FloatFn::FastSinPi
            | FloatFn::FastCosPi
            | FloatFn::FastTanPi => 384,
            // The tie correction of a bfloat fused multiply-add (`narrow::bf16_fma`).
            FloatFn::Fma if f.float_fmt() == Some(FloatFmt::BFloat) => 208,
            // `exp10`'s composition of the device's `exp2` (`exp10`): five products and
            // sums, two fused ones, the device's `exp2`, and the ends' tests: 136 words on
            // `float4`.
            FloatFn::Exp10 => 128,
            // `log10`'s composition of the device's `log2` (`log10`): the device's `log2` and
            // the product, the polynomial's fused steps, and the range test.
            FloatFn::Log10 => 200,
            // The exponent's clamp, the bits' fields, the device's two products, tests and
            // selects (`ldexp_bits`).
            FloatFn::Ldexp => 320,
            _ => 48,
        };
        // Each function the call takes: the device's instruction, or precise `fmod`'s routine's
        // calls, one per lane (the routine itself is the module's).
        let base = base
            + libm::uses(self, f)
                .iter()
                .map(|r| libm::use_words(f.lanes, r.arity()))
                .sum::<u32>();
        let roundings = match self {
            _ if self.closed_on_narrow() => 0,
            FloatFn::Mix => 3,
            FloatFn::MulAdd => 2,
            _ => 1,
        };
        base + roundings * narrow::round_words(f)
    }
}

/// Emit float function `op` on `args` (floats of form `f`; `Ldexp`'s exponent is a held `i32` and
/// `Dot` gives the scalar), by the device's instructions, and precise `fmod` by `lib`'s routine
/// where the module defines it. A `half` function is its `float` function rounded, its zeros' signs
/// and NaNs' signs that function's, but `fmod`, which answers Metal's zero remainder around the
/// exact remainder's routine (`half_fmod_zero`).
pub fn float_fn(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    contract: Contraction,
    args: &[Value],
    lib: &Lib,
) -> BResult<Value> {
    use FloatFn as F;
    let r = float_fn_core(fb, op, f, contract, args, lib)?;
    let r = match op {
        _ if op.closed_on_narrow() => r,
        // A `half` `mix`, the device's `FMix` in `float`, rounds once.
        F::Mix if f.float_fmt() == Some(FloatFmt::Half) => narrow::round(fb, f, r)?,
        // Rounded per step below.
        F::Mix | F::MulAdd | F::Fract | F::FastFract => r,
        F::Dot => narrow::round(fb, f.scalar_form(), r)?,
        F::Fmod => {
            let r = narrow::round(fb, f, r)?;
            half_fmod_zero(fb, f, arg(args, 0)?, r)?
        }
        _ => narrow::round(fb, f, r)?,
    };
    Ok(r)
}

/// The `float` function a `half` call of `op` (form `f`, held in `float`) takes before its
/// rounding, where that is not `op` itself (`op` otherwise, and on another format): `exp10` and
/// `log10`, which no instruction is, are the device's `exp2` and `log2` of the product with
/// `log2(10)` and `log10(2)` (`FastExp10`, `FastLog10`), as `float`'s fast forms are. Every other
/// `half` function is its `float` function's instruction (or composition), rounded.
pub(crate) fn half_form(op: FloatFn, f: Form) -> FloatFn {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return op;
    }
    match op {
        FloatFn::Exp10 => FloatFn::FastExp10,
        FloatFn::Log10 => FloatFn::FastLog10,
        _ => op,
    }
}

/// [`float_fn`] before the final rounding to a narrow format.
fn float_fn_core(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    contract: Contraction,
    args: &[Value],
    lib: &Lib,
) -> BResult<Value> {
    use FloatFn as F;
    let a = arg(args, 0)?;
    let op = half_form(op, f);
    let g1 = |fb: &mut FunctionBuilder<'_>, o: GlslOp| glsl(fb, o, &[a]);
    let times = |fb: &mut FunctionBuilder<'_>, k: f64| -> BResult<Value> {
        let c = wide_splat(fb, f, k)?;
        bin(fb, BinOp::FMul, a, c)
    };
    match op {
        F::Abs => g1(fb, GlslOp::FAbs),
        F::Floor => g1(fb, GlslOp::Floor),
        F::Ceil => g1(fb, GlslOp::Ceil),
        F::Trunc => g1(fb, GlslOp::Trunc),
        F::Rint => g1(fb, GlslOp::RoundEven),
        F::Round => {
            // Metal's definition, halfway cases away from zero, which `GLSL.std.450 Round` leaves
            // to the device: t = trunc(x); |x - t| >= 0.5 ? t + (x < 0 ? -1 : 1) : t. `x - t` is
            // exact. A zero is the device's `Trunc`'s, `-0` below zero (C99's), where Metal
            // answers `+0`.
            let t = g1(fb, GlslOp::Trunc)?;
            let d = bin(fb, BinOp::FSub, a, t)?;
            let d = glsl(fb, GlslOp::FAbs, &[d])?;
            let half = fsplat(fb, f, 0.5)?;
            let up = bin(fb, BinOp::FOrdGreaterThanEqual, d, half)?;
            let z = fsplat(fb, f, 0.0)?;
            let neg = bin(fb, BinOp::FOrdLessThan, a, z)?;
            let (m1, p1) = (fsplat(fb, f, -1.0)?, fsplat(fb, f, 1.0)?);
            let step = fb.select(neg, m1, p1)?;
            let away = bin(fb, BinOp::FAdd, t, step)?;
            fb.select(up, away, t)
        }
        F::Fract => fract_below_one(fb, f, a),
        F::FastFract => g1(fb, GlslOp::Fract),
        F::Sign => g1(fb, GlslOp::FSign),
        // Metal's `clamp(x, +0, 1)`: the device's `NClamp`, Metal's where measured
        // (`DriverDeviation::MetalMinMax`), else noted `PrecisionNote::NativeMinMax`, as the
        // minimum and maximum functions below.
        F::Saturate => {
            let (z, one) = (fsplat(fb, f, 0.0)?, fsplat(fb, f, 1.0)?);
            glsl(fb, GlslOp::NClamp, &[a, z, one])
        }
        F::Min | F::MinNum => glsl(fb, GlslOp::NMin, &[a, arg(args, 1)?]),
        F::Max | F::MaxNum => glsl(fb, GlslOp::NMax, &[a, arg(args, 1)?]),
        F::Clamp => glsl(fb, GlslOp::NClamp, &[a, arg(args, 1)?, arg(args, 2)?]),
        F::Min3 | F::Max3 => {
            let o = if op == F::Min3 {
                GlslOp::NMin
            } else {
                GlslOp::NMax
            };
            let r = glsl(fb, o, &[a, arg(args, 1)?])?;
            glsl(fb, o, &[r, arg(args, 2)?])
        }
        F::Median3 | F::FastMedian3 => median3(fb, [a, arg(args, 1)?, arg(args, 2)?]),
        // The device's `FMix` (`PrecisionNote::NativeMix`).
        F::Mix if matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) => {
            glsl(fb, GlslOp::FMix, &[a, arg(args, 1)?, arg(args, 2)?])
        }
        F::Mix => {
            // The blend `x * (1 - t) + y * t` on `bfloat` and `double`, which Metal's fused form is
            // not measured on: its multiplies and add carry the call's contraction, and each step
            // rounds to a narrow format by the device's rounding (`narrow::round`).
            let (y, t) = (arg(args, 1)?, arg(args, 2)?);
            let one = fsplat(fb, f, 1.0)?;
            let w = fb.binary_with(BinOp::FSub, one, t, contract)?;
            let w = narrow::round(fb, f, w)?;
            let p = fb.binary_with(BinOp::FMul, a, w, contract)?;
            let p = narrow::round(fb, f, p)?;
            let q = fb.binary_with(BinOp::FMul, y, t, contract)?;
            let q = narrow::round(fb, f, q)?;
            let r = fb.binary_with(BinOp::FAdd, p, q, contract)?;
            narrow::round(fb, f, r)
        }
        F::Ldexp => ldexp(fb, f, a, arg(args, 1)?),
        F::CopySign => copysign(fb, f, a, arg(args, 1)?),
        F::MulAdd => {
            // A narrow form takes the unfused choice, each step rounded once by the device.
            let p = bin(fb, BinOp::FMul, a, arg(args, 1)?)?;
            let p = narrow::round(fb, f, p)?;
            let r = bin(fb, BinOp::FAdd, p, arg(args, 2)?)?;
            narrow::round(fb, f, r)
        }
        F::Sqrt => g1(fb, GlslOp::Sqrt),
        F::Fma if f.float_fmt() == Some(FloatFmt::BFloat) => {
            narrow::bf16_fma(fb, f, a, arg(args, 1)?, arg(args, 2)?)
        }
        F::Fma => glsl(fb, GlslOp::Fma, &[a, arg(args, 1)?, arg(args, 2)?]),
        F::Fmod => {
            let y = arg(args, 1)?;
            match lib.fmod(fb, f, a, y)? {
                Some(r) if libm::uses(op, f).contains(&libm::Fun::Fmod) => Ok(r),
                _ => bin(fb, BinOp::FRem, a, y),
            }
        }
        F::Dot => bin(fb, BinOp::Dot, a, arg(args, 1)?),
        F::Rsqrt => g1(fb, GlslOp::InverseSqrt),
        F::FastRsqrt => g1(fb, GlslOp::InverseSqrt),
        F::FastSqrt => g1(fb, GlslOp::Sqrt),
        F::Exp => g1(fb, GlslOp::Exp),
        F::Exp2 => g1(fb, GlslOp::Exp2),
        F::Log => g1(fb, GlslOp::Log),
        F::Log2 | F::FastLog2 => g1(fb, GlslOp::Log2),
        // No instruction is Metal's `exp10` or `log10`: Metal's own compositions of the device's
        // `exp2` and `log2`.
        F::Exp10 => exp10(fb, f, a),
        F::Log10 => log10(fb, f, a),
        // Metal's fast compositions of its `exp2` and `log2` (MSL Table 8.2).
        F::FastExp | F::FastExp10 => {
            let k = if op == F::FastExp {
                std::f64::consts::LOG2_E
            } else {
                std::f64::consts::LOG2_10
            };
            let x = times(fb, k)?;
            glsl(fb, GlslOp::Exp2, &[x])
        }
        F::FastLog | F::FastLog10 => {
            let l = g1(fb, GlslOp::Log2)?;
            let k = if op == F::FastLog {
                std::f64::consts::LN_2
            } else {
                std::f64::consts::LOG10_2
            };
            let c = wide_splat(fb, f, k)?;
            fb.binary_with(BinOp::FMul, l, c, Contraction::Forbidden)
        }
        // The device's instructions in both modes, Metal's fast functions where its functions are
        // Metal's in a fast-math module (`libm::Fun::device_answers`).
        F::Sin | F::FastSin => g1(fb, GlslOp::Sin),
        F::Cos | F::FastCos => g1(fb, GlslOp::Cos),
        F::Tan | F::FastTan => g1(fb, GlslOp::Tan),
        F::Asin | F::FastAsin => g1(fb, GlslOp::Asin),
        F::Acos | F::FastAcos => g1(fb, GlslOp::Acos),
        F::Atan | F::FastAtan => g1(fb, GlslOp::Atan),
        F::Sinh | F::FastSinh => g1(fb, GlslOp::Sinh),
        F::Cosh | F::FastCosh => g1(fb, GlslOp::Cosh),
        F::Tanh | F::FastTanh => g1(fb, GlslOp::Tanh),
        F::Asinh | F::FastAsinh => g1(fb, GlslOp::Asinh),
        F::Acosh | F::FastAcosh => g1(fb, GlslOp::Acosh),
        F::Atanh | F::FastAtanh => g1(fb, GlslOp::Atanh),
        F::SinPi | F::CosPi | F::TanPi | F::FastSinPi | F::FastCosPi | F::FastTanPi => {
            let which = match op {
                F::SinPi | F::FastSinPi => PiScaled::Sin,
                F::CosPi | F::FastCosPi => PiScaled::Cos,
                _ => PiScaled::Tan,
            };
            // The device's `Sin` and `Cos` of `float(π) r`.
            pi_scaled(
                fb,
                f,
                which,
                a,
                &|fb, residue| {
                    let pi = wide_splat(fb, f, std::f64::consts::PI)?;
                    let angle = bin(fb, BinOp::FMul, pi, residue)?;
                    Ok((
                        glsl(fb, GlslOp::Sin, &[angle])?,
                        glsl(fb, GlslOp::Cos, &[angle])?,
                    ))
                },
                &|fb, num, den| bin(fb, BinOp::FDiv, num, den),
            )
        }
        F::Powr => {
            let y = arg(args, 1)?;
            let r = glsl(fb, GlslOp::Pow, &[a, y])?;
            powr_nan(fb, f, a, y, r)
        }
        F::FastPowr => {
            let y = arg(args, 1)?;
            let r = exp2_of_product(fb, y, a)?;
            fast_powr_special(fb, f, a, y, r)
        }
        // C99 Annex F's special cases around the magnitude of `|x|`, which `GLSL.std.450 Pow`
        // defines only for a positive base.
        F::Pow | F::FastPow => {
            let y = arg(args, 1)?;
            let ax = glsl(fb, GlslOp::FAbs, &[a])?;
            let magnitude = if op == F::Pow {
                glsl(fb, GlslOp::Pow, &[ax, y])?
            } else {
                let r = exp2_of_product(fb, y, ax)?;
                infinite_product(fb, f, y, ax, r)?
            };
            pow_special(fb, f, a, y, magnitude)
        }
        F::Atan2 => {
            let x = arg(args, 1)?;
            let generic = glsl(fb, GlslOp::Atan2, &[a, x])?;
            atan2_special(fb, f, a, x, generic)
        }
        F::FastAtan2 { reciprocal_flushed } => {
            // Metal's fast `atan` of its fast quotient `y · rcp(x)`: the device's `Atan` and
            // division.
            let atan = |fb: &mut FunctionBuilder<'_>, q: Value| glsl(fb, GlslOp::Atan, &[q]);
            let divide = |fb: &mut FunctionBuilder<'_>, n: Value, d: Value| -> BResult<Value> {
                bin(fb, BinOp::FDiv, n, d)
            };
            fast_atan2(fb, f, a, arg(args, 1)?, reciprocal_flushed, &atan, &divide)
        }
    }
}

/// `llvm.copysign(a, b)` on floats of form `f` held in an `f32` container: `a`'s bits with the
/// sign bit replaced by `b`'s (LangRef), so a NaN keeps its payload and a zero takes `b`'s sign
/// whatever the device's float arithmetic does with them. A narrow value's container is the
/// widened value, whose sign bit is the format's.
fn copysign(fb: &mut FunctionBuilder<'_>, f: Form, a: Value, b: Value) -> BResult<Value> {
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let ab = fb.convert(ConvOp::Bitcast, a, ut)?;
    let bb = fb.convert(ConvOp::Bitcast, b, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let ab = bin(fb, BinOp::BitwiseAnd, ab, magnitude)?;
    let bb = bin(fb, BinOp::BitwiseAnd, bb, sign)?;
    let r = bin(fb, BinOp::BitwiseOr, ab, bb)?;
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The constant `k` at the container's precision: an internal constant of a sequence that runs
/// in the container (`f32` for a narrow form) before its final rounding.
fn wide_splat(fb: &mut FunctionBuilder<'_>, f: Form, k: f64) -> BResult<Value> {
    let wide = match f.float_fmt() {
        Some(n) if n.narrow() => f.with(Held::Float(FloatFmt::F32)),
        _ => f,
    };
    fsplat(fb, wide, k)
}

// ---- Metal's functions at their special operands ----

/// Whether lanes of `x` (floats of form `f`, held in a 32-bit container) have the bits `bits`:
/// an integer comparison, which no float mode lets a device fold.
fn has_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, bits: u64) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let k = splat(fb, u, bits)?;
    bin(fb, BinOp::IEqual, b, k)
}

/// `r` (a `half` `fmod`'s remainder of `x`, floats of form `f`) with Metal's zero: a zero
/// remainder of a nonzero dividend is `+0`, and a zero dividend's is the dividend itself
/// (`fmod(-2, 0.5)` is `+0`, `fmod(-0, 1)` is `-0`), with or without fast-math flags
/// (`nan_fmod_zero`, `nan_fmod_zero_fast`). A `float` remainder keeps the dividend's sign, as C's
/// `fmod` does, and so does Metal's: it and every other format keep `r`. Tested on the bits, so no
/// signed-zero mode folds it.
fn half_fmod_zero(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, r: Value) -> BResult<Value> {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return Ok(r);
    }
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let zero = splat(fb, u, 0)?;
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let rm = bin(fb, BinOp::BitwiseAnd, rb, magnitude)?;
    let r_zero = bin(fb, BinOp::IEqual, rm, zero)?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let xm = bin(fb, BinOp::BitwiseAnd, xb, magnitude)?;
    let x_zero = bin(fb, BinOp::IEqual, xm, zero)?;
    let z = fb.select(x_zero, xb, zero)?;
    let b = fb.select(r_zero, z, rb)?;
    fb.convert(ConvOp::Bitcast, b, ft)
}

/// The classes of `powr`'s operands `x` and `y` (floats of form `f`, a `float` or a `half` in its
/// 32-bit container) that its special cases read, each told once and shared by every rule that
/// reads it: `y`'s bits and each operand's magnitude's, whether it is a zero by the device's comparison
/// ([`is_zero`]: an `f32` denormal is one where the device reads it so), whether it is an infinity
/// of either sign and whether `x` is one, the last two by integer comparisons of the bits, which
/// no float mode folds.
struct PowrOperands {
    x_magnitude: Value,
    x_zero: Value,
    x_inf: Value,
    x_one: Value,
    y_bits: Value,
    y_magnitude: Value,
    y_zero: Value,
    y_inf: Value,
}

impl PowrOperands {
    fn of(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Self> {
        let u = u32_form(f);
        let ut = ty(fb, u)?;
        let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
        let inf = splat(fb, u, 0x7F80_0000)?;
        let class = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<[Value; 4]> {
            let b = fb.convert(ConvOp::Bitcast, v, ut)?;
            let m = bin(fb, BinOp::BitwiseAnd, b, magnitude)?;
            let zero = is_zero(fb, f, v)?;
            let is_inf = bin(fb, BinOp::IEqual, m, inf)?;
            Ok([b, m, zero, is_inf])
        };
        let [x_bits, x_magnitude, x_zero, x_inf] = class(fb, x)?;
        let [y_bits, y_magnitude, y_zero, y_inf] = class(fb, y)?;
        let one = splat(fb, u, u64::from(1f32.to_bits()))?;
        let x_one = bin(fb, BinOp::IEqual, x_bits, one)?;
        Ok(Self {
            x_magnitude,
            x_zero,
            x_inf,
            x_one,
            y_bits,
            y_magnitude,
            y_zero,
            y_inf,
        })
    }

    /// Whether `log2(x)` is infinite: `x` a zero to Metal or an infinity.
    fn log_infinite(&self, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        bin(fb, BinOp::LogicalOr, self.x_zero, self.x_inf)
    }

    /// Whether neither operand is a NaN: the larger magnitude's bits are an infinity's or less.
    fn numbers(&self, fb: &mut FunctionBuilder<'_>, f: Form) -> BResult<Value> {
        let larger = glsl(fb, GlslOp::UMax, &[self.x_magnitude, self.y_magnitude])?;
        let inf = splat(fb, u32_form(f), 0x7F80_0000)?;
        bin(fb, BinOp::ULessThanEqual, larger, inf)
    }

    /// Metal's NaN where `exp2(y · log2(x))` is one for operands that are not NaNs ([`powr_nan`]),
    /// `negative` where `x` is below zero as the mode reads it: `0 · inf` (`y` a zero, `log2(x)`
    /// infinite) and `inf · 0` (`y` infinite, `x` one).
    fn nan(
        &self,
        fb: &mut FunctionBuilder<'_>,
        negative: Value,
        log_infinite: Value,
    ) -> BResult<Value> {
        let zero_times_inf = bin(fb, BinOp::LogicalAnd, self.y_zero, log_infinite)?;
        let inf_times_zero = bin(fb, BinOp::LogicalAnd, self.y_inf, self.x_one)?;
        let nan = bin(fb, BinOp::LogicalOr, negative, zero_times_inf)?;
        bin(fb, BinOp::LogicalOr, nan, inf_times_zero)
    }

    /// IEEE's `exp2` of an infinite `y · log2(x)` (`x` not negative where it matters): `+0` where
    /// the product is `-inf`, `+inf` where it is `+inf`, by the signs of `log2(x)` (below zero
    /// where `|x|` is below one, a zero included) and of `y` (its sign bit).
    fn infinite_answer(&self, fb: &mut FunctionBuilder<'_>, f: Form) -> BResult<Value> {
        let one = splat(fb, u32_form(f), u64::from(1f32.to_bits()))?;
        let l_negative = bin(fb, BinOp::ULessThan, self.x_magnitude, one)?;
        let y_negative = bin(fb, BinOp::INotEqual, self.y_bits, self.y_magnitude)?;
        let negative = bin(fb, BinOp::LogicalNotEqual, l_negative, y_negative)?;
        let (zero, inf) = (wide_splat(fb, f, 0.0)?, wide_splat(fb, f, f64::INFINITY)?);
        fb.select(negative, zero, inf)
    }
}

/// `powr(x, y)`'s result `r` (floats of form `f`) with a quiet NaN where MSL's `powr`, `exp2(y *
/// log2(x))` for `x >= 0`, is a NaN for operands that are not NaNs: `x` below zero (`-inf`
/// included), `y` zero with `x` a zero or an infinity (`0 * inf`), and `y` infinite with `x` one
/// (`inf * 0`). Devices answer `0` or `1` there (`GLSL.std.450 Pow` leaves `x < 0` and `x = 0,
/// y <= 0` undefined), so the NaN is a capability the device lacks, not an answer it gives
/// otherwise. The operands are classed once ([`PowrOperands`]), by the device's comparisons: an
/// `f32` denormal is a zero, or below zero, as the device reads it (`PrecisionNote::DeviceDenormals`
/// where it is not measured to read it as Metal does). `float` and `half`; `bfloat` keeps `r`
/// (not measured).
pub(super) fn powr_nan(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    let negative = below_zero(fb, f, x)?;
    let log_infinite = c.log_infinite(fb)?;
    let nan = c.nan(fb, negative, log_infinite)?;
    let k = fsplat(fb, f, f64::NAN)?;
    fb.select(nan, k, r)
}

/// Whether lanes of `x` (floats of form `f`) are below zero by the device's comparison.
fn below_zero(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let z = fsplat(fb, f, 0.0)?;
    bin(fb, BinOp::FOrdLessThan, x, z)
}

/// Metal's `fast::powr(x, y)` around `r`, the composition's `exp2(y · log2(x))`
/// ([`exp2_of_product`]): [`powr_nan`]'s NaN, over [`infinite_product`]'s IEEE value, both on one
/// classification of the operands ([`PowrOperands`]). The infinite product is selected wherever `log2(x)` or `y` is
/// infinite and neither operand is a NaN: the lanes of those where the product is `0 · inf`, and
/// those of a negative `x`, are the NaN rule's, which overrides them. `float` and `half`; `bfloat`
/// keeps `r` (not measured).
pub(super) fn fast_powr_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    let negative = below_zero(fb, f, x)?;
    let log_infinite = c.log_infinite(fb)?;
    let nan = c.nan(fb, negative, log_infinite)?;
    let infinite = bin(fb, BinOp::LogicalOr, log_infinite, c.y_inf)?;
    let numbers = c.numbers(fb, f)?;
    let infinite = bin(fb, BinOp::LogicalAnd, infinite, numbers)?;
    let answer = c.infinite_answer(fb, f)?;
    let r = fb.select(infinite, answer, r)?;
    let k = fsplat(fb, f, f64::NAN)?;
    fb.select(nan, k, r)
}

/// Metal's `fract(x)`: the device's `Fract`, rounded once to a narrow format, then bounded below
/// one, as Metal defines it (`fmin(x - floor(x), 0x1.fffffep-1)`): the largest value below one of
/// the format wherever the difference reaches one (`-1e-10` gives `1 - 2^-24`, not `1`). The
/// bound is an ordered comparison, so a NaN `Fract` is kept, the device's NaN. An infinity, a NaN
/// and an `f32` denormal answer as the device's `Fract` does (`PrecisionNote::NativeTranscendentals`
/// off Metal's own functions).
fn fract_below_one(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    // `floor(x)` of a narrow value is a value of the format: the difference takes the device's
    // rounding.
    let d = glsl(fb, GlslOp::Fract, &[x])?;
    let d = narrow::round(fb, f, d)?;
    let one = fsplat(fb, f, 1.0)?;
    let reaches = bin(fb, BinOp::FOrdGreaterThanEqual, d, one)?;
    let below_one = splat(fb, f, below_one_bits(f))?;
    fb.select(reaches, below_one, d)
}

/// Metal's `median3(a, b, c)`: `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and
/// `NMax`, so the result is one of the operands, picked as the device's `NMin` and `NMax` pick
/// (`PrecisionNote::NativeMinMax` on every device: of two NaNs, and of an `f32` denormal, the
/// device's pick).
fn median3(fb: &mut FunctionBuilder<'_>, [a, b, c]: [Value; 3]) -> BResult<Value> {
    let lo = glsl(fb, GlslOp::NMin, &[a, b])?;
    let hi = glsl(fb, GlslOp::NMax, &[a, b])?;
    let m = glsl(fb, GlslOp::NMax, &[c, lo])?;
    glsl(fb, GlslOp::NMin, &[m, hi])
}

/// Whether Metal's `fmin` and `fmax` family is measured on lanes of format `fmt`: `float`
/// (`nan_minmax` without fast math, `nan_minmax_fast` with every flag, and both at zeros and
/// denormals) and `half` (`nan_half`, `nan_half_fast`), where the device's `NMin`, `NMax` and
/// `NClamp`, which every format takes, record `PrecisionNote::NativeMinMax` unless measured to be
/// Metal's. On `bfloat` and `double` Metal's is not measured.
pub fn metal_picks(fmt: Option<FloatFmt>) -> bool {
    matches!(fmt, Some(FloatFmt::F32 | FloatFmt::Half))
}

/// A division of two values of one form: the device's (a unit test substitutes a reference).
pub(super) type DivideOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// A fused multiply-add of three floats (`super::fma::emit`; a unit test substitutes a
/// reference).
pub(super) type FmaOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value>;

/// Metal's `fast::fmod(x, y)` on `float`: `fma(-trunc(q), y, x)` of the quotient `q = x / y`
/// (`divide`, the device's division), the fused multiply-add `fma`. Every lane of probe case
/// 08bc6611 (1,024, specials among them) is this expression on Metal's own quotient; the device's
/// quotient is a near-equivalent of it (`PrecisionNote::DeviceArithmetic`), so `fmod(1e10, 3)` is
/// Metal's `-512` exactly where the device's quotient is Metal's. Metal reads a denormal `x` or `y`
/// as the zero of its sign and answers no denormal, with and without fast math (every pair of 28
/// patterns); the device's are its own (`PrecisionNote::DeviceDenormals`).
pub(super) fn fast_fmod(
    fb: &mut FunctionBuilder<'_>,
    x: Value,
    y: Value,
    divide: DivideOf<'_>,
    fma: FmaOf<'_>,
) -> BResult<Value> {
    let q = divide(fb, x, y)?;
    let t = glsl(fb, GlslOp::Trunc, &[q])?;
    let n = fb.unary(UnOp::FNegate, t)?;
    fma(fb, &[n, y, x])
}

/// Which of Metal's pi-scaled functions [`pi_scaled`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PiScaled {
    Sin,
    Cos,
    Tan,
}

/// `sin(π r)` and `cos(π r)` of the exact residue `r` in `[-1/4, 1/4]` (`GLSL.std.450`
/// `Sin`/`Cos` of `float(π) r`; a unit test substitutes a stand-in).
pub(super) type SinCosOf<'a> =
    &'a dyn Fn(&mut FunctionBuilder<'_>, Value) -> BResult<(Value, Value)>;

/// The quotient of two values, for [`pi_scaled`]'s `tanpi`.
pub(super) type QuotientOf<'a> =
    &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// Metal `sinpi`/`cospi`/`tanpi` of `x` (form `f`, computed in its `f32` container) by an exact
/// period reduction (the old engine's device-measured lowering), not `sin(float(π) * x)`:
/// `float(π)` is `8.74e-8` above π, so the pre-multiply misses every zero crossing (`sinpi(1)`
/// came back `-8.74e-8`, `cospi(0.5)` `-4.37e-8`, and `tan(float(π) * 0.5)` has no pole at all),
/// and it hands `Sin`/`Cos` a large argument (`cospi(1e7)` came back 0). Both functions have
/// period 2, a power of two, so every step but the last multiply is exact:
///
/// ```text
/// x2 = x - 2 roundEven(x / 2)     in [-1, 1]
/// n  = roundEven(2 x2)             in {-2, -1, 0, 1, 2}
/// r  = x2 - n / 2                  in [-1/4, 1/4]
/// sinpi(x) = [ sin, cos, -sin, -cos][n mod 4](π r)
/// cospi(x) = [ cos, -sin, -cos, sin][n mod 4](π r)
/// tanpi(x) = sinpi(x) / cospi(x), and ±1 where |r| = 1/4
/// ```
///
/// This is the form where no routine answers (a `bfloat`; on `float` and `half` the routines are
/// Metal's own forms). The sine and cosine of `π r` are `sin_cos`'s of the residue, the device's
/// `Sin` and `Cos` of `float(π) r` (a unit test substitutes a stand-in). `n` is selected by three
/// float compares (never converted, so a large `x` cannot overflow an integer). At a multiple of
/// one half `r` is exactly `+0`, and the sine and cosine there are selected as `+0` and `1` rather
/// than taken from the device (Vulkan bounds `Sin` and `Cos` only to `2^-11` absolute, so a
/// device's `Sin(+0)` need not be `+0`), so the result is an exact `0` or `±1`, and `tanpi` divides
/// an exact `±1` by the `+0` below into the correctly signed infinity. The quotient is
/// `quotient`'s, the device's `OpFDiv`. For a normal `x` no operand of it is a denormal: a nonzero
/// `sinpi` is at least `π 2^-126` in magnitude (`r` is `x` below 1, and a multiple of `2^-23`
/// above); a denormal `x` reaches the first multiply and difference, which read it as the device
/// does: Metal's reading where the profile's functions are Metal's (the deviation, or flushed
/// before), the device's under native device maths (`PrecisionNote::DeviceDenormals`). At a
/// quarter turn (`r = ±1/4`, exact like every step before the multiply) the two arms are the sine
/// and cosine of `π/4`, equal in magnitude, so `tanpi` is exactly `±1` with the quotient's sign,
/// whatever the device's `Sin` and `Cos` round them to (on lavapipe they differ by one ulp, and the
/// quotient by one). With the zeros and the poles these are the only rational values `tanpi` takes
/// at a rational argument (Niven's theorem), and `sinpi` and `cospi` take a rational value at a
/// dyadic `r` in `[-1/4, 1/4]` only at `r = 0`, so all three are exact on any device wherever their
/// value is a float. Metal's signed zeros are measured, not derived: `sinpi` is `-0` at odd
/// integers (the `-sin` arm negates `+0`), and `cospi` is `+0` at every half-integer, including
/// those whose arm is `-sin`, so that arm selects `+0` when `r` is 0 (a negation, or `0 - sin`,
/// which Metal's compiler folds into one, would give `-0`). A non-finite `x` (tested on its bits,
/// which no fast-math mode folds) gives the quiet NaN `0x7fc00000`, as Metal does for `±inf` and
/// NaN.
pub(super) fn pi_scaled(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    which: PiScaled,
    x: Value,
    sin_cos: SinCosOf<'_>,
    quotient: QuotientOf<'_>,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (half, two, zero) = (c(fb, 0.5)?, c(fb, 2.0)?, c(fb, 0.0)?);
    let (one, minus_one) = (c(fb, 1.0)?, c(fb, -1.0)?);
    // x2 = x - 2 roundEven(x / 2): scalings by powers of two, and a difference below 1.
    let scaled = bin(fb, BinOp::FMul, x, half)?;
    let whole = glsl(fb, GlslOp::RoundEven, &[scaled])?;
    let doubled = bin(fb, BinOp::FMul, whole, two)?;
    let reduced = bin(fb, BinOp::FSub, x, doubled)?;
    let twice = bin(fb, BinOp::FMul, reduced, two)?;
    let quadrant = glsl(fb, GlslOp::RoundEven, &[twice])?;
    let offset = bin(fb, BinOp::FMul, quadrant, half)?;
    let residue = bin(fb, BinOp::FSub, reduced, offset)?;
    let (sine, cosine) = sin_cos(fb, residue)?;
    // On an axis (`r = +0`) the arms are exact whatever the device's `Sin` and `Cos` of `+0` are.
    let axis = bin(fb, BinOp::FOrdEqual, residue, zero)?;
    let sine = fb.select(axis, zero, sine)?;
    let cosine = fb.select(axis, one, cosine)?;
    let at_zero = bin(fb, BinOp::FOrdEqual, quadrant, zero)?;
    let at_one = bin(fb, BinOp::FOrdEqual, quadrant, one)?;
    let at_minus_one = bin(fb, BinOp::FOrdEqual, quadrant, minus_one)?;
    // The arms for n = 0, 1, -1, and ±2 (which select the same value).
    let pick = |fb: &mut FunctionBuilder<'_>, arms: [Value; 4]| -> BResult<Value> {
        let inner = fb.select(at_minus_one, arms[2], arms[3])?;
        let middle = fb.select(at_one, arms[1], inner)?;
        fb.select(at_zero, arms[0], middle)
    };
    let sin_arms = |fb: &mut FunctionBuilder<'_>| -> BResult<[Value; 4]> {
        let ns = fb.unary(UnOp::FNegate, sine)?;
        let nc = fb.unary(UnOp::FNegate, cosine)?;
        Ok([sine, cosine, nc, ns])
    };
    let cos_arms = |fb: &mut FunctionBuilder<'_>| -> BResult<[Value; 4]> {
        let ns = fb.unary(UnOp::FNegate, sine)?;
        let ns = fb.select(axis, zero, ns)?;
        let nc = fb.unary(UnOp::FNegate, cosine)?;
        Ok([cosine, ns, sine, nc])
    };
    let r = match which {
        PiScaled::Sin => {
            let arms = sin_arms(fb)?;
            pick(fb, arms)?
        }
        PiScaled::Cos => {
            let arms = cos_arms(fb)?;
            pick(fb, arms)?
        }
        PiScaled::Tan => {
            let arms = sin_arms(fb)?;
            let num = pick(fb, arms)?;
            let arms = cos_arms(fb)?;
            let den = pick(fb, arms)?;
            let quotient = quotient(fb, num, den)?;
            // A quarter turn: both arms are nonzero, so the quotient's sign bit is the answer's.
            let quarter = c(fb, 0.25)?;
            let magnitude = glsl(fb, GlslOp::FAbs, &[residue])?;
            let at_quarter = bin(fb, BinOp::FOrdEqual, magnitude, quarter)?;
            let unit = copysign(fb, f, one, quotient)?;
            fb.select(at_quarter, unit, quotient)?
        }
    };
    // Finite: the exponent field is not all ones.
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
    let finite = bin(fb, BinOp::INotEqual, e, exp)?;
    let nan = c(fb, f64::NAN)?;
    fb.select(finite, r, nan)
}

/// The words of `infinite_product` on a scalar: about 23 instructions and their constants.
const INFINITE_PRODUCT_WORDS: u32 = 128;

/// `exp2(y · log2(x))` with the product rounded, of the device's `Exp2` and `Log2`: Metal's
/// `fast::powr(x, y)`, and its `fast::pow`'s magnitude of `|x|`.
/// Every lane of probe case 08bc6611 (1,024, specials among them) gives Metal's `fast::powr` as
/// this expression of Metal's own `exp2` and `log2`, and its `fast::pow` at a base that is not
/// negative. Where the product is infinite the callers select IEEE's answer by the operands' bits
/// ([`infinite_product`], [`fast_powr_special`]), so a device's arithmetic on the infinite
/// intermediates cannot move it.
fn exp2_of_product(fb: &mut FunctionBuilder<'_>, y: Value, x: Value) -> BResult<Value> {
    let l = glsl(fb, GlslOp::Log2, &[x])?;
    let p = bin(fb, BinOp::FMul, y, l)?;
    glsl(fb, GlslOp::Exp2, &[p])
}

/// `r`, the value of `exp2(y · log2(x))` (floats of form `f`, `x` not negative where it
/// matters), with the value IEEE arithmetic gives it where the product is infinite, told by the
/// operands' bits ([`PowrOperands`]): `log2(x)` is `-inf` at a base Metal reads as zero (an `f32`
/// denormal included, as the fast `log2` reads it) and `+inf` at `+inf`, and the product of
/// either with a nonzero `y`, or of an infinite `y` with a `log2(x)` that is not zero, is `+inf`
/// (answer `+inf`) or `-inf` (answer `+0`) by their signs. Where the product is a NaN (a zero
/// times an infinity) or an operand is one, `r` is kept for the callers' NaN rules. A fast-math
/// entry declares no `SignedZeroInfNanPreserve`, so a device may compute those infinite
/// intermediates otherwise: the RTX 5080 answered `1` at every one (case f6524207's `fast::pow`
/// at infinite bases and `fast::powr` at zero and denormal ones, case 32931370's `half` `powr`),
/// where Metal and the Mesa venues answer IEEE's value. `fast::powr` reads the
/// same classes for its NaN rule as well ([`fast_powr_special`]).
pub(super) fn infinite_product(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    // `log2(x)` infinite; then a nonzero `y` makes the product infinite.
    let log_infinite = c.log_infinite(fb)?;
    let y_nonzero = fb.unary(UnOp::LogicalNot, c.y_zero)?;
    let by_base = bin(fb, BinOp::LogicalAnd, log_infinite, y_nonzero)?;
    // An infinite `y` times a `log2(x)` that is not zero.
    let x_not_one = fb.unary(UnOp::LogicalNot, c.x_one)?;
    let by_exponent = bin(fb, BinOp::LogicalAnd, c.y_inf, x_not_one)?;
    let infinite = bin(fb, BinOp::LogicalOr, by_base, by_exponent)?;
    let numbers = c.numbers(fb, f)?;
    let infinite = bin(fb, BinOp::LogicalAnd, infinite, numbers)?;
    let answer = c.infinite_answer(fb, f)?;
    fb.select(infinite, answer, r)
}

/// Metal's precise `exp10(x)`, which no instruction is, from the device's `exp2` (floats of form
/// `f` in a 32-bit container, the constants the container's): `E = exp2(p)` of the rounded
/// product `p = x c_hi`, `c_hi + c_lo = log2(10)`, corrected by the product's error
/// `d = fma(x, c_lo, fma(x, c_hi, -p))` as `fma(E, d ln 2, E)`, and `E` itself where it is `+inf`
/// or `+0` (where `x` is infinite `d` is a NaN). Every lane of case 7b283404 on the M3 and the M4
/// is this of Metal's `exp2`. Each step is its own: no device fuses or regroups it.
fn exp10(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let (hi, lo) = libm::split(std::f64::consts::LOG2_10);
    let c_hi = wide_splat(fb, f, f64::from(hi))?;
    let c_lo = wide_splat(fb, f, f64::from(lo))?;
    let p = fb.binary_with(BinOp::FMul, x, c_hi, exact)?;
    let minus_p = fb.unary(UnOp::FNegate, p)?;
    let e = glsl(fb, GlslOp::Fma, &[x, c_hi, minus_p])?;
    let d = glsl(fb, GlslOp::Fma, &[x, c_lo, e])?;
    let big = glsl(fb, GlslOp::Exp2, &[p])?;
    let ln2 = wide_splat(fb, f, f64::from(std::f64::consts::LN_2 as f32))?;
    let t = fb.binary_with(BinOp::FMul, d, ln2, exact)?;
    let r = glsl(fb, GlslOp::Fma, &[big, t, big])?;
    let inf = has_bits(fb, f, big, 0x7F80_0000)?;
    let zero = has_bits(fb, f, big, 0)?;
    let ends = bin(fb, BinOp::LogicalOr, inf, zero)?;
    fb.select(ends, big, r)
}

/// Metal's precise `log10(x)`, which no instruction is, from the device's `log2` (floats of form
/// `f` in a 32-bit container, the constants the container's): the `log2` times `RN(log10 2)`, and
/// strictly inside [`libm::LOG10_NEAR`]'s range next to 1 its polynomial `RN(t fma(p(t), t,
/// lead))` of `t = x - 1`, by fused steps. Each step is its own: no device fuses or regroups it.
fn log10(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let (near, c) = (&libm::LOG10_NEAR, libm::LOG10_2_BITS);
    let exact = Contraction::Forbidden;
    let l = glsl(fb, GlslOp::Log2, &[x])?;
    let k = wide_splat(fb, f, f64::from(f32::from_bits(c)))?;
    let far = fb.binary_with(BinOp::FMul, l, k, exact)?;
    let one = wide_splat(fb, f, 1.0)?;
    let t = fb.binary_with(BinOp::FSub, x, one, exact)?;
    let (last, rest) = near
        .p
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut p = wide_splat(fb, f, f64::from(f32::from_bits(*last)))?;
    for c in rest.iter().rev() {
        let c = wide_splat(fb, f, f64::from(f32::from_bits(*c)))?;
        p = glsl(fb, GlslOp::Fma, &[p, t, c])?;
    }
    let lead = wide_splat(fb, f, f64::from(f32::from_bits(near.lead)))?;
    let q = glsl(fb, GlslOp::Fma, &[p, t, lead])?;
    let close = fb.binary_with(BinOp::FMul, t, q, exact)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let first = splat(fb, u, u64::from(near.low + 1))?;
    let offset = bin(fb, BinOp::ISub, b, first)?;
    let span = splat(fb, u, u64::from(near.end - near.low - 1))?;
    let inside = bin(fb, BinOp::ULessThan, offset, span)?;
    fb.select(inside, close, far)
}

/// The C99 Annex F (F.9.4.4) special cases of `pow(x, y)` around `magnitude`, the value
/// `GLSL.std.450 Pow(|x|, y)` gives, which `Pow` leaves undefined (a negative base, a zero base to
/// `y <= 0`) or whose sign it cannot give (a negative base to an odd integer). Later selects win:
///
/// - `pow(±0, y)` is `+inf` for `y < 0` and `+0` for `y > 0` (signed for odd integral `y`);
/// - `pow(-1, y)` is `1` for integral or infinite `y` (odd integers negate below);
/// - a negative base to an odd integer negates (the sign bit, so `-0` and `-inf` count); a
///   negative finite nonzero base to a non-integral finite exponent is a quiet NaN, which the
///   device's `Pow` does not define;
/// - a NaN operand gives a NaN, overriding the zero-base and `-1` results (C99 makes
///   `pow(±0, NaN)` and `pow(-1, NaN)` NaN, and `Pow` is not bound to propagate one): the device's
///   own NaN of the operands ([`device_nan`]), its sign and payload the device's;
/// - `pow(+1, y)` is `1` for every `y`, even a NaN;
/// - `pow(x, ±0)` is `1` for every `x`, even a NaN.
///
/// Zeros and signs are the device's comparisons: an `f32` denormal is a zero, or below zero, as
/// the device reads it (`PrecisionNote::DeviceDenormals` where it is not measured to read it as
/// Metal does). Every other operand, `pow(x, 1)` included, is the device's `Pow`.
pub(super) fn pow_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    magnitude: Value,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (zero, one, half) = (c(fb, 0.0)?, c(fb, 1.0)?, c(fb, 0.5)?);
    let (inf, ninf, nan) = (
        c(fb, f64::INFINITY)?,
        c(fb, f64::NEG_INFINITY)?,
        c(fb, f64::NAN)?,
    );
    // y integral; y odd (an odd integer halves to a fraction of exactly one half).
    let ty_ = glsl(fb, GlslOp::Trunc, &[y])?;
    let yint = bin(fb, BinOp::FOrdEqual, ty_, y)?;
    let hy = bin(fb, BinOp::FMul, y, half)?;
    let fr = glsl(fb, GlslOp::Fract, &[hy])?;
    let yodd = bin(fb, BinOp::FOrdEqual, fr, half)?;
    // Zero base.
    let xz = is_zero(fb, f, x)?;
    let yneg = bin(fb, BinOp::FOrdLessThan, y, zero)?;
    let zb = fb.select(yneg, inf, zero)?;
    let base = fb.select(xz, zb, magnitude)?;
    // A base of magnitude one.
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let x1 = bin(fb, BinOp::FOrdEqual, ax, one)?;
    let base = fb.select(x1, one, base)?;
    // The sign bit of x (so -0 and -inf count).
    let sneg = sign_set(fb, f, x)?;
    let flip = bin(fb, BinOp::LogicalAnd, sneg, yodd)?;
    let nb = fb.unary(UnOp::FNegate, base)?;
    let r = fb.select(flip, nb, base)?;
    // A negative finite nonzero base to a non-integral exponent.
    let xneg = bin(fb, BinOp::FOrdLessThan, x, zero)?;
    let xfin = bin(fb, BinOp::FOrdGreaterThan, x, ninf)?;
    let fractional = fb.unary(UnOp::LogicalNot, yint)?;
    let bad = bin(fb, BinOp::LogicalAnd, xneg, xfin)?;
    let bad = bin(fb, BinOp::LogicalAnd, bad, fractional)?;
    let r = fb.select(bad, nan, r)?;
    // NaN operands give the device's NaN (after the zero-base and -1 cases, which they
    // override).
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let any_nan = bin(fb, BinOp::LogicalOr, xnan, ynan)?;
    let operand_nan = device_nan(fb, x, y)?;
    let r = fb.select(any_nan, operand_nan, r)?;
    let xp1 = bin(fb, BinOp::FOrdEqual, x, one)?;
    let r = fb.select(xp1, one, r)?;
    let y0 = is_zero(fb, f, y)?;
    fb.select(y0, one, r)
}

/// Whether lanes of `v` (floats of form `f`) are zeros by the device's comparison, either sign:
/// an `f32` denormal is one where the device reads it as zero (`PrecisionNote::DeviceDenormals`
/// where it is not measured to read it as Metal does).
fn is_zero(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    let zero = wide_splat(fb, f, 0.0)?;
    bin(fb, BinOp::FOrdEqual, v, zero)
}

/// A NaN of the device's arithmetic from operands `a` and `b`, at least one of them a NaN: their
/// sum, whose NaN propagation (sign and payload) is the device's own
/// (`PrecisionNote::NativeNanEncoding` where it is not measured to be Metal's).
fn device_nan(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary_with(BinOp::FAdd, a, b, Contraction::Forbidden)
}

/// Whether the sign bit of `v` (floats of form `f`, held in an `f32` container) is set, so `-0`
/// and `-inf` count: the sign of a zero, which no comparison tells.
fn sign_set(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let k31 = splat(fb, u, 31)?;
    let s = bin(fb, BinOp::ShiftRightLogical, bits, k31)?;
    let uz = splat(fb, u, 0)?;
    bin(fb, BinOp::INotEqual, s, uz)
}

/// The C99 Annex F (F.9.1.4) special cases of `atan2(y, x)` around `generic`, the value
/// `GLSL.std.450 Atan2(y, x)` gives, as Metal answers them (probe case 6802ab13, every lane of
/// sixteen operands squared on precise `float` and `half`, and A1 case 58e56ca3). Vulkan leaves
/// `Atan2` undefined where both operands are zero and bounds it only to 4096 ulp elsewhere, so
/// a zero's sign and the axes' angles are the device's (llvmpipe answers `atan2(-0, 1) = +0` and
/// `atan2(inf, inf)` 55 ulp below `π/4`). Each special answer is a constant with `y`'s sign:
///
/// - a zero `y` gives `±0` toward a positive `x` (`+0` included) and `±π` toward a negative one
///   (`-0` and `-inf` included);
/// - an infinite `y` gives `±π/4` toward `+inf`, `±3π/4` toward `-inf`, and `±π/2` otherwise;
/// - a finite nonzero `y` gives `±π/2` over a zero `x`, `±0` toward `+inf`, and `±π` toward
///   `-inf`;
/// - a NaN operand gives a NaN, the device's own of the operands ([`device_nan`]): `Atan2` is not
///   bound to propagate one.
///
/// The constants are `π`, `π/2`, `π/4`, and `3π/4` correctly rounded to the format (Metal's
/// answers: `0x40490fdb`, `0x3fc90fdb`, `0x3f490fdb`, `0x4016cbe4`; `half` `0x4248`, `0x3e48`,
/// `0x3a48`, `0x40b6`, which the final rounding gives from the `f32` constants). A zero is told by
/// the device's comparison, so an `f32` denormal operand is a zero where the device reads it so
/// (`PrecisionNote::DeviceDenormals` where it is not measured to read it as Metal does), and the
/// generic answer, a denormal among them, is the device's `Atan2`.
pub(super) fn atan2_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    generic: Value,
) -> BResult<Value> {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (zero, half_pi, pi) = (c(fb, 0.0)?, c(fb, FRAC_PI_2)?, c(fb, PI)?);
    let (quarter, three_quarters) = (c(fb, FRAC_PI_4)?, c(fb, 3.0 * FRAC_PI_4)?);
    let (yz, xz) = (is_zero(fb, f, y)?, is_zero(fb, f, x)?);
    let (yi, xi) = (fb.unary(UnOp::IsInf, y)?, fb.unary(UnOp::IsInf, x)?);
    let xneg = sign_set(fb, f, x)?;
    // The answer on the x axis: 0 toward +x, π toward -x.
    let axis = fb.select(xneg, pi, zero)?;
    // A zero x (y finite nonzero): π/2; otherwise (an infinite x) the axis.
    let k = fb.select(xz, half_pi, axis)?;
    let diagonal = fb.select(xneg, three_quarters, quarter)?;
    let at_inf = fb.select(xi, diagonal, half_pi)?;
    let k = fb.select(yi, at_inf, k)?;
    let k = fb.select(yz, axis, k)?;
    let y_special = bin(fb, BinOp::LogicalOr, yz, yi)?;
    let x_special = bin(fb, BinOp::LogicalOr, xz, xi)?;
    let special = bin(fb, BinOp::LogicalOr, y_special, x_special)?;
    let signed = copysign(fb, f, k, y)?;
    let r = fb.select(special, signed, generic)?;
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let any_nan = bin(fb, BinOp::LogicalOr, ynan, xnan)?;
    let nan = device_nan(fb, y, x)?;
    fb.select(any_nan, nan, r)
}

/// The arc tangent of a quotient (`GLSL.std.450 Atan`; a unit test substitutes a stand-in
/// because the interpreter does not model it).
pub(super) type AtanOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value) -> BResult<Value>;

/// Metal `fast::atan2(y, x)` on `float` (probe case c11eb8cb, every lane of sixteen operands
/// squared, in both of its compiles): `atan(q)` of the quotient `q = y / x`, turned by `π` when
/// `x` is below zero (by `-π` when `y` is, too), with the formula's special values, which the
/// device's `Atan` of an infinite or NaN quotient leaves undefined:
///
/// - the quotient's sign is the exclusive or of the operands' sign bits, so a zero quotient
///   gives that signed zero and an infinite one (a zero `x`, an infinite `y`, or an overflow)
///   `±π/2`;
/// - `0 / 0` and `inf / inf` give a quiet NaN, and a NaN operand the device's own NaN of the
///   operands ([`device_nan`]);
/// - the turn reads `x < 0` and `y >= 0` as comparisons: `fast::atan2(-0, -1)` is `+π` (where
///   `atan2` answers `-π`), and `fast::atan2(1, -0)` is `-π/2`.
///
/// Zeros are the device's comparisons, so an `f32` denormal operand is a zero where the device
/// reads it so (`denormal::coverage`), and a denormal quotient is the device's `Atan` of it.
///
/// Beyond `2^126` the quotient follows the call's fast-math flags. A call that carries every
/// flag (case c11eb8cb, compiled with fast math) divides over the whole range
/// (`fast::atan2(FLT_MAX, FLT_MAX)` is `atan(1)`): the division is scaled by `1/4` when `|x|` is
/// at least `2^126`, an exact scaling that keeps the divisor in the range where Metal's fast
/// division and Vulkan's `OpFDiv` bound their error. Any other call (`reciprocal_flushed`; case
/// 6802ab13, compiled without fast math, measured with no flag) multiplies `y` by `x`'s
/// reciprocal flushed to zero beyond `2^126` (the reciprocal is an `f32` denormal there), so the
/// quotient is the signed zero, or NaN for an infinite `y` (`fast::atan2(FLT_MAX, FLT_MAX)` is
/// `+0`, `fast::atan2(inf, FLT_MAX)` is NaN). A call with only some of the flags is not
/// measured; it takes the second answer, which differs only where the divisor leaves the range
/// MSL bounds the fast division in.
///
/// Every finite normal pair of our probe (16.7 million, the M4) is `atan(q)` of Metal's fast
/// `atan` and its fast quotient `q = y · rcp(x)`, and `RN(atan(q) ± RN(π))` toward a negative
/// `x`: `divide` gives `q` (Metal's reciprocal's routine, or the device's quotient where it is
/// Metal's), `atan` the arc tangent (Metal's fast `atan`'s routine, or the device's where it is
/// Metal's). Every special value above is exact.
pub(super) fn fast_atan2(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    reciprocal_flushed: bool,
    atan: AtanOf<'_>,
    divide: DivideOf<'_>,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let zero = c(fb, 0.0)?;
    let (pi, minus_pi) = (c(fb, std::f64::consts::PI)?, c(fb, -std::f64::consts::PI)?);
    let half_pi = c(fb, std::f64::consts::FRAC_PI_2)?;
    let nan = c(fb, f64::NAN)?;
    let (yz, xz) = (is_zero(fb, f, y)?, is_zero(fb, f, x)?);
    let (yi, xi) = (fb.unary(UnOp::IsInf, y)?, fb.unary(UnOp::IsInf, x)?);
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let limit = c(fb, 2f64.powi(126))?;
    // The quotient, and where the reciprocal is flushed (`None` when it is not).
    let (q, beyond) = if reciprocal_flushed {
        let q = divide(fb, y, x)?;
        (q, Some(bin(fb, BinOp::FOrdGreaterThan, ax, limit)?))
    } else {
        // Both operands scaled by 1/4 when the divisor is at least 2^126.
        let (one, quarter) = (c(fb, 1.0)?, c(fb, 0.25)?);
        let big = bin(fb, BinOp::FOrdGreaterThanEqual, ax, limit)?;
        let s = fb.select(big, quarter, one)?;
        let ys = bin(fb, BinOp::FMul, y, s)?;
        let xs = bin(fb, BinOp::FMul, x, s)?;
        (divide(fb, ys, xs)?, None)
    };
    // The quotient's sign: the exclusive or of the operands' sign bits.
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let sb = bin(fb, BinOp::BitwiseXor, yb, xb)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let sb = bin(fb, BinOp::BitwiseAnd, sb, sign)?;
    let ft = ty(fb, f)?;
    let signed_zero = fb.convert(ConvOp::Bitcast, sb, ft)?;
    let signed_half_pi = copysign(fb, f, half_pi, signed_zero)?;
    // A zero quotient: a zero y or an infinite x.
    let qz = bin(fb, BinOp::LogicalOr, yz, xi)?;
    let qz = match beyond {
        Some(b) => bin(fb, BinOp::LogicalOr, qz, b)?,
        None => qz,
    };
    // An infinite quotient: a zero x, an infinite y, or a quotient that overflows.
    let q_inf = fb.unary(UnOp::IsInf, q)?;
    let qi = bin(fb, BinOp::LogicalOr, xz, yi)?;
    let qi = bin(fb, BinOp::LogicalOr, qi, q_inf)?;
    let a = atan(fb, q)?;
    let a = fb.select(qz, signed_zero, a)?;
    let a = fb.select(qi, signed_half_pi, a)?;
    // The turn: x below zero, by π when y is at least zero.
    let xneg = bin(fb, BinOp::FOrdLessThan, x, zero)?;
    let ynonneg = bin(fb, BinOp::FOrdGreaterThanEqual, y, zero)?;
    let turn = fb.select(ynonneg, pi, minus_pi)?;
    let turned = bin(fb, BinOp::FAdd, a, turn)?;
    let r = fb.select(xneg, turned, a)?;
    // 0 / 0 and inf / inf (or an infinite y times a flushed reciprocal); then NaN operands.
    let zeros = bin(fb, BinOp::LogicalAnd, yz, xz)?;
    let infinite_x = match beyond {
        Some(b) => bin(fb, BinOp::LogicalOr, xi, b)?,
        None => xi,
    };
    let infs = bin(fb, BinOp::LogicalAnd, yi, infinite_x)?;
    let undefined = bin(fb, BinOp::LogicalOr, zeros, infs)?;
    let r = fb.select(undefined, nan, r)?;
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let any_nan = bin(fb, BinOp::LogicalOr, ynan, xnan)?;
    let operand_nan = device_nan(fb, y, x)?;
    fb.select(any_nan, operand_nan, r)
}

/// The container bits of the largest value below 1.0 of `f`'s format.
fn below_one_bits(f: Form) -> u64 {
    let significand = match f.float_fmt() {
        Some(FloatFmt::F64) => 53,
        Some(FloatFmt::F32) => 24,
        Some(FloatFmt::Half) => 11,
        _ => 8,
    };
    f.const_bits(1.0 - 2f64.powi(-significand))
}

/// The (mantissa bits, exponent bias, all-ones exponent) of a float width.
pub(super) fn float_layout(bits: u8) -> (u32, i64, i64) {
    match bits {
        64 => (52, 1023, 2047),
        32 => (23, 127, 255),
        _ => (10, 15, 31),
    }
}

/// The bits of `2^k` in a `bits`-wide float (infinity when it overflows).
pub(super) fn pow2_bits(bits: u8, k: u32) -> u64 {
    let (m, bias, top) = float_layout(bits);
    let e = i64::from(k) + bias;
    if e >= top {
        (top as u64) << m
    } else {
        (e as u64) << m
    }
}

/// The value of the container float with bit pattern `b`: an `f64` when `bits` is 64, else an
/// `f32` (every narrow float is held in an `f32`).
pub(super) fn float_value(bits: u8, b: u64) -> f64 {
    if bits == 64 {
        f64::from_bits(b)
    } else {
        f64::from(f32::from_bits(b as u32))
    }
}

/// The bits of the largest finite float below `2^k` (`k >= 1`).
pub(super) fn below_pow2_bits(bits: u8, k: u32) -> u64 {
    let (m, bias, top) = float_layout(bits);
    let e = (i64::from(k) - 1 + bias).min(top - 1);
    ((e as u64) << m) | ((1u64 << m) - 1)
}

/// How `air.convert` converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvertKind {
    /// Float to float (`FConvert`, or the value itself at one width).
    Float,
    /// Integer (`signed` source) to float.
    ToFloat { signed: bool },
    /// Float to integer, saturating: NaN to 0, out-of-range to the destination's bound.
    ToInt { signed: bool },
    /// Integer (`signed` source) to integer.
    Int { signed: bool },
    /// An 8-bit float (carried in an `i8`) to a float: exact.
    FromFp8(narrow::Fp8Layout),
    /// An 8-bit float (carried in an `i8`) to an integer: its value as an `f32` (exact, as
    /// [`ConvertKind::FromFp8`]) converted as [`ConvertKind::ToInt`] (saturating, NaN to 0).
    Fp8ToInt {
        fp8: narrow::Fp8Layout,
        signed: bool,
    },
    /// A `float` or `half` to `bool`: true unless the value is a zero or an `f32` denormal, so a
    /// NaN is true. Metal's reading (A1 cases 099ba79b and 9a824047, with and without fast math) is
    /// C++'s `x != 0` with `float` denormals flushed; the exponent field of the `f32` container
    /// decides it, whatever the device's denormal mode (a `half` is held widened, so its denormals
    /// are normal `f32` values and true, as Metal answers).
    ToBool,
    /// An integer to `bool`: C++'s `x != 0`. MSL's `bool`-vector conversion of an integer vector is
    /// what the front end emits as this `air.convert`; a scalar conversion is the front end's own
    /// `icmp ne x, 0`. Signedness does not matter.
    IntToBool,
}

/// Emit `air.convert` of `x` from `from` to `to`, a float source's NaN told by `test` (the
/// entry's [`NanTest::for_width`] of the source's container).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    k: ConvertKind,
    from: Form,
    to: Form,
    x: Value,
    test: NanTest,
) -> BResult<Value> {
    use crate::lower::arith::ops::cast;
    match k {
        ConvertKind::FromFp8(l) => {
            let u = narrow::u32_form(from);
            let v = narrow::fp8_to_f32(fb, l, u, to, x)?;
            // Every 8-bit float value is a value of every wider format: no rounding.
            if to.float_fmt() == Some(FloatFmt::F64) {
                let t = ty(fb, to)?;
                fb.convert(ConvOp::FConvert, v, t)
            } else {
                Ok(v)
            }
        }
        ConvertKind::Fp8ToInt { fp8, signed } => {
            let f = from.with(Held::Float(FloatFmt::F32));
            let v = narrow::fp8_to_f32(fb, fp8, narrow::u32_form(from), f, x)?;
            convert(fb, ConvertKind::ToInt { signed }, f, to, v, test)
        }
        ConvertKind::Float => cast(fb, CastKind::FpResize, from, to, x),
        // A held integer is zero-extended in its container, so the container is zero exactly
        // when the value is.
        ConvertKind::IntToBool => {
            let zero = splat(fb, from, 0)?;
            bin(fb, BinOp::INotEqual, x, zero)
        }
        ConvertKind::ToBool => {
            let u = narrow::u32_form(from);
            let ut = ty(fb, u)?;
            let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
            let exp = splat(fb, u, 0x7F80_0000)?;
            let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
            let zero = splat(fb, u, 0)?;
            bin(fb, BinOp::INotEqual, e, zero)
        }
        ConvertKind::ToFloat { signed } => cast(
            fb,
            if signed {
                CastKind::SiToFp
            } else {
                CastKind::UiToFp
            },
            from,
            to,
            x,
        ),
        ConvertKind::Int { signed } => {
            if from.is_bool() {
                return cast(fb, CastKind::ZExt, from, to, x);
            }
            let widening = to.bits() >= from.bits();
            let c = match (widening, signed) {
                (true, true) => CastKind::SExt,
                (true, false) => CastKind::ZExt,
                (false, _) => CastKind::Trunc,
            };
            if widening && to.bits() == from.bits() {
                return Ok(x);
            }
            cast(fb, c, from, to, x)
        }
        ConvertKind::ToInt { signed } => {
            let n = u32::from(to.bits());
            // A narrow float is exactly its container's value: clamp in the container.
            let fw = from.container();
            let k = if signed { n - 1 } else { n };
            // Clamp into [lo, largest float below 2^k] first, so the conversion is in range. A
            // signed `lo` is `-2^k`, a float of both containers (`k <= 63`): below it the clamp
            // gives the minimum.
            let lo = if signed {
                splat(fb, from, pow2_bits(fw, k) | (1u64 << (fw - 1)))?
            } else {
                fsplat(fb, from, 0.0)?
            };
            let hi_bits = below_pow2_bits(fw, k);
            let hi = splat(fb, from, hi_bits)?;
            let c = glsl(fb, GlslOp::NClamp, &[x, lo, hi])?;
            let t = ty(fb, to)?;
            let op = if signed {
                ConvOp::ConvertFToS
            } else {
                ConvOp::ConvertFToU
            };
            let mut r = fb.convert(op, c, t)?;
            // At or above `2^k` the result is the maximum. The clamp already gives it when the
            // largest float below `2^k` truncates to `2^k - 1` (the float holds every integer up
            // to it); otherwise it is selected.
            if float_value(fw, hi_bits).trunc() < (2f64.powi(k as i32) - 1.0) {
                let limit = splat(fb, from, pow2_bits(fw, k))?;
                let over = bin(fb, BinOp::FOrdGreaterThanEqual, x, limit)?;
                let max = splat(fb, to, ones(if signed { to.bits() - 1 } else { to.bits() }))?;
                r = fb.select(over, max, r)?;
            }
            // `NClamp` takes NaN to the lower bound; Metal converts it to 0, the unsigned lower
            // bound. Without `SignedZeroInfNanPreserve` for the container a device may assume no
            // NaN, in `NClamp` and `OpIsNan` alike, so the bits tell it.
            if signed || test == NanTest::Bits {
                let nan = nan::is_nan(fb, from, test, x)?;
                let z = splat(fb, to, 0)?;
                r = fb.select(nan, z, r)?;
            }
            mask(fb, to, r)
        }
    }
}

/// The packed-channel layout of a normalized `air.pack`/`air.unpack` format: each channel's
/// width, low channel first, and whether channels are signed-normalized; `None` for the packed
/// float formats (`rg11b10f`, `rgb9e5`, [`float_unpack_lanes`]). `rgb565` holds red in bits
/// 0..5, green in 5..11, and blue in 11..16 (device-measured by the old engine); the `snorm`
/// `rgb10a2` fields are two's complement at `rgb10a2`'s positions (measured).
pub fn pack_layout(format: PackFormat) -> Option<(&'static [u32], bool)> {
    Some(match format {
        PackFormat::Unorm4x8 | PackFormat::Unorm4x8Srgb => (&[8, 8, 8, 8], false),
        PackFormat::Snorm4x8 => (&[8, 8, 8, 8], true),
        PackFormat::Unorm2x16 => (&[16, 16], false),
        PackFormat::Snorm2x16 => (&[16, 16], true),
        PackFormat::UnormRgb10a2 => (&[10, 10, 10, 2], false),
        PackFormat::SnormRgb10a2 => (&[10, 10, 10, 2], true),
        PackFormat::UnormRgb565 => (&[5, 6, 5], false),
        PackFormat::UnormRg11b10f | PackFormat::UnormRgb9e5 => return None,
    })
}

/// How a normalized `air.unpack` format converts a channel code `c` to a float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnpackRounding {
    /// `c / d` correctly rounded ([`div_exact`]; snorm: of the magnitude clamped to `d`).
    ///
    /// [`div_exact`]: crate::lower::arith::exact_division::div_exact
    Quotient,
    /// `max(c * fl(1/d), -1)`, the product with the `f32` reciprocal rounded to nearest even
    /// ([`mul_exact`]): Metal's `snorm` `rgb10a2` on every code of every field, where the quotient
    /// differs on 267,279 of 1,049,600 measured words by one ULP.
    ReciprocalProduct,
}

/// The conversion of normalized unpack format `format`'s channels ([`UnpackRounding`]).
pub fn unpack_rounding(format: PackFormat) -> UnpackRounding {
    match format {
        PackFormat::SnormRgb10a2 => UnpackRounding::ReciprocalProduct,
        _ => UnpackRounding::Quotient,
    }
}

/// The float lanes of a packed float format this family unpacks (`rg11b10f`, `rgb9e5`: 3), or
/// `None` for a normalized one. Packing to them is not lowered (Metal's rounding, overflow, and NaN
/// rules for these encodings are not established).
pub fn float_unpack_lanes(format: PackFormat) -> Option<u8> {
    match format {
        PackFormat::UnormRg11b10f | PackFormat::UnormRgb9e5 => Some(3),
        _ => None,
    }
}

/// The AIR width of the packed integer of `format` (`rgb565` packs into a `ushort`).
pub fn packed_bits(format: PackFormat) -> u8 {
    if format == PackFormat::UnormRgb565 {
        16
    } else {
        32
    }
}

/// The error of an `air.pack`/`air.unpack` whose format the emission cannot take (a packed float
/// format to pack to, or the sRGB format without the module's table): selection rules both out.
fn unpackable() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Operand {
        op: crate::spv::Op::ExtInst,
        rule: crate::spirv::error::OperandRule::Arity,
    }
}

/// [`round_exact_product_lanes`] with one scale `m` for every lane.
#[cfg(test)]
pub fn round_exact_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    m: u32,
    snorm: bool,
) -> BResult<Value> {
    round_exact_product_lanes(fb, s, v, &[m], snorm)
}

/// `v` (`f32` lanes of form `s`, in `[0, 1]`, or `[-1, 1]` when `snorm`) times `m[i]` for lane
/// `i` (one scale for every lane when `m` has one; each below `2^16`), rounded to the nearest
/// integer, ties to even, **from the exact product**, as `u32` lanes (two's complement for a
/// negative `snorm` result). The sequence is lane-wise.
///
/// Rounding `fl(v * m)` differs where the multiply itself rounds onto a half-integer; Metal
/// rounds the exact product (measured for `rgb10a2` by the old engine's device case
/// `round-the-exact-product-into-ten-ten-ten-two`). The product is formed exactly in integers,
/// independent of the device's rounding mode and of whether it fuses: with `v = mant * 2^(e -
/// 150)` (`e` the biased exponent, `mant` the significand with its leading one), `v * m =
/// (mant * m) / 2^s` for `s = 150 - e >= 23`. `mant * m < 2^40` is the `UMulExtended` pair;
/// shifted right by 9 with a sticky bit it fits 31 bits, and `s >= 41` (which includes every
/// denormal and zero) gives a product below one half, so 0.
pub fn round_exact_product_lanes(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    m: &[u32],
    snorm: bool,
) -> BResult<Value> {
    let u = narrow::u32_form(s);
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let a = if snorm {
        glsl(fb, GlslOp::FAbs, &[v])?
    } else {
        v
    };
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, a, ut)?;
    let k23 = c(fb, 23)?;
    let e = bin(fb, BinOp::ShiftRightLogical, bits, k23)?;
    let man = c(fb, 0x7F_FFFF)?;
    let mant = bin(fb, BinOp::BitwiseAnd, bits, man)?;
    let lead = c(fb, 0x80_0000)?;
    let mant = bin(fb, BinOp::BitwiseOr, mant, lead)?;
    let m: Vec<u64> = m.iter().map(|m| u64::from(*m)).collect();
    let mk = per_lane(fb, u, &m)?;
    let p = bin(fb, BinOp::UMulExtended, mant, mk)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    // s = 150 - e; zero when s >= 41; t = min(s, 40) - 9 in [14, 31].
    let k150 = c(fb, 150)?;
    let sh = bin(fb, BinOp::ISub, k150, e)?;
    let k41 = c(fb, 41)?;
    let zero = bin(fb, BinOp::UGreaterThanEqual, sh, k41)?;
    let k40 = c(fb, 40)?;
    let sh = glsl(fb, GlslOp::UMin, &[sh, k40])?;
    let k9 = c(fb, 9)?;
    let t = bin(fb, BinOp::ISub, sh, k9)?;
    // x = (mant * m) >> 9 with the dropped bits as a sticky lowest bit.
    let low9 = c(fb, 0x1FF)?;
    let dropped = bin(fb, BinOp::BitwiseAnd, lo, low9)?;
    let z = c(fb, 0)?;
    let one = c(fb, 1)?;
    let lost = bin(fb, BinOp::INotEqual, dropped, z)?;
    let sticky = fb.select(lost, one, z)?;
    let x = bin(fb, BinOp::ShiftRightLogical, lo, k9)?;
    let up = bin(fb, BinOp::ShiftLeftLogical, hi, k23)?;
    let x = bin(fb, BinOp::BitwiseOr, x, up)?;
    let x = bin(fb, BinOp::BitwiseOr, x, sticky)?;
    // Round x / 2^t to nearest even.
    let q = bin(fb, BinOp::ShiftRightLogical, x, t)?;
    let unit = bin(fb, BinOp::ShiftLeftLogical, one, t)?;
    let below = bin(fb, BinOp::ISub, unit, one)?;
    let rem = bin(fb, BinOp::BitwiseAnd, x, below)?;
    let t1 = bin(fb, BinOp::ISub, t, one)?;
    let half = bin(fb, BinOp::ShiftLeftLogical, one, t1)?;
    let gt = bin(fb, BinOp::UGreaterThan, rem, half)?;
    let eq = bin(fb, BinOp::IEqual, rem, half)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
    let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, eq, odd)?;
    let round_up = bin(fb, BinOp::LogicalOr, gt, tie_up)?;
    let inc = fb.select(round_up, one, z)?;
    let r = bin(fb, BinOp::IAdd, q, inc)?;
    let r = fb.select(zero, z, r)?;
    if !snorm {
        return Ok(r);
    }
    let fz = fsplat(fb, s, 0.0)?;
    let neg = bin(fb, BinOp::FOrdLessThan, v, fz)?;
    let nr = bin(fb, BinOp::ISub, z, r)?;
    fb.select(neg, nr, r)
}

/// `air.unpack.<format>`: the packed integer `w` (a `u32`; a `ushort` for `rgb565`) to float
/// lanes of form `f`. Each channel of a normalized format is converted as Metal converts a
/// normalized integer to a float: `c / (2^N - 1)` (snorm: `max(c / (2^(N-1) - 1), -1)`), the
/// quotient correctly rounded ([`div_exact`]), except the `snorm` `rgb10a2`, which multiplies by
/// the `f32` reciprocal ([`unpack_rounding`]); the sRGB format decodes the color channels through
/// Metal's twelve-bit table (`k / 4095` of the step [`srgb::step`]), which `srgb` (the module's
/// table) holds. A narrow form rounds each lane (so rounds twice, as Metal's `half` answers do).
/// The packed float formats decode exactly (`unpack_float`).
///
/// Every channel runs one lane of one vector sequence, its field and divisor per-lane constants:
/// the instructions are lane-wise, so each lane computes exactly what it computed alone. Only the
/// sRGB table reads are per channel.
///
/// [`div_exact`]: crate::lower::arith::exact_division::div_exact
pub fn unpack(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    w: Value,
    srgb: Option<GlobalId>,
) -> BResult<Value> {
    if float_unpack_lanes(format).is_some() {
        return unpack_float(fb, format, f, w);
    }
    let Some((widths, snorm)) = pack_layout(format) else {
        return Err(unpackable());
    };
    if usize::from(f.lanes) != widths.len() {
        return Err(unpackable());
    }
    let s = f.with(Held::Float(FloatFmt::F32));
    let u = narrow::u32_form(s);
    let ut = ty(fb, u)?;
    // Lane `i` holds field `i`: the word shifted down to the field and masked to its width.
    let mut at = Vec::with_capacity(widths.len());
    let mut next = 0u64;
    for n in widths {
        at.push(next);
        next += u64::from(*n);
    }
    let spread = fb.composite_construct(ut, &vec![w; widths.len()])?;
    let sh = per_lane(fb, u, &at)?;
    let v = bin(fb, BinOp::ShiftRightLogical, spread, sh)?;
    let m: Vec<u64> = widths.iter().map(|n| ones(*n as u8)).collect();
    let m = per_lane(fb, u, &m)?;
    let v = bin(fb, BinOp::BitwiseAnd, v, m)?;
    let q = if format == PackFormat::Unorm4x8Srgb {
        // The color channels' steps over 4095, the alpha channel's byte over 255.
        let table = srgb.ok_or_else(unpackable)?;
        let mut k = Vec::with_capacity(widths.len());
        for i in 0..widths.len() {
            let b = fb.composite_extract(v, &[i as u32])?;
            k.push(if i < 3 { srgb::step(fb, table, b)? } else { b });
        }
        let k = fb.composite_construct(ut, &k)?;
        let d: Vec<u32> = (0..widths.len())
            .map(|i| if i < 3 { 4095 } else { 255 })
            .collect();
        div_exact_lanes(fb, u, k, &d)?
    } else if snorm {
        let d: Vec<u32> = widths.iter().map(|n| (1u32 << (n - 1)) - 1).collect();
        let up: Vec<u64> = widths.iter().map(|n| u64::from(32 - n)).collect();
        let up = per_lane(fb, u, &up)?;
        let v = bin(fb, BinOp::ShiftLeftLogical, v, up)?;
        let v = bin(fb, BinOp::ShiftRightArithmetic, v, up)?;
        let z = splat(fb, u, 0)?;
        let neg = bin(fb, BinOp::SLessThan, v, z)?;
        let minus = bin(fb, BinOp::ISub, z, v)?;
        let mag = fb.select(neg, minus, v)?;
        match unpack_rounding(format) {
            UnpackRounding::Quotient => {
                // The magnitude, clamped to the scale (the most negative code is -1 exactly).
                let dk: Vec<u64> = d.iter().map(|d| u64::from(*d)).collect();
                let dk = per_lane(fb, u, &dk)?;
                let mag = glsl(fb, GlslOp::UMin, &[mag, dk])?;
                let q = div_exact_lanes(fb, u, mag, &d)?;
                let nq = fb.unary(UnOp::FNegate, q)?;
                fb.select(neg, nq, q)?
            }
            UnpackRounding::ReciprocalProduct => {
                // `max(c * fl(1/d), -1)`: the magnitude's product with its lane's reciprocal,
                // the code's sign, and the bound, which only the most negative code reaches
                // (`-512 * fl(1/511)` is below -1). Comparing with -1 is exact on every device.
                let r: Vec<u32> = d.iter().map(|d| (1.0f32 / *d as f32).to_bits()).collect();
                let p = mul_exact(fb, u, mag, &r)?;
                let pb = fb.convert(ConvOp::Bitcast, p, ut)?;
                let sign = splat(fb, u, 0x8000_0000)?;
                let nb = bin(fb, BinOp::BitwiseOr, pb, sign)?;
                let b = fb.select(neg, nb, pb)?;
                let st = ty(fb, s)?;
                let q = fb.convert(ConvOp::Bitcast, b, st)?;
                let m1 = fsplat(fb, s, -1.0)?;
                let below = bin(fb, BinOp::FOrdLessThan, q, m1)?;
                fb.select(below, m1, q)?
            }
        }
    } else {
        let d: Vec<u32> = widths.iter().map(|n| (1u32 << n) - 1).collect();
        div_exact_lanes(fb, u, v, &d)?
    };
    narrow::round(fb, f, q)
}

/// `air.unpack` of a packed float format (the field layouts are the old engine's, device-measured
/// against Metal's `rg11b10f<float3>` and `rgb9e5<float3>` reads):
///
/// - `rg11b10f`: red in bits 0..11 and green in 11..22 are unsigned floats with a 5-bit exponent
///   (bias 15) above a 6-bit mantissa; blue in 22..32 has a 5-bit mantissa. Each is the half
///   whose exponent and (left-aligned) mantissa bits they are, infinities and NaNs included.
/// - `rgb9e5`: three 9-bit mantissas (bits 0..9, 9..18, 18..27) share the exponent in bits
///   27..32: each channel is `mantissa * 2^(exponent - 24)`, without an implicit leading one.
///
/// Every channel value is a `half` value (so also a `float` one), so nothing rounds: the
/// `rg11b10f` fields widen as a `half` read from its bits does (`narrow::from_bits`: the device's
/// widening where the module holds 16-bit floats, its NaN bits and a denormal as the module
/// records), and the `rgb9e5` scale `2^(exponent - 24)` is a normal `f32` built from its bits, whose product with
/// the mantissa is exact.
fn unpack_float(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    w: Value,
) -> BResult<Value> {
    let s = f.scalar_form().with(Held::Float(FloatFmt::F32));
    let u = narrow::u32_form(s);
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let mut lanes = Vec::with_capacity(3);
    if format == PackFormat::UnormRgb9e5 {
        let k27 = c(fb, 27)?;
        let e = bin(fb, BinOp::ShiftRightLogical, w, k27)?;
        let k103 = c(fb, 103)?;
        let e = bin(fb, BinOp::IAdd, e, k103)?;
        let k23 = c(fb, 23)?;
        let e = bin(fb, BinOp::ShiftLeftLogical, e, k23)?;
        let st = ty(fb, s)?;
        let scale = fb.convert(ConvOp::Bitcast, e, st)?;
        for at in [0u64, 9, 18] {
            let v = if at == 0 {
                w
            } else {
                let k = c(fb, at)?;
                bin(fb, BinOp::ShiftRightLogical, w, k)?
            };
            let m9 = c(fb, 0x1FF)?;
            let v = bin(fb, BinOp::BitwiseAnd, v, m9)?;
            let v = fb.convert(ConvOp::ConvertUToF, v, st)?;
            lanes.push(bin(fb, BinOp::FMul, v, scale)?);
        }
    } else {
        let half = Form {
            held: Held::Float(FloatFmt::Half),
            lanes: 1,
        };
        for (at, man) in [(0u64, 6u64), (11, 6), (22, 5)] {
            let v = if at == 0 {
                w
            } else {
                let k = c(fb, at)?;
                bin(fb, BinOp::ShiftRightLogical, w, k)?
            };
            let km = c(fb, man)?;
            let e = bin(fb, BinOp::ShiftRightLogical, v, km)?;
            let m5 = c(fb, 0x1F)?;
            let e = bin(fb, BinOp::BitwiseAnd, e, m5)?;
            let k10 = c(fb, 10)?;
            let e = bin(fb, BinOp::ShiftLeftLogical, e, k10)?;
            let mm = c(fb, (1 << man) - 1)?;
            let m = bin(fb, BinOp::BitwiseAnd, v, mm)?;
            let up = c(fb, 10 - man)?;
            let m = bin(fb, BinOp::ShiftLeftLogical, m, up)?;
            let h = bin(fb, BinOp::BitwiseOr, e, m)?;
            lanes.push(narrow::from_bits(fb, half, h)?);
        }
    }
    let t = ty(fb, f)?;
    fb.composite_construct(t, &lanes)
}

/// `a * r` rounded to nearest even as an `f32`, for an integer `0 <= a < 2^10` (`u32` lanes of
/// form `u`) and a positive normal `f32` constant `r` per lane (its bits `r_bits[i]` for lane
/// `i`, at most `2^117`; a lane-wise sequence), with integer operations only, so it is exact on
/// every device whatever its rounding mode and under every fast-math mode (Metal's `snorm`
/// `rgb10a2` unpack multiplies by the reciprocal).
///
/// With `r = m * 2^er` (`m` its 24-bit significand, leading one included), the product `p = a *
/// m` is below `2^34`: `UMulExtended` gives it as two words. Its leading one is at bit `b = 23 +
/// s` with `s` in `[0, 10]` (`m >= 2^23`), so the significand is `p >> s`, rounded to nearest
/// even on the `s` dropped bits (`s <= 10` keeps every shift below 32), and the biased exponent
/// is `s + er + 150`. As in [`div_exact`], the bits are `((s + er + 149) << 23) + q` plus the
/// rounding, which carries into the exponent when `q` is all ones. `a = 0` is `+0`.
///
/// [`div_exact`]: crate::lower::arith::exact_division::div_exact
pub fn mul_exact(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    a: Value,
    r_bits: &[u32],
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let m: Vec<u64> = r_bits
        .iter()
        .map(|r| u64::from((r & 0x7F_FFFF) | 0x80_0000))
        .collect();
    // er + 149 = (biased exponent - 150) + 149.
    let bias: Vec<u64> = r_bits
        .iter()
        .map(|r| u64::from(((r >> 23) & 0xFF).saturating_sub(1)))
        .collect();
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let zero = bin(fb, BinOp::IEqual, a, z)?;
    let mk = per_lane(fb, u, &m)?;
    let p = bin(fb, BinOp::UMulExtended, a, mk)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    // The leading bit: 32 + msb(hi) when hi is nonzero, else msb(lo) (>= 23 unless a = 0).
    let hi_set = bin(fb, BinOp::INotEqual, hi, z)?;
    let hm = glsl(fb, GlslOp::FindUMsb, &[hi])?;
    let k32 = c(fb, 32)?;
    let hm = bin(fb, BinOp::IAdd, hm, k32)?;
    let lm = glsl(fb, GlslOp::FindUMsb, &[lo])?;
    let msb = fb.select(hi_set, hm, lm)?;
    // s = msb - 23, in [0, 10]; t = max(s, 1) keeps the shifts below 32 when s = 0 (then hi = 0
    // and nothing is dropped).
    let k23 = c(fb, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let k10 = c(fb, 10)?;
    let s = glsl(fb, GlslOp::UMin, &[s, k10])?;
    let t = glsl(fb, GlslOp::UMax, &[s, one])?;
    let q = bin(fb, BinOp::ShiftRightLogical, lo, s)?;
    let up_sh = bin(fb, BinOp::ISub, k32, t)?;
    let top = bin(fb, BinOp::ShiftLeftLogical, hi, up_sh)?;
    let q = bin(fb, BinOp::BitwiseOr, q, top)?;
    let unit = bin(fb, BinOp::ShiftLeftLogical, one, s)?;
    let below = bin(fb, BinOp::ISub, unit, one)?;
    let rem = bin(fb, BinOp::BitwiseAnd, lo, below)?;
    let t1 = bin(fb, BinOp::ISub, t, one)?;
    let half = bin(fb, BinOp::ShiftLeftLogical, one, t1)?;
    let some = bin(fb, BinOp::INotEqual, s, z)?;
    let gt = bin(fb, BinOp::UGreaterThan, rem, half)?;
    let eq = bin(fb, BinOp::IEqual, rem, half)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
    let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, eq, odd)?;
    let up = bin(fb, BinOp::LogicalOr, gt, tie_up)?;
    let up = bin(fb, BinOp::LogicalAnd, some, up)?;
    let inc = fb.select(up, one, z)?;
    let kb = per_lane(fb, u, &bias)?;
    let be = bin(fb, BinOp::IAdd, s, kb)?;
    let be = bin(fb, BinOp::ShiftLeftLogical, be, k23)?;
    let bits = bin(fb, BinOp::IAdd, be, q)?;
    let bits = bin(fb, BinOp::IAdd, bits, inc)?;
    let bits = fb.select(zero, z, bits)?;
    let ft = ty(fb, u.with(Held::Float(FloatFmt::F32)))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}
