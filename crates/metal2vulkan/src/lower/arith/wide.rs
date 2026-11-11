//! Wide integers on profiles without `shaderInt64`.
//!
//! On such a profile an `i33`..`i64` value is held as two `u32` words, low word first
//! ([`Held::Wide`]): a scalar is a `uvec2`, and a vector of `n` lanes is a struct of the `n`-lane
//! low-word vector and the `n`-lane high-word vector, so every operation below is lane-wise on
//! ordinary 32-bit vectors. The value is zero-extended from its AIR width like every held integer:
//! the bits of the high word above the width are zero, so an operation that can set them masks its
//! result ([`mask_top`]) and a signed one sign-extends its operands first ([`sext_top`]).
//!
//! Every sequence is exact for every non-poison input and uses only 32-bit integer instructions
//! (no `Int64` capability): add and subtract through `OpIAddCarry`/`OpISubBorrow`, multiply
//! through `OpUMulExtended` and the two cross products, shifts by a dynamic amount through
//! in-range 32-bit shifts and a select on the amount's bit 5, compares from the high words and
//! then the low words, and division and remainder by Hacker's Delight's `divdu` on 32-bit
//! `OpUDiv` (Knuth's algorithm D on 16-bit digits, straight-line: no loop is opened; every
//! divisor is nonzero). The division is one module-scope function per lane count, which every
//! division on that many lanes calls ([`Divide`]; a signed one divides the magnitudes and sets the
//! sign at the call), so a module holds it once, not once per use. A conversion to a float
//! collapses the magnitude to 32 significant bits with a sticky bit before one exactly scaled
//! conversion, so it rounds once (to odd first for `bfloat`, then once to the format); a conversion
//! from a float splits the truncated magnitude at `2^32` exactly.
//!
//! The multiply-high functions and `mad_sat` form the exact 128-bit product from four
//! `OpUMulExtended` (signed: of the sign-extended operands, corrected by subtracting each operand
//! where the other is negative) and take its bits `[N, 2N)` or saturate it; `rotate` and the
//! funnel shifts reduce the amount modulo the width in 32-bit words and shift in range at every
//! width.

use crate::air::intrinsics::AirIntrinsic;
use crate::air::ir::{
    BinOp as AirBin, Callee, CastOp, Inst, InstId, InstKind, IntPredicate, Value,
};
use crate::error::Error;
use crate::lower::arith::legal::{self, FloatFmt, Form, Held};
use crate::lower::arith::math::{self, ConvertKind, IntFn, Sign};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{self, CastKind};
use crate::lower::arith::rounding;
use crate::lower::select::SelectCx;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule, TypeRule};
use crate::spirv::ids::{ConstId, DemoteTy, FuncId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type, UnOp};

use super::{check_operands, lanes_of, not_lowered, operands_of, ty_of, ArithStrategy, Fun};

// ---- The pair ----

/// A wide value as its two words (each a `u32` or a `u32` vector of the value's lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub lo: SValue,
    pub hi: SValue,
}

/// The `u32` form of `lanes` lanes (one word of each lane).
pub fn words_form(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

/// The wide form of `bits` bits and `lanes` lanes.
pub fn wide_form(bits: u8, lanes: u8) -> Form {
    Form {
        held: Held::Wide { bits },
        lanes,
    }
}

fn arity_error() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::CompositeExtract,
        rule: OperandRule::Arity,
    }
}

fn word_const(fb: &mut FunctionBuilder<'_>, lanes: u8, v: u32) -> BResult<ConstId> {
    let u = ops::ty(fb, words_form(1))?;
    let s = fb.module().constant(Constant::Scalar {
        ty: u,
        bits: ConstBits(u64::from(v)),
    })?;
    if lanes == 1 {
        return Ok(s);
    }
    let t = ops::ty(fb, words_form(lanes))?;
    fb.module().constant(Constant::Composite {
        ty: t,
        parts: vec![s; usize::from(lanes)].into(),
    })
}

/// The word constant `v` in every lane.
fn k(fb: &mut FunctionBuilder<'_>, lanes: u8, v: u32) -> BResult<SValue> {
    let c = word_const(fb, lanes, v)?;
    fb.module().const_value(c)
}

/// The wide constant of form `f` with `bits` (masked to the width) in every lane.
pub fn splat(fb: &mut FunctionBuilder<'_>, f: Form, bits: u64) -> BResult<SValue> {
    let bits = bits & ops::ones(f.bits());
    let lo = word_const(fb, f.lanes, bits as u32)?;
    let hi = word_const(fb, f.lanes, (bits >> 32) as u32)?;
    let t = ops::ty(fb, f)?;
    let c = fb.module().constant(Constant::Composite {
        ty: t,
        parts: vec![lo, hi].into(),
    })?;
    fb.module().const_value(c)
}

/// The pair constant with `bits` in every lane of `lanes` lanes.
fn pk(fb: &mut FunctionBuilder<'_>, lanes: u8, bits: u64) -> BResult<Pair> {
    Ok(Pair {
        lo: k(fb, lanes, bits as u32)?,
        hi: k(fb, lanes, (bits >> 32) as u32)?,
    })
}

/// The two words of wide value `x`.
pub fn split(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<Pair> {
    Ok(Pair {
        lo: fb.composite_extract(x, &[0])?,
        hi: fb.composite_extract(x, &[1])?,
    })
}

/// The wide value of form `f` with words `p`.
pub fn join(fb: &mut FunctionBuilder<'_>, f: Form, p: Pair) -> BResult<SValue> {
    let t = ops::ty(fb, f)?;
    fb.composite_construct(t, &[p.lo, p.hi])
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: SValue, b: SValue) -> BResult<SValue> {
    fb.binary(op, a, b)
}

/// `x op c` with a word constant, omitted for a shift by 0.
fn bink(fb: &mut FunctionBuilder<'_>, n: u8, op: BinOp, x: SValue, c: u32) -> BResult<SValue> {
    if c == 0
        && matches!(
            op,
            BinOp::ShiftLeftLogical | BinOp::ShiftRightLogical | BinOp::ShiftRightArithmetic
        )
    {
        return Ok(x);
    }
    let c = k(fb, n, c)?;
    bin(fb, op, x, c)
}

/// The words of `p` with the bits above `f`'s width cleared.
pub fn mask_top(fb: &mut FunctionBuilder<'_>, f: Form, p: Pair) -> BResult<Pair> {
    let bits = f.bits();
    if bits >= 64 {
        return Ok(p);
    }
    let hi = bink(
        fb,
        f.lanes,
        BinOp::BitwiseAnd,
        p.hi,
        ops::ones(bits - 32) as u32,
    )?;
    Ok(Pair { lo: p.lo, hi })
}

/// The words of `p` (zero-extended from `f`'s width) sign-extended to 64 bits.
pub fn sext_top(fb: &mut FunctionBuilder<'_>, f: Form, p: Pair) -> BResult<Pair> {
    let bits = f.bits();
    if bits >= 64 {
        return Ok(p);
    }
    let sh = u32::from(64 - bits);
    let up = bink(fb, f.lanes, BinOp::ShiftLeftLogical, p.hi, sh)?;
    let hi = bink(fb, f.lanes, BinOp::ShiftRightArithmetic, up, sh)?;
    Ok(Pair { lo: p.lo, hi })
}

fn per_word(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Pair, b: Pair) -> BResult<Pair> {
    Ok(Pair {
        lo: bin(fb, op, a.lo, b.lo)?,
        hi: bin(fb, op, a.hi, b.hi)?,
    })
}

/// `a + b` modulo `2^64`.
pub fn add(fb: &mut FunctionBuilder<'_>, a: Pair, b: Pair) -> BResult<Pair> {
    let r = bin(fb, BinOp::IAddCarry, a.lo, b.lo)?;
    let lo = fb.composite_extract(r, &[0])?;
    let c = fb.composite_extract(r, &[1])?;
    let hi = bin(fb, BinOp::IAdd, a.hi, b.hi)?;
    let hi = bin(fb, BinOp::IAdd, hi, c)?;
    Ok(Pair { lo, hi })
}

/// `a - b` modulo `2^64`.
pub fn sub(fb: &mut FunctionBuilder<'_>, a: Pair, b: Pair) -> BResult<Pair> {
    let r = bin(fb, BinOp::ISubBorrow, a.lo, b.lo)?;
    let lo = fb.composite_extract(r, &[0])?;
    let c = fb.composite_extract(r, &[1])?;
    let hi = bin(fb, BinOp::ISub, a.hi, b.hi)?;
    let hi = bin(fb, BinOp::ISub, hi, c)?;
    Ok(Pair { lo, hi })
}

/// `-a` modulo `2^64` (`n` lanes).
pub fn neg(fb: &mut FunctionBuilder<'_>, n: u8, a: Pair) -> BResult<Pair> {
    let z = pk(fb, n, 0)?;
    sub(fb, z, a)
}

/// `a * b` modulo `2^64`.
pub fn mul(fb: &mut FunctionBuilder<'_>, a: Pair, b: Pair) -> BResult<Pair> {
    let p = bin(fb, BinOp::UMulExtended, a.lo, b.lo)?;
    let lo = fb.composite_extract(p, &[0])?;
    let c = fb.composite_extract(p, &[1])?;
    let x = bin(fb, BinOp::IMul, a.lo, b.hi)?;
    let y = bin(fb, BinOp::IMul, a.hi, b.lo)?;
    let hi = bin(fb, BinOp::IAdd, c, x)?;
    let hi = bin(fb, BinOp::IAdd, hi, y)?;
    Ok(Pair { lo, hi })
}

/// `select(c, a, b)` word by word (`c` a `bool` or a `bool` vector of the words' lanes).
pub fn sel(fb: &mut FunctionBuilder<'_>, c: SValue, a: Pair, b: Pair) -> BResult<Pair> {
    Ok(Pair {
        lo: fb.select(c, a.lo, b.lo)?,
        hi: fb.select(c, a.hi, b.hi)?,
    })
}

/// A shift of a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shift {
    Left,
    Logical,
    Arithmetic,
}

/// `x` shifted by `s mod 64` (`s` a word of `n` lanes). Every 32-bit shift is in range: the
/// words move by `s mod 32`, the bits crossing between them by two shifts (`(x >> 1) >> (31 -
/// s)`), and bit 5 of `s` selects the half-swapped result. An arithmetic shift reads `x`'s high
/// word as signed (sign-extend it first).
fn shift(fb: &mut FunctionBuilder<'_>, n: u8, kind: Shift, x: Pair, s: SValue) -> BResult<Pair> {
    let s = bink(fb, n, BinOp::BitwiseAnd, s, 63)?;
    let s5 = bink(fb, n, BinOp::BitwiseAnd, s, 31)?;
    let b5 = bink(fb, n, BinOp::BitwiseAnd, s, 32)?;
    let z = k(fb, n, 0)?;
    let big = bin(fb, BinOp::INotEqual, b5, z)?;
    let k31 = k(fb, n, 31)?;
    let inv = bin(fb, BinOp::ISub, k31, s5)?;
    Ok(match kind {
        Shift::Left => {
            let lo_s = bin(fb, BinOp::ShiftLeftLogical, x.lo, s5)?;
            let c = bink(fb, n, BinOp::ShiftRightLogical, x.lo, 1)?;
            let c = bin(fb, BinOp::ShiftRightLogical, c, inv)?;
            let hi_s = bin(fb, BinOp::ShiftLeftLogical, x.hi, s5)?;
            let hi_s = bin(fb, BinOp::BitwiseOr, hi_s, c)?;
            Pair {
                lo: fb.select(big, z, lo_s)?,
                hi: fb.select(big, lo_s, hi_s)?,
            }
        }
        Shift::Logical | Shift::Arithmetic => {
            let op = if kind == Shift::Arithmetic {
                BinOp::ShiftRightArithmetic
            } else {
                BinOp::ShiftRightLogical
            };
            let hi_s = bin(fb, op, x.hi, s5)?;
            let c = bink(fb, n, BinOp::ShiftLeftLogical, x.hi, 1)?;
            let c = bin(fb, BinOp::ShiftLeftLogical, c, inv)?;
            let lo_s = bin(fb, BinOp::ShiftRightLogical, x.lo, s5)?;
            let lo_s = bin(fb, BinOp::BitwiseOr, lo_s, c)?;
            let fill = if kind == Shift::Arithmetic {
                bin(fb, BinOp::ShiftRightArithmetic, x.hi, k31)?
            } else {
                z
            };
            Pair {
                lo: fb.select(big, hi_s, lo_s)?,
                hi: fb.select(big, fill, hi_s)?,
            }
        }
    })
}

/// `x << (s mod 64)` (`s` a word of `n` lanes).
pub fn shl(fb: &mut FunctionBuilder<'_>, n: u8, x: Pair, s: SValue) -> BResult<Pair> {
    shift(fb, n, Shift::Left, x, s)
}

/// `x >> (s mod 64)`, logical (`s` a word of `n` lanes).
pub fn lshr(fb: &mut FunctionBuilder<'_>, n: u8, x: Pair, s: SValue) -> BResult<Pair> {
    shift(fb, n, Shift::Logical, x, s)
}

/// `x >> s` (logical) for a constant `s` below 64.
fn lshr_const(fb: &mut FunctionBuilder<'_>, n: u8, x: Pair, s: u32) -> BResult<Pair> {
    Ok(match s {
        0 => x,
        1..=31 => {
            let lo = bink(fb, n, BinOp::ShiftRightLogical, x.lo, s)?;
            let c = bink(fb, n, BinOp::ShiftLeftLogical, x.hi, 32 - s)?;
            Pair {
                lo: bin(fb, BinOp::BitwiseOr, lo, c)?,
                hi: bink(fb, n, BinOp::ShiftRightLogical, x.hi, s)?,
            }
        }
        _ => Pair {
            lo: bink(fb, n, BinOp::ShiftRightLogical, x.hi, s - 32)?,
            hi: k(fb, n, 0)?,
        },
    })
}

/// `x << s` for a constant `s` below 64.
fn shl_const(fb: &mut FunctionBuilder<'_>, n: u8, x: Pair, s: u32) -> BResult<Pair> {
    Ok(match s {
        0 => x,
        1..=31 => {
            let hi = bink(fb, n, BinOp::ShiftLeftLogical, x.hi, s)?;
            let c = bink(fb, n, BinOp::ShiftRightLogical, x.lo, 32 - s)?;
            Pair {
                lo: bink(fb, n, BinOp::ShiftLeftLogical, x.lo, s)?,
                hi: bin(fb, BinOp::BitwiseOr, hi, c)?,
            }
        }
        _ => Pair {
            lo: k(fb, n, 0)?,
            hi: bink(fb, n, BinOp::ShiftLeftLogical, x.lo, s - 32)?,
        },
    })
}

/// An unsigned (or, on sign-extended pairs, signed) comparison of two pairs.
fn compare(fb: &mut FunctionBuilder<'_>, pred: IntPredicate, a: Pair, b: Pair) -> BResult<SValue> {
    use IntPredicate as P;
    let (a, b, pred) = match pred {
        P::Ugt => (b, a, P::Ult),
        P::Uge => (b, a, P::Ule),
        P::Sgt => (b, a, P::Slt),
        P::Sge => (b, a, P::Sle),
        p => (a, b, p),
    };
    let lo_eq = bin(fb, BinOp::IEqual, a.lo, b.lo)?;
    let hi_eq = bin(fb, BinOp::IEqual, a.hi, b.hi)?;
    match pred {
        P::Eq => bin(fb, BinOp::LogicalAnd, lo_eq, hi_eq),
        P::Ne => {
            let eq = bin(fb, BinOp::LogicalAnd, lo_eq, hi_eq)?;
            fb.unary(UnOp::LogicalNot, eq)
        }
        P::Ult | P::Ule | P::Slt | P::Sle => {
            let hi_op = if matches!(pred, P::Slt | P::Sle) {
                BinOp::SLessThan
            } else {
                BinOp::ULessThan
            };
            let lo_op = if matches!(pred, P::Ule | P::Sle) {
                BinOp::ULessThanEqual
            } else {
                BinOp::ULessThan
            };
            let hi_lt = bin(fb, hi_op, a.hi, b.hi)?;
            let lo_c = bin(fb, lo_op, a.lo, b.lo)?;
            let tie = bin(fb, BinOp::LogicalAnd, hi_eq, lo_c)?;
            bin(fb, BinOp::LogicalOr, hi_lt, tie)
        }
        _ => Err(arity_error()),
    }
}

/// Whether sign-extended pair `p` is negative.
fn negative(fb: &mut FunctionBuilder<'_>, n: u8, p: Pair) -> BResult<SValue> {
    let z = k(fb, n, 0)?;
    bin(fb, BinOp::SLessThan, p.hi, z)
}

/// `|p|` of sign-extended pair `p` (as an unsigned pair; `-2^63` stays `2^63`).
fn abs(fb: &mut FunctionBuilder<'_>, n: u8, p: Pair) -> BResult<(Pair, SValue)> {
    let neg_p = negative(fb, n, p)?;
    let m = neg(fb, n, p)?;
    Ok((sel(fb, neg_p, m, p)?, neg_p))
}

/// `c ? 1 : 0` in each of `n` lanes.
fn one_if(fb: &mut FunctionBuilder<'_>, n: u8, c: SValue) -> BResult<SValue> {
    let (one, z) = (k(fb, n, 1)?, k(fb, n, 0)?);
    fb.select(c, one, z)
}

/// Quotient and remainder of unsigned `u / v` on two-word lanes, exact for every 64-bit `u` and
/// `v` (Hacker's Delight's `divdu`, straight-line: both of its cases are computed and selected
/// per lane, and no loop is opened). A zero divisor, which LLVM leaves undefined, divides by one
/// (quotient `u`, remainder 0), so no `OpUDiv` below divides by zero.
///
/// A divisor below `2^32` (`v.hi == 0`) divides the high word by it (`u.hi / v.lo`, the
/// quotient's high word) and then the remainder's two words by it ([`div_words`], the quotient's
/// low word). A wider divisor is normalized by its leading zeros `z` (its top word `top`, top bit
/// set); `(u / 2) / top`, shifted right by `31 - z`, is the quotient or one less (at most one less
/// once a nonzero estimate is lowered by one), and one compare of the remainder `u - q·v` with
/// `v` corrects it.
fn udivrem(fb: &mut FunctionBuilder<'_>, n: u8, u: Pair, v: Pair) -> BResult<(Pair, Pair)> {
    let z = k(fb, n, 0)?;
    let narrow = bin(fb, BinOp::IEqual, v.hi, z)?;
    let one = k(fb, n, 1)?;
    let at_least_one = fb.glsl(GlslOp::UMax, &[v.lo, one])?;
    let v = Pair {
        lo: fb.select(narrow, at_least_one, v.lo)?,
        hi: v.hi,
    };
    // The top word of the normalized divisor: `m` is the top bit of `v.hi` (of `1` in a narrow
    // lane, which keeps every shift below in range), and the bits leaving `v.lo` enter by two
    // in-range shifts (`(v.lo >> 1) >> m` is `v.lo >> (32 - z)`).
    let h1 = bink(fb, n, BinOp::BitwiseOr, v.hi, 1)?;
    let m = fb.glsl(GlslOp::FindUMsb, &[h1])?;
    let k31 = k(fb, n, 31)?;
    let lz = bin(fb, BinOp::ISub, k31, m)?;
    let top = bin(fb, BinOp::ShiftLeftLogical, v.hi, lz)?;
    let c = bink(fb, n, BinOp::ShiftRightLogical, v.lo, 1)?;
    let c = bin(fb, BinOp::ShiftRightLogical, c, m)?;
    let top = bin(fb, BinOp::BitwiseOr, top, c)?;
    // The one word the long division divides by, nonzero in every lane.
    let d = fb.select(narrow, v.lo, top)?;
    let qh = bin(fb, BinOp::UDiv, u.hi, d)?;
    let rh = bin(fb, BinOp::UMod, u.hi, d)?;
    // `u / 2`, whose high word is below `2^31 <= top`.
    let hh = bink(fb, n, BinOp::ShiftRightLogical, u.hi, 1)?;
    let hl = bink(fb, n, BinOp::ShiftRightLogical, u.lo, 1)?;
    let cross = bink(fb, n, BinOp::ShiftLeftLogical, u.hi, 31)?;
    let hl = bin(fb, BinOp::BitwiseOr, hl, cross)?;
    let (nh, nl) = (fb.select(narrow, rh, hh)?, fb.select(narrow, u.lo, hl)?);
    let dq = div_words(fb, n, nh, nl, d)?;
    // Undo the normalization and the halving: `(dq << z) >> 31` is `dq >> m`.
    let est = bin(fb, BinOp::ShiftRightLogical, dq, m)?;
    let nonzero = bin(fb, BinOp::INotEqual, est, z)?;
    let dec = one_if(fb, n, nonzero)?;
    let est = bin(fb, BinOp::ISub, est, dec)?;
    let q = Pair {
        lo: fb.select(narrow, dq, est)?,
        hi: fb.select(narrow, qh, z)?,
    };
    // `q·v <= u`, so the remainder is exact modulo `2^64`; it reaches `v` only in a wide lane
    // whose estimate is one short (whose quotient stays below `2^32`).
    let p = mul(fb, q, v)?;
    let r = sub(fb, u, p)?;
    let short = compare(fb, IntPredicate::Uge, r, v)?;
    let inc = one_if(fb, n, short)?;
    let q = Pair {
        lo: bin(fb, BinOp::IAdd, q.lo, inc)?,
        hi: q.hi,
    };
    let less = sub(fb, r, v)?;
    Ok((q, sel(fb, short, less, r)?))
}

/// The quotient word of the two-word dividend `(u1, u0)` by the word `v`, where `0 < v` and
/// `u1 < v` in every lane (Hacker's Delight's `divlu`, Knuth's algorithm D on 16-bit digits):
/// `v` shifted left by its leading zeros `s` (top bit set) and the dividend with it, then two
/// quotient digits ([`digit`]), each from the running remainder, which stays below the shifted
/// divisor and so is exact modulo `2^32`.
fn div_words(
    fb: &mut FunctionBuilder<'_>,
    n: u8,
    u1: SValue,
    u0: SValue,
    v: SValue,
) -> BResult<SValue> {
    let t = fb.glsl(GlslOp::FindUMsb, &[v])?;
    let k31 = k(fb, n, 31)?;
    let s = bin(fb, BinOp::ISub, k31, t)?;
    let vn = bin(fb, BinOp::ShiftLeftLogical, v, s)?;
    let vn1 = bink(fb, n, BinOp::ShiftRightLogical, vn, 16)?;
    let vn0 = bink(fb, n, BinOp::BitwiseAnd, vn, 0xFFFF)?;
    // `u1 << s` takes the bits leaving `u0` by two in-range shifts (`(u0 >> 1) >> t`).
    let hi = bin(fb, BinOp::ShiftLeftLogical, u1, s)?;
    let c = bink(fb, n, BinOp::ShiftRightLogical, u0, 1)?;
    let c = bin(fb, BinOp::ShiftRightLogical, c, t)?;
    let un32 = bin(fb, BinOp::BitwiseOr, hi, c)?;
    let un10 = bin(fb, BinOp::ShiftLeftLogical, u0, s)?;
    let un1 = bink(fb, n, BinOp::ShiftRightLogical, un10, 16)?;
    let un0 = bink(fb, n, BinOp::BitwiseAnd, un10, 0xFFFF)?;
    let q1 = digit(fb, n, un32, un1, vn1, vn0)?;
    let up = bink(fb, n, BinOp::ShiftLeftLogical, un32, 16)?;
    let up = bin(fb, BinOp::IAdd, up, un1)?;
    let taken = bin(fb, BinOp::IMul, q1, vn)?;
    let un21 = bin(fb, BinOp::ISub, up, taken)?;
    let q0 = digit(fb, n, un21, un0, vn1, vn0)?;
    let q1 = bink(fb, n, BinOp::ShiftLeftLogical, q1, 16)?;
    bin(fb, BinOp::BitwiseOr, q1, q0)
}

/// One quotient digit (below `2^16`) of `num·2^16 + low` by the normalized `vn1·2^16 + vn0`
/// (`vn1 >= 2^15`), when the quotient is one digit (Knuth's step D3): the estimate `num / vn1`,
/// with remainder `rhat`, is at most two above the digit, and is lowered while it is `2^16` or
/// more or `q·vn0 > rhat·2^16 + low` (both sides below `2^32` whenever the first test fails),
/// the second time only while `rhat + vn1` stays below `2^16`.
fn digit(
    fb: &mut FunctionBuilder<'_>,
    n: u8,
    num: SValue,
    low: SValue,
    vn1: SValue,
    vn0: SValue,
) -> BResult<SValue> {
    let over = |fb: &mut FunctionBuilder<'_>, q: SValue, rhat: SValue| -> BResult<SValue> {
        let big = bink(fb, n, BinOp::UGreaterThanEqual, q, 0x1_0000)?;
        let lhs = bin(fb, BinOp::IMul, q, vn0)?;
        let rhs = bink(fb, n, BinOp::ShiftLeftLogical, rhat, 16)?;
        let rhs = bin(fb, BinOp::IAdd, rhs, low)?;
        let gt = bin(fb, BinOp::UGreaterThan, lhs, rhs)?;
        bin(fb, BinOp::LogicalOr, big, gt)
    };
    let q = bin(fb, BinOp::UDiv, num, vn1)?;
    let rhat = bin(fb, BinOp::UMod, num, vn1)?;
    let first = over(fb, q, rhat)?;
    let dec = one_if(fb, n, first)?;
    let q = bin(fb, BinOp::ISub, q, dec)?;
    let z = k(fb, n, 0)?;
    let add = fb.select(first, vn1, z)?;
    let rhat = bin(fb, BinOp::IAdd, rhat, add)?;
    // Without a first correction `q` and `rhat` are unchanged and the test fails again.
    let small = bink(fb, n, BinOp::ULessThan, rhat, 0x1_0000)?;
    let again = over(fb, q, rhat)?;
    let second = bin(fb, BinOp::LogicalAnd, small, again)?;
    let dec = one_if(fb, n, second)?;
    bin(fb, BinOp::ISub, q, dec)
}

// ---- The shared division routine ----

/// The census name of a wide division, whose routine's words it is charged once.
pub const DIVIDE: &str = "arith.wide.divide";

/// A module's shared unsigned division routine of wide operands of `lanes` lanes: `fn(a, b) -> {a /
/// b, a % b}` ([`define_divide`]). Every wide `udiv`, `urem`, `sdiv`, and `srem` on that many lanes
/// calls it, whatever its width: the routine divides the two-word values exactly as 64-bit ones,
/// and a width's zero-extended values stay below `2^bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Divide {
    pub lanes: u8,
}

impl Divide {
    /// The routine of wide operands of form `f`.
    pub fn of(f: Form) -> Divide {
        Divide { lanes: f.lanes }
    }

    /// An upper estimate of the routine's SPIR-V words, counted once per module: about 125
    /// straight-line instructions of 4 to 7 words, and its constants. Measured at most 716 words,
    /// at 4 lanes (`wide_division_words_bound_their_emission`).
    pub fn words(self) -> u32 {
        768
    }
}

/// Define `d`'s routine in `mb`: a function of two unsigned two-word values of its lanes,
/// returning the struct of their quotient and remainder (the compact division above).
pub fn define_divide(mb: &mut ModuleBuilder, d: Divide) -> BResult<FuncId> {
    let held = legal::wide_demote(d.lanes);
    let both = DemoteTy::Composite(vec![held.clone(), held.clone()].into());
    let (Some(t), Some(ret)) = (legal::spirv_type(mb, &held)?, legal::spirv_type(mb, &both)?)
    else {
        return Err(BuildError::Type(TypeRule::FunctionSignature));
    };
    let mut fb = mb.routine(ret)?;
    let (a, b) = (fb.param(t)?, fb.param(t)?);
    let (x, y) = (split(&mut fb, a)?, split(&mut fb, b)?);
    let (q, r) = udivrem(&mut fb, d.lanes, x, y)?;
    let f = wide_form(64, d.lanes);
    let (q, r) = (join(&mut fb, f, q)?, join(&mut fb, f, r)?);
    let v = fb.composite_construct(ret, &[q, r])?;
    fb.ret(Some(v))?;
    fb.finish()
}

/// Whether `op` is a division or remainder (a wide one calls [`Divide`]'s routine).
fn is_division(op: AirBin) -> bool {
    matches!(
        op,
        AirBin::UDiv | AirBin::URem | AirBin::SDiv | AirBin::SRem
    )
}

/// Division or remainder `op` on wide values of form `f`, through `routine`, the form's
/// [`define_divide`]. An unsigned one is the call (its quotient `q <= a` and remainder `r < b`,
/// or `q = a` and `r = 0` for a zero divisor, stay below `2^bits`); a signed one divides the
/// magnitudes and negates the quotient where the operands' signs differ, the remainder where the
/// dividend is negative.
pub fn divide(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    a: SValue,
    b: SValue,
    routine: FuncId,
) -> BResult<SValue> {
    let call = |fb: &mut FunctionBuilder<'_>, a: SValue, b: SValue, rem: bool| {
        let qr = fb.call(routine, &[a, b])?.ok_or_else(missing_routine)?;
        fb.composite_extract(qr, &[u32::from(rem)])
    };
    let n = f.lanes;
    match op {
        AirBin::UDiv | AirBin::URem => call(fb, a, b, op == AirBin::URem),
        AirBin::SDiv | AirBin::SRem => {
            let (x, y) = (split(fb, a)?, split(fb, b)?);
            let (x, y) = (sext_top(fb, f, x)?, sext_top(fb, f, y)?);
            let (mx, nx) = abs(fb, n, x)?;
            let (my, ny) = abs(fb, n, y)?;
            let (mx, my) = (join(fb, f, mx)?, join(fb, f, my)?);
            let m = call(fb, mx, my, op == AirBin::SRem)?;
            let m = split(fb, m)?;
            let flip = if op == AirBin::SDiv {
                bin(fb, BinOp::LogicalNotEqual, nx, ny)?
            } else {
                nx
            };
            let nm = neg(fb, n, m)?;
            let r = sel(fb, flip, nm, m)?;
            let r = mask_top(fb, f, r)?;
            join(fb, f, r)
        }
        _ => Err(arity_error()),
    }
}

/// A division without its routine, or a routine without a result.
fn missing_routine() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::FunctionCall,
        rule: OperandRule::Callee,
    }
}

// ---- Operations ----

/// An integer binary operator other than a division on wide values of form `f` (a division is
/// [`divide`]).
pub fn binary(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    a: SValue,
    b: SValue,
) -> BResult<SValue> {
    let n = f.lanes;
    let (x, y) = (split(fb, a)?, split(fb, b)?);
    let r = match op {
        AirBin::Add => add(fb, x, y)?,
        AirBin::Sub => sub(fb, x, y)?,
        AirBin::Mul => mul(fb, x, y)?,
        AirBin::And => per_word(fb, BinOp::BitwiseAnd, x, y)?,
        AirBin::Or => per_word(fb, BinOp::BitwiseOr, x, y)?,
        AirBin::Xor => per_word(fb, BinOp::BitwiseXor, x, y)?,
        // A shift by the width or more is poison: any amount's low six bits will do.
        AirBin::Shl => shl(fb, n, x, y.lo)?,
        AirBin::LShr => lshr(fb, n, x, y.lo)?,
        AirBin::AShr => {
            let x = sext_top(fb, f, x)?;
            shift(fb, n, Shift::Arithmetic, x, y.lo)?
        }
        AirBin::UDiv
        | AirBin::URem
        | AirBin::SDiv
        | AirBin::SRem
        | AirBin::FAdd
        | AirBin::FSub
        | AirBin::FMul
        | AirBin::FDiv
        | AirBin::FRem => return Err(arity_error()),
    };
    let r = mask_top(fb, f, r)?;
    join(fb, f, r)
}

/// `icmp` on wide values of form `f`.
pub fn int_compare(
    fb: &mut FunctionBuilder<'_>,
    pred: IntPredicate,
    f: Form,
    a: SValue,
    b: SValue,
) -> BResult<SValue> {
    use IntPredicate as P;
    let (mut x, mut y) = (split(fb, a)?, split(fb, b)?);
    if matches!(pred, P::Sgt | P::Sge | P::Slt | P::Sle) {
        x = sext_top(fb, f, x)?;
        y = sext_top(fb, f, y)?;
    }
    compare(fb, pred, x, y)
}

/// Integer `x` of form `from` (`bool`, a 32-bit container, or wide) as a 64-bit pair,
/// sign-extended when `signed`.
fn to_pair(fb: &mut FunctionBuilder<'_>, from: Form, x: SValue, signed: bool) -> BResult<Pair> {
    let n = from.lanes;
    match from.held {
        Held::Bool => {
            let one = pk(fb, n, if signed { u64::MAX } else { 1 })?;
            let z = pk(fb, n, 0)?;
            sel(fb, x, one, z)
        }
        Held::Int { container: 32, .. } => {
            if signed {
                let lo = ops::sext(fb, from, x)?;
                let hi = bink(fb, n, BinOp::ShiftRightArithmetic, lo, 31)?;
                Ok(Pair { lo, hi })
            } else {
                Ok(Pair {
                    lo: x,
                    hi: k(fb, n, 0)?,
                })
            }
        }
        Held::Wide { .. } => {
            let p = split(fb, x)?;
            if signed {
                sext_top(fb, from, p)
            } else {
                Ok(p)
            }
        }
        Held::Int { .. } | Held::Float(_) => Err(arity_error()),
    }
}

/// Pair `p` (of `to`'s lanes) as a value of integer form `to`: truncated to its width.
fn from_pair(fb: &mut FunctionBuilder<'_>, to: Form, p: Pair) -> BResult<SValue> {
    match to.held {
        Held::Bool => {
            let one = k(fb, to.lanes, 1)?;
            let b = bin(fb, BinOp::BitwiseAnd, p.lo, one)?;
            let z = k(fb, to.lanes, 0)?;
            bin(fb, BinOp::INotEqual, b, z)
        }
        Held::Int { container: 32, .. } => ops::mask(fb, to, p.lo),
        Held::Wide { .. } => {
            let p = mask_top(fb, to, p)?;
            join(fb, to, p)
        }
        Held::Int { .. } | Held::Float(_) => Err(arity_error()),
    }
}

/// The float form of `f`'s container (a narrow float's `f32`).
fn container_float(f: Form) -> Form {
    f.with(Held::Float(if f.container() == 64 {
        FloatFmt::F64
    } else {
        FloatFmt::F32
    }))
}

/// Float `x` (form `from`) truncated toward zero to a wide integer of form `to` (LLVM
/// `fptoui`/`fptosi`: a value out of range is poison). The truncated magnitude `m` splits as
/// `h * 2^32 + r` with `h = trunc(m * 2^-32)`: the scaling is exact, `h * 2^32` is exact, and
/// `r = m - h * 2^32` is exact (the low bits of `m`'s significand), so both words convert
/// exactly.
fn float_to_pair(
    fb: &mut FunctionBuilder<'_>,
    from: Form,
    to: Form,
    x: SValue,
    signed: bool,
) -> BResult<Pair> {
    let n = to.lanes;
    let cf = container_float(from);
    let m = if signed {
        fb.glsl(GlslOp::FAbs, &[x])?
    } else {
        x
    };
    let down = ops::fsplat(fb, cf, 2f64.powi(-32))?;
    let up = ops::fsplat(fb, cf, 2f64.powi(32))?;
    let h = bin(fb, BinOp::FMul, m, down)?;
    let h = fb.glsl(GlslOp::Trunc, &[h])?;
    let hh = bin(fb, BinOp::FMul, h, up)?;
    let r = bin(fb, BinOp::FSub, m, hh)?;
    let ut = ops::ty(fb, words_form(n))?;
    let hi = fb.convert(ConvOp::ConvertFToU, h, ut)?;
    let lo = fb.convert(ConvOp::ConvertFToU, r, ut)?;
    let p = Pair { lo, hi };
    if !signed {
        return Ok(p);
    }
    let z = ops::fsplat(fb, cf, 0.0)?;
    let is_neg = bin(fb, BinOp::FOrdLessThan, x, z)?;
    let np = neg(fb, n, p)?;
    sel(fb, is_neg, np, p)
}

/// Wide `x` (form `from`, read as signed when `signed`) converted to float form `to`, rounding
/// once: to `double` as `hi * 2^32 + lo` (both exact, one rounding add); otherwise the magnitude
/// shifted right by `s` (so it has at most 32 significant bits) with the bits shifted out folded
/// into a sticky lowest bit, converted (to odd for `bfloat`, by `dir` for an `f32` rounded toward
/// zero or an infinity, [`rounding::magnitude_f32`], which the sticky bit keeps exact), and scaled
/// back by `2^s` exactly. A magnitude below `2^32` takes `s = 0`, so below `2^24` the `f32` is
/// exact and a `half` rounds it by integer arithmetic ([`narrow::round_integer`], or
/// [`rounding::to_half`] by `dir`).
fn pair_to_float(
    fb: &mut FunctionBuilder<'_>,
    from: Form,
    to: Form,
    x: SValue,
    (signed, dir): (bool, Option<rounding::Directed>),
) -> BResult<SValue> {
    let n = from.lanes;
    let p = to_pair(fb, from, x, signed)?;
    let (m, is_neg) = if signed {
        let (m, s) = abs(fb, n, p)?;
        (m, Some(s))
    } else {
        (p, None)
    };
    let t = ops::ty(fb, to)?;
    let r = if to.float_fmt() == Some(FloatFmt::F64) {
        if dir.is_some() {
            return Err(arity_error());
        }
        let fh = fb.convert(ConvOp::ConvertUToF, m.hi, t)?;
        let fl = fb.convert(ConvOp::ConvertUToF, m.lo, t)?;
        let up = ops::fsplat(fb, to, 2f64.powi(32))?;
        let fh = bin(fb, BinOp::FMul, fh, up)?;
        bin(fb, BinOp::FAdd, fh, fl)?
    } else {
        // s = the high word's bit length (FindUMsb of 0 is -1).
        let msb = fb.glsl(GlslOp::FindUMsb, &[m.hi])?;
        let s = bink(fb, n, BinOp::IAdd, msb, 1)?;
        let t32 = lshr(fb, n, m, s)?;
        let back = shl(fb, n, t32, s)?;
        let lost = compare(fb, IntPredicate::Ne, back, m)?;
        let one = k(fb, n, 1)?;
        let z = k(fb, n, 0)?;
        let sticky = fb.select(lost, one, z)?;
        let m32 = bin(fb, BinOp::BitwiseOr, t32.lo, sticky)?;
        let f32f = to.with(Held::Float(FloatFmt::F32));
        let f = match (to.float_fmt(), dir) {
            (Some(FloatFmt::BFloat), _) => narrow::int_to_f32_odd(fb, words_form(n), to, m32)?,
            (Some(FloatFmt::F32), Some(d)) => {
                rounding::magnitude_f32(fb, words_form(n), to, (m32, is_neg), d)?
            }
            _ => {
                let ft = ops::ty(fb, f32f)?;
                fb.convert(ConvOp::ConvertUToF, m32, ft)?
            }
        };
        fb.glsl(GlslOp::Ldexp, &[f, s])?
    };
    let r = match is_neg {
        Some(c) => {
            let nr = fb.unary(UnOp::FNegate, r)?;
            fb.select(c, nr, r)?
        }
        None => r,
    };
    match (to.float_fmt(), dir) {
        (Some(FloatFmt::Half), Some(d)) => {
            rounding::to_half(fb, to.with(Held::Float(FloatFmt::F32)), r, d)
        }
        _ => narrow::round_integer(fb, to, r),
    }
}

/// A numeric cast with a wide side.
pub fn cast(
    fb: &mut FunctionBuilder<'_>,
    kind: CastKind,
    from: Form,
    to: Form,
    x: SValue,
) -> BResult<SValue> {
    match kind {
        CastKind::Trunc | CastKind::ZExt | CastKind::SExt => {
            let p = to_pair(fb, from, x, kind == CastKind::SExt)?;
            from_pair(fb, to, p)
        }
        CastKind::FpToUi | CastKind::FpToSi => {
            let p = float_to_pair(fb, from, to, x, kind == CastKind::FpToSi)?;
            from_pair(fb, to, p)
        }
        CastKind::UiToFp | CastKind::SiToFp => {
            pair_to_float(fb, from, to, x, (kind == CastKind::SiToFp, None))
        }
        CastKind::FpResize => Err(arity_error()),
    }
}

/// `air.convert` with a wide side (saturating from a float, as `math::convert`).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    kind: ConvertKind,
    from: Form,
    to: Form,
    x: SValue,
) -> BResult<SValue> {
    match kind {
        ConvertKind::Int { signed } => {
            let signed = signed && !from.is_bool();
            let p = to_pair(fb, from, x, signed)?;
            from_pair(fb, to, p)
        }
        ConvertKind::ToFloat { signed } => pair_to_float(fb, from, to, x, (signed, None)),
        ConvertKind::ToFloatDirected { signed, dir } => {
            pair_to_float(fb, from, to, x, (signed, Some(dir)))
        }
        // Clamped as pairs to the range both share, then truncated or extended.
        ConvertKind::SatInt {
            src_signed,
            dst_signed,
        } => {
            use IntPredicate as P;
            let n = from.lanes;
            let mut p = to_pair(fb, from, x, src_signed)?;
            let (lo, hi) = rounding::sat_bounds((from.bits(), src_signed), (to.bits(), dst_signed));
            let (lt, gt) = if src_signed {
                (P::Slt, P::Sgt)
            } else {
                (P::Ult, P::Ugt)
            };
            if let Some(l) = lo {
                let lp = pk(fb, n, l as i64 as u64)?;
                let below = compare(fb, lt, p, lp)?;
                p = sel(fb, below, lp, p)?;
            }
            if let Some(h) = hi {
                let hp = pk(fb, n, h as i64 as u64)?;
                let above = compare(fb, gt, p, hp)?;
                p = sel(fb, above, hp, p)?;
            }
            from_pair(fb, to, p)
        }
        ConvertKind::ToInt { signed, integral } => {
            // Rounded to an integral value first; the NaN test reads the operand itself.
            let v = rounding::integral(fb, integral, x)?;
            let nb = u32::from(to.bits());
            let fw = from.container();
            let kk = if signed { nb - 1 } else { nb };
            // Clamp into [lo, the largest float below 2^k] so the conversion is in range.
            let lo = if signed {
                let lo = math::pow2_bits(fw, kk);
                let (mb, _, top) = math::float_layout(fw);
                let lo = if lo >> mb == top as u64 {
                    math::below_pow2_bits(fw, kk)
                } else {
                    lo
                };
                ops::splat(fb, from, lo | (1u64 << (fw - 1)))?
            } else {
                ops::fsplat(fb, from, 0.0)?
            };
            let hi = ops::splat(fb, from, math::below_pow2_bits(fw, kk))?;
            let c = fb.glsl(GlslOp::NClamp, &[v, lo, hi])?;
            let n = to.lanes;
            let mut p = float_to_pair(fb, from, to, c, signed)?;
            let limit = ops::splat(fb, from, math::pow2_bits(fw, kk))?;
            let over = bin(fb, BinOp::FOrdGreaterThanEqual, v, limit)?;
            let max = pk(fb, n, ops::ones(kk as u8))?;
            p = sel(fb, over, max, p)?;
            if signed {
                let under = bin(fb, BinOp::FOrdLessThan, v, lo)?;
                let min = pk(fb, n, (!0u64) << (nb - 1))?;
                p = sel(fb, under, min, p)?;
                let nan = fb.unary(UnOp::IsNan, x)?;
                let z = pk(fb, n, 0)?;
                p = sel(fb, nan, z, p)?;
            }
            from_pair(fb, to, p)
        }
        // A wide value is held zero-extended, so it is zero exactly when both words are.
        ConvertKind::IntToBool => {
            let p = split(fb, x)?;
            let z = pk(fb, from.lanes, 0)?;
            compare(fb, IntPredicate::Ne, p, z)
        }
        // The value as an `f32` (exact), then the saturating conversion.
        ConvertKind::Fp8ToInt { fp8, signed } => {
            let f = from.with(Held::Float(FloatFmt::F32));
            let v = narrow::fp8_to_f32(fb, fp8, narrow::u32_form(from), f, x)?;
            let k = ConvertKind::ToInt {
                signed,
                integral: rounding::Integral::Truncate,
            };
            convert(fb, k, f, to, v)
        }
        ConvertKind::Float
        | ConvertKind::FloatDirected(_)
        | ConvertKind::FromFp8(_)
        | ConvertKind::ToBool => Err(arity_error()),
    }
}

/// The words of wide value `x` of form `f`, lane by lane, low word first (`2 * lanes` words).
pub fn to_words(fb: &mut FunctionBuilder<'_>, f: Form, x: SValue) -> BResult<Vec<SValue>> {
    let p = split(fb, x)?;
    if f.lanes == 1 {
        return Ok(vec![p.lo, p.hi]);
    }
    let mut out = Vec::with_capacity(2 * usize::from(f.lanes));
    for i in 0..u32::from(f.lanes) {
        out.push(fb.composite_extract(p.lo, &[i])?);
        out.push(fb.composite_extract(p.hi, &[i])?);
    }
    Ok(out)
}

/// The wide value of form `f` from its words, lane by lane, low word first (a missing high word
/// is zero); the bits above the width are cleared.
pub fn from_words(fb: &mut FunctionBuilder<'_>, f: Form, ws: &[SValue]) -> BResult<SValue> {
    let word = |fb: &mut FunctionBuilder<'_>, i: usize| -> BResult<SValue> {
        match ws.get(i) {
            Some(w) => Ok(*w),
            None => k(fb, 1, 0),
        }
    };
    let p = if f.lanes == 1 {
        Pair {
            lo: word(fb, 0)?,
            hi: word(fb, 1)?,
        }
    } else {
        let t = ops::ty(fb, words_form(f.lanes))?;
        let mut los = Vec::with_capacity(usize::from(f.lanes));
        let mut his = Vec::with_capacity(usize::from(f.lanes));
        for i in 0..usize::from(f.lanes) {
            los.push(word(fb, 2 * i)?);
            his.push(word(fb, 2 * i + 1)?);
        }
        Pair {
            lo: fb.composite_construct(t, &los)?,
            hi: fb.composite_construct(t, &his)?,
        }
    };
    let p = mask_top(fb, f, p)?;
    join(fb, f, p)
}

/// A bitcast with a wide side through 32-bit words (both sides 64 bits per wide lane).
pub fn bitcast(fb: &mut FunctionBuilder<'_>, from: Form, to: Form, x: SValue) -> BResult<SValue> {
    let total = u32::from(from.bits()) * u32::from(from.lanes);
    let n = u8::try_from(total / 32).map_err(|_| arity_error())?;
    let ws = if from.is_wide() {
        to_words(fb, from, x)?
    } else {
        ops::pack_words(fb, from, n, x)?
    };
    if to.is_wide() {
        from_words(fb, to, &ws)
    } else {
        ops::unpack_words(fb, to, &ws)
    }
}

/// The low word of wide scalar `x`, or of a 64-bit integer (the index word of a GEP).
pub fn low_word(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let t = fb.value_ty(x)?;
    if matches!(fb.module().types().get(t)?, Type::Vector { .. }) {
        return fb.composite_extract(x, &[0]);
    }
    let u = ops::ty(fb, words_form(1))?;
    fb.convert(ConvOp::UConvert, x, u)
}

/// `u32` value `w` zero-extended to integer form `f` (a 32- or 64-bit container, or wide).
pub fn zext_word(fb: &mut FunctionBuilder<'_>, f: Form, w: SValue) -> BResult<SValue> {
    if f.is_wide() {
        let z = k(fb, f.lanes, 0)?;
        return join(fb, f, Pair { lo: w, hi: z });
    }
    if f.container() == 32 {
        return Ok(w);
    }
    let t = ops::ty(fb, f)?;
    fb.convert(ConvOp::UConvert, w, t)
}

/// A `u32` result `r` (of `f`'s lanes) as a wide value of form `f`.
fn widened(fb: &mut FunctionBuilder<'_>, f: Form, r: SValue) -> BResult<SValue> {
    let z = k(fb, f.lanes, 0)?;
    join(fb, f, Pair { lo: r, hi: z })
}

/// `u32` operand `v` (a scalar) in every lane of `n`.
fn broadcast(fb: &mut FunctionBuilder<'_>, n: u8, v: SValue) -> BResult<SValue> {
    if n == 1 {
        return Ok(v);
    }
    let t = ops::ty(fb, words_form(n))?;
    fb.composite_construct(t, &vec![v; usize::from(n)])
}

/// The byte-reversed word `w`.
fn bswap32(fb: &mut FunctionBuilder<'_>, n: u8, w: SValue) -> BResult<SValue> {
    let mut acc: Option<SValue> = None;
    for i in 0..4u32 {
        let b = bink(fb, n, BinOp::ShiftRightLogical, w, 8 * i)?;
        let b = bink(fb, n, BinOp::BitwiseAnd, b, 0xFF)?;
        let b = bink(fb, n, BinOp::ShiftLeftLogical, b, 8 * (3 - i))?;
        acc = Some(match acc {
            None => b,
            Some(a) => bin(fb, BinOp::BitwiseOr, a, b)?,
        });
    }
    acc.ok_or_else(arity_error)
}

/// The high 64 bits of the 128-bit product of unsigned pairs `a` and `b`.
fn mul_hi_u(fb: &mut FunctionBuilder<'_>, n: u8, a: Pair, b: Pair) -> BResult<Pair> {
    let mx = |fb: &mut FunctionBuilder<'_>, x, y| -> BResult<Pair> {
        let p = bin(fb, BinOp::UMulExtended, x, y)?;
        Ok(Pair {
            lo: fb.composite_extract(p, &[0])?,
            hi: fb.composite_extract(p, &[1])?,
        })
    };
    let z = k(fb, n, 0)?;
    let p00 = mx(fb, a.lo, b.lo)?;
    let p01 = mx(fb, a.lo, b.hi)?;
    let p10 = mx(fb, a.hi, b.lo)?;
    let p11 = mx(fb, a.hi, b.hi)?;
    // Word 1 of the product: p00.hi + p01.lo + p10.lo, whose carries go to words 2 and 3.
    let s1 = add(fb, Pair { lo: p00.hi, hi: z }, Pair { lo: p01.lo, hi: z })?;
    let s1 = add(fb, s1, Pair { lo: p10.lo, hi: z })?;
    // Words 2 and 3: p11 + p01.hi + p10.hi + the carry of word 1.
    let r = add(fb, p11, Pair { lo: p01.hi, hi: z })?;
    let r = add(fb, r, Pair { lo: p10.hi, hi: z })?;
    add(fb, r, Pair { lo: s1.hi, hi: z })
}

/// The high 64 bits of the 128-bit product of pairs `a` and `b`: unsigned, or, when `signed`, of
/// the sign-extended pairs read as two's complement (the unsigned high half less `b` where
/// `a < 0` and less `a` where `b < 0`, modulo `2^64`).
fn mul_high(fb: &mut FunctionBuilder<'_>, n: u8, a: Pair, b: Pair, signed: bool) -> BResult<Pair> {
    let mut hi = mul_hi_u(fb, n, a, b)?;
    if signed {
        let z = pk(fb, n, 0)?;
        let an = negative(fb, n, a)?;
        let bn = negative(fb, n, b)?;
        let tb = sel(fb, an, b, z)?;
        let ta = sel(fb, bn, a, z)?;
        hi = sub(fb, hi, tb)?;
        hi = sub(fb, hi, ta)?;
    }
    Ok(hi)
}

/// The 128-bit product of pairs `a` and `b` as its low and high 64 bits ([`mul_high`]).
fn mul_full(
    fb: &mut FunctionBuilder<'_>,
    n: u8,
    a: Pair,
    b: Pair,
    signed: bool,
) -> BResult<(Pair, Pair)> {
    Ok((mul(fb, a, b)?, mul_high(fb, n, a, b, signed)?))
}

/// Bits `[bits, 2 * bits)` of the 128-bit value `(lo, hi)` (unmasked above `bits`).
fn product_high(
    fb: &mut FunctionBuilder<'_>,
    n: u8,
    bits: u8,
    lo: Pair,
    hi: Pair,
) -> BResult<Pair> {
    if bits >= 64 {
        return Ok(hi);
    }
    let down = lshr_const(fb, n, lo, u32::from(bits))?;
    let up = shl_const(fb, n, hi, u32::from(64 - bits))?;
    per_word(fb, BinOp::BitwiseOr, down, up)
}

/// `x mod bits` of pair `x` as a word, for a width of 33 to 64 bits: its low six bits at 64, else
/// `((hi mod bits) * (2^32 mod bits) + lo mod bits) mod bits`, in which no word overflows.
fn amount_mod(fb: &mut FunctionBuilder<'_>, n: u8, bits: u8, x: Pair) -> BResult<SValue> {
    if bits >= 64 {
        return bink(fb, n, BinOp::BitwiseAnd, x.lo, 63);
    }
    let w = u32::from(bits);
    let hm = bink(fb, n, BinOp::UMod, x.hi, w)?;
    let hm = bink(fb, n, BinOp::IMul, hm, ((1u64 << 32) % u64::from(w)) as u32)?;
    let lm = bink(fb, n, BinOp::UMod, x.lo, w)?;
    let s = bin(fb, BinOp::IAdd, hm, lm)?;
    bink(fb, n, BinOp::UMod, s, w)
}

/// `a * b + c` saturated to the width of `f` from the exact 128-bit result (`a`, `b`, `c` the
/// operands' pairs, sign-extended when `signed`).
fn mad_sat(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    signed: bool,
    a: Pair,
    b: Pair,
    c: Pair,
) -> BResult<Pair> {
    let n = f.lanes;
    let bits = f.bits();
    let (lo, hi) = mul_full(fb, n, a, b, signed)?;
    let lo = add(fb, lo, c)?;
    let carry = compare(fb, IntPredicate::Ult, lo, c)?;
    let one = pk(fb, n, 1)?;
    let z = pk(fb, n, 0)?;
    let cp = sel(fb, carry, one, z)?;
    let mut hi = add(fb, hi, cp)?;
    if !signed {
        let max = pk(fb, n, ops::ones(bits))?;
        let zero = compare(fb, IntPredicate::Eq, hi, z)?;
        let fits = if bits < 64 {
            let le = compare(fb, IntPredicate::Ule, lo, max)?;
            bin(fb, BinOp::LogicalAnd, zero, le)?
        } else {
            zero
        };
        return sel(fb, fits, lo, max);
    }
    // A negative addend adds its sign extension, -1, to the high half.
    let cn = negative(fb, n, c)?;
    let all = pk(fb, n, u64::MAX)?;
    let ext = sel(fb, cn, all, z)?;
    hi = add(fb, hi, ext)?;
    // The result is one 64-bit value when the high half is the low half's sign extension.
    let s = bink(fb, n, BinOp::ShiftRightArithmetic, lo.hi, 31)?;
    let fits = compare(fb, IntPredicate::Eq, hi, Pair { lo: s, hi: s })?;
    let min = pk(fb, n, (!0u64) << (bits - 1))?;
    let max = pk(fb, n, ops::ones(bits - 1))?;
    let under = compare(fb, IntPredicate::Slt, lo, min)?;
    let r = sel(fb, under, min, lo)?;
    let over = compare(fb, IntPredicate::Slt, max, r)?;
    let r = sel(fb, over, max, r)?;
    let neg = negative(fb, n, hi)?;
    let sat = sel(fb, neg, min, max)?;
    sel(fb, fits, r, sat)
}

/// Integer function `op` on wide values of form `f` (the implemented set: [`int_fn_lowered`]).
pub fn int_fn(
    fb: &mut FunctionBuilder<'_>,
    op: IntFn,
    f: Form,
    args: &[SValue],
) -> BResult<SValue> {
    let n = f.lanes;
    let arg = |i: usize| -> BResult<SValue> { args.get(i).copied().ok_or_else(arity_error) };
    let signed = |s: Sign| s == Sign::Signed;
    let a = split(fb, arg(0)?)?;
    let ext = |fb: &mut FunctionBuilder<'_>, s: Sign, p: Pair| -> BResult<Pair> {
        if signed(s) {
            sext_top(fb, f, p)
        } else {
            Ok(p)
        }
    };
    let ext_arg = |fb: &mut FunctionBuilder<'_>, s: Sign, i: usize| -> BResult<Pair> {
        let p = split(fb, arg(i)?)?;
        ext(fb, s, p)
    };
    let lt = |s: Sign| {
        if signed(s) {
            IntPredicate::Slt
        } else {
            IntPredicate::Ult
        }
    };
    let bits = f.bits();
    let r: Pair = match op {
        IntFn::Abs => {
            let x = sext_top(fb, f, a)?;
            abs(fb, n, x)?.0
        }
        IntFn::AbsDiff(s) => {
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let c = compare(fb, lt(s), y, x)?;
            let d1 = sub(fb, x, y)?;
            let d2 = sub(fb, y, x)?;
            sel(fb, c, d1, d2)?
        }
        IntFn::Min(s) | IntFn::Max(s) | IntFn::Min3(s) | IntFn::Max3(s) => {
            let max = matches!(op, IntFn::Max(_) | IntFn::Max3(_));
            let pick = |fb: &mut FunctionBuilder<'_>, x: Pair, y: Pair| -> BResult<Pair> {
                let c = compare(fb, lt(s), x, y)?;
                if max {
                    sel(fb, c, y, x)
                } else {
                    sel(fb, c, x, y)
                }
            };
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let mut r = pick(fb, x, y)?;
            if matches!(op, IntFn::Min3(_) | IntFn::Max3(_)) {
                let z = ext_arg(fb, s, 2)?;
                r = pick(fb, r, z)?;
            }
            r
        }
        IntFn::Median3(s) => {
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let z = ext_arg(fb, s, 2)?;
            let c = compare(fb, lt(s), x, y)?;
            let (lo, hi) = (sel(fb, c, x, y)?, sel(fb, c, y, x)?);
            let c = compare(fb, lt(s), hi, z)?;
            let mid = sel(fb, c, hi, z)?;
            let c = compare(fb, lt(s), lo, mid)?;
            sel(fb, c, mid, lo)?
        }
        IntFn::Clamp(s) => {
            let x = ext(fb, s, a)?;
            let lo = ext_arg(fb, s, 1)?;
            let hi = ext_arg(fb, s, 2)?;
            let below = compare(fb, lt(s), x, lo)?;
            let r = sel(fb, below, lo, x)?;
            let above = compare(fb, lt(s), hi, r)?;
            sel(fb, above, hi, r)?
        }
        IntFn::AddSat(s) | IntFn::SubSat(s) => {
            let add_op = matches!(op, IntFn::AddSat(_));
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let r = if add_op {
                add(fb, x, y)?
            } else {
                sub(fb, x, y)?
            };
            if !signed(s) {
                if add_op {
                    // Below 64 bits the sum cannot wrap: clamp to the width's maximum.
                    let max = pk(fb, n, ops::ones(bits))?;
                    let over = if bits < 64 {
                        compare(fb, IntPredicate::Ult, max, r)?
                    } else {
                        compare(fb, IntPredicate::Ult, r, x)?
                    };
                    sel(fb, over, max, r)?
                } else {
                    let under = compare(fb, IntPredicate::Ult, x, y)?;
                    let z = pk(fb, n, 0)?;
                    sel(fb, under, z, r)?
                }
            } else {
                let min = pk(fb, n, (!0u64) << (bits - 1))?;
                let max = pk(fb, n, ops::ones(bits - 1))?;
                if bits < 64 {
                    // One bit more than the width: no wrap; clamp to [MIN, MAX].
                    let under = compare(fb, IntPredicate::Slt, r, min)?;
                    let r = sel(fb, under, min, r)?;
                    let over = compare(fb, IntPredicate::Slt, max, r)?;
                    sel(fb, over, max, r)?
                } else {
                    // Overflow iff the result's sign differs from what the operands allow.
                    let xs = bin(fb, BinOp::BitwiseXor, x.hi, r.hi)?;
                    let t = if add_op {
                        bin(fb, BinOp::BitwiseXor, y.hi, r.hi)?
                    } else {
                        bin(fb, BinOp::BitwiseXor, x.hi, y.hi)?
                    };
                    let both = bin(fb, BinOp::BitwiseAnd, xs, t)?;
                    let z = k(fb, n, 0)?;
                    let over = bin(fb, BinOp::SLessThan, both, z)?;
                    let xn = negative(fb, n, x)?;
                    let sat = sel(fb, xn, min, max)?;
                    sel(fb, over, sat, r)?
                }
            }
        }
        IntFn::HAdd(s) | IntFn::RHAdd(s) => {
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let one = k(fb, n, 1)?;
            let kind = if signed(s) {
                Shift::Arithmetic
            } else {
                Shift::Logical
            };
            let hx = shift(fb, n, kind, x, one)?;
            let hy = shift(fb, n, kind, y, one)?;
            let both = if matches!(op, IntFn::HAdd(_)) {
                bin(fb, BinOp::BitwiseAnd, x.lo, y.lo)?
            } else {
                bin(fb, BinOp::BitwiseOr, x.lo, y.lo)?
            };
            let carry = bin(fb, BinOp::BitwiseAnd, both, one)?;
            let z = k(fb, n, 0)?;
            let r = add(fb, hx, hy)?;
            add(fb, r, Pair { lo: carry, hi: z })?
        }
        IntFn::MulHi(s) | IntFn::MadHi(s) => {
            // Bits [N, 2N) of the exact product of the operands (sign-extended when signed).
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let mut h = if bits >= 64 {
                mul_high(fb, n, x, y, signed(s))?
            } else {
                let (lo, hi) = mul_full(fb, n, x, y, signed(s))?;
                product_high(fb, n, bits, lo, hi)?
            };
            if matches!(op, IntFn::MadHi(_)) {
                let c = split(fb, arg(2)?)?;
                h = add(fb, h, c)?;
            }
            h
        }
        IntFn::MadSat(s) => {
            let (x, y) = (ext(fb, s, a)?, ext_arg(fb, s, 1)?);
            let c = ext_arg(fb, s, 2)?;
            mad_sat(fb, f, signed(s), x, y, c)?
        }
        IntFn::ExtractBits(s) => {
            // (x << (64 - o - c)) >> (64 - c), 0 for a zero count (as `extract_bits64`).
            let o = broadcast(fb, n, arg(1)?)?;
            let c = broadcast(fb, n, arg(2)?)?;
            let z = k(fb, n, 0)?;
            let none = bin(fb, BinOp::IEqual, c, z)?;
            let k64 = k(fb, n, 64)?;
            let up = bin(fb, BinOp::ISub, k64, o)?;
            let up = bin(fb, BinOp::ISub, up, c)?;
            let down = bin(fb, BinOp::ISub, k64, c)?;
            let v = shl(fb, n, a, up)?;
            let kind = if signed(s) {
                Shift::Arithmetic
            } else {
                Shift::Logical
            };
            let v = shift(fb, n, kind, v, down)?;
            let zp = pk(fb, n, 0)?;
            sel(fb, none, zp, v)?
        }
        IntFn::InsertBits => {
            let y = split(fb, arg(1)?)?;
            let o = broadcast(fb, n, arg(2)?)?;
            let c = broadcast(fb, n, arg(3)?)?;
            let z = k(fb, n, 0)?;
            let none = bin(fb, BinOp::IEqual, c, z)?;
            let k64 = k(fb, n, 64)?;
            let down = bin(fb, BinOp::ISub, k64, c)?;
            let all = pk(fb, n, u64::MAX)?;
            let m = lshr(fb, n, all, down)?;
            let zp = pk(fb, n, 0)?;
            let m = sel(fb, none, zp, m)?;
            let m = shl(fb, n, m, o)?;
            let keep = Pair {
                lo: fb.unary(UnOp::Not, m.lo)?,
                hi: fb.unary(UnOp::Not, m.hi)?,
            };
            let keep = per_word(fb, BinOp::BitwiseAnd, a, keep)?;
            let ins = shl(fb, n, y, o)?;
            let ins = per_word(fb, BinOp::BitwiseAnd, ins, m)?;
            per_word(fb, BinOp::BitwiseOr, keep, ins)?
        }
        IntFn::Popcount => {
            let cl = fb.unary(UnOp::BitCount, a.lo)?;
            let ch = fb.unary(UnOp::BitCount, a.hi)?;
            let r = bin(fb, BinOp::IAdd, cl, ch)?;
            return widened(fb, f, r);
        }
        IntFn::ReverseBits => {
            let full = Pair {
                lo: fb.unary(UnOp::BitReverse, a.hi)?,
                hi: fb.unary(UnOp::BitReverse, a.lo)?,
            };
            let pad = k(fb, n, u32::from(64 - bits))?;
            lshr(fb, n, full, pad)?
        }
        IntFn::Bswap => {
            let full = Pair {
                lo: bswap32(fb, n, a.hi)?,
                hi: bswap32(fb, n, a.lo)?,
            };
            let pad = k(fb, n, u32::from(64 - bits))?;
            lshr(fb, n, full, pad)?
        }
        IntFn::Rotate | IntFn::Fshl | IntFn::Fshr => {
            // Every shift is in range: s = amount mod N, and the second shift is by N - 1 - s
            // after a shift by one (the bits above N are masked below).
            let (y, amount) = match op {
                IntFn::Rotate => (a, arg(1)?),
                _ => (split(fb, arg(1)?)?, arg(2)?),
            };
            let amount = split(fb, amount)?;
            let s = amount_mod(fb, n, bits, amount)?;
            let n1 = k(fb, n, u32::from(bits) - 1)?;
            let rest = bin(fb, BinOp::ISub, n1, s)?;
            if op == IntFn::Fshr {
                // (y >> s) | ((a << 1) << (N - 1 - s)).
                let lo = lshr(fb, n, y, s)?;
                let a1 = shl_const(fb, n, a, 1)?;
                let hi = shl(fb, n, a1, rest)?;
                per_word(fb, BinOp::BitwiseOr, hi, lo)?
            } else {
                // (a << s) | ((y >> 1) >> (N - 1 - s)).
                let hi = shl(fb, n, a, s)?;
                let y1 = lshr_const(fb, n, y, 1)?;
                let lo = lshr(fb, n, y1, rest)?;
                per_word(fb, BinOp::BitwiseOr, hi, lo)?
            }
        }
        IntFn::Clz => {
            // 31 - FindUMsb(w) is 32 for a zero word.
            let k31 = k(fb, n, 31)?;
            let mh = fb.glsl(GlslOp::FindUMsb, &[a.hi])?;
            let ch = bin(fb, BinOp::ISub, k31, mh)?;
            let ml = fb.glsl(GlslOp::FindUMsb, &[a.lo])?;
            let cl = bin(fb, BinOp::ISub, k31, ml)?;
            let cl = bink(fb, n, BinOp::IAdd, cl, 32)?;
            let z = k(fb, n, 0)?;
            let hz = bin(fb, BinOp::IEqual, a.hi, z)?;
            let r = fb.select(hz, cl, ch)?;
            let r = bink(fb, n, BinOp::ISub, r, u32::from(64 - bits))?;
            return widened(fb, f, r);
        }
        IntFn::Ctz => {
            let z = k(fb, n, 0)?;
            let ll = fb.glsl(GlslOp::FindILsb, &[a.lo])?;
            let lh = fb.glsl(GlslOp::FindILsb, &[a.hi])?;
            let lh = bink(fb, n, BinOp::IAdd, lh, 32)?;
            let lnz = bin(fb, BinOp::INotEqual, a.lo, z)?;
            let r = fb.select(lnz, ll, lh)?;
            let zero = compare(fb, IntPredicate::Eq, a, Pair { lo: z, hi: z })?;
            let w = k(fb, n, u32::from(bits))?;
            let r = fb.select(zero, w, r)?;
            return widened(fb, f, r);
        }
    };
    let r = mask_top(fb, f, r)?;
    join(fb, f, r)
}

/// Whether integer function `op` is lowered on wide values of form `f`: every one, `bswap` on
/// whole 16-bit units (as LLVM defines it).
pub fn int_fn_lowered(op: IntFn, f: Form) -> bool {
    match op {
        IntFn::Bswap => f.bits().is_multiple_of(16),
        _ => true,
    }
}

// ---- Strategy ----

/// A lowering of an instruction with a wide operand or result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WideOp {
    /// An integer binary operator on wide values of `form`.
    Bin { op: AirBin, form: Form },
    /// `icmp` on wide values of `form`.
    Compare { pred: IntPredicate, form: Form },
    /// A numeric cast with a wide side.
    Cast {
        kind: CastKind,
        from: Form,
        to: Form,
    },
    /// `air.convert` with a wide side.
    Convert {
        kind: ConvertKind,
        from: Form,
        to: Form,
    },
    /// A bitcast with a wide side, through 32-bit words.
    Bitcast { from: Form, to: Form },
    /// `select` of wide values of `form`: of the pair on a scalar (its condition is a scalar),
    /// word by word on a vector (its condition may be a vector).
    Select { form: Form },
    /// `extractelement` from a vector of form `vec` (wide or not): lane `at`, or the lane a
    /// dynamic index names (wide when `wide_index`).
    Extract {
        vec: Form,
        at: Option<u32>,
        wide_index: bool,
    },
    /// `insertelement` into a vector of form `vec`, as [`WideOp::Extract`].
    Insert {
        vec: Form,
        at: Option<u32>,
        wide_index: bool,
    },
    /// `shufflevector` of wide vectors into form `out`: result lane `i` is lane `mask[i]` of the
    /// concatenation.
    Shuffle { mask: Box<[u32]>, out: Form },
    /// An integer function on wide values of `form`.
    IntFn { f: IntFn, form: Form },
}

impl WideOp {
    /// The census name.
    pub fn name(&self) -> &'static str {
        match self {
            WideOp::Bin { op, .. } if is_division(*op) => DIVIDE,
            WideOp::Bin { .. } => "arith.wide.int",
            WideOp::Compare { .. } => "arith.wide.compare",
            WideOp::Cast { .. } => "arith.wide.cast",
            WideOp::Convert { .. } => "arith.wide.convert",
            WideOp::Bitcast { .. } => "arith.wide.bitcast",
            WideOp::Select { .. } => "arith.wide.select",
            WideOp::Extract { .. } | WideOp::Insert { .. } => "arith.wide.element",
            WideOp::Shuffle { .. } => "arith.wide.shuffle",
            WideOp::IntFn { .. } => "arith.wide.int_fn",
        }
    }

    /// An upper estimate of the SPIR-V words per use, at most 6 words per instruction. A division's
    /// is its call (measured at most 11 words for an unsigned one and 207 for a signed one,
    /// `wide_division_words_bound_their_emission`); its routine's are the module's, once
    /// ([`Divide::words`]).
    pub fn words(&self) -> u32 {
        match self {
            WideOp::Bin { op, .. } if is_division(*op) => {
                if matches!(op, AirBin::SDiv | AirBin::SRem) {
                    384
                } else {
                    32
                }
            }
            WideOp::Bin { .. } => 192,
            WideOp::Compare { .. } => 96,
            // A directed rounding's magnitude and `half` steps beyond the plain conversion.
            WideOp::Convert {
                kind: ConvertKind::ToFloatDirected { .. },
                to,
                ..
            } => 1024 + rounding::directed_words(*to),
            WideOp::Cast { .. } | WideOp::Convert { .. } => {
                // The float paths: the collapse to 32 bits (two shifts, a compare) and rounding.
                1024
            }
            WideOp::Bitcast { from, to } => {
                64 + 96 * u32::from(from.lanes.max(to.lanes)) + narrow_words(*from, *to)
            }
            // A scalar's is one `OpSelect` of the pair (6 words); a vector's the operands' splits
            // (four 5-word extracts), two selects, and the join (37).
            WideOp::Select { form } if form.lanes == 1 => 8,
            WideOp::Select { .. } => 40,
            WideOp::Extract { .. } | WideOp::Insert { .. } => 64,
            // The operands' splits (20 words), two `OpVectorShuffle` of 5 words plus one per
            // lane, and the join (5).
            WideOp::Shuffle { mask, .. } => 40 + 2 * u32::try_from(mask.len()).unwrap_or(u32::MAX),
            // Measured at most 712 words for a signed `mad_sat` and 452 for every other function
            // (`wide_integer_function_words_bound_their_emission`).
            WideOp::IntFn { f, .. } => match f {
                IntFn::MadSat(_) => 1024,
                IntFn::Bswap => 512,
                _ => 640,
            },
        }
    }

    /// The module-scope routine the operation calls, if any: a division's.
    pub fn routine(&self) -> Option<Divide> {
        match self {
            WideOp::Bin { op, form } if is_division(*op) => Some(Divide::of(*form)),
            _ => None,
        }
    }

    /// The operands the operation reads (a prefix of the instruction's; all when `None`).
    pub fn arity(&self) -> Option<usize> {
        match self {
            WideOp::Cast { .. } | WideOp::Convert { .. } | WideOp::Bitcast { .. } => Some(1),
            WideOp::Extract { at: Some(_), .. } => Some(1),
            WideOp::Insert { at: Some(_), .. } => Some(2),
            WideOp::IntFn { f, .. } => Some(f.arity()),
            _ => None,
        }
    }
}

/// The words a bitcast spends moving narrow floats to or from their bits.
fn narrow_words(from: Form, to: Form) -> u32 {
    let one = |f: Form| {
        if f.is_narrow() {
            192 * u32::from(f.lanes)
        } else {
            0
        }
    };
    one(from) + one(to)
}

/// Emit `op` on operand values `xs`; `routine` is the module's function of [`WideOp::routine`]
/// when the operation has one.
pub fn emit(
    fb: &mut FunctionBuilder<'_>,
    op: &WideOp,
    xs: &[SValue],
    routine: Option<FuncId>,
) -> BResult<SValue> {
    let a = |i: usize| -> BResult<SValue> { xs.get(i).copied().ok_or_else(arity_error) };
    match op {
        WideOp::Bin { op, form } if is_division(*op) => {
            let r = routine.ok_or_else(missing_routine)?;
            divide(fb, *op, *form, a(0)?, a(1)?, r)
        }
        WideOp::Bin { op, form } => binary(fb, *op, *form, a(0)?, a(1)?),
        WideOp::Compare { pred, form } => int_compare(fb, *pred, *form, a(0)?, a(1)?),
        WideOp::Cast { kind, from, to } => cast(fb, *kind, *from, *to, a(0)?),
        WideOp::Convert { kind, from, to } => convert(fb, *kind, *from, *to, a(0)?),
        WideOp::Bitcast { from, to } => bitcast(fb, *from, *to, a(0)?),
        WideOp::Select { form } if form.lanes == 1 => fb.select(a(0)?, a(1)?, a(2)?),
        WideOp::Select { form } => {
            let (c, t, f) = (a(0)?, split(fb, a(1)?)?, split(fb, a(2)?)?);
            let r = sel(fb, c, t, f)?;
            join(fb, *form, r)
        }
        WideOp::Extract {
            vec,
            at,
            wide_index,
        } => {
            let v = a(0)?;
            let i = match at {
                Some(_) => None,
                None => Some(index(fb, a(1)?, *wide_index)?),
            };
            let lane = |fb: &mut FunctionBuilder<'_>, w: SValue| -> BResult<SValue> {
                match (at, i) {
                    (Some(at), _) => fb.composite_extract(w, &[*at]),
                    (None, Some(i)) => fb.vector_extract_dynamic(w, i),
                    (None, None) => Err(arity_error()),
                }
            };
            if vec.is_wide() {
                let p = split(fb, v)?;
                let r = Pair {
                    lo: lane(fb, p.lo)?,
                    hi: lane(fb, p.hi)?,
                };
                join(fb, vec.scalar_form(), r)
            } else {
                let y = lane(fb, v)?;
                // A store of a narrow lane takes its bits from its vector's.
                if let Some(at) = at {
                    if super::narrow::may_be_narrow(fb, y)? {
                        fb.note_lane(y, v, *at);
                    }
                }
                Ok(y)
            }
        }
        WideOp::Insert {
            vec,
            at,
            wide_index,
        } => {
            let (v, e) = (a(0)?, a(1)?);
            let i = match at {
                Some(_) => None,
                None => Some(index(fb, a(2)?, *wide_index)?),
            };
            let put = |fb: &mut FunctionBuilder<'_>, w: SValue, x: SValue| -> BResult<SValue> {
                match (at, i) {
                    (Some(at), _) => fb.composite_insert(x, w, &[*at]),
                    (None, Some(i)) => fb.vector_insert_dynamic(w, x, i),
                    (None, None) => Err(arity_error()),
                }
            };
            if vec.is_wide() {
                let (p, x) = (split(fb, v)?, split(fb, e)?);
                let r = Pair {
                    lo: put(fb, p.lo, x.lo)?,
                    hi: put(fb, p.hi, x.hi)?,
                };
                join(fb, *vec, r)
            } else {
                // Narrow lanes' bits follow the insert.
                match at {
                    Some(at) => super::narrow::insert(fb, v, e, *at),
                    None => put(fb, v, e),
                }
            }
        }
        WideOp::Shuffle { mask, out } => {
            let (p, q) = (split(fb, a(0)?)?, split(fb, a(1)?)?);
            let r = Pair {
                lo: fb.vector_shuffle(p.lo, q.lo, mask)?,
                hi: fb.vector_shuffle(p.hi, q.hi, mask)?,
            };
            join(fb, *out, r)
        }
        WideOp::IntFn { f, form } => int_fn(fb, *f, *form, xs),
    }
}

/// A dynamic element index: a wide one's low word (an index of 2^32 or more is out of range,
/// poison, whatever lane it names).
fn index(fb: &mut FunctionBuilder<'_>, i: SValue, wide: bool) -> BResult<SValue> {
    if wide {
        fb.composite_extract(i, &[0])
    } else {
        Ok(i)
    }
}

// ---- Selection ----

/// Whether `inst` has a wide integer operand or result on the selection's profile, so this
/// module lowers it (aggregate operations, `freeze`, and `select` of aggregates move the
/// representation as it is and stay with the general strategies).
pub fn involved(cx: &SelectCx<'_>, inst: &Inst) -> bool {
    if !legal::holds_wide(cx.profile) {
        return false;
    }
    let types = cx.entry.module.types();
    let wide = |t: Option<crate::air::ir::TypeId>| t.is_some_and(|t| legal::is_wide_int(types, t));
    match &inst.kind {
        InstKind::ExtractValue { .. } | InstKind::InsertValue { .. } | InstKind::Freeze(_) => false,
        InstKind::Call(call) => {
            wide(Some(inst.ty)) || call.args.iter().any(|v| wide(cx.value_ty(*v)))
        }
        kind => {
            let mut any = wide(Some(inst.ty));
            kind.for_each_value(|v| any |= wide(cx.value_ty(v)));
            any
        }
    }
}

/// The form of `v`, wide where the profile holds it so; with `chunk`, a chunked vector's form is
/// that of one chunk of `chunk` lanes.
fn form(cx: &SelectCx<'_>, v: Value, chunk: Option<u8>) -> Result<Form, Error> {
    type_form(cx, ty_of(cx, v)?, chunk)
}

fn result_form(cx: &SelectCx<'_>, inst: &Inst, chunk: Option<u8>) -> Result<Form, Error> {
    type_form(cx, inst.ty, chunk)
}

fn type_form(
    cx: &SelectCx<'_>,
    ty: crate::air::ir::TypeId,
    chunk: Option<u8>,
) -> Result<Form, Error> {
    if let (Some(k), Some((held, _))) = (chunk, legal::chunked(cx.entry.module.types(), ty)) {
        return Ok(Form {
            held: legal::held_on(cx.profile, held)?,
            lanes: k,
        });
    }
    legal::wide_form_on(cx.entry, cx.profile, ty)
}

/// Whether a lane-wise operation `inst` on chunked vectors has a wide integer lane on the
/// selection's profile, so this module lowers each chunk ([`select_in`]).
pub fn involved_chunked(cx: &SelectCx<'_>, inst: &Inst) -> bool {
    if !legal::holds_wide(cx.profile) {
        return false;
    }
    let types = cx.entry.module.types();
    let wide = |t: Option<crate::air::ir::TypeId>| {
        t.is_some_and(|t| {
            legal::chunked(types, t)
                .is_some_and(|(h, _)| matches!(h, Held::Int { container: 64, .. }))
                || legal::is_wide_int(types, t)
        })
    };
    match &inst.kind {
        InstKind::Call(call) => {
            wide(Some(inst.ty)) || call.args.iter().any(|v| wide(cx.value_ty(*v)))
        }
        kind => {
            let mut any = wide(Some(inst.ty));
            kind.for_each_value(|v| any |= wide(cx.value_ty(v)));
            any
        }
    }
}

/// Select the lowering of `inst`, which [`involved`] holds.
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<ArithStrategy, Error> {
    select_in(cx, at, inst, None)
}

/// Select `inst`; with `chunk`, for one chunk of `chunk` lanes of a lane-wise operation on
/// chunked vectors of wide integers (its chunked operands and result take the chunk's form).
pub fn select_in(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    if let InstKind::Call(call) = &inst.kind {
        return select_call(cx, inst, call, chunk);
    }
    let mut ops = operands_of(&inst.kind);
    if let InstKind::ExtractElement { idx, .. } | InstKind::InsertElement { idx, .. } = &inst.kind {
        if matches!(idx, Value::Const(_)) {
            ops.retain(|v| v != idx);
        }
    }
    check_operands(cx, &ops)?;
    let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    let op = match &inst.kind {
        InstKind::BinOp { op, a, .. } => {
            let f = form(cx, *a, chunk)?;
            if !f.is_wide() || op.is_float() {
                return Err(cx.invariant("a wide binary operator on mismatched operands"));
            }
            WideOp::Bin { op: *op, form: f }
        }
        InstKind::ICmp { pred, a, .. } => {
            let f = form(cx, *a, chunk)?;
            if !f.is_wide() {
                return Err(cx.invariant("a wide compare on mismatched operands"));
            }
            WideOp::Compare {
                pred: *pred,
                form: f,
            }
        }
        InstKind::Cast { op, v } => select_cast(cx, *op, *v, inst, chunk)?,
        InstKind::Select { .. } => {
            let f = result_form(cx, inst, chunk)?;
            if !f.is_wide() {
                return Err(cx.invariant("a wide select of a narrow value"));
            }
            WideOp::Select { form: f }
        }
        InstKind::ExtractElement { vec, idx } => {
            let vf = form(cx, *vec, chunk)?;
            let lanes = u64::from(lanes_of(cx, *vec)?);
            match (idx, super::const_int(cx, *idx)) {
                (Value::Const(_), Some(i)) if i < lanes => WideOp::Extract {
                    vec: vf,
                    at: Some(u32::try_from(i).map_err(|_| not_lowered())?),
                    wide_index: false,
                },
                (Value::Const(_), _) => return Ok(ArithStrategy::Undef(result)),
                _ => WideOp::Extract {
                    vec: vf,
                    at: None,
                    wide_index: form(cx, *idx, chunk)?.is_wide(),
                },
            }
        }
        InstKind::InsertElement { vec, idx, .. } => {
            let vf = form(cx, *vec, chunk)?;
            let lanes = u64::from(lanes_of(cx, *vec)?);
            match (idx, super::const_int(cx, *idx)) {
                (Value::Const(_), Some(i)) if i < lanes => WideOp::Insert {
                    vec: vf,
                    at: Some(u32::try_from(i).map_err(|_| not_lowered())?),
                    wide_index: false,
                },
                (Value::Const(_), _) => return Ok(ArithStrategy::Undef(result)),
                _ => WideOp::Insert {
                    vec: vf,
                    at: None,
                    wide_index: form(cx, *idx, chunk)?.is_wide(),
                },
            }
        }
        InstKind::ShuffleVector { a, b, mask } => {
            let (fa, fb_) = (form(cx, *a, chunk)?, form(cx, *b, chunk)?);
            let out = result_form(cx, inst, chunk)?;
            if !fa.is_wide() || !fb_.is_wide() || !out.is_wide() || out.lanes == 1 {
                return Err(not_lowered());
            }
            let total = u32::from(fa.lanes) + u32::from(fb_.lanes);
            let mut comps = Vec::with_capacity(mask.len());
            for m in mask.iter() {
                // A poison lane may be any value: take component 0.
                let c = m.unwrap_or(0);
                if c >= total {
                    return Err(cx.invariant("a shuffle index beyond its operands"));
                }
                comps.push(c);
            }
            WideOp::Shuffle {
                mask: comps.into(),
                out,
            }
        }
        _ => return Err(not_lowered()),
    };
    if let Some(d) = op.routine() {
        cx.families.arith.divides.insert(d);
    }
    Ok(ArithStrategy::Wide(op))
}

fn select_cast(
    cx: &mut SelectCx<'_>,
    op: CastOp,
    v: Value,
    inst: &Inst,
    chunk: Option<u8>,
) -> Result<WideOp, Error> {
    let from = form(cx, v, chunk)?;
    let to = result_form(cx, inst, chunk)?;
    let int_like = |f: Form| f.is_bool() || f.is_wide() || (f.is_int() && f.container() == 32);
    let kind = match op {
        CastOp::Trunc => CastKind::Trunc,
        CastOp::ZExt => CastKind::ZExt,
        CastOp::SExt => CastKind::SExt,
        CastOp::FPToUI => CastKind::FpToUi,
        CastOp::FPToSI => CastKind::FpToSi,
        CastOp::UIToFP => CastKind::UiToFp,
        CastOp::SIToFP => CastKind::SiToFp,
        CastOp::Bitcast => return select_bitcast(from, to),
        CastOp::FPTrunc | CastOp::FPExt => {
            return Err(cx.invariant("a float resize with a wide side"))
        }
        CastOp::PtrToInt | CastOp::IntToPtr | CastOp::AddrSpaceCast => {
            return Err(cx.invariant("a pointer cast routed to arith"))
        }
    };
    let ok = match kind {
        CastKind::Trunc | CastKind::ZExt | CastKind::SExt => int_like(from) && int_like(to),
        CastKind::FpToUi | CastKind::FpToSi => from.is_float() && to.is_wide(),
        CastKind::UiToFp | CastKind::SiToFp => from.is_wide() && to.is_float(),
        CastKind::FpResize => false,
    };
    if !ok || from.lanes != to.lanes {
        return Err(not_lowered());
    }
    Ok(WideOp::Cast { kind, from, to })
}

fn select_bitcast(from: Form, to: Form) -> Result<WideOp, Error> {
    if from == to {
        return Err(not_lowered());
    }
    let side = |f: Form| {
        if f.is_wide() {
            f.bits() == 64
        } else {
            !f.is_bool() && ops::word_lanes(f)
        }
    };
    let total = u32::from(from.bits()) * u32::from(from.lanes);
    if !side(from)
        || !side(to)
        || total != u32::from(to.bits()) * u32::from(to.lanes)
        || total % 32 != 0
    {
        return Err(not_lowered());
    }
    Ok(WideOp::Bitcast { from, to })
}

fn select_call(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    let fun = match &call.callee {
        Callee::Air {
            op: AirIntrinsic::Convert(c),
            ..
        } => return select_convert(cx, inst, call, c, chunk),
        Callee::Air {
            op: AirIntrinsic::Math(m),
            ..
        } => super::air_math(m.f)?,
        Callee::Llvm { op, .. } => super::llvm_math(op)?,
        _ => return Err(not_lowered()),
    };
    let op = match fun {
        Fun::Int(f) => f,
        // Unsigned `abs` is the operand itself.
        Fun::Copy => {
            let x = *call
                .args
                .first()
                .ok_or_else(|| cx.invariant("an intrinsic without operands"))?;
            check_operands(cx, &[x])?;
            super::lowered_type(cx.entry, cx.profile, inst.ty)?;
            return Ok(ArithStrategy::Copy);
        }
        Fun::Float(_) | Fun::SinCos => return Err(not_lowered()),
    };
    let args = call
        .args
        .get(..op.arity())
        .ok_or_else(|| cx.invariant("too few intrinsic operands"))?;
    check_operands(cx, args)?;
    super::lowered_type(cx.entry, cx.profile, inst.ty)?;
    let first = *args
        .first()
        .ok_or_else(|| cx.invariant("an intrinsic without operands"))?;
    let f = form(cx, first, chunk)?;
    if !f.is_wide() || !int_fn_lowered(op, f) {
        return Err(not_lowered());
    }
    // The bit-field offset and count are `u32` scalars.
    let fields: &[usize] = match op {
        IntFn::ExtractBits(_) => &[1, 2],
        IntFn::InsertBits => &[2, 3],
        _ => &[],
    };
    for i in fields {
        let v = *args
            .get(*i)
            .ok_or_else(|| cx.invariant("too few intrinsic operands"))?;
        let vf = form(cx, v, chunk)?;
        if !(vf.is_int() && vf.container() == 32 && vf.lanes == 1) {
            return Err(not_lowered());
        }
    }
    // Every other operand has the first one's form.
    for (i, v) in args.iter().enumerate() {
        if !fields.contains(&i) && form(cx, *v, chunk)? != f {
            return Err(not_lowered());
        }
    }
    Ok(ArithStrategy::Wide(WideOp::IntFn { f: op, form: f }))
}

fn select_convert(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    c: &crate::air::intrinsics::Convert,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    use crate::air::intrinsics::NumKind;
    let x = *call
        .args
        .first()
        .ok_or_else(|| cx.invariant("air.convert without an operand"))?;
    check_operands(cx, &[x])?;
    super::lowered_type(cx.entry, cx.profile, inst.ty)?;
    let from = form(cx, x, chunk)?;
    let to = result_form(cx, inst, chunk)?;
    if from.lanes != to.lanes {
        return Err(not_lowered());
    }
    // An 8-bit float to a wide integer: its value, then the saturating conversion.
    if let Some(fmt) = c.src_fp8 {
        let carrier = from.is_int() && from.bits() == 8;
        if c.dst_kind == NumKind::Float || !carrier || !to.is_wide() {
            return Err(not_lowered());
        }
        cx.precision
            .insert(crate::pipeline::artifact::PrecisionNote::Fp8Value);
        return Ok(ArithStrategy::Wide(WideOp::Convert {
            kind: ConvertKind::Fp8ToInt {
                fp8: narrow::Fp8Layout::of(fmt),
                signed: c.dst_kind == NumKind::Signed,
            },
            from,
            to,
        }));
    }
    // A wide integer to `bool` is C++'s `x != 0`.
    if to.is_bool() {
        if c.src_kind == NumKind::Float || !from.is_wide() {
            return Err(not_lowered());
        }
        return Ok(ArithStrategy::Wide(WideOp::Convert {
            kind: ConvertKind::IntToBool,
            from,
            to,
        }));
    }
    let kind = match (c.dst_kind, c.src_kind) {
        (NumKind::Float, NumKind::Float) => return Err(not_lowered()),
        (NumKind::Float, s) => {
            let signed = s == NumKind::Signed;
            let dst = to.float_fmt();
            match (rounding::Directed::of(c.rounding), dst) {
                (_, Some(fmt)) if rounding::converts_exactly(from, signed, fmt) => {
                    ConvertKind::ToFloat { signed }
                }
                // Nearest even: the conversion's rounding where the entry declares
                // `RoundingModeRTE` for its width (a `half` or `bfloat` rounds by integer
                // arithmetic).
                (None, Some(FloatFmt::F32)) => {
                    if c.rounding.is_some() {
                        cx.families.arith.rounds_even = true;
                    }
                    ConvertKind::ToFloat { signed }
                }
                (None, _) => ConvertKind::ToFloat { signed },
                (Some(dir), Some(FloatFmt::F32 | FloatFmt::Half)) => {
                    ConvertKind::ToFloatDirected { signed, dir }
                }
                // A `bfloat` by a directed rounding from a wide integer (`air.convert` names no
                // `double`).
                (Some(_), _) => return Err(not_lowered()),
            }
        }
        (d, NumKind::Float) => {
            let integral = rounding::Integral::of(c.rounding);
            match integral {
                rounding::Integral::Truncate => {}
                rounding::Integral::Even => {
                    super::leave_denormals(cx, from, super::denormal::Covered::Deviation)
                }
                rounding::Integral::Up | rounding::Integral::Down => {
                    super::leave_denormals(cx, from, super::denormal::Covered::Unary)
                }
            }
            ConvertKind::ToInt {
                signed: d == NumKind::Signed,
                integral,
            }
        }
        (d, s) if c.saturate => ConvertKind::SatInt {
            src_signed: s == NumKind::Signed,
            dst_signed: d == NumKind::Signed,
        },
        (_, s) => ConvertKind::Int {
            signed: s == NumKind::Signed,
        },
    };
    let int_like = |f: Form| f.is_bool() || f.is_wide() || (f.is_int() && f.container() == 32);
    let ok = match kind {
        ConvertKind::ToFloat { .. } | ConvertKind::ToFloatDirected { .. } => {
            from.is_wide() && to.is_float()
        }
        ConvertKind::ToInt { .. } => from.is_float() && to.is_wide(),
        ConvertKind::Int { .. } | ConvertKind::SatInt { .. } => {
            int_like(from) && int_like(to) && !to.is_bool()
        }
        ConvertKind::Float
        | ConvertKind::FloatDirected(_)
        | ConvertKind::FromFp8(_)
        | ConvertKind::Fp8ToInt { .. }
        | ConvertKind::ToBool
        | ConvertKind::IntToBool => false,
    };
    if !ok {
        return Err(not_lowered());
    }
    Ok(ArithStrategy::Wide(WideOp::Convert { kind, from, to }))
}
