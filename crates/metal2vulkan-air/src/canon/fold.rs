//! Constant folding for SCCP.
//!
//! Folds exactly what the spec lists: integer arithmetic and comparisons, `fcmp` on constants
//! (exact IEEE comparison of the stored values), casts among integers, same-width bit casts, int to
//! pointer of zero and pointer to int of null, `freeze` of a constant, and element and member
//! access on constant vectors and aggregates. SCCP does **no float arithmetic folding**: the
//! device's denormal and fast-math behavior is the lowering's contract. Anything else, and any
//! operation whose result would be poison (division by zero, an oversized shift, an out-of-range
//! element), is left unfolded. A fold never guesses: `None` means "not a constant here".
//!
//! The static-initializer interpreter alone also evaluates the basic float operations where
//! every IEEE 754 evaluation agrees ([`Folder::static_float_binop`]): a static initializer is the
//! deferred form of an MSL constant expression, which the pipeline build evaluates once, before any
//! entry runs, not the device's per-invocation arithmetic.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::LlvmIntOp;
use crate::ir::{BinOp, CastOp, Const, ConstId, ConstTable, Constant, FloatPredicate};
use crate::ir::{IntPredicate, Type, TypeId, TypeTable};

/// The most elements a fold expands a `zeroinitializer`/`undef` aggregate into. Larger aggregates
/// are left unfolded (a precision limit, never a semantic one).
const EXPAND_LIMIT: u64 = 4096;

/// Folds over one module's types and constants.
pub(super) struct Folder<'a> {
    pub types: &'a TypeTable,
    pub consts: &'a mut ConstTable,
    /// Ledger bytes of the member lists the current fold holds, returned when it ends.
    pub held: usize,
}

fn mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sext(v: u64, bits: u8) -> i64 {
    let shift = 64 - u32::from(bits.clamp(1, 64));
    ((v << shift) as i64) >> shift
}

/// The exact value of a stored floating-point bit pattern.
fn float_value(ty: &Type, bits: u64) -> Option<f64> {
    Some(match ty {
        Type::Half => half_to_f64(bits as u16),
        Type::BFloat => f64::from(f32::from_bits((bits as u32) << 16)),
        Type::Float => f64::from(f32::from_bits(bits as u32)),
        Type::Double => f64::from_bits(bits),
        _ => return None,
    })
}

/// IEEE binary16 to f64 (exact: every half is a double).
fn half_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f64::from(h & 0x3ff);
    sign * match exp {
        0 => frac * 2f64.powi(-24),
        0x1f if frac == 0.0 => f64::INFINITY,
        0x1f => f64::NAN,
        e => (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The float formats a static initializer's arithmetic is evaluated in (bfloat is not).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatFormat {
    Half,
    Float,
    Double,
}

impl FloatFormat {
    fn of(ty: &Type) -> Option<FloatFormat> {
        match ty {
            Type::Half => Some(FloatFormat::Half),
            Type::Float => Some(FloatFormat::Float),
            Type::Double => Some(FloatFormat::Double),
            _ => None,
        }
    }

    /// The least positive normal value.
    fn min_normal(self) -> f64 {
        match self {
            FloatFormat::Half => 2f64.powi(-14),
            FloatFormat::Float => f64::from(f32::MIN_POSITIVE),
            FloatFormat::Double => f64::MIN_POSITIVE,
        }
    }

    /// The sign bit of the stored bits.
    fn sign(self) -> u64 {
        match self {
            FloatFormat::Half => 1 << 15,
            FloatFormat::Float => 1 << 31,
            FloatFormat::Double => 1 << 63,
        }
    }

    /// Whether `x` (a value of this format) is finite, nonzero, and normal.
    fn plain(self, x: f64) -> bool {
        x.is_finite() && x.abs() >= self.min_normal()
    }
}

/// The bits of the half nearest to `x` (ties to even), for `|x|` at least the least normal half;
/// `None` when it rounds past the largest finite half. Exact integer arithmetic on `x`'s bits.
fn round_to_half(x: f64) -> Option<u16> {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let mut exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    if !(-14..=15).contains(&exp) {
        return None;
    }
    let frac = bits & ((1u64 << 52) - 1);
    let (shift, half) = (42u32, 1u64 << 41);
    let mut m = frac >> shift;
    let rem = frac & ((1u64 << shift) - 1);
    if rem > half || (rem == half && m & 1 == 1) {
        m += 1;
    }
    if m == 1 << 10 {
        m = 0;
        exp += 1;
    }
    if exp > 15 {
        return None;
    }
    Some(sign | (((exp + 15) as u16) << 10) | m as u16)
}

/// The bits of the float (`mant` stored mantissa bits, `exp` exponent bits) nearest to the
/// integer `(-1)^neg * mag`, ties to even; out of range is infinity.
fn round_int_to_float(neg: bool, mag: u64, mant: u32, exp: u32) -> u64 {
    let sign = if neg { 1u64 << (mant + exp) } else { 0 };
    if mag == 0 {
        return sign;
    }
    let (mant, exp) = (u64::from(mant), u64::from(exp));
    let bias = (1u64 << (exp - 1)) - 1;
    let mut e = u64::from(63 - mag.leading_zeros());
    let mut m = if e <= mant {
        mag << (mant - e)
    } else {
        let shift = e - mant;
        let mut m = mag >> shift;
        let rem = mag & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && m & 1 == 1) {
            m += 1;
        }
        m
    };
    if m == 1u64 << (mant + 1) {
        m >>= 1;
        e += 1;
    }
    if e > bias {
        return sign | (((1u64 << exp) - 1) << mant);
    }
    sign | ((e + bias) << mant) | (m & ((1u64 << mant) - 1))
}

impl Folder<'_> {
    fn ty(&self, c: ConstId) -> Option<TypeId> {
        self.consts.ty(c)
    }

    fn value(&self, c: ConstId) -> Option<&Const> {
        self.consts.get(c).map(|k| &k.value)
    }

    /// The width of a scalar integer type.
    fn int_bits(&self, ty: TypeId) -> Option<u8> {
        match self.types.get(ty)? {
            Type::Int(w) => Some(*w),
            _ => None,
        }
    }

    /// The value of a scalar integer constant (`zeroinitializer` is 0).
    pub fn int(&self, c: ConstId) -> Option<u64> {
        self.int_bits(self.ty(c)?)?;
        match self.value(c)? {
            Const::Int(v) => Some(*v),
            Const::Null => Some(0),
            _ => None,
        }
    }

    /// Whether the scalar integer `c` decides `c op x` and `x op c` whatever `x` holds: zero for
    /// `and` and `mul`, all ones for `or`.
    pub fn absorbs(&self, op: BinOp, c: ConstId) -> bool {
        let (Some(w), Some(v)) = (self.ty(c).and_then(|t| self.int_bits(t)), self.int(c)) else {
            return false;
        };
        match op {
            BinOp::And | BinOp::Mul => v == 0,
            BinOp::Or => v == mask(w),
            _ => false,
        }
    }

    fn mk(&mut self, ty: TypeId, value: Const, m: &mut Meter) -> Result<ConstId, Error> {
        self.consts.intern(Constant { ty, value }, m)
    }

    fn mk_int(&mut self, ty: TypeId, v: u64, m: &mut Meter) -> Result<Option<ConstId>, Error> {
        let Some(w) = self.int_bits(ty) else {
            return Ok(None);
        };
        self.mk(ty, Const::Int(v & mask(w)), m).map(Some)
    }

    /// `a op b` over scalar integers.
    pub fn binop(
        &mut self,
        op: BinOp,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let (Some(w), Some(x), Some(y)) = (self.int_bits(ty), self.int(a), self.int(b)) else {
            return Ok(None);
        };
        let (sx, sy) = (sext(x, w), sext(y, w));
        let min = sext(1u64 << (w - 1), w);
        let r = match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::UDiv if y != 0 => x / y,
            BinOp::URem if y != 0 => x % y,
            BinOp::SDiv if sy != 0 && !(sx == min && sy == -1) => sx.wrapping_div(sy) as u64,
            BinOp::SRem if sy != 0 && !(sx == min && sy == -1) => sx.wrapping_rem(sy) as u64,
            BinOp::Shl if y < u64::from(w) => x << y,
            BinOp::LShr if y < u64::from(w) => x >> y,
            BinOp::AShr if y < u64::from(w) => (sx >> y) as u64,
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            _ => return Ok(None),
        };
        self.mk_int(ty, r, m)
    }

    /// `llvm.{s,u}{max,min}` over scalar integers.
    pub fn int_minmax(
        &mut self,
        op: LlvmIntOp,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let (Some(w), Some(x), Some(y)) = (self.int_bits(ty), self.int(a), self.int(b)) else {
            return Ok(None);
        };
        let (sx, sy) = (sext(x, w), sext(y, w));
        let r = match op {
            LlvmIntOp::UMax => x.max(y),
            LlvmIntOp::UMin => x.min(y),
            LlvmIntOp::SMax => sx.max(sy) as u64,
            LlvmIntOp::SMin => sx.min(sy) as u64,
            _ => return Ok(None),
        };
        self.mk_int(ty, r, m)
    }

    /// `air.normalize_function_constant_predicate(x)`: 1 when `x` is a known non-zero integer, 0
    /// when it is zero or undefined (`undef`, `poison`, or `None`: a value computed from an
    /// unsupplied function constant).
    pub fn normalize_predicate(
        &mut self,
        ty: TypeId,
        x: Option<ConstId>,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let v = match x {
            None => 0,
            Some(c) => match self.value(c) {
                Some(Const::Undef | Const::Poison) => 0,
                _ => match self.int(c) {
                    Some(v) => u64::from(v != 0),
                    None => return Ok(None),
                },
            },
        };
        self.mk_int(ty, v, m)
    }

    /// `icmp pred a, b` over scalar integers (and null pointers), typed `ty` (`i1`).
    pub fn icmp(
        &mut self,
        pred: IntPredicate,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let both_null = matches!(self.value(a), Some(Const::Null))
            && matches!(self.value(b), Some(Const::Null));
        if both_null {
            let r = matches!(
                pred,
                IntPredicate::Eq
                    | IntPredicate::Uge
                    | IntPredicate::Ule
                    | IntPredicate::Sge
                    | IntPredicate::Sle
            );
            return self.mk_int(ty, u64::from(r), m);
        }
        let Some(w) = self.ty(a).and_then(|t| self.int_bits(t)) else {
            return Ok(None);
        };
        let (Some(x), Some(y)) = (self.int(a), self.int(b)) else {
            return Ok(None);
        };
        let (sx, sy) = (sext(x, w), sext(y, w));
        let r = match pred {
            IntPredicate::Eq => x == y,
            IntPredicate::Ne => x != y,
            IntPredicate::Ugt => x > y,
            IntPredicate::Uge => x >= y,
            IntPredicate::Ult => x < y,
            IntPredicate::Ule => x <= y,
            IntPredicate::Sgt => sx > sy,
            IntPredicate::Sge => sx >= sy,
            IntPredicate::Slt => sx < sy,
            IntPredicate::Sle => sx <= sy,
        };
        self.mk_int(ty, u64::from(r), m)
    }

    fn float(&self, c: ConstId) -> Option<f64> {
        let ty = self.types.get(self.ty(c)?)?;
        match self.value(c)? {
            Const::Float(bits) => float_value(ty, *bits),
            Const::Null => float_value(ty, 0),
            _ => None,
        }
    }

    /// A static initializer's `fadd`, `fsub`, `fmul`, or `fdiv` of two scalar `half`, `float`,
    /// or `double` constants, typed `ty`. Only the static-initializer interpreter calls it; SCCP
    /// never folds float arithmetic.
    ///
    /// Evaluated only where every IEEE 754 evaluation agrees, whatever the operation's fast-math
    /// flags and the evaluator's denormal mode: each operand is finite, nonzero, and normal, and
    /// the exact result is either zero (an exact cancellation, `+0` under round to nearest even)
    /// or finite and above the least normal value before rounding. The result is the exact value
    /// rounded to nearest even: `float` and `double` in host IEEE arithmetic, `half` by rounding
    /// the `double` result once more, which is innocuous for these operations (53 ≥ 2·11 + 2
    /// significand bits). Anything else is `None`.
    pub fn static_float_binop(
        &mut self,
        op: BinOp,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(format) = self.types.get(ty).and_then(FloatFormat::of) else {
            return Ok(None);
        };
        let (Some(x), Some(y)) = (self.float(a), self.float(b)) else {
            return Ok(None);
        };
        if self.ty(a) != Some(ty) || self.ty(b) != Some(ty) {
            return Ok(None);
        }
        if !format.plain(x) || !format.plain(y) {
            return Ok(None);
        }
        let exact = match op {
            BinOp::FAdd => x + y,
            BinOp::FSub => x - y,
            BinOp::FMul => x * y,
            BinOp::FDiv => x / y,
            _ => return Ok(None),
        };
        // Only a sum or difference of nonzero operands is exactly zero; a zero product or
        // quotient is an underflow.
        let cancels = matches!(op, BinOp::FAdd | BinOp::FSub);
        let bits = if exact == 0.0 && cancels {
            0
        } else if !exact.is_finite() || exact.abs() <= format.min_normal() {
            return Ok(None);
        } else {
            match format {
                FloatFormat::Double => exact.to_bits(),
                FloatFormat::Float => {
                    let (x, y) = (x as f32, y as f32);
                    let r = match op {
                        BinOp::FAdd => x + y,
                        BinOp::FSub => x - y,
                        BinOp::FMul => x * y,
                        _ => x / y,
                    };
                    if !r.is_finite() {
                        return Ok(None);
                    }
                    u64::from(r.to_bits())
                }
                FloatFormat::Half => match round_to_half(exact) {
                    Some(h) => u64::from(h),
                    None => return Ok(None),
                },
            }
        };
        self.mk(ty, Const::Float(bits), m).map(Some)
    }

    /// A static initializer's `fneg` of a scalar `half`, `float`, or `double` constant that is
    /// finite, nonzero, and normal: its sign flipped. Anything else is `None`.
    pub fn static_float_neg(
        &mut self,
        ty: TypeId,
        a: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(format) = self.types.get(ty).and_then(FloatFormat::of) else {
            return Ok(None);
        };
        let (Some(x), Some(Const::Float(bits))) = (self.float(a), self.value(a)) else {
            return Ok(None);
        };
        if self.ty(a) != Some(ty) || !format.plain(x) {
            return Ok(None);
        }
        let bits = *bits ^ format.sign();
        self.mk(ty, Const::Float(bits), m).map(Some)
    }

    /// `fcmp pred a, b` over scalar floats: an exact IEEE comparison, typed `ty` (`i1`).
    pub fn fcmp(
        &mut self,
        pred: FloatPredicate,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let (Some(x), Some(y)) = (self.float(a), self.float(b)) else {
            return Ok(None);
        };
        let uno = x.is_nan() || y.is_nan();
        use FloatPredicate as P;
        let r = match pred {
            P::False => false,
            P::True => true,
            P::Ord => !uno,
            P::Uno => uno,
            P::Oeq => !uno && x == y,
            P::Ogt => !uno && x > y,
            P::Oge => !uno && x >= y,
            P::Olt => !uno && x < y,
            P::Ole => !uno && x <= y,
            P::One => !uno && x != y,
            P::Ueq => uno || x == y,
            P::Ugt => uno || x > y,
            P::Uge => uno || x >= y,
            P::Ult => uno || x < y,
            P::Ule => uno || x <= y,
            P::Une => uno || x != y,
        };
        self.mk_int(ty, u64::from(r), m)
    }

    /// `op v to ty`.
    pub fn cast(
        &mut self,
        op: CastOp,
        v: ConstId,
        to: TypeId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(from) = self.ty(v) else {
            return Ok(None);
        };
        if from == to && op == CastOp::Bitcast {
            return Ok(Some(v));
        }
        let (Some(src), Some(dst)) = (self.types.get(from), self.types.get(to)) else {
            return Ok(None);
        };
        match (op, src, dst) {
            (CastOp::Trunc | CastOp::ZExt, Type::Int(_), Type::Int(_)) => match self.int(v) {
                Some(x) => self.mk_int(to, x, m),
                None => Ok(None),
            },
            (CastOp::SExt, Type::Int(w), Type::Int(_)) => match self.int(v) {
                Some(x) => self.mk_int(to, sext(x, *w) as u64, m),
                None => Ok(None),
            },
            (CastOp::IntToPtr, Type::Int(_), Type::Ptr { .. }) if self.int(v) == Some(0) => {
                self.mk(to, Const::Null, m).map(Some)
            }
            (CastOp::PtrToInt, Type::Ptr { .. }, Type::Int(_))
                if matches!(self.value(v), Some(Const::Null)) =>
            {
                self.mk_int(to, 0, m)
            }
            (CastOp::Bitcast, Type::Int(w), t) if t.float_bits() == Some(u32::from(*w)) => {
                match self.int(v) {
                    Some(x) => self.mk(to, Const::Float(x), m).map(Some),
                    None => Ok(None),
                }
            }
            (CastOp::Bitcast, s, Type::Int(w)) if s.float_bits() == Some(u32::from(*w)) => {
                match self.value(v) {
                    Some(Const::Float(bits)) => {
                        let bits = *bits;
                        self.mk_int(to, bits, m)
                    }
                    Some(Const::Null) => self.mk_int(to, 0, m),
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// `freeze c`: a constant without undefined parts is itself; `undef`/`poison` freezes to zero.
    /// An aggregate with undefined parts is left for the emitter.
    pub fn freeze(&mut self, c: ConstId, m: &mut Meter) -> Result<Option<ConstId>, Error> {
        let Some(ty) = self.ty(c) else {
            return Ok(None);
        };
        match self.value(c) {
            Some(Const::Undef | Const::Poison) => self.mk(ty, Const::Null, m).map(Some),
            Some(_) if self.fully_defined(c, m)? => Ok(Some(c)),
            _ => Ok(None),
        }
    }

    /// Whether no part of `c` is `undef` or `poison`. The worklist and visited set are charged
    /// as they grow and released on return.
    pub fn fully_defined(&self, c: ConstId, m: &mut Meter) -> Result<bool, Error> {
        let mut held = 0usize;
        let r = self.walk_defined(c, m, &mut held);
        m.release(held);
        r
    }

    fn walk_defined(&self, c: ConstId, m: &mut Meter, held: &mut usize) -> Result<bool, Error> {
        let mut stack = vec![c];
        let mut seen = std::collections::HashSet::new();
        // Each constant is pushed per parent edge and remembered once.
        let entry = 3 * std::mem::size_of::<ConstId>();
        while let Some(k) = stack.pop() {
            m.tick(1)?;
            if !seen.insert(k) {
                continue;
            }
            match self.value(k) {
                Some(Const::Undef | Const::Poison) | None => return Ok(false),
                Some(v) => {
                    let children = v.children();
                    let bytes = entry.saturating_mul(children.len().max(1));
                    m.charge(bytes)?;
                    *held = held.saturating_add(bytes);
                    stack.extend_from_slice(children);
                }
            }
        }
        Ok(true)
    }

    /// Run one fold, then return the ledger bytes its temporary lists held.
    fn transient<R>(
        &mut self,
        m: &mut Meter,
        fold: impl FnOnce(&mut Self, &mut Meter) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let r = fold(self, m);
        m.release(self.held);
        self.held = 0;
        r
    }

    /// The member types of an aggregate or vector type.
    fn member_types(&self, ty: TypeId) -> Option<Vec<TypeId>> {
        match self.types.get(ty)? {
            Type::Vector { elem, len } => {
                let n = u64::from(*len);
                (n <= EXPAND_LIMIT).then(|| vec![*elem; n as usize])
            }
            Type::Array { elem, len } => (*len <= EXPAND_LIMIT).then(|| vec![*elem; *len as usize]),
            Type::Struct { fields, .. } => Some(fields.to_vec()),
            _ => None,
        }
    }

    /// The members of a constant aggregate or vector, expanding `zeroinitializer`, `undef`, and
    /// `poison` into per-member constants.
    fn members(&mut self, c: ConstId, m: &mut Meter) -> Result<Option<Vec<ConstId>>, Error> {
        let Some(ty) = self.ty(c) else {
            return Ok(None);
        };
        let fill = match self.value(c) {
            Some(Const::Aggregate(elems)) => {
                let bytes = std::mem::size_of_val(&**elems);
                m.charge(bytes)?;
                self.held = self.held.saturating_add(bytes);
                return Ok(self.value(c).map(|v| v.children().to_vec()));
            }
            Some(Const::Null) => Const::Null,
            Some(Const::Undef) => Const::Undef,
            Some(Const::Poison) => Const::Poison,
            _ => return Ok(None),
        };
        let Some(tys) = self.member_types(ty) else {
            return Ok(None);
        };
        let bytes = tys.len().saturating_mul(2 * std::mem::size_of::<ConstId>());
        m.charge(bytes)?;
        self.held = self.held.saturating_add(bytes);
        let mut out = Vec::with_capacity(tys.len());
        for t in tys {
            out.push(self.mk(t, fill.clone(), m)?);
        }
        Ok(Some(out))
    }

    /// `shufflevector a, b, mask`, typed `ty`: lane `i` is lane `mask[i]` of `a` then `b`, and a
    /// `None` lane is `poison`.
    pub fn shuffle(
        &mut self,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        mask: &[Option<u32>],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.shuffle_in(ty, a, b, mask, m))
    }

    /// `shufflevector a, b, mask`, typed `ty`: lane `i` is lane `mask[i]` of `a` then `b`, and a
    /// `None` lane is `poison`.
    fn shuffle_in(
        &mut self,
        ty: TypeId,
        a: ConstId,
        b: ConstId,
        mask: &[Option<u32>],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(Type::Vector { elem, .. }) = self.types.get(ty) else {
            return Ok(None);
        };
        let elem = *elem;
        let (Some(xs), Some(ys)) = (self.members(a, m)?, self.members(b, m)?) else {
            return Ok(None);
        };
        let mut lanes = Vec::with_capacity(mask.len());
        for lane in mask {
            let c = match lane {
                None => self.mk(elem, Const::Poison, m)?,
                Some(i) => {
                    let i = *i as usize;
                    match xs
                        .get(i)
                        .or_else(|| i.checked_sub(xs.len()).and_then(|j| ys.get(j)))
                    {
                        Some(&c) => c,
                        None => return Ok(None),
                    }
                }
            };
            lanes.push(c);
        }
        self.mk(ty, Const::Aggregate(lanes.into_boxed_slice()), m)
            .map(Some)
    }

    /// `llvm.ctpop` over a scalar integer.
    pub fn ctpop(
        &mut self,
        ty: TypeId,
        x: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        match self.int(x) {
            Some(v) => self.mk_int(ty, u64::from(v.count_ones()), m),
            None => Ok(None),
        }
    }

    /// An integer-to-float conversion (`sitofp`/`uitofp`, `air.convert.f.*.{s,u}.*`) of a scalar
    /// integer constant to the float type `to`, rounded to nearest even as both Metal and Vulkan
    /// specify for these conversions. Exact integer arithmetic, no host float rounding mode.
    pub fn int_to_float(
        &mut self,
        x: ConstId,
        signed: bool,
        to: TypeId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let (Some(w), Some(v)) = (self.ty(x).and_then(|t| self.int_bits(t)), self.int(x)) else {
            return Ok(None);
        };
        let (neg, mag) = if signed && w > 1 && sext(v, w) < 0 {
            (true, sext(v, w).unsigned_abs())
        } else {
            (false, v & mask(w))
        };
        let (mant_bits, exp_bits) = match self.types.get(to) {
            Some(Type::Half) => (10u32, 5u32),
            Some(Type::Float) => (23, 8),
            Some(Type::Double) => (52, 11),
            _ => return Ok(None),
        };
        let bits = round_int_to_float(neg, mag, mant_bits, exp_bits);
        self.mk(to, Const::Float(bits), m).map(Some)
    }

    /// `air.convert` of the integer or `bool` constant `x` (a scalar, or a vector converted lane by
    /// lane) to type `to`: to `bool`, C++'s `x != 0`; to another integer, zero- or sign-extended by
    /// the source's signedness (`signed`; a `bool` is zero-extended) or truncated, as C++ converts
    /// it; a scalar to a float, [`Folder::int_to_float`]. A vector to floats is `None`.
    pub fn convert_int(
        &mut self,
        x: ConstId,
        signed: bool,
        to: TypeId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.convert_int_in(x, signed, to, m))
    }

    fn convert_int_in(
        &mut self,
        x: ConstId,
        signed: bool,
        to: TypeId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(Type::Vector { elem, .. }) = self.types.get(to) else {
            return self.convert_scalar(x, signed, to, m);
        };
        let elem = *elem;
        if self.types.get(elem).is_some_and(Type::is_float) {
            return Ok(None);
        }
        let Some(lanes) = self.members(x, m)? else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(lanes.len());
        for lane in lanes {
            match self.convert_scalar(lane, signed, elem, m)? {
                Some(c) => out.push(c),
                None => return Ok(None),
            }
        }
        self.mk(to, Const::Aggregate(out.into_boxed_slice()), m)
            .map(Some)
    }

    /// [`Folder::convert_int`] of a scalar.
    fn convert_scalar(
        &mut self,
        x: ConstId,
        signed: bool,
        to: TypeId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        if self.types.get(to).is_some_and(Type::is_float) {
            return self.int_to_float(x, signed, to, m);
        }
        let (Some(from), Some(dst), Some(v)) = (
            self.ty(x).and_then(|t| self.int_bits(t)),
            self.int_bits(to),
            self.int(x),
        ) else {
            return Ok(None);
        };
        let r = match (dst, from) {
            (1, _) => u64::from(v & mask(from) != 0),
            (_, 1) => v & 1,
            _ if signed => sext(v, from) as u64,
            _ => v & mask(from),
        };
        self.mk_int(to, r, m)
    }

    /// `extractelement vec, idx`.
    pub fn extract_element(
        &mut self,
        vec: ConstId,
        idx: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.extract_element_in(vec, idx, m))
    }

    /// `extractelement vec, idx`.
    fn extract_element_in(
        &mut self,
        vec: ConstId,
        idx: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(i) = self.int(idx) else {
            return Ok(None);
        };
        Ok(self
            .members(vec, m)?
            .and_then(|e| usize::try_from(i).ok().and_then(|i| e.get(i).copied())))
    }

    /// `insertelement vec, elt, idx`, typed `ty`.
    pub fn insert_element(
        &mut self,
        ty: TypeId,
        vec: ConstId,
        elt: ConstId,
        idx: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.insert_element_in(ty, vec, elt, idx, m))
    }

    /// `insertelement vec, elt, idx`, typed `ty`.
    fn insert_element_in(
        &mut self,
        ty: TypeId,
        vec: ConstId,
        elt: ConstId,
        idx: ConstId,
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some(i) = self.int(idx).and_then(|i| usize::try_from(i).ok()) else {
            return Ok(None);
        };
        let Some(mut elems) = self.members(vec, m)? else {
            return Ok(None);
        };
        let Some(slot) = elems.get_mut(i) else {
            return Ok(None);
        };
        *slot = elt;
        self.mk(ty, Const::Aggregate(elems.into_boxed_slice()), m)
            .map(Some)
    }

    /// `extractvalue agg, indices`.
    pub fn extract_value(
        &mut self,
        agg: ConstId,
        indices: &[u32],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.extract_value_in(agg, indices, m))
    }

    /// `extractvalue agg, indices`.
    fn extract_value_in(
        &mut self,
        agg: ConstId,
        indices: &[u32],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let mut c = agg;
        for &i in indices {
            let Some(elems) = self.members(c, m)? else {
                return Ok(None);
            };
            let Some(&next) = elems.get(i as usize) else {
                return Ok(None);
            };
            c = next;
        }
        Ok(Some(c))
    }

    /// `insertvalue agg, val, indices`.
    pub fn insert_value(
        &mut self,
        agg: ConstId,
        val: ConstId,
        indices: &[u32],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        self.transient(m, |f, m| f.insert_value_in(agg, val, indices, m))
    }

    /// `insertvalue agg, val, indices`.
    fn insert_value_in(
        &mut self,
        agg: ConstId,
        val: ConstId,
        indices: &[u32],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        // Descend, keeping each level's members, then rebuild from the innermost level out.
        let mut path: Vec<(TypeId, Vec<ConstId>, usize)> = Vec::new();
        let mut c = agg;
        for &i in indices {
            let (Some(ty), Some(elems)) = (self.ty(c), self.members(c, m)?) else {
                return Ok(None);
            };
            let Some(&next) = elems.get(i as usize) else {
                return Ok(None);
            };
            path.push((ty, elems, i as usize));
            c = next;
        }
        let mut new = val;
        while let Some((ty, mut elems, i)) = path.pop() {
            if let Some(slot) = elems.get_mut(i) {
                *slot = new;
            }
            new = self.mk(ty, Const::Aggregate(elems.into_boxed_slice()), m)?;
        }
        Ok(Some(new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_values_are_exact() {
        assert_eq!(half_to_f64(0x3c00), 1.0);
        assert_eq!(half_to_f64(0xc000), -2.0);
        assert_eq!(half_to_f64(0x0001), 2f64.powi(-24));
        assert_eq!(half_to_f64(0x7c00), f64::INFINITY);
        assert!(half_to_f64(0x7e00).is_nan());
        assert_eq!(half_to_f64(0x8000), 0.0);
    }

    #[test]
    fn doubles_round_to_the_nearest_even_normal_half() {
        assert_eq!(round_to_half(1.0), Some(0x3c00));
        assert_eq!(round_to_half(-2.0), Some(0xc000));
        assert_eq!(round_to_half(2f64.powi(-14)), Some(0x0400));
        assert_eq!(round_to_half(65504.0), Some(0x7bff));
        // Ties: 1 + 2^-11 is halfway to the odd neighbour and rounds down; 1 + 3·2^-11 rounds up
        // to the even one; just above a tie rounds up.
        assert_eq!(round_to_half(1.0 + 2f64.powi(-11)), Some(0x3c00));
        assert_eq!(round_to_half(1.0 + 3.0 * 2f64.powi(-11)), Some(0x3c02));
        assert_eq!(
            round_to_half(1.0 + 2f64.powi(-11) + 2f64.powi(-40)),
            Some(0x3c01)
        );
        // A carry into the exponent, and past the largest finite half.
        assert_eq!(round_to_half(2.0 - 2f64.powi(-12)), Some(0x4000));
        assert_eq!(round_to_half(65519.0), Some(0x7bff));
        assert_eq!(round_to_half(65520.0), None);
        assert_eq!(round_to_half(2f64.powi(-15)), None);
    }

    #[test]
    fn integers_round_to_the_nearest_even_float() {
        let f32_of = |neg, mag| f32::from_bits(round_int_to_float(neg, mag, 23, 8) as u32);
        assert_eq!(f32_of(false, 0), 0.0);
        assert_eq!(f32_of(false, 1), 1.0);
        assert_eq!(f32_of(true, 7), -7.0);
        assert_eq!(f32_of(false, 16_777_217), 16_777_216.0);
        assert_eq!(f32_of(false, 16_777_219), 16_777_220.0);
        assert_eq!(f32_of(false, u64::from(u32::MAX)), 4_294_967_296.0);
        for v in [3u64, 1 << 40, (1 << 53) + 1, u64::MAX] {
            assert_eq!(f32_of(false, v), v as f32, "{v}");
            let d = f64::from_bits(round_int_to_float(false, v, 52, 11));
            assert_eq!(d, v as f64, "{v}");
        }
        let half = |mag| round_int_to_float(false, mag, 10, 5);
        assert_eq!(half(1), 0x3c00);
        assert_eq!(half(2049), 0x6800);
        assert_eq!(half(2051), 0x6802);
        assert_eq!(half(65504), 0x7bff);
        assert_eq!(half(65519), 0x7bff);
        assert_eq!(half(65520), 0x7c00);
        assert_eq!(round_int_to_float(true, 2, 10, 5), 0xc000);
    }

    #[test]
    fn aggregate_folds_return_their_temporary_ledger_bytes() {
        let mut m = Meter::new(&crate::budget::Budget::default());
        let mut types = TypeTable::new();
        let mut consts = ConstTable::new();
        let i32t = types.intern(Type::Int(32), &mut m).unwrap();
        let v4 = types
            .intern(Type::Vector { elem: i32t, len: 4 }, &mut m)
            .unwrap();
        let int = |consts: &mut ConstTable, m: &mut Meter, v| {
            consts
                .intern(
                    Constant {
                        ty: i32t,
                        value: Const::Int(v),
                    },
                    m,
                )
                .unwrap()
        };
        let lanes: Vec<ConstId> = (0..4).map(|v| int(&mut consts, &mut m, v)).collect();
        let vec = consts
            .intern(
                Constant {
                    ty: v4,
                    value: Const::Aggregate(lanes.into()),
                },
                &mut m,
            )
            .unwrap();
        let zero = consts
            .intern(
                Constant {
                    ty: v4,
                    value: Const::Null,
                },
                &mut m,
            )
            .unwrap();
        let idx = int(&mut consts, &mut m, 2);
        let nine = int(&mut consts, &mut m, 9);
        let mut f = Folder {
            types: &types,
            consts: &mut consts,
            held: 0,
        };
        // A fold interns what it builds; run it twice so the second run interns nothing new and
        // must leave the ledger exactly where it found it.
        let fold = |f: &mut Folder<'_>, m: &mut Meter, op: &dyn Fn(&mut Folder<'_>, &mut Meter)| {
            op(f, m);
            let (before, len) = (m.ledger(), f.consts.len());
            op(f, m);
            assert_eq!(f.consts.len(), len);
            assert_eq!(m.ledger(), before);
            assert_eq!(f.held, 0);
        };
        fold(&mut f, &mut m, &|f, m| {
            f.extract_element(vec, idx, m).unwrap().unwrap();
        });
        fold(&mut f, &mut m, &|f, m| {
            f.insert_element(v4, zero, nine, idx, m).unwrap().unwrap();
        });
        fold(&mut f, &mut m, &|f, m| {
            f.shuffle(v4, vec, zero, &[Some(0), Some(5), None, Some(3)], m)
                .unwrap();
        });
        fold(&mut f, &mut m, &|f, m| {
            f.freeze(vec, m).unwrap().unwrap();
        });
    }

    #[test]
    fn sign_extension_and_masks() {
        assert_eq!(sext(0xff, 8), -1);
        assert_eq!(sext(0x7f, 8), 127);
        assert_eq!(sext(1, 1), -1);
        assert_eq!(mask(64), u64::MAX);
        assert_eq!(mask(3), 7);
    }
}
