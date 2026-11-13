//! Vectors of other than 2 to 4 lanes: a `<N x T>` with `N` not in `2..=4` (`<1 x T>`, `<8 x
//! half>`, `<16 x i8>`, `<64 x float>`, ...) is held as a [`DemoteTy::Composite`] of **chunks**
//! ([`legal::chunk_lanes`]): 4-lane SPIR-V vectors, the last one holding the remaining 1 to 4 lanes
//! (a 1-lane chunk is a scalar member). No profile needs `Vector16`, and a chunked value is an
//! ordinary layout-free struct for demotion, phis, `select` on a scalar condition, `freeze` (but
//! of integer lanes held narrower than their container: masked per chunk), and aggregate
//! insertion and extraction.
//!
//! Operations:
//!
//! - **Lane-wise** operations (binary operators, `fneg`, compares, numeric casts, `select` on a
//!   vector condition, and the lane-wise math and conversion intrinsics) run **per chunk**
//!   ([`ArithStrategy::Chunked`]): the operation is selected once per distinct chunk size with
//!   the chunk's form, and emitted per chunk on the chunk members of the chunked operands (a
//!   scalar operand, such as `extract_bits`' offset, is passed to every chunk).
//! - **Lane moves** (`extractelement`, `insertelement`, `shufflevector`) address a lane as its
//!   chunk member and component; a dynamic index selects among the lanes by comparison (an index
//!   out of range gives lane 0 for an extract and leaves the vector unchanged for an insert,
//!   both of which LLVM's poison result permits).
//! - **Bitcasts** with a chunked side go through 32-bit words ([`bitcast`]): lanes of 8, 16, 32,
//!   or 64 bits are packed low first into words and unpacked the same way (a narrow float lane
//!   moves as its format bits, a 64-bit lane as a word pair), so no 64-bit integer is needed
//!   for the move itself.
//!
//! - **Two-word lanes**: without `shaderInt64` a chunk of 64-bit integer lanes is their wide
//!   representation (a `uvec2` for a one-lane chunk, else a struct of the low-word and high-word
//!   vectors), a lane-wise operation is the wide lowering's per chunk (`wide::select_in`), a lane
//!   move moves both words of the lane, a dynamic index of 64 bits reads its low word (an index of
//!   `2^32` or more is out of range), and a bitcast takes the lane's words as they are.
//!
//! - **`dot`** of chunked vectors sums each chunk's `OpDot` (a product for a one-lane chunk) in
//!   `f32` and rounds a narrow result once; **`sincos`** of one takes each chunk's sine and
//!   cosine, and its cosine is stored as one chunked leaf (`memory::leaf`).
//!
//! Not lowered (`NotImplemented`): bitcasts whose lanes are `i1` or of another width, or whose
//! total is not a whole number of words.

use crate::air::intrinsics::{AirIntrinsic, MathFn};
use crate::air::ir::{Call, Callee, CastOp, Inst, InstId, InstKind, TypeId as AirTypeId, Value};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::arith::legal::{self, chunk_lanes, not_lowered, Form, Held};
use crate::lower::arith::libm::Lib;
use crate::lower::arith::math::{self, FloatFn};
use crate::lower::arith::{narrow, operands_of, ops, wide, ArithStrategy};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{DemoteTy, Value as SValue};
use crate::spirv::ir::{BinOp, Constant, Contraction, ConvOp};

/// How a scalar or vector value is held: a scalar (`lanes == 1`, not chunked), a SPIR-V vector
/// (2 to 4 lanes), or chunked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub held: Held,
    pub lanes: u16,
    pub chunked: bool,
}

impl Shape {
    /// The native form of a scalar or SPIR-V vector shape.
    pub(super) fn form(self) -> Form {
        Form {
            held: self.held,
            lanes: u8::try_from(self.lanes).unwrap_or(1),
        }
    }

    /// Where lane `i` lives: its member path, or, for a lane of two words inside a vector of two
    /// or more lanes, the paths of its low and high words.
    fn path(self, i: u32) -> LanePath {
        let (mut prefix, size, j) = if self.chunked {
            let c = i / 4;
            let sizes = chunk_lanes(u32::from(self.lanes));
            (vec![c], sizes.get(c as usize).copied().unwrap_or(1), i % 4)
        } else {
            (Vec::new(), u8::try_from(self.lanes).unwrap_or(1), i)
        };
        if size == 1 {
            return LanePath::Whole(prefix);
        }
        if matches!(self.held, Held::Wide { .. }) {
            let mut hi = prefix.clone();
            prefix.extend([0, j]);
            hi.extend([1, j]);
            return LanePath::Words(prefix, hi);
        }
        prefix.push(j);
        LanePath::Whole(prefix)
    }

    /// The form of one lane.
    pub(super) fn lane_form(self) -> Form {
        Form {
            held: self.held,
            lanes: 1,
        }
    }

    /// The lowered representation.
    pub fn demote(self) -> DemoteTy {
        if self.chunked {
            legal::chunked_demote(self.held, u32::from(self.lanes))
        } else {
            self.form().demote()
        }
    }
}

/// The shape of AIR type `ty` on the profile (64-bit integer lanes as two words without
/// `shaderInt64`), or the refusal of one arith does not hold.
pub(super) fn shape_of(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Shape, Error> {
    if let Some((held, lanes)) = legal::chunked(cx.entry.module.types(), ty) {
        let held = legal::held_on(cx.profile, held)?;
        let lanes = u16::try_from(lanes).map_err(|_| not_lowered())?;
        return Ok(Shape {
            held,
            lanes,
            chunked: true,
        });
    }
    let f = legal::wide_form_on(cx.entry, cx.profile, ty)?;
    Ok(Shape {
        held: f.held,
        lanes: u16::from(f.lanes),
        chunked: false,
    })
}

/// Where one lane of a scalar, vector, or chunked value lives ([`Shape::path`]).
enum LanePath {
    /// The member at this path (the value itself when empty).
    Whole(Vec<u32>),
    /// A two-word lane: its low word and its high word, each a component of a word vector.
    Words(Vec<u32>, Vec<u32>),
}

/// Whether `ty` is a chunked vector type.
fn is_chunked(cx: &SelectCx<'_>, ty: AirTypeId) -> bool {
    legal::chunked(cx.entry.module.types(), ty).is_some()
}

fn value_chunked(cx: &SelectCx<'_>, v: Value) -> bool {
    cx.value_ty(v).is_some_and(|t| is_chunked(cx, t))
}

/// Whether `freeze` of chunked `v` is lane-wise: `v` is no `undef`, and its lanes are integers
/// held narrower than their container, which a poison lane (a `nuw` operation's,
/// [`ops::int_binary`]) may exceed ([`ArithStrategy::Freeze`]).
fn freezes_per_chunk(cx: &SelectCx<'_>, v: Value) -> bool {
    let padded = cx
        .value_ty(v)
        .and_then(|t| legal::chunked(cx.entry.module.types(), t))
        .is_some_and(|(held, _)| Form { held, lanes: 1 }.legalized());
    padded && !matches!(cx.constant(v), Some(legal::LoweredConst::Undef(_)))
}

/// The value operands of `inst` as emission reads them (a call's arguments).
fn value_operands(inst: &Inst) -> Vec<Value> {
    match &inst.kind {
        InstKind::Call(call) => call.args.to_vec(),
        kind => operands_of(kind),
    }
}

/// Whether instruction `inst` reads or produces a chunked vector.
pub fn involved(cx: &SelectCx<'_>, inst: &Inst) -> bool {
    is_chunked(cx, inst.ty) || value_operands(inst).iter().any(|v| value_chunked(cx, *v))
}

/// The lane count of a chunked instruction's lane-wise operation: its result's, else its first
/// chunked operand's.
fn lanes_of(cx: &SelectCx<'_>, inst: &Inst) -> Option<u32> {
    let types = cx.entry.module.types();
    if let Some((_, n)) = legal::chunked(types, inst.ty) {
        return Some(n);
    }
    value_operands(inst)
        .iter()
        .find_map(|v| cx.value_ty(*v).and_then(|t| legal::chunked(types, t)))
        .map(|(_, n)| n)
}

/// The integer form of a dynamic lane index `v` (a held integer, or two words, of which the low
/// one names the lane: an index of `2^32` or more is out of range), or a refusal.
fn index_form(cx: &SelectCx<'_>, v: Value) -> Result<Form, Error> {
    let t = cx
        .value_ty(v)
        .ok_or_else(|| cx.invariant("a lane index without a type"))?;
    let f = legal::wide_form_on(cx.entry, cx.profile, t)?;
    if !(f.is_int() || f.is_wide()) || f.lanes != 1 {
        return Err(not_lowered());
    }
    Ok(f)
}

/// The constant lane index `v`, if it is one.
fn const_index(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c)?.value {
        crate::air::ir::Const::Int(b) => Some(b),
        _ => None,
    }
}

/// Select the strategy of instruction `inst` that involves a chunked vector, metering in `m`
/// the walks the store of `air.sincos` takes.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    use ArithStrategy as S;
    match &inst.kind {
        // Whole-value operations: the composite as it is. A `freeze` of padded integer lanes
        // masks them, per chunk (the lane-wise arm below).
        InstKind::Freeze(v) if !freezes_per_chunk(cx, *v) => {
            super::select_in(cx, at, inst, None, m)
        }
        InstKind::ExtractValue { .. } | InstKind::InsertValue { .. } => {
            super::select_in(cx, at, inst, None, m)
        }
        InstKind::Select { c, .. } if !value_chunked(cx, *c) => {
            super::select_in(cx, at, inst, None, m)
        }
        InstKind::ExtractElement { vec, idx } => {
            let ty = cx
                .value_ty(*vec)
                .ok_or_else(|| cx.invariant("a vector without a type"))?;
            let src = shape_of(cx, ty)?;
            super::check_operands(cx, &[*vec])?;
            let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
            Ok(match (idx, const_index(cx, *idx)) {
                (Value::Const(_), Some(i)) if i < u64::from(src.lanes) => S::LaneExtract {
                    src,
                    at: Some(i as u32),
                    index: None,
                },
                (Value::Const(_), _) => S::Undef(result),
                _ => {
                    super::check_operands(cx, &[*idx])?;
                    S::LaneExtract {
                        src,
                        at: None,
                        index: Some(index_form(cx, *idx)?),
                    }
                }
            })
        }
        InstKind::InsertElement { vec, elt, idx } => {
            let dst = shape_of(cx, inst.ty)?;
            super::check_operands(cx, &[*vec, *elt])?;
            let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
            Ok(match (idx, const_index(cx, *idx)) {
                (Value::Const(_), Some(i)) if i < u64::from(dst.lanes) => S::LaneInsert {
                    dst,
                    at: Some(i as u32),
                    index: None,
                },
                (Value::Const(_), _) => S::Undef(result),
                _ => {
                    super::check_operands(cx, &[*idx])?;
                    S::LaneInsert {
                        dst,
                        at: None,
                        index: Some(index_form(cx, *idx)?),
                    }
                }
            })
        }
        InstKind::ShuffleVector { a, b, mask } => {
            super::check_operands(cx, &[*a, *b])?;
            let ty = |cx: &SelectCx<'_>, v: Value| {
                cx.value_ty(v)
                    .ok_or_else(|| cx.invariant("a vector without a type"))
            };
            let (sa, sb) = (shape_of(cx, ty(cx, *a)?)?, shape_of(cx, ty(cx, *b)?)?);
            let out = shape_of(cx, inst.ty)?;
            let total = u32::from(sa.lanes) + u32::from(sb.lanes);
            let mut lanes = Vec::with_capacity(mask.len());
            for m in mask.iter() {
                // A poison lane may be any value: take lane 0.
                let l = m.unwrap_or(0);
                if l >= total {
                    return Err(cx.invariant("a shuffle index beyond its operands"));
                }
                lanes.push(l);
            }
            // A simdgroup matrix's fragment: its elements 2..63 are zero or undefined
            // (`memory::fragment`), so zero is their value or a refinement of it.
            let fragment = cx.memory.is_fragment(at) && fragment_shape(out);
            Ok(S::LaneShuffle {
                a: sa,
                b: sb,
                out,
                mask: lanes.into(),
                fragment,
            })
        }
        InstKind::Call(call) if is_math(call, MathFn::SinCos) => {
            super::select_sincos(cx, at, inst, call, m)
        }
        InstKind::Call(call) if is_math(call, MathFn::Dot) => select_dot(cx, inst, call),
        InstKind::Cast {
            op: CastOp::Bitcast,
            v,
        } => {
            super::check_operands(cx, &[*v])?;
            let ty = cx
                .value_ty(*v)
                .ok_or_else(|| cx.invariant("a bitcast operand without a type"))?;
            let (from, to) = (shape_of(cx, ty)?, shape_of(cx, inst.ty)?);
            if !word_movable(from) || !word_movable(to) {
                return Err(not_lowered());
            }
            Ok(S::WordBitcast { from, to })
        }
        _ => {
            let n = lanes_of(cx, inst).ok_or_else(|| cx.invariant("no chunked lane count"))?;
            if !is_chunked(cx, inst.ty) {
                // A lane-wise operation has a result of its operands' lanes; anything else
                // (`dot`) is not lowered per chunk.
                return Err(not_lowered());
            }
            // Chunks of two-word lanes are the wide lowering's.
            let wide_lanes = wide::involved_chunked(cx, inst);
            let mut parts: Vec<(u8, ArithStrategy)> = Vec::new();
            for k in chunk_lanes(n) {
                if parts.iter().all(|(s, _)| *s != k) {
                    let part = if wide_lanes {
                        wide::select_in(cx, at, inst, Some(k))?
                    } else {
                        super::select_in(cx, at, inst, Some(k), m)?
                    };
                    parts.push((k, part));
                }
            }
            let arity = parts
                .first()
                .map(|(_, s)| super::arity(s))
                .ok_or_else(|| cx.invariant("a chunked vector without chunks"))?;
            let ops = value_operands(inst);
            let take = arity.unwrap_or(ops.len()).min(ops.len());
            let wide: Vec<bool> = ops
                .iter()
                .take(take)
                .map(|v| value_chunked(cx, *v))
                .collect();
            let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
            Ok(S::Chunked {
                lanes: u16::try_from(n).map_err(|_| not_lowered())?,
                parts: parts.into(),
                wide: wide.into(),
                result,
            })
        }
    }
}

/// Whether `call` is the `air` math function `f`.
fn is_math(call: &Call, f: MathFn) -> bool {
    matches!(
        &call.callee,
        Callee::Air {
            op: AirIntrinsic::Math(m),
            ..
        } if m.f == f
    )
}

/// `air.dot` of two chunked float vectors, on every device: the sum of each chunk's dot product by
/// the device's `OpDot`, rounded once to a narrow format. Metal's `dot` is the products' fused
/// chain over every lane; `dot`'s device precision (`DeviceArithmetic`) covers the order of the
/// sum, which Vulkan leaves to the device for `OpDot` too. Its NaN is the device's.
fn select_dot(cx: &mut SelectCx<'_>, inst: &Inst, call: &Call) -> Result<ArithStrategy, Error> {
    let (Some(&a), Some(&b)) = (call.args.first(), call.args.get(1)) else {
        return Err(cx.invariant("air.dot without its operands"));
    };
    super::check_operands(cx, &[a, b])?;
    super::lowered_type(cx.entry, cx.profile, inst.ty)?;
    let ty = cx
        .value_ty(a)
        .ok_or_else(|| cx.invariant("a dot operand without a type"))?;
    let shape = shape_of(cx, ty)?;
    let elem = shape.lane_form();
    let f64_ok = elem.bits() != 64 || FloatFn::Dot.allows_f64();
    if !shape.chunked || !elem.is_float() || !f64_ok || cx.value_ty(b) != Some(ty) {
        return Err(not_lowered());
    }
    let answer = elem.float_fmt().and_then(|fmt| {
        super::nan::function_answer(FloatFn::Dot, false, super::every_flag(call.fmf), fmt)
    });
    super::note(cx, FloatFn::Dot.precision());
    super::device_float(cx, elem, true, answer.is_none());
    super::fp_env(cx, call.fmf, elem);
    let s = ArithStrategy::ChunkedDot {
        lanes: shape.lanes,
        elem,
        contract: super::contraction(call.fmf),
    };
    Ok(super::with_nan(cx, s, answer))
}

/// The dot product of chunked vectors `a` and `b` of `lanes` lanes of `elem`: each chunk's `OpDot`
/// (a multiply for a one-lane chunk), summed, and rounded once to a narrow format.
pub fn dot(
    fb: &mut FunctionBuilder<'_>,
    lanes: u16,
    elem: Form,
    contract: Contraction,
    a: SValue,
    b: SValue,
) -> BResult<SValue> {
    let mut sum: Option<SValue> = None;
    for (c, k) in chunk_lanes(u32::from(lanes)).into_iter().enumerate() {
        let x = fb.composite_extract(a, &[c as u32])?;
        let y = fb.composite_extract(b, &[c as u32])?;
        let part = if k == 1 {
            fb.binary_with(BinOp::FMul, x, y, contract)?
        } else {
            fb.binary(BinOp::Dot, x, y)?
        };
        sum = Some(match sum {
            None => part,
            Some(s) => fb.binary_with(BinOp::FAdd, s, part, contract)?,
        });
    }
    let sum = sum.ok_or_else(operand_error)?;
    narrow::round(fb, elem, sum)
}

/// The words of [`dot`] on `lanes` lanes of `elem`.
pub fn dot_words(lanes: u16, elem: Form) -> u32 {
    let chunks = u32::from(lanes).div_ceil(4);
    24u32
        .saturating_mul(chunks)
        .saturating_add(8 + narrow::round_words(elem))
}

/// The sine and cosine of `x` of shape `s`: per chunk for a chunked vector, Metal's fast ones for
/// `air.fast_sincos` on `float` (`fast`, [`sin_cos_fns`]), each the device's instruction as `lib`
/// reads it ([`math::float_fn`]).
pub fn sin_cos(
    fb: &mut FunctionBuilder<'_>,
    s: Shape,
    x: SValue,
    fast: bool,
    lib: &Lib,
) -> BResult<(SValue, SValue)> {
    let (sin, cos) = sin_cos_fns(fast);
    let one = |fb: &mut FunctionBuilder<'_>, f: Form, x: SValue| -> BResult<(SValue, SValue)> {
        let c = Contraction::Forbidden;
        Ok((
            math::float_fn(fb, sin, f, c, &[x], lib)?,
            math::float_fn(fb, cos, f, c, &[x], lib)?,
        ))
    };
    if !s.chunked {
        return one(fb, s.form(), x);
    }
    let (mut sins, mut coss) = (Vec::new(), Vec::new());
    for (c, k) in chunk_lanes(u32::from(s.lanes)).into_iter().enumerate() {
        let xc = fb.composite_extract(x, &[c as u32])?;
        let (sn, cs) = one(
            fb,
            Form {
                held: s.held,
                lanes: k,
            },
            xc,
        )?;
        sins.push(sn);
        coss.push(cs);
    }
    let t = legal::spirv_type(fb.module(), &s.demote())?.ok_or_else(operand_error)?;
    Ok((
        fb.composite_construct(t, &sins)?,
        fb.composite_construct(t, &coss)?,
    ))
}

/// The sine and cosine [`sin_cos`] computes: `air.fast_sincos`'s on `float` when `fast`.
pub fn sin_cos_fns(fast: bool) -> (FloatFn, FloatFn) {
    if fast {
        (FloatFn::FastSin, FloatFn::FastCos)
    } else {
        (FloatFn::Sin, FloatFn::Cos)
    }
}

/// The words of [`sin_cos`] on shape `s`.
pub fn sin_cos_words(s: Shape, fast: bool) -> u32 {
    let (sin, cos) = sin_cos_fns(fast);
    let per = |f: Form| sin.words(f).saturating_add(cos.words(f)).saturating_add(8);
    if !s.chunked {
        return per(s.form());
    }
    chunk_lanes(u32::from(s.lanes))
        .into_iter()
        .map(|k| {
            per(Form {
                held: s.held,
                lanes: k,
            })
        })
        .fold(16u32, u32::saturating_add)
}

/// Lanes a word bitcast moves: 8, 16, 32, or 64 bits of a non-`bool` type, whole words in all.
fn word_movable(s: Shape) -> bool {
    let bits = Form {
        held: s.held,
        lanes: 1,
    }
    .bits();
    matches!(bits, 8 | 16 | 32 | 64)
        && !matches!(s.held, Held::Bool)
        && (u32::from(bits) * u32::from(s.lanes)) % 32 == 0
}

/// An upper estimate of the SPIR-V words of a lane move over `n` lanes.
pub fn lane_words(n: u16, per: u32) -> u32 {
    per.saturating_mul(u32::from(n)).saturating_add(16)
}

/// A lane move, for its word bound ([`move_words`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMove {
    Extract,
    Insert,
    Shuffle,
}

/// Words to read a two-word lane ([`lane`]): two `OpCompositeExtract` of at most 3 indices (7
/// words each) and the `uvec2` construct (5).
const WIDE_READ: u32 = 2 * 7 + 5;
/// Words to write a two-word lane ([`with_lane`]): its split (two 1-index extracts, 5 words each)
/// and two `OpCompositeInsert` of at most 3 indices (8 each).
const WIDE_WRITE: u32 = 2 * 5 + 2 * 8;
/// A two-word lane's share of building a vector ([`build`]): its split (10 words), its
/// constituent in the low-word and high-word constructs (2), and at most half of a 2-lane
/// chunk's two construct headers, join, and member constituent (`(2 * 3 + 5 + 1) / 2`).
const WIDE_BUILD: u32 = 2 * 5 + 2 + 6;
/// Words per lane of a dynamic index: the lane constant (4), `OpIEqual` (5), `OpSelect` (6).
const PICK: u32 = 4 + 5 + 6;

/// An upper estimate of the SPIR-V words of lane move `m` on shape `s` (dynamic: by a dynamic
/// index), from the emission below. A lane held whole is one extract or insert (the flat bounds); a
/// two-word lane is read, written, and gathered word by word (19, 26, and 18 words, and 15 per lane
/// of a dynamic index), each move with 8 words of slack for a new type.
pub fn move_words(s: Shape, m: LaneMove, dynamic: bool) -> u32 {
    let n = s.lanes;
    if !matches!(s.held, Held::Wide { .. }) {
        return match (m, dynamic) {
            (LaneMove::Extract | LaneMove::Insert, false) => 8,
            (LaneMove::Extract, true) => lane_words(n, 24),
            (LaneMove::Insert, true) => lane_words(n, 40),
            (LaneMove::Shuffle, _) => lane_words(n, 16),
        };
    }
    match (m, dynamic) {
        (LaneMove::Extract, false) => WIDE_READ + 8,
        (LaneMove::Insert, false) => WIDE_WRITE + 8,
        (LaneMove::Extract, true) => lane_words(n, WIDE_READ + PICK) + 8,
        (LaneMove::Insert, true) => lane_words(n, WIDE_READ + PICK + WIDE_BUILD) + 8,
        (LaneMove::Shuffle, _) => lane_words(n, WIDE_READ + WIDE_BUILD) + 8,
    }
}

/// The words of a [`ArithStrategy::WordBitcast`].
pub fn bitcast_words(from: Shape, to: Shape) -> u32 {
    let per = |s: Shape| 64 + narrow::bits_words(s.form().scalar_form());
    lane_words(from.lanes, per(from)).saturating_add(lane_words(to.lanes, per(to)))
}

// ---- Emission ----

fn operand_error() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::CompositeExtract,
        rule: OperandRule::Arity,
    }
}

/// Lane `i` of `v` (shape `s`) as a scalar (a two-word lane as its `uvec2`).
fn lane(fb: &mut FunctionBuilder<'_>, s: Shape, v: SValue, i: u32) -> BResult<SValue> {
    match s.path(i) {
        LanePath::Whole(p) if p.is_empty() => Ok(v),
        LanePath::Whole(p) => fb.composite_extract(v, &p),
        LanePath::Words(lo, hi) => {
            let lo = fb.composite_extract(v, &lo)?;
            let hi = fb.composite_extract(v, &hi)?;
            wide::join(fb, s.lane_form(), wide::Pair { lo, hi })
        }
    }
}

/// `v` (shape `s`) with lane `i` replaced by scalar `x`.
fn with_lane(
    fb: &mut FunctionBuilder<'_>,
    s: Shape,
    v: SValue,
    i: u32,
    x: SValue,
) -> BResult<SValue> {
    match s.path(i) {
        LanePath::Whole(p) if p.is_empty() => Ok(x),
        LanePath::Whole(p) => fb.composite_insert(x, v, &p),
        LanePath::Words(lo, hi) => {
            let w = wide::split(fb, x)?;
            let v = fb.composite_insert(w.lo, v, &lo)?;
            fb.composite_insert(w.hi, v, &hi)
        }
    }
}

/// Every lane of `v` (shape `s`) as a scalar.
fn lanes(fb: &mut FunctionBuilder<'_>, s: Shape, v: SValue) -> BResult<Vec<SValue>> {
    let mut out = Vec::with_capacity(usize::from(s.lanes));
    for i in 0..u32::from(s.lanes) {
        out.push(lane(fb, s, v, i)?);
    }
    Ok(out)
}

/// The scalar or vector of `xs.len()` lanes of `held` with scalar lanes `xs` (two-word lanes
/// gathered into their low-word and high-word vectors).
fn vector_of(fb: &mut FunctionBuilder<'_>, held: Held, xs: &[SValue]) -> BResult<SValue> {
    let lanes = u8::try_from(xs.len()).map_err(|_| operand_error())?;
    if lanes == 1 {
        return xs.first().copied().ok_or_else(operand_error);
    }
    let f = Form { held, lanes };
    if !f.is_wide() {
        let t = ops::ty(fb, f)?;
        return fb.composite_construct(t, xs);
    }
    let t = ops::ty(fb, wide::words_form(lanes))?;
    let (mut los, mut his) = (Vec::with_capacity(xs.len()), Vec::with_capacity(xs.len()));
    for x in xs {
        let w = wide::split(fb, *x)?;
        los.push(w.lo);
        his.push(w.hi);
    }
    let lo = fb.composite_construct(t, &los)?;
    let hi = fb.composite_construct(t, &his)?;
    wide::join(fb, f, wide::Pair { lo, hi })
}

/// The value of shape `s` with scalar lanes `xs`.
fn build(fb: &mut FunctionBuilder<'_>, s: Shape, xs: &[SValue]) -> BResult<SValue> {
    if !s.chunked {
        return vector_of(fb, s.held, xs);
    }
    let mut members = Vec::new();
    let mut at = 0usize;
    for k in chunk_lanes(u32::from(s.lanes)) {
        let part = xs.get(at..at + usize::from(k)).ok_or_else(operand_error)?;
        members.push(vector_of(fb, s.held, part)?);
        at += usize::from(k);
    }
    let Some(t) = legal::spirv_type(fb.module(), &s.demote())? else {
        return Err(operand_error());
    };
    fb.composite_construct(t, &members)
}

/// A dynamic lane index `i` of form `index` as a held integer: a two-word index's low word (an
/// index of `2^32` or more is out of range, so any lane or none will do).
fn index_value(fb: &mut FunctionBuilder<'_>, index: Form, i: SValue) -> BResult<(Form, SValue)> {
    if index.is_wide() {
        Ok((wide::words_form(1), wide::low_word(fb, i)?))
    } else {
        Ok((index, i))
    }
}

/// Emit a lane-wise operation per chunk: `parts` holds the operation per chunk size, `wide`
/// which operands are chunked.
pub fn emit_chunked(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    lanes: u16,
    parts: &[(u8, ArithStrategy)],
    wide: &[bool],
    result: &DemoteTy,
    xs: &[SValue],
) -> BResult<SValue> {
    let mut members = Vec::new();
    for (c, k) in chunk_lanes(u32::from(lanes)).into_iter().enumerate() {
        let Some((_, s)) = parts.iter().find(|(n, _)| *n == k) else {
            return Err(cx.fail("a chunk size without a strategy"));
        };
        let mut xc = Vec::with_capacity(xs.len());
        for (x, w) in xs.iter().zip(wide) {
            xc.push(if *w {
                fb.composite_extract(*x, &[c as u32])?
            } else {
                *x
            });
        }
        members.push(super::value(cx, fb, s, &xc)?);
    }
    let Some(t) = legal::spirv_type(fb.module(), result)? else {
        return Err(cx.fail("a chunked result arith does not type"));
    };
    fb.composite_construct(t, &members)
}

/// `index == k` for each lane `k` of `n`, as selects over `xs` (lane 0 when out of range).
fn pick(fb: &mut FunctionBuilder<'_>, index: Form, i: SValue, xs: &[SValue]) -> BResult<SValue> {
    let mut r = xs.first().copied().ok_or_else(operand_error)?;
    for (k, x) in xs.iter().enumerate().skip(1) {
        let kk = ops::splat(fb, index, k as u64)?;
        let eq = fb.binary(BinOp::IEqual, i, kk)?;
        r = fb.select(eq, *x, r)?;
    }
    Ok(r)
}

/// `extractelement`: lane `at` (constant) or lane `index` (dynamic, of form `form`).
pub fn extract(
    fb: &mut FunctionBuilder<'_>,
    src: Shape,
    at: Option<u32>,
    form: Option<Form>,
    xs: &[SValue],
) -> BResult<SValue> {
    let v = xs.first().copied().ok_or_else(operand_error)?;
    match (at, form) {
        (Some(i), _) => lane(fb, src, v, i),
        (None, Some(f)) => {
            let i = xs.get(1).copied().ok_or_else(operand_error)?;
            let (f, i) = index_value(fb, f, i)?;
            let ls = lanes(fb, src, v)?;
            pick(fb, f, i, &ls)
        }
        (None, None) => Err(operand_error()),
    }
}

/// `insertelement` of `xs[1]` into `xs[0]` at lane `at` (constant) or `xs[2]` (dynamic).
pub fn insert(
    fb: &mut FunctionBuilder<'_>,
    dst: Shape,
    at: Option<u32>,
    form: Option<Form>,
    xs: &[SValue],
) -> BResult<SValue> {
    let v = xs.first().copied().ok_or_else(operand_error)?;
    let e = xs.get(1).copied().ok_or_else(operand_error)?;
    match (at, form) {
        (Some(i), _) => with_lane(fb, dst, v, i, e),
        (None, Some(f)) => {
            let i = xs.get(2).copied().ok_or_else(operand_error)?;
            let (f, i) = index_value(fb, f, i)?;
            let mut ls = lanes(fb, dst, v)?;
            for (k, l) in ls.iter_mut().enumerate() {
                let kk = ops::splat(fb, f, k as u64)?;
                let eq = fb.binary(BinOp::IEqual, i, kk)?;
                *l = fb.select(eq, e, *l)?;
            }
            build(fb, dst, &ls)
        }
        (None, None) => Err(operand_error()),
    }
}

/// `shufflevector` of `xs[0]` (shape `a`) and `xs[1]` (shape `b`) into shape `out`.
pub fn shuffle(
    fb: &mut FunctionBuilder<'_>,
    a: Shape,
    b: Shape,
    out: Shape,
    mask: &[u32],
    xs: &[SValue],
) -> BResult<SValue> {
    let va = xs.first().copied().ok_or_else(operand_error)?;
    let vb = xs.get(1).copied().ok_or_else(operand_error)?;
    let na = u32::from(a.lanes);
    let mut picked = Vec::with_capacity(mask.len());
    for m in mask {
        picked.push(if *m < na {
            lane(fb, a, va, *m)
        } else {
            lane(fb, b, vb, *m - na)
        }?);
    }
    build(fb, out, &picked)
}

/// The lanes a simdgroup matrix's fragment holds: elements 0 and 1 (`memory::fragment`).
const FRAGMENT_LANES: usize = 2;

/// Whether a chunked shape of 64 lanes, each held whole, can be built as a fragment
/// ([`shuffle_fragment`]): its first chunk holds both fragment lanes.
fn fragment_shape(s: Shape) -> bool {
    s.chunked
        && s.lanes == 64
        && !matches!(s.held, Held::Wide { .. })
        && chunk_lanes(u32::from(s.lanes))
            .first()
            .is_some_and(|&k| usize::from(k) >= FRAGMENT_LANES)
}

/// An upper estimate of [`shuffle_fragment`]'s words on result shape `s`: two lane reads (an
/// `OpCompositeExtract` of two indices, 6 words each), the zero lane and the null chunk (3 words
/// each), the first chunk's construct (7), and the result's (3 and a word per chunk), with 8
/// words of slack.
pub fn fragment_shuffle_words(s: Shape) -> u32 {
    let chunks = chunk_lanes(u32::from(s.lanes)).len() as u32;
    let fixed: u32 = 2 * 6 + 3 + 3 + 7 + 3 + 8;
    fixed.saturating_add(chunks)
}

/// `shufflevector` whose result is a simdgroup matrix's fragment: lanes 0 and 1 as the mask picks
/// them, every other lane zero (a null chunk past the first).
pub fn shuffle_fragment(
    fb: &mut FunctionBuilder<'_>,
    a: Shape,
    b: Shape,
    out: Shape,
    mask: &[u32],
    xs: &[SValue],
) -> BResult<SValue> {
    let va = xs.first().copied().ok_or_else(operand_error)?;
    let vb = xs.get(1).copied().ok_or_else(operand_error)?;
    let na = u32::from(a.lanes);
    let mut picked = Vec::with_capacity(4);
    for m in mask.iter().take(FRAGMENT_LANES) {
        picked.push(if *m < na {
            lane(fb, a, va, *m)
        } else {
            lane(fb, b, vb, *m - na)
        }?);
    }
    let first = *picked.first().ok_or_else(operand_error)?;
    let lane_ty = fb.value_ty(first)?;
    let zero = fb.module().constant(Constant::Null { ty: lane_ty })?;
    let zero = fb.module().const_value(zero)?;
    let DemoteTy::Composite(chunks) = out.demote() else {
        return Err(operand_error());
    };
    let mut members = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            let DemoteTy::Vector { len, .. } = chunk else {
                return Err(operand_error());
            };
            picked.resize(usize::from(*len), zero);
            members.push(vector_of(fb, out.held, &picked)?);
            continue;
        }
        let Some(t) = legal::spirv_type(fb.module(), chunk)? else {
            return Err(operand_error());
        };
        let null = fb.module().constant(Constant::Null { ty: t })?;
        members.push(fb.module().const_value(null)?);
    }
    let Some(t) = legal::spirv_type(fb.module(), &out.demote())? else {
        return Err(operand_error());
    };
    fb.composite_construct(t, &members)
}

/// The 32-bit words of `v` (shape `s`), low lane first.
fn words_of(fb: &mut FunctionBuilder<'_>, s: Shape, v: SValue) -> BResult<Vec<SValue>> {
    let lane = s.form().scalar_form();
    let u = narrow::u32_form(lane);
    let ut = ops::ty(fb, u)?;
    let bits = u32::from(lane.bits());
    let mut words = Vec::new();
    let mut acc: Option<SValue> = None;
    let mut at = 0u32;
    for x in lanes(fb, s, v)? {
        match (bits, lane.held) {
            // Two words already.
            (_, Held::Wide { .. }) => {
                words.push(fb.composite_extract(x, &[0])?);
                words.push(fb.composite_extract(x, &[1])?);
            }
            (64, _) => {
                let pair = ops::ty(
                    fb,
                    Form {
                        held: u.held,
                        lanes: 2,
                    },
                )?;
                let p = fb.convert(ConvOp::Bitcast, x, pair)?;
                words.push(fb.composite_extract(p, &[0])?);
                words.push(fb.composite_extract(p, &[1])?);
            }
            (32, Held::Float(_)) => words.push(fb.convert(ConvOp::Bitcast, x, ut)?),
            (32, _) => words.push(x),
            _ => {
                // An integer lane masked to its width: a poison lane (a `nuw` operation's,
                // `ops::int_binary`) may exceed it, and never reaches its neighbour's bits.
                let b = if lane.is_narrow() {
                    narrow::to_bits(fb, lane, x)?
                } else {
                    ops::mask(fb, lane, x)?
                };
                let placed = if at == 0 {
                    b
                } else {
                    let k = ops::splat(fb, u, u64::from(at))?;
                    fb.binary(BinOp::ShiftLeftLogical, b, k)?
                };
                acc = Some(match acc {
                    None => placed,
                    Some(w) => fb.binary(BinOp::BitwiseOr, w, placed)?,
                });
                at += bits;
                if at == 32 {
                    words.extend(acc.take());
                    at = 0;
                }
            }
        }
    }
    Ok(words)
}

/// The value of shape `s` whose words are `ws`, low lane first.
fn from_words(fb: &mut FunctionBuilder<'_>, s: Shape, ws: &[SValue]) -> BResult<SValue> {
    let lane = s.form().scalar_form();
    let u = narrow::u32_form(lane);
    let bits = u32::from(lane.bits());
    let lt = ops::ty(fb, lane)?;
    let mut xs = Vec::with_capacity(usize::from(s.lanes));
    let mut next = ws.iter().copied();
    let mut word: Option<SValue> = None;
    let mut at = 0u32;
    for _ in 0..s.lanes {
        match (bits, lane.held) {
            (64, _) => {
                let (lo, hi) = (
                    next.next().ok_or_else(operand_error)?,
                    next.next().ok_or_else(operand_error)?,
                );
                let pair = ops::ty(
                    fb,
                    Form {
                        held: u.held,
                        lanes: 2,
                    },
                )?;
                let p = fb.composite_construct(pair, &[lo, hi])?;
                // A two-word lane is the pair itself.
                xs.push(if lane.is_wide() {
                    p
                } else {
                    fb.convert(ConvOp::Bitcast, p, lt)?
                });
            }
            (32, Held::Float(_)) => {
                let w = next.next().ok_or_else(operand_error)?;
                xs.push(fb.convert(ConvOp::Bitcast, w, lt)?);
            }
            (32, _) => xs.push(next.next().ok_or_else(operand_error)?),
            _ => {
                if at == 0 {
                    word = Some(next.next().ok_or_else(operand_error)?);
                }
                let w = word.ok_or_else(operand_error)?;
                let v = if at == 0 {
                    w
                } else {
                    let k = ops::splat(fb, u, u64::from(at))?;
                    fb.binary(BinOp::ShiftRightLogical, w, k)?
                };
                let m = ops::splat(fb, u, (1u64 << bits) - 1)?;
                let v = fb.binary(BinOp::BitwiseAnd, v, m)?;
                xs.push(if lane.is_narrow() {
                    narrow::from_bits(fb, lane, v)?
                } else {
                    v
                });
                at = (at + bits) % 32;
            }
        }
    }
    build(fb, s, &xs)
}

/// A bitcast with a chunked side, through 32-bit words (both sides move whole words of lanes
/// of 8, 16, 32, or 64 bits, `word_movable`).
pub fn bitcast(fb: &mut FunctionBuilder<'_>, from: Shape, to: Shape, v: SValue) -> BResult<SValue> {
    let ws = words_of(fb, from, v)?;
    from_words(fb, to, &ws)
}
