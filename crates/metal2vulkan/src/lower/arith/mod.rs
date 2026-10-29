//! `lower::arith`: math, conversions, comparisons, and composites.
//!
//! `lower::select` routes every non-pointer arithmetic instruction here (binary operators,
//! `fneg`, numeric casts and bitcasts, `icmp`/`fcmp`, `select`, `freeze`, vector and aggregate
//! element operations) together with the `air` math, conversion, and packing intrinsics and the
//! `llvm` math intrinsics. [`select`] decides one [`ArithStrategy`] per instruction; [`emit`]
//! routes it to the emission core in [`ops`] (the LLVM instruction set) and [`math`] (the
//! intrinsics). `air.sincos` stores its cosine through the access family's store. Everything not
//! lowered yet is refused as `Unsupported{Lowering(Arith), NotImplemented}`: `air.pack` to the
//! packed float formats `rg11b10f` and `rgb9e5` (their unpacking is lowered), `i1` results of
//! `air.convert` from anything but a `float` or `half` (whose reading is Metal's), and the few
//! chunked-vector operations [`chunk`] lists. A vector of other than 2 to 4 lanes is held as a
//! composite of chunks and operated on per chunk ([`chunk`]).
//!
//! This family also owns the **lowered representation of non-pointer values** ([`legal`]):
//! [`lowered_type`] is what `select` demotes block-crossing values and phis to and what it requires
//! of terminator operands, [`constant`] is which constants lowering materializes, and
//! [`spirv_type`] and [`materialize`] are their emission. The representation depends on the AIR
//! type and `shaderInt64` only: `i1` is `bool`, other integers are held zero-extended in a 32- or
//! 64-bit unsigned container (so `i8`/`i16`/`i2`/`i40` need no narrow capability), `i33`..`i64` as
//! two `u32` words without `shaderInt64` ([`wide`]: every operation on them is
//! `ArithStrategy::Wide`, per chunk in a chunked vector; division unrolled without a loop), `float`
//! and `double` are native (`f64` needs `shaderFloat64`), and `half` and `bfloat` are held widened
//! in an `f32` that is always a value of the narrow format ([`narrow`]): each operation rounds its
//! `f32` result back to the format, so no profile needs `shaderFloat16`; where the device's 16-bit
//! `+ - *` is Metal's `half` arithmetic and the profile has `shaderFloat16`, a `half` `+ - *` is
//! that instruction, kept apart, between exact conversions instead ([`ArithStrategy::F16`]): where
//! it is measured to be, and where the 16-bit float controls the module then declares make it so
//! by the Vulkan spec ([`narrow::controlled_modes`]).
//!
//! Precision: a `half` rounding keeps half denormals, as Metal does; a `float` or `half` fused
//! multiply-add (`air.fma`, `llvm.fma`, and `llvm.fmuladd`, which Metal fuses) rounds once where
//! the profile has `OpFmaKHR` or the device's 16-bit `Fma` is Metal's, and is the device's `Fma`
//! elsewhere (`PrecisionNote::NativeFma`, [`fma`]); operations whose Vulkan precision is looser
//! than correct rounding add
//! `PrecisionNote::DeviceArithmetic` (`fdiv`, `frem`, `sqrt`, a `double` or `bfloat` `fma`, `fmod`,
//! `dot`) or `PrecisionNote::DeviceTranscendentals` (the transcendental functions) to the plan.
//! Float arithmetic is emitted in the AIR's order, one instruction per operation, each decorated by
//! its own fast-math flags: one carrying both of LLVM's `contract` and `reassoc` is left
//! undecorated, so the device's compiler groups and fuses it as those flags allow; any other is
//! `NoContraction`, so no device fuses or regroups operations that LLVM keeps apart. Arithmetic
//! the flags let a compiler regroup or fuse with its neighbour, and a `bfloat` or `double`
//! `llvm.fmuladd`, add `PrecisionNote::FastMathRewrites`: its value is any of those the flags allow, and Metal's
//! compiler may take another than the device's. Every other lowering is exact for every non-poison
//! input within the device's floating-point environment: signed zeros, infinities, and NaNs are
//! preserved by `SignedZeroInfNanPreserve` where an operation without `nnan`/`ninf`/`nsz` requires
//! it and the profile supports it (else `PrecisionNote::DeviceSignedZeroInfNan`), while denormal
//! flushing and the rounding mode are the device default (`f32` denormals noted
//! `PrecisionNote::DeviceDenormals` where the device is not measured to read them as Metal does,
//! [`denormal`]; the rounding mode decides the inexact `FConvert` narrowings and integer-to-float
//! conversions, which Vulkan leaves to the device without `RoundingModeRTE`), but in an entry that
//! rounds to `half` under the float controls ([`narrow::controlled_modes`]), which declares
//! `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits and the modes the
//! device's independence ties to them: `RoundingModeRTE` for 32 bits on a device whose rounding
//! modes are not independent, Metal's own rounding of `float` arithmetic, and for 64 bits the
//! rounding and denormals IEEE 754 gives `double`. A function the device has an instruction for is
//! that instruction, its special operands the device's ([`math`]); the normalized and sRGB
//! `air.pack`/`air.unpack` conversions are GLSL.std.450's `Pack`/`Unpack` instructions where it has
//! one and the device's multiply, `RoundEven` and clamp elsewhere ([`pack`],
//! `PrecisionNote::DevicePackConversion`).
//!
//! A NaN result's bits are the device's ([`nan`]): where Metal answers one the device is not
//! measured to answer (its canonical `0x7FC00000` or `0x7E00` on a device without
//! `NanEncoding::MetalCanonical`, and a NaN that carries an operand's sign or bits on every
//! device), the module records `PrecisionNote::NativeNanEncoding`.

pub mod aggregate;
pub mod chunk;
pub mod color_write;
pub mod denormal;
pub mod divide;
pub mod exact_division;
pub mod extended_range;
pub mod fma;
pub mod half_chain;
pub mod legal;
pub mod libm;
pub mod math;
pub mod nan;
pub mod narrow;
pub mod ops;
pub mod output_precision;
pub mod pack;
pub mod srgb;
pub mod texel_round;
pub mod wide;

#[cfg(test)]
mod tests;

use crate::air::intrinsics::{
    AirIntrinsic, Convert, LlvmFloatOp, LlvmIntOp, LlvmIntrinsic, MathFn, NumKind, Pack, PackDir,
    PackFormat, Sign as AirSign,
};
use crate::air::intrinsics::{Elem, Math};
use crate::air::ir::{
    BinOp as AirBin, BlockId, CastOp, Const, FastMath, FloatPredicate, Inst, InstId, InstKind,
    IntPredicate, Type, TypeId as AirTypeId, Value, ValueId,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::device::profile::{DriverWorkaround, FloatFunctions, NanEncoding};
use crate::device::{DeviceProfile, FloatWidth};
use crate::error::Error;
use crate::lower::access;
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::pipeline::artifact::{EntryMath, PrecisionNote};
use crate::spirv::builder::{FunctionBuilder, MaskForm, ModuleBuilder, ProductForm, QuantizeForm};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{Contraction, ExecMode, UnOp};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use legal::{check, demote_of, form_on, not_lowered};
pub use legal::{FloatFmt, Form, Held, LoweredConst};
use math::{ConvertKind, FloatFn, FloatPrecision, IntFn, Sign};
use ops::{BitcastPlan, CastKind};

/// The lowered representation of a non-pointer AIR value of type `ty` on profile `p`: a type every
/// lowered value of `ty` has, which is always demotable. A type this family does not lower yet is
/// `Unsupported{Lowering(Arith), NotImplemented}`; a type that needs a capability the profile lacks
/// is a typed refusal naming it.
///
/// On a profile without `shaderInt64` every 64-bit integer scalar or vector is held as two words
/// ([`legal::widen`]), each chunk of a chunked vector of them too.
pub fn lowered_type(e: &Entry, p: &DeviceProfile, ty: AirTypeId) -> Result<DemoteTy, Error> {
    let mut d = demote_of(e.module.types(), ty)?;
    if legal::holds_wide(p) {
        d = legal::widen(&d);
    }
    check(p, &d)?;
    Ok(d)
}

/// `v` as a constant lowering materializes, in its profile-independent representation (64-bit
/// integers native). It is `Some` only for constants of a type [`lowered_type`] lowers on some
/// profile; `select` checks the profile. Emission materializes [`constant_in`].
pub fn constant(e: &Entry, v: Value) -> Option<LoweredConst> {
    let Value::Const(c) = v else { return None };
    legal::lower_const(e, c)
}

/// [`constant`] in the representation of the plan's profile: 64-bit integers as two words where
/// the plan holds them so.
pub fn constant_in(e: &Entry, plan: &ArithPlan, v: Value) -> Option<LoweredConst> {
    let k = constant(e, v)?;
    Some(if plan.wide { legal::widen_const(k) } else { k })
}

/// [`constant`] in lowered representation `d`: the constant itself, or its wide form when `d` is
/// the wide representation of its type; `None` when neither is `d`.
pub fn constant_as(e: &Entry, v: Value, d: &DemoteTy) -> Option<LoweredConst> {
    let k = constant(e, v)?;
    if k.demote() == *d {
        return Some(k);
    }
    let w = legal::widen_const(k);
    (w.demote() == *d).then_some(w)
}

/// The SPIR-V type of lowered representation `ty`, or `None` for one this family does not
/// produce.
pub fn spirv_type(mb: &mut ModuleBuilder, ty: &DemoteTy) -> BResult<Option<TypeId>> {
    legal::spirv_type(mb, ty)
}

/// Materialize constant `k` in the open function, or `None` for one this family does not
/// produce.
pub fn materialize(fb: &mut FunctionBuilder<'_>, k: &LoweredConst) -> BResult<Option<SValue>> {
    legal::materialize(fb, k)
}

/// The arith strategies. Operands are the instruction's value operands (a call's arguments) in
/// order; each variant carries the forms it was selected for.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArithStrategy {
    /// An integer binary operator on held integers.
    Int {
        op: AirBin,
        form: Form,
    },
    /// A binary operator on `i1` values.
    Bool {
        op: AirBin,
        form: Form,
    },
    /// A float binary operator on floats of `form`, in the AIR's order; `contract` is `Forbidden`
    /// unless the instruction carries both `contract` and `reassoc` ([`contraction`]).
    Float {
        op: AirBin,
        form: Form,
        contract: Contraction,
    },
    /// A `half` `fadd`, `fsub` or `fmul` on lanes of `form` by the device's 16-bit arithmetic,
    /// Metal's own in the forms that keep it apart (measured, `DriverDeviation::MetalHalfArithmetic`,
    /// or by the Vulkan spec under the 16-bit float controls, [`ops::f16_binary`]): the operands
    /// narrowed to 16 bits, exactly, the instruction kept apart, and its result widened, exactly.
    F16 {
        op: AirBin,
        form: Form,
    },
    /// `fneg` on floats of `form`: the device's `OpFNegate`.
    FNeg {
        form: Form,
    },
    IntCompare {
        pred: IntPredicate,
        form: Form,
    },
    BoolCompare {
        pred: IntPredicate,
    },
    FloatCompare {
        pred: FloatPredicate,
        form: Form,
    },
    Cast {
        kind: CastKind,
        from: Form,
        to: Form,
    },
    Bitcast {
        plan: BitcastPlan,
        from: Form,
        to: Form,
    },
    /// `OpSelect` (any lowered representation).
    Select,
    /// The first operand itself (`freeze` of a value, unsigned `abs`).
    Copy,
    /// The zero of the result's representation (`freeze` of `undef`/`poison`).
    Zero(DemoteTy),
    /// `OpUndef` of the result's representation (an element index known out of range).
    Undef(DemoteTy),
    ExtractConst(u32),
    /// `OpVectorExtractDynamic`, its index of form `index` made an integer
    /// ([`ops::element_index`]).
    ExtractDynamic {
        index: Form,
    },
    InsertConst(u32),
    /// `OpVectorInsertDynamic`, its index of form `index` made an integer
    /// ([`ops::element_index`]).
    InsertDynamic {
        index: Form,
    },
    /// Component indices into the concatenation of the two operands.
    Shuffle(Box<[u32]>),
    ExtractValue(Box<[u32]>),
    InsertValue(Box<[u32]>),
    /// `extractvalue` of a 2-4-lane `bool` vector member at `path`, which its aggregate holds as
    /// `u32` lanes of 0 and 1 (`legal::as_member`): the extract, then a comparison with 0.
    ExtractBools {
        path: Box<[u32]>,
        lanes: u8,
    },
    /// `insertvalue` of a 2-4-lane `bool` vector into such a member at `path`: a select of 1 and
    /// 0, then the insert.
    InsertBools {
        path: Box<[u32]>,
        lanes: u8,
    },
    IntFn {
        f: IntFn,
        form: Form,
    },
    /// A float function; `contract` (from the call's fast-math flags) governs the multiply and
    /// add of a sequence such as `mix`.
    FloatFn {
        f: FloatFn,
        form: Form,
        contract: Contraction,
    },
    /// A fused multiply-add on `float` or `half` lanes ([`fma`]): `OpFmaKHR`, the device's 16-bit
    /// `Fma` or `OpFmaKHR`, or the device's `Fma`.
    Fma(fma::Fma),
    /// `air.fast_fmod` on `float`: Metal's `fma(-trunc(x / y), y, x)` of the device's quotient
    /// (`math::fast_fmod`), the fused multiply-add `fma`.
    FastFmod {
        fma: fma::Fma,
    },
    /// An `fdiv` carrying `arcp` on `float` lanes, Metal's fast division: the device's
    /// `OpFDiv(x, y)`, contracting as `contract` allows (`PrecisionNote::NativeDivision`).
    FastDivide {
        form: Form,
        contract: Contraction,
    },
    /// Metal's precise division on `float` or `half` lanes (an `fdiv` without `arcp` on `float`,
    /// any `fdiv` on `half`): the device's `OpFDiv`, its divisor kept in Vulkan's range
    /// ([`divide::native`], `PrecisionNote::NativeDivision`).
    NativeDivide {
        form: Form,
    },
    Convert {
        kind: ConvertKind,
        from: Form,
        to: Form,
    },
    /// `air.pack`/`air.unpack` of `format` between float lanes of form `floats` and a `u32`.
    Pack {
        dir: PackDir,
        format: PackFormat,
        floats: Form,
    },
    /// A `zext`/`sext` to a 64-bit integer that the access family consumes as a narrow GEP index
    /// (`access::index`): nothing is emitted, and no 64-bit value is formed.
    IndexExtension,
    /// `i64` arithmetic whose only uses are narrow GEP indices, held as its low word
    /// (`access::index`): a `u32` operation, so no 64-bit value is formed.
    IndexWord(access::index::IndexWord),
    /// A lane-wise operation on chunked vectors of `lanes` lanes: `parts` is the operation selected
    /// for each distinct chunk size, `wide` which operands are chunked, and `result` the result's
    /// representation.
    Chunked {
        lanes: u16,
        parts: Box<[(u8, ArithStrategy)]>,
        wide: Box<[bool]>,
        result: DemoteTy,
    },
    /// `extractelement` from a vector of shape `src`: lane `at`, or the lane a dynamic index of
    /// form `index` names.
    LaneExtract {
        src: chunk::Shape,
        at: Option<u32>,
        index: Option<Form>,
    },
    /// `insertelement` into a vector of shape `dst`, as [`ArithStrategy::LaneExtract`].
    LaneInsert {
        dst: chunk::Shape,
        at: Option<u32>,
        index: Option<Form>,
    },
    /// `shufflevector` with a chunked operand or result: result lane `i` is lane `mask[i]` of the
    /// operands' concatenation.
    LaneShuffle {
        a: chunk::Shape,
        b: chunk::Shape,
        out: chunk::Shape,
        mask: Box<[u32]>,
        /// The result is a simdgroup matrix's fragment (`MemoryPlan::is_fragment`): only lanes 0
        /// and 1 are moved, and the others are zero ([`chunk::shuffle_fragment`]).
        fragment: bool,
    },
    /// A bitcast with a chunked side, through 32-bit words.
    WordBitcast {
        from: chunk::Shape,
        to: chunk::Shape,
    },
    /// `air.sincos` on floats of `shape` (chunked per chunk): the sine is the result, and the
    /// cosine is stored through the pointer operand by the access family's store `store`, selected
    /// for a store of the result's type through that pointer.
    SinCos {
        shape: chunk::Shape,
        store: Box<access::AccessStrategy>,
        /// `air.fast_sincos` on `float`: [`FloatFn::FastSin`] and [`FloatFn::FastCos`], which
        /// flush past the last evaluated quadrant as Metal's `fast::sincos` does.
        fast: bool,
    },
    /// `air.dot` of chunked vectors of `lanes` lanes of `elem` ([`chunk::dot`]).
    ChunkedDot {
        lanes: u16,
        elem: Form,
        contract: Contraction,
    },
    /// An operation with a wide integer operand or result on a profile without `shaderInt64`,
    /// on the two words of each value ([`wide`]).
    Wide(wide::WideOp),
    /// An `insertvalue` or `extractvalue` whose result is an aggregate that holds a pointer: no
    /// SPIR-V value ([`aggregate`]).
    PointerAggregate,
    /// An `extractvalue` of a texture or sampler handle with one static candidate: that
    /// candidate's constant `u32` handle index ([`aggregate`]).
    Handle(u32),
    /// An `extractvalue` from an aggregate that holds a pointer, whose member one `insertvalue`
    /// supplied: that inserted value ([`aggregate::forwarded`]).
    Forward(ValueId),
    /// An `extractvalue` of a pointer member that a null or undefined pointer constant, or an
    /// undefined aggregate, supplied: the null pointer in the result's representation
    /// ([`aggregate`]).
    Null(DemoteTy),
    /// An `extractvalue` of member `path` of the member-wise phi `of` ([`aggregate::phi_type`]): a
    /// composite extract from the phi's value.
    Member {
        of: ValueId,
        path: Box<[u32]>,
        /// The lanes of a `bool` vector member, held as `u32` lanes (`legal::as_member`): the
        /// extract is compared with 0.
        bools: Option<u8>,
    },
}

impl FamilyStrategy for ArithStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        use ArithStrategy as S;
        Cow::Borrowed(match self {
            S::Int { form, .. } | S::IntCompare { form, .. } if form.legalized() => {
                "arith.int.legalized"
            }
            S::Int { .. } | S::IntCompare { .. } => "arith.int",
            S::Bool { .. } | S::BoolCompare { .. } => "arith.bool",
            S::Float { form, .. } if matches!(form.float_fmt(), Some(legal::FloatFmt::Half)) => {
                "arith.float.half"
            }
            S::Float { form, .. } if form.is_narrow() => "arith.float.bfloat",
            S::F16 { .. } => ops::F16,
            S::Float { .. } | S::FNeg { .. } | S::FloatCompare { .. } => "arith.float",
            S::FastDivide { .. } => FAST_DIVIDE,
            S::NativeDivide { .. } => divide::NATIVE,
            S::Cast { .. } => "arith.cast",
            S::Bitcast {
                plan: BitcastPlan::Packed { .. },
                ..
            } => "arith.bitcast.packed",
            S::Bitcast {
                plan: BitcastPlan::Words { .. },
                ..
            } => "arith.bitcast.words",
            S::Bitcast {
                plan: BitcastPlan::Lanes,
                ..
            } => "arith.bitcast.lanes",
            S::Bitcast { .. } => "arith.bitcast",
            S::Select => "arith.select",
            S::Copy | S::Zero(_) | S::Undef(_) => "arith.copy",
            S::ExtractConst(_)
            | S::ExtractDynamic { .. }
            | S::InsertConst(_)
            | S::InsertDynamic { .. } => "arith.element",
            S::Shuffle(_) => "arith.shuffle",
            S::ExtractValue(_)
            | S::InsertValue(_)
            | S::ExtractBools { .. }
            | S::InsertBools { .. } => "arith.aggregate",
            S::IntFn { .. } => "arith.int_fn",
            S::FloatFn { form, .. } if form.is_narrow() => "arith.float_fn.narrow",
            S::FloatFn { f, .. } => match f.precision() {
                FloatPrecision::Exact => "arith.float_fn",
                FloatPrecision::DeviceArithmetic => "arith.float_fn.device_arithmetic",
                FloatPrecision::DeviceTranscendental => "arith.float_fn.device_transcendental",
            },
            S::Fma(f) => f.name(),
            S::FastFmod { .. } => FAST_FMOD,
            S::Convert {
                kind: ConvertKind::ToInt { .. },
                ..
            } => "arith.convert.saturating",
            S::Convert {
                kind: ConvertKind::FromFp8(_),
                ..
            } => "arith.convert.fp8",
            S::Convert { .. } => "arith.convert",
            S::Pack {
                dir: PackDir::Pack, ..
            } => "arith.pack",
            S::Pack { .. } => "arith.unpack",
            S::IndexExtension => "arith.index_extension",
            S::IndexWord(_) => "arith.index_word",
            S::SinCos { .. } => "arith.sincos",
            S::ChunkedDot { .. } => "arith.chunked.dot",
            S::Chunked { .. } => "arith.chunked",
            S::LaneExtract { .. } | S::LaneInsert { .. } => "arith.element.chunked",
            S::LaneShuffle { fragment: true, .. } => "arith.shuffle.fragment",
            S::LaneShuffle { .. } => "arith.shuffle.chunked",
            S::WordBitcast { .. } => "arith.bitcast.chunked",
            S::Wide(w) => w.name(),
            S::PointerAggregate => "arith.aggregate.pointer",
            S::Handle(_) => "arith.aggregate.handle",
            S::Forward(_) => "arith.aggregate.forward",
            S::Null(_) => "arith.aggregate.null",
            S::Member { .. } => "arith.aggregate.member",
        })
    }

    fn words(&self) -> u32 {
        // Generous per-use bounds: each instruction is at most 6 words plus its share of new
        // constants and types (a vector constant adds its lanes).
        use ArithStrategy as S;
        match self {
            S::IndexExtension | S::PointerAggregate => 0,
            S::Handle(_) | S::Forward(_) => 4,
            S::Member { path, .. } => {
                6u32.saturating_add(u32::try_from(path.len()).unwrap_or(u32::MAX))
            }
            // Two operand conversions (a shift pair or a select each) and the operation.
            S::IndexWord(_) => 32,
            S::Copy => 4,
            S::Zero(d) | S::Undef(d) | S::Null(d) => {
                8u32.saturating_add(8u32.saturating_mul(leaves(d)))
            }
            S::Int { form, .. } | S::IntCompare { form, .. } if form.legalized() => 64,
            S::Int { .. } | S::IntCompare { .. } => 16,
            S::Bool { .. } | S::BoolCompare { .. } | S::FloatCompare { .. } => 32,
            S::Float { form, .. } => 8 + narrow::round_words(*form),
            S::F16 { form, .. } => ops::f16_binary_words(*form),
            // The device's division.
            S::FastDivide { .. } => 16,
            S::NativeDivide { form } => divide::native_words(*form),
            // The bits' two conversions, the operation, and the mask's and its types' share.
            S::FNeg { .. } | S::Select => 8,
            S::Cast { from, to, .. } => 64 + conversion_words(*from, *to),
            S::Bitcast {
                plan: BitcastPlan::Packed { .. } | BitcastPlan::Words { .. } | BitcastPlan::Lanes,
                from,
                to,
            } => (32 * u32::from(from.lanes) + format_bits_words(*from))
                .saturating_add(32 * u32::from(to.lanes) + format_bits_words(*to))
                .saturating_add(32),
            S::Bitcast { .. } => 8,
            S::ExtractConst(_)
            | S::ExtractDynamic { .. }
            | S::InsertConst(_)
            | S::InsertDynamic { .. } => 8,
            S::Shuffle(c) | S::ExtractValue(c) | S::InsertValue(c) => {
                8u32.saturating_add(u32::try_from(c.len()).unwrap_or(u32::MAX))
            }
            // The access, the conversion, and its two (or one) splat constants.
            S::ExtractBools { path, .. } | S::InsertBools { path, .. } => {
                40u32.saturating_add(u32::try_from(path.len()).unwrap_or(u32::MAX))
            }
            S::IntFn { f, form } => f.words(form.bits()),
            S::FloatFn { f, form, .. } => f.words(*form),
            S::Fma(f) => f.words(),
            // The device's division, the truncation, its negation, and the fused multiply-add.
            S::FastFmod { fma } => 80 + fma.words(),
            S::Convert {
                kind: ConvertKind::FromFp8(_),
                ..
            } => 320,
            // The value (as `FromFp8`), then the saturating conversion from `f32`.
            S::Convert {
                kind: ConvertKind::Fp8ToInt { .. },
                from,
                to,
            } => 320 + 160 + conversion_words(from.with(Held::Float(FloatFmt::F32)), *to),
            S::Convert { from, to, .. } => 160 + conversion_words(*from, *to),
            S::Pack {
                dir,
                format,
                floats,
            } => pack::words(*format, *dir, *floats),
            S::SinCos { shape, store, fast } => {
                chunk::sin_cos_words(*shape, *fast).saturating_add(store.words())
            }
            S::ChunkedDot { lanes, elem, .. } => chunk::dot_words(*lanes, *elem),
            S::Chunked {
                lanes, parts, wide, ..
            } => {
                let per = 4u32.saturating_mul(wide.len() as u32 + 1);
                legal::chunk_lanes(u32::from(*lanes))
                    .into_iter()
                    .map(|k| {
                        parts
                            .iter()
                            .find(|(n, _)| *n == k)
                            .map_or(0, |(_, s)| s.words())
                            .saturating_add(per)
                    })
                    .fold(16u32, u32::saturating_add)
            }
            S::LaneExtract { src, at, .. } => {
                chunk::move_words(*src, chunk::LaneMove::Extract, at.is_none())
            }
            S::LaneInsert { dst, at, .. } => {
                chunk::move_words(*dst, chunk::LaneMove::Insert, at.is_none())
            }
            S::LaneShuffle {
                out,
                fragment: true,
                ..
            } => chunk::fragment_shuffle_words(*out),
            S::LaneShuffle { out, .. } => chunk::move_words(*out, chunk::LaneMove::Shuffle, false),
            S::WordBitcast { from, to } => chunk::bitcast_words(*from, *to),
            S::Wide(w) => w.words(),
        }
    }

    /// An aggregate that holds a pointer has no value.
    fn yields_value(&self) -> bool {
        !matches!(self, ArithStrategy::PointerAggregate)
    }
}

/// The extra words a conversion from `from` to `to` spends on narrow floats: a final rounding (by
/// integer arithmetic from an integer to `half`), a round-to-odd step from `double` or into
/// `bfloat` (twice as long from a 64-bit integer).
fn conversion_words(from: Form, to: Form) -> u32 {
    if !to.is_narrow() {
        return 0;
    }
    if !from.is_float() && to.float_fmt() == Some(FloatFmt::Half) {
        return narrow::round_integer_words(to);
    }
    let odd = if from.float_fmt() == Some(FloatFmt::F64) || !from.is_float() {
        if from.container() == 64 && !from.is_float() {
            192
        } else {
            96
        }
    } else {
        0
    };
    odd + narrow::round_words(to)
}

/// The words `f` spends moving to or from its format bits: a narrow float vector converts once as a
/// vector.
fn format_bits_words(f: Form) -> u32 {
    if f.is_narrow() {
        192
    } else {
        0
    }
}

/// The scalar leaves of `d` (for word bounds).
fn leaves(d: &DemoteTy) -> u32 {
    match d {
        DemoteTy::Vector { len, .. } => u32::from(*len),
        DemoteTy::Composite(parts) => parts.iter().map(leaves).fold(0u32, u32::saturating_add),
        _ => 1,
    }
}

/// The arith family's part of the per-function plan (`select::FamilyPlans`): the entry's
/// floating-point environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArithPlan {
    /// The float container widths whose operations must keep signed zeros, infinities, and NaNs:
    /// those LLVM requires it of (an operation without all of `nnan`, `ninf`, and `nsz`).
    pub preserve: BTreeSet<u8>,
    /// The widths whose `SignedZeroInfNanPreserve` execution mode the entry declares: those of
    /// `preserve` the profile supports.
    pub modes: BTreeSet<u8>,
    /// Whether 64-bit integers are held as two words (the profile lacks `shaderInt64`), so emission
    /// materializes their constants in that form ([`constant_in`]).
    pub wide: bool,
    /// The shared division routines the wide divisions call, each defined once by [`declare`].
    pub divides: BTreeSet<wide::Divide>,
    /// The value each forwarding `extractvalue` reads ([`aggregate::forwarded`]), and the
    /// member-wise phi each member extraction reads.
    pub forwards: BTreeMap<InstId, Value>,
    /// The composite representation of each member-wise phi: a `phi` of an aggregate that holds
    /// a pointer ([`aggregate::phi_type`]).
    pub phis: BTreeMap<InstId, DemoteTy>,
    /// How each member-wise phi's value on the edge from a block is assembled from its incoming
    /// aggregate's members ([`aggregate::incoming`]).
    pub incoming: BTreeMap<(InstId, BlockId), aggregate::Build>,
    /// The `half` additions, subtractions and products held at `float` precision, unrounded,
    /// between a chain's roundings on a device without `shaderFloat16` ([`half_chain`]).
    pub held_at_float: BTreeSet<InstId>,
    /// Whether a call takes Metal's precise `fmod` on `float` or `half` lanes ([`libm::uses`]),
    /// recorded at selection; [`finish`] decides [`ArithPlan::fmod`] from it.
    pub fmod_uses: bool,
    /// Whether the module defines precise `fmod`'s routine ([`declare`]): where a call takes it
    /// and the device's `OpFRem` is not Metal's `fmod` in the entry's mode
    /// ([`libm::Fun::device_answers`]), decided by [`finish`].
    pub fmod: bool,
    /// The mode of the device's own `float` functions where the profile's are Metal's
    /// (`FloatFunctions::Metal`): the mode Metal compiles this entry in ([`libm::module_mode`],
    /// decided by [`finish`]). `None` on any other profile.
    pub metal_functions: Option<libm::Mode>,
    /// Whether a selected fused multiply-add takes the device's `Fma` ([`fma::Route::Native`]),
    /// whose note [`finish`] records.
    pub native_fma: bool,
    /// How the module emits a multiply the device must round on its own, decided by [`finish`] from
    /// the same fact as [`ArithPlan::metal_functions`]: a device whose `float` functions are
    /// Metal's reaches Metal through SPIRV-Cross, which loses a `-0` product of a `NoContraction`
    /// `OpFMul`, so it takes `Fma(a, b, -0.0)`; any other device takes the `OpFMul`, which has no
    /// `-0.0` addend for it to answer.
    pub product: ProductForm,
    /// How the module clears the bits above a narrow integer's width in a 32-bit container
    /// ([`ops::mask`]), decided by [`finish`] from the profile's driver workarounds:
    /// `OpBitFieldUExtract` on a driver whose compiler drops the mask after a signed clamp
    /// (`DriverWorkaround::NarrowMaskAfterSignedClamp`), else the mask.
    pub mask: MaskForm,
    /// Whether a function converts the texels of a slot or attachment bound in an extended-range,
    /// padded or bfloat form ([`ArithPlan::converts`]) by the device's arithmetic
    /// ([`extended_range`]): [`finish`] then records `PrecisionNote::DevicePackConversion`.
    pub extended_range: bool,
    /// Whether a `half` fused multiply-add takes the device's 16-bit `Fma` ([`fma::Route::F16`]),
    /// recorded at selection: the module then requires `VkProperty::MetalHalfFma`.
    pub f16_fma: bool,
    /// Whether a `float` operation leaves its `f32` denormals to the device because the device
    /// reads them as Metal does ([`denormal::device_flushes`]), recorded at selection: the module
    /// then requires `VkProperty::MetalFloatDenormals`.
    pub device_denormals: bool,
    /// How the module rounds an `f32` to `half` ([`narrow::round`]), decided by [`finish`] from
    /// the profile ([`narrow::quantize_form`]): the device's conversion to 16 bits and its
    /// widening under the float controls that make it Metal's, else that conversion as it is
    /// where the profile has `shaderFloat16` (noted `NativeHalfConversion`), else the device's
    /// `OpQuantizeToF16` alone (noted `NativeHalfQuantize` on every such profile: a result below
    /// the smallest normal half may be a zero).
    pub quantize: QuantizeForm,
    /// Whether [`ArithPlan::quantize`] is the device's own rounding (native half rounding,
    /// [`narrow::native_rounding`]), decided with it by [`finish`].
    pub native_half: bool,
    /// Whether a `half`'s NaN bits are left to the device's conversion where the module rounds
    /// through the conversion under the float controls ([`QuantizeForm::Controlled`]): a `half`
    /// texel takes the rounding's NaN (`narrow::round_texel`). Decided by [`finish`], on every
    /// profile that rounds so.
    pub native_half_nans: bool,
    /// Whether Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` on `float` or `half`
    /// takes the device's `NMin`, `NMax` or `NClamp` measured to be Metal's
    /// (`DriverDeviation::MetalMinMax`), recorded at selection: the module then requires
    /// `VkProperty::MetalMinMax`. Every such call takes those instructions on every profile; one
    /// not measured so records `NativeMinMax` ([`native_min_max`]).
    pub device_min_max: bool,
    /// Whether a `half` `+`, `-` or `*` takes the device's 16-bit arithmetic
    /// ([`ArithStrategy::F16`]), measured to be Metal's (`DriverDeviation::MetalHalfArithmetic`),
    /// recorded at selection: the module then requires `VkProperty::MetalHalfArithmetic`.
    pub f16_arith: bool,
    /// The precision notes the plan's operations record unless their entry's quotient and
    /// functions are Metal's own (the profile's `float` functions are Metal's and Metal compiles
    /// the entry with fast math), recorded at selection ([`OpNotes::unless_metal_fast`]);
    /// [`finish`] records them where they are not.
    pub unless_metal_fast: BTreeSet<PrecisionNote>,
}

impl ArithPlan {
    /// Record that a function reads or writes the texels of a slot or attachment bound in
    /// `form`, which convert by the device's arithmetic ([`ArithPlan::extended_range`]).
    pub fn converts(&mut self, _form: crate::metal::format::ExtendedRange) {
        self.extended_range = true;
    }

    /// Record fused multiply-add `fma`, selected: the device's 16-bit `Fma`
    /// ([`ArithPlan::f16_fma`]), or the device's `Fma` ([`ArithPlan::native_fma`]).
    pub fn take_fma(&mut self, fma: fma::Fma) {
        self.f16_fma |= fma.route == fma::Route::F16;
        self.native_fma |= fma.route == fma::Route::Native;
    }
}

/// The words of the module-scope routines the plan's strategies call, each with the census name
/// of the strategies that call it: counted once per module, not per use.
pub fn routine_words(plan: &ArithPlan) -> impl Iterator<Item = (&'static str, u64)> + '_ {
    let fmod = plan
        .fmod
        .then_some((libm::FMOD_NAME, u64::from(libm::FMOD_WORDS)));
    plan.divides
        .iter()
        .map(|d| (wide::DIVIDE, u64::from(d.words())))
        .chain(fmod)
}

/// The census name of `air.fast_fmod` on `float` ([`ArithStrategy::FastFmod`]).
pub const FAST_FMOD: &str = "arith.fmod.fast";
/// The census name of an `arcp` division ([`ArithStrategy::FastDivide`]).
pub const FAST_DIVIDE: &str = "arith.fdiv.fast";

/// Complete the arith family's plan once every instruction and parameter is selected: declare
/// `SignedZeroInfNanPreserve` for each required width the profile supports, and note
/// `DeviceSignedZeroInfNan` for one it does not (the device's behavior is accepted and
/// reported); then, with the entry's execution modes known, decide whether precise `fmod` takes
/// its routine ([`ArithPlan::fmod`]: on a profile whose `float` functions are Metal's, not in an
/// entry Metal compiles with precise functions, [`libm::module_mode`]), record the notes of the
/// operations that are Metal's own in a fast-math entry only where it is not
/// ([`ArithPlan::unless_metal_fast`]), and choose the form of a multiply the device must round on
/// its own by the same fact ([`ArithPlan::product`]) and the form of a narrow integer's mask by the
/// driver's workarounds ([`ArithPlan::mask`]).
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    cx.families.arith.wide = legal::holds_wide(cx.profile);
    cx.families.arith.quantize = narrow::quantize_form(cx.profile);
    cx.families.arith.native_half = narrow::native_rounding(cx.families.arith.quantize).is_some();
    // Under the float controls a `half`'s NaN is the device's conversion's.
    cx.families.arith.native_half_nans = cx.families.arith.quantize == QuantizeForm::Controlled;
    // An extended-range, padded or bfloat texel's conversion is the device's arithmetic.
    if cx.families.arith.extended_range {
        cx.precision.insert(PrecisionNote::DevicePackConversion);
    }
    cx.families.arith.mask = if cx
        .profile
        .fields()
        .workarounds
        .contains(&DriverWorkaround::NarrowMaskAfterSignedClamp)
    {
        MaskForm::Extract
    } else {
        MaskForm::And
    };
    let fc = cx.profile.fields().float_controls;
    let widths: Vec<u8> = cx.families.arith.preserve.iter().copied().collect();
    for w in widths {
        let supported = match w {
            32 => fc.f32.signed_zero_inf_nan_preserve,
            64 => fc.f64.signed_zero_inf_nan_preserve,
            _ => false,
        };
        if supported {
            cx.families.arith.modes.insert(w);
        } else {
            cx.precision.insert(PrecisionNote::DeviceSignedZeroInfNan);
        }
    }
    let arith = &mut cx.families.arith;
    // The measured device fact, never the driver's name.
    arith.metal_functions = match cx.profile.fields().float_functions {
        FloatFunctions::Metal => Some(libm::module_mode(&arith.modes)),
        FloatFunctions::Native => None,
    };
    // Every family selected its fused multiply-adds by now (a cube position's at selection).
    if arith.native_fma {
        cx.precision.insert(PrecisionNote::NativeFma);
    }
    let metal = arith.metal_functions;
    // The same fact decides the multiply's form.
    arith.product = match metal {
        Some(_) => ProductForm::NegativeZeroFma,
        None => ProductForm::Multiply,
    };
    // Precise `fmod` takes its routine but where the device's `OpFRem` is Metal's `fmod`.
    arith.fmod = arith.fmod_uses && !metal.is_some_and(|m| libm::Fun::Fmod.device_answers(m));
    // An operation that is Metal's own in a fast-math entry records its notes only outside one.
    if !metal_fast(arith.metal_functions) {
        cx.precision.extend(arith.unless_metal_fast.iter().copied());
    }
    Ok(())
}

/// The arith family's module-scope objects: the shared wide division routines and precise
/// `fmod`'s routine (the `GLSL.std.450` import is the serializer's, and constants are
/// interned where they are used).
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The function of each of the plan's [`ArithPlan::divides`].
    pub divides: BTreeMap<wide::Divide, FuncId>,
    /// Precise `fmod`'s routine, when [`ArithPlan::fmod`].
    pub fmod: Option<FuncId>,
}

/// Declare the arith family's module-scope objects and entry execution modes: one function per
/// shared division routine the plan calls, and one `SignedZeroInfNanPreserve` per width the plan
/// selected.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    for w in &plan.families.arith.modes {
        modes.push(ExecMode::SignedZeroInfNanPreserve { width: *w });
    }
    let mut decl = Declared::default();
    for d in &plan.families.arith.divides {
        decl.divides.insert(*d, wide::define_divide(mb, *d)?);
    }
    if plan.families.arith.fmod {
        decl.fmod = Some(libm::define_fmod(mb)?);
    }
    Ok(decl)
}

/// Record that a float operation on form `f` with fast-math flags `fmf` needs signed zeros,
/// infinities, and NaNs preserved at `f`'s container width unless it carries all of `nnan`,
/// `ninf`, and `nsz` (LLVM semantics; a Vulkan device may otherwise ignore them). The simd
/// family's float calls record here too: the simdgroup and quad sums, prefix sums, minima, and
/// maxima, and the matrix and AGX multiply-adds, whose steps are `f32` `Fma`s.
pub(crate) fn fp_env(cx: &mut SelectCx<'_>, fmf: FastMath, f: Form) {
    let relaxed = FastMath::NNAN.union(FastMath::NINF).union(FastMath::NSZ);
    if f.is_float() && !fmf.contains(relaxed) {
        cx.families.arith.preserve.insert(f.container());
    }
}

// ---- Selection ----

/// The AIR type of `v`.
fn ty_of(cx: &SelectCx<'_>, v: Value) -> Result<AirTypeId, Error> {
    cx.value_ty(v)
        .ok_or_else(|| cx.invariant("an arith operand without a type"))
}

/// The form of `v` on the profile.
fn form(cx: &SelectCx<'_>, v: Value) -> Result<Form, Error> {
    form_on(cx.entry, cx.profile, ty_of(cx, v)?)
}

/// The form of the instruction's result.
fn result_form(cx: &SelectCx<'_>, inst: &Inst) -> Result<Form, Error> {
    form_on(cx.entry, cx.profile, inst.ty)
}

/// Every operand must be a value emission produces in its lowered representation: a constant
/// this family materializes, or a value of a type lowered on the profile (instruction results
/// are selected on their own; parameters by their family).
fn check_operands(cx: &SelectCx<'_>, operands: &[Value]) -> Result<(), Error> {
    for v in operands {
        if matches!(v, Value::Const(_)) && cx.constant(*v).is_none() {
            return Err(not_lowered());
        }
        cx.lowered_repr(*v, ty_of(cx, *v)?)?;
    }
    Ok(())
}

/// The value operands of a non-call instruction, in order.
fn operands_of(kind: &InstKind) -> Vec<Value> {
    let mut out = Vec::new();
    kind.for_each_value(|v| out.push(v));
    out
}

/// The integer constant `v`, if it is one.
fn const_int(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c)?.value {
        Const::Int(b) => Some(b),
        _ => None,
    }
}

/// The lane count of vector value `v`.
fn lanes_of(cx: &SelectCx<'_>, v: Value) -> Result<u32, Error> {
    let t = ty_of(cx, v)?;
    match cx.entry.module.types().get(t) {
        Some(Type::Vector { len, .. }) => Ok(*len),
        _ => Err(cx.invariant("a vector operation on a non-vector")),
    }
}

/// The SPIR-V contraction of an operation with fast-math flags `fmf`: the AIR's own permission,
/// nothing of Metal's choices. An undecorated SPIR-V operation lets the device both fuse it and
/// regroup it with its neighbours (Vulkan assumes `AllowContract` and `AllowReassoc` for it;
/// `NoContraction` forbids both). LLVM grants the two separately, `contract` and `reassoc`, so an
/// operation that carries both stays undecorated, and the device's compiler groups and fuses it as
/// it chooses; any other is `NoContraction`, which keeps it apart and in the AIR's order: an
/// operation without `contract` is never fused, and one without `reassoc` (every operation of a
/// precise module) never regrouped.
fn contraction(fmf: FastMath) -> Contraction {
    if fmf.contains(FastMath::CONTRACT.union(FastMath::REASSOC)) {
        Contraction::Allowed
    } else {
        Contraction::Forbidden
    }
}

/// Note `PrecisionNote::FastMathRewrites` when float arithmetic `op` with flags `fmf` on operands
/// `a` and `b` lets a compiler rewrite it with the operation defining one of them, as LLVM's flags
/// allow: `op` and the operand are additions, subtractions, or multiplies that both carry
/// `reassoc` (a regrouping), or `op` is an addition or subtraction and the operand a multiply, both
/// carrying `contract`, the multiply's `fpext` to `float` looked through (a fusion). Where both
/// carry `contract` and `reassoc` the module leaves them undecorated and the device's compiler
/// takes its own rewrite; elsewhere it keeps them apart ([`contraction`]); Metal's compiler takes
/// its own in either case. A negation between them is looked through: it is exact, and a compiler
/// folds it into either operation. So are lane selections, insertions and extractions: a compiler
/// splits vectors into lanes before it regroups. Arithmetic that chains only through memory is not
/// seen here.
fn note_rewrites(cx: &mut SelectCx<'_>, (op, fmf): (AirBin, FastMath), a: Value, b: Value) {
    if !matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul) {
        return;
    }
    let sum = matches!(op, AirBin::FAdd | AirBin::FSub);
    let rewritable = |def: AirBin, def_fmf: FastMath, widened: bool| {
        let regrouped =
            !widened && fmf.contains(FastMath::REASSOC) && def_fmf.contains(FastMath::REASSOC);
        let fused = sum
            && def == AirBin::FMul
            && fmf.contains(FastMath::CONTRACT)
            && def_fmf.contains(FastMath::CONTRACT);
        regrouped || fused
    };
    if [a, b]
        .into_iter()
        .any(|v| float_arith_def(cx, v, LOOK_THROUGH, false, &rewritable))
    {
        cx.precision.insert(PrecisionNote::FastMathRewrites);
    }
}

/// How many negations, widenings, lane selections, insertions and extractions [`float_arith_def`]
/// looks through: a bound on its walk.
const LOOK_THROUGH: u32 = 8;

/// Whether a float addition, subtraction, or multiply defining `v`, through negations, `fpext`s
/// and lane selections, insertions and extractions, at most `depth` of them, satisfies `f` of its
/// operator, its fast-math flags, and whether the walk crossed a widening (`widened`, or one of its
/// own): either source of a selection of two vectors' lanes or of an insertion.
fn float_arith_def(
    cx: &SelectCx<'_>,
    v: Value,
    depth: u32,
    widened: bool,
    f: &dyn Fn(AirBin, FastMath, bool) -> bool,
) -> bool {
    let Value::Inst(id) = v else {
        return false;
    };
    let Some(inst) = cx.body.insts.get(id) else {
        return false;
    };
    if let InstKind::BinOp {
        op: op @ (AirBin::FAdd | AirBin::FSub | AirBin::FMul),
        fmf,
        ..
    } = &inst.kind
    {
        return f(*op, *fmf, widened);
    }
    let Some(depth) = depth.checked_sub(1) else {
        return false;
    };
    match &inst.kind {
        InstKind::FNeg { v: negated, .. } => float_arith_def(cx, *negated, depth, widened, f),
        InstKind::Cast {
            op: CastOp::FPExt,
            v: narrow,
        } => float_arith_def(cx, *narrow, depth, true, f),
        InstKind::ShuffleVector { a, b, .. } => {
            float_arith_def(cx, *a, depth, widened, f) || float_arith_def(cx, *b, depth, widened, f)
        }
        InstKind::InsertElement { vec, elt, .. } => {
            float_arith_def(cx, *elt, depth, widened, f)
                || float_arith_def(cx, *vec, depth, widened, f)
        }
        InstKind::ExtractElement { vec, .. } => float_arith_def(cx, *vec, depth, widened, f),
        _ => false,
    }
}

fn note(cx: &mut SelectCx<'_>, p: FloatPrecision) {
    cx.precision.extend(precision_note(p));
}

/// The note that records precision class `p`, if it records one.
fn precision_note(p: FloatPrecision) -> Option<PrecisionNote> {
    match p {
        FloatPrecision::Exact => None,
        FloatPrecision::DeviceArithmetic => Some(PrecisionNote::DeviceArithmetic),
        FloatPrecision::DeviceTranscendental => Some(PrecisionNote::DeviceTranscendentals),
    }
}

/// Record that `float` operations on form `f` are left to the device's denormal handling (with
/// `denormals`) and its NaN encoding (with `nans`): Metal flushes `f32` denormal operands and
/// results and answers `0x7fc00000` for a NaN result, which no Vulkan device is required to do.
/// Other formats record nothing: a `half` is held widened (never an `f32` denormal) and rounded
/// by this translation, and Metal's `double` and `bfloat` environments are not measured.
fn device_float(cx: &mut SelectCx<'_>, f: Form, denormals: bool, nans: bool) {
    cx.precision.extend(device_float_notes(f, denormals, nans));
}

/// The notes [`device_float`] records.
fn device_float_notes(f: Form, denormals: bool, nans: bool) -> impl Iterator<Item = PrecisionNote> {
    let single = f.float_fmt() == Some(FloatFmt::F32);
    [
        (denormals, PrecisionNote::DeviceDenormals),
        (nans, PrecisionNote::DeviceNanEncoding),
    ]
    .into_iter()
    .filter_map(move |(left, note)| (single && left).then_some(note))
}

/// The precision notes a float operation's lowering records for the operation itself, by where they
/// hold: `always` in every entry, and `unless_metal_fast` in every entry but one whose quotient and
/// functions are Metal's own (the profile's `float` functions are Metal's, and Metal compiles the
/// entry with fast math), where the operation is Metal's own and
/// its notes name no difference from Metal. Selection records the first ([`record`]) and holds the
/// second in [`ArithPlan::unless_metal_fast`] until [`finish`] knows the entry's math mode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpNotes {
    /// The notes the operation records in every entry.
    pub always: BTreeSet<PrecisionNote>,
    /// The notes it records unless the entry is Metal's fast one.
    pub unless_metal_fast: BTreeSet<PrecisionNote>,
}

impl OpNotes {
    /// `notes`, held in every entry.
    fn everywhere(notes: BTreeSet<PrecisionNote>) -> OpNotes {
        OpNotes {
            always: notes,
            unless_metal_fast: BTreeSet::new(),
        }
    }

    /// `notes`, held in every entry but Metal's fast one where `own`, else in every entry.
    fn unless_own(notes: BTreeSet<PrecisionNote>, own: bool) -> OpNotes {
        if own {
            OpNotes {
                always: BTreeSet::new(),
                unless_metal_fast: notes,
            }
        } else {
            OpNotes::everywhere(notes)
        }
    }

    /// Both sets of `other` added to this one's.
    fn with(mut self, other: OpNotes) -> OpNotes {
        self.always.extend(other.always);
        self.unless_metal_fast.extend(other.unless_metal_fast);
        self
    }

    /// The notes the operation records in an entry whose `float` functions are Metal's in mode
    /// `metal` (`None` on a profile whose functions are the device's), as [`finish`] decides
    /// them.
    pub fn in_entry(self, metal: Option<libm::Mode>) -> BTreeSet<PrecisionNote> {
        let mut notes = self.always;
        if !metal_fast(metal) {
            notes.extend(self.unless_metal_fast);
        }
        notes
    }
}

/// Whether an entry whose `float` functions are Metal's in mode `metal` (`None`: the device's) is
/// one Metal compiles with fast math: its `arcp` quotient and its functions are then Metal's own
/// ([`OpNotes`]).
fn metal_fast(metal: Option<libm::Mode>) -> bool {
    metal == Some(libm::Mode::Fast)
}

/// Record `notes`, an operation's: those that hold in every entry now, the others once [`finish`]
/// knows whether the entry is Metal's fast one.
fn record(cx: &mut SelectCx<'_>, notes: OpNotes) {
    cx.precision.extend(notes.always);
    cx.families
        .arith
        .unless_metal_fast
        .extend(notes.unless_metal_fast);
}

/// The precision notes the lowering of the float binary operator `op` with fast-math flags `fmf`
/// on operands of format `fmt` records for the operator itself on profile `p`: a division or
/// remainder is the device's arithmetic, and a `float` operator leaves its denormals to the device
/// where it is not measured to read them as Metal does (Metal flushes `f32` denormals,
/// [`leave_denormals`]); a `float` remainder also leaves the NaN encoding to it, where `+ - *`
/// answer Metal's `0x7fc00000`. Metal's precise division ([`divide::precise`]: an `fdiv` on
/// `float` without `arcp`, or on `half`) is the device's `OpFDiv` (`NativeDivision`), whose
/// denormals no device is measured to read as Metal does. Metal's fast division (an `fdiv` on
/// `float` with `arcp`) is the device's `OpFDiv(x, y)` too; where the profile's `float` functions
/// are Metal's, the quotient of an entry Metal compiles with fast math is Metal's own fast
/// division, and the notes hold only outside one. LLVM's `frem` on `float` without every
/// fast-math flag is the device's `OpFRem`, `x - y · trunc(x / y)` at its precision, as Metal
/// computes it of its precise quotient. Beside these, an addition, subtraction, or multiply whose
/// flags let the device fuse or regroup it with an operand's definition records
/// `FastMathRewrites`, which the operands decide (`note_rewrites`). `None` for an integer
/// operator.
///
/// [`select`] records exactly these for such an operator (a `half` result's rounding is noted from
/// the emitted module); validation reads them to tell which operations the product computes as a
/// near-equivalent of Metal's, as [`PrecisionNote::of_float_operator`].
pub fn float_binop_notes(
    op: AirBin,
    fmt: FloatFmt,
    fmf: FastMath,
    p: &DeviceProfile,
) -> Option<OpNotes> {
    if !op.is_float() {
        return None;
    }
    // The device leaves the NaN Metal answers to the device where the profile does.
    let native_nan = |answers: bool| {
        (answers && nan::encoding(p) == NanEncoding::Native)
            .then_some(PrecisionNote::NativeNanEncoding)
    };
    if divide::precise(op, fmt, fmf) {
        return Some(OpNotes::everywhere(
            [Some(PrecisionNote::NativeDivision), native_nan(true)]
                .into_iter()
                .flatten()
                .chain(device_float_notes(scalar_float(fmt), true, false))
                .collect(),
        ));
    }
    // A `float` remainder with every flag is Metal's `fast::fmod`, with that function's notes.
    if op == AirBin::FRem && fmt == FloatFmt::F32 && every_flag(fmf) {
        let f = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 1,
        };
        return float_call(FloatFn::Fmod, true, fmf, f, FloatFacts::on(p)).map(|l| l.notes);
    }
    // A fast `float` division is the device's `OpFDiv`, its `f32` denormals the device's: no
    // device is measured to read a quotient's as Metal does.
    if op == AirBin::FDiv && fmt == FloatFmt::F32 {
        let mut notes = BTreeSet::from([
            PrecisionNote::DeviceArithmetic,
            PrecisionNote::NativeDivision,
            PrecisionNote::DeviceDenormals,
        ]);
        notes.extend(native_nan(true));
        return Some(OpNotes::unless_own(
            notes,
            p.fields().float_functions == FloatFunctions::Metal,
        ));
    }
    // A `float` `+ - *` and remainder answer the device's NaN, noted where the profile does not
    // state it is Metal's ([`nan`]), and leave their denormals to the device: `+ - *` but where it
    // is measured to read them as Metal does ([`leave_denormals`]), a remainder on every device.
    // Metal's `half` and `bfloat` remainders are not measured.
    let answer = binop_answer(op, fmt);
    let nans = answer.is_none();
    let denormals = op == AirBin::FRem || !denormal::device_flushes(p);
    let mut notes: BTreeSet<_> = device_float_notes(scalar_float(fmt), denormals, nans).collect();
    if matches!(op, AirBin::FDiv | AirBin::FRem) {
        notes.insert(PrecisionNote::DeviceArithmetic);
    }
    notes.extend(native_nan(answer.is_some()));
    Some(OpNotes::everywhere(notes))
}

/// The NaN Metal answers for the float binary operator `op` on format `fmt`, which its lowering
/// leaves to the device ([`with_nan`]): `float` `frem`'s canonical NaN beside
/// [`nan::binary_answer`]'s.
fn binop_answer(op: AirBin, fmt: FloatFmt) -> Option<nan::NanAnswer> {
    match (op, fmt) {
        (AirBin::FRem, FloatFmt::F32) => Some(nan::NanAnswer::Canonical),
        _ => nan::binary_answer(op, fmt),
    }
}

/// Strategy `s`, whose result's NaN Metal answers as `answer` where the lowering does not answer
/// it itself, with that NaN left to the device ([`nan`]): noted `NativeNanEncoding` where the
/// device is not measured to answer it ([`nan::NanAnswer::device_answers`]).
fn with_nan(
    cx: &mut SelectCx<'_>,
    s: ArithStrategy,
    answer: Option<nan::NanAnswer>,
) -> ArithStrategy {
    note_native_nan(cx, answer);
    s
}

/// Record `NativeNanEncoding` where Metal answers a NaN (`answer`) the device is not measured to
/// answer ([`nan::NanAnswer::device_answers`]).
fn note_native_nan(cx: &mut SelectCx<'_>, answer: Option<nan::NanAnswer>) {
    if answer.is_some_and(|a| !a.device_answers(nan::encoding(cx.profile))) {
        cx.precision.insert(PrecisionNote::NativeNanEncoding);
    }
}

/// Leave an operation's `f32` denormals (on `float` lanes of form `f`) to the device: where the
/// device is measured to read them as Metal does and the operation is one the measured deviation
/// covers (`covered`, [`denormal::device_flushes`]), the module requires the property; elsewhere
/// they are the device's own, noted `DeviceDenormals` (as the operation's notes record it).
/// Another format leaves nothing: a `half` is held widened, never an `f32` denormal, and Metal's
/// `bfloat` and `double` environments are not measured.
pub(super) fn leave_denormals(cx: &mut SelectCx<'_>, f: Form, covered: denormal::Covered) {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return;
    }
    if covered == denormal::Covered::Deviation && denormal::device_flushes(cx.profile) {
        cx.families.arith.device_denormals = true;
    } else {
        cx.precision.insert(PrecisionNote::DeviceDenormals);
    }
}

/// A scalar of float format `fmt`.
fn scalar_float(fmt: FloatFmt) -> Form {
    Form {
        held: Held::Float(fmt),
        lanes: 1,
    }
}

/// Select the strategy of instruction `inst` of this family, metering in `m` the walks the
/// access family takes for the store of `air.sincos`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    if let InstKind::Cast {
        op: CastOp::ZExt | CastOp::SExt,
        ..
    } = &inst.kind
    {
        if access::index::is_index_extension(cx, at) {
            return Ok(ArithStrategy::IndexExtension);
        }
    }
    if let InstKind::BinOp { .. } = &inst.kind {
        if access::index::is_index_word(cx, at) {
            return access::index::select_word(cx, inst).map(ArithStrategy::IndexWord);
        }
    }
    if aggregate::involved(cx, inst) {
        return aggregate::select(cx, at, inst);
    }
    if chunk::involved(cx, inst) {
        return chunk::select(cx, at, inst, m);
    }
    if wide::involved(cx, inst) {
        return wide::select(cx, at, inst);
    }
    select_in(cx, at, inst, None, m)
}

/// The form of `v`; with `chunk`, a chunked vector's form is that of one chunk of `chunk` lanes.
fn form_in(cx: &SelectCx<'_>, v: Value, chunk: Option<u8>) -> Result<Form, Error> {
    let t = ty_of(cx, v)?;
    chunk_form(cx, t, chunk)
}

/// [`form_in`] of the instruction's result.
fn result_form_in(cx: &SelectCx<'_>, inst: &Inst, chunk: Option<u8>) -> Result<Form, Error> {
    chunk_form(cx, inst.ty, chunk)
}

fn chunk_form(cx: &SelectCx<'_>, t: AirTypeId, chunk: Option<u8>) -> Result<Form, Error> {
    if let (Some(k), Some((held, _))) = (chunk, legal::chunked(cx.entry.module.types(), t)) {
        let f = Form { held, lanes: k };
        legal::check_scalar(cx.profile, f.scalar())?;
        return Ok(f);
    }
    form_on(cx.entry, cx.profile, t)
}

/// Select `inst`; with `chunk`, for one chunk of `chunk` lanes of a lane-wise operation on
/// chunked vectors (its chunked operands and result take the chunk's form).
fn select_in(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    chunk: Option<u8>,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    use ArithStrategy as S;
    if let InstKind::Call(call) = &inst.kind {
        return select_call(cx, at, inst, call, chunk, m);
    }
    let mut ops = operands_of(&inst.kind);
    // A constant element index is a literal of the instruction, never a materialized value (an
    // `i64` index needs no `shaderInt64`).
    if let InstKind::ExtractElement { idx, .. } | InstKind::InsertElement { idx, .. } = &inst.kind {
        if matches!(idx, Value::Const(_)) {
            ops.retain(|v| v != idx);
        }
    }
    check_operands(cx, &ops)?;
    // The result is a value of a lowered type (or the operation is refused).
    let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    Ok(match &inst.kind {
        InstKind::BinOp { op, a, b, fmf, .. } => {
            let f = form_in(cx, *a, chunk)?;
            match f.held {
                Held::Bool if !op.is_float() => S::Bool { op: *op, form: f },
                Held::Int { .. } if !op.is_float() => S::Int { op: *op, form: f },
                // LLVM's `frem` on `float` is Metal's (measured on the M3 Max): with every
                // fast-math flag its `fast::fmod`, as `air.fast_fmod` is lowered; otherwise
                // `x - y * trunc(x / y)` of its precise quotient, which the device's `OpFRem` is at
                // the device's precision, below.
                Held::Float(FloatFmt::F32) if *op == AirBin::FRem && every_flag(*fmf) => {
                    return select_float_fn(cx, (FloatFn::Fmod, true), *fmf, f);
                }
                // Metal divides `float` without `arcp` correctly rounded, a denormal operand
                // read as a zero and a denormal quotient flushed (a probe cases 914cc4d2, 711cc190,
                // and c10cba13, every lane), and `half` under every flag set rounded once from the
                // exact quotient (4c363561, 54f0fe8e, every lane): the device's quotient, its
                // divisor kept in Vulkan's range ([`divide::native`]), its NaN the device's (Metal
                // answers its canonical one). No device is measured to read a quotient's denormals
                // as Metal does.
                Held::Float(fmt) if divide::precise(*op, fmt, *fmf) => {
                    let notes = float_binop_notes(*op, fmt, *fmf, cx.profile);
                    record(cx, notes.unwrap_or_default());
                    note_rewrites(cx, (*op, *fmf), *a, *b);
                    fp_env(cx, *fmf, f);
                    leave_denormals(cx, f, denormal::Covered::Never);
                    let s = S::NativeDivide { form: f };
                    with_nan(cx, s, Some(nan::NanAnswer::Canonical))
                }
                Held::Float(fmt) if op.is_float() => {
                    let notes = float_binop_notes(*op, fmt, *fmf, cx.profile);
                    record(cx, notes.unwrap_or_default());
                    note_rewrites(cx, (*op, *fmf), *a, *b);
                    fp_env(cx, *fmf, f);
                    let contract = contraction(*fmf);
                    // Metal divides `float` under `arcp` as `x` times its fast reciprocal of `y`
                    // (a probe cases 914cc4d2 and 711cc190, every flag set measured, every lane),
                    // its reciprocal flushing past `2^126` and at a denormal `y`: the device's
                    // `OpFDiv(x, y)`, Metal's own where the device's functions are Metal's in an
                    // entry Metal compiles with fast math. Its NaN is the device's, noted where it
                    // is not measured to be Metal's canonical one; its `f32` denormals the
                    // device's ([`float_binop_notes`]).
                    if *op == AirBin::FDiv
                        && fmf.contains(FastMath::ARCP)
                        && f.float_fmt() == Some(FloatFmt::F32)
                    {
                        let s = S::FastDivide { form: f, contract };
                        with_nan(cx, s, Some(nan::NanAnswer::Canonical))
                    } else if let Some(basis) = f16_arithmetic(cx.profile, *op, f) {
                        // The device's own `half` arithmetic, Metal's in the forms that keep it
                        // apart, which the AIR's flags allow whatever they are. Only the measured
                        // basis is a property the module requires; the float controls are modes
                        // its entry declares.
                        cx.families.arith.f16_arith |= basis == F16Basis::Measured;
                        let s = S::F16 { op: *op, form: f };
                        with_nan(cx, s, nan::binary_answer(*op, fmt))
                    } else {
                        // A `half` chain's operation held at `float` precision: its operands,
                        // `half`s held widened or held operations' `float`s, operated on as
                        // `float`s and left unrounded ([`half_chain`]), noted.
                        let form = match cx.families.arith.held_at_float.contains(&at) {
                            true => {
                                cx.precision.insert(PrecisionNote::FastMathRewrites);
                                f.with(Held::Float(FloatFmt::F32))
                            }
                            false => f,
                        };
                        let s = S::Float {
                            op: *op,
                            form,
                            contract,
                        };
                        // Metal reads an `f32` denormal operand of `+ - *` as a zero and flushes a
                        // denormal result: the device's reading, on the deviation. A remainder's
                        // are the device's on every device ([`float_binop_notes`]).
                        if matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul) {
                            leave_denormals(cx, f, denormal::Covered::Deviation);
                        }
                        with_nan(cx, s, binop_answer(*op, fmt))
                    }
                }
                _ => return Err(cx.invariant("a binary operator on mismatched operands")),
            }
        }
        InstKind::FNeg { v, fmf } => {
            let f = form_in(cx, *v, chunk)?;
            // The device's sign flip: it makes no NaN. Metal's `float` `fneg` flips the sign bit
            // alone, keeping a NaN's payload and a denormal, whatever the flags; Vulkan lets
            // `OpFNegate` flush a denormal (`DeviceDenormals`), and states no NaN's bits, noted
            // `NativeNanEncoding` where Metal answers one ([`nan::fneg_answer`]).
            device_float(cx, f, true, false);
            fp_env(cx, *fmf, f);
            let nan = f.float_fmt().and_then(nan::fneg_answer);
            with_nan(cx, S::FNeg { form: f }, nan)
        }
        InstKind::ICmp { pred, a, .. } => {
            let f = form_in(cx, *a, chunk)?;
            if f.is_bool() {
                S::BoolCompare { pred: *pred }
            } else {
                S::IntCompare {
                    pred: *pred,
                    form: f,
                }
            }
        }
        InstKind::FCmp { pred, a, fmf, .. } => {
            let f = form_in(cx, *a, chunk)?;
            // Metal compares an `f32` denormal as a zero; the result is a `bool`, never a NaN.
            fp_env(cx, *fmf, f);
            leave_denormals(cx, f, denormal::Covered::Deviation);
            S::FloatCompare {
                pred: *pred,
                form: f,
            }
        }
        InstKind::Cast { op, v } => select_cast(cx, *op, *v, inst, chunk)?,
        InstKind::Select { .. } => S::Select,
        InstKind::Freeze(v) => match cx.constant(*v) {
            Some(LoweredConst::Undef(_)) => S::Zero(result),
            _ => S::Copy,
        },
        // A constant index out of range (or `undef`) gives poison.
        InstKind::ExtractElement { vec, idx } => match (idx, const_int(cx, *idx)) {
            (Value::Const(_), Some(i)) if i < u64::from(lanes_of(cx, *vec)?) => {
                S::ExtractConst(u32::try_from(i).map_err(|_| not_lowered())?)
            }
            (Value::Const(_), _) => S::Undef(result),
            _ => S::ExtractDynamic {
                index: form_in(cx, *idx, None)?,
            },
        },
        InstKind::InsertElement { vec, idx, .. } => match (idx, const_int(cx, *idx)) {
            (Value::Const(_), Some(i)) if i < u64::from(lanes_of(cx, *vec)?) => {
                S::InsertConst(u32::try_from(i).map_err(|_| not_lowered())?)
            }
            (Value::Const(_), _) => S::Undef(result),
            _ => S::InsertDynamic {
                index: form_in(cx, *idx, None)?,
            },
        },
        InstKind::ShuffleVector { a, b, mask } => {
            let total = lanes_of(cx, *a)?.saturating_add(lanes_of(cx, *b)?);
            let mut comps = Vec::with_capacity(mask.len());
            for m in mask.iter() {
                // A poison lane may be any value: take component 0.
                let c = m.unwrap_or(0);
                if c >= total {
                    return Err(cx.invariant("a shuffle index beyond its operands"));
                }
                comps.push(c);
            }
            S::Shuffle(comps.into())
        }
        // A `bool` vector member is held as `u32` lanes.
        InstKind::ExtractValue { indices, .. } => {
            match legal::bool_lanes(cx.entry.module.types(), inst.ty) {
                Some(lanes) => S::ExtractBools {
                    path: indices.clone(),
                    lanes,
                },
                None => S::ExtractValue(indices.clone()),
            }
        }
        InstKind::InsertValue { val, indices, .. } => {
            match legal::bool_lanes(cx.entry.module.types(), ty_of(cx, *val)?) {
                Some(lanes) => S::InsertBools {
                    path: indices.clone(),
                    lanes,
                },
                None => S::InsertValue(indices.clone()),
            }
        }
        _ => return Err(not_lowered()),
    })
}

/// Select the cast `op` of `v`, instruction `inst`; with `chunk`, for one chunk (as
/// [`select_in`]).
fn select_cast(
    cx: &mut SelectCx<'_>,
    op: CastOp,
    v: Value,
    inst: &Inst,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    let from = form_in(cx, v, chunk)?;
    let to = result_form_in(cx, inst, chunk)?;
    let kind = match op {
        CastOp::Trunc => CastKind::Trunc,
        CastOp::ZExt => CastKind::ZExt,
        CastOp::SExt => CastKind::SExt,
        CastOp::FPTrunc | CastOp::FPExt => {
            // Between `float` and `double` a denormal (and a NaN's payload) is the device's.
            if from.bits() == 64 || to.bits() == 64 {
                device_float(cx, from, true, true);
                device_float(cx, to, true, true);
            }
            let s = ArithStrategy::Cast {
                kind: CastKind::FpResize,
                from,
                to,
            };
            // A `float` narrowed to `half` answers Metal's canonical `half` NaN.
            let answer = match (from.float_fmt(), to.float_fmt()) {
                (Some(a), Some(b)) => nan::cast_answer(op, a, b),
                _ => None,
            };
            return Ok(with_nan(cx, s, answer));
        }
        CastOp::FPToUI => CastKind::FpToUi,
        CastOp::FPToSI => CastKind::FpToSi,
        CastOp::UIToFP => CastKind::UiToFp,
        CastOp::SIToFP => CastKind::SiToFp,
        CastOp::Bitcast if chunk.is_some() => {
            return Err(cx.invariant("a bitcast selected per chunk"))
        }
        CastOp::Bitcast => {
            let plan = ops::bitcast_plan(from, to).ok_or_else(not_lowered)?;
            if let BitcastPlan::Packed { total } = plan {
                // The packed integer is a legal type on this profile.
                legal::check_scalar(cx.profile, ops::packed_form(total).scalar())?;
            }
            return Ok(ArithStrategy::Bitcast { plan, from, to });
        }
        CastOp::PtrToInt | CastOp::IntToPtr | CastOp::AddrSpaceCast => {
            return Err(cx.invariant("a pointer cast routed to arith"))
        }
    };
    Ok(ArithStrategy::Cast { kind, from, to })
}

fn sign(s: AirSign) -> Sign {
    match s {
        AirSign::Signed => Sign::Signed,
        AirSign::Unsigned => Sign::Unsigned,
    }
}

/// The arith function of a decoded `air` math intrinsic, or a refusal.
enum Fun {
    Int(IntFn),
    Float(FloatFn),
    Copy,
    /// `sincos`, which also stores (see [`select_sincos`]).
    SinCos,
}

fn air_math(f: MathFn) -> Result<Fun, Error> {
    use FloatFn as F;
    use MathFn as M;
    Ok(match f {
        M::Fabs => Fun::Float(F::Abs),
        M::Sqrt => Fun::Float(F::Sqrt),
        M::Rsqrt => Fun::Float(F::Rsqrt),
        M::Exp => Fun::Float(F::Exp),
        M::Exp2 => Fun::Float(F::Exp2),
        M::Exp10 => Fun::Float(F::Exp10),
        M::Log => Fun::Float(F::Log),
        M::Log2 => Fun::Float(F::Log2),
        M::Log10 => Fun::Float(F::Log10),
        M::Sin => Fun::Float(F::Sin),
        M::Cos => Fun::Float(F::Cos),
        M::Tan => Fun::Float(F::Tan),
        M::Asin => Fun::Float(F::Asin),
        M::Acos => Fun::Float(F::Acos),
        M::Atan => Fun::Float(F::Atan),
        M::Sinh => Fun::Float(F::Sinh),
        M::Cosh => Fun::Float(F::Cosh),
        M::Tanh => Fun::Float(F::Tanh),
        M::Asinh => Fun::Float(F::Asinh),
        M::Acosh => Fun::Float(F::Acosh),
        M::Atanh => Fun::Float(F::Atanh),
        M::SinPi => Fun::Float(F::SinPi),
        M::CosPi => Fun::Float(F::CosPi),
        M::TanPi => Fun::Float(F::TanPi),
        M::Floor => Fun::Float(F::Floor),
        M::Ceil => Fun::Float(F::Ceil),
        M::Round => Fun::Float(F::Round),
        M::Rint => Fun::Float(F::Rint),
        M::Trunc => Fun::Float(F::Trunc),
        M::Fract => Fun::Float(F::Fract),
        M::FSign => Fun::Float(F::Sign),
        M::Saturate => Fun::Float(F::Saturate),
        M::FMin | M::Min => Fun::Float(F::Min),
        M::FMax | M::Max => Fun::Float(F::Max),
        M::Powr => Fun::Float(F::Powr),
        M::Fmod => Fun::Float(F::Fmod),
        M::Atan2 => Fun::Float(F::Atan2),
        M::Fma => Fun::Float(F::Fma),
        M::Mix => Fun::Float(F::Mix),
        M::Clamp => Fun::Float(F::Clamp),
        M::FMin3 => Fun::Float(F::Min3),
        M::FMax3 => Fun::Float(F::Max3),
        M::FMedian3 => Fun::Float(F::Median3),
        M::Ldexp => Fun::Float(F::Ldexp),
        M::Dot => Fun::Float(F::Dot),
        M::Abs(AirSign::Signed) => Fun::Int(IntFn::Abs),
        M::Abs(AirSign::Unsigned) => Fun::Copy,
        M::AbsDiff(s) => Fun::Int(IntFn::AbsDiff(sign(s))),
        M::IMin(s) => Fun::Int(IntFn::Min(sign(s))),
        M::IMax(s) => Fun::Int(IntFn::Max(sign(s))),
        M::IClamp(s) => Fun::Int(IntFn::Clamp(sign(s))),
        M::IMin3(s) => Fun::Int(IntFn::Min3(sign(s))),
        M::IMax3(s) => Fun::Int(IntFn::Max3(sign(s))),
        M::MulHi(s) => Fun::Int(IntFn::MulHi(sign(s))),
        M::MadHi(s) => Fun::Int(IntFn::MadHi(sign(s))),
        M::MadSat(s) => Fun::Int(IntFn::MadSat(sign(s))),
        M::Pow => Fun::Float(F::Pow),
        M::AddSat(s) => Fun::Int(IntFn::AddSat(sign(s))),
        M::SubSat(s) => Fun::Int(IntFn::SubSat(sign(s))),
        M::HAdd(s) => Fun::Int(IntFn::HAdd(sign(s))),
        M::RHAdd(s) => Fun::Int(IntFn::RHAdd(sign(s))),
        M::ExtractBits(s) => Fun::Int(IntFn::ExtractBits(sign(s))),
        M::InsertBits(_) => Fun::Int(IntFn::InsertBits),
        M::Popcount => Fun::Int(IntFn::Popcount),
        M::ReverseBits => Fun::Int(IntFn::ReverseBits),
        M::Bswap => Fun::Int(IntFn::Bswap),
        M::Rotate => Fun::Int(IntFn::Rotate),
        M::Clz { .. } => Fun::Int(IntFn::Clz),
        M::Ctz { .. } => Fun::Int(IntFn::Ctz),
        M::SinCos => Fun::SinCos,
        // Canonicalization replaces every runtime routine's call (`air::canon::routine`).
        M::Frexp | M::Ilogb => {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "a runtime routine's call survived canonicalization",
            )))
        }
    })
}

fn llvm_math(op: &LlvmIntrinsic) -> Result<Fun, Error> {
    Ok(match *op {
        LlvmIntrinsic::Float { op, .. } => Fun::Float(match op {
            LlvmFloatOp::FMulAdd => FloatFn::MulAdd,
            LlvmFloatOp::Fma => FloatFn::Fma,
            LlvmFloatOp::FAbs => FloatFn::Abs,
            LlvmFloatOp::MinNum => FloatFn::MinNum,
            LlvmFloatOp::MaxNum => FloatFn::MaxNum,
            // Apple's math library defines precise `air.exp2`, `air.log2`, and `air.sqrt` as
            // these calls, so they are those functions.
            LlvmFloatOp::Exp2 => FloatFn::Exp2,
            LlvmFloatOp::Log2 => FloatFn::Log2,
            LlvmFloatOp::Sqrt => FloatFn::Sqrt,
            LlvmFloatOp::CopySign => FloatFn::CopySign,
        }),
        LlvmIntrinsic::Int { op, .. } => Fun::Int(match op {
            LlvmIntOp::SMax => IntFn::Max(Sign::Signed),
            LlvmIntOp::SMin => IntFn::Min(Sign::Signed),
            LlvmIntOp::UMax => IntFn::Max(Sign::Unsigned),
            LlvmIntOp::UMin => IntFn::Min(Sign::Unsigned),
            LlvmIntOp::Fshl => IntFn::Fshl,
            LlvmIntOp::Fshr => IntFn::Fshr,
            LlvmIntOp::Bswap => IntFn::Bswap,
            LlvmIntOp::Ctpop => IntFn::Popcount,
            LlvmIntOp::UAddSat => IntFn::AddSat(Sign::Unsigned),
            LlvmIntOp::USubSat => IntFn::SubSat(Sign::Unsigned),
            LlvmIntOp::SAddSat => IntFn::AddSat(Sign::Signed),
            LlvmIntOp::SSubSat => IntFn::SubSat(Sign::Signed),
        }),
        LlvmIntrinsic::Abs { .. } => Fun::Int(IntFn::Abs),
        LlvmIntrinsic::Cttz { .. } => Fun::Int(IntFn::Ctz),
        LlvmIntrinsic::Ctlz { .. } => Fun::Int(IntFn::Clz),
        LlvmIntrinsic::Memset { .. }
        | LlvmIntrinsic::Memcpy { .. }
        | LlvmIntrinsic::Memmove { .. }
        | LlvmIntrinsic::Agx(_) => return Err(not_lowered()),
    })
}

/// The float function a call of `op` on form `f` computes, given whether its `air` symbol is the
/// `fast_` variant. The `fast_` variants on `float` that answer differently from the precise ones
/// are their own functions; the fast-math accuracy table is single precision, so a narrow `fast_`
/// call keeps the precise function:
///
/// - `air.fast_tanh`, `air.fast_sinh`, `air.fast_cosh`, `air.fast_atanh`, `air.fast_sin`,
///   `air.fast_cos` and `air.fast_tan` are the device's instructions, which a device whose
///   functions are Metal's answers as Metal's fast functions only in a fast-math module (and its
///   `Tanh` as Metal's precise `tanh` in every module): their own variants for their notes
///   ([`FloatFn::FastTanh`], [`FloatFn::FastSinh`], [`FloatFn::FastCosh`],
///   [`FloatFn::FastAtanh`], [`FloatFn::FastSin`], [`FloatFn::FastCos`], [`FloatFn::FastTan`]);
/// - `air.fast_asinh` with every flag is MSL Table 8.2's formula, the canonical NaN at `-inf`
///   ([`FloatFn::FastAsinh`]);
/// - `air.fast_acosh` is MSL Table 8.2's formula of Metal's fast `log` and `sqrt`
///   ([`FloatFn::FastAcosh`]), the device's `Acosh`;
/// - `air.fast_fract` is not bounded below one, and with every flag is the canonical NaN at an
///   infinity ([`FloatFn::FastFract`]);
/// - `air.fast_fmedian3` answers `b` for two NaN first operands ([`FloatFn::FastMedian3`]);
/// - `air.fast_pow`'s magnitude is `exp2(y · log2|x|)` ([`FloatFn::FastPow`]);
/// - `air.fast_atan2` is `atan(y / x)` turned by `±π`, NaN at `0 / 0` and `inf / inf`
///   ([`FloatFn::FastAtan2`]);
/// - `air.fast_asin`, `air.fast_acos`, and `air.fast_atan` are Metal's fast functions, which a
///   device that is Metal's own answers only in a fast-math module and which answer `fast::asin`
///   of a zero and of a denormal as `+0` and `fast::atan` of one as its signed zero
///   ([`FloatFn::FastAsin`], [`FloatFn::FastAcos`], [`FloatFn::FastAtan`]);
/// - `air.fast_rsqrt` is Metal's fast `rsqrt` ([`FloatFn::FastRsqrt`]), apart from the correctly
///   rounded precise one;
/// - `air.fast_sqrt` is the device's `Sqrt` ([`FloatFn::FastSqrt`]);
/// - `air.fast_exp`, `air.fast_exp10`, `air.fast_log`, `air.fast_log10`, and `air.fast_powr` are
///   MSL Table 8.2's formulas `exp2(x · log2 e)`, `exp2(x · log2 10)`, `log2(x) · ln 2`,
///   `log2(x) · log10 2`, and `exp2(y · log2 x)` ([`FloatFn::FastExp`], [`FloatFn::FastExp10`],
///   [`FloatFn::FastLog`], [`FloatFn::FastLog10`], [`FloatFn::FastPowr`]); `air.fast_pow`'s
///   magnitude is the last of `|x|`;
/// - `air.fast_log2` is Metal's `log2`, the precise one ([`FloatFn::FastLog2`]), which the fast
///   formulas' `log2` is too;
/// - `air.fast_sinpi`, `air.fast_cospi`, and `air.fast_tanpi` are Metal's precise functions,
///   which a device whose functions are Metal's answers by its `Sin` and `Cos` within Apple's
///   fast bound ([`FloatFn::FastSinPi`], [`FloatFn::FastCosPi`], [`FloatFn::FastTanPi`]).
///
/// `air.fast_fmod` on `float` is selected beside this, as [`ArithStrategy::FastFmod`].
///
/// `llvm.minnum` and `llvm.maxnum` on `float`, Metal's fast-math form of `a < b ? a : b` and
/// `a > b ? a : b` (with every flag), are Metal's `fmin` and `fmax` ([`FloatFn::Min`],
/// [`FloatFn::Max`]), with any flags; on the other formats they stay LLVM's (`FloatFn::MinNum`).
///
/// A call's fast-math flags `fmf` do not change the function: Metal's `sign`, for one, answers
/// `+0` for `-0` under fast math too (A1 case f9a12252). The exceptions measured are
/// `fast_atan2`'s quotient beyond a divisor of `2^126`, which a call without every flag takes
/// from Metal's reciprocal, flushed there (probe cases 6802ab13 and c11eb8cb), and the device's
/// reciprocal here (`FloatFn::FastAtan2`), and `fast_asinh` at
/// infinities, which only a call with every flag answers as a NaN (A1 case aed3faaa without
/// flags, our probe with them). The other flag combinations are not measured; they take the
/// answer of a call without flags. A function the device has an instruction for answers its
/// special operands as the instruction does in every combination (`fract`, `sign`).
fn refine_float(op: FloatFn, fast: bool, fmf: FastMath, f: Form) -> FloatFn {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return op;
    }
    // `a < b ? a : b` and `a > b ? a : b` under fast math are `llvm.minnum` and `llvm.maxnum`
    // with every flag, which Metal answers as `fmin` and `fmax` (`nan_minmax_fast`, every lane);
    // without a flag Metal answers them the same, word for word (those two calls of
    // `nan_minmax_fast` with their flags removed, every pair of 20 patterns, NaNs and denormals
    // among them), and so with some flags (`nnan` on `llvm.minnum`, `nsz arcp` on `llvm.maxnum`).
    match op {
        FloatFn::MinNum => return FloatFn::Min,
        FloatFn::MaxNum => return FloatFn::Max,
        _ => {}
    }
    if !fast {
        return op;
    }
    match op {
        FloatFn::Tanh => FloatFn::FastTanh,
        FloatFn::Asinh if every_flag(fmf) => FloatFn::FastAsinh,
        FloatFn::Sinh => FloatFn::FastSinh,
        FloatFn::Cosh => FloatFn::FastCosh,
        FloatFn::Acosh => FloatFn::FastAcosh,
        FloatFn::Atanh => FloatFn::FastAtanh,
        FloatFn::Exp10 => FloatFn::FastExp10,
        FloatFn::Log10 => FloatFn::FastLog10,
        FloatFn::Fract => FloatFn::FastFract,
        FloatFn::Median3 => FloatFn::FastMedian3,
        FloatFn::Pow => FloatFn::FastPow,
        FloatFn::Sin => FloatFn::FastSin,
        FloatFn::Cos => FloatFn::FastCos,
        FloatFn::Tan => FloatFn::FastTan,
        FloatFn::Asin => FloatFn::FastAsin,
        FloatFn::Acos => FloatFn::FastAcos,
        FloatFn::Atan => FloatFn::FastAtan,
        FloatFn::Sqrt => FloatFn::FastSqrt,
        FloatFn::Rsqrt => FloatFn::FastRsqrt,
        FloatFn::Exp => FloatFn::FastExp,
        FloatFn::Log => FloatFn::FastLog,
        FloatFn::Log2 => FloatFn::FastLog2,
        FloatFn::Powr => FloatFn::FastPowr,
        FloatFn::Atan2 => FloatFn::FastAtan2 {
            by_reciprocal: !every_flag(fmf),
        },
        FloatFn::SinPi => FloatFn::FastSinPi,
        FloatFn::CosPi => FloatFn::FastCosPi,
        FloatFn::TanPi => FloatFn::FastTanPi,
        op => op,
    }
}

/// Whether `fmf` holds every fast-math flag, as Metal's compiler gives each `float` operation
/// under fast math: the state the `fast_` functions are measured in.
fn every_flag(fmf: FastMath) -> bool {
    let every = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag));
    fmf.contains(every)
}

fn select_call(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &crate::air::ir::Call,
    chunk: Option<u8>,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    use crate::air::intrinsics::Precision;
    use crate::air::ir::Callee;
    let mut fast = false;
    let fun = match &call.callee {
        Callee::Air {
            op: AirIntrinsic::Math(m),
            ..
        } => {
            fast = m.precision == Precision::Fast;
            air_math(m.f)?
        }
        Callee::Air {
            op: AirIntrinsic::Convert(c),
            ..
        } => return select_convert(cx, inst, call, c, chunk),
        Callee::Air {
            op: AirIntrinsic::Pack(p),
            ..
        } if chunk.is_none() => return select_pack(cx, inst, call, p),
        Callee::Llvm { op, .. } => llvm_math(op)?,
        _ => return Err(not_lowered()),
    };
    let arity = match &fun {
        Fun::Int(f) => f.arity(),
        Fun::Float(f) => f.arity(),
        Fun::Copy => 1,
        // A chunked `sincos` is selected whole ([`chunk::select`]), never per chunk.
        Fun::SinCos if chunk.is_some() => return Err(cx.invariant("air.sincos selected per chunk")),
        Fun::SinCos => return select_sincos(cx, at, inst, call, m),
    };
    let args = call
        .args
        .get(..arity)
        .ok_or_else(|| cx.invariant("too few intrinsic operands"))?;
    check_operands(cx, args)?;
    lowered_type(cx.entry, cx.profile, inst.ty)?;
    let first = *args
        .first()
        .ok_or_else(|| cx.invariant("an intrinsic without operands"))?;
    let f = form_in(cx, first, chunk)?;
    Ok(match fun {
        Fun::Copy => ArithStrategy::Copy,
        Fun::SinCos => return Err(cx.invariant("sincos selected as a plain function")),
        Fun::Int(op) => {
            let ok = match (f.held, op) {
                (Held::Int { bits, .. }, IntFn::Bswap) => bits % 16 == 0,
                (Held::Int { .. }, _) => true,
                _ => false,
            };
            if !ok {
                return Err(not_lowered());
            }
            ArithStrategy::IntFn { f: op, form: f }
        }
        Fun::Float(op) => select_float_fn(cx, (op, fast), call.fmf, f)?,
    })
}

/// The strategy of float function `op` (`fast` its `fast_` variant) with fast-math flags `fmf`
/// on operands of form `f`: a math call's, or LLVM's `frem`'s, which is `fmod`.
fn select_float_fn(
    cx: &mut SelectCx<'_>,
    (op, fast): (FloatFn, bool),
    fmf: FastMath,
    f: Form,
) -> Result<ArithStrategy, Error> {
    let facts = FloatFacts::on(cx.profile);
    let lowered = float_call(op, fast, fmf, f, facts).ok_or_else(not_lowered)?;
    record(cx, lowered.notes);
    fp_env(cx, fmf, f);
    if let FloatLowering::Fn(op) = lowered.how {
        let fmod = libm::uses(op, f).contains(&libm::Fun::Fmod);
        let arith = &mut cx.families.arith;
        arith.fmod_uses |= fmod;
        arith.device_min_max |= device_min_max(cx.profile, op, f);
    }
    if let FloatLowering::Fma(fma) | FloatLowering::FastFmod(fma) = lowered.how {
        cx.families.arith.take_fma(fma);
    }
    let s = match lowered.how {
        FloatLowering::Fma(fma) => ArithStrategy::Fma(fma),
        FloatLowering::FastFmod(fma) => ArithStrategy::FastFmod { fma },
        FloatLowering::Fn(op) => ArithStrategy::FloatFn {
            f: op,
            form: f,
            contract: contraction(fmf),
        },
    };
    // The device reads the call's `f32` denormals as Metal does where the table says so, on the
    // deviation; elsewhere they are its own, as the call's notes say.
    if let Some(covered) = call_coverage(lowered.how, facts) {
        leave_denormals(cx, f, covered);
    }
    Ok(with_nan(cx, s, lowered.nan))
}

/// Where the device reads the `f32` denormals of a float function call lowered as `how` on a
/// profile with float facts `facts` as Metal does ([`denormal::coverage`]): a fused multiply-add
/// and `fast::fmod` by the device's own `float` instruction as `fma` (`OpFmaKHR`, or the device's
/// `Fma`), and the function's own row. Where the profile's
/// functions are Metal's, Metal's fast `sinpi`, `cospi` and `tanpi` are the device's form
/// ([`FloatFn::FastSinPi`], `math::pi_scaled`), whose first multiply and difference read the
/// operand, on the deviation. `None` for a lowering the table does not list.
fn call_coverage(how: FloatLowering, facts: FloatFacts) -> Option<denormal::Covered> {
    match how {
        FloatLowering::Fma(fma) | FloatLowering::FastFmod(fma) if fma.route.device_float() => {
            denormal::coverage(FloatFn::Fma)
        }
        FloatLowering::Fn(FloatFn::FastSinPi | FloatFn::FastCosPi | FloatFn::FastTanPi)
            if facts.metal_functions =>
        {
            Some(denormal::Covered::Deviation)
        }
        FloatLowering::Fn(op) => denormal::coverage(op),
        FloatLowering::Fma(_) | FloatLowering::FastFmod(_) => None,
    }
}

/// Whether the device leaves a call lowered as `how` (on lanes of form `f`) its `f32` denormals
/// on a profile with float facts `facts`, as [`leave_denormals`] notes them: where its reading is
/// covered ([`call_coverage`]) but not by the profile's deviation (`facts.device_denormals`).
fn coverage_left(how: FloatLowering, f: Form, facts: FloatFacts) -> bool {
    f.float_fmt() == Some(FloatFmt::F32)
        && call_coverage(how, facts)
            .is_some_and(|c| !(c == denormal::Covered::Deviation && facts.device_denormals))
}

/// Whether Metal's minimum and maximum function `op` on lanes of form `f` takes the device's
/// `NMin`, `NMax` or `NClamp` measured to be Metal's: `fmin`, `fmax`, `min3`, `max3`, `clamp` and
/// `saturate` ([`min_max_family`]) on `float` and `half` ([`math::metal_picks`]) where profile `p`
/// takes `DriverDeviation::MetalMinMax`.
fn device_min_max(p: &DeviceProfile, op: FloatFn, f: Form) -> bool {
    use crate::error::DriverDeviation;
    min_max_family(op)
        && math::metal_picks(f.float_fmt())
        && p.takes_driver_deviation(DriverDeviation::MetalMinMax)
}

/// Whether float function `op` is one of Metal's minimum and maximum functions, which pick one of
/// their operands: `fmin`, `fmax`, `min3`, `max3`, `clamp`, and `saturate`, which is Metal's
/// `clamp(x, +0, 1)` at every operand (a NaN, either zero and an `f32` denormal answer `+0`, as
/// `fmax` of them and `+0` answers `+0`).
fn min_max_family(op: FloatFn) -> bool {
    matches!(
        op,
        FloatFn::Min
            | FloatFn::Max
            | FloatFn::Clamp
            | FloatFn::Min3
            | FloatFn::Max3
            | FloatFn::Saturate
    )
}

/// Whether Metal's minimum and maximum function `op` on lanes of form `f` takes the device's
/// `NMin`, `NMax` or `NClamp` as they are, not measured to be Metal's, noted `NativeMinMax`: one of
/// [`device_min_max`]'s functions on a profile that does not take `DriverDeviation::MetalMinMax`
/// (`native_min_max`, [`FloatFacts::native_min_max`]), `saturate` on every other float format,
/// which no device is measured to answer as Metal's `clamp(x, +0, 1)`, and `median3` and
/// `fast::median3` on every device: `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and
/// `NMax` (`math::median3`), Metal's answer but where two operands are NaN, whose pick is the
/// device's (Metal's `fast::median3` answers `b` for two NaN first operands), and an `f32`
/// denormal, which Metal compares as a zero.
fn native_min_max(op: FloatFn, f: Form, native_min_max: bool) -> bool {
    let picks = math::metal_picks(f.float_fmt());
    (min_max_family(op) && ((picks && native_min_max) || (op == FloatFn::Saturate && !picks)))
        || matches!(op, FloatFn::Median3 | FloatFn::FastMedian3)
}

/// Why the device's 16-bit `+ - *` answers as Metal's `half` arithmetic ([`f16_arithmetic`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum F16Basis {
    /// Measured (`DriverDeviation::MetalHalfArithmetic`): the module requires
    /// `VkProperty::MetalHalfArithmetic`.
    Measured,
    /// The 16-bit float controls the module's `half` roundings take ([`QuantizeForm::Controlled`],
    /// [`narrow::controlled_modes`]), which its entry then declares ([`narrow::computes_in_16_bits`]):
    /// no fact.
    Controlled,
}

/// Whether the float binary operator `op` on lanes of form `f` takes the device's 16-bit
/// arithmetic ([`ArithStrategy::F16`]), and on what basis: a `half` `fadd`, `fsub` or `fmul`, on
/// one to four lanes (a chunk's own form), where profile `p` has `shaderFloat16` and takes
/// `DriverDeviation::MetalHalfArithmetic` ([`F16Basis::Measured`]) or rounds to `half` under the
/// 16-bit float controls ([`F16Basis::Controlled`]). Under those controls the instruction is Metal's
/// by the Vulkan spec ([`ops::f16_binary`]). A `half` `fdiv` is the device's `float` quotient of
/// the widened operands, rounded to `half`: Metal's fast and relaxed compiles answer the device's
/// 16-bit quotient of widened operands as a `float` one, it has no form that keeps it apart, and
/// Vulkan's precision table does not make a 16-bit quotient correctly rounded. A `half` `frem` is
/// not measured.
fn f16_arithmetic(p: &DeviceProfile, op: AirBin, f: Form) -> Option<F16Basis> {
    use crate::error::DriverDeviation;
    let applies = matches!(op, AirBin::FAdd | AirBin::FSub | AirBin::FMul)
        && f.float_fmt() == Some(FloatFmt::Half)
        && (1..=4).contains(&f.lanes)
        && p.fields().leaf.float16;
    if !applies {
        None
    } else if p.takes_driver_deviation(DriverDeviation::MetalHalfArithmetic) {
        Some(F16Basis::Measured)
    } else if narrow::quantize_form(p) == QuantizeForm::Controlled {
        Some(F16Basis::Controlled)
    } else {
        None
    }
}

/// How [`select_call`] lowers a float function call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatLowering {
    /// A fused multiply-add.
    Fma(fma::Fma),
    /// `air.fast_fmod` on `float`, its quotient the device's.
    FastFmod(fma::Fma),
    /// The float function.
    Fn(FloatFn),
}

/// A float function call's lowering, the precision notes it records, and the NaN Metal answers
/// that its lowering does not.
struct FloatCall {
    how: FloatLowering,
    notes: OpNotes,
    nan: Option<nan::NanAnswer>,
}

/// The facts of a profile a float function call's lowering and notes read.
#[derive(Clone, Copy, Debug)]
struct FloatFacts {
    /// The fused multiply-add routes ([`fma::routes`]).
    routes: fma::Routes,
    /// Whether the profile's `float` functions are Metal's (`FloatFunctions::Metal`).
    metal_functions: bool,
    /// Whether the device reads `f32` denormals as Metal does ([`denormal::device_flushes`]).
    device_denormals: bool,
    /// Which NaN the device's arithmetic answers ([`nan::encoding`]).
    nans: NanEncoding,
    /// Whether Metal's minimum and maximum functions are the device's not measured to be Metal's:
    /// the profile does not take `DriverDeviation::MetalMinMax` ([`native_min_max`]).
    native_min_max: bool,
}

impl FloatFacts {
    /// Profile `p`'s.
    fn on(p: &DeviceProfile) -> FloatFacts {
        FloatFacts {
            routes: fma::routes(p),
            metal_functions: p.fields().float_functions == FloatFunctions::Metal,
            device_denormals: denormal::device_flushes(p),
            nans: nan::encoding(p),
            native_min_max: !p.takes_driver_deviation(crate::error::DriverDeviation::MetalMinMax),
        }
    }
}

/// How much of a call of float function `op` (with the `fast_` prefix where `fast`) on form `f`
/// is Metal's own in an entry Metal compiles with fast math, on a device whose `float` functions
/// are Metal's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalOwn {
    /// None of it: the device's instruction is not Metal's function in a fast-math module, the
    /// device answers it by a near-equivalent ([`libm::Fun::device_near_equivalent`]), or no
    /// `float` or `half` function is called.
    No,
    /// Its function and its division: a `fast_` call of no listed function (`fast::rsqrt`,
    /// `fast::sqrt`), whose device instruction the module compiles as Metal's fast function, and
    /// whose division is Metal's fast one.
    Function,
    /// Wholly: every function the call takes the device answers in a fast-math module as Metal's
    /// own function ([`libm::Fun::device_answers`]), whose every word, denormals and NaNs among
    /// them, is Metal's (`DriverDeviation::MetalFloatFunctions`); its division is Metal's fast
    /// one.
    Wholly,
}

impl MetalOwn {
    /// The call of `op` (`fast_` where `fast`) on form `f`.
    fn of(op: FloatFn, fast: bool, f: Form) -> MetalOwn {
        if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
            return MetalOwn::No;
        }
        let uses = libm::uses(op, f);
        let own = |r: &libm::Fun| r.device_answers(libm::Mode::Fast) && !r.device_near_equivalent();
        if !uses.iter().all(own) {
            MetalOwn::No
        } else if !uses.is_empty() {
            MetalOwn::Wholly
        } else if fast && f.float_fmt() == Some(FloatFmt::F32) {
            MetalOwn::Function
        } else {
            MetalOwn::No
        }
    }
}

/// The notes of a call of float function `op` (with the `fast_` prefix where `fast`) on form `f`
/// lowered as the function itself, on a profile whose `float` functions are Metal's where
/// `metal_functions`: where they are not, the device's precision where the call takes the device's
/// instruction for one of Metal's functions, `fabs`, `sign`, `fract` and `fast::sqrt` included
/// ([`FloatFn::native_instruction`], `NativeTranscendentals`), else the function's precision
/// ([`FloatFn::precision`]); `round`'s zero, C99's, on every device
/// ([`FloatFn::composed_on_every_device`], `NativeTranscendentals`);
/// and its division's (`DeviceArithmetic` for [`FloatFn::divides`], and `NativeDivision` where the
/// functions are not Metal's), and its `f32` denormals' and NaNs' where they are left to the
/// device (`denormals`, `nans`, [`device_float_notes`]).
///
/// A call of precise `fmod` by its routine, whose answer is the one right value, Metal's on every
/// device ([`libm::Fun::correctly_rounded`]), records no precision note, but on `half`, whose zero
/// remainder of a nonzero dividend is the `float` remainder's sign, where Metal answers `+0`, in
/// every entry on every device ([`FloatFn::half_zero_of_float`], `NativeTranscendentals`). Where the profile's
/// `float` functions are Metal's, a call
/// that is Metal's own in a fast-math entry ([`MetalOwn`]) records its function's and its
/// division's notes only outside one, and, where it is wholly Metal's own, its denormals' and
/// NaNs' too ([`OpNotes`]). Of its fast functions, `fast::atan2` is Apple's formula
/// `atan(y / x)` turned by `±π` (MSL Table 8.2) of Metal's own fast `atan` and fast reciprocal,
/// and answers Metal's word on every lane of case f6524207 through MoltenVK but two, whose
/// divisors are past `2^126`, where Table 8.2 states no accuracy for the quotient: within Apple's
/// bound, which the user's fast-math sign-off accepts ("fast-math within Apple's bound",
/// 2026-10-03), so it is Metal's own too. Its fast `sinpi`, `cospi`, and `tanpi`, the device's
/// near-equivalents, keep their notes, as does every call in an entry that keeps NaNs.
fn function_notes(
    op: FloatFn,
    fast: bool,
    f: Form,
    metal_functions: bool,
    (denormals, nans): (bool, bool),
) -> OpNotes {
    let uses = libm::uses(op, f);
    // Off Metal's own functions, the device's instruction for one of Metal's functions, and the
    // device's `fast::sqrt`: its precision is Vulkan's, and its `f32` denormals the device's.
    let native = !metal_functions;
    let native_uses = native && (uses.iter().any(|r| r.native()) || op.native_instruction());
    let exact = !native_uses && !uses.is_empty() && uses.iter().all(|r| r.correctly_rounded());
    let mut function = BTreeSet::new();
    if native_uses {
        function.insert(PrecisionNote::NativeTranscendentals);
    } else if !exact {
        function.extend(precision_note(op.precision()));
    }
    // `round`'s composition on every device: C99's `-0`, where Metal answers `+0`.
    if op.composed_on_every_device() {
        function.insert(PrecisionNote::NativeTranscendentals);
    }
    if op.divides() {
        function.insert(PrecisionNote::DeviceArithmetic);
        if native {
            function.insert(PrecisionNote::NativeDivision);
        }
    }
    let denormals = denormals || native_uses;
    let environment: BTreeSet<_> = device_float_notes(f, denormals, nans).collect();
    let own = if metal_functions {
        MetalOwn::of(op, fast, f)
    } else {
        MetalOwn::No
    };
    // A `half` `fmod`'s zero, the `float` remainder's sign in either math mode.
    let half_zero = BTreeSet::from_iter(
        op.half_zero_of_float(f)
            .then_some(PrecisionNote::NativeTranscendentals),
    );
    OpNotes::unless_own(function, own != MetalOwn::No)
        .with(OpNotes::unless_own(environment, own == MetalOwn::Wholly))
        .with(OpNotes::everywhere(half_zero))
}

/// The note of fused multiply-add `fma`: `NativeFma` where it takes the device's `Fma`
/// ([`fma::Route::Native`]), which [`finish`] records for every such route a family takes.
fn native_fma(fma: fma::Fma) -> OpNotes {
    match fma.route {
        fma::Route::Native => OpNotes::everywhere(BTreeSet::from([PrecisionNote::NativeFma])),
        _ => OpNotes::default(),
    }
}

/// The lowering of float function `op` called with the `fast_` prefix if `fast` and with
/// fast-math flags `fmf`, on operands of form `f`, on a profile with float facts `facts`
/// ([`FloatFacts`]); `None` where the form refuses the function (it is not a float, or a `double`
/// the function does not take). A Metal NaN the call answers is the device's, noted
/// `NativeNanEncoding` where the device is not measured to answer it (as [`with_nan`] records it),
/// and Metal's minimum and maximum functions are the device's, noted `NativeMinMax` where they are
/// not measured to be Metal's ([`native_min_max`]); the `f32` denormals the device reads where the
/// table covers the call but the profile lacks the deviation are noted `DeviceDenormals`
/// ([`coverage_left`]).
fn float_call(
    op: FloatFn,
    fast: bool,
    fmf: FastMath,
    f: Form,
    facts: FloatFacts,
) -> Option<FloatCall> {
    let mut call = float_call_of(op, fast, fmf, f, facts)?;
    let mut native = BTreeSet::new();
    if call.nan.is_some_and(|a| !a.device_answers(facts.nans)) {
        native.insert(PrecisionNote::NativeNanEncoding);
    }
    if let FloatLowering::Fn(op) = call.how {
        if native_min_max(op, f, facts.native_min_max) {
            native.insert(PrecisionNote::NativeMinMax);
        }
    }
    // The `f32` denormals the device reads where it is not measured to read them as Metal does
    // ([`leave_denormals`]).
    if coverage_left(call.how, f, facts) {
        native.insert(PrecisionNote::DeviceDenormals);
    }
    call.notes = call.notes.with(OpNotes::everywhere(native));
    Some(call)
}

/// [`float_call`] before the device's NaN and minimum and maximum notes.
fn float_call_of(
    op: FloatFn,
    fast: bool,
    fmf: FastMath,
    f: Form,
    facts: FloatFacts,
) -> Option<FloatCall> {
    if !f.is_float() || (f.bits() == 64 && !op.allows_f64()) {
        return None;
    }
    let FloatFacts {
        routes,
        metal_functions,
        ..
    } = facts;
    let op = refine_float(op, fast, fmf, f);
    // `float` and `half` fused multiply-adds take the profile's route ([`fma`]); a `bfloat` one
    // corrects its tie, and a `double` one is `GLSL.std.450 Fma` at the device's precision.
    // `llvm.fmuladd` on `float` and `half` is the fused one too: LLVM lets the implementation
    // choose, and Metal fuses it (a probe case f7850ba7, every lane), so it needs no
    // `FastMathRewrites`.
    if matches!(op, FloatFn::Fma | FloatFn::MulAdd) {
        if let Some(fma) = fma::Fma::by(routes, f) {
            // The device's `float` instruction answers the device's NaN, noted; a `half` result's
            // rounding answers the device's `half` NaN on every route, noted where Metal's is
            // measured.
            let khr = fma.route.device_float();
            let nan = match f.float_fmt() {
                Some(FloatFmt::Half) => {
                    nan::function_answer(op, fast, every_flag(fmf), FloatFmt::Half)
                }
                _ => khr.then_some(nan::NanAnswer::Canonical),
            };
            return Some(FloatCall {
                how: FloatLowering::Fma(fma),
                notes: native_fma(fma),
                nan,
            });
        }
    }
    // `mix` and `dot` are the device's `FMix` and `OpDot` on every device. Metal's `mix(x, y, t)`
    // on `float` and `half` is `fma(t, y, fma(-t, x, x))` and its `dot` the products' fused chain,
    // with and without fast-math flags; the device's instructions may round or sum otherwise (a
    // tree, an unfused sum, or, under fast-math flags, `b * (a + c)` for
    // `dot(float2(a, b), float2(b, c))`), noted `NativeMix` and `NativeDot`, Vulkan's bounds
    // inherited from `x * (1 - a) + y * a` and the sum of products. Where the profile's `float`
    // functions are Metal's, a `float` one reaches Metal's own `mix` and `dot`; a `half` one is
    // the instruction on its `float` container, rounded once, noted. Their NaN is the device's,
    // noted as for any function.
    if matches!(op, FloatFn::Mix | FloatFn::Dot) {
        let fmt = f.float_fmt()?;
        let own = metal_functions && fmt == FloatFmt::F32;
        let mut notes = BTreeSet::new();
        if !own {
            notes.insert(if op == FloatFn::Mix {
                PrecisionNote::NativeMix
            } else {
                PrecisionNote::NativeDot
            });
            if op == FloatFn::Mix {
                notes.extend(device_float_notes(f, true, false));
            }
        }
        let nan = match (op, fmt) {
            (FloatFn::Mix, FloatFmt::F32) => Some(nan::NanAnswer::Canonical),
            _ => nan::function_answer(op, fast, every_flag(fmf), fmt),
        };
        return Some(FloatCall {
            how: FloatLowering::Fn(op),
            notes: OpNotes::everywhere(notes),
            nan,
        });
    }
    // `air.fast_fmod` on `float` is Metal's composition around a fused multiply-add, of the
    // device's quotient; the fast-math table is single precision, so a narrow call keeps `fmod`.
    // Where the profile's `float` functions are Metal's, its `fma` is Metal's own and its quotient
    // Metal's fast one in a fast-math entry, so its notes hold only outside one. Metal reads a
    // denormal operand as the zero of its sign and answers no denormal, in both modes; the
    // device's reading is noted where the profile lacks the deviation ([`coverage_left`]).
    if fast && op == FloatFn::Fmod && f.float_fmt() == Some(FloatFmt::F32) {
        if let Some(fma) = fma::Fma::by(routes, f) {
            let khr = fma.route.device_float();
            let notes = BTreeSet::from([PrecisionNote::DeviceArithmetic]);
            return Some(FloatCall {
                how: FloatLowering::FastFmod(fma),
                notes: OpNotes::unless_own(notes, metal_functions && khr).with(native_fma(fma)),
                nan: khr.then_some(nan::NanAnswer::Canonical),
            });
        }
    }
    // The NaN Metal answers, where the lowering does not answer it itself.
    let answer = f
        .float_fmt()
        .and_then(|fmt| nan::function_answer(op, fast, every_flag(fmf), fmt));
    let metal_nans = f.float_fmt().is_some_and(|fmt| op.metal_nans(fmt));
    let nan = answer.filter(|_| !metal_nans);
    // A function whose denormals Metal's table reads ([`denormal::coverage`]), and Metal's fast
    // `sinpi`, `cospi` and `tanpi` where the device's functions are Metal's, are noted where the
    // device's reading is left ([`call_coverage`], [`float_call`]); the device's instruction for
    // one of Metal's functions records its denormals itself ([`function_notes`]).
    let fast_pi = matches!(
        op,
        FloatFn::FastSinPi | FloatFn::FastCosPi | FloatFn::FastTanPi
    );
    let mut notes = function_notes(
        op,
        fast,
        f,
        metal_functions,
        (
            !op.metal_denormals() && denormal::coverage(op).is_none() && !fast_pi,
            !metal_nans && answer.is_none(),
        ),
    );
    // A `bfloat` or `double` `llvm.fmuladd` is fused or not, as the implementation chooses.
    if op == FloatFn::MulAdd {
        notes.always.insert(PrecisionNote::FastMathRewrites);
    }
    Some(FloatCall {
        how: FloatLowering::Fn(op),
        notes,
        nan,
    })
}

/// The precision notes this family's lowering records for the `air` math call `call` with
/// fast-math flags `fmf` on profile `p`: exactly what [`select`] records for the call itself
/// (its operands' and result's own instructions record theirs; a `half` result's rounding is
/// noted from the emitted module: native half rounding's where its form is not Metal's). `None` where the family does not lower `call` as a float
/// function: an integer function, `sincos` (selected with its store), or a function its type
/// refuses.
///
/// Validation reads it, through the AIR symbol Metal compiles an MSL function to, to tell which
/// operations the product computes as a near-equivalent of Metal's, as
/// [`PrecisionNote::of_math_call`].
pub fn float_call_notes(call: Math, fmf: FastMath, p: &DeviceProfile) -> Option<OpNotes> {
    let Ok(Fun::Float(op)) = air_math(call.f) else {
        return None;
    };
    let held = Held::Float(match call.ty.elem {
        Elem::F16 => FloatFmt::Half,
        Elem::BF16 => FloatFmt::BFloat,
        Elem::F32 => FloatFmt::F32,
        _ => return None,
    });
    let f = Form {
        held,
        lanes: u8::try_from(call.ty.lanes).ok()?,
    };
    let fast = call.precision == crate::air::intrinsics::Precision::Fast;
    float_call(op, fast, fmf, f, FloatFacts::on(p)).map(|lowered| lowered.notes)
}

/// The mode of the `float` functions of an entry of math mode `math` on profile `p`
/// ([`ArithPlan::metal_functions`]): Metal's in that mode where the profile's are Metal's,
/// `None` elsewhere.
fn metal_functions_in(p: &DeviceProfile, math: EntryMath) -> Option<libm::Mode> {
    (p.fields().float_functions == FloatFunctions::Metal).then_some(match math {
        EntryMath::Fast => libm::Mode::Fast,
        EntryMath::Precise => libm::Mode::Precise,
    })
}

impl PrecisionNote {
    /// The precision notes a compile for profile `p` records for the AIR float operator `op`
    /// (`fadd`, `fsub`, `fmul`, `fdiv` or `frem`) on lanes of `width`, with fast-math flags
    /// `fmf`, in an entry of math mode `math`: what a module states for the operator itself. Its
    /// operands' instructions record their own, and a `half` result's rounding is noted from the
    /// emitted module. `None` for an integer operator. The precision contract a tool reads to tell
    /// which operations the crate computes as a near-equivalent of Metal's.
    pub fn of_float_operator(
        op: AirBin,
        width: FloatWidth,
        fmf: FastMath,
        p: &DeviceProfile,
        math: EntryMath,
    ) -> Option<BTreeSet<PrecisionNote>> {
        let fmt = match width {
            FloatWidth::F16 => FloatFmt::Half,
            FloatWidth::F32 => FloatFmt::F32,
            FloatWidth::F64 => FloatFmt::F64,
        };
        float_binop_notes(op, fmt, fmf, p).map(|notes| notes.in_entry(metal_functions_in(p, math)))
    }

    /// The precision notes a compile for profile `p` records for the AIR math call `call` (an
    /// `air.*` function, as `metal2vulkan_air::intrinsics::AirIntrinsic::parse` decodes its
    /// symbol) with fast-math flags `fmf`, in an entry of math mode `math`: the notes for the call
    /// itself. `None` where the crate does not lower `call` as a float function: an integer
    /// function, `sincos` (lowered with its store), or a type it refuses.
    pub fn of_math_call(
        call: Math,
        fmf: FastMath,
        p: &DeviceProfile,
        math: EntryMath,
    ) -> Option<BTreeSet<PrecisionNote>> {
        float_call_notes(call, fmf, p).map(|notes| notes.in_entry(metal_functions_in(p, math)))
    }
}

/// `air.sincos(x, ptr)`: the sine is the result and the cosine is stored through `ptr` (a thread
/// pointer to the result's type). The store is the access family's: it is selected as the store
/// of the call's own result through `ptr` (the store the call performs, with the type the
/// signature gives it and the type's ABI alignment, which a `thread T&` has), and emitted with
/// the cosine bound to the result while it runs.
pub(super) fn select_sincos(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &crate::air::ir::Call,
    m: &mut Meter,
) -> Result<ArithStrategy, Error> {
    let (Some(&x), Some(&ptr)) = (call.args.first(), call.args.get(1)) else {
        return Err(cx.invariant("air.sincos without its operands"));
    };
    check_operands(cx, &[x])?;
    lowered_type(cx.entry, cx.profile, inst.ty)?;
    let shape = chunk::shape_of(cx, ty_of(cx, x)?)?;
    let f = shape.lane_form();
    if !f.is_float() || (f.bits() == 64 && !FloatFn::Sin.allows_f64()) {
        return Err(not_lowered());
    }
    // `fast::sincos` answers as `fast::sin` and `fast::cos` do, `+0` for both past the last
    // evaluated quadrant (a probe cases 914cc4d2 and 711cc190, every lane); the fast-math table is
    // single precision, so a narrow call keeps the precise functions.
    let symbol_fast = matches!(
        &call.callee,
        crate::air::ir::Callee::Air {
            op: AirIntrinsic::Math(math),
            ..
        } if math.precision == crate::air::intrinsics::Precision::Fast
    );
    let fast = symbol_fast && f.float_fmt() == Some(FloatFmt::F32);
    let (sin, cos) = chunk::sin_cos_fns(fast);
    // Each result's NaN is the device's, noted as a call of its own's (`float_call`'s rule): Metal's
    // `sincos` answers its `sin` and `cos`, word for word (cases 7b283404 and f6524207, every
    // pair). Each records its notes as a call of its own (`function_notes`).
    let metal_functions = cx.profile.fields().float_functions == FloatFunctions::Metal;
    for op in [sin, cos] {
        let answer = f
            .float_fmt()
            .and_then(|fmt| nan::function_answer(op, symbol_fast, every_flag(call.fmf), fmt));
        let metal_nans = f.float_fmt().is_some_and(|fmt| op.metal_nans(fmt));
        let notes = function_notes(
            op,
            fast,
            f,
            metal_functions,
            (!op.metal_denormals(), !metal_nans && answer.is_none()),
        );
        record(cx, notes);
        note_native_nan(cx, answer.filter(|_| !metal_nans));
    }
    fp_env(cx, call.fmf, f);
    let align = cx
        .memory
        .layouts
        .abi_align(inst.ty)
        .and_then(crate::air::ir::Align::from_bytes);
    let store = Inst {
        ty: inst.ty,
        kind: InstKind::Store {
            val: Value::Inst(at),
            ptr,
            align,
            volatile: false,
            atomic: None,
        },
    };
    let store = access::select(cx, at, &store, m)?;
    Ok(ArithStrategy::SinCos {
        shape,
        store: Box::new(store),
        fast,
    })
}

fn select_pack(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    p: &Pack,
) -> Result<ArithStrategy, Error> {
    // The float lane count; packing to a packed float format is not lowered.
    let lanes = match (
        math::pack_layout(p.format),
        math::float_unpack_lanes(p.format),
    ) {
        (Some((widths, _)), _) => widths.len(),
        (None, Some(n)) if p.dir == PackDir::Unpack => usize::from(n),
        _ => return Err(not_lowered()),
    };
    let x = *call
        .args
        .first()
        .ok_or_else(|| cx.invariant("air.pack without an operand"))?;
    check_operands(cx, &[x])?;
    let (floats, packed) = match p.dir {
        PackDir::Pack => (form(cx, x)?, result_form(cx, inst)?),
        PackDir::Unpack => (result_form(cx, inst)?, form(cx, x)?),
    };
    let word = Form {
        held: Held::Int {
            bits: math::packed_bits(p.format),
            container: 32,
        },
        lanes: 1,
    };
    if !floats.is_float() || usize::from(floats.lanes) != lanes || packed != word {
        return Err(cx.invariant("an air.pack whose types disagree with its format"));
    }
    // The device's conversion: GLSL.std.450's instruction for the format, or its multiply,
    // `RoundEven` and clamp, whose rounding Vulkan leaves to the device (`DevicePackConversion`).
    // Where the profile's `float` functions are Metal's, a `float` one of the four formats
    // GLSL.std.450 names reaches Metal's own `pack_float_to_*` or `unpack_*_to_float`.
    let own = FloatFacts::on(cx.profile).metal_functions
        && floats.float_fmt() == Some(FloatFmt::F32)
        && pack::instruction(p.format, p.dir).is_some();
    if !own {
        cx.precision.insert(PrecisionNote::DevicePackConversion);
    }
    Ok(ArithStrategy::Pack {
        dir: p.dir,
        format: p.format,
        floats,
    })
}

fn select_convert(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    c: &Convert,
    chunk: Option<u8>,
) -> Result<ArithStrategy, Error> {
    let x = *call
        .args
        .first()
        .ok_or_else(|| cx.invariant("air.convert without an operand"))?;
    check_operands(cx, &[x])?;
    let from = form_in(cx, x, chunk)?;
    let to = result_form_in(cx, inst, chunk)?;
    // An integer to `bool` is C++'s `x != 0`.
    if to.is_bool() && c.src_kind != NumKind::Float {
        if !from.is_int() || from.lanes != to.lanes {
            return Err(not_lowered());
        }
        return Ok(ArithStrategy::Convert {
            kind: ConvertKind::IntToBool,
            from,
            to,
        });
    }
    // A `float` or `half` to `bool` is Metal's measured reading. No reading exists of a `bfloat`
    // (whose denormals are `f32` denormals) or `double` source.
    if to.is_bool() {
        let measured = matches!(from.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half));
        if c.src_kind != NumKind::Float || !measured || from.lanes != to.lanes {
            return Err(not_lowered());
        }
        return Ok(ArithStrategy::Convert {
            kind: ConvertKind::ToBool,
            from,
            to,
        });
    }
    if let Some(fmt) = c.src_fp8 {
        let fp8 = narrow::Fp8Layout::of(fmt);
        let to_int = c.dst_kind != NumKind::Float && to.is_int();
        if from.bits() != 8 || !(to.is_float() || to_int) || from.lanes != to.lanes {
            return Err(cx.invariant("an 8-bit float air.convert whose types disagree"));
        }
        let kind = if to_int {
            // The format's documented value, not read on any Metal device.
            cx.precision.insert(PrecisionNote::Fp8Value);
            ConvertKind::Fp8ToInt {
                fp8,
                signed: c.dst_kind == NumKind::Signed,
            }
        } else {
            ConvertKind::FromFp8(fp8)
        };
        return Ok(ArithStrategy::Convert { kind, from, to });
    }
    let kind = match (c.dst_kind, c.src_kind) {
        (NumKind::Float, NumKind::Float) => ConvertKind::Float,
        (NumKind::Float, s) => ConvertKind::ToFloat {
            signed: s == NumKind::Signed,
        },
        (d, NumKind::Float) => ConvertKind::ToInt {
            signed: d == NumKind::Signed,
        },
        (_, s) => ConvertKind::Int {
            signed: s == NumKind::Signed,
        },
    };
    let consistent = match kind {
        ConvertKind::Float => from.is_float() && to.is_float(),
        ConvertKind::ToFloat { .. } => !from.is_float() && to.is_float(),
        ConvertKind::ToInt { .. } => from.is_float() && to.is_int(),
        ConvertKind::Int { .. } => !from.is_float() && to.is_int(),
        ConvertKind::FromFp8(_)
        | ConvertKind::Fp8ToInt { .. }
        | ConvertKind::ToBool
        | ConvertKind::IntToBool => false,
    };
    if !consistent || from.lanes != to.lanes {
        return Err(cx.invariant("an air.convert whose kinds disagree with its types"));
    }
    let s = ArithStrategy::Convert { kind, from, to };
    // Metal's widening of a `half` NaN, as `fpext`'s.
    let nan = match (kind, from.float_fmt(), to.float_fmt()) {
        (ConvertKind::Float, Some(a), Some(b)) => nan::convert_answer(a, b),
        _ => None,
    };
    Ok(with_nan(cx, s, nan))
}

// ---- Emission ----

fn operand(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<SValue> {
    match cx.plan.values.id(v) {
        Some(id) => cx.operand(fb, id),
        None => Err(cx.fail("an arith operand out of the value range")),
    }
}

/// The SPIR-V values of the first `n` operands (all when `None`) of instruction `inst`.
fn operand_values(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    inst: &Inst,
    n: Option<usize>,
) -> BResult<Vec<SValue>> {
    let vals: Vec<Value> = match &inst.kind {
        InstKind::Call(call) => call.args.to_vec(),
        kind => operands_of(kind),
    };
    let take = n.unwrap_or(vals.len()).min(vals.len());
    let mut out = Vec::with_capacity(take);
    for v in vals.into_iter().take(take) {
        out.push(operand(cx, fb, v)?);
    }
    Ok(out)
}

fn nth(cx: &mut EmitCx<'_>, xs: &[SValue], i: usize) -> BResult<SValue> {
    match xs.get(i) {
        Some(x) => Ok(*x),
        None => Err(cx.fail("an arith strategy with too few operands")),
    }
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &ArithStrategy,
) -> BResult<()> {
    use ArithStrategy as S;
    let body = cx.body;
    let Some(inst) = body.insts.get(at) else {
        return Err(cx.fail("an arith strategy names a missing instruction"));
    };
    let Some(result) = cx.plan.values.inst(at) else {
        return Err(cx.fail("an arith instruction out of the value range"));
    };
    let xs = operand_values(cx, fb, inst, arity(s))?;
    let x = match s {
        // Its only uses read its operand: it has no value of its own.
        S::IndexExtension => return Ok(()),
        // No SPIR-V value: its extractions resolve through provenance.
        S::PointerAggregate => return Ok(()),
        S::Handle(k) => crate::memory::leaf::k32(fb, *k)?,
        S::Forward(v) => cx.operand(fb, *v)?,
        S::Null(d) => aggregate::null_value(cx, fb, d)?,
        S::Member { of, path, bools } => {
            let x = cx.operand(fb, *of)?;
            let x = fb.composite_extract(x, path)?;
            match bools {
                Some(lanes) => ops::member_to_bools(fb, x, *lanes)?,
                None => x,
            }
        }
        S::IndexWord(w) => access::index::emit_word(cx, fb, w)?,
        S::SinCos { shape, store, fast } => {
            let p = nth(cx, &xs, 0)?;
            let lib = lib(cx);
            let (sin, cos) = chunk::sin_cos(fb, *shape, p, *fast, &lib)?;
            // The access family's store reads the stored value as the call's result.
            cx.define(result, cos);
            access::emit(cx, fb, at, store)?;
            sin
        }
        _ => value(cx, fb, s, &xs)?,
    };
    cx.define(result, x);
    Ok(())
}

/// What the entry being emitted's float function calls take ([`libm::Lib`]).
pub(crate) fn lib(cx: &EmitCx<'_>) -> libm::Lib {
    libm::Lib::new(cx.decl.families.arith.fmod)
}

/// The operands strategy `s` reads (a prefix of the instruction's; all when `None`).
fn arity(s: &ArithStrategy) -> Option<usize> {
    use ArithStrategy as S;
    match s {
        S::IntFn { f, .. } => Some(f.arity()),
        S::FloatFn { f, .. } => Some(f.arity()),
        // A constant element index is a literal, never materialized.
        S::Copy
        | S::Convert { .. }
        | S::Pack { .. }
        | S::ExtractConst(_)
        | S::SinCos { .. }
        | S::WordBitcast { .. }
        | S::LaneExtract { at: Some(_), .. } => Some(1),
        S::InsertConst(_) | S::LaneInsert { at: Some(_), .. } | S::ChunkedDot { .. } => Some(2),
        // An index word reads its operands itself (an index extension has no value).
        S::Zero(_)
        | S::Undef(_)
        | S::IndexExtension
        | S::IndexWord(_)
        | S::PointerAggregate
        | S::Handle(_)
        | S::Forward(_)
        | S::Null(_)
        | S::Member { .. } => Some(0),
        S::Chunked { parts, .. } => parts.first().and_then(|(_, p)| arity(p)),
        S::Wide(w) => w.arity(),
        _ => None,
    }
}

/// The value of strategy `s` on operand values `xs` (every strategy but the ones [`emit`]
/// handles itself).
fn value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &ArithStrategy,
    xs: &[SValue],
) -> BResult<SValue> {
    use ArithStrategy as S;
    let a = |cx: &mut EmitCx<'_>, i| nth(cx, xs, i);
    Ok(match s {
        S::IndexExtension
        | S::IndexWord(_)
        | S::SinCos { .. }
        | S::PointerAggregate
        | S::Handle(_)
        | S::Forward(_)
        | S::Null(_)
        | S::Member { .. } => return Err(cx.fail("a strategy with no value of its own")),
        S::Chunked {
            lanes,
            parts,
            wide,
            result,
        } => chunk::emit_chunked(cx, fb, *lanes, parts, wide, result, xs)?,
        S::ChunkedDot {
            lanes,
            elem,
            contract,
        } => chunk::dot(fb, *lanes, *elem, *contract, a(cx, 0)?, a(cx, 1)?)?,
        S::LaneExtract { src, at, index } => chunk::extract(fb, *src, *at, *index, xs)?,
        S::LaneInsert { dst, at, index } => chunk::insert(fb, *dst, *at, *index, xs)?,
        S::LaneShuffle {
            a,
            b,
            out,
            mask,
            fragment: false,
        } => chunk::shuffle(fb, *a, *b, *out, mask, xs)?,
        S::LaneShuffle {
            a,
            b,
            out,
            mask,
            fragment: true,
        } => chunk::shuffle_fragment(fb, *a, *b, *out, mask, xs)?,
        S::WordBitcast { from, to } => {
            let p = nth(cx, xs, 0)?;
            chunk::bitcast(fb, *from, *to, p)?
        }
        S::Int { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::int_binary(fb, *op, *form, p, q)?
        }
        S::Bool { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::bool_binary(fb, *op, *form, p, q)?
        }
        S::Float { op, form, contract } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::float_binary(fb, *op, *form, *contract, p, q)?
        }
        S::F16 { op, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::f16_binary(fb, *op, form.lanes, p, q)?
        }
        S::FastDivide { contract, .. } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            fb.binary_with(crate::spirv::ir::BinOp::FDiv, p, q, *contract)?
        }
        S::NativeDivide { form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            divide::native(fb, *form, p, q)?
        }
        S::FNeg { .. } => {
            let p = a(cx, 0)?;
            fb.unary(UnOp::FNegate, p)?
        }
        S::IntCompare { pred, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::int_compare(fb, *pred, *form, p, q)?
        }
        S::BoolCompare { pred } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::bool_compare(fb, *pred, p, q)?
        }
        S::FloatCompare { pred, form } => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            ops::float_compare(fb, *pred, *form, p, q)?
        }
        S::Cast { kind, from, to } => {
            let p = a(cx, 0)?;
            ops::cast(fb, *kind, *from, *to, p)?
        }
        S::Bitcast { plan, from, to } => {
            let p = a(cx, 0)?;
            ops::bitcast(fb, *plan, *from, *to, p)?
        }
        S::Select => {
            let (c, t, f) = (a(cx, 0)?, a(cx, 1)?, a(cx, 2)?);
            fb.select(c, t, f)?
        }
        S::Copy => a(cx, 0)?,
        S::Zero(d) => match legal::zero(d) {
            Some(k) => match legal::materialize(fb, &k)? {
                Some(x) => x,
                None => return Err(cx.fail("a zero arith does not materialize")),
            },
            None => return Err(cx.fail("a zero of a non-arith type")),
        },
        S::Undef(d) => match legal::spirv_type(fb.module(), d)? {
            Some(t) => fb.undef(t)?,
            None => return Err(cx.fail("an undef of a non-arith type")),
        },
        S::ExtractConst(i) => {
            let v = a(cx, 0)?;
            let y = fb.composite_extract(v, &[*i])?;
            // A store of a narrow lane takes its bits from its vector's.
            if narrow::may_be_narrow(fb, y)? {
                fb.note_lane(y, v, *i);
            }
            y
        }
        S::ExtractDynamic { index } => {
            let (v, i) = (a(cx, 0)?, a(cx, 1)?);
            let i = ops::element_index(fb, i, *index)?;
            fb.vector_extract_dynamic(v, i)?
        }
        S::InsertConst(i) => {
            let (v, e) = (a(cx, 0)?, a(cx, 1)?);
            // Narrow lanes' bits follow the insert.
            narrow::insert(fb, v, e, *i)?
        }
        S::InsertDynamic { index } => {
            let (v, e, i) = (a(cx, 0)?, a(cx, 1)?, a(cx, 2)?);
            let i = ops::element_index(fb, i, *index)?;
            fb.vector_insert_dynamic(v, e, i)?
        }
        S::Shuffle(c) => {
            let (p, q) = (a(cx, 0)?, a(cx, 1)?);
            fb.vector_shuffle(p, q, c)?
        }
        S::ExtractValue(idx) => {
            let v = a(cx, 0)?;
            fb.composite_extract(v, idx)?
        }
        S::InsertValue(idx) => {
            let (agg, v) = (a(cx, 0)?, a(cx, 1)?);
            fb.composite_insert(v, agg, idx)?
        }
        S::ExtractBools { path, lanes } => {
            let v = a(cx, 0)?;
            let x = fb.composite_extract(v, path)?;
            ops::member_to_bools(fb, x, *lanes)?
        }
        S::InsertBools { path, lanes } => {
            let (agg, v) = (a(cx, 0)?, a(cx, 1)?);
            let x = ops::bools_to_member(fb, v, *lanes)?;
            fb.composite_insert(x, agg, path)?
        }
        S::IntFn { f, form } => math::int_fn(fb, *f, *form, xs)?,
        S::Wide(w) => {
            let routine = match w.routine() {
                Some(d) => match cx.decl.families.arith.divides.get(&d) {
                    Some(f) => Some(*f),
                    None => return Err(cx.fail("a wide division whose routine is not declared")),
                },
                None => None,
            };
            wide::emit(fb, w, xs, routine)?
        }
        S::FloatFn { f, form, contract } => {
            let lib = lib(cx);
            math::float_fn(fb, *f, *form, *contract, xs, &lib)?
        }
        S::Fma(f) => fma::emit(fb, *f, xs)?,
        S::FastFmod { fma: f } => {
            let (x, y) = match xs {
                [x, y, ..] => (*x, *y),
                _ => return Err(cx.fail("a fast fmod without its operands")),
            };
            let fma = |fb: &mut FunctionBuilder<'_>, args: &[SValue]| fma::emit(fb, *f, args);
            let divide = |fb: &mut FunctionBuilder<'_>, p: SValue, q: SValue| {
                fb.binary_with(crate::spirv::ir::BinOp::FDiv, p, q, Contraction::Forbidden)
            };
            math::fast_fmod(fb, x, y, &divide, &fma)?
        }
        S::Convert { kind, from, to } => {
            let p = a(cx, 0)?;
            // An 8-bit float source converts through `f32`.
            let width = from.float_fmt().map_or(32, FloatFmt::container);
            let test = nan::NanTest::for_width(&cx.plan.families.arith.modes, width);
            math::convert(fb, *kind, *from, *to, p, test)?
        }
        S::Pack {
            dir,
            format,
            floats,
        } => {
            let p = a(cx, 0)?;
            match dir {
                PackDir::Pack => pack::pack(fb, *format, *floats, p)?,
                PackDir::Unpack => pack::unpack(fb, *format, *floats, p)?,
            }
        }
    })
}
