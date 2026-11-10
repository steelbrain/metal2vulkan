//! Value legality: how every non-pointer AIR value is held in SPIR-V.
//!
//! The representation depends on the AIR type and one profile bit, `shaderInt64`; constants are
//! selected without a profile and widened where emission's plan holds words ([`widen_const`]).
//! Otherwise the profile only decides whether a representation is available ([`check`]):
//!
//! - `i1` is `bool`; every other integer is held **zero-extended** in an unsigned 32-bit
//!   container (`i2`..`i32`) or 64-bit container (`i33`..`i64`), or, without `shaderInt64`, as
//!   two `u32` words ([`Held::Wide`], [`widen`], `super::wide`). The bits above the AIR width are
//!   always zero, so an operation that can set them (add, sub, mul, shl, signed results) masks
//!   its result, and a signed operation sign-extends its operands first. `i8`/`i16` therefore
//!   never need `Int8`/`Int16`.
//! - `float` is `f32` and `double` is `f64` (needs `shaderFloat64`, else refused). `half` and
//!   `bfloat` are **held widened** in an `f32` that is always exactly a value of the narrow format,
//!   on every profile ([`super::narrow`]): no profile needs `shaderFloat16`, and a constant
//!   materializes as its exact `f32` value.
//! - Vectors of 2 to 4 of those elements are SPIR-V vectors; a vector of any other length is
//!   **chunked** ([`chunk_lanes`]): a layout-free struct of 4-lane vectors, the last member holding
//!   the remaining 1 to 4 lanes (a scalar when one remains).
//! - Arrays and structs held as SSA values are [`DemoteTy::Composite`], emitted as a SPIR-V
//!   struct without layout (an array member per element), up to [`MAX_SCALARS`] scalar leaves
//!   in total (a vector lane counts as one) and [`MAX_DEPTH`] levels. The total is computed from
//!   the AIR type before any representation is built, so a small type that expands to many
//!   leaves (`[64 x [64 x [64 x i32]]]`) is refused without allocating its expansion.
//! - A 2-4-lane `bool` vector that is a member of such an aggregate is held as `u32` lanes of 0
//!   and 1 ([`as_member`]): an insertion selects them from the `bool`s and an extraction compares
//!   them with 0. SPIRV-Cross (MoltenVK) declares every `bool` struct member as a `short` and
//!   assigns a `bool` vector to it without a conversion, which Metal refuses; a `u32` member needs
//!   none. A scalar `bool` member converts implicitly and stays a `bool`. A chunked `bool` vector
//!   keeps its `bool` chunks, which SPIRV-Cross renders the same way: an open gap.

use crate::air::error::AbiError;
use crate::air::ir::{Const, ConstId, Type, TypeId as AirTypeId, TypeTable};
use crate::air::Entry;
use crate::device::requirements::VkFeature;
use crate::device::DeviceProfile;
use crate::error::{
    Capability, Error, Feature, InputError, InputErrorKind, Location, LowerFamily, Unsupported,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId as SConstId, DemoteTy, ScalarTy, TypeId, Value as SValue};
use crate::spirv::ir::{ConstBits, Constant, Type as SType};

/// The most scalar leaves (vector lanes counted singly) an aggregate SSA value may expand to. It
/// bounds the memory of every [`DemoteTy`], [`LoweredConst`], and SPIR-V type or constant this
/// family builds for one value, independently of how the AIR type nests.
pub const MAX_SCALARS: u64 = 256;
/// The deepest aggregate nesting held as an SSA value.
pub const MAX_DEPTH: u32 = 8;

pub(super) fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Arith)).into()
}

/// How one AIR scalar is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Held {
    Bool,
    /// An `i<bits>` held zero-extended in an unsigned `container`-bit integer (32 or 64).
    Int {
        bits: u8,
        container: u8,
    },
    Float(FloatFmt),
    /// An `i<bits>` (33 to 64 bits) held zero-extended as two `u32` words, low word first, on a
    /// profile without `shaderInt64` ([`super::wide`]): a scalar is a `uvec2`, a vector of `n`
    /// lanes a struct of two `u32` vectors of `n` lanes (the low words, then the high words). Only
    /// [`wide_form_on`] produces it; [`form_on`] keeps refusing such a type.
    Wide {
        bits: u8,
    },
}

/// A float format and its container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFmt {
    /// IEEE binary16, held widened in an `f32`.
    Half,
    /// bfloat16 (the high half of an `f32`), held widened in an `f32`.
    BFloat,
    F32,
    F64,
}

impl FloatFmt {
    /// The AIR width in bits.
    pub fn bits(self) -> u8 {
        match self {
            FloatFmt::Half | FloatFmt::BFloat => 16,
            FloatFmt::F32 => 32,
            FloatFmt::F64 => 64,
        }
    }

    /// The container width in bits.
    pub fn container(self) -> u8 {
        match self {
            FloatFmt::F64 => 64,
            _ => 32,
        }
    }

    /// Held widened in a larger container.
    pub fn narrow(self) -> bool {
        matches!(self, FloatFmt::Half | FloatFmt::BFloat)
    }

    /// The container bits of `x` rounded to this format (to nearest even).
    pub fn const_bits(self, x: f64) -> u64 {
        match self {
            FloatFmt::F64 => x.to_bits(),
            FloatFmt::F32 => u64::from((x as f32).to_bits()),
            FloatFmt::Half => self.widen_bits(u64::from(f64_to_f16(x))),
            FloatFmt::BFloat => {
                let f = x as f32;
                let mut b = f.to_bits();
                if f.is_nan() {
                    return u64::from((b | 0x0040_0000) & 0xFFFF_0000);
                }
                // Round to odd in f32 first, so the one rounding to bfloat is correct.
                if f.is_finite() && f64::from(f) != x {
                    if f64::from(f).abs() > x.abs() {
                        b -= 1;
                    }
                    b |= 1;
                }
                let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
                u64::from(r)
            }
        }
    }

    /// The container bits of this format's bits `b` (exact: the same value, and a NaN its sign
    /// and payload, a signalling one not quieted).
    pub fn widen_bits(self, b: u64) -> u64 {
        match self {
            FloatFmt::Half if b & 0x7C00 == 0x7C00 && b & 0x3FF != 0 => {
                ((b & 0x8000) << 16) | 0x7F80_0000 | ((b & 0x3FF) << 13)
            }
            FloatFmt::Half => u64::from((f16_to_f64(b as u16) as f32).to_bits()),
            FloatFmt::BFloat => (b & 0xFFFF) << 16,
            _ => b,
        }
    }

    /// This format's bits of a container value `c` that is one of its values (the inverse of
    /// [`FloatFmt::widen_bits`]).
    #[cfg(test)]
    pub fn narrow_bits(self, c: u64) -> u64 {
        match self {
            FloatFmt::Half => {
                let x = f32::from_bits(c as u32);
                if x.is_nan() {
                    let top = (c >> 13) & 0x3FF;
                    let q = if top == 0 { 0x200 } else { 0 };
                    return ((c >> 16) & 0x8000) | 0x7C00 | top | q;
                }
                u64::from(f64_to_f16(f64::from(x)))
            }
            FloatFmt::BFloat => (c >> 16) & 0xFFFF,
            _ => c,
        }
    }
}

/// `x` rounded to an IEEE half (round to nearest even).
pub fn f64_to_f16(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if x.is_nan() {
        return 0x7E00;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a < 2f64.powi(-14) {
        // Subnormal (a carry into 1024 is the smallest normal's encoding).
        return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
    }
    let mut e = a.log2().floor() as i32;
    if a / 2f64.powi(e) < 1.0 {
        e -= 1;
    } else if a / 2f64.powi(e) >= 2.0 {
        e += 1;
    }
    let mut m = (a / 2f64.powi(e) * 1024.0).round_ties_even();
    if m >= 2048.0 {
        m /= 2.0;
        e += 1;
    }
    sign | (((e + 15) as u16) << 10) | ((m as u16).wrapping_sub(1024) & 0x3FF)
}

/// The value of IEEE half bits `h` (NaNs keep their payload's top bits).
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1F);
    let m = f64::from(h & 0x3FF);
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::from(f32::from_bits(
            (u32::from(h & 0x8000) << 16) | 0x7F80_0000 | (u32::from(h & 0x3FF) << 13),
        )),
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// How one AIR scalar or vector is held: its element and lane count (1 for a scalar).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub held: Held,
    pub lanes: u8,
}

impl Form {
    /// The lowered scalar type of the element.
    pub fn scalar(self) -> ScalarTy {
        match self.held {
            Held::Bool => ScalarTy::Bool,
            Held::Int { container, .. } => ScalarTy::Int { bits: container },
            Held::Float(f) => ScalarTy::Float {
                bits: f.container(),
            },
            // The element of each of its two words.
            Held::Wide { .. } => ScalarTy::Int { bits: 32 },
        }
    }

    /// The lowered representation.
    pub fn demote(self) -> DemoteTy {
        if let Held::Wide { .. } = self.held {
            return wide_demote(self.lanes);
        }
        if self.lanes == 1 {
            DemoteTy::Scalar(self.scalar())
        } else {
            DemoteTy::Vector {
                elem: self.scalar(),
                len: self.lanes,
            }
        }
    }

    /// The same lanes holding `held`.
    pub fn with(self, held: Held) -> Form {
        Form { held, ..self }
    }

    /// The AIR element width in bits (1 for `bool`).
    pub fn bits(self) -> u8 {
        match self.held {
            Held::Bool => 1,
            Held::Int { bits, .. } | Held::Wide { bits } => bits,
            Held::Float(f) => f.bits(),
        }
    }

    /// The container width in bits (a wide integer's two words are 64 bits).
    pub fn container(self) -> u8 {
        match self.held {
            Held::Bool => 1,
            Held::Int { container, .. } => container,
            Held::Float(f) => f.container(),
            Held::Wide { .. } => 64,
        }
    }

    pub fn is_bool(self) -> bool {
        self.held == Held::Bool
    }

    pub fn is_int(self) -> bool {
        matches!(self.held, Held::Int { .. })
    }

    pub fn is_float(self) -> bool {
        matches!(self.held, Held::Float { .. })
    }

    /// An integer held as two words.
    pub fn is_wide(self) -> bool {
        matches!(self.held, Held::Wide { .. })
    }

    /// An integer held in a container wider than its AIR width.
    pub fn legalized(self) -> bool {
        matches!(self.held, Held::Int { bits, container } if bits < container)
    }

    /// The float format, for a float form.
    pub fn float_fmt(self) -> Option<FloatFmt> {
        match self.held {
            Held::Float(f) => Some(f),
            _ => None,
        }
    }

    /// A narrow float held widened.
    pub fn is_narrow(self) -> bool {
        self.float_fmt().is_some_and(FloatFmt::narrow)
    }

    /// A value held in a container wider than its AIR width (a legalized integer or a narrow
    /// float): its container bits are not its AIR bits.
    pub fn padded(self) -> bool {
        self.legalized() || self.is_narrow()
    }

    /// The container bits of float constant `x` in this form's format (to nearest even).
    pub fn const_bits(self, x: f64) -> u64 {
        match self.held {
            Held::Float(f) => f.const_bits(x),
            _ => x as u64,
        }
    }
}

/// The representation of a wide integer of `lanes` lanes: a `uvec2` (low, high) for a scalar, else
/// a struct of the low-word vector and the high-word vector.
pub fn wide_demote(lanes: u8) -> DemoteTy {
    let word = ScalarTy::Int { bits: 32 };
    if lanes == 1 {
        DemoteTy::Vector { elem: word, len: 2 }
    } else {
        let v = DemoteTy::Vector {
            elem: word,
            len: lanes,
        };
        DemoteTy::Composite(vec![v.clone(), v].into())
    }
}

/// `d` with every 64-bit integer scalar or vector replaced by its wide representation.
pub fn widen(d: &DemoteTy) -> DemoteTy {
    match d {
        DemoteTy::Scalar(ScalarTy::Int { bits: 64 }) => wide_demote(1),
        DemoteTy::Vector {
            elem: ScalarTy::Int { bits: 64 },
            len,
        } => wide_demote(*len),
        DemoteTy::Composite(parts) => DemoteTy::Composite(parts.iter().map(widen).collect()),
        other => other.clone(),
    }
}

/// Constant `k` with every 64-bit integer scalar or vector in its wide representation.
pub fn widen_const(k: LoweredConst) -> LoweredConst {
    let word = ScalarTy::Int { bits: 32 };
    let lo = |b: u64| b & 0xFFFF_FFFF;
    let hi = |b: u64| b >> 32;
    match k {
        LoweredConst::Scalar {
            ty: ScalarTy::Int { bits: 64 },
            bits,
        } => LoweredConst::Vector {
            elem: word,
            lanes: vec![lo(bits), hi(bits)].into(),
        },
        LoweredConst::Vector {
            elem: ScalarTy::Int { bits: 64 },
            lanes,
        } => {
            let n = u8::try_from(lanes.len()).unwrap_or(u8::MAX);
            LoweredConst::Composite {
                ty: wide_demote(n),
                parts: vec![
                    LoweredConst::Vector {
                        elem: word,
                        lanes: lanes.iter().map(|b| lo(*b)).collect(),
                    },
                    LoweredConst::Vector {
                        elem: word,
                        lanes: lanes.iter().map(|b| hi(*b)).collect(),
                    },
                ]
                .into(),
            }
        }
        LoweredConst::Composite { ty, parts } => LoweredConst::Composite {
            ty: widen(&ty),
            parts: parts.into_vec().into_iter().map(widen_const).collect(),
        },
        LoweredConst::Undef(d) => LoweredConst::Undef(widen(&d)),
        other => other,
    }
}

impl LoweredConst {
    /// The lowered representation of the constant.
    pub fn demote(&self) -> DemoteTy {
        match self {
            LoweredConst::Scalar { ty, .. } => DemoteTy::Scalar(*ty),
            LoweredConst::Vector { elem, lanes } => DemoteTy::Vector {
                elem: *elem,
                len: u8::try_from(lanes.len()).unwrap_or(u8::MAX),
            },
            LoweredConst::Composite { ty, .. } | LoweredConst::Undef(ty) => ty.clone(),
        }
    }
}

/// How AIR scalar type `t` is held, or `None` when it is not an arithmetic scalar this family
/// lowers (pointers, labels, ...).
pub fn held_of(t: &Type) -> Option<Held> {
    Some(match *t {
        Type::Int(1) => Held::Bool,
        Type::Int(bits @ 2..=32) => Held::Int {
            bits,
            container: 32,
        },
        Type::Int(bits @ 33..=64) => Held::Int {
            bits,
            container: 64,
        },
        Type::Half => Held::Float(FloatFmt::Half),
        Type::BFloat => Held::Float(FloatFmt::BFloat),
        Type::Float => Held::Float(FloatFmt::F32),
        Type::Double => Held::Float(FloatFmt::F64),
        _ => return None,
    })
}

/// Whether AIR type `ty` is an integer or 2-4-lane integer vector of 33 to 64 bits: a type held
/// in a 64-bit container, or as two words without `shaderInt64`.
pub fn is_wide_int(types: &TypeTable, ty: AirTypeId) -> bool {
    match types.get(ty) {
        Some(Type::Int(33..=64)) => true,
        Some(Type::Vector { elem, len }) if (2..=4).contains(len) => {
            matches!(types.get(*elem), Some(Type::Int(33..=64)))
        }
        _ => false,
    }
}

/// The lane counts of the chunks of a chunked vector of `lanes` lanes: 4-lane chunks, the last one
/// holding the remaining 1 to 4 lanes.
pub fn chunk_lanes(lanes: u32) -> Vec<u8> {
    (0..lanes.div_ceil(4))
        .map(|c| (lanes - 4 * c).min(4) as u8)
        .collect()
}

/// The element and lane count of AIR type `ty` when it is a vector held chunked: one of other than
/// 2 to 4 lanes of an element this family holds.
pub fn chunked(types: &TypeTable, ty: AirTypeId) -> Option<(Held, u32)> {
    match types.get(ty)? {
        Type::Vector { elem, len } if !(2..=4).contains(len) && *len > 0 => {
            Some((held_of(types.get(*elem)?)?, *len))
        }
        _ => None,
    }
}

/// The lowered representation of a chunked vector of `lanes` lanes of `held`.
pub fn chunked_demote(held: Held, lanes: u32) -> DemoteTy {
    DemoteTy::Composite(
        chunk_lanes(lanes)
            .into_iter()
            .map(|k| Form { held, lanes: k }.demote())
            .collect(),
    )
}

/// The form of AIR scalar or vector type `ty` (profile-independent). A chunked vector has no
/// form (its chunks do, [`chunked`]).
pub fn form_of(types: &TypeTable, ty: AirTypeId) -> Result<Form, Error> {
    match types.get(ty).ok_or_else(not_lowered)? {
        Type::Vector { elem, len } => {
            let lanes = u8::try_from(*len)
                .ok()
                .filter(|l| (2..=4).contains(l))
                .ok_or_else(not_lowered)?;
            let held = types.get(*elem).and_then(held_of).ok_or_else(not_lowered)?;
            Ok(Form { held, lanes })
        }
        t => Ok(Form {
            held: held_of(t).ok_or_else(not_lowered)?,
            lanes: 1,
        }),
    }
}

/// The lowered representation of AIR type `ty`, profile-independent (every type [`check`] may
/// then refuse on a profile).
pub fn demote_of(types: &TypeTable, ty: AirTypeId) -> Result<DemoteTy, Error> {
    scalars_within(types, ty, 0, MAX_SCALARS)?;
    demote_at(types, ty)
}

/// The number of scalar leaves of AIR type `ty` (vector lanes counted singly), refusing a type
/// this family does not hold, one nested deeper than [`MAX_DEPTH`], or one with more than
/// `budget` leaves. It never expands the type: an array visits its element once with the budget
/// divided by its length, and a struct gives each field what its earlier fields left. Every
/// visited type has at least one leaf, so the work is bounded by `budget` times the depth.
fn scalars_within(types: &TypeTable, ty: AirTypeId, depth: u32, budget: u64) -> Result<u64, Error> {
    if depth > MAX_DEPTH {
        return Err(not_lowered());
    }
    match types.get(ty).ok_or_else(not_lowered)? {
        Type::Array { elem, len } => {
            if *len == 0 || *len > budget {
                return Err(not_lowered());
            }
            let each = scalars_within(types, *elem, depth + 1, budget / len)?;
            Ok(each * len)
        }
        Type::Struct { fields, .. } => {
            if fields.is_empty() || fields.len() as u64 > budget {
                return Err(not_lowered());
            }
            let mut used = 0u64;
            for f in fields.iter() {
                used += scalars_within(types, *f, depth + 1, budget - used)?;
            }
            Ok(used)
        }
        _ => {
            let lanes = match chunked(types, ty) {
                Some((_, n)) => u64::from(n),
                None => u64::from(form_of(types, ty)?.lanes),
            };
            if lanes > budget {
                return Err(not_lowered());
            }
            Ok(lanes)
        }
    }
}

/// The representation of `ty`, which [`scalars_within`] has bounded.
fn demote_at(types: &TypeTable, ty: AirTypeId) -> Result<DemoteTy, Error> {
    match types.get(ty).ok_or_else(not_lowered)? {
        Type::Array { elem, len } => {
            let e = as_member(demote_at(types, *elem)?);
            let len = usize::try_from(*len).map_err(|_| not_lowered())?;
            Ok(DemoteTy::Composite(vec![e; len].into()))
        }
        Type::Struct { fields, .. } => {
            let parts = fields
                .iter()
                .map(|f| demote_at(types, *f).map(as_member))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DemoteTy::Composite(parts.into()))
        }
        _ => match chunked(types, ty) {
            Some((held, lanes)) => Ok(chunked_demote(held, lanes)),
            None => Ok(form_of(types, ty)?.demote()),
        },
    }
}

/// The representation of an aggregate member whose own representation is `d`: a 2-4-lane `bool`
/// vector is held as `u32` lanes of 0 and 1, every other member as itself.
pub fn as_member(d: DemoteTy) -> DemoteTy {
    match d {
        DemoteTy::Vector {
            elem: ScalarTy::Bool,
            len,
        } => DemoteTy::Vector {
            elem: ScalarTy::Int { bits: 32 },
            len,
        },
        d => d,
    }
}

/// The lane count of AIR type `ty` where it is a 2-4-lane `bool` vector: an aggregate holds such
/// a member as `u32` lanes ([`as_member`]), so inserting or extracting it converts.
pub fn bool_lanes(types: &TypeTable, ty: AirTypeId) -> Option<u8> {
    match types.get(ty)? {
        Type::Vector { elem, len } if (2..=4).contains(len) => {
            matches!(types.get(*elem)?, Type::Int(1)).then_some(u8::try_from(*len).ok()?)
        }
        _ => None,
    }
}

/// Whether profile `p` can hold scalar `s`: 64-bit integers need `shaderInt64`, 64-bit floats
/// `shaderFloat64` (both refused naming the feature). No lowered representation is a 16-bit float
/// (narrow floats are held widened), so one is not this family's.
pub fn check_scalar(p: &DeviceProfile, s: ScalarTy) -> Result<(), Error> {
    let leaf = &p.fields().leaf;
    match s {
        ScalarTy::Int { bits: 64 } if !leaf.int64 => Err(Unsupported::needs(
            Feature::Int64Arithmetic,
            Capability::Feature(VkFeature::ShaderInt64),
        )
        .into()),
        ScalarTy::Float { bits: 64 } if !leaf.float64 => Err(Unsupported::needs(
            Feature::Float64Arithmetic,
            Capability::Feature(VkFeature::ShaderFloat64),
        )
        .into()),
        ScalarTy::Float { bits: 16 } => Err(not_lowered()),
        _ => Ok(()),
    }
}

/// [`check_scalar`] over every scalar of `d`.
pub fn check(p: &DeviceProfile, d: &DemoteTy) -> Result<(), Error> {
    match d {
        DemoteTy::Scalar(s) | DemoteTy::Vector { elem: s, .. } => check_scalar(p, *s),
        DemoteTy::Composite(parts) => parts.iter().try_for_each(|m| check(p, m)),
        _ => Err(not_lowered()),
    }
}

/// [`form_of`] checked on `p`.
pub fn form_on(e: &Entry, p: &DeviceProfile, ty: AirTypeId) -> Result<Form, Error> {
    let f = form_of(e.module.types(), ty)?;
    check_scalar(p, f.scalar())?;
    Ok(f)
}

/// Whether profile `p` holds 64-bit integers as two words: it lacks `shaderInt64`.
pub fn holds_wide(p: &DeviceProfile) -> bool {
    !p.fields().leaf.int64
}

/// [`form_on`], except that an integer of 33 to 64 bits is [`Held::Wide`] on a profile without
/// `shaderInt64`. Only families that lower the wide form call it; every other family keeps refusing
/// such a type through [`form_on`].
pub fn wide_form_on(e: &Entry, p: &DeviceProfile, ty: AirTypeId) -> Result<Form, Error> {
    let f = form_of(e.module.types(), ty)?;
    Ok(f.with(held_on(p, f.held)?))
}

/// Refuse a 64-bit integer (a scalar or 2-4-lane vector of AIR type `ty`) as a stage interface
/// value of entry `e`. Metal's compiler rejects `long` and `ulong` as a vertex attribute, a varying
/// (flat or not), a color output or input, and a stage builtin ("type 'long' is not valid for
/// attribute"), so such an input is not Metal's, on every profile: `InvalidInput`
/// (`Abi(RoleTypeMismatch)`), never a capability that would not help.
pub fn refuse_int64_interface(e: &Entry, ty: AirTypeId) -> Result<(), Error> {
    if !is_wide_int(e.module.types(), ty) {
        return Ok(());
    }
    Err(Error::InvalidInput(InputError::new(
        InputErrorKind::Abi(AbiError::RoleTypeMismatch),
        Location::metadata(Some(e.ep.name.clone()), None, None),
    )))
}

/// Refuse a `bfloat` scalar or vector of AIR type `ty` as a fragment function's color output of
/// entry `e`. Apple's Metal compiler rejects such a return type ("invalid return type 'bfloat4'
/// (vector of 4 'bfloat' values) for fragment function": an Apple M4 under macOS 27.0.1, into
/// `R16Float`, `RG16Float`, `RGBA16Float` and the three bfloat formats alike),
/// so no fragment function it compiles writes one: `InvalidInput` (`Abi(RoleTypeMismatch)`) on
/// every profile, as [`refuse_int64_interface`]. What Metal's runtime does with AIR made another
/// way is not measured.
pub fn refuse_bfloat_color(e: &Entry, ty: AirTypeId) -> Result<(), Error> {
    let types = e.module.types();
    let scalar = match types.get(ty) {
        Some(Type::Vector { elem, .. }) => types.get(*elem),
        other => other,
    };
    if !matches!(scalar, Some(Type::BFloat)) {
        return Ok(());
    }
    Err(Error::InvalidInput(InputError::new(
        InputErrorKind::Abi(AbiError::RoleTypeMismatch),
        Location::metadata(Some(e.ep.name.clone()), None, None),
    )))
}

/// How profile `p` holds an element held as `h` profile-independently: an integer of 33 to 64
/// bits as [`Held::Wide`] without `shaderInt64`, anything else as it is once `p` can hold its
/// scalar ([`check_scalar`]). The element of a chunked vector takes its form here.
pub fn held_on(p: &DeviceProfile, h: Held) -> Result<Held, Error> {
    match h {
        Held::Int {
            bits,
            container: 64,
        } if holds_wide(p) => Ok(Held::Wide { bits }),
        _ => {
            check_scalar(p, Form { held: h, lanes: 1 }.scalar())?;
            Ok(h)
        }
    }
}

// ---- Constants ----

/// A constant as lowering materializes it, in its lowered representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LoweredConst {
    /// A scalar; integer bits are zero-extended from the AIR width.
    Scalar { ty: ScalarTy, bits: u64 },
    /// A vector, one entry per lane.
    Vector { elem: ScalarTy, lanes: Box<[u64]> },
    /// An aggregate of type `ty`, one part per member.
    Composite {
        ty: DemoteTy,
        parts: Box<[LoweredConst]>,
    },
    /// A top-level `undef` or `poison` (`OpUndef`). Inside an aggregate or vector such a part is
    /// zero.
    Undef(DemoteTy),
}

fn mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// The bits of a scalar constant held as `h` (undef and poison lanes are zero).
fn scalar_bits(value: &Const, h: Held) -> Option<u64> {
    match (value, h) {
        (Const::Int(b), Held::Bool) => Some(b & 1),
        (Const::Int(b), Held::Int { bits, .. }) => Some(b & mask(bits)),
        (Const::Float(b), Held::Float(f)) => Some(f.widen_bits(b & mask(f.bits()))),
        (Const::Null | Const::Undef | Const::Poison, _) => Some(0),
        _ => None,
    }
}

/// The zero value of `d`.
pub fn zero(d: &DemoteTy) -> Option<LoweredConst> {
    Some(match d {
        DemoteTy::Scalar(s) => LoweredConst::Scalar { ty: *s, bits: 0 },
        DemoteTy::Vector { elem, len } => LoweredConst::Vector {
            elem: *elem,
            lanes: vec![0; usize::from(*len)].into(),
        },
        DemoteTy::Composite(parts) => LoweredConst::Composite {
            ty: d.clone(),
            parts: parts.iter().map(zero).collect::<Option<Vec<_>>>()?.into(),
        },
        _ => return None,
    })
}

/// Constant `id` in its lowered representation, or `None` when this family does not
/// materialize it (addresses, constant expressions, types it does not hold).
pub fn lower_const(e: &Entry, id: ConstId) -> Option<LoweredConst> {
    let k = e.module.consts().get(id)?;
    let d = demote_of(e.module.types(), k.ty).ok()?;
    match &k.value {
        Const::Undef | Const::Poison => Some(LoweredConst::Undef(d)),
        _ => lower_part(e, id, 0),
    }
}

/// Constant `id` with undef and poison held as zero.
fn lower_part(e: &Entry, id: ConstId, depth: u32) -> Option<LoweredConst> {
    if depth > MAX_DEPTH {
        return None;
    }
    let types = e.module.types();
    let k = e.module.consts().get(id)?;
    let d = demote_of(types, k.ty).ok()?;
    if matches!(k.value, Const::Null | Const::Undef | Const::Poison) {
        return zero(&d);
    }
    if let Some((held, lanes)) = chunked(types, k.ty) {
        return chunked_const(e, &k.value, held, lanes, d);
    }
    match (&d, &k.value) {
        (DemoteTy::Scalar(_), v) => {
            let h = held_of(types.get(k.ty)?)?;
            Some(LoweredConst::Scalar {
                ty: form_of(types, k.ty).ok()?.scalar(),
                bits: scalar_bits(v, h)?,
            })
        }
        (DemoteTy::Vector { elem, .. }, Const::Aggregate(parts)) => {
            let h = form_of(types, k.ty).ok()?.held;
            let lanes = parts
                .iter()
                .map(|p| scalar_bits(&e.module.consts().get(*p)?.value, h))
                .collect::<Option<Vec<_>>>()?;
            Some(LoweredConst::Vector {
                elem: *elem,
                lanes: lanes.into(),
            })
        }
        (DemoteTy::Composite(_), Const::Aggregate(parts)) => Some(LoweredConst::Composite {
            ty: d.clone(),
            parts: parts
                .iter()
                .map(|p| lower_part(e, *p, depth + 1).map(const_as_member))
                .collect::<Option<Vec<_>>>()?
                .into(),
        }),
        _ => None,
    }
}

/// Constant `k` as an aggregate member ([`as_member`]): a `bool` vector's lanes, 0 and 1, as
/// `u32`s.
pub fn const_as_member(k: LoweredConst) -> LoweredConst {
    match k {
        LoweredConst::Vector {
            elem: ScalarTy::Bool,
            lanes,
        } => LoweredConst::Vector {
            elem: ScalarTy::Int { bits: 32 },
            lanes,
        },
        k => k,
    }
}

/// A constant chunked vector: its lanes grouped into the chunk constants.
fn chunked_const(
    e: &Entry,
    value: &Const,
    held: Held,
    lanes: u32,
    ty: DemoteTy,
) -> Option<LoweredConst> {
    let Const::Aggregate(parts) = value else {
        return None;
    };
    if parts.len() != lanes as usize {
        return None;
    }
    let bits = parts
        .iter()
        .map(|p| scalar_bits(&e.module.consts().get(*p)?.value, held))
        .collect::<Option<Vec<_>>>()?;
    let elem = Form { held, lanes: 1 }.scalar();
    let mut chunks = Vec::new();
    let mut at = 0usize;
    for k in chunk_lanes(lanes) {
        let part = bits.get(at..at + usize::from(k))?;
        chunks.push(if k == 1 {
            LoweredConst::Scalar {
                ty: elem,
                bits: *part.first()?,
            }
        } else {
            LoweredConst::Vector {
                elem,
                lanes: part.into(),
            }
        });
        at += usize::from(k);
    }
    Some(LoweredConst::Composite {
        ty,
        parts: chunks.into(),
    })
}

/// The SPIR-V type of lowered representation `d`, or `None` for one this family does not
/// produce.
pub fn spirv_type(mb: &mut ModuleBuilder, d: &DemoteTy) -> BResult<Option<TypeId>> {
    Ok(Some(match d {
        DemoteTy::Scalar(s) => match scalar_type(mb, *s)? {
            Some(t) => t,
            None => return Ok(None),
        },
        DemoteTy::Vector { elem, len } => {
            let Some(e) = scalar_type(mb, *elem)? else {
                return Ok(None);
            };
            mb.ty(SType::Vector { elem: e, len: *len })?
        }
        DemoteTy::Composite(parts) => {
            let mut members = Vec::with_capacity(parts.len());
            for p in parts.iter() {
                match spirv_type(mb, p)? {
                    Some(t) => members.push(t),
                    None => return Ok(None),
                }
            }
            mb.ty(SType::Struct {
                members: members.into(),
                layout: None,
            })?
        }
        // A pointer member of a member-wise phi's composite is the access family's.
        DemoteTy::FatPtr { .. } | DemoteTy::ArenaOffset | DemoteTy::HandleIndex => {
            return crate::lower::access::spirv_type(mb, d)
        }
    }))
}

fn scalar_type(mb: &mut ModuleBuilder, s: ScalarTy) -> BResult<Option<TypeId>> {
    Ok(Some(match s {
        ScalarTy::Bool => mb.ty(SType::Bool)?,
        ScalarTy::Int {
            bits: bits @ (32 | 64),
        } => mb.ty(SType::Int {
            width: bits,
            signed: false,
        })?,
        ScalarTy::Float {
            bits: bits @ (16 | 32 | 64),
        } => mb.ty(SType::Float { width: bits })?,
        _ => return Ok(None),
    }))
}

/// The module-scope constant of `k`, or `None` for `undef` or a type this family does not
/// produce.
fn const_id(mb: &mut ModuleBuilder, k: &LoweredConst) -> BResult<Option<SConstId>> {
    Ok(Some(match k {
        LoweredConst::Scalar { ty, bits } => {
            let Some(t) = scalar_type(mb, *ty)? else {
                return Ok(None);
            };
            mb.constant(Constant::Scalar {
                ty: t,
                bits: ConstBits(*bits),
            })?
        }
        LoweredConst::Vector { elem, lanes } => {
            let Some(e) = scalar_type(mb, *elem)? else {
                return Ok(None);
            };
            let len = u8::try_from(lanes.len()).unwrap_or(u8::MAX);
            let ty = mb.ty(SType::Vector { elem: e, len })?;
            let mut parts = Vec::with_capacity(lanes.len());
            for b in lanes.iter() {
                parts.push(mb.constant(Constant::Scalar {
                    ty: e,
                    bits: ConstBits(*b),
                })?);
            }
            mb.constant(Constant::Composite {
                ty,
                parts: parts.into(),
            })?
        }
        LoweredConst::Composite { ty, parts } => {
            let Some(t) = spirv_type(mb, ty)? else {
                return Ok(None);
            };
            let mut ids = Vec::with_capacity(parts.len());
            for p in parts.iter() {
                match const_id(mb, p)? {
                    Some(c) => ids.push(c),
                    None => return Ok(None),
                }
            }
            mb.constant(Constant::Composite {
                ty: t,
                parts: ids.into(),
            })?
        }
        LoweredConst::Undef(_) => return Ok(None),
    }))
}

/// Materialize `k` in the open function, or `None` for one this family does not produce.
pub fn materialize(fb: &mut FunctionBuilder<'_>, k: &LoweredConst) -> BResult<Option<SValue>> {
    if let LoweredConst::Undef(d) = k {
        let Some(t) = spirv_type(fb.module(), d)? else {
            return Ok(None);
        };
        return fb.undef(t).map(Some);
    }
    match const_id(fb.module(), k)? {
        Some(c) => fb.module().const_value(c).map(Some),
        None => Ok(None),
    }
}
