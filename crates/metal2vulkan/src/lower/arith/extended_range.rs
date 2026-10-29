//! Apple's extended-range texels (`metal::format::ExtendedRange`): Metal's value of each raw code a
//! slot reads, and the code of each value it writes, exact on every device.
//!
//! A channel's 10-bit code `c` holds `(c - 384) / 510`: a read answers it correctly rounded
//! ([`texel_round::ratio_to_f32`](super::texel_round::ratio_to_f32) of `|c - 384|` over 1020,
//! doubled), alpha clamped to `[0, 1]` first, and a `half` result rounds that value with ties away
//! from zero ([`decode`]); a write rounds the exact product of the value with 510 to nearest even
//! ([`math::round_exact_product_lanes`](super::math::round_exact_product_lanes), the magnitude
//! below 1.5 first) and clamps the code (alpha to `[384, 894]`), a NaN 0. An `_sRGB` format's color
//! channel goes through Metal's own tables, symmetric about code 384 ([`declare`]): a read answers
//! `s / 8192` for the step `s` of `|c - 384|`, signed; a write rounds the magnitude (below 2 first)
//! to `2^-13`, nearest even (an exact product and `RoundEven`), and the offset is the largest whose
//! first index is at most that ([`search`]), at most 384 below the zero code. Both tables were read
//! from an Apple M3 Max and an Apple M4 (byte-identical): every code read, and every threshold
//! between two adjacent codes found by bisection over the `f32`s, every half, and a dense grid
//! written. The MSL transfer `((x + 0.055) / 1.055)^2.4` rounded to `2^-13` gives every decode
//! step; it misses 32 of the 639 encode thresholds, which Metal places by its own arithmetic.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math::round_exact_product_lanes;
use crate::lower::arith::ops::{fsplat, splat};
use crate::lower::arith::texel_round::{f32_to_half, ratio_to_f32, HalfRounding};
use crate::metal::format::ExtendedRange;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type};

pub mod blend;
pub(crate) mod blend_color;
pub use blend::Blend;
pub(crate) mod table;
pub use table::{declare, declare_padded, search};
use table::{word, TABLE_LEN};

/// The census name the table is charged to ([`TABLE_WORDS`]), once per module that declares it
/// ([`ArithPlan::extended_range_srgb`](super::ArithPlan::extended_range_srgb)).
pub const CENSUS_NAME: &str = "texture.extended_range_srgb";

/// An upper estimate of the SPIR-V words of the table ([`declare`]): its distinct constants,
/// the composite, and the variable, charged once per module.
pub const TABLE_WORDS: u32 = 641 * 4 + TABLE_LEN + 3 + 8;

/// The census name a padded `_sRGB` form's table is charged to ([`PADDED_TABLE_WORDS`]), once
/// per module and form that declares it
/// ([`ArithPlan::padded_srgb`](super::ArithPlan::padded_srgb)).
pub const PADDED_CENSUS_NAME: &str = "texture.padded_srgb";

/// An upper estimate of the SPIR-V words of a padded `_sRGB` form's table ([`declare_padded`]):
/// at most one distinct constant per code, the composite, and the variable.
pub const PADDED_TABLE_WORDS: u32 = TABLE_LEN * 4 + TABLE_LEN + 3 + 8;

/// An upper estimate of the SPIR-V words of one texel's [`decode`]: the codes, the quotients'
/// exact sequence on four lanes, the signs, and three table reads.
pub const DECODE_WORDS: u32 = 140 * 6;

/// An upper estimate of the SPIR-V words of one texel's [`encode`]: the clean value, the exact
/// products on four lanes, the codes, three searches of ten steps, and the packing.
pub const ENCODE_WORDS: u32 = 220 * 6;

/// Four `u32` lanes.
fn words() -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 4,
    }
}

/// Four `f32` lanes.
fn floats() -> Form {
    words().with(Held::Float(FloatFmt::F32))
}

/// The four codes of raw texel `raw` (the raw format's `uvec4` as a read gives it) in Metal's
/// channel order, red first; `BGR10_XR`'s alpha is the code of 1, and `BGRA10_XR`'s is clamped
/// to `[384, 894]`, the codes of `[0, 1]`. A two-plane texel's alpha is its alpha byte (the
/// second component), an 8-bit unorm code.
pub(crate) fn codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    match form {
        ExtendedRange::Bgr10 { .. } | ExtendedRange::Planes { .. } => {
            let w = fb.composite_extract(raw, &[0])?;
            let mut lanes = Vec::with_capacity(4);
            for shift in [20u32, 10, 0] {
                let s = fb.u32_const(shift)?;
                let s = fb.module().const_value(s)?;
                let x = fb.binary(BinOp::ShiftRightLogical, w, s)?;
                let m = fb.u32_const(0x3FF)?;
                let m = fb.module().const_value(m)?;
                lanes.push(fb.binary(BinOp::BitwiseAnd, x, m)?);
            }
            if let ExtendedRange::Planes { .. } = form {
                let a = fb.composite_extract(raw, &[1])?;
                let m = fb.u32_const(0xFF)?;
                let m = fb.module().const_value(m)?;
                lanes.push(fb.binary(BinOp::BitwiseAnd, a, m)?);
            } else {
                let one = fb.u32_const(894)?;
                lanes.push(fb.module().const_value(one)?);
            }
            fb.composite_construct(ut, &lanes)
        }
        ExtendedRange::Bgra10 { .. } => {
            let six = splat(fb, u, 6)?;
            let c = fb.binary(BinOp::ShiftRightLogical, raw, six)?;
            // Red is the third field, blue the first.
            let c = fb.vector_shuffle(c, c, &[2, 1, 0, 3])?;
            let lo = per_lane(fb, [0, 0, 0, 384])?;
            let hi = per_lane(fb, [1023, 1023, 1023, 894])?;
            fb.glsl(GlslOp::UClamp, &[c, lo, hi])
        }
        // A padded format's words hold each code in their top 10 bits; blue is 0 and alpha the
        // code of 1 (`value_of_codes` divides every lane by 1023).
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb => {
            let six = splat(fb, u, 6)?;
            let c = fb.binary(BinOp::ShiftRightLogical, raw, six)?;
            let fill = per_lane(fb, [0, 0, 0, 1023])?;
            fb.vector_shuffle(c, fill, padded_lanes(form))
        }
        // `RGB10A2Unorm_sRGB`'s word: red, green and blue in 10-bit fields from bit 0, alpha's
        // 2-bit code in bits 30 and 31.
        ExtendedRange::Rgb10A2Srgb => {
            let w = fb.composite_extract(raw, &[0])?;
            let w = fb.composite_construct(ut, &[w, w, w, w])?;
            let shifts = per_lane(fb, [0, 10, 20, 30])?;
            let c = fb.binary(BinOp::ShiftRightLogical, w, shifts)?;
            let m = per_lane(fb, [0x3FF, 0x3FF, 0x3FF, 0x3])?;
            fb.binary(BinOp::BitwiseAnd, c, m)
        }
        // A bfloat form has no codes of this kind: its texels convert by [`bfloat_value`] and
        // [`bfloat_texel`], and no attachment of one is held.
        ExtendedRange::R16Bfloat | ExtendedRange::Rg16Bfloat | ExtendedRange::Rgba16Bfloat => Err(
            crate::spirv::error::BuildError::Type(crate::spirv::error::TypeRule::FunctionSignature),
        ),
    }
}

/// The shuffle that keeps a padded form's channels of a first vector and takes the rest from a
/// second: red alone, or red and green.
fn padded_lanes(form: ExtendedRange) -> &'static [u32] {
    match form.raw_components() {
        1 => &[0, 5, 6, 7],
        _ => &[0, 1, 6, 7],
    }
}

/// A padded `_sRGB` form's conversion without its table ([`declare_padded`]), which its
/// selection declares (`ArithPlan::converts`).
fn missing_table() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Operand {
        op: crate::spv::Op::AccessChain,
        rule: crate::spirv::error::OperandRule::NotPointer,
    }
}

/// A `uvec4` constant of the four lanes `v`.
fn per_lane(fb: &mut FunctionBuilder<'_>, v: [u32; 4]) -> BResult<Value> {
    let ut = crate::lower::arith::ops::ty(fb, words())?;
    let e = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let mut parts = Vec::with_capacity(4);
    for x in v {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(u64::from(x)),
        })?);
    }
    let c = fb.module().constant(Constant::Composite {
        ty: ut,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// Metal's value of raw texel `raw` of a slot bound in `form` (the raw format's `uvec4`), a
/// `vec4` of `f32`, red first. `table` is the module's ([`declare`]) for an `_sRGB` form, whose
/// color channels then read through it, and `None` for a linear one. For a `half` result
/// (`half`) each lane is rounded to `half` with ties away from zero, held as the `f32` of that
/// value: Metal rounds every tie so (an `_sRGB` color step `s / 8192` of an odd multiple of half
/// a `half` ulp: each of the 13,458 ties of `BGRA10_XR_sRGB`'s and 25,502 of `BGR10_XR_sRGB`'s
/// reads), and a linear lane's `(c - 384) / 510` is never a tie.
pub fn decode(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    table: Option<GlobalId>,
    half: bool,
) -> BResult<Value> {
    if form.bfloat() {
        return bfloat_value(fb, form, raw, half);
    }
    let c = codes(fb, form, raw)?;
    value_of_codes(fb, form, c, table, half)
}

/// An upper estimate of the SPIR-V words of one texel's [`steps`]: the codes, the shifted codes
/// less the zero code's, and an `_sRGB` twin's three signed table steps.
pub const STEPS_WORDS: u32 = 48 * 6;

/// Whether a texel of `form` has [`steps`]: every form but a padded or bfloat one, whose lanes
/// the exact sums read on other grids or not at all.
pub fn has_steps(form: ExtendedRange) -> bool {
    !form.padded() && !form.bfloat()
}

/// The grid steps of raw texel `raw` of a slot bound in `form` (one that [`has_steps`]), as `i32`
/// bits on four `u32` lanes, red first: the integers an exact sum (`BicubicTexels::EXTENDED`)
/// recovers from [`decode`]'s value `v` as `RoundEven(v · D)`, taken from the codes alone. On a
/// color lane's grid `16 · 510` a code `c` is `16 (c - 384)` (`v = (c - 384) / 510`), on an
/// `_sRGB` twin's grid `8192` its decode step `±s` (`v = ±s / 8192`), and on alpha's grid
/// `8 · 510` a two-plane texel's byte `b` is `16 b` (`v = b / 255`) and an alpha code `8 (c -
/// 384)` (`BGR10_XR`'s the code of 1). `table` is the module's ([`declare`]) for an `_sRGB` form.
pub fn steps(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    let c = codes(fb, form, raw)?;
    let planes = matches!(form, ExtendedRange::Planes { .. });
    let (shift, zero) = match planes {
        true => ([4, 4, 4, 4], [16 * 384, 16 * 384, 16 * 384, 0]),
        false => ([4, 4, 4, 3], [16 * 384, 16 * 384, 16 * 384, 8 * 384]),
    };
    let shift = per_lane(fb, shift)?;
    let zero = per_lane(fb, zero)?;
    let up = fb.binary(BinOp::ShiftLeftLogical, c, shift)?;
    let linear = fb.binary(BinOp::ISub, up, zero)?;
    let Some(table) = table else {
        return Ok(linear);
    };
    // An `_sRGB` twin's color lane: the step of `|c - 384|` in the table, negated below 384.
    let ut = crate::lower::arith::ops::ty(fb, words())?;
    let mask = fb.u32_const(0xFFFF)?;
    let mask = fb.module().const_value(mask)?;
    let at = fb.u32_const(384)?;
    let at = fb.module().const_value(at)?;
    let mut lanes = Vec::with_capacity(4);
    for lane in 0..3u32 {
        let code = fb.composite_extract(c, &[lane])?;
        let neg = fb.binary(BinOp::ULessThan, code, at)?;
        let up = fb.binary(BinOp::ISub, code, at)?;
        let down = fb.binary(BinOp::ISub, at, code)?;
        let n = fb.select(neg, down, up)?;
        let w = word(fb, table, n)?;
        let step = fb.binary(BinOp::BitwiseAnd, w, mask)?;
        let minus = fb.unary(crate::spirv::ir::UnOp::SNegate, step)?;
        lanes.push(fb.select(neg, minus, step)?);
    }
    lanes.push(fb.composite_extract(linear, &[3])?);
    fb.composite_construct(ut, &lanes)
}

/// The grids of [`steps`] per lane, red first, as `f32`: `16 · 510` (an `_sRGB` twin's `8192`)
/// on the color lanes and `8 · 510` on alpha, by which a border texel's value (0 or 1 per lane)
/// is its steps.
pub fn step_grids(fb: &mut FunctionBuilder<'_>, form: ExtendedRange) -> BResult<Value> {
    let color = match form.srgb() {
        true => 8192.0,
        false => 16.0 * 510.0,
    };
    per_lane_f(fb, [color, color, color, 8.0 * 510.0])
}

/// Metal's tile value of raw texel `raw` of a color attachment of `form` (a framebuffer fetch's
/// read; byte-identical, a `float` and a `half` read alike): a `vec4` of
/// `f32`, red first, each lane a `half` value. The tile holds no color code above 1022: a linear
/// form's code and `BGRA10_XR_sRGB`'s are clamped to 1022 and decoded ([`decode`]),
/// `BGR10_XR_sRGB`'s and the two-plane twin's decoded and then clamped to the linear values of
/// codes 0 and 1022, `[-384 / 510, 638 / 510]`; each lane is rounded to `half` with ties away from
/// zero (only an `_sRGB` step can tie). Alpha is [`decode`]'s.
pub fn tile_value(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    let f = floats();
    let c = codes(fb, form, raw)?;
    let by_code = !form.srgb() || matches!(form, ExtendedRange::Bgra10 { .. });
    let c = match by_code {
        true => {
            let top = per_lane(fb, [1022, 1022, 1022, u32::MAX])?;
            fb.glsl(GlslOp::UMin, &[c, top])?
        }
        false => c,
    };
    let v = value_of_codes(fb, form, c, table, false)?;
    let v = match by_code {
        true => v,
        false => {
            let lo = (-384.0f64 / 510.0) as f32;
            let hi = (638.0f64 / 510.0) as f32;
            let lo = per_lane_f(fb, [lo, lo, lo, f32::NEG_INFINITY])?;
            let hi = per_lane_f(fb, [hi, hi, hi, f32::INFINITY])?;
            fb.glsl(GlslOp::FClamp, &[v, lo, hi])?
        }
    };
    f32_to_half(fb, f, v, HalfRounding::NearestAway)
}

/// Metal's read of the tile memory of an attachment of `form` that holds `held` (a `vec4` of
/// `f32`, red first, each lane a `half`: what the pass's clear, load or last write left there,
/// the `half` shadow's texel): a fetch's value, and a blend's destination. Its color lanes are
/// clamped to the blend's source range (`BGRA10_XR_sRGB`'s [`SRGB_SOURCE_RANGE`], every other
/// form's [`SOURCE_RANGE`]), alpha to `[0, 1]`, a NaN to the low end (`blend::clamped`, the
/// blend's own clamp of its source), and `BGR10_XR`'s alpha is one: the tile holds what a write
/// wrote, unclamped, and stores its encode, but every read clamps (on an Apple
/// M4: unblended `half` writes past the range, NaNs and the infinities, each read back by a
/// fetch and blended over, for every form; their codes are the writes' own encodes, code 1023
/// for 2.0 where the read is 1022's value). [`tile_value`] of a texel's codes is such a read
/// already.
pub fn tile_read(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, held: Value) -> BResult<Value> {
    let read = blend::clamped(fb, held, source_range(form))?;
    match form {
        ExtendedRange::Bgr10 { .. } => {
            let one = fsplat(fb, floats(), 1.0)?;
            fb.vector_shuffle(read, one, &[0, 1, 2, 7])
        }
        _ => Ok(read),
    }
}

pub(crate) use blend_color::{SOURCE_RANGE, SRGB_SOURCE_RANGE};

/// The range Metal clamps a blend's `half` source color to before blending into an attachment
/// of `form` ([`SOURCE_RANGE`], [`SRGB_SOURCE_RANGE`]).
fn source_range(form: ExtendedRange) -> (f32, f32) {
    match form {
        ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
        _ => SOURCE_RANGE,
    }
}

/// [`decode`]'s value of the codes `c` ([`codes`]) of `form`.
fn value_of_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    c: Value,
    table: Option<GlobalId>,
    half: bool,
) -> BResult<Value> {
    let (u, f) = (words(), floats());
    // A padded format's lane reads `c / 1023` correctly rounded, its `half` to nearest even
    // (every code; `c / 1023` is never a `half` tie), `RG10Unorm_X12_sRGB`'s too.
    // `R10Unorm_X6_sRGB`'s red reads its code's decode step `s / 8192` (every code), its `half`
    // with ties away from zero (every code); so do `RGB10A2Unorm_sRGB`'s red, green and blue,
    // and its alpha `a / 3` correctly rounded, never a `half` tie (an Apple M3 Max, every code).
    if form.padded() {
        let d = match form {
            ExtendedRange::Rgb10A2Srgb => per_lane(fb, [1023, 1023, 1023, 3])?,
            _ => splat(fb, u, 1023)?,
        };
        let v = ratio_to_f32(fb, u, c, d)?;
        let (v, rounding) = match form {
            ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb => {
                let table = table.ok_or_else(missing_table)?;
                let colors = srgb_channels(form);
                let ft = crate::lower::arith::ops::ty(fb, f)?;
                let mut lanes = Vec::with_capacity(4);
                for lane in 0..4u32 {
                    lanes.push(match lane < colors {
                        true => {
                            let code = fb.composite_extract(c, &[lane])?;
                            table_step(fb, table, code)?
                        }
                        false => fb.composite_extract(v, &[lane])?,
                    });
                }
                let v = fb.composite_construct(ft, &lanes)?;
                (v, HalfRounding::NearestAway)
            }
            _ => (v, HalfRounding::NearestEven),
        };
        return match half {
            true => f32_to_half(fb, f, v, rounding),
            false => Ok(v),
        };
    }
    // A two-plane texel's alpha byte `b` reads `b / 255`: its zero code is 0, and its quotient
    // is over 255, not doubled.
    let planes = matches!(form, ExtendedRange::Planes { .. });
    let zero_code = match planes {
        true => per_lane(fb, [384, 384, 384, 0])?,
        false => splat(fb, u, 384)?,
    };
    let neg = fb.binary(BinOp::ULessThan, c, zero_code)?;
    let up = fb.binary(BinOp::ISub, c, zero_code)?;
    let down = fb.binary(BinOp::ISub, zero_code, c)?;
    let n = fb.select(neg, down, up)?;
    // |c - 384| / 510, correctly rounded: the quotient over 1020 (its operands within the
    // routine's `k <= d`), doubled exactly.
    let (d, two) = match planes {
        true => (
            per_lane(fb, [1020, 1020, 1020, 255])?,
            per_lane_f(fb, [2.0, 2.0, 2.0, 1.0])?,
        ),
        false => (splat(fb, u, 1020)?, fsplat(fb, f, 2.0)?),
    };
    let q = ratio_to_f32(fb, u, n, d)?;
    let mut v = fb.binary(BinOp::FMul, q, two)?;
    if let Some(table) = table {
        let ft = crate::lower::arith::ops::ty(fb, f)?;
        let mut lanes = Vec::with_capacity(4);
        for lane in 0..3u32 {
            let i = fb.composite_extract(n, &[lane])?;
            lanes.push(table_step(fb, table, i)?);
        }
        lanes.push(fb.composite_extract(v, &[3])?);
        v = fb.composite_construct(ft, &lanes)?;
    }
    let minus = fb.unary(crate::spirv::ir::UnOp::FNegate, v)?;
    let v = fb.select(neg, minus, v)?;
    match half {
        true => f32_to_half(fb, f, v, HalfRounding::NearestAway),
        false => Ok(v),
    }
}

/// The value `s / 8192` of the decode step `s` in the low half of word `i` (a `u32` scalar) of
/// `table`, exact.
fn table_step(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let f1 = fb.module().ty(Type::Float { width: 32 })?;
    let mask = fb.u32_const(0xFFFF)?;
    let mask = fb.module().const_value(mask)?;
    let scale = fb.module().constant(Constant::Scalar {
        ty: f1,
        bits: ConstBits(u64::from((1.0f32 / 8192.0).to_bits())),
    })?;
    let scale = fb.module().const_value(scale)?;
    let w = word(fb, table, i)?;
    let s = fb.binary(BinOp::BitwiseAnd, w, mask)?;
    let s = fb.convert(ConvOp::ConvertUToF, s, f1)?;
    fb.binary(BinOp::FMul, s, scale)
}

/// The raw texel (the raw format's `uvec4`) a write of `v` (a `vec4` of `f32`, red first) to a
/// slot bound in `form` stores, as Metal writes it. `table` is the module's ([`declare`]) for an
/// `_sRGB` form, whose color channels then encode through it, and `None` for a linear one.
pub fn encode(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    // A padded format's code is the exact product of the value clamped to [0, 1] with 1023,
    // rounded to nearest (every value of a grid and around every midpoint), in
    // its word's top 10 bits, the low 6 bits zero.
    if form.padded() {
        let u = words();
        let k = padded_codes(fb, form, v, table)?;
        if form == ExtendedRange::Rgb10A2Srgb {
            return pack(fb, form, k);
        }
        let six = splat(fb, u, 6)?;
        let w = fb.binary(BinOp::ShiftLeftLogical, k, six)?;
        let z = splat(fb, u, 0)?;
        return fb.vector_shuffle(w, z, padded_lanes(form));
    }
    if form.bfloat() {
        return bfloat_texel(fb, form, v);
    }
    let code = linear_codes(fb, form, v, table)?;
    pack(fb, form, code)
}

/// The codes a write of `v` (a `vec4` of `f32`, red first) to a slot of a padded form stores
/// ([`encode`]): the exact product of the value, clamped to `[0, 1]` (a NaN 0), with 1023 (alpha's
/// with 3), rounded to nearest, an `_sRGB` form's color codes through its `table`.
fn padded_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    let f = floats();
    let zero = fsplat(fb, f, 0.0)?;
    let nan = fb.unary(crate::spirv::ir::UnOp::IsNan, v)?;
    let v = fb.select(nan, zero, v)?;
    let one = fsplat(fb, f, 1.0)?;
    let v = fb.glsl(GlslOp::FClamp, &[v, zero, one])?;
    let scales: &[u32] = match form {
        ExtendedRange::Rgb10A2Srgb => &[1023, 1023, 1023, 3],
        _ => &[1023],
    };
    let k = round_exact_product_lanes(fb, f, v, scales, false)?;
    match form {
        ExtendedRange::R10X6Srgb | ExtendedRange::Rg10X12Srgb | ExtendedRange::Rgb10A2Srgb => {
            let table = table.ok_or_else(missing_table)?;
            padded_srgb_codes(fb, form, table, v, k)
        }
        _ => Ok(k),
    }
}

/// The codes (red first) a write of `v` (a `vec4` of `f32`, red first) to a slot of `form` stores,
/// as [`encode`] packs them: a linear or `_sRGB` form's color codes and alpha's, or a padded
/// form's. `table` is the form's ([`declare`], [`declare_padded`]) where it has one.
pub(crate) fn write_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    match form.padded() {
        true => padded_codes(fb, form, v, table),
        false => linear_codes(fb, form, v, table),
    }
}

/// The lanes a bfloat form keeps of a first vector, the rest taken from a second: red; red and
/// green; or all four.
fn bfloat_lanes(form: ExtendedRange) -> &'static [u32] {
    match form.raw_components() {
        1 => &[0, 5, 6, 7],
        2 => &[0, 1, 6, 7],
        _ => &[0, 1, 2, 3],
    }
}

/// Metal's value of raw texel `raw` of a bfloat form ([`decode`]; `metal::format::ExtendedRange`'s
/// `R16Bfloat`): each channel's code shifted up 16 bits as a `float`, any NaN code (its magnitude
/// past `0x7f80`) the quiet NaN `0x7fc00000`; a channel the form lacks 0 and alpha 1; a `half`
/// result of a read or a gather each lane rounded to nearest even (an Apple M4: every code of each
/// bfloat format, `float` and `half` reads, `read_write`, nearest and linear samples alike;
/// gathers); a sample's `half` result rounds toward zero instead ([`bfloat_sample_half`]). Integer
/// operations on the bits alone, so no float mode changes the answer.
fn bfloat_value(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    half: bool,
) -> BResult<Value> {
    let (u, f) = (words(), floats());
    let ft = crate::lower::arith::ops::ty(fb, f)?;
    let low = splat(fb, u, 0x7FFF)?;
    let mag = fb.binary(BinOp::BitwiseAnd, raw, low)?;
    let inf = splat(fb, u, 0x7F80)?;
    let nan = fb.binary(BinOp::UGreaterThan, mag, inf)?;
    let sixteen = splat(fb, u, 16)?;
    let bits = fb.binary(BinOp::ShiftLeftLogical, raw, sixteen)?;
    let quiet = splat(fb, u, 0x7FC0_0000)?;
    let bits = fb.select(nan, quiet, bits)?;
    let fill = per_lane(fb, [0, 0, 0, 1.0f32.to_bits()])?;
    let bits = fb.vector_shuffle(bits, fill, bfloat_lanes(form))?;
    let v = fb.convert(ConvOp::Bitcast, bits, ft)?;
    match half {
        true => f32_to_half(fb, f, v, HalfRounding::NearestEven),
        false => Ok(v),
    }
}

/// `v`, a bfloat sample's `vec4` of `float` values ([`decode`] without `half`, or the border
/// texel), as the sample's `half` result: each lane rounded toward zero, a finite value past
/// `65504` saturating there, an infinity and a NaN kept (an Apple M4 under macOS
/// 27.0.1: 18,432 positions of an 8 x 4 texture of each bfloat format holding NaN, infinite,
/// overflowing and `half`-denormal codes, nearest and linear filters under every address mode,
/// each `half` result the `float` result's toward zero, where a read's and a gather's round to
/// nearest even). Integer operations on the bits alone.
pub(crate) fn bfloat_sample_half(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    f32_to_half(fb, floats(), v, HalfRounding::TowardZeroSaturating)
}

/// The raw texel a kernel's write of `v` (a `vec4` of `f32`; a `half` value held widened)
/// stores to a bfloat form ([`encode`]): each lane's top 16 bits (toward zero), a NaN the quiet
/// NaN `0x7fc0`, the components past the form's zero (an Apple M4: per bfloat code
/// the `float`s of its bits and three neighbours of each tie, and every `half` code, into each
/// bfloat format). Integer operations on the bits alone.
fn bfloat_texel(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, v: Value) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let abs = splat(fb, u, 0x7FFF_FFFF)?;
    let mag = fb.binary(BinOp::BitwiseAnd, bits, abs)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let nan = fb.binary(BinOp::UGreaterThan, mag, inf)?;
    let sixteen = splat(fb, u, 16)?;
    let code = fb.binary(BinOp::ShiftRightLogical, bits, sixteen)?;
    let quiet = splat(fb, u, 0x7FC0)?;
    let code = fb.select(nan, quiet, code)?;
    let zero = splat(fb, u, 0)?;
    fb.vector_shuffle(code, zero, bfloat_lanes(form))
}

/// The raw texel a render pass stores to an attachment of the bfloat form `form` whose tile holds
/// `v` (a `vec4` of `half` values held as `f32`; `stage_io::extended_color::bfloat_output`): each
/// lane's bfloat code nearest, ties to even, a NaN `0x7fc0`, the components past the form's zero
/// (an Apple M4 under macOS 27.0.1: every `half` code output into each bfloat
/// format). Not a kernel write's code, which drops the low 16 bits ([`bfloat_texel`]). Integer
/// operations on the bits alone; no sum wraps (the largest magnitude below a NaN's is the
/// infinity's).
pub(crate) fn bfloat_stored(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let abs = splat(fb, u, 0x7FFF_FFFF)?;
    let mag = fb.binary(BinOp::BitwiseAnd, bits, abs)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let nan = fb.binary(BinOp::UGreaterThan, mag, inf)?;
    let sixteen = splat(fb, u, 16)?;
    let one = splat(fb, u, 1)?;
    let kept = fb.binary(BinOp::ShiftRightLogical, bits, sixteen)?;
    let odd = fb.binary(BinOp::BitwiseAnd, kept, one)?;
    let half_step = splat(fb, u, 0x7FFF)?;
    let biased = fb.binary(BinOp::IAdd, bits, half_step)?;
    let biased = fb.binary(BinOp::IAdd, biased, odd)?;
    let code = fb.binary(BinOp::ShiftRightLogical, biased, sixteen)?;
    let quiet = splat(fb, u, 0x7FC0)?;
    let code = fb.select(nan, quiet, code)?;
    let zero = splat(fb, u, 0)?;
    fb.vector_shuffle(code, zero, bfloat_lanes(form))
}

/// [`encode`] of `v` into a texel of a linear or `_sRGB` form that keeps the channels `kept`
/// (red, green, blue, alpha) of `old`, the raw texel the write overwrites, as Metal's blended
/// write under a write mask keeps them (every masked configuration, each texel): each kept
/// channel's code as [`decode`] reads it (a color code as it is, 1023 too; `BGRA10_XR`'s alpha
/// clamped to `[384, 894]`; a two-plane texel's alpha byte), packed as a write packs it: plane 0's
/// and `BGR10_XR`'s top two bits as the write sets them (zeros and ones), `BGRA10_XR`'s six low
/// bits zero.
pub fn encode_keeping(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    table: Option<GlobalId>,
    (old, kept): (Value, [bool; 4]),
) -> BResult<Value> {
    let new = linear_codes(fb, form, v, table)?;
    let old = codes(fb, form, old)?;
    let lanes: Vec<u32> = kept
        .iter()
        .zip(0..4u32)
        .map(|(&kept, k)| match kept {
            true => 4 + k,
            false => k,
        })
        .collect();
    let code = fb.vector_shuffle(new, old, &lanes)?;
    pack(fb, form, code)
}

/// The codes a write of `v` (a `vec4` of `f32`, red first) to a slot of a linear or `_sRGB` form
/// stores ([`encode`]): each color code, and alpha's (a two-plane form's alpha byte).
fn linear_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    let (u, f) = (words(), floats());
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let zero = fsplat(fb, f, 0.0)?;
    let nan = fb.unary(crate::spirv::ir::UnOp::IsNan, v)?;
    let v = fb.select(nan, zero, v)?;
    // Alpha's value clamped to [0, 1], whose codes are [384, 894].
    let lo = per_lane_f(fb, [-2.0, -2.0, -2.0, 0.0])?;
    let hi = per_lane_f(fb, [2.0, 2.0, 2.0, 1.0])?;
    let v = fb.glsl(GlslOp::FClamp, &[v, lo, hi])?;
    let neg = fb.binary(BinOp::FOrdLessThan, v, zero)?;
    let a = fb.glsl(GlslOp::FAbs, &[v])?;
    let cap = fsplat(fb, f, 1.5)?;
    let m = fb.glsl(GlslOp::FMin, &[a, cap])?;
    // A two-plane texel's alpha byte is the exact product with 255, rounded to nearest even.
    let planes = matches!(form, ExtendedRange::Planes { .. });
    let scales: &[u32] = match planes {
        true => &[510, 510, 510, 255],
        false => &[510],
    };
    let k = round_exact_product_lanes(fb, f, m, scales, false)?;
    let zero_code = splat(fb, u, 384)?;
    let up = fb.binary(BinOp::IAdd, zero_code, k)?;
    let down = fb.binary(BinOp::ISub, zero_code, k)?;
    // A negative code wraps past 2^31: clamp the down codes as signed.
    let i32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let it = fb.module().ty(Type::Vector { elem: i32_, len: 4 })?;
    let down = fb.convert(ConvOp::Bitcast, down, it)?;
    let iz = fb.module().constant(Constant::Null { ty: it })?;
    let iz = fb.module().const_value(iz)?;
    let down = fb.glsl(GlslOp::SMax, &[down, iz])?;
    let down = fb.convert(ConvOp::Bitcast, down, ut)?;
    let top = splat(fb, u, 1023)?;
    let up = fb.glsl(GlslOp::UMin, &[up, top])?;
    let mut code = fb.select(neg, down, up)?;
    // A two-plane texel's alpha lane holds its byte.
    if planes {
        code = fb.vector_shuffle(code, k, &[0, 1, 2, 7])?;
    }
    if let Some(table) = table {
        let f1 = fb.module().ty(Type::Float { width: 32 })?;
        let u1 = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let grid = fb.module().constant(Constant::Scalar {
            ty: f1,
            bits: ConstBits(u64::from(8192.0f32.to_bits())),
        })?;
        let grid = fb.module().const_value(grid)?;
        let k384 = fb.u32_const(384)?;
        let k384 = fb.module().const_value(k384)?;
        let mut lanes = Vec::with_capacity(4);
        for lane in 0..3u32 {
            // |x| below 2 times 2^13 is exact; RoundEven is exact.
            let x = fb.composite_extract(a, &[lane])?;
            let x = fb.binary(BinOp::FMul, x, grid)?;
            let x = fb.glsl(GlslOp::RoundEven, &[x])?;
            let i = fb.convert(ConvOp::ConvertFToU, x, u1)?;
            let n = search(fb, table, i)?;
            let n_down = fb.glsl(GlslOp::UMin, &[n, k384])?;
            let up = fb.binary(BinOp::IAdd, k384, n)?;
            let down = fb.binary(BinOp::ISub, k384, n_down)?;
            let is_neg = fb.composite_extract(neg, &[lane])?;
            lanes.push(fb.select(is_neg, down, up)?);
        }
        lanes.push(fb.composite_extract(code, &[3])?);
        code = fb.composite_construct(ut, &lanes)?;
    }
    Ok(code)
}

/// The raw texel (the raw format's `uvec4`) of the codes `code` (red first; a two-plane form's
/// alpha its byte) of a linear or `_sRGB` form, or of `RGB10A2Unorm_sRGB`.
pub(crate) fn pack(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    code: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let planes = matches!(form, ExtendedRange::Planes { .. });
    match form {
        ExtendedRange::Bgr10 { .. } | ExtendedRange::Planes { .. } => {
            // Bits 30 and 31 are written as ones by `BGR10_XR` and as zeros by plane 0 of the
            // two-plane format, as Metal writes them.
            let top = match planes {
                true => fb.u32_const(0)?,
                false => fb.u32_const(0xC000_0000)?,
            };
            let mut w = fb.module().const_value(top)?;
            for (lane, shift) in [(0u32, 20u32), (1, 10), (2, 0)] {
                let c = fb.composite_extract(code, &[lane])?;
                let s = fb.u32_const(shift)?;
                let s = fb.module().const_value(s)?;
                let c = fb.binary(BinOp::ShiftLeftLogical, c, s)?;
                w = fb.binary(BinOp::BitwiseOr, w, c)?;
            }
            let z = fb.u32_const(0)?;
            let z = fb.module().const_value(z)?;
            let alpha = match planes {
                true => fb.composite_extract(code, &[3])?,
                false => z,
            };
            fb.composite_construct(ut, &[w, alpha, z, z])
        }
        ExtendedRange::Bgra10 { .. } => {
            // Blue's field first; each code in its field's top 10 bits.
            let c = fb.vector_shuffle(code, code, &[2, 1, 0, 3])?;
            let six = splat(fb, u, 6)?;
            fb.binary(BinOp::ShiftLeftLogical, c, six)
        }
        // `RGB10A2Unorm_sRGB`'s fields from bit 0, alpha in bits 30 and 31.
        ExtendedRange::Rgb10A2Srgb => {
            let shifts = per_lane(fb, [0, 10, 20, 30])?;
            let k = fb.binary(BinOp::ShiftLeftLogical, code, shifts)?;
            let mut w = fb.composite_extract(k, &[0])?;
            for lane in 1..4u32 {
                let x = fb.composite_extract(k, &[lane])?;
                w = fb.binary(BinOp::BitwiseOr, w, x)?;
            }
            let z = fb.u32_const(0)?;
            let z = fb.module().const_value(z)?;
            fb.composite_construct(ut, &[w, z, z, z])
        }
        // Another padded form's words are [`encode`]'s own, and a bfloat form's.
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb
        | ExtendedRange::R16Bfloat
        | ExtendedRange::Rg16Bfloat
        | ExtendedRange::Rgba16Bfloat => Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        )),
    }
}

/// The codes a write of `v` (clamped to `[0, 1]`, a NaN 0) to a padded `_sRGB` form stores
/// through its `table` ([`declare_padded`]), `k` the linear codes `RG10Unorm_X12` would store
/// (an Apple M4 and an Apple M3 Max, byte-identical: values next to every threshold, every `half`,
/// and a grid). `R10Unorm_X6_sRGB`'s red is the largest code whose first encode index is at most
/// `RoundEven(v · 8192)` (exact: `v` below 1 times `2^13`), as an extended-range `_sRGB` channel's
/// offset is ([`search`]); `RG10Unorm_X12_sRGB`'s red and green are the table's code of each linear
/// code. `RGB10A2Unorm_sRGB`'s red, green and blue are `R10Unorm_X6_sRGB`'s red's (an Apple M3
/// Max, 16,385 values). The other lanes stay `k`'s.
fn padded_srgb_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    table: GlobalId,
    v: Value,
    k: Value,
) -> BResult<Value> {
    let ut = crate::lower::arith::ops::ty(fb, words())?;
    let u1 = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f1 = fb.module().ty(Type::Float { width: 32 })?;
    let channels = srgb_channels(form);
    let mut lanes = Vec::with_capacity(4);
    for lane in 0..4u32 {
        let code = match (lane < channels, form) {
            (false, _) => fb.composite_extract(k, &[lane])?,
            (true, ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb) => {
                let grid = fb.module().constant(Constant::Scalar {
                    ty: f1,
                    bits: ConstBits(u64::from(8192.0f32.to_bits())),
                })?;
                let grid = fb.module().const_value(grid)?;
                let x = fb.composite_extract(v, &[lane])?;
                let x = fb.binary(BinOp::FMul, x, grid)?;
                let x = fb.glsl(GlslOp::RoundEven, &[x])?;
                let i = fb.convert(ConvOp::ConvertFToU, x, u1)?;
                search(fb, table, i)?
            }
            (true, _) => {
                let i = fb.composite_extract(k, &[lane])?;
                word(fb, table, i)?
            }
        };
        lanes.push(code);
    }
    fb.composite_construct(ut, &lanes)
}

/// The lanes, from red, a padded `_sRGB` form's table converts: its color channels.
fn srgb_channels(form: ExtendedRange) -> u32 {
    match form {
        ExtendedRange::Rgb10A2Srgb => 3,
        _ => form.raw_components(),
    }
}

/// A `vec4` constant of the four `f32` lanes `v`.
fn per_lane_f(fb: &mut FunctionBuilder<'_>, v: [f32; 4]) -> BResult<Value> {
    let ft = crate::lower::arith::ops::ty(fb, floats())?;
    let e = fb.module().ty(Type::Float { width: 32 })?;
    let mut parts = Vec::with_capacity(4);
    for x in v {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(u64::from(x.to_bits())),
        })?);
    }
    let c = fb.module().constant(Constant::Composite {
        ty: ft,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

#[cfg(test)]
pub(crate) mod tests;
