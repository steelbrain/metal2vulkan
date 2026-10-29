//! The emulated filters' arithmetic after the fetches, as module routines that every sample of the
//! module calls: one definition per module, whatever the number of samples.
//!
//! Each routine takes the bound format's `BicubicTexels` bits (or its `TexelFormat` class) as an
//! argument, the texture slot's specialization constant at every call, so once a device
//! specializes the module and inlines a call, the paths of the other formats fold away. Each is
//! started by `ModuleBuilder::routine` or `ModuleBuilder::texel_routine`, so on a profile whose
//! driver converts every function to one inlined at each call
//! (`DriverWorkaround::FunctionsInlinedAtEveryCall`, MoltenVK 1.4.1) it states `DontInline`.
//!
//! - [`Plane`] (a 2D bicubic sample): the Catmull-Rom weights (`a = -1/2`) of each axis's fraction
//!   `k / 256` in `f32`, the footprint's sixteen taps summed row by row and then the rows, in the
//!   device's multiply and add; where the sample is bilinear (Metal's rule for the 32-bit float
//!   formats, and for a level under four texels on either axis) the device's `FMix` of the inner
//!   taps; then the normalized formats' clamps. The answer is the device's arithmetic, not Metal's
//!   datapath (`PrecisionNote::BicubicFilterArithmetic`).
//! - [`Line`] (a 1D bicubic sample, bilinear over taps `i` and `i + 1` whatever the format): the
//!   device's `FMix` of the two taps, then the normalized formats' clamps.
//! - [`Quad`] (a two-plane YCbCr texture's linear sample read in place, `super::super::in_place`,
//!   bilinear over the footprint's taps `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, and `(i + 1, j + 1)`).
//!   `exact` weighs each tap `wx · wy` in units of `2^-16`, the integer weights `256 - k` and `k`
//!   per axis (a merged axis's fraction arrives as 0), and truncates each code's term before the
//!   half-up step, as Metal's 2D sums do: a packed field's by its own `s`
//!   (`PackedTexels::filter_truncation`: 7 on YCbCr's grid `2048`).
//!
//! The codes are recovered exactly: a YCbCr field's channels (`PackedTexels::YCbCr2048`,
//! `YCbCr8192`, `YCbCrExtended8192`) arrive as their steps `k / D`, exact, and sum on the grid
//! `2048` or `8192` (`S = D`), each 2D term `k · w` truncated by `2^7`; the extended-range field's
//! steps are signed and reach `2D`, so its quotient divides by `2D` and doubles, exactly, and a
//! YCbCr field's `half` result rounds the exact `f32` with ties away from zero.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::splat;
use crate::lower::arith::texel_round::{self, HalfRounding};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::{PackedTexels, TexelFormat};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type, UnOp};

/// The routines a module defines for its emulated samples, by the samples' dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Routine {
    /// A 2D (or 2D array) texture's bicubic sample: [`Plane`].
    Plane,
    /// A 1D (or 1D array) texture's bicubic sample: [`Line`].
    Line,
    /// A two-plane YCbCr texture's linear sample read in place: [`Quad`].
    Quad,
}

impl Routine {
    /// An upper estimate of the routines' SPIR-V words, counted once per module
    /// (`bicubic_routines_words_bound_their_emission`).
    pub fn words(self) -> u32 {
        match self {
            Routine::Plane => PLANE_WORDS,
            Routine::Line => LINE_WORDS,
            Routine::Quad => 5440,
        }
    }

    /// The census name of the routines.
    pub fn census_name(self) -> &'static str {
        match self {
            Routine::Plane => "sampler.bicubic.plane",
            Routine::Line => "sampler.bicubic.line",
            Routine::Quad => "sampler.linear.quad",
        }
    }
}

/// [`Routine::words`] of [`Routine::Plane`].
const PLANE_WORDS: u32 = 1024;

/// [`Routine::words`] of [`Routine::Line`].
const LINE_WORDS: u32 = 256;

/// A 2D bicubic sample's routine (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    /// `fn(16 × vec4, kx: u32, ky: u32, lkx: u32, lky: u32, bilinear: bool, flags: u32) -> vec4`:
    /// the filter of the footprint `taps[4j + i]` (row `j` on the `t` axis, column `i` on the `s`
    /// axis) at the fractions `kx / 256` and `ky / 256`, or where `bilinear` the inner taps'
    /// bilinear filter at `lkx / 256` and `lky / 256`, clamped as `flags` say.
    pub sum: FuncId,
}

/// A 1D bicubic sample's routine (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Line {
    /// `fn(t0: vec4, t1: vec4, k: u32, flags: u32) -> vec4`: the bilinear filter of the two taps
    /// at the fraction `k / 256`, clamped as `flags` say.
    pub sum: FuncId,
}

/// Define the 2D bicubic sample's routine in `mb`.
pub fn define_plane(mb: &mut ModuleBuilder) -> BResult<Plane> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let mut taps = Vec::with_capacity(16);
    for _ in 0..16 {
        taps.push(fb.param(t.vf)?);
    }
    let kx = fb.param(t.u)?;
    let ky = fb.param(t.u)?;
    let lkx = fb.param(t.u)?;
    let lky = fb.param(t.u)?;
    let bilinear = fb.param(t.b)?;
    let flags = fb.param(t.u)?;
    let fb_ = &mut fb;
    let wx = catmull_rom(fb_, t, kx)?;
    let wy = catmull_rom(fb_, t, ky)?;
    // Each row's sum under the `s` axis's weights, then the rows' under the `t` axis's.
    let weighted = |fb: &mut FunctionBuilder<'_>, xs: &[Value], ws: &[Value]| -> BResult<Value> {
        let mut acc: Option<Value> = None;
        for (x, w) in xs.iter().zip(ws) {
            let term = bin(fb, BinOp::VectorTimesScalar, *x, *w)?;
            acc = Some(match acc {
                Some(a) => bin(fb, BinOp::FAdd, a, term)?,
                None => term,
            });
        }
        match acc {
            Some(a) => Ok(a),
            None => f4(fb, 0.0),
        }
    };
    let mut rows = Vec::with_capacity(4);
    for row in taps.chunks(4) {
        rows.push(weighted(fb_, row, &wx)?);
    }
    let cubic = weighted(fb_, &rows, &wy)?;
    // The inner taps' bilinear filter: rows 1 and 2, columns 1 and 2.
    let (&[_, _, _, _, _, a11, a12, _, _, a21, a22, ..], fx, fy) = (
        taps.as_slice(),
        fraction(fb_, t, lkx)?,
        fraction(fb_, t, lky)?,
    ) else {
        return Err(crate::spirv::error::BuildError::Operand {
            op: crate::spv::Op::FunctionCall,
            rule: crate::spirv::error::OperandRule::Arity,
        });
    };
    let fx = broadcast(fb_, t.vf, fx)?;
    let fy = broadcast(fb_, t.vf, fy)?;
    let top = fb_.glsl(GlslOp::FMix, &[a11, a12, fx])?;
    let bottom = fb_.glsl(GlslOp::FMix, &[a21, a22, fx])?;
    let linear = fb_.glsl(GlslOp::FMix, &[top, bottom, fy])?;
    let r = fb_.select(bilinear, linear, cubic)?;
    let r = clamped(fb_, flags, r)?;
    Ok(Plane {
        sum: finish(fb, r)?,
    })
}

/// Define the 1D bicubic sample's routine in `mb`.
pub fn define_line(mb: &mut ModuleBuilder) -> BResult<Line> {
    let t = types(mb)?;
    let mut fb = mb.routine(t.vf)?;
    let t0 = fb.param(t.vf)?;
    let t1 = fb.param(t.vf)?;
    let k = fb.param(t.u)?;
    let flags = fb.param(t.u)?;
    let fb_ = &mut fb;
    let f = fraction(fb_, t, k)?;
    let f = broadcast(fb_, t.vf, f)?;
    let r = fb_.glsl(GlslOp::FMix, &[t0, t1, f])?;
    let r = clamped(fb_, flags, r)?;
    Ok(Line {
        sum: finish(fb, r)?,
    })
}

/// The fraction `k / 256` (an `f32`) of `k` (a `u32`, below 256), exact.
fn fraction(fb: &mut FunctionBuilder<'_>, t: Tys, k: Value) -> BResult<Value> {
    let x = fb.convert(ConvOp::ConvertUToF, k, t.f)?;
    let step = f1(fb, 1.0 / 256.0)?;
    bin(fb, BinOp::FMul, x, step)
}

/// The Catmull-Rom weights (`a = -1/2`) of taps `i - 1 .. i + 2` at fraction `k / 256`, four
/// `f32` scalars, in the device's arithmetic: `((A f + B) f + C) f + D` lane by lane, the
/// polynomials `-f³/2 + f² - f/2`, `3f³/2 - 5f²/2 + 1`, `-3f³/2 + 2f² + f/2`, and `f³/2 - f²/2`.
fn catmull_rom(fb: &mut FunctionBuilder<'_>, t: Tys, k: Value) -> BResult<[Value; 4]> {
    let f = fraction(fb, t, k)?;
    let lanes = |fb: &mut FunctionBuilder<'_>, x: [f32; 4]| -> BResult<Value> {
        let mut parts = Vec::with_capacity(4);
        for v in x {
            parts.push(f1(fb, v)?);
        }
        fb.composite_construct(t.vf, &parts)
    };
    let a = lanes(fb, [-0.5, 1.5, -1.5, 0.5])?;
    let b = lanes(fb, [1.0, -2.5, 2.0, -0.5])?;
    let c = lanes(fb, [-0.5, 0.0, 0.5, 0.0])?;
    let d = lanes(fb, [0.0, 1.0, 0.0, 0.0])?;
    let w = bin(fb, BinOp::VectorTimesScalar, a, f)?;
    let w = bin(fb, BinOp::FAdd, w, b)?;
    let w = bin(fb, BinOp::VectorTimesScalar, w, f)?;
    let w = bin(fb, BinOp::FAdd, w, c)?;
    let w = bin(fb, BinOp::VectorTimesScalar, w, f)?;
    let w = bin(fb, BinOp::FAdd, w, d)?;
    Ok([
        fb.composite_extract(w, &[0])?,
        fb.composite_extract(w, &[1])?,
        fb.composite_extract(w, &[2])?,
        fb.composite_extract(w, &[3])?,
    ])
}

/// `r` clamped into the bound format's range: `[0, 1]` for normalized unsigned channels
/// (`BicubicTexels::UNORM`), `[-1, 1]` for signed ones (`BicubicTexels::SNORM`), else as it is.
fn clamped(fb: &mut FunctionBuilder<'_>, flags: Value, r: Value) -> BResult<Value> {
    let (zero, one, neg) = (f4(fb, 0.0)?, f4(fb, 1.0)?, f4(fb, -1.0)?);
    let unorm = fb.glsl(GlslOp::FClamp, &[r, zero, one])?;
    let is_unorm = flag(fb, flags, BicubicTexels::UNORM)?;
    let r = fb.select(is_unorm, unorm, r)?;
    let snorm = fb.glsl(GlslOp::FClamp, &[r, neg, one])?;
    let is_snorm = flag(fb, flags, BicubicTexels::SNORM)?;
    fb.select(is_snorm, snorm, r)
}

/// A two-plane YCbCr texture's in-place linear sample's routine (module docs). The taps are the
/// footprint's row `j`, then row `j + 1`, each from column `i`; `kx` and `ky` are the fractions'
/// 1/256 steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quad {
    /// `fn(t00: vec4, t01: vec4, t10: vec4, t11: vec4, kx: u32, ky: u32, flags: u32, half: bool)
    /// -> vec4`: the exact sum on the field's grid, taking a merged axis's fraction as 0.
    pub exact: FuncId,
}

/// Define the in-place linear sample's routine in `mb`.
pub fn define_quad(mb: &mut ModuleBuilder) -> BResult<Quad> {
    Ok(Quad {
        exact: define_exact_quad(mb)?,
    })
}

// ---- Types and constants ----

/// The four lanes of a texel.
const LANES: u8 = 4;

fn uform(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

fn fform(lanes: u8) -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    }
}

fn bform(lanes: u8) -> Form {
    Form {
        held: Held::Bool,
        lanes,
    }
}

/// The routines' types.
#[derive(Clone, Copy)]
struct Tys {
    f: TypeId,
    u: TypeId,
    b: TypeId,
    vf: TypeId,
    vu: TypeId,
}

fn types(mb: &mut ModuleBuilder) -> BResult<Tys> {
    let f = mb.ty(Type::Float { width: 32 })?;
    let u = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let b = mb.ty(Type::Bool)?;
    let vf = mb.ty(Type::Vector {
        elem: f,
        len: LANES,
    })?;
    let vu = mb.ty(Type::Vector {
        elem: u,
        len: LANES,
    })?;
    Ok(Tys { f, u, b, vf, vu })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// A float operation no device may contract or regroup.
fn exact_op(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary_no_contraction(op, a, b)
}

/// The `u32` scalar `v`.
fn u1(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    splat(fb, uform(1), u64::from(v))
}

/// The `uvec4` of `v` in every lane.
fn u4(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    splat(fb, uform(LANES), u64::from(v))
}

/// The `f32` scalar `x`.
fn f1(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<Value> {
    splat(fb, fform(1), u64::from(x.to_bits()))
}

/// The `vec4` of `x` in every lane.
fn f4(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<Value> {
    splat(fb, fform(LANES), u64::from(x.to_bits()))
}

/// The vector of scalar `x` in every lane.
fn broadcast(fb: &mut FunctionBuilder<'_>, ty: TypeId, x: Value) -> BResult<Value> {
    fb.composite_construct(ty, &[x; LANES as usize])
}

/// Whether bit `bit` of `flags` (a `u32`) is set.
fn flag(fb: &mut FunctionBuilder<'_>, flags: Value, bit: u32) -> BResult<Value> {
    let m = u1(fb, bit)?;
    let z = u1(fb, 0)?;
    let and = bin(fb, BinOp::BitwiseAnd, flags, m)?;
    bin(fb, BinOp::INotEqual, and, z)
}

/// The function's return value `r`, and the function.
fn finish(mut fb: FunctionBuilder<'_>, r: Value) -> BResult<FuncId> {
    fb.ret(Some(r))?;
    fb.finish()
}

// ---- The exact sums ----

/// The grid of an `EXACT` format: `D` (`16 · 255`, `16 · 127`, `65535`,
/// or `32767`; with `SRGB` `4095` for the color lanes and `16 · 255` for
/// alpha; with `EXTENDED` `16 · 510` for the color lanes, `8192` an `_sRGB` twin's, and `8 · 510`
/// for alpha) as `u32` lanes, the bounds of a tap's code (`±D`, `±2D` for `EXTENDED`) as `f32`
/// lanes, the bounds of a sum's step (`[0, D]`, `[-D, D]` for a signed format,
/// `[-2D, 2D]` for `EXTENDED`) as `i32` bits, and what the result's quotient reads: the grid it
/// divides by (`D`, or `2D` for `EXTENDED`, whose steps reach past `D`), the factor that restores
/// it (1 or 2, exact), and the lanes whose `half` result rounds ties away from zero (an
/// `EXTENDED` `_sRGB` twin's color lanes, a YCbCr field's). A tap's code is recovered on `scale`
/// (`f32` lanes) and
/// multiplied by `per` (`u32` lanes): `D` and 1 (no `per`), but on a packed normalized format's
/// lanes (its `TexelFormat::PACKED` field with `PackedTexels::filter_grids`, where the routine
/// reads such a class: [`Packed`]), which take the field's grid `D` each, their channel's code
/// scale `S` (also the code's bound) and `D / S`. Such a routine's grid also states whether the
/// field names its grids and the field's truncation `s` per lane
/// (`PackedTexels::filter_truncation`), which [`Truncation::of`] takes there. A field whose steps
/// reach `±2D` (`PackedTexels::filter_extended`) bounds its codes by `±2S` and divides by `2D`.
struct Grid {
    scale: Value,
    per: Option<Value>,
    cap: Value,
    neg_cap: Value,
    lo: Value,
    hi: Value,
    divisor: Value,
    factor: Value,
    away: Value,
    packed: Option<(Value, Value)>,
}

/// Whether a routine's treatment may name a packed normalized format's grids: a `TexelFormat`
/// class's routine (the linear filter's), not a `BicubicTexels` one's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Packed {
    Field,
    No,
}

fn grid(fb: &mut FunctionBuilder<'_>, t: Tys, flags: Value, packed: Packed) -> BResult<Grid> {
    let wide16 = flag(fb, flags, BicubicTexels::WIDE16)?;
    let snorm = flag(fb, flags, BicubicTexels::SNORM)?;
    let (d16s, d16u) = (u1(fb, 32767)?, u1(fb, 65535)?);
    let (d8s, d8u) = (u1(fb, 16 * 127)?, u1(fb, 16 * 255)?);
    let d16 = fb.select(snorm, d16s, d16u)?;
    let d8 = fb.select(snorm, d8s, d8u)?;
    let d = fb.select(wide16, d16, d8)?;
    let d = broadcast(fb, t.vu, d)?;
    let srgb = flag(fb, flags, BicubicTexels::SRGB)?;
    let (color, alpha) = (u1(fb, 4095)?, u1(fb, 16 * 255)?);
    let srgb_d = fb.composite_construct(t.vu, &[color, color, color, alpha])?;
    let d = fb.select(srgb, srgb_d, d)?;
    let extended = flag(fb, flags, BicubicTexels::EXTENDED)?;
    let (linear, steps, alpha) = (u1(fb, 16 * 510)?, u1(fb, 8192)?, u1(fb, 8 * 510)?);
    let color = fb.select(srgb, steps, linear)?;
    let extended_d = fb.composite_construct(t.vu, &[color, color, color, alpha])?;
    let d = fb.select(extended, extended_d, d)?;
    let (one, two) = (u1(fb, 1)?, u1(fb, 2)?);
    let factor_u = fb.select(extended, two, one)?;
    let factor_u = broadcast(fb, t.vu, factor_u)?;
    let divisor = bin(fb, BinOp::IMul, d, factor_u)?;
    let factor = fb.convert(ConvOp::ConvertUToF, factor_u, t.vf)?;
    let lo = |fb: &mut FunctionBuilder<'_>, divisor: Value| -> BResult<Value> {
        let neg = fb.unary(UnOp::SNegate, divisor)?;
        let z = u4(fb, 0)?;
        let signed = bin(fb, BinOp::LogicalOr, snorm, extended)?;
        fb.select(signed, neg, z)
    };
    let srgb_steps = bin(fb, BinOp::LogicalAnd, extended, srgb)?;
    let (yes, no) = (splat(fb, bform(1), 1)?, splat(fb, bform(1), 0)?);
    let lanes = crate::lower::arith::ops::ty(fb, bform(LANES))?;
    let color_lanes = fb.composite_construct(lanes, &[yes, yes, yes, no])?;
    let none = splat(fb, bform(LANES), 0)?;
    let away = fb.select(srgb_steps, color_lanes, none)?;
    let Some(p) = packed_grids(fb, t, flags, packed)? else {
        let cap = fb.convert(ConvOp::ConvertUToF, divisor, t.vf)?;
        return Ok(Grid {
            scale: fb.convert(ConvOp::ConvertUToF, d, t.vf)?,
            per: None,
            cap,
            neg_cap: fb.unary(UnOp::FNegate, cap)?,
            lo: lo(fb, divisor)?,
            hi: divisor,
            divisor,
            factor,
            away,
            packed: None,
        });
    };
    let packed = p.named;
    // A packed field's lanes: its grids, each tap's code recovered on its channel's scale
    // (clamped there, within `±2S` where the field's steps reach `±2D`) and moved onto the grid;
    // such a field's step divided by `2D` and doubled, and a YCbCr field's `half` rounded with
    // ties away from zero.
    let scale = fb.select(packed, p.scales, d)?;
    let scale = fb.convert(ConvOp::ConvertUToF, scale, t.vf)?;
    let wide = bin(fb, BinOp::LogicalAnd, packed, p.wide)?;
    let span = fb.select(wide, two, one)?;
    let span = broadcast(fb, t.vu, span)?;
    let field_cap = bin(fb, BinOp::IMul, p.scales, span)?;
    let cap = fb.select(packed, field_cap, divisor)?;
    let cap = fb.convert(ConvOp::ConvertUToF, cap, t.vf)?;
    let neg_cap = fb.unary(UnOp::FNegate, cap)?;
    let ones = u4(fb, 1)?;
    let per = Some(fb.select(packed, p.per, ones)?);
    let field_divisor = bin(fb, BinOp::IMul, p.d, span)?;
    let divisor = fb.select(packed, field_divisor, divisor)?;
    let span = fb.convert(ConvOp::ConvertUToF, span, t.vf)?;
    let factor = fb.select(packed, span, factor)?;
    let rounds_away = bin(fb, BinOp::LogicalAnd, packed, p.away)?;
    let away = fb.select(rounds_away, color_lanes, away)?;
    Ok(Grid {
        scale,
        per,
        cap,
        neg_cap,
        lo: lo(fb, divisor)?,
        hi: divisor,
        divisor,
        factor,
        away,
        packed: Some((packed, p.truncation)),
    })
}

/// The code of normalized texel `x` on grid `g`, as `i32` bits: `RoundEven(x · D)` (module docs),
/// or on a packed normalized format's lane `RoundEven(x · S) · (D / S)`, its code on its channel's
/// scale `S` moved onto the grid (the device's value may miss `c / S` by more than half a grid
/// step, never by half a code). The product is clamped into the code bounds first (`[-D, D]`,
/// `[-2D, 2D]` for an extended-range lane, `[-S, S]` on a packed lane), which changes no texel's of
/// the format, so that a texel of a format this routine does not serve (its result is selected
/// away) still converts and shifts within range.
fn code(fb: &mut FunctionBuilder<'_>, g: &Grid, x: Value) -> BResult<Value> {
    let y = fb.binary_no_contraction(BinOp::FMul, x, g.scale)?;
    let y = fb.glsl(GlslOp::NClamp, &[y, g.neg_cap, g.cap])?;
    let y = fb.glsl(GlslOp::RoundEven, &[y])?;
    let it = crate::lower::arith::ops::ty(fb, uform(LANES))?;
    let c = fb.convert(ConvOp::ConvertFToS, y, it)?;
    match g.per {
        Some(per) => bin(fb, BinOp::IMul, c, per),
        None => Ok(c),
    }
}

/// The value `n / D` of grid step `n` (`i32` bits, within the grid's step bounds): `+0` for
/// zero, the magnitude divided exactly, rounded to `f32`, or to `half` when `half`, and the sign
/// set. An extended-range lane divides by `2D` and doubles, exactly; its `_sRGB` color lanes'
/// `half` is the exact `f32` rounded with ties away from zero.
fn ratio(fb: &mut FunctionBuilder<'_>, g: &Grid, n: Value, half: Value) -> BResult<Value> {
    let rf = texel_round::grid_value(fb, uform(LANES), n, g.divisor, |fb, m, d| {
        texel_round::ratio_to_f32(fb, uform(LANES), m, d)
    })?;
    let rh = texel_round::grid_value(fb, uform(LANES), n, g.divisor, |fb, m, d| {
        texel_round::ratio_to_half(fb, uform(LANES), m, d)
    })?;
    let rf = exact_op(fb, BinOp::FMul, rf, g.factor)?;
    let rh = exact_op(fb, BinOp::FMul, rh, g.factor)?;
    let ra = texel_round::f32_to_half(fb, fform(LANES), rf, HalfRounding::NearestAway)?;
    let rh = fb.select(g.away, ra, rh)?;
    fb.select(half, rh, rf)
}

/// The parameters of a [`Quad`] routine, in order.
struct QuadParams {
    t00: Value,
    t01: Value,
    t10: Value,
    t11: Value,
    kx: Value,
    ky: Value,
    flags: Value,
    half: Value,
}

fn quad_params(fb: &mut FunctionBuilder<'_>, t: Tys) -> BResult<QuadParams> {
    Ok(QuadParams {
        t00: fb.param(t.vf)?,
        t01: fb.param(t.vf)?,
        t10: fb.param(t.vf)?,
        t11: fb.param(t.vf)?,
        kx: fb.param(t.u)?,
        ky: fb.param(t.u)?,
        flags: fb.param(t.u)?,
        half: fb.param(t.b)?,
    })
}

/// `fn(t00, t01, t10, t11, kx, ky, flags, half) -> vec4` (module docs, [`Quad::exact`]): Metal's
/// sum over the four taps under the weights `w = wx · wy` in units of `2^-16`, the axes' weights
/// `256 - k` and `k` (a merged axis's fraction arrives as 0), each term truncated: `T = Σ ⌊c · w /
/// 2^s⌋` for the codes `c` on the format's grid, then the step `⌊(T + 2^(15 - s)) / 2^(16 -
/// s)⌋`, clamped and divided by the grid exactly. `s` is 8 for a 16-bit format's codes and 4 for
/// every 12-bit grid: an sRGB color lane's step is truncated, and an 8-bit lane's `16 ·
/// code` term is a multiple of 16, so its sum is the exact one, rounded half up. An extended-range
/// format's `s`
/// is 7 on a color lane and 4 on alpha (every lane of random 2D footprints of the
/// four formats). An unsigned format's product `c · w` is below `2^32` (codes below `2^16`, weights
/// at most `2^16`), a `u32` word shifted logically; a signed format's codes (at most `2^15 - 1` in
/// magnitude) keep it below `2^31`, an `i32` shifted arithmetically (toward negative infinity), as
/// an extended-range format's do (below `2^14`). The terms' sum is below `2^24` in magnitude (the
/// weights sum to `2^16`, and the codes are below `2^12` where `s` is 4, below `2^14` where it is
/// 7, and below `2^16` where it is 8), and so is the biased one.
fn define_exact_quad(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let t = types(mb)?;
    let mut fb = mb.texel_routine(t.vf)?;
    let q = quad_params(&mut fb, t)?;
    let fb_ = &mut fb;
    let g = grid(fb_, t, q.flags, Packed::Field)?;
    let n = quad_sum(fb_, t, &g, &q, |fb, x| code(fb, &g, x))?;
    let r = ratio(fb_, &g, n, q.half)?;
    finish(fb, r)
}

/// The exact quad's step ([`define_exact_quad`]) over grid `g` of `q`'s taps (`(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`), each tap's code `code_of` it (taken where its term is),
/// under `q`'s fractions and flags: the terms' step rounded half up and clamped into the grid's
/// step bounds, which the caller divides.
fn quad_sum(
    fb_: &mut FunctionBuilder<'_>,
    t: Tys,
    g: &Grid,
    q: &QuadParams,
    mut code_of: impl FnMut(&mut FunctionBuilder<'_>, Value) -> BResult<Value>,
) -> BResult<Value> {
    let (kx, ky, flags) = (q.kx, q.ky, q.flags);
    let k256 = u1(fb_, 256)?;
    let wx = [bin(fb_, BinOp::ISub, k256, kx)?, kx];
    let wy = [bin(fb_, BinOp::ISub, k256, ky)?, ky];
    let tr = Truncation::of(fb_, t, flags, g)?;
    let rows = [[q.t00, q.t01], [q.t10, q.t11]];
    let mut total = u4(fb_, 0)?;
    for (row, b) in rows.into_iter().zip(wy) {
        for (x, a) in row.into_iter().zip(wx) {
            let c = code_of(fb_, x)?;
            let w = bin(fb_, BinOp::IMul, a, b)?;
            let w = broadcast(fb_, t.vu, w)?;
            let term = tr.term(fb_, c, w)?;
            total = bin(fb_, BinOp::IAdd, total, term)?;
        }
    }
    let n = tr.step(fb_, total)?;
    fb_.glsl(GlslOp::SClamp, &[n, g.lo, g.hi])
}

/// How a 2D or 3D exact sum truncates its terms: `s` per lane (8 for a 16-bit format's codes, 4 for
/// every 12-bit grid, for an extended-range format 7 on a color lane and 4 on alpha, and on a
/// packed field's grids the field's own, `PackedTexels::filter_truncation`: 7 on YCbCr's grid
/// `2048`), the bias `2^(15 - s)` and shift `16 - s` of the step, and whether the codes are signed
/// (a signed normalized or an extended-range format's), whose terms and sum shift arithmetically
/// (toward negative infinity).
struct Truncation {
    s: Value,
    bias: Value,
    shift: Value,
    signed: Value,
}

impl Truncation {
    fn of(fb: &mut FunctionBuilder<'_>, t: Tys, flags: Value, g: &Grid) -> BResult<Truncation> {
        let wide16 = flag(fb, flags, BicubicTexels::WIDE16)?;
        let snorm = flag(fb, flags, BicubicTexels::SNORM)?;
        let extended = flag(fb, flags, BicubicTexels::EXTENDED)?;
        let (a, b) = (u1(fb, 8)?, u1(fb, 4)?);
        let plain = fb.select(wide16, a, b)?;
        let plain = broadcast(fb, t.vu, plain)?;
        let seven = u1(fb, 7)?;
        let lanes = fb.composite_construct(t.vu, &[seven, seven, seven, b])?;
        let s = fb.select(extended, lanes, plain)?;
        let s = match g.packed {
            Some((named, own)) => fb.select(named, own, s)?,
            None => s,
        };
        let (k15, k16, one) = (u4(fb, 15)?, u4(fb, 16)?, u4(fb, 1)?);
        let up = bin(fb, BinOp::ISub, k15, s)?;
        let bias = bin(fb, BinOp::ShiftLeftLogical, one, up)?;
        let shift = bin(fb, BinOp::ISub, k16, s)?;
        let signed = bin(fb, BinOp::LogicalOr, snorm, extended)?;
        Ok(Truncation {
            s,
            bias,
            shift,
            signed,
        })
    }

    /// The term `⌊c · w / 2^s⌋` of code `c` under weight `w` (`uvec4` lanes).
    fn term(&self, fb: &mut FunctionBuilder<'_>, c: Value, w: Value) -> BResult<Value> {
        let prod = bin(fb, BinOp::IMul, c, w)?;
        let unsigned = bin(fb, BinOp::ShiftRightLogical, prod, self.s)?;
        let signed = bin(fb, BinOp::ShiftRightArithmetic, prod, self.s)?;
        fb.select(self.signed, signed, unsigned)
    }

    /// The grid step `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋` of the terms' sum `T`.
    fn step(&self, fb: &mut FunctionBuilder<'_>, total: Value) -> BResult<Value> {
        let total = bin(fb, BinOp::IAdd, total, self.bias)?;
        let unsigned = bin(fb, BinOp::ShiftRightLogical, total, self.shift)?;
        let signed = bin(fb, BinOp::ShiftRightArithmetic, total, self.shift)?;
        fb.select(self.signed, signed, unsigned)
    }
}

/// A packed field's grids ([`packed_grids`]), each `uvec4` lanes.
struct PackedGrids {
    /// Whether the treatment names the field's grids.
    named: Value,
    /// The grids `D`.
    d: Value,
    /// The channels' code scales `S`.
    scales: Value,
    /// `D / S`.
    per: Value,
    /// The 2D and 3D sums' truncation `s` (`PackedTexels::filter_truncation`).
    truncation: Value,
    /// Whether the field's codes and steps reach `±2D` (`PackedTexels::filter_extended`), a
    /// `bool`.
    wide: Value,
    /// Whether a `half` result rounds the step's value with ties away from zero
    /// (`PackedTexels::ycbcr`), a `bool`.
    away: Value,
}

/// Whether treatment `flags` names a packed field's grids (its `TexelFormat::PACKED` field with
/// `PackedTexels::filter_grids`, under `BicubicTexels::EXACT`: `TexelFormat::packed_exact`), and
/// per lane that field's grids (each lane 1 for any other field); `None` for a routine whose
/// treatment names none.
fn packed_grids(
    fb: &mut FunctionBuilder<'_>,
    t: Tys,
    flags: Value,
    packed: Packed,
) -> BResult<Option<PackedGrids>> {
    if packed == Packed::No {
        return Ok(None);
    }
    let shift = u1(fb, TexelFormat::PACKED)?;
    let field = bin(fb, BinOp::ShiftRightLogical, flags, shift)?;
    let four_bits = u1(fb, 0xF)?;
    let field = bin(fb, BinOp::BitwiseAnd, field, four_bits)?;
    let no = splat(fb, bform(1), 0)?;
    let (mut named, mut wide, mut away) = (no, no, no);
    let ones = u4(fb, 1)?;
    let (mut d, mut s, mut per, mut truncation) = (ones, ones, ones, ones);
    for p in PackedTexels::ALL {
        let (Some(grids), Some(cut)) = (p.filter_grids(), p.filter_truncation()) else {
            continue;
        };
        // A field without code scales (a YCbCr field's exact steps) recovers its codes on the
        // grids.
        let scales = p.scales().unwrap_or(grids);
        let k = u1(fb, p as u32)?;
        let is = bin(fb, BinOp::IEqual, field, k)?;
        let lanes = |fb: &mut FunctionBuilder<'_>, v: [u32; 4]| -> BResult<Value> {
            let mut l = Vec::with_capacity(4);
            for x in v {
                l.push(u1(fb, x)?);
            }
            fb.composite_construct(t.vu, &l)
        };
        let mut steps = grids;
        for (g, s) in steps.iter_mut().zip(scales) {
            *g /= s.max(1);
        }
        let these_d = lanes(fb, grids)?;
        let these_s = lanes(fb, scales)?;
        let these_per = lanes(fb, steps)?;
        let these_cut = lanes(fb, cut)?;
        d = fb.select(is, these_d, d)?;
        s = fb.select(is, these_s, s)?;
        per = fb.select(is, these_per, per)?;
        truncation = fb.select(is, these_cut, truncation)?;
        named = bin(fb, BinOp::LogicalOr, named, is)?;
        if p.filter_extended() {
            wide = bin(fb, BinOp::LogicalOr, wide, is)?;
        }
        if p.ties_away() {
            away = bin(fb, BinOp::LogicalOr, away, is)?;
        }
    }
    let exact = flag(fb, flags, BicubicTexels::EXACT)?;
    let named = bin(fb, BinOp::LogicalAnd, named, exact)?;
    Ok(Some(PackedGrids {
        named,
        d,
        scales: s,
        per,
        truncation,
        wide,
        away,
    }))
}
