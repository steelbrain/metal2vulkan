//! The emulated filters' arithmetic after the fetches, as module routines that every sample of the
//! module calls: one definition per module, whatever the number of samples.
//!
//! Each routine takes the bound format's `BicubicTexels` bits (or its `TexelFormat` class) as an
//! argument, the texture slot's specialization constant at every call, so once a device
//! specializes the module and inlines a call, the paths of the other formats fold away. They are
//! plain functions.
//!
//! - [`Plane`] (a 2D bicubic sample): the Catmull-Rom weights (`a = -1/2`) of each axis's fraction
//!   `k / 256` in `f32`, the footprint's sixteen taps summed row by row and then the rows, in the
//!   device's multiply and add; where the sample is bilinear (Metal's rule for the 32-bit float
//!   formats, and for a level under four texels on either axis) the device's `FMix` of the inner
//!   taps; then the normalized formats' clamps. The answer is the device's arithmetic, not Metal's
//!   datapath (`PrecisionNote::BicubicFilterArithmetic`).
//! - [`Line`] (a 1D bicubic sample, bilinear over taps `i` and `i + 1` whatever the format): the
//!   device's `FMix` of the two taps, then the normalized formats' clamps.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::splat;
use crate::metal::bicubic::BicubicTexels;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type};

/// The routines a module defines for its emulated samples, by the samples' dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Routine {
    /// A 2D (or 2D array) texture's bicubic sample: [`Plane`].
    Plane,
    /// A 1D (or 1D array) texture's bicubic sample: [`Line`].
    Line,
}

impl Routine {
    /// An upper estimate of the routines' SPIR-V words, counted once per module
    /// (`bicubic_routines_words_bound_their_emission`).
    pub fn words(self) -> u32 {
        match self {
            Routine::Plane => PLANE_WORDS,
            Routine::Line => LINE_WORDS,
        }
    }

    /// The census name of the routines.
    pub fn census_name(self) -> &'static str {
        match self {
            Routine::Plane => "sampler.bicubic.plane",
            Routine::Line => "sampler.bicubic.line",
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
    let mut fb = mb.function(t.vf)?;
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
    let mut fb = mb.function(t.vf)?;
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

/// The routines' types.
#[derive(Clone, Copy)]
struct Tys {
    f: TypeId,
    u: TypeId,
    b: TypeId,
    vf: TypeId,
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
    // Declared here, so the module keeps the order of its type declarations.
    mb.ty(Type::Vector {
        elem: u,
        len: LANES,
    })?;
    Ok(Tys { f, u, b, vf })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// The `u32` scalar `v`.
fn u1(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    splat(fb, uform(1), u64::from(v))
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
