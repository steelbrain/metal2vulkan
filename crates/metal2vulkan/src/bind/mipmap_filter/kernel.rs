//! The filter kernel ([`super::mipmap_filter_module`]) of one [`super::MipmapFilterKernel`]: one
//! invocation per destination texel, its taps from the level's tap tables (eight in 3D, four where
//! the level above is one texel deep), each source texel's words read once and decoded to its
//! `float` channels by its format's layout, each channel the weighted sum of its taps (a tap of
//! weight 0 left out), and the sums encoded back (`layout.rs`). The products and sums are the
//! device's `float` arithmetic in the kernel's order, which it may contract (`cpu.rs` bounds the
//! answer). The kernel is built with its format's layout, so that no branch on a format is left
//! to the GPU.

use super::fetch::{self, Component, Recover};
use super::layout::{header, Field, Layout, Rule, Texel, EXTENDED_SCALE, SRGB_SCALE, TAP_WORDS};
use super::{DESTINATION, PARAMS, SOURCE, SOURCE_IMAGE};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::pack;
use crate::pipeline::sym::ResourceSlot;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp,
    ImageOperands, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, Scope, Semantics,
    StorageClass, Type, TypedModule, UnOp,
};

/// Invocations of one workgroup, along x.
pub(super) const LOCAL_SIZE: u32 = 64;

/// What a kernel is built with (`MipmapFilterKernel::spec`): its format's layout, whether it takes
/// the depth taps (a 3D level whose level above is more than one texel deep), and the first
/// level's sampled view where it fetches its taps from it instead of reading the scratch buffer
/// (the first later level of a chain that reads its first level in place: `fetch.rs`).
pub(super) struct Spec {
    pub(super) layout: Layout,
    pub(super) depth: bool,
    pub(super) fetch: Option<Fetch>,
}

/// The first level's sampled view a kernel fetches its taps from (`fetch.rs`): the components a
/// fetch returns, and whether the view is a 3D one (a 3D texture's) or a 2D array of every layer.
#[derive(Clone, Copy)]
pub(super) struct Fetch {
    pub(super) components: [Option<Component>; 4],
    pub(super) volume: bool,
}

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

/// What every part of the kernel takes: the types, the buffers, and what the kernel is built
/// with.
#[derive(Clone, Copy)]
struct K<'s> {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    /// The level above: its words in the scratch buffer, or, where the spec fetches
    /// ([`Spec::fetch`]), the first level's sampled view, loaded.
    source: Value,
    destination: Value,
    params: Value,
    spec: &'s Spec,
}

/// A field without its channel's value (`encode` takes one per channel the layout decodes).
fn missing_channel() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::CompositeExtract,
        rule: OperandRule::Arity,
    }
}

fn c32(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

fn cf(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: f32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: k.f32_,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    f.module().const_value(c)
}

fn bin(f: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    f.binary(op, a, b)
}

/// `a op v` for a `u32` constant `v`.
fn bink(f: &mut FunctionBuilder<'_>, op: BinOp, a: Value, v: u32) -> BResult<Value> {
    let c = c32(f, v)?;
    f.binary(op, a, c)
}

/// `a op v` for an `f32` constant `v`.
fn fopk(f: &mut FunctionBuilder<'_>, k: &K<'_>, op: BinOp, a: Value, v: f32) -> BResult<Value> {
    let c = cf(f, k, v)?;
    f.binary(op, a, c)
}

/// The scalar `float` form `lower::arith::pack`'s transfers take.
const SCALAR: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

fn select(f: &mut FunctionBuilder<'_>, c: Value, t: Value, e: Value) -> BResult<Value> {
    f.select(c, t, e)
}

/// Word `i` (a value) of the level's params.
fn param(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let p = f.access_chain(k.params, &[zero, i])?;
    f.load(p, MemAccess::default())
}

/// Word `i` (a constant) of the level's params.
fn paramk(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: u32) -> BResult<Value> {
    let i = c32(f, i)?;
    param(f, k, i)
}

/// `2^e` as an `f32` value from `e + 127` (a `u32` value from 1 to 254).
fn pow2_biased(f: &mut FunctionBuilder<'_>, k: &K<'_>, biased: Value) -> BResult<Value> {
    let bits = bink(f, BinOp::ShiftLeftLogical, biased, 23)?;
    f.convert(ConvOp::Bitcast, bits, k.f32_)
}

/// `v` with a NaN read as 0.
fn number(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: Value) -> BResult<Value> {
    let nan = f.unary(UnOp::IsNan, v)?;
    let zero = cf(f, k, 0.0)?;
    select(f, nan, zero, v)
}

/// `v` clamped to `[lo, hi]`.
fn clamp(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: Value, lo: f32, hi: f32) -> BResult<Value> {
    let lo = cf(f, k, lo)?;
    let hi = cf(f, k, hi)?;
    f.glsl(GlslOp::FClamp, &[v, lo, hi])
}

/// Source texel `index`'s words, read once from the scratch buffer: its bytes from bit 0 of the
/// first (a texel narrower than a word shifted down from the word that holds all of it).
fn texel(f: &mut FunctionBuilder<'_>, k: &K<'_>, index: Value) -> BResult<Vec<Value>> {
    let zero = c32(f, 0)?;
    let bytes = k.spec.layout.bytes;
    if bytes < 4 {
        let byte = bink(f, BinOp::IMul, index, bytes)?;
        let word = bink(f, BinOp::ShiftRightLogical, byte, 2)?;
        let lane = bink(f, BinOp::BitwiseAnd, byte, 3)?;
        let lane = bink(f, BinOp::ShiftLeftLogical, lane, 3)?;
        let p = f.access_chain(k.source, &[zero, word])?;
        let w = f.load(p, MemAccess::default())?;
        return Ok(vec![bin(f, BinOp::ShiftRightLogical, w, lane)?]);
    }
    let per = bytes / 4;
    let first = bink(f, BinOp::IMul, index, per)?;
    let mut words = Vec::with_capacity(per as usize);
    for j in 0..per {
        let at = if j == 0 {
            first
        } else {
            bink(f, BinOp::IAdd, first, j)?
        };
        let p = f.access_chain(k.source, &[zero, at])?;
        words.push(f.load(p, MemAccess::default())?);
    }
    Ok(words)
}

/// The field at bit `offset` of a texel's `words` ([`texel`]), masked by `mask`: a field never
/// crosses a 32-bit word.
fn bits(f: &mut FunctionBuilder<'_>, words: &[Value], offset: u32, mask: u32) -> BResult<Value> {
    let w = words[(offset / 32) as usize];
    let w = match offset % 32 {
        0 => w,
        shift => bink(f, BinOp::ShiftRightLogical, w, shift)?,
    };
    match mask {
        u32::MAX => Ok(w),
        mask => bink(f, BinOp::BitwiseAnd, w, mask),
    }
}

/// The `f32` value of a signed integer held in a `u32` value.
fn signed_float(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: Value) -> BResult<Value> {
    let s = f.convert(ConvOp::Bitcast, v, k.i32_)?;
    f.convert(ConvOp::ConvertSToF, s, k.f32_)
}

/// The value of float bits `b` of a 5-bit exponent and `mantissa` mantissa bits (a half's, or an
/// unsigned field of `RG11B10Float`, placed as the half of the same bits) by the device's widening
/// (`UnpackHalf2x16`).
fn small_float_value(f: &mut FunctionBuilder<'_>, b: Value, mantissa: u32) -> BResult<Value> {
    let h = match 10 - mantissa {
        0 => b,
        shift => bink(f, BinOp::ShiftLeftLogical, b, shift)?,
    };
    let pair = f.glsl(GlslOp::UnpackHalf2x16, &[h])?;
    f.composite_extract(pair, &[0])
}

/// The bits of `v` as a half by the device's narrowing (`PackHalf2x16`, the high half 0).
fn half_bits_of(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: Value) -> BResult<Value> {
    let vec2 = f.module().ty(Type::Vector {
        elem: k.f32_,
        len: 2,
    })?;
    let zero = cf(f, k, 0.0)?;
    let pair = f.composite_construct(vec2, &[v, zero])?;
    f.glsl(GlslOp::PackHalf2x16, &[pair])
}

/// The bits of `v` as an unsigned float of a 5-bit exponent and `mantissa` mantissa bits
/// (`RG11B10Float`'s, which no device instruction narrows to; `cpu::small_float_bits`).
fn small_float_bits(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    v: Value,
    mantissa: u32,
) -> BResult<Value> {
    let kk = 23 - mantissa;
    let inf = 31 << mantissa;
    let b = f.convert(ConvOp::Bitcast, v, k.u32_)?;
    let sign = bink(f, BinOp::ShiftRightLogical, b, 31)?;
    let a = bink(f, BinOp::BitwiseAnd, b, 0x7fff_ffff)?;
    let nan = bink(f, BinOp::UGreaterThan, a, 0x7f80_0000)?;
    let e = bink(f, BinOp::ShiftRightLogical, a, 23)?;
    // A normal result: the exponent rebiased, the mantissa rounded to nearest even.
    let rebiased = bink(f, BinOp::ISub, a, 112 << 23)?;
    let lsb = bink(f, BinOp::ShiftRightLogical, rebiased, kk)?;
    let lsb = bink(f, BinOp::BitwiseAnd, lsb, 1)?;
    let rn = bink(f, BinOp::IAdd, rebiased, (1 << (kk - 1)) - 1)?;
    let rn = bin(f, BinOp::IAdd, rn, lsb)?;
    let rn = bink(f, BinOp::ShiftRightLogical, rn, kk)?;
    let cap = c32(f, inf)?;
    let rn = f.glsl(GlslOp::UMin, &[rn, cap])?;
    // A subnormal (or zero) result: the significand shifted down, rounded to nearest even.
    let shift = c32(f, kk + 113)?;
    let shift = bin(f, BinOp::ISub, shift, e)?;
    let gone = bink(f, BinOp::UGreaterThanEqual, shift, 32)?;
    let one = c32(f, 1)?;
    let most = c32(f, 31)?;
    let sh = f.glsl(GlslOp::UClamp, &[shift, one, most])?;
    let sig = bink(f, BinOp::BitwiseAnd, a, 0x7f_ffff)?;
    let sig = bink(f, BinOp::BitwiseOr, sig, 0x80_0000)?;
    let low = bin(f, BinOp::ShiftRightLogical, sig, sh)?;
    let low = bink(f, BinOp::BitwiseAnd, low, 1)?;
    let less = bink(f, BinOp::ISub, sh, 1)?;
    let half = bin(f, BinOp::ShiftLeftLogical, one, less)?;
    let rs = bin(f, BinOp::IAdd, sig, half)?;
    let rs = bink(f, BinOp::ISub, rs, 1)?;
    let rs = bin(f, BinOp::IAdd, rs, low)?;
    let rs = bin(f, BinOp::ShiftRightLogical, rs, sh)?;
    let zero = c32(f, 0)?;
    let rs = select(f, gone, zero, rs)?;
    let is_normal = bink(f, BinOp::UGreaterThanEqual, e, 113)?;
    let r = select(f, is_normal, rn, rs)?;
    let negative = bin(f, BinOp::INotEqual, sign, zero)?;
    let r = select(f, negative, zero, r)?;
    let quiet = c32(f, inf | (1 << (mantissa - 1)))?;
    select(f, nan, quiet, r)
}

/// `t` of `v`'s magnitude with `v`'s sign: `transfer(|v|)`, negated where `v < 0`.
fn signed_transfer(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    v: Value,
    transfer: fn(&mut FunctionBuilder<'_>, Form, Value) -> BResult<Value>,
) -> BResult<Value> {
    let a = f.glsl(GlslOp::FAbs, &[v])?;
    let t = transfer(f, SCALAR, a)?;
    let minus = f.unary(UnOp::FNegate, t)?;
    let zero = cf(f, k, 0.0)?;
    let below = bin(f, BinOp::FOrdLessThan, v, zero)?;
    select(f, below, minus, t)
}

/// The extended-range colour code of value `v` (`Rule::Extended`): `RoundEven(clamp(v, lo, hi)
/// 510) + 384`, the code clamped to `[0, 1023]`, a NaN 0.
fn extended_code(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    v: Value,
    lo: f32,
    hi: f32,
) -> BResult<Value> {
    let v = number(f, k, v)?;
    let v = clamp(f, k, v, lo, hi)?;
    let v = fopk(f, k, BinOp::FMul, v, 510.0)?;
    let v = f.glsl(GlslOp::RoundEven, &[v])?;
    let c = f.convert(ConvOp::ConvertFToS, v, k.i32_)?;
    let c = f.convert(ConvOp::Bitcast, c, k.u32_)?;
    let c = bink(f, BinOp::IAdd, c, 384)?;
    let c = f.convert(ConvOp::Bitcast, c, k.i32_)?;
    let zero = f.module().constant(Constant::Scalar {
        ty: k.i32_,
        bits: ConstBits(0),
    })?;
    let zero = f.module().const_value(zero)?;
    let top = f.module().constant(Constant::Scalar {
        ty: k.i32_,
        bits: ConstBits(1023),
    })?;
    let top = f.module().const_value(top)?;
    let c = f.glsl(GlslOp::SClamp, &[c, zero, top])?;
    f.convert(ConvOp::Bitcast, c, k.u32_)
}

/// The value of field `fl` of a texel's `words` (`layout::Rule`).
fn value(f: &mut FunctionBuilder<'_>, k: &K<'_>, words: &[Value], fl: &Field) -> BResult<Value> {
    let b = bits(f, words, fl.offset, fl.mask())?;
    match fl.rule {
        Rule::Unorm => {
            let v = f.convert(ConvOp::ConvertUToF, b, k.f32_)?;
            fopk(f, k, BinOp::FMul, v, fl.scale())
        }
        Rule::Snorm => {
            let shift = 32 - fl.width;
            let v = bink(f, BinOp::ShiftLeftLogical, b, shift)?;
            let v = bink(f, BinOp::ShiftRightArithmetic, v, shift)?;
            let v = signed_float(f, k, v)?;
            let v = fopk(f, k, BinOp::FMul, v, fl.scale())?;
            let minus_one = cf(f, k, -1.0)?;
            f.glsl(GlslOp::FMax, &[v, minus_one])
        }
        Rule::Srgb8 => {
            let v = f.convert(ConvOp::ConvertUToF, b, k.f32_)?;
            let v = fopk(f, k, BinOp::FMul, v, SRGB_SCALE)?;
            pack::srgb_decode(f, SCALAR, v)
        }
        Rule::Float { exponent: 8, .. } => f.convert(ConvOp::Bitcast, b, k.f32_),
        Rule::Float { mantissa, .. } => small_float_value(f, b, mantissa),
        Rule::Extended { alpha } => {
            let c = if alpha {
                let lo = c32(f, 384)?;
                let hi = c32(f, 894)?;
                f.glsl(GlslOp::UClamp, &[b, lo, hi])?
            } else {
                b
            };
            let o = bink(f, BinOp::ISub, c, 384)?;
            let v = signed_float(f, k, o)?;
            fopk(f, k, BinOp::FMul, v, EXTENDED_SCALE)
        }
        Rule::ExtendedSrgb => {
            let o = bink(f, BinOp::ISub, b, 384)?;
            let v = signed_float(f, k, o)?;
            let v = fopk(f, k, BinOp::FMul, v, EXTENDED_SCALE)?;
            signed_transfer(f, k, v, pack::srgb_decode)
        }
        Rule::Padding(_) => cf(f, k, 0.0),
    }
}

/// Field `fl`'s code of value `v` (`layout::Rule`), in its low `width` bits.
fn code(f: &mut FunctionBuilder<'_>, k: &K<'_>, fl: &Field, v: Value) -> BResult<Value> {
    match fl.rule {
        Rule::Unorm => {
            let v = number(f, k, v)?;
            let v = clamp(f, k, v, 0.0, 1.0)?;
            let v = fopk(f, k, BinOp::FMul, v, fl.den() as f32)?;
            let v = f.glsl(GlslOp::RoundEven, &[v])?;
            f.convert(ConvOp::ConvertFToU, v, k.u32_)
        }
        Rule::Snorm => {
            let v = number(f, k, v)?;
            let v = clamp(f, k, v, -1.0, 1.0)?;
            let v = fopk(f, k, BinOp::FMul, v, fl.den() as f32)?;
            let v = f.glsl(GlslOp::RoundEven, &[v])?;
            let c = f.convert(ConvOp::ConvertFToS, v, k.i32_)?;
            let c = f.convert(ConvOp::Bitcast, c, k.u32_)?;
            bink(f, BinOp::BitwiseAnd, c, fl.mask())
        }
        Rule::Srgb8 => {
            let v = number(f, k, v)?;
            let v = clamp(f, k, v, 0.0, 1.0)?;
            let v = pack::srgb_encode(f, SCALAR, v)?;
            let v = fopk(f, k, BinOp::FMul, v, 255.0)?;
            let v = f.glsl(GlslOp::RoundEven, &[v])?;
            f.convert(ConvOp::ConvertFToU, v, k.u32_)
        }
        Rule::Float { exponent: 8, .. } => f.convert(ConvOp::Bitcast, v, k.u32_),
        Rule::Float { signed: true, .. } => half_bits_of(f, k, v),
        Rule::Float { mantissa, .. } => small_float_bits(f, k, v, mantissa),
        Rule::Extended { alpha } => {
            let (lo, hi) = if alpha { (0.0, 1.0) } else { (-1.0, 1.5) };
            extended_code(f, k, v, lo, hi)
        }
        Rule::ExtendedSrgb => {
            let v = number(f, k, v)?;
            let t = signed_transfer(f, k, v, pack::srgb_encode)?;
            extended_code(f, k, t, -1.0, 1.5)
        }
        Rule::Padding(p) => c32(f, p & fl.mask()),
    }
}

/// The `float` channels of a texel of the kernel's layout from its `words` (`cpu::decode`).
fn decode(f: &mut FunctionBuilder<'_>, k: &K<'_>, words: &[Value]) -> BResult<Vec<Value>> {
    match k.spec.layout.texel {
        Texel::Fields(fields) => fields
            .iter()
            .filter(|fl| fl.channel())
            .map(|fl| value(f, k, words, fl))
            .collect(),
        Texel::Rgb9e5 => {
            let w = words[0];
            let e = bink(f, BinOp::ShiftRightLogical, w, 27)?;
            let e = bink(f, BinOp::IAdd, e, 127 - 24)?;
            let scale = pow2_biased(f, k, e)?;
            let mut out = Vec::with_capacity(3);
            for c in 0..3 {
                let m = bits(f, words, 9 * c, 511)?;
                let m = f.convert(ConvOp::ConvertUToF, m, k.f32_)?;
                out.push(bin(f, BinOp::FMul, m, scale)?);
            }
            Ok(out)
        }
    }
}

/// `2^(24 - e)` from a shared exponent `e` (a `u32` value from 0 to 32).
fn rgb9e5_scale(f: &mut FunctionBuilder<'_>, k: &K<'_>, e: Value) -> BResult<Value> {
    let biased = c32(f, 24 + 127)?;
    let biased = bin(f, BinOp::ISub, biased, e)?;
    pow2_biased(f, k, biased)
}

/// `placed` ORed into word `word` of a texel's `words`.
fn place(
    f: &mut FunctionBuilder<'_>,
    words: &mut [Value; 4],
    placed: Value,
    word: u32,
) -> BResult<()> {
    let w = &mut words[word as usize];
    *w = bin(f, BinOp::BitwiseOr, *w, placed)?;
    Ok(())
}

/// The words of a texel of the kernel's layout from its `float` channels (`cpu::encode`).
fn encode(f: &mut FunctionBuilder<'_>, k: &K<'_>, v: &[Value]) -> BResult<[Value; 4]> {
    let zero = c32(f, 0)?;
    let mut words = [zero; 4];
    match k.spec.layout.texel {
        Texel::Fields(fields) => {
            let mut values = v.iter();
            for fl in fields {
                let c = match fl.rule {
                    Rule::Padding(_) => code(f, k, fl, zero)?,
                    _ => {
                        let v = values.next().copied().ok_or_else(missing_channel)?;
                        code(f, k, fl, v)?
                    }
                };
                let placed = match fl.offset % 32 {
                    0 => c,
                    shift => bink(f, BinOp::ShiftLeftLogical, c, shift)?,
                };
                place(f, &mut words, placed, fl.offset / 32)?;
            }
        }
        Texel::Rgb9e5 => {
            let mut c = Vec::with_capacity(3);
            for &x in v {
                let x = number(f, k, x)?;
                c.push(clamp(f, k, x, 0.0, 65408.0)?);
            }
            let most = f.glsl(GlslOp::FMax, &[c[0], c[1]])?;
            let most = f.glsl(GlslOp::FMax, &[most, c[2]])?;
            // The shared exponent: `max(floor(log2(most)), -16) + 16`, one more where the largest
            // mantissa rounds to 512.
            let eb = f.convert(ConvOp::Bitcast, most, k.u32_)?;
            let eb = bink(f, BinOp::ShiftRightLogical, eb, 23)?;
            let eb = bink(f, BinOp::BitwiseAnd, eb, 0xff)?;
            let least = c32(f, 111)?;
            let e = f.glsl(GlslOp::UMax, &[eb, least])?;
            let e = bink(f, BinOp::ISub, e, 111)?;
            let scale = rgb9e5_scale(f, k, e)?;
            let s = bin(f, BinOp::FMul, most, scale)?;
            let s = fopk(f, k, BinOp::FAdd, s, 0.5)?;
            let s = f.glsl(GlslOp::Floor, &[s])?;
            let full = cf(f, k, 512.0)?;
            let bump = bin(f, BinOp::FOrdEqual, s, full)?;
            let one = c32(f, 1)?;
            let step = select(f, bump, one, zero)?;
            let e = bin(f, BinOp::IAdd, e, step)?;
            let scale = rgb9e5_scale(f, k, e)?;
            let mut w = bink(f, BinOp::ShiftLeftLogical, e, 27)?;
            let top = c32(f, 511)?;
            for (i, x) in c.into_iter().enumerate() {
                let m = bin(f, BinOp::FMul, x, scale)?;
                let m = fopk(f, k, BinOp::FAdd, m, 0.5)?;
                let m = f.glsl(GlslOp::Floor, &[m])?;
                let m = f.convert(ConvOp::ConvertFToU, m, k.u32_)?;
                let m = f.glsl(GlslOp::UMin, &[m, top])?;
                let m = match i {
                    0 => m,
                    i => bink(f, BinOp::ShiftLeftLogical, m, 9 * i as u32)?,
                };
                w = bin(f, BinOp::BitwiseOr, w, m)?;
            }
            words[0] = w;
        }
    }
    Ok(words)
}

/// The taps of `coordinate` along one axis: `(x0, x1, w0, w1)` from the axis's table at word
/// `table` and the level above's `big` texels (`layout::axis_tap`).
fn axis(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    table: Value,
    coordinate: Value,
    big: Value,
) -> BResult<[Value; 4]> {
    let at = bink(f, BinOp::IMul, coordinate, TAP_WORDS)?;
    let at = bin(f, BinOp::IAdd, table, at)?;
    let x0 = param(f, k, at)?;
    let at = bink(f, BinOp::IAdd, at, 1)?;
    let t = param(f, k, at)?;
    let t = f.convert(ConvOp::Bitcast, t, k.f32_)?;
    let next = bink(f, BinOp::IAdd, x0, 1)?;
    let last = bink(f, BinOp::ISub, big, 1)?;
    let x1 = f.glsl(GlslOp::UMin, &[next, last])?;
    let one = cf(f, k, 1.0)?;
    let w0 = bin(f, BinOp::FSub, one, t)?;
    Ok([x0, x1, w0, t])
}

/// The words of the first level's texel `at` (`x`, `y`, and the layer or slice) fetched from
/// its sampled view, as the scratch buffer would hold them (`fetch.rs`): each component's code
/// recovered and placed at its bit offset, a texel narrower than a word in the low bits of the
/// first.
fn fetched(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    fetch: &Fetch,
    at: [Value; 3],
) -> BResult<Vec<Value>> {
    let uvec3 = f.module().ty(Type::Vector {
        elem: k.u32_,
        len: 3,
    })?;
    let coord = f.composite_construct(uvec3, &at)?;
    let ops = ImageOperands {
        lod: Some(c32(f, 0)?),
        ..ImageOperands::default()
    };
    let value = f.fetch(k.source, coord, &ops)?;
    let zero = c32(f, 0)?;
    let mut words = vec![zero; k.spec.layout.bytes.div_ceil(4) as usize];
    for (lane, component) in fetch.components.iter().enumerate() {
        let Some(component) = *component else {
            continue;
        };
        let v = f.composite_extract(value, &[lane as u32])?;
        let code = recovered(f, k, component, v)?;
        let shifted = match component.offset % 32 {
            0 => code,
            shift => bink(f, BinOp::ShiftLeftLogical, code, shift)?,
        };
        let w = &mut words[(component.offset / 32) as usize];
        *w = bin(f, BinOp::BitwiseOr, *w, shifted)?;
    }
    Ok(words)
}

/// The code of a fetched component `v` (a `float`, or a `u32` where the view is an integer
/// one), recovered by its rule (`fetch.rs`), in its low `bits` bits.
fn recovered(
    f: &mut FunctionBuilder<'_>,
    k: &K<'_>,
    component: Component,
    v: Value,
) -> BResult<Value> {
    let bits = component.bits;
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let scaled = |f: &mut FunctionBuilder<'_>, scale: u32| -> BResult<Value> {
        let c = cf(f, k, scale as f32)?;
        let product = f.binary(BinOp::FMul, v, c)?;
        f.glsl(GlslOp::RoundEven, &[product])
    };
    match component.recover {
        Recover::Unorm => {
            let r = scaled(f, mask)?;
            f.convert(ConvOp::ConvertFToU, r, k.u32_)
        }
        Recover::Snorm => {
            let r = scaled(f, mask >> 1)?;
            let r = f.convert(ConvOp::ConvertFToS, r, k.i32_)?;
            let r = f.convert(ConvOp::Bitcast, r, k.u32_)?;
            bink(f, BinOp::BitwiseAnd, r, mask)
        }
        Recover::Half => {
            let b = f.convert(ConvOp::Bitcast, v, k.u32_)?;
            half_bits(f, b)
        }
        Recover::Bits if bits == 32 => Ok(v),
        Recover::Bits => bink(f, BinOp::BitwiseAnd, v, mask),
    }
}

/// The half-float bits of `f32` bits `b` whose value is a half float, exactly: a normal or
/// subnormal half's value and sign, an infinity, and a NaN as a quiet NaN with the top ten bits
/// of its mantissa (`fetch.rs`). `b` holds a half's value: a smaller one reads as the zero of
/// its sign, a larger one as no value.
fn half_bits(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let sign = bink(f, BinOp::ShiftRightLogical, b, 16)?;
    let sign = bink(f, BinOp::BitwiseAnd, sign, 0x8000)?;
    let e = bink(f, BinOp::ShiftRightLogical, b, 23)?;
    let e = bink(f, BinOp::BitwiseAnd, e, 0xff)?;
    let m = bink(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let top = bink(f, BinOp::ShiftRightLogical, m, 13)?;
    // A normal half: exponent `e - 112` (of 1 to 30).
    let ne = bink(f, BinOp::ISub, e, 112)?;
    let ne = bink(f, BinOp::ShiftLeftLogical, ne, 10)?;
    let normal = bin(f, BinOp::BitwiseOr, ne, top)?;
    // A subnormal half (`e` of 103 to 112): the significand shifted down by `126 - e`.
    let hidden = bink(f, BinOp::BitwiseOr, m, 0x80_0000)?;
    let shift = c32(f, 126)?;
    let shift = bin(f, BinOp::ISub, shift, e)?;
    let most = c32(f, 31)?;
    let shift = f.glsl(GlslOp::UMin, &[shift, most])?;
    let subnormal = bin(f, BinOp::ShiftRightLogical, hidden, shift)?;
    // An infinity, or a NaN kept quiet.
    let zero = c32(f, 0)?;
    let is_nan = bin(f, BinOp::INotEqual, m, zero)?;
    let quiet = c32(f, 0x7e00)?;
    let inf = c32(f, 0x7c00)?;
    let special = select(f, is_nan, quiet, inf)?;
    let special = bin(f, BinOp::BitwiseOr, special, top)?;
    let low = c32(f, 103)?;
    let least = c32(f, 113)?;
    let all = c32(f, 255)?;
    let h = bin(f, BinOp::UGreaterThanEqual, e, low)?;
    let h = select(f, h, subnormal, zero)?;
    let is_normal = bin(f, BinOp::UGreaterThanEqual, e, least)?;
    let h = select(f, is_normal, normal, h)?;
    let is_special = bin(f, BinOp::IEqual, e, all)?;
    let h = select(f, is_special, special, h)?;
    bin(f, BinOp::BitwiseOr, h, sign)
}

/// Destination texel `i`'s words: its taps from the level's tap tables (eight in 3D; four where
/// the level above is one texel deep), each source texel's words read once and decoded, each
/// channel the weighted sum of its taps in tap order (a tap of weight 0 left out), the sums
/// encoded (`cpu::level`).
fn filtered(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value) -> BResult<[Value; 4]> {
    let mut small = [i; 3];
    let mut big = [i; 3];
    let mut tables = [i; 3];
    let axes = if k.spec.depth { 3 } else { 2 };
    for a in 0..axes {
        small[a as usize] = paramk(f, k, header::EXTENT + a)?;
        big[a as usize] = paramk(f, k, header::ABOVE + a)?;
        tables[a as usize] = paramk(f, k, header::TAPS + a)?;
    }
    let x = bin(f, BinOp::UMod, i, small[0])?;
    let r = bin(f, BinOp::UDiv, i, small[0])?;
    let y = bin(f, BinOp::UMod, r, small[1])?;
    let r = bin(f, BinOp::UDiv, r, small[1])?;
    let ax = axis(f, k, tables[0], x, big[0])?;
    let ay = axis(f, k, tables[1], y, big[1])?;
    // The depth axis, where it has two taps: the slices, the layer's first slice, the taps.
    let depth = if k.spec.depth {
        let z = bin(f, BinOp::UMod, r, small[2])?;
        let layer = bin(f, BinOp::UDiv, r, small[2])?;
        let first = bin(f, BinOp::IMul, layer, big[2])?;
        Some((first, axis(f, k, tables[2], z, big[2])?))
    } else {
        None
    };
    let count = if k.spec.depth { 8 } else { 4 };
    let zero = cf(f, k, 0.0)?;
    let mut sums: Vec<Value> = Vec::new();
    for t in 0..count {
        let (sx, wx) = if t & 1 == 1 {
            (ax[1], ax[3])
        } else {
            (ax[0], ax[2])
        };
        let (sy, wy) = if (t >> 1) & 1 == 1 {
            (ay[1], ay[3])
        } else {
            (ay[0], ay[2])
        };
        let wxy = bin(f, BinOp::FMul, wx, wy)?;
        // The image: the layer's (`r`, one slice each), or the layer's depth tap's.
        let (image, w) = match depth {
            Some((first, az)) => {
                let (sz, wz) = if (t >> 2) & 1 == 1 {
                    (az[1], az[3])
                } else {
                    (az[0], az[2])
                };
                (
                    bin(f, BinOp::IAdd, first, sz)?,
                    bin(f, BinOp::FMul, wxy, wz)?,
                )
            }
            None => (r, wxy),
        };
        let words = match &k.spec.fetch {
            // The first level's view: a 2D array's layer or a 3D texture's slice is `image`.
            Some(fetch) => fetched(f, k, fetch, [sx, sy, image])?,
            None => {
                let index = bin(f, BinOp::IMul, image, big[1])?;
                let index = bin(f, BinOp::IAdd, index, sy)?;
                let index = bin(f, BinOp::IMul, index, big[0])?;
                let index = bin(f, BinOp::IAdd, index, sx)?;
                texel(f, k, index)?
            }
        };
        let values = decode(f, k, &words)?;
        let weightless = bin(f, BinOp::FOrdEqual, w, zero)?;
        for (c, v) in values.into_iter().enumerate() {
            let term = bin(f, BinOp::FMul, w, v)?;
            let term = select(f, weightless, zero, term)?;
            match sums.get_mut(c) {
                Some(sum) => *sum = bin(f, BinOp::FAdd, *sum, term)?,
                None => sums.push(term),
            }
        }
    }
    encode(f, k, &sums)
}

/// Destination texel `i`'s `words` written: a sub-word texel ORed into the zeroed level, a
/// wider one stored word by word.
fn store_texel(f: &mut FunctionBuilder<'_>, k: &K<'_>, i: Value, words: [Value; 4]) -> BResult<()> {
    let zero = c32(f, 0)?;
    let bytes = k.spec.layout.bytes;
    if bytes < 4 {
        let byte = bink(f, BinOp::IMul, i, bytes)?;
        let word = bink(f, BinOp::ShiftRightLogical, byte, 2)?;
        let lane = bink(f, BinOp::BitwiseAnd, byte, 3)?;
        let lane = bink(f, BinOp::ShiftLeftLogical, lane, 3)?;
        let v = bin(f, BinOp::ShiftLeftLogical, words[0], lane)?;
        let p = f.access_chain(k.destination, &[zero, word])?;
        f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, v)?;
        return Ok(());
    }
    let per = bytes / 4;
    let first = bink(f, BinOp::IMul, i, per)?;
    for (j, w) in words.into_iter().take(per as usize).enumerate() {
        let at = match j {
            0 => first,
            j => bink(f, BinOp::IAdd, first, j as u32)?,
        };
        let p = f.access_chain(k.destination, &[zero, at])?;
        f.store(p, w, MemAccess::default())?;
    }
    Ok(())
}

/// The filter module of `spec` as a typed module, before serialization: the module
/// [`super::mipmap_filter_module`] serializes, which the CPU interpreter runs. Its bindings are
/// the symbols `SOURCE` (or `SOURCE_IMAGE`, the first level's sampled view, where the spec
/// fetches), `DESTINATION` and `PARAMS` (the module docs).
pub(super) fn typed_module(spec: &Spec) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let source = match &spec.fetch {
        Some(view) => {
            let sampled_type = if fetch::integer(&view.components) {
                u32_
            } else {
                f32_
            };
            let (dim, arrayed) = if view.volume {
                (Dim::D3, false)
            } else {
                (Dim::D2, true)
            };
            let ty = b.ty(Type::Image(ImageType {
                sampled_type,
                dim,
                depth: false,
                arrayed,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            b.resource(SOURCE_IMAGE, StorageClass::UniformConstant, ty)?
        }
        None => words_buffer(&mut b, SOURCE, u32_)?,
    };
    let destination = words_buffer(&mut b, DESTINATION, u32_)?;
    let params = words_buffer(&mut b, PARAMS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let source = f.module().global_value(source)?;
    // The sampled view is loaded once, where every tap's fetch reads it.
    let source = match spec.fetch {
        Some(_) => f.load(source, MemAccess::default())?,
        None => source,
    };
    let destination = f.module().global_value(destination)?;
    let params = f.module().global_value(params)?;
    let k = K {
        u32_,
        i32_,
        f32_,
        source,
        destination,
        params,
        spec,
    };
    let texels = paramk(&mut f, &k, header::TEXELS)?;
    let row = paramk(&mut f, &k, header::ROW)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let gx = f.composite_extract(invocation, &[0])?;
    let gy = f.composite_extract(invocation, &[1])?;
    let i = bin(&mut f, BinOp::IMul, gy, row)?;
    let i = bin(&mut f, BinOp::IAdd, i, gx)?;
    let inside = bin(&mut f, BinOp::ULessThan, i, texels)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        let words = filtered(f, &k, i)?;
        store_texel(f, &k, i, words)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, body, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}
