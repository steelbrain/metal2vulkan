//! The conversion of a texel to a `half` result on a device that samples through Metal's own
//! texture unit (`VkProperty::MetalSampler`: MoltenVK on Apple's GPU).
//!
//! A read, gather, or sample with a `half` result takes the device's `f32` texel (Metal's `float`
//! result of the same access there, filtered where the access filters) rounded to `half` to
//! nearest even by the module's rounding (`narrow::round_texel`), on every device: the device's
//! native conversion. Metal's own `half` result differs from it for a few classes, which the
//! module notes (`PrecisionNote::HalfTexelConversion`, `texture::half_texels`): a 32-bit float
//! texel's sample (Metal rounds toward zero, saturating at `±65504`), the 16-bit normalized grids
//! (Metal's `float` value of `65455 / 65535`, `65519 / 65535` and `32759 / 32767` rounds twice),
//! and a format whose conversion is not measured. Every other class's `float` result rounds to
//! Metal's `half` (`metal_float_steps_round_to_their_half`).
//!
//! **The routine** ([`define`]): `fn(x: vec4, flags: u32, channel: u32) -> vec4`, called with the
//! slot's `SpecKey::TexelFormat` class (`metal::texel_format`), keeps one more of Metal's rules: a
//! two-plane YCbCr format's one texel on the grid `8192` (`TexelFormat::YCBCR_8192`: a read or a
//! sample of one texel, `TexelLanes::Texel`, and a 2D gather's lanes, `Gathered`) rounds its step
//! with ties away from zero (`texel_round::f32_to_half_from_even`), as Metal's does (every channel
//! of five formats on the grid `8192`, and the gathers; measured on an Apple M3 Max, macOS 26.7).
//! Every other texel, a value that may be filtered among them (`TexelLanes::Filtered`), rounds to
//! nearest even.
//!
//! **The plain arm.** A slot's access whose texel an image instruction hands back
//! ([`TexelOrigin::Image`]) converts it under the slot's `SpecKey::PlainHalfTexels`
//! specialization, which states that the bound class has no such YCbCr field
//! (`TexelFormat::plain_on_metal_sampler`): there by the device's conversion of the texel itself
//! (`narrow::round_texel`, which Metal's compiler may fold into the image instruction), else by
//! the routine, so specialization leaves one of them. The class stays push data, so only a slot
//! that sees both kinds of class makes a second pipeline. A computed texel keeps the routine.
//!
//! Elsewhere the device's own sampler hands back its `float` texel, and the `half` is that texel
//! rounded to nearest even, no routine (`texture::device_texels`).

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::splat;
use crate::lower::arith::texel_round;
use crate::metal::texel_format::TexelFormat;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{FuncId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, Type as SType};

/// Where the texel a conversion takes comes from: what Metal's compiler may fold its conversion
/// to `half` into ([`narrow::round_texel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelOrigin {
    /// An image instruction's texel (a read, fetch, sample or gather), or lanes and selections of
    /// such texels and constants: no arithmetic computes it.
    Image,
    /// A value the translation computes from texels (an emulated filter's sum).
    Computed,
}

/// Which lanes of an access's value hold which channel of one texel: whether a YCbCr field's one
/// texel is there to round with ties away (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLanes {
    /// One texel's channels, red to alpha: a read, or a sample whose sampler reads one texel of
    /// one level.
    Texel,
    /// Every lane channel `c` (0 red to 3 alpha) of its own texel: a 2D gather's.
    Gathered(u32),
    /// A value that may be filtered: a sample through a sampler that may filter, or a cube
    /// gather (its corner texels are made up).
    Filtered,
}

impl TexelLanes {
    /// The routine's `channel` argument: the gathered channel, 4 for a texel, 5 for a value that
    /// may be filtered.
    pub fn code(self) -> u32 {
        match self {
            TexelLanes::Gathered(c) => c.min(3),
            TexelLanes::Texel => TEXEL_LANES,
            TexelLanes::Filtered => FILTERED_LANES,
        }
    }

    /// The argument as a constant of `fb`'s module.
    pub fn value(self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let c = fb.u32_const(self.code())?;
        fb.module().const_value(c)
    }
}

/// The `channel` argument of one texel's lanes ([`TexelLanes::Texel`]).
pub const TEXEL_LANES: u32 = 4;

/// The `channel` argument of a value that may be filtered ([`TexelLanes::Filtered`]).
pub const FILTERED_LANES: u32 = 5;

/// An upper estimate of the routine's words, counted once per module that defines it, and the bound
/// on its size in every `QuantizeForm`, which a device that inlines it at each call compiles once
/// per call (`the_half_texel_routine_words_bound_its_emission`).
pub const ROUTINE_WORDS: u64 = 400;

/// An upper estimate of one call's words: the call and its two constant arguments.
pub const CALL_WORDS: u32 = 16;

/// The census name of the routine.
pub const CENSUS_NAME: &str = "texture.half_texel";

fn fform() -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    }
}

/// The held form of a texel's four lanes as `half`s.
fn hform() -> Form {
    Form {
        held: Held::Float(FloatFmt::Half),
        lanes: 4,
    }
}

/// Define the routine `fn(x: vec4, flags: u32, channel: u32) -> vec4` in `mb` (module docs;
/// `channel` is [`TexelLanes::code`]): a YCbCr field's one texel on the grid `8192` rounded to
/// nearest with ties away from zero, every other texel to nearest even (`narrow::round_texel`).
/// It reads the packed field's top bit ([`TexelFormat::YCBCR_8192`]) of the class, and `channel`.
pub fn define(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let vf = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let mut fb = mb.function(vf)?;
    let x = fb.param(vf)?;
    let flags = fb.param(u32_)?;
    let channel = fb.param(u32_)?;
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| {
        let s = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        };
        splat(fb, s, u64::from(v))
    };
    let m = u1(&mut fb, TexelFormat::YCBCR_8192)?;
    let z = u1(&mut fb, 0)?;
    let field = fb.binary(BinOp::BitwiseAnd, flags, m)?;
    let ycbcr = fb.binary(BinOp::INotEqual, field, z)?;
    let texel_lanes = u1(&mut fb, TEXEL_LANES)?;
    let single = fb.binary(BinOp::ULessThanEqual, channel, texel_lanes)?;
    let ycbcr = fb.binary(BinOp::LogicalAnd, ycbcr, single)?;
    let even = narrow::round_texel(&mut fb, hform(), x)?;
    let b = fb.module().ty(SType::Bool)?;
    let no = fb.module().constant(Constant::Scalar {
        ty: b,
        bits: ConstBits(0),
    })?;
    let toward_zero = fb.module().const_value(no)?;
    let away = texel_round::f32_to_half_from_even(&mut fb, fform(), (x, even), toward_zero)?;
    let r = fb.select(ycbcr, away, even)?;
    fb.ret(Some(r))?;
    fb.finish()
}

/// The held `half` texel of image texel `x` (a `vec4` of `f32`), converted by `routine` under the
/// slot's treatment constant `flags` (its `TexelFormat` specialization) and `lanes`, its
/// [`TexelLanes::code`] (a constant, or the shader's choice where a runtime sampler's word decides
/// whether a sample reads one texel).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    routine: FuncId,
    x: SValue,
    flags: SValue,
    lanes: SValue,
) -> BResult<SValue> {
    // Every call of the routine returns its texel.
    fb.call(routine, &[x, flags, lanes])?
        .ok_or(BuildError::Operand {
            op: crate::spv::Op::FunctionCall,
            rule: OperandRule::Callee,
        })
}
