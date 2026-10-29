//! Narrow floats held widened: `half` and `bfloat` values are held in an `f32` that is a value of
//! the narrow format wherever the device's own rounding to `half` rounds.
//!
//! An operation on narrow floats runs in `f32` and rounds its result back to the format
//! ([`round`]) by the device's own rounding to `half`, in the module's [`QuantizeForm`]. Where the
//! profile has `shaderFloat16` and the float controls that make the conversion Metal's by the
//! Vulkan spec (`RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits, and
//! the modes the device's independence ties to them), it is that conversion and its widening, each
//! decorated `NoContraction`, in an entry declaring those modes ([`QuantizeForm::Controlled`],
//! [`controlled_modes`]): exact on every such device whose driver keeps the pair (not NVIDIA's,
//! `DriverWorkaround::HalfConversionPairFolded`), with no fact and no note. Elsewhere, where the
//! profile has `shaderFloat16`, it is the device's conversion to 16 bits and its widening
//! ([`QuantizeForm::Converted`]), recording `PrecisionNote::NativeHalfConversion`; without it, the
//! device's `OpQuantizeToF16` alone ([`QuantizeForm::Device`]), whose result below the smallest
//! normal half may be a zero, recording `PrecisionNote::NativeHalfQuantize` on every such profile
//! ([`quantizes`]); a texel's `half` takes [`round`] and keeps the NaN the device decoded
//! ([`round_texel`]) unless the profile leaves NaN bits to the device under native half rounding
//! or the float controls; `bfloat` rounds through an
//! exact integer round-to-nearest-even of the `f32` bits. Where the device's 16-bit `+ - *`, kept
//! apart, is Metal's `half` arithmetic and the profile has `shaderFloat16`, a `half` `+ - *` is
//! that instruction between exact conversions instead, with no rounding here (`ops::f16_binary`):
//! where it is measured to be (`DriverDeviation::MetalHalfArithmetic`), and under the float
//! controls of [`QuantizeForm::Controlled`], which make it so by the Vulkan spec and which the
//! entry then declares ([`computes_in_16_bits`]). For `+ - * /` and square root the double
//! rounding (the exact result to `f32`, then to the narrow format) is innocuous, because `f32`
//! carries at least `2p + 2` significand bits for both formats; a `half` fused multiply-add's is
//! not, so it is the device's 16-bit one where that is Metal's, and else the device's `f32` one
//! rounded to `half`, noted (`super::fma`), and a `bfloat` one is corrected ([`bf16_fma`]).
//!
//! A `half` read from its bits is the device's widening of them on every profile ([`from_bits`]):
//! the bits read as a 16-bit float and converted where the module can hold one
//! ([`device_from_bits`]), else `GLSL.std.450 UnpackHalf2x16` of the bits as the low half of a
//! word ([`unpacked_from_bits`]), which needs no 16-bit type: the device's value of every `half`
//! but a NaN's bits and, on a device that may flush 16-bit denormals, a denormal, each recorded
//! ([`widens_bits`]). A `half`'s bits from its value, where they are not known from the bits it
//! was read from, are the device's narrowing of it on every profile ([`to_bits`]): its conversion
//! to a 16-bit float read as a word where the module can hold one ([`converted_bits`]), else
//! `GLSL.std.450 PackHalf2x16` ([`packed_bits`]), the same two notes recorded
//! ([`encodes_bits`]). Every other crossing is exact: a `bfloat`'s move between a value and its
//! format bits with integer shifts (NaN payloads included); an integer converts to `half` through
//! `f32` without
//! double rounding (any integer the `f32` step rounds is at least `2^24`, far above the largest
//! half) and rounds by integer arithmetic, not `OpQuantizeToF16`, around a driver's defect
//! ([`round_integer`]), an integer converts to `bfloat` through a round-to-odd `f32`
//! ([`int_to_f32_odd`], [`int64_to_f32_odd`]), a `double` narrows through a round-to-odd
//! `f32` ([`f64_to_f32_odd`]), and a `bfloat` fused multiply-add corrects its one inexact case
//! ([`bf16_fma`]), so the final rounding is the only one that counts.

use crate::budget::Meter;
use crate::device::profile::{DriverWorkaround, Independence};
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::pipeline::artifact::PrecisionNote;
use crate::spirv::builder::{FunctionBuilder, NarrowFmt, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConstBits, ConstKind, Constant, ConvOp, ExecMode, GlslOp, Inst, Type, TypedModule, UnOp,
    ValueOrigin,
};

/// The unsigned 32-bit integer form with `f`'s lanes.
pub fn u32_form(f: Form) -> Form {
    f.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// An upper estimate of the SPIR-V words of one [`round`] on form `f`, where the module's
/// [`QuantizeForm`] is not at hand: [`round_words_in`] of [`QuantizeForm::Device`], the largest.
pub fn round_words(f: Form) -> u32 {
    round_words_in(QuantizeForm::Device, f)
}

/// An upper estimate of the SPIR-V words of one [`round`] on form `f` in quantize form `q`: for
/// `half`, the device's quantize or its conversion and widening, at most two instructions with
/// their types and decorations, within the bound of 64 words and 4 per lane, in every form; a
/// dozen instructions for `bfloat`.
pub fn round_words_in(q: QuantizeForm, f: Form) -> u32 {
    let lanes = u32::from(f.lanes);
    match (f.float_fmt(), q) {
        (
            Some(FloatFmt::Half),
            QuantizeForm::Device | QuantizeForm::Converted | QuantizeForm::Controlled,
        ) => 64 + 4 * lanes,
        (Some(FloatFmt::BFloat), _) => 96,
        _ => 0,
    }
}

/// `x` (an `f32` of form `f`) rounded to `f`'s narrow format; the value itself for `float` and
/// `double` forms.
///
/// `half`: the device's rounding to `half`, in the module's [`QuantizeForm`].
/// [`QuantizeForm::Device`]: the device's `OpQuantizeToF16(x)` alone. The SPIR-V specification lets
/// the quantize answer a zero for a magnitude too small for a normal half, and a device may (NIR's
/// `fquantize2f16` flushes every one), where Metal answers the half denormal; Vulkan leaves its
/// rounding mode to the device. The module records `PrecisionNote::NativeHalfQuantize`
/// ([`quantizes`]).
/// [`QuantizeForm::Converted`]: [`converted`]'s alone, the device's conversion.
/// [`QuantizeForm::Controlled`]: [`controlled`]'s alone, Metal's value by the Vulkan spec.
pub fn round(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match f.held {
        Held::Float(FloatFmt::Half) => {
            let wide = f.with(Held::Float(FloatFmt::F32));
            match fb.module().quantize_form() {
                QuantizeForm::Converted => converted(fb, wide, x),
                QuantizeForm::Controlled => controlled(fb, wide, x),
                QuantizeForm::Device => fb.unary(UnOp::QuantizeToF16, x),
            }
        }
        Held::Float(FloatFmt::BFloat) => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let r = bf16_round_bits(fb, u, b)?;
            let ft = ty(fb, f)?;
            fb.convert(ConvOp::Bitcast, r, ft)
        }
        _ => Ok(x),
    }
}

/// How profile `p`'s modules round an `f32` to `half` ([`QuantizeForm`]): the device's conversion
/// to 16 bits and back under the float controls that make it Metal's by the Vulkan spec where the
/// profile has `shaderFloat16` and them ([`QuantizeForm::Controlled`], [`controlled_modes`]), else
/// that conversion as it is where the profile has `shaderFloat16` ([`QuantizeForm::Converted`]),
/// else the device's `OpQuantizeToF16` ([`QuantizeForm::Device`]).
pub fn quantize_form(p: &DeviceProfile) -> QuantizeForm {
    let float16 = p.fields().leaf.float16;
    if float16 && controlled_modes(p).is_some() {
        QuantizeForm::Controlled
    } else if float16 {
        QuantizeForm::Converted
    } else {
        QuantizeForm::Device
    }
}

/// The precision note of a module whose `half` roundings take `form`, the device's own rounding:
/// `NativeHalfConversion` for [`QuantizeForm::Converted`], `NativeHalfQuantize` for
/// [`QuantizeForm::Device`] (a quantize's result below the smallest normal half is the device's on
/// every profile, which no measurement covers). `None` for the float controls' conversion, Metal's
/// by the Vulkan spec.
pub fn native_rounding(form: QuantizeForm) -> Option<PrecisionNote> {
    match form {
        QuantizeForm::Converted => Some(PrecisionNote::NativeHalfConversion),
        QuantizeForm::Device => Some(PrecisionNote::NativeHalfQuantize),
        QuantizeForm::Controlled => None,
    }
}

/// `x` (`f32` lanes of form `wide`) rounded to `half` by the device's conversion to 16 bits,
/// `OpFConvert`, and widened back ([`QuantizeForm::Converted`]), recorded as
/// `PrecisionNote::NativeHalfConversion`. The device's compiler may fold the conversion into the
/// instruction computing `x`: Metal's compiler then rounds toward zero (`a1_half_tanh_regimes`,
/// case 27353187: 1,204 of its 16,384 halves one ulp toward zero on MoltenVK). NVIDIA's compiler
/// (RTX 3090) drops the pair, and so does lavapipe's where measured, so there the value is not
/// rounded at all (the note's stated bound), and the store's conversion to bits rounds it
/// ([`converted_bits`]).
fn converted(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let h = match wide.lanes {
        1 => e,
        len => fb.module().ty(Type::Vector { elem: e, len })?,
    };
    let w = ty(fb, wide)?;
    let n = fb.convert(ConvOp::FConvert, x, h)?;
    fb.convert(ConvOp::FConvert, n, w)
}

/// `x` (`f32` lanes of form `wide`) rounded to `half` by the device's conversion to 16 bits and
/// its widening, each an `OpFConvert` decorated `NoContraction` ([`QuantizeForm::Controlled`]),
/// in an entry that declares [`controlled_modes`]: Metal's `half(float)` on every `f32` by the
/// Vulkan spec. `NoContraction` keeps the pair: on a float operation without it Vulkan assumes
/// `AllowTransform`, under which Mesa's compiler folds the widening of a narrowing into nothing and
/// hands `x` on unrounded.
fn controlled(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let h = match wide.lanes {
        1 => e,
        len => fb.module().ty(Type::Vector { elem: e, len })?,
    };
    let w = ty(fb, wide)?;
    let n = fb.convert_no_contraction(x, h)?;
    fb.convert_no_contraction(n, w)
}

/// The execution modes an entry that rounds to `half` in [`QuantizeForm::Controlled`] declares on
/// profile `p`, or `None` where the profile lacks one or its driver folds the pair
/// (`DriverWorkaround::HalfConversionPairFolded`): `RoundingModeRTE`, `DenormPreserve` and
/// `SignedZeroInfNanPreserve` for 16 bits, which make an `OpFConvert` from 32 to 16 bits Metal's
/// `half(float)`. The Vulkan spec ("Precision and Operation of SPIR-V Instructions"): conversions
/// between types are "Correctly rounded", "according to IEEE 754 roundTiesToEven" where the entry
/// declares `RoundingModeRTE`; under `DenormPreserve` "the denormalized result and any denormalized
/// operands must be preserved"; and `OpFConvert` is among the operations that keep signed zeros,
/// infinities and NaNs in an entry declaring `SignedZeroInfNanPreserve` for one of its widths.
/// Then the modes the device's independence ties to them (RuntimeSpirv VUIDs 06289-06292):
/// `RoundingModeRTE` for 64 bits where rounding modes are independent for 32 bits only, and for 32
/// and 64 bits where they are not independent; `DenormPreserve` for 64 bits where denormal modes
/// are independent for 32 bits only. A device whose denormal modes are not independent at all
/// would have to preserve `float` denormals with them, changing how the module's `float`
/// arithmetic reads them (`PrecisionNote::DeviceDenormals`), so the form is not taken there.
pub fn controlled_modes(p: &DeviceProfile) -> Option<Vec<ExecMode>> {
    if p.fields()
        .workarounds
        .contains(&DriverWorkaround::HalfConversionPairFolded)
    {
        return None;
    }
    let fc = p.fields().float_controls;
    let half = fc.f16;
    if !(half.rounding_rte && half.denorm_preserve && half.signed_zero_inf_nan_preserve) {
        return None;
    }
    let mut modes = vec![
        ExecMode::RoundingModeRte { width: 16 },
        ExecMode::DenormPreserve { width: 16 },
        ExecMode::SignedZeroInfNanPreserve { width: 16 },
    ];
    match fc.rounding_independence {
        Independence::All => {}
        Independence::Only32Bit if fc.f64.rounding_rte => {
            modes.push(ExecMode::RoundingModeRte { width: 64 });
        }
        Independence::None if fc.f32.rounding_rte && fc.f64.rounding_rte => {
            modes.push(ExecMode::RoundingModeRte { width: 32 });
            modes.push(ExecMode::RoundingModeRte { width: 64 });
        }
        Independence::Only32Bit | Independence::None => return None,
    }
    match fc.denorm_independence {
        Independence::All => {}
        Independence::Only32Bit if fc.f64.denorm_preserve => {
            modes.push(ExecMode::DenormPreserve { width: 64 });
        }
        Independence::Only32Bit | Independence::None => return None,
    }
    Some(modes)
}

/// Whether module `m` rounds to `half` through [`converted`] or [`controlled`]: an `OpFConvert` of
/// an `f32` to 16 bits whose result another widens back. The module then records
/// `PrecisionNote::NativeHalfConversion` ([`QuantizeForm::Converted`]) or declares
/// [`controlled_modes`] ([`QuantizeForm::Controlled`]). A `half` fused multiply-add on the
/// device's 16-bit `Fma` (`super::fma`) converts its operands, each a `half` already, exactly, and
/// widens the `Fma`'s
/// result, which is no such pair. Two passes over the instructions, with one bit per value of the
/// module charged to `meter` while they run.
pub fn rounds_by_conversion(m: &TypedModule, meter: &mut Meter) -> Result<bool, Error> {
    // The width of `v`'s float lanes, if it is a float or a vector of floats.
    let width = |v: Value| -> Option<u8> {
        let ty = m.value(v).ok()?.ty;
        let ty = match m.types().get(ty).ok()? {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        match m.types().get(ty).ok()? {
            Type::Float { width } => Some(*width),
            _ => None,
        }
    };
    let fconverts = || {
        m.blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|n| match (&n.inst, n.result) {
                (
                    Inst::Convert {
                        op: ConvOp::FConvert,
                        a,
                        ..
                    },
                    Some(r),
                ) => Some((*a, r)),
                _ => None,
            })
    };
    let words = m.values().len().div_ceil(64);
    meter.charge(8 * words)?;
    let mut narrowed = vec![0u64; words];
    let mut any = false;
    for (a, r) in fconverts() {
        if width(a) == Some(32) && width(r) == Some(16) {
            if let Some(w) = narrowed.get_mut(r.index() / 64) {
                *w |= 1 << (r.index() % 64);
                any = true;
            }
        }
    }
    let is_narrowed = |v: Value| {
        narrowed
            .get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    };
    let found = any && fconverts().any(|(a, r)| is_narrowed(a) && width(r) == Some(32));
    drop(narrowed);
    meter.release(8 * words);
    Ok(found)
}

/// An upper estimate of the SPIR-V words of one [`round_texel`] on form `f`: [`round`]'s, and the
/// NaN test's four instructions with their two constants.
pub fn round_texel_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => round_words(f) + 40,
        _ => 0,
    }
}

/// Texel value `x` (`f32` lanes of form `f`: the device's decode of an image texel) as a texture
/// access hands it to the function in `f`'s narrow format. A `half` is [`round`]ed, which keeps
/// half subnormals on their grid to nearest even where the module converts (a device's quantize,
/// [`QuantizeForm::Device`], may answer one as its signed zero,
/// `PrecisionNote::NativeHalfQuantize`), so a half-float texel, which the device decodes to its
/// exact `f32` value (a half subnormal is a normal `f32`), keeps its value: every subnormal, both
/// zeros, both infinities, and `65504`. A NaN is the device's decode itself, since
/// `OpQuantizeToF16` may return another NaN; it is told by its bits, so the test holds whatever
/// float modes the device compiles the module under. Under native half rounding
/// ([`native_rounding`]) the rounding's own NaN stands instead: the note the module records bounds
/// a NaN's encoding to the device's, so the keep's four instructions a texel are left out. So too
/// under the float controls ([`QuantizeForm::Controlled`], `ModuleBuilder::native_half_nans`): the
/// conversion keeps a NaN a NaN, its bits the device's, as the texel's decode is
/// (`PrecisionNote::DeviceNanEncoding`, which the access records). Another form is [`round`]ed.
pub fn round_texel(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let native = fb.module().native_half();
    match f.held {
        Held::Float(FloatFmt::Half) if native || fb.module().native_half_nans() => round(fb, f, x),
        Held::Float(FloatFmt::Half) => {
            let r = round(fb, f, x)?;
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let abs = splat(fb, u, 0x7FFF_FFFF)?;
            let mag = bin(fb, BinOp::BitwiseAnd, b, abs)?;
            let inf = splat(fb, u, 0x7F80_0000)?;
            let nan = bin(fb, BinOp::UGreaterThan, mag, inf)?;
            fb.select(nan, x, r)
        }
        _ => round(fb, f, x),
    }
}

/// An upper estimate of the SPIR-V words of one [`round_integer`] on form `f`: twelve instructions
/// and eight constants for `half` (a vector's constants carry their lanes); [`round`]'s for every
/// other form.
pub fn round_integer_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => 136 + 8 * u32::from(f.lanes),
        _ => round_words(f),
    }
}

/// `x` (an `f32` of form `f`: an integer converted to `f32`, so exact wherever its magnitude is
/// below `2^24` and at least `2^24` elsewhere) rounded to `f`'s narrow format; the value itself
/// for `float` and `double` forms, [`round`] for `bfloat`.
///
/// `half`: by integer arithmetic on the `f32` bits, with no `OpQuantizeToF16`: a workaround for a
/// driver defect, not a rounding choice. NVIDIA's `OpQuantizeToF16` of an integer conversion keeps
/// the integer unrounded (driver 615.71.09: 2049 stays 2049, which is no `half`), so the device's
/// rounding cannot stand there; an integer's `half` has one right answer, which this computes on
/// every device and in every rounding mode. A nonzero integer below `2^24` is a normal `f32` whose
/// half keeps the top 11 of its 24 significand bits, so adding `0xFFF` plus the lowest kept bit and
/// clearing the 13 bits below it rounds the exact value to nearest even, once; a carry out of the
/// fraction raises the exponent, as it must. Zero stays zero. A result above 65504 is the infinity
/// of the integer's sign: those are exactly the integers from 65520 up (65519 rounds to 65504;
/// 65520, halfway between 65504 and `2^16`, rounds to the even `2^16`), and every integer the `f32`
/// step rounded, which is at least `2^24`. Metal answers every one of these (measured: 2051 is
/// 2052, 4097 is 4096, 65519 is 65504).
pub fn round_integer(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match f.held {
        Held::Float(FloatFmt::Half) => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let k13 = splat(fb, u, 13)?;
            let one = splat(fb, u, 1)?;
            let lsb = bin(fb, BinOp::ShiftRightLogical, b, k13)?;
            let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
            let half_ulp = splat(fb, u, 0xFFF)?;
            let bias = bin(fb, BinOp::IAdd, half_ulp, lsb)?;
            let r = bin(fb, BinOp::IAdd, b, bias)?;
            let keep = splat(fb, u, 0xFFFF_E000)?;
            let r = bin(fb, BinOp::BitwiseAnd, r, keep)?;
            // Past the largest half: the infinity of the sign.
            let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
            let mag = bin(fb, BinOp::BitwiseAnd, r, abs_mask)?;
            let largest = splat(fb, u, 0x477F_E000)?;
            let over = bin(fb, BinOp::UGreaterThan, mag, largest)?;
            let sign_mask = splat(fb, u, 0x8000_0000)?;
            let sign = bin(fb, BinOp::BitwiseAnd, b, sign_mask)?;
            let inf = splat(fb, u, 0x7F80_0000)?;
            let inf = bin(fb, BinOp::BitwiseOr, sign, inf)?;
            let r = fb.select(over, inf, r)?;
            let ft = ty(fb, f)?;
            fb.convert(ConvOp::Bitcast, r, ft)
        }
        _ => round(fb, f, x),
    }
}

/// Whether module `m` rounds to `half` through `OpQuantizeToF16` ([`round`] and [`round_texel`]
/// in [`QuantizeForm::Device`]): the module then records `PrecisionNote::NativeHalfQuantize`
/// ([`native_rounding`]) on every profile: Vulkan leaves the quantize's rounding mode, and its
/// result below the smallest normal half, to the device.
pub fn quantizes(m: &TypedModule) -> bool {
    m.blocks().iter().any(|b| {
        b.insts.iter().any(|n| {
            matches!(
                n.inst,
                Inst::Unary {
                    op: UnOp::QuantizeToF16,
                    ..
                }
            )
        })
    })
}

/// The `f32` bits `b` (form `u`) rounded to the nearest `bfloat` (ties to even), as `f32` bits
/// with the low half clear. A NaN stays a NaN of its sign (quieted); a finite value that rounds
/// past the largest `bfloat` becomes the infinity of its sign, as the carry into the exponent
/// gives.
fn bf16_round_bits(fb: &mut FunctionBuilder<'_>, u: Form, b: Value) -> BResult<Value> {
    let k16 = splat(fb, u, 16)?;
    let one = splat(fb, u, 1)?;
    let hi_mask = splat(fb, u, 0xFFFF_0000)?;
    let lsb = bin(fb, BinOp::ShiftRightLogical, b, k16)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
    let half_ulp = splat(fb, u, 0x7FFF)?;
    let bias = bin(fb, BinOp::IAdd, half_ulp, lsb)?;
    let r = bin(fb, BinOp::IAdd, b, bias)?;
    let r = bin(fb, BinOp::BitwiseAnd, r, hi_mask)?;
    // NaN: the magnitude bits above the infinity's.
    let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let mag = bin(fb, BinOp::BitwiseAnd, b, abs_mask)?;
    let nan = bin(fb, BinOp::UGreaterThan, mag, inf)?;
    let quiet = splat(fb, u, 0x0040_0000)?;
    let q = bin(fb, BinOp::BitwiseOr, b, quiet)?;
    let q = bin(fb, BinOp::BitwiseAnd, q, hi_mask)?;
    fb.select(nan, q, r)
}

/// An upper estimate of the SPIR-V words of one [`to_bits`] or [`from_bits`] on one lane of
/// form `f`: for half, the device's conversion or pack and their 16-bit types and zero, a few
/// instructions of at most nine words; a shift for bfloat.
pub fn bits_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => 48,
        Some(FloatFmt::BFloat) => 24,
        _ => 0,
    }
}

/// The format bits (zero-extended in `u32` lanes) of narrow value `x` of form `f`. A `bfloat`'s
/// are exact for every value of the format. A `half`'s are the device's narrowing of the value
/// ([`encode`]): exact for every `half` value but a NaN's bits and, on a device that may flush
/// 16-bit denormals, a denormal, each recorded ([`encodes_bits`]).
///
/// Bits already known are reused: see `known_bits`, so a `half` loaded and stored unchanged keeps
/// its bits, a NaN's payload included; and a lane extract's bits are taken from its vector's when
/// those are known. Each is the value's bits, so the result does not depend on which is taken but
/// for a NaN's, which the device's narrowing may encode otherwise.
pub fn to_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let Some(fmt) = narrow_fmt(f) else {
        return encode(fb, f, x);
    };
    if let Some(b) = known_bits(fb, fmt, f, x)? {
        return Ok(b);
    }
    let mut lane = None;
    if f.lanes == 1 {
        if let Some((v, i)) = fb.lane_of(x) {
            if let Some(n) = vector_lanes(fb, v)? {
                let vf = Form { lanes: n, ..f };
                if let Some(vb) = known_bits(fb, fmt, vf, v)? {
                    let b = fb.composite_extract(vb, &[i])?;
                    fb.note_bits(fmt, x, b, 0, false);
                    return Ok(b);
                }
                // The lane of one encode of the vector, shared by its lanes' stores in this
                // block, is offered for the lane's own encode.
                let (vb, group) = match fb.encode_of(fmt, v) {
                    Some(e) => e,
                    None => {
                        let vb = encode(fb, vf, v)?;
                        let group = fb.offer_group();
                        fb.note_encode(fmt, v, vb, group);
                        (vb, group)
                    }
                };
                lane = Some((fb.composite_extract(vb, &[i])?, group));
            }
        }
    }
    let b = encode(fb, f, x)?;
    fb.note_bits(fmt, x, b, 0, false);
    if let Some((c, group)) = lane {
        fb.offer(group, b, c);
    }
    Ok(b)
}

/// The format bits of `x` (form `f`, format `fmt`) when they are known without encoding the
/// value ([`encode`]), cleared above each lane's 16 bits: the bits `x` was decoded from
/// ([`from_bits`] notes them) or computed with earlier in the open block, a constant's bits
/// computed at translation time ([`constant_bits`]), or, for a vector built by lane inserts
/// ([`note_insert`]), the inserts repeated on the lanes' known bits. `None` otherwise.
fn known_bits(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
) -> BResult<Option<Value>> {
    if let Some((bits, dirty)) = fb.bits_of(fmt, x) {
        return clean(fb, fmt, f, x, bits, dirty).map(Some);
    }
    if let Some(k) = constant_bits(fb, f, x)? {
        return Ok(Some(k));
    }
    if f.lanes > 1 {
        return inserted_bits(fb, fmt, f, x);
    }
    Ok(None)
}

/// `bits` (the noted bits of `x`, dirty in the lanes `dirty` has a bit for) cleared above each
/// lane's 16 bits, noted for `x` as clean in the open block.
fn clean(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
    bits: Value,
    dirty: u32,
) -> BResult<Value> {
    if dirty == 0 {
        return Ok(bits);
    }
    let m = splat(fb, u32_form(f), 0xFFFF)?;
    let b = bin(fb, BinOp::BitwiseAnd, bits, m)?;
    fb.note_bits(fmt, x, b, 0, false);
    Ok(b)
}

/// The most lane inserts [`inserted_bits`] repeats on bits: a bound on its work (a longer chain
/// is encoded, [`encode`]).
const MAX_INSERTS: usize = 16;

/// Where a lane insert's bits come from.
enum LaneBits {
    /// Noted bits, dirty or not.
    Noted(Value, bool),
    /// A constant's bits.
    Const(u64),
    /// Lane `.2` of a vector's noted bits (dirty in the lanes `.1` has a bit for).
    Of(Value, u32, u32),
}

/// Where the first vector of a chain of lane inserts gets its bits from.
enum BaseBits {
    Noted(Value, u32),
    Undef,
    Const(Vec<u64>),
}

/// The bits of vector `x` built by a chain of lane inserts ([`note_insert`]) from a vector
/// whose bits are known (noted, a constant's, or an `undef`'s, which may be any bits), each
/// inserted lane's bits known (noted, a constant's, or a lane of a vector's noted bits): the
/// same inserts on the bits. Each lane then holds the bits of the lane `x` holds, so they are
/// `x`'s bits. Checked before anything is emitted; `None` (nothing emitted) when a link is
/// unknown.
fn inserted_bits(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
) -> BResult<Option<Value>> {
    let mut steps: Vec<(LaneBits, u32)> = Vec::new();
    let mut at = x;
    let base = loop {
        if !steps.is_empty() {
            if let Some((b, d)) = fb.bits_of(fmt, at) {
                break BaseBits::Noted(b, d);
            }
            if fb.is_undef(at) {
                break BaseBits::Undef;
            }
            if let Some(k) = constant_lanes(fb, f, at)? {
                break BaseBits::Const(k);
            }
        }
        if steps.len() == MAX_INSERTS {
            return Ok(None);
        }
        let Some((v, e, i)) = fb.insert_of(at) else {
            return Ok(None);
        };
        let lane = Form { lanes: 1, ..f };
        let e_bits = if let Some((b, d)) = fb.bits_of(fmt, e) {
            LaneBits::Noted(b, d != 0)
        } else if let Some(k) = constant_lanes(fb, lane, e)? {
            LaneBits::Const(k.first().copied().unwrap_or(0))
        } else if let Some((w, j)) = fb.lane_of(e) {
            match fb.bits_of(fmt, w) {
                Some((wb, wd)) => LaneBits::Of(wb, wd, j),
                None => return Ok(None),
            }
        } else {
            return Ok(None);
        };
        steps.push((e_bits, i));
        at = v;
    };
    let all = u32::MAX >> (32 - u32::from(f.lanes).clamp(1, 32));
    let (mut rb, mut dirty) = match base {
        BaseBits::Noted(b, d) => (b, d),
        BaseBits::Undef => {
            let t = ty(fb, u32_form(f))?;
            (fb.undef(t)?, all)
        }
        BaseBits::Const(k) => (materialize(fb, f, &k)?, 0),
    };
    for (e_bits, i) in steps.into_iter().rev() {
        let (eb, d) = match e_bits {
            LaneBits::Noted(b, d) => (b, d),
            LaneBits::Const(k) => (splat(fb, u32_form(Form { lanes: 1, ..f }), k)?, false),
            LaneBits::Of(wb, wd, j) => (fb.composite_extract(wb, &[j])?, wd >> j & 1 != 0),
        };
        rb = fb.composite_insert(eb, rb, &[i])?;
        let bit = 1u32.checked_shl(i).unwrap_or(0);
        dirty = if d { dirty | bit } else { dirty & !bit };
    }
    fb.note_bits(fmt, x, rb, dirty, false);
    clean(fb, fmt, f, x, rb, dirty).map(Some)
}

/// Note that narrow vector `y` is `v` with lane `index` replaced by `e` (an `f32` scalar), for
/// `inserted_bits`; nothing for any other lane type.
pub fn note_insert(
    fb: &mut FunctionBuilder<'_>,
    y: Value,
    v: Value,
    e: Value,
    index: u32,
) -> BResult<()> {
    if may_be_narrow(fb, e)? {
        fb.note_insert(y, v, e, index);
    }
    Ok(())
}

/// `insertelement` of lane `e` into vector `v` at lane `index`: the `f32` insert, noted for
/// `inserted_bits`. When `e` is a narrow decode (its bits noted by [`from_bits`]) and `v` one too
/// or an `undef`, the same insert on the bits is decoded as a vector first and offered for the
/// `f32` insert: it holds the same lanes (each lane the decode of the bits it was decoded from; an
/// `undef` lane any value), and the insert's bits are noted, so a chain of such inserts decodes
/// once if the module is smaller for it.
pub fn insert(fb: &mut FunctionBuilder<'_>, v: Value, e: Value, index: u32) -> BResult<Value> {
    let mut offered = None;
    if may_be_narrow(fb, e)? {
        if let Some(n) = vector_lanes(fb, v)? {
            offered = decoded_insert(fb, v, e, index, n)?;
        }
    }
    let y = fb.composite_insert(e, v, &[index])?;
    note_insert(fb, y, v, e, index)?;
    if let Some((fmt, rb, dirty, yd)) = offered {
        fb.note_bits(fmt, y, rb, dirty, true);
        let group = fb.offer_group();
        fb.offer(group, y, yd);
    }
    Ok(y)
}

/// The bits insert and its vector decode for [`insert`] when both operands are decodes of known
/// bits (or the vector an `undef`): the format, the inserted bits and their dirty lanes, and the
/// decode. A declined offer leaves nothing behind: its instructions, the constants and vector
/// types only they read leave, and the integer scalars it reads are the lane decode's own.
fn decoded_insert(
    fb: &mut FunctionBuilder<'_>,
    v: Value,
    e: Value,
    index: u32,
    n: u8,
) -> BResult<Option<(NarrowFmt, Value, u32, Value)>> {
    for (fmt, held) in [
        (NarrowFmt::Half, FloatFmt::Half),
        (NarrowFmt::BFloat, FloatFmt::BFloat),
    ] {
        let Some((eb, ed)) = fb.decoded_from(fmt, e) else {
            continue;
        };
        let f = Form {
            held: Held::Float(held),
            lanes: n,
        };
        let all = u32::MAX >> (32 - u32::from(n).clamp(1, 32));
        let (vb, vd) = if let Some(noted) = fb.decoded_from(fmt, v) {
            noted
        } else if fb.is_undef(v) {
            let t = ty(fb, u32_form(f))?;
            (fb.undef(t)?, all)
        } else {
            return Ok(None);
        };
        let rb = fb.composite_insert(eb, vb, &[index])?;
        let bit = 1u32.checked_shl(index).unwrap_or(0);
        let dirty = if ed & 1 != 0 { vd | bit } else { vd & !bit };
        let yd = from_bits(fb, f, rb)?;
        return Ok(Some((fmt, rb, dirty, yd)));
    }
    Ok(None)
}

/// The builder's name of `f`'s narrow format, if it is one.
fn narrow_fmt(f: Form) -> Option<NarrowFmt> {
    match f.held {
        Held::Float(FloatFmt::Half) => Some(NarrowFmt::Half),
        Held::Float(FloatFmt::BFloat) => Some(NarrowFmt::BFloat),
        _ => None,
    }
}

/// Whether `y` may hold a narrow float: an `f32` scalar.
pub fn may_be_narrow(fb: &mut FunctionBuilder<'_>, y: Value) -> BResult<bool> {
    let t = fb.value_ty(y)?;
    Ok(matches!(
        fb.module().types().get(t)?,
        Type::Float { width: 32 }
    ))
}

/// The lane count of `v` when it is a vector.
fn vector_lanes(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Option<u8>> {
    let t = fb.value_ty(v)?;
    Ok(match fb.module().types().get(t)? {
        Type::Vector { len, .. } => Some(*len),
        _ => None,
    })
}

/// The format bits of narrow value `x` of form `f`, where no known bits stand: a `bfloat`'s its
/// `f32` bits moved down, exact; a `half`'s the device's narrowing of the value, its conversion to
/// a 16-bit float where the module can hold one (every [`QuantizeForm`] but
/// [`QuantizeForm::Device`], [`converted_bits`]), else `GLSL.std.450 PackHalf2x16`
/// ([`packed_bits`]).
fn encode(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match (f.held, fb.module().quantize_form()) {
        (Held::Float(FloatFmt::BFloat), _) => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let k16 = splat(fb, u, 16)?;
            bin(fb, BinOp::ShiftRightLogical, b, k16)
        }
        (_, QuantizeForm::Device) => packed_bits(fb, f, x),
        (_, QuantizeForm::Converted | QuantizeForm::Controlled) => converted_bits(fb, f, x),
    }
}

/// The `half` bits of `x` (`f32` lanes of form `f`), zero-extended in `u32` lanes, by the device's
/// conversion to 16 bits: the lanes converted at once, one `OpFConvert` (decorated `NoContraction`
/// under the float controls, as [`controlled`]'s), and each beside a 16-bit zero read as a word
/// (`OpBitcast` maps the lower-numbered component to a word's lower-ordered bits), two lanes at a
/// time as four. Three instructions for one or two lanes, six for three or four.
///
/// Every `half` value converts to its own bits, by the Vulkan spec ("Precision and Operation of
/// SPIR-V Instructions": a conversion is "Correctly rounded"), but a NaN, whose bits are the
/// device's (the SPIR-V spec's `SignedZeroInfNanPreserve`: "Bit patterns for NaNs might not be
/// preserved"), where Metal moves a `half` NaN with its bits, and a denormal, which the device may
/// flush to the zero of its sign without `DenormPreserve` for 16 bits: kept under the float
/// controls ([`QuantizeForm::Controlled`], [`controlled_modes`]), measured kept on MoltenVK
/// (`DriverDeviation::MetalHalfArithmetic`, whose measurement narrows every `half`), and elsewhere
/// within `PrecisionNote::NativeHalfConversion`'s bound; the module records
/// `PrecisionNote::NativeNanEncoding` ([`encodes_bits`]). A value that is no `half` value (under
/// native half rounding the device's compiler may fold [`converted`]'s pair and hand `x` on
/// unrounded, as NVIDIA's and lavapipe's do) takes the device's rounding, within
/// `PrecisionNote::NativeHalfConversion`'s bound, which that module records.
fn converted_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let halves =
        |fb: &mut FunctionBuilder<'_>, len: u8| fb.module().ty(Type::Vector { elem: e, len });
    let words = |fb: &mut FunctionBuilder<'_>, lanes: u8| ty(fb, u32_form(Form { lanes, ..f }));
    let narrow = |fb: &mut FunctionBuilder<'_>, to| match fb.module().quantize_form() {
        QuantizeForm::Controlled => fb.convert_no_contraction(x, to),
        _ => fb.convert(ConvOp::FConvert, x, to),
    };
    if f.lanes <= 1 {
        let h = narrow(fb, e)?;
        let k = fb.module().constant(Constant::Null { ty: e })?;
        let z = fb.module().const_value(k)?;
        let pair = halves(fb, 2)?;
        let p = fb.composite_construct(pair, &[h, z])?;
        let w = words(fb, 1)?;
        return fb.convert(ConvOp::Bitcast, p, w);
    }
    let n = u32::from(f.lanes);
    let hv = halves(fb, f.lanes)?;
    let h = narrow(fb, hv)?;
    let pair = halves(fb, 2)?;
    let k = fb.module().constant(Constant::Null { ty: pair })?;
    let zeros = fb.module().const_value(k)?;
    // At most two parts: a vector holds at most four lanes. Index `n` is a zero.
    let mut parts = Vec::with_capacity(2);
    let mut at = 0u32;
    while at < n {
        let part = if n - at >= 2 {
            let q = fb.vector_shuffle(h, zeros, &[at, n, at + 1, n])?;
            let w = words(fb, 2)?;
            at += 2;
            fb.convert(ConvOp::Bitcast, q, w)?
        } else {
            let p = fb.vector_shuffle(h, zeros, &[at, n])?;
            let w = words(fb, 1)?;
            at += 1;
            fb.convert(ConvOp::Bitcast, p, w)?
        };
        parts.push(part);
    }
    match parts.as_slice() {
        [one] => Ok(*one),
        _ => {
            let ut = ty(fb, u32_form(f))?;
            fb.composite_construct(ut, &parts)
        }
    }
}

/// The `half` bits of `x` (`f32` lanes of form `f`), zero-extended in `u32` lanes, by
/// `GLSL.std.450 PackHalf2x16` ([`QuantizeForm::Device`], a profile without `shaderFloat16`): each
/// lane packed beside a zero, so a word's low 16 bits are the lane's `half` and its high 16 bits
/// the zero's. Two instructions a lane, and a vector's construct.
///
/// It needs no 16-bit type: the instruction is core to every Vulkan device. Vulkan gives it no
/// precision of its own ("implementation-defined"); every `half` value is an `f32`, and the
/// device's narrowing is its answer: a NaN packs to a NaN whose bits are the device's, where Metal
/// moves a `half` NaN with its bits, and a denormal may be flushed to the zero of its sign without
/// `DenormPreserve` for 16 bits, which such a module does not declare. The module records
/// `PrecisionNote::NativeNanEncoding` and `PrecisionNote::NativeHalfConversion`
/// ([`encodes_bits`]).
fn packed_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let wide = |lanes: u8| Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    };
    if f.lanes <= 1 {
        let z = fsplat(fb, wide(1), 0.0)?;
        let pair = ty(fb, wide(2))?;
        let p = fb.composite_construct(pair, &[x, z])?;
        return fb.glsl(GlslOp::PackHalf2x16, &[p]);
    }
    let n = u32::from(f.lanes);
    // Index `n` is a zero.
    let zeros = fsplat(fb, wide(2), 0.0)?;
    let mut words = Vec::with_capacity(usize::from(f.lanes));
    for i in 0..n {
        let p = fb.vector_shuffle(x, zeros, &[i, n])?;
        words.push(fb.glsl(GlslOp::PackHalf2x16, &[p])?);
    }
    let ut = ty(fb, u32_form(f))?;
    fb.composite_construct(ut, &words)
}

/// Whether module `m` narrows a `half` to its bits by the device ([`to_bits`]): a
/// `GLSL.std.450 PackHalf2x16` ([`packed_bits`], in [`QuantizeForm::Device`]), or an `OpBitcast`
/// to integers of an `OpFConvert` from 32 to 16 bits, or of a construct or shuffle of one
/// ([`converted_bits`]). It moves a `half` between its value and its bits as [`widens_bits`]'s
/// widening does the other way, with the same notes: the module records
/// `PrecisionNote::NativeNanEncoding` (a NaN's bits the device's); under the float controls
/// ([`QuantizeForm::Controlled`]) its entry declares [`controlled_modes`], and elsewhere its
/// denormals are the measured fact's (`DriverDeviation::MetalHalfArithmetic`) or the device's
/// (`PrecisionNote::NativeHalfConversion`). At most three passes over the instructions, with two
/// bits per value of the module charged to `meter` while they run.
pub fn encodes_bits(m: &TypedModule, meter: &mut Meter) -> Result<bool, Error> {
    let packs = m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
        matches!(
            n.inst,
            Inst::Glsl {
                op: GlslOp::PackHalf2x16,
                ..
            }
        )
    });
    if packs {
        return Ok(true);
    }
    // The width of `v`'s float lanes, if it is a float or a vector of floats; integers are `0`.
    let width = |v: Value| -> Option<u8> {
        let ty = m.value(v).ok()?.ty;
        let ty = match m.types().get(ty).ok()? {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        match m.types().get(ty).ok()? {
            Type::Float { width } => Some(*width),
            Type::Int { .. } => Some(0),
            _ => None,
        }
    };
    let insts = || m.blocks().iter().flat_map(|b| &b.insts);
    let words = m.values().len().div_ceil(64);
    meter.charge(16 * words)?;
    let mut narrowed = vec![0u64; words];
    let mut gathered = vec![0u64; words];
    let set = |bits: &mut [u64], v: Value| {
        if let Some(w) = bits.get_mut(v.index() / 64) {
            *w |= 1 << (v.index() % 64);
        }
    };
    let has = |bits: &[u64], v: Value| {
        bits.get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    };
    for n in insts() {
        if let (
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            },
            Some(r),
        ) = (&n.inst, n.result)
        {
            if width(*a) == Some(32) && width(r) == Some(16) {
                set(&mut narrowed, r);
            }
        }
    }
    for n in insts() {
        let taken = match &n.inst {
            Inst::CompositeConstruct { parts } => parts.iter().any(|p| has(&narrowed, *p)),
            Inst::VectorShuffle { a, b, .. } => has(&narrowed, *a) || has(&narrowed, *b),
            _ => false,
        };
        if let (true, Some(r)) = (taken, n.result) {
            set(&mut gathered, r);
        }
    }
    let found = insts().any(|n| match (&n.inst, n.result) {
        (
            Inst::Convert {
                op: ConvOp::Bitcast,
                a,
                ..
            },
            Some(r),
        ) => (has(&narrowed, *a) || has(&gathered, *a)) && width(r) == Some(0),
        _ => false,
    });
    drop((narrowed, gathered));
    meter.release(16 * words);
    Ok(found)
}

/// The format bits of `x` (form `f`) when `x` is a plain constant every lane of which is a
/// value of the format, computed at translation time by [`half_bits_of_f32`] and checked by
/// [`f32_of_half_bits`]: `None` for anything else (the device's narrowing then runs, [`encode`]).
/// A `null` is zero bits.
pub fn constant_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Value>> {
    match constant_lanes(fb, f, x)? {
        Some(k) => materialize(fb, f, &k).map(Some),
        None => Ok(None),
    }
}

/// The lanes' bits [`constant_bits`] folds `x` to, without emitting anything.
fn constant_lanes(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Vec<u64>>> {
    let ValueOrigin::Const(c) = fb.origin(x)? else {
        return Ok(None);
    };
    let lane = |bits: u64| -> Option<u64> {
        let b = u32::try_from(bits).ok()?;
        match f.held {
            Held::Float(FloatFmt::Half) => {
                let h = half_bits_of_f32(b);
                (f32_of_half_bits(h) == b).then_some(u64::from(h))
            }
            Held::Float(FloatFmt::BFloat) => (b & 0xFFFF == 0).then_some(u64::from(b >> 16)),
            _ => None,
        }
    };
    let consts = fb.module().consts();
    Ok(match &consts.get(c)?.kind {
        ConstKind::Plain(Constant::Scalar { bits, .. }) if f.lanes == 1 => {
            lane(bits.0).map(|b| vec![b])
        }
        ConstKind::Plain(Constant::Null { .. }) => Some(vec![0; usize::from(f.lanes)]),
        ConstKind::Plain(Constant::Composite { parts, .. })
            if parts.len() == usize::from(f.lanes) =>
        {
            parts
                .iter()
                .map(|p| consts.scalar_bits(*p).and_then(|b| lane(b.0)))
                .collect()
        }
        _ => None,
    })
}

/// The `u32` constant of form `u32_form(f)` with lanes `lanes`.
fn materialize(fb: &mut FunctionBuilder<'_>, f: Form, lanes: &[u64]) -> BResult<Value> {
    let u = u32_form(f);
    if f.lanes == 1 {
        return splat(fb, u, lanes.first().copied().unwrap_or(0));
    }
    let e = ty(fb, Form { lanes: 1, ..u })?;
    let mut parts = Vec::with_capacity(lanes.len());
    for b in lanes {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(*b),
        })?);
    }
    let t = ty(fb, u)?;
    let k = fb.module().constant(Constant::Composite {
        ty: t,
        parts: parts.into(),
    })?;
    fb.module().const_value(k)
}

/// The half bits of the `f32` with bits `b`, on the host, exact when `b` is a half value (a NaN's
/// payload moved down under the all-ones exponent, quieted where its top bits are all zero): the
/// bits [`constant_bits`] folds a constant to.
pub fn half_bits_of_f32(b: u32) -> u32 {
    let a = b & 0x7FFF_FFFF;
    let sign = (b >> 16) & 0x8000;
    let t = a >> 13;
    let bias = if a >= 0x7F80_0000 {
        224 << 10
    } else {
        112 << 10
    };
    let normal = if a.wrapping_sub(0x7F80_0001) < 0x1FFF {
        0x7E00
    } else {
        t.wrapping_sub(bias)
    };
    let r = if a < 0x3880_0000 {
        // `|x| · 2^24`, an integer below 2^10 for a half denormal or zero.
        (f32::from_bits(a) * 16_777_216.0) as u32
    } else {
        normal
    };
    sign | r
}

/// The `f32` bits of the exact value of the half with bits `h` (its low 16 bits), on the host: a
/// half's exponent and fraction moved up and rebiased, a denormal's integer times `2^-24`, an
/// infinity's or NaN's payload moved up under the all-ones exponent.
pub fn f32_of_half_bits(h: u32) -> u32 {
    let em = h & 0x7FFF;
    let bias = if em >= 0x7C00 { 224 << 23 } else { 112 << 23 };
    let r = if em < 0x0400 {
        (em as f32 * (1.0 / 16_777_216.0)).to_bits()
    } else {
        (em << 13).wrapping_add(bias)
    };
    ((h & 0x8000) << 16) | r
}

/// The narrow value (an `f32` of form `f`) of format bits `h`, the low 16 bits of each `u32`
/// lane; the bits above them are ignored, so a load need not clear them.
///
/// A `half` is the device's widening of its bits on every profile: wherever the module can hold a
/// 16-bit float (every [`QuantizeForm`] but [`QuantizeForm::Device`]) the bits read as one and
/// converted ([`device_from_bits`]), and in [`QuantizeForm::Device`], which only a profile without
/// `shaderFloat16` takes, `GLSL.std.450 UnpackHalf2x16` of each lane's word
/// ([`unpacked_from_bits`]), which needs no 16-bit type. A `bfloat` is its bits moved up, exact
/// for every bit pattern.
///
/// The result's bits are noted: a later [`to_bits`] of the value, wherever the value is, returns
/// `h` (masked to its low 16 bits) without converting back, so a narrow value stored unchanged
/// keeps its bits whatever the widening made of them.
pub fn from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    let x = match (f.held, fb.module().quantize_form()) {
        (Held::Float(FloatFmt::Half), QuantizeForm::Device) => unpacked_from_bits(fb, f, h)?,
        (Held::Float(FloatFmt::Half), QuantizeForm::Converted | QuantizeForm::Controlled) => {
            device_from_bits(fb, f, h)?
        }
        _ => {
            let u = u32_form(f);
            let k16 = splat(fb, u, 16)?;
            let b = bin(fb, BinOp::ShiftLeftLogical, h, k16)?;
            let ft = ty(fb, f)?;
            fb.convert(ConvOp::Bitcast, b, ft)?
        }
    };
    if let Some(fmt) = narrow_fmt(f) {
        let all = u32::MAX >> (32 - u32::from(f.lanes).clamp(1, 32));
        fb.note_bits(fmt, x, h, all, true);
    }
    Ok(x)
}

/// [`from_bits`] of the `half` lanes of form `f` that `words` hold as memory holds them, from half
/// `first` (0 or 1) of the first word on: lane `k` is the low 16 bits of word `(first + k) / 2`
/// where `first + k` is even, else its high 16 bits; `h` is the same lanes' bits as [`from_bits`]
/// takes them. The words are widened by the device at once, one or two of them, and the lanes
/// taken from them: where the module can hold a 16-bit float, read as 16-bit floats (one
/// `OpBitcast` of the words) and converted, one `OpFConvert`, three instructions for a `half4` from
/// its two words, and a lone `half` whose word is a 16-bit unit's zero extension
/// (`FunctionBuilder::widened16_of`) is that unit read as a 16-bit float and converted, two; in
/// [`QuantizeForm::Device`], each word's `UnpackHalf2x16`, which reads both its halves, so a
/// `half2` of one word is one instruction, a lone `half` two and a `half4` three. Widening `h`
/// instead gathers each lane's word and shifts it first. Lanes past two words take [`from_bits`]
/// of `h`. Either way `h` is noted as [`from_bits`] notes it.
pub fn from_word_halves(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    words: &[Value],
    first: u32,
    h: Value,
) -> BResult<Value> {
    let lanes = u32::from(f.lanes.max(1));
    let n = (first + lanes).div_ceil(2) as usize;
    let fits = matches!(f.held, Held::Float(FloatFmt::Half)) && first < 2 && n <= 2;
    let Some(words) = words.get(..n).filter(|_| fits) else {
        return from_bits(fb, f, h);
    };
    let x = match fb.module().quantize_form() {
        QuantizeForm::Device => unpacked_words(fb, f, words, first)?,
        QuantizeForm::Converted | QuantizeForm::Controlled => converted_words(fb, f, words, first)?,
    };
    let all = u32::MAX >> (32 - lanes.clamp(1, 32));
    fb.note_bits(NarrowFmt::Half, x, h, all, true);
    Ok(x)
}

/// [`from_word_halves`]' widening where the module can hold a 16-bit float: `words` (one or two)
/// read as 16-bit floats, lanes `first..first + f.lanes` taken from them and converted to 32 bits;
/// a lone `half` of a 16-bit unit's zero extension, that unit read as a 16-bit float.
fn converted_words(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    words: &[Value],
    first: u32,
) -> BResult<Value> {
    let lanes = u32::from(f.lanes.max(1));
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let wide = ty(fb, f)?;
    // A lone `half` loaded as a 16-bit unit: the unit itself read as a 16-bit float.
    if let ([w], 0, 1) = (words, first, lanes) {
        if let Some(unit) = fb.widened16_of(*w) {
            let x = fb.convert(ConvOp::Bitcast, unit, e)?;
            return fb.convert(ConvOp::FConvert, x, wide);
        }
    }
    let packed = match words {
        [w] => *w,
        _ => {
            let uv = ty(fb, u32_form(Form { lanes: 2, ..f }))?;
            fb.composite_construct(uv, words)?
        }
    };
    let comps = 2 * words.len() as u32;
    let halves = fb.module().ty(Type::Vector {
        elem: e,
        len: comps as u8,
    })?;
    let p = fb.convert(ConvOp::Bitcast, packed, halves)?;
    let taken = match (first, lanes) {
        (0, l) if l == comps => p,
        (_, 1) => fb.composite_extract(p, &[first])?,
        _ => {
            let picked: Vec<u32> = (first..first + lanes).collect();
            fb.vector_shuffle(p, p, &picked)?
        }
    };
    fb.convert(ConvOp::FConvert, taken, wide)
}

/// [`from_word_halves`]' widening in [`QuantizeForm::Device`]: each of `words` (one or two)
/// unpacked by `UnpackHalf2x16` into the pair of `f32`s its halves widen to (its low 16 bits the
/// first), and lanes `first..first + f.lanes` taken from the pairs.
fn unpacked_words(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    words: &[Value],
    first: u32,
) -> BResult<Value> {
    let lanes = u32::from(f.lanes.max(1));
    let mut pairs = Vec::with_capacity(2);
    for w in words {
        pairs.push(fb.glsl(GlslOp::UnpackHalf2x16, &[*w])?);
    }
    match (pairs.as_slice(), first, lanes) {
        ([p], 0, 2) => Ok(*p),
        ([p, ..], _, 1) => fb.composite_extract(*p, &[first]),
        ([a, b], _, _) => {
            let picked: Vec<u32> = (first..first + lanes).collect();
            fb.vector_shuffle(*a, *b, &picked)
        }
        _ => Err(BuildError::Type(TypeRule::VectorShape)),
    }
}

/// The `half` values of bits `h` (the low 16 bits of each `u32` lane of form [`u32_form`]`(f)`)
/// widened by `GLSL.std.450 UnpackHalf2x16` ([`QuantizeForm::Device`], a profile without
/// `shaderFloat16`): each lane's word unpacked and its first component taken, whose value the
/// unpack reads from the word's low 16 bits, the bits above them ignored. Two instructions for a
/// lane, and a lane extract and the vector's construct for a vector.
///
/// It needs no 16-bit type: the instruction is core to every Vulkan device. Vulkan gives it no
/// precision of its own ("implementation-defined"); every `half` is an `f32`, and the device's
/// widening is its answer. A NaN widens to a NaN whose bits are the device's, where Metal widens
/// every `half` NaN to `0x7FC00000` and moves one unwidened with its bits; the module records
/// `PrecisionNote::NativeNanEncoding` ([`widens_bits`]). A denormal "obtained via unpacking an
/// integer into a vector of values with smaller bit width" may be flushed to the zero of its sign
/// without `DenormPreserve` for 16 bits, which such a module does not declare, within
/// `PrecisionNote::NativeHalfConversion`'s bound, which the module records.
fn unpacked_from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    let lane = |fb: &mut FunctionBuilder<'_>, word: Value| -> BResult<Value> {
        let pair = fb.glsl(GlslOp::UnpackHalf2x16, &[word])?;
        fb.composite_extract(pair, &[0])
    };
    if f.lanes <= 1 {
        return lane(fb, h);
    }
    let mut lanes = Vec::with_capacity(usize::from(f.lanes));
    for i in 0..u32::from(f.lanes) {
        let word = fb.composite_extract(h, &[i])?;
        lanes.push(lane(fb, word)?);
    }
    let wide = ty(fb, f)?;
    fb.composite_construct(wide, &lanes)
}

/// The `half` values of bits `h` (the low 16 bits of each `u32` lane of form [`u32_form`]`(f)`)
/// widened by the device: the lanes' words reinterpreted as 16-bit floats, two words at a time as
/// four (`OpBitcast` maps a word's lower-ordered bits to the lower-numbered component, so each
/// lane's half is an even component), those components gathered, and the gathered 16-bit floats
/// converted to 32 bits, one `OpFConvert`. Three instructions for one or two lanes, six for three
/// or four.
///
/// Every `half` but a NaN and a denormal widens to its exact value, by the Vulkan spec
/// ("Precision and Operation of SPIR-V Instructions"): a conversion is "Correctly rounded", and
/// every `half` is an `f32`. A NaN widens to a NaN whose bits are the device's (the SPIR-V spec's
/// `SignedZeroInfNanPreserve`: "Bit patterns for NaNs might not be preserved"), where Metal widens
/// every `half` NaN to `0x7FC00000` and moves one unwidened with its bits; the module records
/// `PrecisionNote::NativeNanEncoding` ([`widens_bits`]). A denormal "obtained via unpacking an
/// integer into a vector of values with smaller bit width" is kept under `DenormPreserve` for 16
/// bits, which the float controls declare ([`QuantizeForm::Controlled`], [`controlled_modes`]); on
/// MoltenVK the measured `DriverDeviation::MetalHalfArithmetic` reads every `half` through this
/// widening as Metal widens it, and the module requires that fact; elsewhere the device may flush
/// it to the zero of its sign, within `PrecisionNote::NativeHalfConversion`'s bound, which the
/// module records.
fn device_from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let halves =
        |fb: &mut FunctionBuilder<'_>, len: u8| fb.module().ty(Type::Vector { elem: e, len });
    let wide = ty(fb, f)?;
    if f.lanes <= 1 {
        let pair = halves(fb, 2)?;
        let p = fb.convert(ConvOp::Bitcast, h, pair)?;
        let x = fb.composite_extract(p, &[0])?;
        return fb.convert(ConvOp::FConvert, x, wide);
    }
    // At most two parts: a vector holds at most four lanes.
    let mut parts = Vec::with_capacity(2);
    let mut at = 0u8;
    while at < f.lanes {
        let part = if f.lanes - at >= 2 {
            let words = match f.lanes {
                2 => h,
                _ => fb.vector_shuffle(h, h, &[u32::from(at), u32::from(at) + 1])?,
            };
            let quad = halves(fb, 4)?;
            at += 2;
            fb.convert(ConvOp::Bitcast, words, quad)?
        } else {
            let word = fb.composite_extract(h, &[u32::from(at)])?;
            let pair = halves(fb, 2)?;
            at += 1;
            fb.convert(ConvOp::Bitcast, word, pair)?
        };
        parts.push(part);
    }
    let (a, b) = match parts.as_slice() {
        [a] => (*a, *a),
        [a, b] => (*a, *b),
        _ => return Err(BuildError::Type(TypeRule::VectorShape)),
    };
    // Lane `k`'s half is component `2k` of the parts' concatenation: the first part holds four.
    let even: Vec<u32> = (0..u32::from(f.lanes)).map(|k| 2 * k).collect();
    let gathered = fb.vector_shuffle(a, b, &even)?;
    fb.convert(ConvOp::FConvert, gathered, wide)
}

/// Whether module `m` widens a `half` from its bits by the device ([`from_bits`],
/// [`from_word_halves`]): a `GLSL.std.450 UnpackHalf2x16` ([`unpacked_from_bits`], in
/// [`QuantizeForm::Device`]), or an `OpFConvert` to 32 bits of an `OpBitcast` to a 16-bit float or
/// a vector of them, or of 16-bit floats taken from one by a component extract or a shuffle
/// ([`device_from_bits`]). The module then records `PrecisionNote::NativeNanEncoding` (a NaN's
/// bits the device's); under the float controls ([`QuantizeForm::Controlled`]) its entry declares
/// [`controlled_modes`], and elsewhere its denormals are the measured fact's
/// (`DriverDeviation::MetalHalfArithmetic`, which measures the conversion, not the unpack) or the
/// device's (`PrecisionNote::NativeHalfConversion`). The device's 16-bit arithmetic and
/// conversions widen what they compute, which no bitcast gives. At most four passes over the
/// instructions, with two bits per value of the module charged to `meter` while they run.
pub fn widens_bits(m: &TypedModule, meter: &mut Meter) -> Result<bool, Error> {
    let unpacks = m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
        matches!(
            n.inst,
            Inst::Glsl {
                op: GlslOp::UnpackHalf2x16,
                ..
            }
        )
    });
    if unpacks {
        return Ok(true);
    }
    // Whether `ty` is a 16-bit float or a vector of them (`halves`), or a 32-bit one.
    let is = |ty, halves: bool| -> bool {
        let lane = match m.types().get(ty) {
            Ok(Type::Vector { elem, .. }) => *elem,
            _ => ty,
        };
        let width = if halves { 16 } else { 32 };
        matches!(m.types().get(lane), Ok(Type::Float { width: w }) if *w == width)
    };
    let insts = || m.blocks().iter().flat_map(|b| &b.insts);
    let words = m.values().len().div_ceil(64);
    meter.charge(16 * words)?;
    let mut halves = vec![0u64; words];
    let mut lanes = vec![0u64; words];
    let set = |bits: &mut [u64], v: Value| {
        if let Some(w) = bits.get_mut(v.index() / 64) {
            *w |= 1 << (v.index() % 64);
        }
    };
    let has = |bits: &[u64], v: Value| {
        bits.get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    };
    for n in insts() {
        if let (
            Inst::Convert {
                op: ConvOp::Bitcast,
                ..
            },
            Some(r),
        ) = (&n.inst, n.result)
        {
            if m.value(r).is_ok_and(|v| is(v.ty, true)) {
                set(&mut halves, r);
            }
        }
    }
    for n in insts() {
        let taken = match &n.inst {
            Inst::CompositeExtract { composite, .. } => has(&halves, *composite),
            Inst::VectorShuffle { a, b, .. } => has(&halves, *a) || has(&halves, *b),
            _ => false,
        };
        if let (true, Some(r)) = (taken, n.result) {
            set(&mut lanes, r);
        }
    }
    let found = insts().any(|n| match (&n.inst, n.result) {
        (
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            },
            Some(r),
        ) => (has(&lanes, *a) || has(&halves, *a)) && m.value(r).is_ok_and(|v| is(v.ty, false)),
        _ => false,
    });
    drop((halves, lanes));
    meter.release(16 * words);
    Ok(found)
}

/// Whether module `m` computes in 16-bit floats: an `OpFAdd`, `OpFSub` or `OpFMul`, an `OpFmaKHR`
/// or a `GLSL.std.450 Fma` whose result is a 16-bit float or a vector of them, the device's own
/// `half` arithmetic (`super::ops::f16_binary`, `super::fma`). Under the 16-bit float controls
/// ([`QuantizeForm::Controlled`]) the entry then declares [`controlled_modes`], which make that
/// arithmetic Metal's. One pass over the instructions.
pub fn computes_in_16_bits(m: &TypedModule) -> bool {
    let half = |v: Value| -> bool {
        let Ok(ty) = m.value(v).map(|v| v.ty) else {
            return false;
        };
        let ty = match m.types().get(ty) {
            Ok(Type::Vector { elem, .. }) => *elem,
            _ => ty,
        };
        matches!(m.types().get(ty), Ok(Type::Float { width: 16 }))
    };
    m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
        let arithmetic = match &n.inst {
            Inst::Binary { op, .. } => matches!(op, BinOp::FAdd | BinOp::FSub | BinOp::FMul),
            Inst::Fma { .. } => true,
            Inst::Glsl { op, .. } => *op == GlslOp::Fma,
            _ => false,
        };
        arithmetic && n.result.is_some_and(half)
    })
}

/// `x` (`double` lanes of form `from`) converted to `f32` rounding to odd: the result is
/// truncated toward zero and its lowest bit set when inexact, so one later rounding to a format
/// of at most 22 significand bits is correctly rounded. Independent of the device's rounding
/// mode (the step back toward zero undoes a rounding away).
pub fn f64_to_f32_odd(fb: &mut FunctionBuilder<'_>, from: Form, x: Value) -> BResult<Value> {
    let f32f = from.with(Held::Float(FloatFmt::F32));
    let u = u32_form(from);
    let t32 = ty(fb, f32f)?;
    let ut = ty(fb, u)?;
    let t64 = ty(fb, from)?;
    let r = fb.convert(ConvOp::FConvert, x, t32)?;
    let back = fb.convert(ConvOp::FConvert, r, t64)?;
    let ab = fb.glsl(GlslOp::FAbs, &[back])?;
    let ax = fb.glsl(GlslOp::FAbs, &[x])?;
    let away = bin(fb, BinOp::FOrdGreaterThan, ab, ax)?;
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let one = splat(fb, u, 1)?;
    let toward = bin(fb, BinOp::ISub, rb, one)?;
    let rb = fb.select(away, toward, rb)?;
    let inexact = bin(fb, BinOp::FUnordNotEqual, back, x)?;
    let odd = bin(fb, BinOp::BitwiseOr, rb, one)?;
    let rb = fb.select(inexact, odd, rb)?;
    fb.convert(ConvOp::Bitcast, rb, t32)
}

/// Unsigned magnitude `m` (`u32` lanes of form `u`) converted to `f32` rounding to odd: the
/// bits below the 24 the significand keeps are folded into a sticky lowest bit before an exact
/// conversion, then scaled back exactly.
pub fn int_to_f32_odd(fb: &mut FunctionBuilder<'_>, u: Form, to: Form, m: Value) -> BResult<Value> {
    let msb = fb.glsl(GlslOp::FindUMsb, &[m])?;
    let k23 = splat(fb, u, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let z = splat(fb, u, 0)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s)?;
    let kept = bin(fb, BinOp::ShiftLeftLogical, t, s)?;
    let lost = bin(fb, BinOp::INotEqual, kept, m)?;
    let one = splat(fb, u, 1)?;
    let sticky = fb.select(lost, one, z)?;
    let t = bin(fb, BinOp::BitwiseOr, t, sticky)?;
    let f32f = to.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, f32f)?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// Unsigned 64-bit magnitude `m` (`u64` lanes of form `w`) converted to `f32` rounding to odd,
/// as [`int_to_f32_odd`] does for 32 bits: the most significant bit is found from the two
/// halves, the bits below the 24 kept are folded into a sticky bit, and the (at most 24-bit)
/// rest converts exactly and is scaled back exactly.
pub fn int64_to_f32_odd(
    fb: &mut FunctionBuilder<'_>,
    w: Form,
    to: Form,
    m: Value,
) -> BResult<Value> {
    let u = u32_form(w);
    let ut = ty(fb, u)?;
    let wt = ty(fb, w)?;
    let lo = fb.convert(ConvOp::UConvert, m, ut)?;
    let k32w = splat(fb, w, 32)?;
    let hi = bin(fb, BinOp::ShiftRightLogical, m, k32w)?;
    let hi = fb.convert(ConvOp::UConvert, hi, ut)?;
    let mlo = fb.glsl(GlslOp::FindUMsb, &[lo])?;
    let mhi = fb.glsl(GlslOp::FindUMsb, &[hi])?;
    let k32 = splat(fb, u, 32)?;
    let mhi = bin(fb, BinOp::IAdd, mhi, k32)?;
    let z = splat(fb, u, 0)?;
    let hz = bin(fb, BinOp::IEqual, hi, z)?;
    let msb = fb.select(hz, mlo, mhi)?;
    let k23 = splat(fb, u, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let s64 = fb.convert(ConvOp::UConvert, s, wt)?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s64)?;
    let kept = bin(fb, BinOp::ShiftLeftLogical, t, s64)?;
    let lost = bin(fb, BinOp::INotEqual, kept, m)?;
    let (one, z64) = (splat(fb, w, 1)?, splat(fb, w, 0)?);
    let sticky = fb.select(lost, one, z64)?;
    let t = bin(fb, BinOp::BitwiseOr, t, sticky)?;
    let f32f = to.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, f32f)?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// `fma(a, b, c)` of `bfloat` values held in `f32` (form `f`), as an `f32` whose one rounding
/// to `bfloat` ([`round`]) is the correctly rounded fused result.
///
/// `a * b` is exact in `f32` (two 8-bit significands), so `s = Fma(a, b, c)` is the exact sum
/// `a * b + c` rounded once to `f32` in the device's mode, fused or not. Rounding `s` to
/// `bfloat` is then correct unless `s` is exactly a `bfloat` tie (its low 16 bits `0x8000`)
/// while the exact sum is not: only then does the direction of the first rounding decide. That
/// direction is exact: with `big`/`small` the addends of larger/smaller magnitude, `t = s -
/// big` is exact (Sterbenz, or the sum itself is exact), so comparing `small` with `t` says
/// whether the exact sum lies above or below `s`, and `s` moves one `f32` ulp that way before
/// the rounding. The correction applies only when the rounded product `p` is a finite normal
/// `f32`: the product has at most 16 significant bits, so such a `p` is exact (below `2^-126`
/// its gap to `2^-126` exceeds half a denormal ulp). An overflowed (`±inf`) or denormal `p`
/// keeps the device's `s` unchanged, so the result is exact while `a * b` is a normal `f32` and
/// is never made worse otherwise (there `bfloat` follows the device's `float` behavior).
pub fn bf16_fma(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    use crate::spirv::ir::Contraction;
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let p = fb.binary_with(BinOp::FMul, a, b, Contraction::Forbidden)?;
    let ap = fb.glsl(GlslOp::FAbs, &[p])?;
    let ac = fb.glsl(GlslOp::FAbs, &[c])?;
    let p_big = bin(fb, BinOp::FOrdGreaterThanEqual, ap, ac)?;
    let big = fb.select(p_big, p, c)?;
    let small = fb.select(p_big, c, p)?;
    let t = fb.binary_with(BinOp::FSub, s, big, Contraction::Forbidden)?;
    let above = bin(fb, BinOp::FOrdGreaterThan, small, t)?;
    let below = bin(fb, BinOp::FOrdLessThan, small, t)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let sb = fb.convert(ConvOp::Bitcast, s, ut)?;
    let low = splat(fb, u, 0xFFFF)?;
    let low = bin(fb, BinOp::BitwiseAnd, sb, low)?;
    let tie_bits = splat(fb, u, 0x8000)?;
    let tie = bin(fb, BinOp::IEqual, low, tie_bits)?;
    // The correction's premise: `p` is the exact product, a finite normal `f32` (exponent
    // field 1..=254, so `bits(|p|) - 2^23 < 254 * 2^23` unsigned). Otherwise `s` is kept.
    let pb = fb.convert(ConvOp::Bitcast, ap, ut)?;
    let min_normal = splat(fb, u, 0x0080_0000)?;
    let pe = bin(fb, BinOp::ISub, pb, min_normal)?;
    let normal_span = splat(fb, u, 0x7F00_0000)?;
    let p_normal = bin(fb, BinOp::ULessThan, pe, normal_span)?;
    let tie = bin(fb, BinOp::LogicalAnd, tie, p_normal)?;
    let k31 = splat(fb, u, 31)?;
    let sign = bin(fb, BinOp::ShiftRightLogical, sb, k31)?;
    let z = splat(fb, u, 0)?;
    let neg = bin(fb, BinOp::INotEqual, sign, z)?;
    let one = splat(fb, u, 1)?;
    let plus = bin(fb, BinOp::IAdd, sb, one)?;
    let minus = bin(fb, BinOp::ISub, sb, one)?;
    // One ulp toward +inf / -inf (the magnitude grows toward the sign's infinity).
    let up = fb.select(neg, minus, plus)?;
    let down = fb.select(neg, plus, minus)?;
    let nb = fb.select(above, up, sb)?;
    let nb = fb.select(below, down, nb)?;
    let nb = fb.select(tie, nb, sb)?;
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, nb, ft)
}

/// Float resize (`fpext`/`fptrunc`/float `air.convert`) of `x` from `from` to `to`, exact up to
/// the one rounding to `to`'s format.
pub fn resize(fb: &mut FunctionBuilder<'_>, from: Form, to: Form, x: Value) -> BResult<Value> {
    let (Held::Float(src), Held::Float(dst)) = (from.held, to.held) else {
        return Ok(x);
    };
    if src == dst {
        return Ok(x);
    }
    // Narrow or widen the container first; only `double -> f32` rounds there.
    let x = match (src, dst) {
        (FloatFmt::F64, FloatFmt::Half | FloatFmt::BFloat) => f64_to_f32_odd(fb, from, x)?,
        (FloatFmt::F64, _) | (_, FloatFmt::F64) => {
            let t = ty(fb, to)?;
            fb.convert(ConvOp::FConvert, x, t)?
        }
        _ => x,
    };
    // Every value of a narrower format is one of the wider (half and bfloat are not ordered,
    // so between them the destination rounds).
    let widening = matches!(
        (src, dst),
        (
            FloatFmt::Half | FloatFmt::BFloat,
            FloatFmt::F32 | FloatFmt::F64
        ) | (FloatFmt::F32, FloatFmt::F64)
    );
    if widening {
        Ok(x)
    } else {
        round(fb, to, x)
    }
}

/// The layout of an 8-bit float format (`air.convert` sources, carried in `i8` lanes). The
/// names follow LLVM's `APFloat` semantics of the same spellings: `f8e4m3` and `f8e5m2` are
/// IEEE-like (an all-ones exponent is infinity or NaN), `f8e4m3fn` is finite (only
/// `S.1111.111` is NaN).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp8Layout {
    pub exp: u32,
    pub man: u32,
    pub bias: u32,
    pub ieee: bool,
}

impl Fp8Layout {
    pub fn of(f: crate::air::intrinsics::Fp8) -> Fp8Layout {
        use crate::air::intrinsics::Fp8;
        match f {
            Fp8::E4M3 => Fp8Layout {
                exp: 4,
                man: 3,
                bias: 7,
                ieee: true,
            },
            Fp8::E4M3Fn => Fp8Layout {
                exp: 4,
                man: 3,
                bias: 7,
                ieee: false,
            },
            Fp8::E5M2 => Fp8Layout {
                exp: 5,
                man: 2,
                bias: 15,
                ieee: true,
            },
        }
    }

    /// The value of bits `b` (the reference decoding, for tests and constants).
    #[cfg(test)]
    pub fn value(self, b: u8) -> f64 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = (u32::from(b) >> self.man) & ((1 << self.exp) - 1);
        let m = u32::from(b) & ((1 << self.man) - 1);
        let top = (1 << self.exp) - 1;
        if self.ieee && e == top {
            return if m == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }
        if !self.ieee && e == top && m == (1 << self.man) - 1 {
            return f64::NAN;
        }
        let m = f64::from(m) / f64::from(1u32 << self.man);
        if e == 0 {
            sign * m * 2f64.powi(1 - self.bias as i32)
        } else {
            sign * (1.0 + m) * 2f64.powi(e as i32 - self.bias as i32)
        }
    }
}

/// The value (an `f32` of form `to`'s lanes) of 8-bit float bits `b` (zero-extended in `u32`
/// lanes of form `u`). Exact: every 8-bit float value is an `f32` (and a `half` and a `bfloat`)
/// value.
pub fn fp8_to_f32(
    fb: &mut FunctionBuilder<'_>,
    l: Fp8Layout,
    u: Form,
    to: Form,
    b: Value,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let top = u64::from((1u32 << l.exp) - 1);
    let mmax = u64::from((1u32 << l.man) - 1);
    let (k24, k23, kman) = (c(fb, 24)?, c(fb, 23)?, c(fb, u64::from(l.man))?);
    let k80 = c(fb, 0x80)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, k80)?;
    let sign = bin(fb, BinOp::ShiftLeftLogical, sign, k24)?;
    let e = bin(fb, BinOp::ShiftRightLogical, b, kman)?;
    let et = c(fb, top)?;
    let e = bin(fb, BinOp::BitwiseAnd, e, et)?;
    let mm = c(fb, mmax)?;
    let m = bin(fb, BinOp::BitwiseAnd, b, mm)?;
    let up = c(fb, u64::from(23 - l.man))?;
    let mup = bin(fb, BinOp::ShiftLeftLogical, m, up)?;
    let rebias = c(fb, u64::from(127 - l.bias))?;
    let ne = bin(fb, BinOp::IAdd, e, rebias)?;
    let ne = bin(fb, BinOp::ShiftLeftLogical, ne, k23)?;
    let normal = bin(fb, BinOp::BitwiseOr, ne, mup)?;
    // Denormal: m * 2^(1 - bias - man), normalized from its most significant bit.
    let msb = fb.glsl(GlslOp::FindUMsb, &[m])?;
    let dbias = c(fb, u64::from(127 + 1 - l.bias - l.man))?;
    let de = bin(fb, BinOp::IAdd, msb, dbias)?;
    let de = bin(fb, BinOp::ShiftLeftLogical, de, k23)?;
    let sh = bin(fb, BinOp::ISub, k23, msb)?;
    let k31 = c(fb, 31)?;
    let sh = fb.glsl(GlslOp::UMin, &[sh, k31])?;
    let dm = bin(fb, BinOp::ShiftLeftLogical, m, sh)?;
    let man = c(fb, 0x7F_FFFF)?;
    let dm = bin(fb, BinOp::BitwiseAnd, dm, man)?;
    let den = bin(fb, BinOp::BitwiseOr, de, dm)?;
    let z = c(fb, 0)?;
    let m_zero = bin(fb, BinOp::IEqual, m, z)?;
    let den = fb.select(m_zero, z, den)?;
    let e_zero = bin(fb, BinOp::IEqual, e, z)?;
    let r = fb.select(e_zero, den, normal)?;
    let e_top = bin(fb, BinOp::IEqual, e, et)?;
    let r = if l.ieee {
        let inf = c(fb, 0x7F80_0000)?;
        let special = bin(fb, BinOp::BitwiseOr, inf, mup)?;
        fb.select(e_top, special, r)?
    } else {
        let m_top = bin(fb, BinOp::IEqual, m, mm)?;
        let nan_bits = bin(fb, BinOp::LogicalAnd, e_top, m_top)?;
        let qnan = c(fb, 0x7FC0_0000)?;
        fb.select(nan_bits, qnan, r)?
    };
    let r = bin(fb, BinOp::BitwiseOr, sign, r)?;
    let ft = ty(fb, to.with(Held::Float(FloatFmt::F32)))?;
    fb.convert(ConvOp::Bitcast, r, ft)
}
