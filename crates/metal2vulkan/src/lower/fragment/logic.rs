//! The private logic operation of a render pipeline (`LoweringInputs::logic_operation`), computed
//! by the fragment function on a device without Vulkan's `logicOp`.
//!
//! Metal applies the operation to the code of each channel of a normalized or integer attachment
//! the function writes, the fragment's code against the stored one, blending ignored and the write
//! mask kept after it (`pipeline::private_state`'s module docs). On a device that orders
//! framebuffer fetch, the function's first output's write to such an attachment reads the pixel's
//! texel, in the interlocked storage form (`fetch`) or through the attachment's input attachment
//! under rasterization-order access, and stores the operation's result itself ([`operated`]):
//!
//! - the stored codes from the texel the read gives: a normalized channel's value times its
//!   largest code, rounded to nearest even (exact: the read's value is a code over that largest
//!   code), an integer channel's bits;
//! - the fragment's codes from its output by the device's own arithmetic: a normalized channel's
//!   value clamped (`NClamp`: a NaN the low end, 0, or -1 for a signed channel, as Metal's logic
//!   operation takes it) times its largest code in `float`, rounded to nearest even
//!   (`RoundEven`), where Metal rounds the exact product to nearest; the two differ only where
//!   the `float` product rounds onto or across a half-code tie
//!   (`PrecisionNote::DeviceColorConversion`); an integer channel's value saturated to the
//!   channel's range, as Metal saturates it;
//! - the operation of the two, by bits, in the channel's width;
//! - the result written as the value the device's write stores as that code: a normalized code
//!   over the largest code, an integer's bits.
//!
//! A signed normalized channel reads its lowest code, `-2^(n-1)`, as -1, as it reads the code
//! above it, and stores a result of the lowest code as -1, the code above: so does Metal's logic
//! operation (probe `logic_operation`, an Apple M3 Max under macOS 27.0.1).
//!
//! The formats this serves ([`channels`]): every channel an 8- or 16-bit unorm or snorm one, or an
//! 8-, 16- or 32-bit integer one. Metal's operation into a 10-bit or a packed 16-bit format is not
//! its per-channel rule (the same probe: a `RGB10A2Unorm` copy stores a code beside the rounded
//! product, a `B5G6R5Unorm` one past it, and `RGB10A2Uint` saturates the operation's 32-bit
//! result), and is refused there (`pipeline::private_state::logic_operation`). So is a write the
//! operation would read otherwise than as the format's codes: a narrower integer output than 32
//! bits ([`takes`]), an attachment held through a word view, under alpha to coverage in the storage
//! form, and, through an input attachment, one held with its channels rotated or with Metal's alpha
//! in red.
//!
//! The write mask applies after the operation: the storage form takes the kept channels back from
//! the texel it read (`fetch`), and over an input attachment the device's write mask keeps them.
//! Each read is ordered after every earlier fragment's write of the pixel (the pixel interlock, or
//! rasterization-order attachment access), so overlapping fragments operate in primitive order, as
//! Metal's do; a multisampled pipeline's function runs per sample and operates on its own sample's
//! codes.

use std::collections::BTreeMap;

use crate::air::abi::{AbiInt, OutputRole};
use crate::air::ir::TypeId as AirTypeId;
use crate::error::{Attachment, Error, Feature, PrivateRenderField, Unsupported};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::format::{write_conversion, WriteConversion};
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::pipeline::sym::{SampledType, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type as SType, UnOp};
use crate::vk;

/// How an attachment's channels hold their codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicNumeric {
    Unorm,
    Snorm,
    Uint,
    Sint,
}

/// The channels of an attachment the function computes the logic operation of ([`channels`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicChannels {
    pub numeric: LogicNumeric,
    /// The bits of Metal's red, green, blue and alpha; `None` where the format lacks one.
    pub bits: [Option<u32>; 4],
}

/// An upper estimate of the words one channel's operation adds: the two codes, the operation, the
/// mask and the value written.
const CHANNEL_WORDS: u64 = 64;

/// An upper estimate of the words the operation of one write adds.
pub(crate) const WRITE_WORDS: u64 = 4 * CHANNEL_WORDS + 24;

/// The channels of a color attachment of `pf` whose logic operation the function computes (see
/// the module docs): every channel an 8- or 16-bit unorm or snorm one, or an 8-, 16- or 32-bit
/// integer one; `None` for another format.
pub(crate) fn channels(pf: PixelFormat) -> Option<LogicChannels> {
    match write_conversion(pf)? {
        WriteConversion::Rules(rules) => {
            let numeric = match rules.iter().flatten().next()? {
                ChannelWrite::Unorm { .. } => LogicNumeric::Unorm,
                ChannelWrite::Snorm { .. } => LogicNumeric::Snorm,
                _ => return None,
            };
            let mut bits = [None; 4];
            for (b, rule) in bits.iter_mut().zip(rules) {
                *b = match (numeric, rule) {
                    (_, None) => None,
                    (LogicNumeric::Unorm, Some(ChannelWrite::Unorm { bits: n @ (8 | 16) }))
                    | (
                        LogicNumeric::Snorm,
                        Some(ChannelWrite::Snorm {
                            bits: n @ (8 | 16), ..
                        }),
                    ) => Some(u32::from(n)),
                    _ => return None,
                };
            }
            Some(LogicChannels { numeric, bits })
        }
        WriteConversion::Exact => {
            let numeric = match crate::metal::format::sampled_type(pf)? {
                SampledType::Uint => LogicNumeric::Uint,
                SampledType::Sint => LogicNumeric::Sint,
                _ => return None,
            };
            let block = crate::metal::format::direct_format(pf)?.block()?;
            let width = block.component_bits.filter(|b| matches!(b, 8 | 16 | 32))?;
            let n = crate::metal::format::color_output_components(pf)?;
            let mut bits = [None; 4];
            for b in bits.iter_mut().take(usize::from(n)) {
                *b = Some(width);
            }
            Some(LogicChannels { numeric, bits })
        }
        _ => None,
    }
}

/// The read class of an attachment of `channels`: `float` texels for a normalized format, `uint`
/// ones for an integer format (a signed one's bits).
pub(crate) fn reads_float(channels: LogicChannels) -> bool {
    matches!(channels.numeric, LogicNumeric::Unorm | LogicNumeric::Snorm)
}

fn u32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

fn i32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })
}

fn f32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

fn u32_value(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn f32_value(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<SValue> {
    let one = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    ops::fsplat(fb, one, f64::from(v))
}

fn i32_value(fb: &mut FunctionBuilder<'_>, v: i32) -> BResult<SValue> {
    let one = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    let bits = ops::splat(fb, one, u64::from(v as u32))?;
    let i32_ = i32_ty(fb)?;
    fb.convert(ConvOp::Bitcast, bits, i32_)
}

/// `op` of the source code `s` and the stored code `d` (`u32`s), by bits, unmasked.
fn operate(fb: &mut FunctionBuilder<'_>, op: vk::LogicOp, s: SValue, d: SValue) -> BResult<SValue> {
    use vk::LogicOp as L;
    let not = |fb: &mut FunctionBuilder<'_>, x| fb.unary(UnOp::Not, x);
    Ok(match op {
        L::CLEAR => u32_value(fb, 0)?,
        L::SET => u32_value(fb, u32::MAX)?,
        L::COPY => s,
        L::COPY_INVERTED => not(fb, s)?,
        L::NO_OP => d,
        L::INVERT => not(fb, d)?,
        L::AND => fb.binary(BinOp::BitwiseAnd, s, d)?,
        L::NAND => {
            let x = fb.binary(BinOp::BitwiseAnd, s, d)?;
            not(fb, x)?
        }
        L::OR => fb.binary(BinOp::BitwiseOr, s, d)?,
        L::NOR => {
            let x = fb.binary(BinOp::BitwiseOr, s, d)?;
            not(fb, x)?
        }
        L::XOR => fb.binary(BinOp::BitwiseXor, s, d)?,
        L::EQUIVALENT => {
            let x = fb.binary(BinOp::BitwiseXor, s, d)?;
            not(fb, x)?
        }
        L::AND_REVERSE => {
            let nd = not(fb, d)?;
            fb.binary(BinOp::BitwiseAnd, s, nd)?
        }
        L::AND_INVERTED => {
            let ns = not(fb, s)?;
            fb.binary(BinOp::BitwiseAnd, ns, d)?
        }
        L::OR_REVERSE => {
            let nd = not(fb, d)?;
            fb.binary(BinOp::BitwiseOr, s, nd)?
        }
        // `OR_INVERTED`, the last of Metal's sixteen.
        _ => {
            let ns = not(fb, s)?;
            fb.binary(BinOp::BitwiseOr, ns, d)?
        }
    })
}

/// The largest code of a channel of `bits` of `numeric` (a normalized one's).
fn largest(numeric: LogicNumeric, bits: u32) -> f32 {
    match numeric {
        LogicNumeric::Snorm => ((1u32 << (bits - 1)) - 1) as f32,
        _ => ((1u64 << bits) - 1) as f32,
    }
}

/// The mask of a channel of `bits`.
fn mask_of(bits: u32) -> u32 {
    match bits {
        32 => u32::MAX,
        b => (1u32 << b) - 1,
    }
}

/// One channel: the operation `op` of the output lane `x` (an `f32` for a normalized channel, the
/// `u32` bits of an integer one) over the stored lane `stored` (the read's `f32`, or `u32` bits),
/// in a channel of `bits` of `numeric`, as the lane written (see the module docs).
fn channel(
    fb: &mut FunctionBuilder<'_>,
    op: vk::LogicOp,
    (numeric, bits): (LogicNumeric, u32),
    x: SValue,
    stored: SValue,
) -> BResult<SValue> {
    let u32_ = u32_ty(fb)?;
    let i32_ = i32_ty(fb)?;
    let mask = u32_value(fb, mask_of(bits))?;
    let (s, d) = match numeric {
        LogicNumeric::Unorm | LogicNumeric::Snorm => {
            let max = f32_value(fb, largest(numeric, bits))?;
            let lo = f32_value(
                fb,
                if numeric == LogicNumeric::Snorm {
                    -1.0
                } else {
                    0.0
                },
            )?;
            let one = f32_value(fb, 1.0)?;
            let clamped = fb.glsl(GlslOp::NClamp, &[x, lo, one])?;
            let scaled = fb.binary(BinOp::FMul, clamped, max)?;
            let s = fb.glsl(GlslOp::RoundEven, &[scaled])?;
            let read = fb.binary(BinOp::FMul, stored, max)?;
            let d = fb.glsl(GlslOp::RoundEven, &[read])?;
            match numeric {
                LogicNumeric::Unorm => (
                    fb.convert(ConvOp::ConvertFToU, s, u32_)?,
                    fb.convert(ConvOp::ConvertFToU, d, u32_)?,
                ),
                _ => {
                    let s = fb.convert(ConvOp::ConvertFToS, s, i32_)?;
                    let d = fb.convert(ConvOp::ConvertFToS, d, i32_)?;
                    (
                        fb.convert(ConvOp::Bitcast, s, u32_)?,
                        fb.convert(ConvOp::Bitcast, d, u32_)?,
                    )
                }
            }
        }
        // Saturated to the channel's range, as Metal saturates an integer write.
        LogicNumeric::Uint => {
            let s = match bits {
                32 => x,
                _ => fb.glsl(GlslOp::UMin, &[x, mask])?,
            };
            (s, stored)
        }
        LogicNumeric::Sint => {
            let s = match bits {
                32 => x,
                _ => {
                    let signed = fb.convert(ConvOp::Bitcast, x, i32_)?;
                    let lo = i32_value(fb, -(1i32 << (bits - 1)))?;
                    let hi = i32_value(fb, (1i32 << (bits - 1)) - 1)?;
                    let c = fb.glsl(GlslOp::SClamp, &[signed, lo, hi])?;
                    fb.convert(ConvOp::Bitcast, c, u32_)?
                }
            };
            (s, stored)
        }
    };
    let d = fb.binary(BinOp::BitwiseAnd, d, mask)?;
    let r = operate(fb, op, s, d)?;
    let r = fb.binary(BinOp::BitwiseAnd, r, mask)?;
    // The value the device's write stores as the code: a signed code's bits sign-extended.
    let signed = |fb: &mut FunctionBuilder<'_>, r: SValue| -> BResult<SValue> {
        if bits == 32 {
            return fb.convert(ConvOp::Bitcast, r, i32_);
        }
        let shift = u32_value(fb, 32 - bits)?;
        let up = fb.binary(BinOp::ShiftLeftLogical, r, shift)?;
        let up = fb.convert(ConvOp::Bitcast, up, i32_)?;
        fb.binary(BinOp::ShiftRightArithmetic, up, shift)
    };
    match numeric {
        LogicNumeric::Unorm => {
            let f32_ = f32_ty(fb)?;
            let v = fb.convert(ConvOp::ConvertUToF, r, f32_)?;
            let max = f32_value(fb, largest(numeric, bits))?;
            fb.binary(BinOp::FDiv, v, max)
        }
        LogicNumeric::Snorm => {
            let f32_ = f32_ty(fb)?;
            let r = signed(fb, r)?;
            let v = fb.convert(ConvOp::ConvertSToF, r, f32_)?;
            let max = f32_value(fb, largest(numeric, bits))?;
            fb.binary(BinOp::FDiv, v, max)
        }
        LogicNumeric::Uint => Ok(r),
        LogicNumeric::Sint => {
            let r = signed(fb, r)?;
            fb.convert(ConvOp::Bitcast, r, u32_)
        }
    }
}

/// The value a write of output `part` (of `form`: one to four `f32` lanes of a `half` or `float`
/// output, or `u32` lanes of an integer one) stores in an attachment of `channels` under Metal's
/// logic operation `op`, over the attachment's texel `texel` in Metal's channel order (four `f32`
/// lanes for a normalized format, four `u32` lanes of an integer one's bits), in the output's form
/// (see the module docs). A lane the format lacks is the output's.
pub(crate) fn operated(
    fb: &mut FunctionBuilder<'_>,
    op: vk::LogicOp,
    channels: LogicChannels,
    (form, part): (Form, SValue),
    texel: SValue,
) -> BResult<SValue> {
    let mut lanes = Vec::with_capacity(usize::from(form.lanes));
    for (lane, bits) in (0..u32::from(form.lanes)).zip(channels.bits) {
        let x = match form.lanes {
            1 => part,
            _ => fb.composite_extract(part, &[lane])?,
        };
        let Some(bits) = bits else {
            lanes.push(x);
            continue;
        };
        let stored = fb.composite_extract(texel, &[lane])?;
        lanes.push(channel(fb, op, (channels.numeric, bits), x, stored)?);
    }
    match lanes.as_slice() {
        [one] => Ok(*one),
        _ => {
            let ty = ops::ty(fb, form)?;
            fb.composite_construct(ty, &lanes)
        }
    }
}

/// Whether the operation takes an output of `form` to an attachment of `channels`: a `float` or
/// `half` output for a normalized format, a 32-bit integer one for an integer format (a narrower
/// integer output's operation is not built).
pub(crate) fn takes(channels: LogicChannels, form: Form) -> bool {
    match reads_float(channels) {
        true => form.is_float(),
        false => matches!(form.held, Held::Int { bits: 32, .. }),
    }
}

/// The refusal of the operation at attachment `color` where the function's write there is outside
/// what this module computes.
pub(crate) fn refused(color: u32) -> Error {
    let gap = Feature::PrivateRenderState(PrivateRenderField::LogicOperation);
    Unsupported::not_implemented(gap)
        .at_attachment(Attachment::Color(color))
        .into()
}

/// An attachment whose logic operation the function computes over its input attachment, on a
/// device that orders framebuffer fetch through rasterization-order input attachments (see the
/// module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicInput {
    /// The pipeline's operation.
    pub op: vk::LogicOp,
    /// The attachment's channels.
    pub channels: LogicChannels,
}

/// Plan the attachments the function writes through a first output under the pipeline's private
/// logic operation (`LoweringInputs::logic_operation`) on a device that orders framebuffer fetch
/// through input attachments, before the stage interface: each one whose format the operation
/// serves ([`channels`]) and that the pipeline's state names (`FramebufferFetchInputs`) reads its
/// input attachment, and states `RuntimeFact::FramebufferFetch`; a normalized one's codes are the
/// device's conversion (`PrecisionNote::DeviceColorConversion`). An attachment held through a
/// word, with its channels rotated, or with Metal's alpha in red, and an output the operation does
/// not take ([`takes`]), are refused naming the operation. (On a device that orders fetch through
/// the pixel interlock, the storage form computes it, `fetch`.)
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let Some(op) = cx.inputs.logic_operation else {
        return Ok(());
    };
    if cx.stage != Stage::Fragment || !super::fetch::inputs_order(cx.profile) {
        return Ok(());
    }
    let outputs: Vec<(u32, AirTypeId)> = cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(crate::lower::stage_io::output_types(cx)?.1)
        .filter_map(|(o, ty)| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } => Some((color, ty)),
            _ => None,
        })
        .collect();
    let mut planned = BTreeMap::new();
    for (color, ty) in outputs {
        let Some(pf) = cx
            .inputs
            .framebuffer_fetch
            .as_ref()
            .and_then(|i| i.attachments.get(&color).copied().flatten())
            .map(|a| a.pixel_format)
        else {
            continue;
        };
        let Some(channels) = channels(pf) else {
            continue;
        };
        let held_otherwise = cx.inputs.word_colors.contains_key(&color)
            || cx.inputs.rotated_colors.contains(&color)
            || cx.inputs.alpha_in_red_colors.contains(&color);
        let form =
            super::color::form(cx, ty).map_err(|e| e.at_attachment(Attachment::Color(color)))?;
        if held_otherwise || !takes(channels, form) {
            return Err(refused(color));
        }
        cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
        cx.facts.insert(RuntimeFact::FramebufferFetch { color });
        if reads_float(channels) {
            cx.precision.insert(PrecisionNote::DeviceColorConversion);
        }
        planned.insert(color, LogicInput { op, channels });
    }
    let reads = !planned.is_empty();
    cx.families.fragment.logic_inputs = planned;
    match reads {
        true => super::fetch::plan_sample(cx),
        false => Ok(()),
    }
}

/// The value a write of first output `part` (of `form`) to attachment `color` stores, whose logic
/// operation the function computes over its input attachment ([`plan`]): the operation of the
/// output's codes over those of the texel read there ([`operated`]).
pub(crate) fn input_operated(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    let Some(li) = cx.plan.families.fragment.logic_inputs.get(&color).copied() else {
        return Err(cx.fail("a logic operation of an attachment the plan does not name"));
    };
    let texel = super::color::input_texel(cx, fb, color)?;
    operated(fb, li.op, li.channels, (form, part), texel)
}

/// The words the planned operations add at `rets` returns: one input attachment each, and at
/// each return the read and the operation.
pub(super) fn words(cx: &SelectCx<'_>, rets: u64) -> u64 {
    let planned = cx.families.fragment.logic_inputs.len() as u64;
    planned.saturating_mul(32u64.saturating_add(WRITE_WORDS.saturating_mul(rets.max(1))))
}
