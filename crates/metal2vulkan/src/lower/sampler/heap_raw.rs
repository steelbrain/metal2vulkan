//! Samples and gathers of an argument buffer's texture that a heap holds by its raw texels alone: a
//! `BGR10_XR`, `BGRA10_XR`, `R10Unorm_X6`, `RG10Unorm_X12`, `R16Bfloat`, `RG16Bfloat` or
//! `RGBA16Bfloat` texture (`ExtendedRange::HEAP_READ`).
//!
//! The binder holds such a texture's raw view in its heap group's unsigned integer class and the
//! neutral in its float class (`bind::heap`), and its handle's class index names its form past the
//! heap's own classes (`metal::texel_format::heap_extended_base`). A float read decodes the raw
//! texel where the index names a form (`texture::heap_raw`). A sample or gather of such a group
//! takes the device's access, which reads the neutral for such a texture, and where the handle
//! names a form, Metal's answer instead: the slot's emulation of that form's sample or gather
//! (`extended`), its taps fetched from the unsigned integer class at the same element and decoded
//! by the form the index names, through the heap's decode routine (`texture::heap_raw::define`).
//! The form is known at run time alone, so the emulation's per-form choices are the form's own at
//! run time: the format's bits for the exact routine and the border texel (`extended::flags`), a
//! bfloat form's nearest sample under either filter, and its `half` sample's rounding toward zero.
//!
//! **Built** as the slot's emulation is: a 2D texture's sample or gradient sample without
//! comparison, or gather of a component without comparison, with a `float` or `half` result, a
//! constant texel offset or none, no `min_lod_clamp`, and no observed residency, through a
//! constexpr sampler with one filter for minification and magnification (nearest or linear), no
//! mip filter, and no anisotropy; or through a runtime sampler slot not of the plan's cubic
//! variant (its `SamplerState` specialization's word), where the word states the same, its address
//! modes, border color and coordinates followed; elsewhere the device's access, the neutral. A
//! `half` gather is a bfloat form's alone: another form's is not measured. Every other sampling of
//! such a group (an argument buffer's sampler's among them) states `RuntimeFact::HeapRawSamples`
//! for its stage, and so does one through a runtime sampler slot, naming the slot, whose bound
//! sampler the planner checks.

use crate::air::abi::BorderColor;
use crate::lower::emit::EmitCx;
use crate::metal::format::ExtendedRange;
use crate::metal::sampler::SamplerWord;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::bicubic::{scalar, value, Addressing, TapAddress, Tys};
use super::extended::{At, Decode, Texture, AXES};
use super::{CombinedImage, Pixel, Sampling, Source};

/// A heap sample or gather that answers a texture a heap holds by its raw texels alone as Metal
/// does (module docs), beside the device's access.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HeapRaw {
    /// The sampler's state the emulation follows.
    pub state: State,
    /// A gather of this component (0 to 3, red first) instead of a sample.
    pub gather: Option<u32>,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The result is `half`, else `float`.
    pub half: bool,
}

/// The state of the sampler a raw heap sample follows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    /// A constexpr sampler's, known at translation: linear minification and magnification,
    /// else nearest; the addressing of the `s` and `t` axes, the border color, and whether the
    /// coordinate is in pixels.
    Constant {
        linear: bool,
        address: Vec<TapAddress>,
        border: BorderColor,
        pixel: bool,
    },
    /// The runtime sampler in slot `sampler`: its `SamplerState` specialization's word states the
    /// filter, the address modes, the border color, and the coordinates; emulated where it states
    /// one filter for minification and magnification, no mip filter, and no anisotropy.
    Slot { sampler: u32 },
}

/// The census words of the emulation's head: the handle's class index, its test and the form, the
/// word's tests, the branch and its merge, the unsigned element's access chain and load, and the
/// form's bits' selects.
const HEAD_WORDS: u32 = 96 + 7 * 16;

/// The census words of one call of the heap's decode routine: the call, its operands, and its
/// constant.
const DECODE_CALL_WORDS: u32 = 24;

impl HeapRaw {
    /// Whether it answers every texture a heap holds by its raw texels alone as Metal does, but
    /// for its runtime sampler's state (`RuntimeFact::HeapRawSamples` names that slot): every
    /// one but a `half` gather, whose non-bfloat forms are not measured.
    pub(crate) fn whole(&self) -> bool {
        !(self.half && self.gather.is_some())
    }

    /// The estimated words: the head, and per filter the emulation takes (a filtered sample's, a
    /// nearest one's, or a gather's) its position, footprint and taps' addressing (a word's under
    /// every mode), its fetches and their decodes' calls, the routine's call, and Metal's offset
    /// rule on each axis an offset moves; a `half` nearest sample's two roundings and their select;
    /// the result's composite.
    pub(crate) fn words(&self) -> u32 {
        let axes = AXES as u32;
        let moved = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32);
        let offset = moved * super::position::OFFSET_RULE_WORDS;
        let position = axes * super::position::POSITION_WORDS;
        let (word, taps) = match &self.state {
            State::Constant { pixel: true, .. } => (0, 0),
            State::Constant { pixel: false, .. } => (position, 0),
            State::Slot { .. } => (
                super::linear::WORD_WORDS + position + axes * super::position::MODE_POSITION_WORDS,
                axes * super::linear::WORD_TAP_WORDS,
            ),
        };
        let linear = super::linear::PLANE_WORDS + 4 * DECODE_CALL_WORDS + offset + word + 2 * taps;
        let nearest =
            super::extended::NEAREST_WORDS + 2 * DECODE_CALL_WORDS + 64 + offset + word + taps;
        let body = match (self.gather, &self.state) {
            (Some(_), _) => linear + 4 * 16,
            (None, State::Constant { linear: false, .. }) => nearest,
            (None, _) => super::extended::BRANCH_WORDS + linear + nearest,
        };
        HEAD_WORDS + body + 16
    }
}

/// The value of sampling `s` with emulation `raw`, `device` the device's access's value (its
/// result type, `s.result`): where `s`'s heap handle names a form a heap reads by its raw texels
/// alone (and a runtime sampler's word states what the emulation answers, and for a `half`
/// gather the form is a bfloat one), Metal's sample or gather of that form, fetched from the
/// group's unsigned integer class (module docs); else `device`.
pub(crate) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    raw: &HeapRaw,
    device: SValue,
) -> BResult<SValue> {
    let (group, handle) = match s.source {
        Source::Heap { handle, key } => (key.group, handle),
        Source::HeapSampler { handle, key, .. } => (key.group, handle),
        Source::Combined(_) => return Err(cx.fail("a raw heap sample of a slot")),
    };
    let f32_routine = cx.decl.families.texture.heap_raw.get(&false).copied();
    let half_routine = cx.decl.families.texture.heap_raw.get(&true).copied();
    let Some(f32_routine) = f32_routine else {
        return Err(cx.fail("a raw heap sample without its routine"));
    };
    if raw.half && half_routine.is_none() {
        return Err(cx.fail("a raw heap `half` sample without its routine"));
    }
    let t = Tys::new(fb)?;
    let h = cx.operand(fb, handle)?;
    let flags = crate::lower::texture::heap_flags(cx, fb, h)?;
    let (form, member) = crate::lower::texture::heap_raw_form(fb, flags)?;
    let bfloat = bfloat_form(fb, &t, form)?;
    // A runtime sampler's word, and whether it states a filter the emulation answers.
    let word = match &raw.state {
        State::Slot { sampler } => Some(super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?),
        State::Constant { .. } => None,
    };
    let mut take = member;
    let mut linear_word = None;
    if let Some(w) = word {
        let mask = scalar(fb, t.u32_, u64::from(SamplerWord::LINEAR_FILTER))?;
        let mask = value(fb, mask)?;
        let filter = fb.binary(BinOp::BitwiseAnd, w, mask)?;
        let linear = scalar(fb, t.u32_, u64::from(SamplerWord::LINEAR))?;
        let linear = value(fb, linear)?;
        let zero = scalar(fb, t.u32_, 0)?;
        let zero = value(fb, zero)?;
        let is_linear = fb.binary(BinOp::IEqual, filter, linear)?;
        let is_nearest = fb.binary(BinOp::IEqual, filter, zero)?;
        let answers = fb.binary(BinOp::LogicalOr, is_linear, is_nearest)?;
        take = fb.binary(BinOp::LogicalAnd, take, answers)?;
        linear_word = Some(is_linear);
    }
    // A `half` gather of a form but a bfloat one is not measured: the device's.
    if raw.half && raw.gather.is_some() {
        take = fb.binary(BinOp::LogicalAnd, take, bfloat)?;
    }
    let (t, word) = (&t, word);
    let emulated: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let img = crate::lower::texture::heap_raw_image(cx, fb, group, h)?;
        let flags = form_flags(fb, t, form)?;
        let tex = Texture {
            img,
            image: CombinedImage {
                plane: AXES as u8,
                arrayed: false,
                texel: crate::lower::texture::TexelClass::Uint,
            },
            decode: Decode::Heap {
                form,
                bfloat,
                routines: (f32_routine, half_routine),
            },
            flags,
            swizzle: None,
            // A heap element's access reads no slot's `OffsetBaseAxes`.
            parent: None,
            size: None,
        };
        let at = At {
            coord: super::emit::plane_coord(cx, fb, s)?,
            layer: None,
            offset: raw.offset.as_deref(),
            passed: None,
            level: None,
            half: raw.half,
        };
        let addressing = match (&raw.state, word) {
            (
                State::Constant {
                    address,
                    border,
                    pixel,
                    ..
                },
                _,
            ) => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            (State::Slot { .. }, Some(word)) => Addressing::Word { word, axes: AXES },
            (State::Slot { .. }, None) => return Err(cx.fail("a raw heap sample without its word")),
        };
        // Whether Metal filters linearly: a linear sampler's sample of a form but a bfloat one.
        let filter = match (raw.gather, &raw.state, linear_word) {
            (Some(_), _, _) | (None, State::Constant { linear: false, .. }, _) => None,
            (None, State::Constant { linear: true, .. }, _) => {
                Some(fb.unary(crate::spirv::ir::UnOp::LogicalNot, bfloat)?)
            }
            (None, State::Slot { .. }, Some(linear)) => {
                let not_bfloat = fb.unary(crate::spirv::ir::UnOp::LogicalNot, bfloat)?;
                Some(fb.binary(BinOp::LogicalAnd, linear, not_bfloat)?)
            }
            (None, State::Slot { .. }, None) => {
                return Err(cx.fail("a raw heap sample without its word's filter"))
            }
        };
        let x = match (raw.gather, filter) {
            (Some(component), _) => {
                super::extended::gathered(cx, fb, t, &at, &tex, addressing, component)?
            }
            (None, None) => super::extended::nearest(cx, fb, t, &at, &tex, addressing)?,
            (None, Some(filter)) => {
                let (at, tex) = (&at, &tex);
                let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                    let x = super::extended::filtered(cx, fb, t, at, tex, addressing)?;
                    Ok(ArmEnd::FallThrough(vec![x]))
                });
                let nearest: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                    let x = super::extended::nearest(cx, fb, t, at, tex, addressing)?;
                    Ok(ArmEnd::FallThrough(vec![x]))
                });
                let Merged(merged) = fb.if_(cx, filter, filtered, Some(nearest))?;
                match merged[..] {
                    [x] => x,
                    _ => return Err(cx.fail("a raw heap sample's filter without its texel")),
                }
            }
        };
        let x = crate::lower::texture::with_word(cx, fb, &s.result, x, None)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let device_arm: ArmFn<'_, EmitCx<'_>> =
        Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![device])));
    let Merged(merged) = fb.if_(cx, take, emulated, Some(device_arm))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a raw heap sample's branch without its value")),
    }
}

/// Whether the form at index `form` (a `u32`) of `ExtendedRange::HEAP_READ` is a bfloat one (a
/// `bool`): the bfloat forms close the list.
fn bfloat_form(fb: &mut FunctionBuilder<'_>, t: &Tys, form: SValue) -> BResult<SValue> {
    let first = ExtendedRange::HEAP_READ
        .iter()
        .position(|f| f.bfloat())
        .unwrap_or(ExtendedRange::HEAP_READ.len());
    let first = scalar(fb, t.u32_, first as u64)?;
    let first = value(fb, first)?;
    fb.binary(BinOp::UGreaterThanEqual, form, first)
}

/// The bits of the form at index `form` (a `u32`) of `ExtendedRange::HEAP_READ` that the exact
/// routine and the border texel read (`extended::flags`), selected at run time.
fn form_flags(fb: &mut FunctionBuilder<'_>, t: &Tys, form: SValue) -> BResult<SValue> {
    let mut v = None;
    for (k, &f) in (0u64..).zip(ExtendedRange::HEAP_READ.iter()) {
        let bits = scalar(fb, t.u32_, u64::from(super::extended::flags(f)))?;
        let bits = value(fb, bits)?;
        v = Some(match v {
            None => bits,
            Some(v) => {
                let k = scalar(fb, t.u32_, k)?;
                let k = value(fb, k)?;
                let is = fb.binary(BinOp::IEqual, form, k)?;
                fb.select(is, bits, v)?
            }
        });
    }
    v.ok_or_else(super::bicubic::missing_routine)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bfloat forms close `ExtendedRange::HEAP_READ`, as [`bfloat_form`]'s test reads them.
    #[test]
    fn the_bfloat_forms_close_the_heap_forms() {
        let first = ExtendedRange::HEAP_READ
            .iter()
            .position(|f| f.bfloat())
            .unwrap();
        assert!(ExtendedRange::HEAP_READ[first..].iter().all(|f| f.bfloat()));
        assert!(!ExtendedRange::HEAP_READ[..first].iter().any(|f| f.bfloat()));
    }
}
