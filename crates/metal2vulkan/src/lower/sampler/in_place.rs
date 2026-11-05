//! Samples of a two-plane YCbCr texture whose planes the function reads in place
//! (`bind::YCbCrInPlace`, `LoweringInputs::ycbcr_in_place`): the slot is the planes' uniform texel
//! buffer, and every tap of a sample is the texel Metal reads there, computed from the codes.
//!
//! **A tap** at texel `(x, y)` (`bind::ycbcr_convert`'s module docs, measured there): the luma
//! code at its place, the four nearest chroma pairs (column `⌊x / 2⌋` and the one after it for an
//! odd `x`, before it for an even one, each clamped to the plane, rows alike) weighted 9/3/3/1 and
//! rounded to a code, `C = (S + 8) >> 4`, each code the byte or the high 10 bits of its 16-bit
//! word; then the conversion of the surface's range and matrix (`bind::ycbcr_read`): every
//! channel `v / 2048` exactly, alpha 1. An element of the texel buffer holds two luma codes or one
//! chroma pair, so a tap fetches five elements. The texture's size, its planes' first elements
//! and rows, and its conversion are the slot's push field (`PushField::YCbCrInPlace`), read once
//! where it dominates.
//!
//! **A sample** is Metal's filter of those texels: a nearest sample the tap at Metal's position,
//! a linear one the exact routine's sum of the four taps of Metal's footprint on the grid `2048`
//! (`PackedTexels::YCbCr2048`, the class the texture's image of converted texels is filtered
//! under), through the extended-range emulation's footprint and routines (`super::extended`), its
//! texel offset placed as there. Metal samples such a texture's one level whatever the sampler's
//! mip filter, level, bias or gradients, so the sample reads the base level alone. **A gather** of
//! a component is the four taps of the linear footprint, each the texel Metal reads there (a border
//! tap the border texel), that component of each in Vulkan's order, as the extended-range
//! emulation gathers (`extended::gathered`); it follows no filter. A 4:2:0
//! texture samples repeat, mirrored repeat and mirror-clamp-to-edge as clamp-to-edge: a constexpr
//! sampler's modes arrive mapped (`LoweringInputs::address_maps`), and a runtime sampler's word is
//! mapped here ([`mapped_word`]). A border tap reads the border color through the format's
//! channels, alpha 1.
//!
//! **The device's sampler** (`LoweringInputs::ycbcr_planes_sampled`): where the consumer holds
//! the planes as an image of the multi-planar 4:2:0 format over their bytes and every sample of the
//! slot is one the device's own sampler Y′CbCr conversion takes (no border address mode, no texel
//! offset placed by a parent's size, no gather: `VUID-RuntimeSpirv-OpTypeSampledImage-12206`
//! admits only samples through that conversion), the slot is a combined image sampler whose immutable sampler
//! is that conversion (`ResourceKind::YCbCrPlanes`), and a sample is the device's sample of the
//! image at level 0: a linear one at Metal's coordinate (normalized from pixels by the push field's
//! size, its texel offset moved in texels), a nearest one at the centre of the luma texel Metal
//! reads, clamped to the edge (where the device's linear filter of its conversion reads that
//! texel's luma and Metal's 9/3/3/1 chroma weights). The answer is the device's conversion and
//! filter, not Metal's integer arithmetic. No `ConstOffset` or `Offset` reaches the sample
//! (`VUID-RuntimeSpirv-ConstOffset-10718`), and the size queries read the push field as above.

use crate::air::ir::ValueId;
use crate::lower::emit::EmitCx;
use crate::metal::PixelFormat;
use crate::pipeline::sym::{PushField, SlotRef};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageOperands, Type as SType};

use super::bicubic::{Addressing, Tys};
use super::extended::{Decode, State, Texture};
use super::{CombinedImage, Pixel};

/// A sample of an in-place texture slot (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The texture slot, under its in-place variant.
    pub texture: u32,
    /// The sampler's state the sample follows: a constexpr sampler's modes as the texture's map
    /// sends them, or a runtime sampler's word.
    pub state: State,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The offset moves an axis that may repeat or mirror: it is placed by the slot's
    /// `SpecKey::OffsetBaseAxes` tests, as an extended-range sample places it.
    pub offset_base: bool,
    /// The plane coordinate: two floats.
    pub coord: ValueId,
    /// The result is `half`, else `float`.
    pub half: bool,
    /// A gather of this component (0 to 3, red first) instead of a sample (module docs).
    pub gather: Option<u32>,
    /// The lowered type of the result (`{T, i8}`, its status word 0).
    pub result: DemoteTy,
}

/// The census words of a slot's constants: the push field's words and the conversion's selects.
const PREPARE_WORDS: u32 = 192;
/// The census words of one tap: five fetches, the chroma filter, and the conversion.
const TAP_WORDS: u32 = 224;
/// The census words of a runtime sampler's mapped word.
const WORD_WORDS: u32 = 96;

impl Sample {
    /// The census name of the sample's form.
    pub(crate) fn name(&self) -> &'static str {
        if self.gather.is_some() {
            return "sampler.ycbcr_in_place.gather";
        }
        match self.state {
            State::Constant { linear: true, .. } => "sampler.ycbcr_in_place.linear",
            State::Constant { linear: false, .. } => "sampler.ycbcr_in_place.nearest",
            State::Slot { .. } => "sampler.ycbcr_in_place.runtime",
        }
    }

    /// The estimated words of the sample: the slot's constants, a linear filter's four taps and
    /// its footprint (a runtime sampler's nearest tap and its own footprint besides, the word and
    /// its branch; a gather's four taps and footprint alone, a runtime sampler's word besides), and
    /// Metal's offset rule on each lane of an offset.
    pub(crate) fn words(&self) -> u32 {
        let position = 2 * super::position::POSITION_WORDS;
        let linear = super::linear::PLANE_WORDS + 4 * TAP_WORDS + position;
        let nearest = super::extended::NEAREST_WORDS + TAP_WORDS + position;
        let rule = match self.offset_base {
            true => super::position::OFFSET_RULE_WORDS + super::position::PARENT_RULE_WORDS,
            false => super::position::OFFSET_RULE_WORDS,
        };
        let offset = self.offset.as_deref().map_or(0, |o| o.len() as u32) * rule;
        let filters = match (&self.state, self.gather) {
            (State::Constant { .. }, Some(_)) => linear,
            (State::Slot { .. }, Some(_)) => linear + WORD_WORDS,
            (State::Constant { linear: true, .. }, None) => linear,
            (State::Constant { linear: false, .. }, None) => nearest,
            (State::Slot { .. }, None) => {
                linear + nearest + WORD_WORDS + super::extended::BRANCH_WORDS
            }
        };
        PREPARE_WORDS + filters + offset + 16
    }
}

/// The constants of an in-place texture's taps, read from the slot's push field
/// (`bind::YCbCrInPlace::words`).
#[derive(Clone, Copy)]
pub(crate) struct Planes {
    /// The luma plane's first element and the elements between its rows (`u32`s).
    luma: [SValue; 2],
    /// The chroma plane's first element and the elements between its rows (`u32`s).
    chroma: [SValue; 2],
    /// The chroma plane's last column and row (`u32`s): `⌈w / 2⌉ - 1`, `⌈h / 2⌉ - 1`.
    last: [SValue; 2],
    /// The shift that takes an element's lane to its code: 6 of 10-bit codes, 0 of 8-bit ones.
    code_shift: SValue,
    /// The shift that counts a code as a 10-bit one: 0 of 10-bit codes, 2 of 8-bit ones.
    up: SValue,
    /// The codes are full range (a `bool`).
    full: SValue,
    /// The largest code `m = 2^b - 1`, and `⌊m / 2⌋`, of full range's luma (`u32`s).
    max: [SValue; 2],
    /// Red's, green's and blue's `(kb, kr)` (`i32`s) of the surface's range and matrix.
    k: [[SValue; 2]; 3],
}

/// The scalar and vector types a tap computes in.
struct T {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    vec4: TypeId,
    uvec2: TypeId,
}

impl T {
    fn new(fb: &mut FunctionBuilder<'_>) -> BResult<T> {
        let m = fb.module();
        let u32_ = m.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let i32_ = m.ty(SType::Int {
            width: 32,
            signed: true,
        })?;
        let f32_ = m.ty(SType::Float { width: 32 })?;
        let vec4 = m.ty(SType::Vector { elem: f32_, len: 4 })?;
        let uvec2 = m.ty(SType::Vector { elem: u32_, len: 2 })?;
        Ok(T {
            u32_,
            i32_,
            f32_,
            vec4,
            uvec2,
        })
    }

    fn k(&self, fb: &mut FunctionBuilder<'_>, ty: TypeId, bits: u32) -> BResult<SValue> {
        let c = fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(bits)),
        })?;
        fb.module().const_value(c)
    }

    fn u(&self, fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
        self.k(fb, self.u32_, v)
    }

    fn i(&self, fb: &mut FunctionBuilder<'_>, v: i32) -> BResult<SValue> {
        self.k(fb, self.i32_, v as u32)
    }

    fn f(&self, fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<SValue> {
        self.k(fb, self.f32_, v.to_bits())
    }
}

/// The words of slot `slot`'s push field (`bind::YCbCrInPlace::words`), each a `u32`, in the
/// open block.
struct Words {
    size: [SValue; 2],
    luma: [SValue; 2],
    chroma: [SValue; 2],
    conversion: SValue,
}

fn words(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, slot: SlotRef) -> BResult<Words> {
    use crate::bind::YCbCrInPlace as W;
    let field = cx.push_field(fb, PushField::YCbCrInPlace(slot))?;
    let low = fb.composite_extract(field, &[0])?;
    let high = fb.composite_extract(field, &[1])?;
    // Word `i` of the field: lane `i % 4` of its first `uvec4` or its second.
    let mut word = |i: usize| match i < 4 {
        true => fb.composite_extract(low, &[i as u32]),
        false => fb.composite_extract(high, &[(i - 4) as u32]),
    };
    Ok(Words {
        size: [word(W::WIDTH)?, word(W::HEIGHT)?],
        luma: [word(W::LUMA_AT)?, word(W::LUMA_ROW)?],
        chroma: [word(W::CHROMA_AT)?, word(W::CHROMA_ROW)?],
        conversion: word(W::CONVERSION)?,
    })
}

/// The width (`axis` 0) or the height (`axis` 1) of slot `slot`'s in-place texture, a `u32`
/// from its push field, in the open block: a size query's answer at its one level.
pub(crate) fn extent(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: SlotRef,
    axis: u8,
) -> BResult<SValue> {
    let [width, height] = words(cx, fb, slot)?.size;
    match axis {
        0 => Ok(width),
        1 => Ok(height),
        _ => Err(cx.fail("an in-place texture's extent past its two axes")),
    }
}

/// The constants of slot `slot`'s taps (its push field's words), and the texture's size (a
/// `uvec2`), in the open block.
pub(crate) fn prepare(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: SlotRef,
) -> BResult<(Planes, SValue)> {
    use crate::bind::{YCbCrMatrix as M, YCbCrRange as R};
    let t = T::new(fb)?;
    let w = words(cx, fb, slot)?;
    let size = fb.composite_construct(t.uvec2, &w.size)?;
    let one = t.u(fb, 1)?;
    // The chroma plane's last column and row, `⌈n / 2⌉ - 1`.
    let last_of = |fb: &mut FunctionBuilder<'_>, n: SValue| -> BResult<SValue> {
        let up = fb.binary(BinOp::IAdd, n, one)?;
        let half = fb.binary(BinOp::ShiftRightLogical, up, one)?;
        fb.binary(BinOp::ISub, half, one)
    };
    let [width, height] = w.size;
    let last = [last_of(fb, width)?, last_of(fb, height)?];
    // The conversion word: the codes' bits, full range, the matrix.
    let field_at = |fb: &mut FunctionBuilder<'_>, shift: u32, mask: u32| -> BResult<SValue> {
        let shift = t.u(fb, shift)?;
        let mask = t.u(fb, mask)?;
        let v = fb.binary(BinOp::ShiftRightLogical, w.conversion, shift)?;
        fb.binary(BinOp::BitwiseAnd, v, mask)
    };
    let bits = field_at(fb, 0, 0xff)?;
    let ten = t.u(fb, 10)?;
    let ten = fb.binary(BinOp::IEqual, bits, ten)?;
    let pick = |fb: &mut FunctionBuilder<'_>, [a, b]: [u32; 2]| -> BResult<SValue> {
        let (a, b) = (t.u(fb, a)?, t.u(fb, b)?);
        fb.select(ten, a, b)
    };
    let code_shift = pick(fb, [6, 0])?;
    let up = pick(fb, [0, 2])?;
    let max = [pick(fb, [1023, 255])?, pick(fb, [511, 127])?];
    let full = field_at(fb, 8, 1)?;
    let full = fb.binary(BinOp::IEqual, full, one)?;
    let matrix = field_at(fb, 9, 3)?;
    let bt709 = fb.binary(BinOp::IEqual, matrix, one)?;
    let two = t.u(fb, 2)?;
    let bt2020 = fb.binary(BinOp::IEqual, matrix, two)?;
    // A coefficient of each matrix (BT.601, BT.709, BT.2020), chosen by the matrix.
    let choose = |fb: &mut FunctionBuilder<'_>, [a, b, c]: [i32; 3]| -> BResult<SValue> {
        let (a, b, c) = (t.i(fb, a)?, t.i(fb, b)?, t.i(fb, c)?);
        let low = fb.select(bt709, b, a)?;
        fb.select(bt2020, c, low)
    };
    // Per channel, each matrix's `(kb, kr)` of `range`.
    let by_channel = |range| {
        let of = crate::bind::ycbcr_convert::coefficients;
        let [[a0, a1, a2], [b0, b1, b2], [c0, c1, c2]] =
            [M::Bt601, M::Bt709, M::Bt2020].map(|m| of(range, m));
        [[a0, b0, c0], [a1, b1, c1], [a2, b2, c2]]
    };
    let sides: [fn((i32, i32)) -> i32; 2] = [|(kb, _)| kb, |(_, kr)| kr];
    let zero = t.i(fb, 0)?;
    let mut k = [[zero; 2]; 3];
    for ((video, full_range), out) in by_channel(R::Video)
        .into_iter()
        .zip(by_channel(R::Full))
        .zip(k.iter_mut())
    {
        for (side, coefficient) in sides.into_iter().zip(out.iter_mut()) {
            let v = choose(fb, video.map(side))?;
            let f = choose(fb, full_range.map(side))?;
            *coefficient = fb.select(full, f, v)?;
        }
    }
    let planes = Planes {
        luma: w.luma,
        chroma: w.chroma,
        last,
        code_shift,
        up,
        full,
        max,
        k,
    };
    Ok((planes, size))
}

/// The element `at + row · y + x` of the planes' texel buffer `img` (`u32`s): a `uvec4` whose
/// first two lanes it holds.
fn element(
    fb: &mut FunctionBuilder<'_>,
    img: SValue,
    [at, row]: [SValue; 2],
    [x, y]: [SValue; 2],
) -> BResult<SValue> {
    let r = fb.binary(BinOp::IMul, row, y)?;
    let e = fb.binary(BinOp::IAdd, at, r)?;
    let e = fb.binary(BinOp::IAdd, e, x)?;
    fb.fetch(img, e, &ImageOperands::default())
}

/// The texel Metal reads at `at` (a `uvec2` inside the texture) of the in-place texture whose
/// planes' texel buffer is `img` and whose constants `p` holds (module docs): a `vec4` of its
/// converted red, green and blue, each a step of `1 / 2048`, and alpha 1. Every such step is a
/// `half` exactly, so a `half` result's rounding leaves it, whatever `_half` asks.
pub(crate) fn texel(
    fb: &mut FunctionBuilder<'_>,
    p: &Planes,
    img: SValue,
    at: SValue,
    _half: bool,
) -> BResult<SValue> {
    let t = T::new(fb)?;
    let x = fb.composite_extract(at, &[0])?;
    let y = fb.composite_extract(at, &[1])?;
    let (zero, one) = (t.u(fb, 0)?, t.u(fb, 1)?);
    // The luma code: lane `x & 1` of element `⌊x / 2⌋` of its row.
    let pair_x = fb.binary(BinOp::ShiftRightLogical, x, one)?;
    let v = element(fb, img, p.luma, [pair_x, y])?;
    let odd_x = fb.binary(BinOp::BitwiseAnd, x, one)?;
    let odd_x = fb.binary(BinOp::INotEqual, odd_x, zero)?;
    let l0 = fb.composite_extract(v, &[0])?;
    let l1 = fb.composite_extract(v, &[1])?;
    let luma = fb.select(odd_x, l1, l0)?;
    let luma = fb.binary(BinOp::ShiftRightLogical, luma, p.code_shift)?;
    // On each axis, the chroma texel's own column (or row) and its neighbour: the one after it
    // for an odd index, before it for an even one, clamped to the plane.
    let near = |fb: &mut FunctionBuilder<'_>, c: SValue, last: SValue| -> BResult<[SValue; 2]> {
        let own = fb.binary(BinOp::ShiftRightLogical, c, one)?;
        let odd = fb.binary(BinOp::BitwiseAnd, c, one)?;
        let odd = fb.binary(BinOp::INotEqual, odd, zero)?;
        let after = fb.binary(BinOp::IAdd, own, one)?;
        let after = fb.glsl(GlslOp::UMin, &[after, last])?;
        let before = fb.glsl(GlslOp::UMax, &[own, one])?;
        let before = fb.binary(BinOp::ISub, before, one)?;
        Ok([own, fb.select(odd, after, before)?])
    };
    let [cx0, cx1] = near(fb, x, p.last[0])?;
    let [cy0, cy1] = near(fb, y, p.last[1])?;
    let mut sums = [zero; 2];
    for (weight, site) in [
        (9, [cx0, cy0]),
        (3, [cx1, cy0]),
        (3, [cx0, cy1]),
        (1, [cx1, cy1]),
    ] {
        let v = element(fb, img, p.chroma, site)?;
        let w = t.u(fb, weight)?;
        for (lane, sum) in sums.iter_mut().enumerate() {
            let c = fb.composite_extract(v, &[lane as u32])?;
            let c = fb.binary(BinOp::ShiftRightLogical, c, p.code_shift)?;
            let c = fb.binary(BinOp::IMul, c, w)?;
            *sum = fb.binary(BinOp::IAdd, *sum, c)?;
        }
    }
    // Each chroma code `(S + 8) >> 4`, counted as a 10-bit code, less 512.
    let (eight, four, mid) = (t.u(fb, 8)?, t.u(fb, 4)?, t.i(fb, 512)?);
    let mut chroma = [mid; 2];
    for (c, sum) in chroma.iter_mut().zip(sums) {
        let s = fb.binary(BinOp::IAdd, sum, eight)?;
        let code = fb.binary(BinOp::ShiftRightLogical, s, four)?;
        let code = fb.binary(BinOp::ShiftLeftLogical, code, p.up)?;
        let code = fb.convert(ConvOp::Bitcast, code, t.i32_)?;
        *c = fb.binary(BinOp::ISub, code, mid)?;
    }
    // The luma term: video range `9576 ((Y << u) - 64)`, full `256 ⌊(32768 Y + ⌊m / 2⌋) / m⌋`.
    let video = fb.binary(BinOp::ShiftLeftLogical, luma, p.up)?;
    let video = fb.convert(ConvOp::Bitcast, video, t.i32_)?;
    let black = t.i(fb, 64)?;
    let video = fb.binary(BinOp::ISub, video, black)?;
    let scale = t.i(fb, 9576)?;
    let video = fb.binary(BinOp::IMul, video, scale)?;
    let wide = t.u(fb, 32768)?;
    let full = fb.binary(BinOp::IMul, luma, wide)?;
    let full = fb.binary(BinOp::IAdd, full, p.max[1])?;
    let full = fb.binary(BinOp::UDiv, full, p.max[0])?;
    let full = fb.convert(ConvOp::Bitcast, full, t.i32_)?;
    let step = t.i(fb, 256)?;
    let full = fb.binary(BinOp::IMul, full, step)?;
    let l = fb.select(p.full, full, video)?;
    // Each channel `clamp((L + kb Cb' + kr Cr' + 2048) >> 12, 0, 2048) / 2048`.
    let (round, twelve, low) = (t.i(fb, 2048)?, t.u(fb, 12)?, t.i(fb, 0)?);
    let grid = t.f(fb, 1.0 / 2048.0)?;
    let mut lanes = Vec::with_capacity(4);
    for [kb, kr] in p.k {
        let b = fb.binary(BinOp::IMul, kb, chroma[0])?;
        let r = fb.binary(BinOp::IMul, kr, chroma[1])?;
        let sum = fb.binary(BinOp::IAdd, l, b)?;
        let sum = fb.binary(BinOp::IAdd, sum, r)?;
        let sum = fb.binary(BinOp::IAdd, sum, round)?;
        let v = fb.binary(BinOp::ShiftRightArithmetic, sum, twelve)?;
        let v = fb.glsl(GlslOp::SClamp, &[v, low, round])?;
        let v = fb.convert(ConvOp::ConvertSToF, v, t.f32_)?;
        lanes.push(fb.binary(BinOp::FMul, v, grid)?);
    }
    lanes.push(t.f(fb, 1.0)?);
    fb.composite_construct(t.vec4, &lanes)
}

/// The class bits a sample's exact routine and border texels read: the converted 4:2:0 formats'
/// (`metal::texel_format::texel_format`), one for both widths (`one_class_for_both_widths`).
pub(crate) fn class() -> u32 {
    crate::metal::texel_format::texel_format(PixelFormat::YCBCR84202P).0
}

/// Runtime sampler word `word` with each plane axis's mode as a 4:2:0 texture samples it:
/// mirror-clamp-to-edge (1), repeat (2) and mirrored repeat (3) as clamp-to-edge (0).
fn mapped_word(fb: &mut FunctionBuilder<'_>, t: &T, word: SValue) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let mut w = word;
    let (one, three, seven) = (t.u(fb, 1)?, t.u(fb, 3)?, t.u(fb, 7)?);
    for shift in SamplerWord::ADDRESS_SHIFT.into_iter().take(2) {
        let s = t.u(fb, shift)?;
        let mode = fb.binary(BinOp::ShiftRightLogical, w, s)?;
        let mode = fb.binary(BinOp::BitwiseAnd, mode, seven)?;
        let above = fb.binary(BinOp::UGreaterThanEqual, mode, one)?;
        let within = fb.binary(BinOp::ULessThanEqual, mode, three)?;
        let wraps = fb.binary(BinOp::LogicalAnd, above, within)?;
        let keep = t.u(fb, !(7 << shift))?;
        let edge = fb.binary(BinOp::BitwiseAnd, w, keep)?;
        w = fb.select(wraps, edge, w)?;
    }
    Ok(w)
}

/// The texel of in-place sample `s` through the device's own sampler Y′CbCr conversion
/// (`LoweringInputs::ycbcr_planes_sampled`, the module docs' **The device's sampler**), in the open
/// block: the slot's combined image `sampled` sampled at level 0, a linear sample at Metal's
/// coordinate (its texel offset moved in texels), a nearest one at the centre of the luma texel
/// Metal reads (clamped to the edge, as a 4:2:0 texture addresses every mode but a border one).
fn device_sample(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sample,
    slot: SlotRef,
    (coord, sampled): (SValue, SValue),
) -> BResult<SValue> {
    let t = T::new(fb)?;
    let w = words(cx, fb, slot)?;
    let vec2 = fb.module().ty(SType::Vector {
        elem: t.f32_,
        len: 2,
    })?;
    let size = fb.composite_construct(t.uvec2, &w.size)?;
    let size = fb.convert(ConvOp::ConvertUToF, size, vec2)?;
    // Whether the sampler reads pixel coordinates and filters linearly: a constexpr sampler's
    // state, or a runtime sampler's word.
    let (pixel, linear) = match &s.state {
        State::Constant { linear, pixel, .. } => (Known::Fixed(*pixel), Known::Fixed(*linear)),
        State::Slot { sampler } => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
            let bit = t.u(fb, crate::metal::sampler::SamplerWord::PIXEL_COORDINATES)?;
            let set = fb.binary(BinOp::BitwiseAnd, word, bit)?;
            let pixel = fb.binary(BinOp::IEqual, set, bit)?;
            let tys = Tys::new(fb)?;
            let linear = super::extended::word_linear(fb, &tys, word)?;
            (Known::Run(pixel), Known::Run(linear))
        }
    };
    // The coordinate in texels, and normalized.
    let texels = fb.binary(BinOp::FMul, coord, size)?;
    let (mut texels, mut normal) = match pixel {
        Known::Fixed(true) => (coord, fb.binary(BinOp::FDiv, coord, size)?),
        Known::Fixed(false) => (texels, coord),
        Known::Run(pixel) => {
            let normal = fb.binary(BinOp::FDiv, coord, size)?;
            let b2 = bvec2(fb)?;
            let pixel2 = fb.composite_construct(b2, &[pixel, pixel])?;
            (
                fb.select(pixel2, coord, texels)?,
                fb.select(pixel2, normal, coord)?,
            )
        }
    };
    if let Some(offset) = s.offset.as_deref() {
        let lanes: Vec<SValue> = offset
            .iter()
            .map(|&k| t.f(fb, k as f32))
            .collect::<BResult<_>>()?;
        let k = fb.composite_construct(vec2, &lanes)?;
        texels = fb.binary(BinOp::FAdd, texels, k)?;
        let moved = fb.binary(BinOp::FDiv, k, size)?;
        normal = fb.binary(BinOp::FAdd, normal, moved)?;
    }
    // A nearest sample's luma texel, clamped to the edge, at its centre.
    let nearest = |fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        let floor = fb.glsl(GlslOp::Floor, &[texels])?;
        let zero = t.f(fb, 0.0)?;
        let zero = fb.composite_construct(vec2, &[zero, zero])?;
        let one = t.f(fb, 1.0)?;
        let one = fb.composite_construct(vec2, &[one, one])?;
        let last = fb.binary(BinOp::FSub, size, one)?;
        let texel = fb.glsl(GlslOp::FClamp, &[floor, zero, last])?;
        let half = t.f(fb, 0.5)?;
        let half = fb.composite_construct(vec2, &[half, half])?;
        let centre = fb.binary(BinOp::FAdd, texel, half)?;
        fb.binary(BinOp::FDiv, centre, size)
    };
    let at = match linear {
        Known::Fixed(true) => normal,
        Known::Fixed(false) => nearest(fb)?,
        Known::Run(linear) => {
            let centre = nearest(fb)?;
            let b2 = bvec2(fb)?;
            let linear2 = fb.composite_construct(b2, &[linear, linear])?;
            fb.select(linear2, normal, centre)?
        }
    };
    let lod = t.f(fb, 0.0)?;
    let ops = ImageOperands {
        lod: Some(lod),
        ..ImageOperands::default()
    };
    let x = fb.sample(sampled, at, None, false, &ops)?;
    // A `half` result is held widened, rounded to a `half` as every sampled texel's is
    // (`texture::held_texel`).
    match s.half {
        false => Ok(x),
        true => {
            let half = crate::lower::arith::Form {
                held: crate::lower::arith::Held::Float(crate::lower::arith::FloatFmt::Half),
                lanes: 4,
            };
            crate::lower::arith::narrow::round_texel(fb, half, x)
        }
    }
}

/// A value fixed as the module is made, or known only as it runs.
#[derive(Clone, Copy)]
enum Known {
    Fixed(bool),
    Run(SValue),
}

/// The `bvec2` type.
fn bvec2(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    let b = fb.module().ty(SType::Bool)?;
    fb.module().ty(SType::Vector { elem: b, len: 2 })
}

/// The texel of in-place sample `s` in the open block, with its status word 0 (module docs).
pub(crate) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sample,
) -> BResult<SValue> {
    let coord = cx.operand(fb, s.coord)?;
    let (decl, img) = crate::lower::texture::slot_image(cx, fb, s.texture)?;
    let Some(slot) = cx
        .plan
        .families
        .texture
        .slots
        .get(&s.texture)
        .map(|p| p.slot)
    else {
        return Err(cx.fail(format!("texture slot {} is not planned", s.texture)));
    };
    if decl.planes_sampler.is_some() {
        // The device's conversion takes no gather (`device_takes` keeps such a slot off it).
        if s.gather.is_some() {
            return Err(cx.fail("an in-place gather under the device's sampler"));
        }
        let x = device_sample(cx, fb, s, slot, (coord, img))?;
        return crate::lower::texture::with_word(cx, fb, &s.result, x, None);
    }
    let (planes, size) = prepare(cx, fb, slot)?;
    let t = Tys::new(fb)?;
    let tt = T::new(fb)?;
    let parent = match s.offset_base {
        true => match cx
            .decl
            .families
            .texture
            .offset_base
            .get(&s.texture)
            .copied()
        {
            Some(base) => Some(base),
            None => {
                return Err(cx.fail(format!(
                    "texture slot {} has no OffsetBaseAxes specialization",
                    s.texture
                )))
            }
        },
        false => None,
    };
    let flags = tt.u(fb, class())?;
    let tex = Texture {
        img,
        image: CombinedImage {
            plane: 2,
            arrayed: false,
            texel: decl.texel,
        },
        decode: Decode::InPlace(planes),
        flags,
        swizzle: None,
        parent,
        size: Some(size),
    };
    let at = super::extended::At {
        coord,
        layer: None,
        offset: s.offset.as_deref(),
        passed: None,
        level: None,
        half: s.half,
    };
    if let Some(component) = s.gather {
        let addressing = match &s.state {
            State::Constant {
                address,
                border,
                pixel,
                ..
            } => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            State::Slot { sampler } => {
                let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
                let word = mapped_word(fb, &tt, word)?;
                Addressing::Word { word, axes: 2 }
            }
        };
        let x = super::extended::gathered(cx, fb, &t, &at, &tex, addressing, component)?;
        return crate::lower::texture::with_word(cx, fb, &s.result, x, None);
    }
    let x = match &s.state {
        State::Constant { .. } => super::extended::filters(cx, fb, &t, &at, &tex, &s.state)?,
        State::Slot { sampler } => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
            let word = mapped_word(fb, &tt, word)?;
            let addressing = Addressing::Word { word, axes: 2 };
            let linear = super::extended::word_linear(fb, &t, word)?;
            let (t, tex, at) = (&t, &tex, &at);
            let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = super::extended::filtered(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let nearest: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = super::extended::nearest(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, linear, filtered, Some(nearest))?;
            match merged[..] {
                [x] => x,
                _ => return Err(cx.fail("an in-place sample's branch without its texel")),
            }
        }
    };
    crate::lower::texture::with_word(cx, fb, &s.result, x, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both widths' converted 4:2:0 formats filter on one class, with Metal's exact sum and its
    /// border texels, so one variant serves both.
    #[test]
    fn one_class_for_both_widths() {
        use crate::metal::bicubic::BicubicTexels;
        let ten = crate::metal::texel_format::texel_format(PixelFormat::YCBCR104202P).0;
        assert_eq!(class(), ten);
        assert_ne!(class() & BicubicTexels::EXACT, 0);
        assert_ne!(class() >> BicubicTexels::BORDER & 0xfff, 0);
    }
}
