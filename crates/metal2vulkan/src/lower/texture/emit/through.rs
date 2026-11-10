//! Reads through a sampler (Metal 4.1 `read(sampler s, coord[, offset][, lod])`, MSL 4.1 §6.13;
//! the semantics are the `texture` module docs'): the sampler's state in the function, the
//! coordinate and level it addresses, and the zero and border texels it hands back.

use crate::lower::emit::EmitCx;
use crate::lower::sampler::Pixel;
use crate::metal::sampler::SamplerWord;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, Dim, GlslOp, ImageQuery, ImageUsage, Type as SType, UnOp,
};

use super::super::{ImageDecl, ImageRef, Read, ReadSampler};
use super::{compose, coord_len, u32s};

/// A test the sampler's state decides: known as the function is emitted (a constexpr sampler's),
/// or a `bool` the function computes from a runtime sampler's word.
#[derive(Clone, Copy)]
enum Flag {
    Known(bool),
    Value(SValue),
}

impl Flag {
    /// `a` where the flag holds, else `b`.
    fn select(self, fb: &mut FunctionBuilder<'_>, a: SValue, b: SValue) -> BResult<SValue> {
        match self {
            Flag::Known(true) => Ok(a),
            Flag::Known(false) => Ok(b),
            Flag::Value(v) => fb.select(v, a, b),
        }
    }

    /// The flag and `x`: `None` where it never holds.
    fn and(self, fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<Option<SValue>> {
        match self {
            Flag::Known(true) => Ok(Some(x)),
            Flag::Known(false) => Ok(None),
            Flag::Value(v) => Ok(Some(fb.binary(BinOp::LogicalAnd, v, x)?)),
        }
    }
}

/// The state a read through a sampler addresses by, in the function ([`ReadSampler`]).
pub(super) struct State {
    /// Per plane axis (`s`, `t`, `r`): an index outside the level reads the border texel.
    border: [Flag; 3],
    /// The read reads at its level, not the base level: the sampler has a mip filter.
    mip: Flag,
    /// A texture buffer's index outside it reads the nearest texel, neither the first nor the
    /// border: a runtime `[[sampler(n)]]` whose `s` address mode is `clamp_to_edge` (module
    /// docs).
    buffer_clamps: Flag,
    /// The border color, `MTLSamplerBorderColor`'s value: a constant, or the word's field.
    color: Result<u32, SValue>,
}

/// The state of `s` in the function: a constexpr sampler's as constants, a runtime one's read from
/// its word (`SamplerWord::READ_STATE`'s fields), a slot's `SamplerState` specialization or an
/// argument buffer's sampler table entry.
pub(super) fn state(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: ReadSampler,
) -> BResult<State> {
    let (word, slot) = match s {
        ReadSampler::Constant { border, color, mip } => {
            return Ok(State {
                border: border.map(Flag::Known),
                mip: Flag::Known(mip),
                buffer_clamps: Flag::Known(false),
                color: Ok(color),
            })
        }
        ReadSampler::Slot(n) => (
            crate::lower::sampler::sampler_word(cx, fb, Pixel::Spec(n))?,
            true,
        ),
        ReadSampler::Heap(h) => (
            crate::lower::sampler::sampler_word(cx, fb, Pixel::Heap(h))?,
            false,
        ),
    };
    let field = |fb: &mut FunctionBuilder<'_>, shift: u32, mask: u32| -> BResult<SValue> {
        let shift = u32s(fb, 1, shift)?;
        let mask = u32s(fb, 1, mask)?;
        let v = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
        fb.binary(BinOp::BitwiseAnd, v, mask)
    };
    // `ClampToZero` (4) and `ClampToBorderColor` (5) read the border; `ClampToEdge` (0),
    // `MirrorClampToEdge` (1), `Repeat` (2) and `MirrorRepeat` (3) the edge.
    let zero = crate::metal::SamplerAddressMode::ClampToZero.as_raw();
    let mut modes = Vec::with_capacity(3);
    let mut border = [Flag::Known(false); 3];
    for (a, shift) in SamplerWord::ADDRESS_SHIFT.into_iter().enumerate() {
        let mode = field(fb, shift, 0b111)?;
        let first = u32s(fb, 1, zero)?;
        border[a] = Flag::Value(fb.binary(BinOp::UGreaterThanEqual, mode, first)?);
        modes.push(mode);
    }
    let mip = field(fb, SamplerWord::MIP_FILTER_SHIFT, 0b11)?;
    let none = u32s(fb, 1, 0)?;
    let mip = Flag::Value(fb.binary(BinOp::INotEqual, mip, none)?);
    let buffer_clamps = match slot {
        true => {
            let edge = u32s(
                fb,
                1,
                crate::metal::SamplerAddressMode::ClampToEdge.as_raw(),
            )?;
            Flag::Value(fb.binary(BinOp::IEqual, modes[0], edge)?)
        }
        false => Flag::Known(false),
    };
    let color = Err(field(fb, SamplerWord::BORDER_SHIFT, 0b11)?);
    Ok(State {
        border,
        mip,
        buffer_clamps,
        color,
    })
}

/// What a read through a sampler accesses, and what it hands back in place of the texel.
pub(super) struct Addressed {
    /// The coordinate the access reads: the plane addressed into the level, then the layer as
    /// the read's.
    pub coord: SValue,
    /// The image's size at the level the access reads, its layers included.
    pub size: SValue,
    /// The level the access reads: the read's where the sampler has a mip filter, else the base
    /// level, and the base level where the read's level is not the image's.
    pub level: Option<SValue>,
    /// The read's level and layer are the image's; elsewhere it hands back zero in every lane.
    /// `None`: always.
    pub inside: Option<SValue>,
    /// The plane coordinate is outside the level on an axis that reads the border texel, which
    /// the read hands back. `None`: never.
    pub border: Option<SValue>,
}

/// The addressing of read `r` of `img` (`decl`) at level `level` (the read's own, where it has
/// one) through a sampler of state `s`. Its level, where the image's, is its own under a mip
/// filter, else the base level. Its signed plane coordinate plus its offset is tested lane-wise
/// against the image's size at that level and clamped into `[0, size - 1]`; a lane outside on a
/// border axis reads the border texel, and a texture buffer's index outside reads its first texel,
/// or the border texel where the `t` axis is a border axis, unless the sampler clamps it
/// ([`State::buffer_clamps`]). The layer is composed unclamped
/// ([`compose`]), and tested against the image's layers.
pub(super) fn address(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (decl, img): (&ImageDecl, SValue),
    r: &Read,
    level: Option<SValue>,
    s: &State,
) -> BResult<Addressed> {
    let Some(t) = r.through else {
        return Err(cx.fail("an addressed coordinate of a read without its sampler"));
    };
    let at = &r.at;
    let sampled = decl.usage == ImageUsage::Sampled;
    // A level that is the constant 0 is every image's; any other is tested against the levels.
    let constant = match level {
        Some(l) => fb.const_scalar(l)?,
        None => Some(0),
    };
    let (level, level_ok) = match level {
        Some(l) if sampled && constant != Some(0) => {
            let levels = fb.query(ImageQuery::Levels, img)?;
            let ok = fb.binary(BinOp::ULessThan, l, levels)?;
            let base = u32s(fb, 1, 0)?;
            let read = s.mip.select(fb, l, base)?;
            (Some(fb.select(ok, read, base)?), Some(ok))
        }
        l => (l, None),
    };
    let size = match level {
        Some(l) if sampled => fb.query(ImageQuery::SizeLod(l), img)?,
        _ => fb.query(ImageQuery::Size, img)?,
    };
    let mut plane = cx.operand(fb, at.plane)?;
    if let Some(o) = t.offset {
        let o = cx.operand(fb, o)?;
        plane = fb.binary(BinOp::IAdd, plane, o)?;
    }
    let n = coord_len(decl);
    let lanes = at.lanes;
    let extent = if lanes == n {
        size
    } else if lanes == 1 {
        fb.composite_extract(size, &[0])?
    } else {
        let first: Vec<u32> = (0..u32::from(lanes)).collect();
        fb.vector_shuffle(size, size, &first)?
    };
    let one = u32s(fb, lanes, 1)?;
    let top = fb.binary(BinOp::ISub, extent, one)?;
    let below = fb.glsl(GlslOp::SMin, &[plane, top])?;
    let zero = u32s(fb, lanes, 0)?;
    let clamped = fb.glsl(GlslOp::SMax, &[below, zero])?;
    let within = fb.binary(BinOp::ULessThan, plane, extent)?;
    // A lane outside on a border axis. A texture buffer's index is addressed as the first texel
    // of a row outside its one row: the `t` axis's mode says whether that is the border texel, and
    // a sampler that clamps the index never reads it (module docs).
    let buffer = matches!(decl.dim, Dim::Buffer);
    let mut border = None;
    for a in 0..lanes {
        let inside = match lanes {
            1 => within,
            _ => fb.composite_extract(within, &[u32::from(a)])?,
        };
        let outside = fb.unary(UnOp::LogicalNot, inside)?;
        let axis = if buffer { 1 } else { usize::from(a) };
        if let Some(out) = s.border[axis].and(fb, outside)? {
            border = Some(match border {
                Some(b) => fb.binary(BinOp::LogicalOr, b, out)?,
                None => out,
            });
        }
    }
    if buffer {
        border = match (border, s.buffer_clamps) {
            (_, Flag::Known(true)) | (None, _) => None,
            (Some(b), Flag::Known(false)) => Some(b),
            (Some(b), Flag::Value(clamps)) => {
                let kept = fb.unary(UnOp::LogicalNot, clamps)?;
                Some(fb.binary(BinOp::LogicalAnd, b, kept)?)
            }
        };
    }
    // A texture buffer's index outside reads its first texel where the sampler does not clamp it.
    let plane = match decl.dim {
        Dim::Buffer => {
            let first = u32s(fb, 1, 0)?;
            let unclamped = fb.select(within, clamped, first)?;
            s.buffer_clamps.select(fb, clamped, unclamped)?
        }
        _ => clamped,
    };
    let coord = compose(cx, fb, decl, at, plane)?;
    let layer_ok = match at.layer {
        Some(l) => {
            let l = cx.operand(fb, l)?;
            let layers = fb.composite_extract(size, &[u32::from(lanes)])?;
            Some(fb.binary(BinOp::ULessThan, l, layers)?)
        }
        None => None,
    };
    let inside = match (level_ok, layer_ok) {
        (Some(a), Some(b)) => Some(fb.binary(BinOp::LogicalAnd, a, b)?),
        (a, b) => a.or(b),
    };
    Ok(Addressed {
        coord,
        size,
        level,
        inside,
        border,
    })
}

/// `texel`, read `r`'s texel in its result form, with what the read hands back in place of it
/// ([`Addressed`]): a slot's border texel where the plane coordinate is outside on a border axis
/// (a heap element's selects the out-of-bounds texel instead, before converting it), and zero in
/// every lane where its level or layer is not the image's.
pub(super) fn finish(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: &Read,
    (s, a): (&State, &Addressed),
    texel: SValue,
) -> BResult<SValue> {
    let texel = match (a.border, r.image) {
        (Some(out), ImageRef::Slot(index)) => {
            let border = border_texel(cx, fb, index, s, texel)?;
            fb.select(out, border, texel)?
        }
        _ => texel,
    };
    match a.inside {
        Some(inside) => {
            let ty = fb.value_ty(texel)?;
            let zero = fb.module().constant(Constant::Null { ty })?;
            let zero = fb.module().const_value(zero)?;
            fb.select(inside, texel, zero)
        }
        None => Ok(texel),
    }
}

/// Metal's border texel of texture slot `index` for the color of `s`, in the form of `like` (a
/// `float`, `half` or 32-bit integer scalar or vector of up to four lanes): lane `c` is 1 where bit
/// `3c + color` of the slot's `BorderTexels` specialization is set, else 0.
fn border_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: u32,
    s: &State,
    like: SValue,
) -> BResult<SValue> {
    let Some(bits) = cx.decl.families.texture.border_texels.get(&index).copied() else {
        return Err(cx.fail(format!(
            "texture slot {index} has no border texels constant"
        )));
    };
    let ty = fb.value_ty(like)?;
    let (elem, lanes) = match fb.module().types().get(ty)? {
        SType::Vector { elem, len } => (*elem, *len),
        _ => (ty, 1),
    };
    let one = match fb.module().types().get(elem)? {
        SType::Float { width: 32 } => u64::from(1.0f32.to_bits()),
        SType::Float { width: 16 } => 0x3c00,
        SType::Int { width: 32, .. } => 1,
        _ => return Err(cx.fail("a border texel of a texel neither float nor 32-bit integer")),
    };
    let one = fb.module().constant(Constant::Scalar {
        ty: elem,
        bits: ConstBits(one),
    })?;
    let one = fb.module().const_value(one)?;
    let nil = fb.module().constant(Constant::Null { ty: elem })?;
    let nil = fb.module().const_value(nil)?;
    let low = u32s(fb, 1, 1)?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for c in 0..u32::from(lanes) {
        let bit = match s.color {
            Ok(color) => u32s(fb, 1, 3 * c + color)?,
            Err(color) => {
                let first = u32s(fb, 1, 3 * c)?;
                fb.binary(BinOp::IAdd, first, color)?
            }
        };
        let v = fb.binary(BinOp::ShiftRightLogical, bits, bit)?;
        let v = fb.binary(BinOp::BitwiseAnd, v, low)?;
        let set = fb.binary(BinOp::IEqual, v, low)?;
        parts.push(fb.select(set, one, nil)?);
    }
    match lanes {
        1 => Ok(parts[0]),
        _ => fb.composite_construct(ty, &parts),
    }
}
