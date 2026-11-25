//! Which texture slots a fragment module reads only at the fragment's own pixel.
//!
//! A runtime that serves a draw's read of a texture its own render pass renders through a copy
//! (the runtime's `texture/attached_reads.rs`) needs to copy only the texels the draw can read.
//! A fragment reads at its own pixel when every texel access of the slot is an `OpImageFetch` of
//! a single-sampled, non-arrayed 2D image at level 0, with no offset, whose coordinate is, lane
//! for lane, Metal's saturating `uint(x)` of `FragCoord`'s x and y, or that clamped below the
//! image's level-0 size less one (the shader's containment, `lower::texture`'s `clamp_below`,
//! where the device's robustness does not return Metal's out-of-bounds texel; its level clamp
//! `UMin(0, levels - 1)` is level 0).
//!
//! - **The conversion** is the one `lower::arith::math`'s `ToInt` emits for `uint(x)`:
//!   `ConvertFToU(NClamp(x, 0, largest float below 2^32))`, then, where they are emitted, the
//!   maximum selected where `x >= 2^32` and 0 where `x` is a NaN (`lower::arith::nan`'s
//!   `is_nan`: `OpIsNan`, or `(bits(x) & 0x7fffffff) > 0x7f800000`). Its `x` may be
//!   `FragCoord`'s lanes, or the pixel-center form `floor(FragCoord) + 0.5` lowering gives a
//!   `[[position]]` read under sample shading (`lower::stage_io`'s `pixel_center`).
//! - **Why it is the own pixel.** `FragCoord`'s x and y are the shaded location's, a sample's
//!   under sample shading, inside the pixel `[p, p + 1)` (Vulkan's Fragment Shading: "the x and
//!   y components ... hold the location of the sample or the pixel center"); a helper
//!   invocation's are those of a pixel of the 2×2 quad of a shaded one (the quad scope:
//!   "neighboring framebuffer locations"), at most one pixel beside it, so possibly left of or
//!   above the framebuffer. Each is finite, and below `2^24` on a framebuffer narrower and
//!   shorter than `2^24` pixels, which the fact presumes (Metal's largest texture is 16384
//!   pixels a side; a runtime checks its render area): there neither select takes its other
//!   arm, the clamp takes the coordinate to `max(x, 0)`, and the conversion truncates it to
//!   `max(p, 0)`. A bare `OpConvertFToU`, undefined at a helper's negative coordinate, leaves the
//!   slot out.
//! - **The texels read.** A fragment exists only inside its draw's scissors (`[x0, x1)` on an
//!   axis), and a helper beside one, so each read lies in `[max(x0 - 1, 0), x1 + 1)`, or under
//!   the shader's clamp below the level's width `w`,
//!   `[min(max(x0 - 1, 0), w - 1), min(x1 + 1, w))` (a read at or past `w` reads no texel under
//!   the device's robustness). A fragment covers one
//!   pixel where the pass has no fragment shading rate or density map.
//!
//! The test is structural and exact, never a name: a slot qualifies only where every use of each
//! of its bindings (the texture's own, a combined image sampler's or a separate image's of it)
//! is one of the operations above, a size, level or sample-count query, which reads no
//! texel, or a load, `OpSampledImage`, `OpImage` or copy that carries the image on to those. A
//! coordinate is traced through copies, bitcasts and conversions between integers of at least 16
//! bits, extracts, shuffles and constructs, and the steps above. Anything else on the way (other
//! arithmetic, a select on another condition, a clamp or limit below `2^24`, a phi, a parameter,
//! a load of another variable, a constant lane) leaves the slot out, as does any other use of its
//! image: a sample, a gather, a storage read or write, a texel pointer, a call, a store, a phi or
//! select of the image.

use std::collections::{BTreeSet, HashMap};

use crate::budget::Meter;
use crate::error::Error;
use crate::pipeline::sym::ResourceKind;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConstKind, Constant, ConvOp, Dim, ExecModel, GlobalKind, GlslOp, ImageOperands,
    ImageQuery, Inst, InstNode, IoDir, Terminator, Type, TypedModule, UnOp, ValueOrigin,
};

/// No slot: a value that holds no slot's image.
const NONE: u32 = u32::MAX;

/// The deepest coordinate trace followed; a deeper one leaves its slot out.
const MOST_STEPS: u32 = 32;

/// `2^24`: a framebuffer coordinate lies below it on a framebuffer narrower and shorter than
/// `2^24` pixels, which the fact presumes (the module docs).
const FRAMEBUFFER_BOUND: f32 = 16_777_216.0;

#[cfg(test)]
mod tests;

/// One lane of a traced coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    /// A float in `[p, p + 1)`, `p` the coordinate on axis 0 (x) or 1 (y) of the pixel a
    /// fragment shades or a helper invocation stands for, finite, at least -1 and below
    /// [`FRAMEBUFFER_BOUND`].
    Pixel(u8),
    /// That pixel's coordinate as a float: `floor` of a [`Lane::Pixel`].
    Floor(u8),
    /// That pixel's coordinate as an integer, `max(p, 0)`.
    Index(u8),
    /// That integer clamped below the bound view's first level's size on its axis:
    /// `UMin(index, size - 1)`.
    Clamped(u8),
    /// The constant `0.5`.
    Half,
    /// Anything else.
    Other,
}

/// The texture slots of fragment module `m` whose every texel read is at the fragment's own pixel
/// (the module docs), sorted; empty for a module of any other stage.
///
/// It reads each instruction at most three times, whatever the order of the blocks. One pass lists
/// the instructions that carry an image on (a load, `OpSampledImage`, `OpImage` or copy whose
/// result is an image, a sampled image or a pointer to one), and a worklist over that list from
/// the slots' variables finds every value that holds a slot's image. A second pass judges each
/// use of them, stopping once every slot is out. At the first fetch it traces, a third pass
/// places each instruction's result in its block, so the trace finds a defining instruction at
/// once. Each value's slot is a table by value, so finding one is a step whatever the number of
/// images. The list, the table and the places are charged to `meter` while held; what else it
/// holds grows with the coordinates traced alone.
pub fn own_pixel_textures(m: &TypedModule, meter: &mut Meter) -> Result<Vec<u32>, Error> {
    if m.entry().model != ExecModel::Fragment {
        return Ok(Vec::new());
    }
    let mut frag_coord = BTreeSet::new();
    // The slots' variables, and each one's slot.
    let mut seeds: Vec<(Value, u32)> = Vec::new();
    for g in m.globals() {
        match &g.kind {
            GlobalKind::BuiltIn {
                builtin: crate::spv::BuiltIn::FragCoord,
                dir: IoDir::Input,
            } => {
                frag_coord.insert(g.value);
            }
            GlobalKind::Resource(slot) => {
                let texture = match slot.kind {
                    ResourceKind::Texture | ResourceKind::CombinedTextureSampler(_) => {
                        Some(slot.index)
                    }
                    ResourceKind::SeparateTexture(_) => slot.separate_texture_of().map(|(t, _)| t),
                    _ => None,
                };
                if let Some(texture) = texture {
                    seeds.push((g.value, texture));
                }
            }
            _ => {}
        }
    }
    let slots: BTreeSet<u32> = seeds.iter().map(|&(_, slot)| slot).collect();
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    // Each value that holds a slot's image, by value: its slot, or `NONE`.
    let table = 4 * m.values().len();
    meter.charge(table)?;
    let mut carried = vec![NONE; m.values().len()];
    for &(v, slot) in &seeds {
        if let Some(c) = carried.get_mut(v.index()) {
            *c = slot;
        }
    }
    // Then what carries one on: a load, `OpSampledImage`, `OpImage` or copy of it, whose result
    // holds an image, found from the instructions that carry one on, by the value each takes it
    // from, whatever the order of the blocks.
    let mut carriers: Vec<(Value, Value)> = Vec::new();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        if let (Some(r), Some(from)) = (n.result, carried_on(&n.inst)) {
            if holds_image(m, r)? {
                carriers.push((from, r));
            }
        }
    }
    let listed = std::mem::size_of_val(carriers.as_slice());
    meter.charge(listed)?;
    carriers.sort_unstable();
    let mut work = seeds;
    while let Some((v, slot)) = work.pop() {
        let start = carriers.partition_point(|&(from, _)| from < v);
        let from_v = carriers
            .iter()
            .skip(start)
            .take_while(|&&(from, _)| from == v);
        for &(_, r) in from_v {
            if let Some(c) = carried.get_mut(r.index()).filter(|c| **c == NONE) {
                *c = slot;
                work.push((r, slot));
            }
        }
    }
    drop(carriers);
    meter.release(listed);
    let mut out: BTreeSet<u32> = BTreeSet::new();
    let mut memo = HashMap::new();
    let mut used = Vec::new();
    let mut places: Option<Vec<u32>> = None;
    let slot_of = |v: Value| carried.get(v.index()).copied().filter(|&slot| slot != NONE);
    'blocks: for b in m.blocks() {
        // A phi or a return of the image leaves its slot out.
        let merged = b
            .phis
            .iter()
            .flat_map(|phi| phi.incoming.iter().map(|(v, _)| *v));
        let returned = match &b.term {
            Terminator::ReturnValue(v) => Some(*v),
            _ => None,
        };
        out.extend(merged.chain(returned).filter_map(slot_of));
        for n in &b.insts {
            n.inst.for_each_value(|v| {
                if let Some(slot) = slot_of(v) {
                    used.push((v, slot));
                }
            });
            for (v, slot) in used.drain(..) {
                let own = match use_of(n, v) {
                    _ if out.contains(&slot) => true,
                    Use::NoTexel => true,
                    Use::Elsewhere => false,
                    Use::Fetch { coord, ops } => {
                        if places.is_none() {
                            places = Some(placed(m, meter)?);
                        }
                        let trace = Trace {
                            m,
                            frag_coord: &frag_coord,
                            carried: &carried,
                            slot,
                            places: places.as_deref().unwrap_or(&[]),
                        };
                        trace.fetch_at_own_pixel(v, coord, ops, &mut memo)?
                    }
                };
                if !own {
                    out.insert(slot);
                }
            }
            if out.len() == slots.len() {
                break 'blocks;
            }
        }
    }
    if places.is_some() {
        meter.release(4 * m.values().len());
    }
    meter.release(table);
    Ok(slots.difference(&out).copied().collect())
}

/// Where each instruction's result lies in its block's instructions, by value (`u32::MAX` for a
/// value no instruction defines), charged to `meter`.
fn placed(m: &TypedModule, meter: &mut Meter) -> Result<Vec<u32>, Error> {
    meter.charge(4 * m.values().len())?;
    let mut places = vec![u32::MAX; m.values().len()];
    for b in m.blocks() {
        for (i, n) in b.insts.iter().enumerate() {
            let (Some(r), Ok(i)) = (n.result, u32::try_from(i)) else {
                continue;
            };
            if let Some(place) = places.get_mut(r.index()) {
                *place = i;
            }
        }
    }
    Ok(places)
}

/// How an instruction uses a value that holds a slot's image.
pub(super) enum Use<'n> {
    /// It carries the image on, or queries its size, levels or samples: it reads no texel.
    NoTexel,
    /// `OpImageFetch` of it at `coord` with `ops`.
    Fetch {
        coord: Value,
        ops: &'n ImageOperands,
    },
    /// A sample, gather, storage access, texel pointer, call, store, select, composite, or the
    /// image anywhere else.
    Elsewhere,
}

/// How `n` uses `v`, which holds a slot's image.
pub(super) fn use_of(n: &InstNode, v: Value) -> Use<'_> {
    match &n.inst {
        // What carries the image on: its result is the image, or one sampled.
        inst if carried_on(inst) == Some(v) && n.result.is_some() => Use::NoTexel,
        // A query of its size, levels or samples reads no texel.
        Inst::ImageQuery { image, .. } if *image == v => Use::NoTexel,
        Inst::ImageFetch {
            image,
            coord,
            ops,
            sparse: false,
        } if *image == v && *coord != v => Use::Fetch { coord: *coord, ops },
        _ => Use::Elsewhere,
    }
}

/// The value an instruction that carries an image on takes it from: a load, `OpSampledImage`,
/// `OpImage`, or a copy; `None` for any other instruction.
pub(super) fn carried_on(inst: &Inst) -> Option<Value> {
    match inst {
        Inst::Load { ptr: from, .. }
        | Inst::SampledImage { image: from, .. }
        | Inst::Image(from)
        | Inst::Unary {
            op: UnOp::CopyObject,
            a: from,
        } => Some(*from),
        _ => None,
    }
}

/// Whether `v` is an image, a sampled image, or a pointer to one.
pub(super) fn holds_image(m: &TypedModule, v: Value) -> BResult<bool> {
    let ty = match m.types().get(m.value(v)?.ty)? {
        Type::Pointer { pointee, .. } => m.types().get(*pointee)?,
        ty => ty,
    };
    Ok(matches!(ty, Type::Image(_) | Type::SampledImage { .. }))
}

/// The traced coordinates' lanes, by slot and value.
type Memo = HashMap<(u32, Value), Vec<Lane>>;

/// What a coordinate trace of one slot reads of the module.
struct Trace<'m> {
    m: &'m TypedModule,
    /// The `FragCoord` input variables.
    frag_coord: &'m BTreeSet<Value>,
    /// Each value's slot, by value, where it holds a slot's image ([`NONE`] elsewhere).
    carried: &'m [u32],
    /// The slot traced.
    slot: u32,
    /// Where each instruction's result lies in its block ([`placed`]).
    places: &'m [u32],
}

impl<'m> Trace<'m> {
    /// The instruction defining `v`, found in its block at its place; `None` for any other value.
    fn def(&self, v: Value) -> Option<&'m Inst> {
        let ValueOrigin::Inst(b) = self.m.value(v).ok()?.origin else {
            return None;
        };
        let at = *self.places.get(v.index())?;
        let n = self.m.blocks().get(b.index())?.insts.get(at as usize)?;
        (n.result == Some(v)).then_some(&n.inst)
    }

    /// Whether `OpImageFetch` of `image` at `coord` with `ops` reads the fragment's own pixel of
    /// level 0 of a single-sampled, non-arrayed 2D image, with no offset.
    fn fetch_at_own_pixel(
        &self,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
        memo: &mut Memo,
    ) -> BResult<bool> {
        let m = self.m;
        let flat_2d = matches!(
            m.types().get(m.value(image)?.ty)?,
            Type::Image(t) if t.dim == Dim::D2 && !t.arrayed && !t.multisampled
        );
        let unadorned = ops.bias.is_none()
            && ops.grad.is_none()
            && ops.offset.is_none()
            && ops.sample.is_none()
            && ops.min_lod.is_none();
        let level_zero = match ops.lod {
            None => true,
            Some(lod) => self.level_zero(lod)?,
        };
        let no_offset = match ops.const_offset {
            None => true,
            Some(offset) => self.all_zero(offset)?,
        };
        if !(flat_2d && unadorned && level_zero && no_offset) {
            return Ok(false);
        }
        let lanes = self.lanes(coord, 0, memo)?;
        Ok(matches!(
            lanes.as_slice(),
            [
                Lane::Index(0) | Lane::Clamped(0),
                Lane::Index(1) | Lane::Clamped(1)
            ]
        ))
    }

    /// Whether level `v` is 0: a zero constant, or `UMin` of one with anything (the shader's
    /// clamp of a level below the image's level count, `lower::texture`'s `clamp_below`).
    fn level_zero(&self, v: Value) -> BResult<bool> {
        if self.all_zero(v)? {
            return Ok(true);
        }
        let Some(Inst::Glsl {
            op: GlslOp::UMin,
            args,
            ..
        }) = self.def(v)
        else {
            return Ok(false);
        };
        for arg in args.iter() {
            if self.all_zero(*arg)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether `v` holds the slot's image: loaded from one of its variables, through
    /// `OpSampledImage`, `OpImage` and copies.
    fn slot_image(&self, v: Value) -> bool {
        self.carried.get(v.index()) == Some(&self.slot)
    }

    /// The axes of `v`'s lanes where `v` is the size of the slot's image's level 0 less one
    /// (`ISub(OpImageQuerySizeLod(image, 0), 1)`, the level as [`Trace::level_zero`] takes it),
    /// lane for lane; `None` for anything else.
    fn size_less_one(&self, v: Value) -> BResult<Option<Vec<u8>>> {
        let Some(Inst::Binary {
            op: BinOp::ISub,
            a: size,
            b: one,
            ..
        }) = self.def(v)
        else {
            return Ok(None);
        };
        if !self.all_scalars(*one, 1)? {
            return Ok(None);
        }
        let Some(Inst::ImageQuery {
            query: ImageQuery::SizeLod(level),
            image,
        }) = self.def(*size)
        else {
            return Ok(None);
        };
        if !self.slot_image(*image) || !self.level_zero(*level)? {
            return Ok(None);
        }
        let flat_2d = matches!(
            self.m.types().get(self.m.value(*image)?.ty)?,
            Type::Image(t) if t.dim == Dim::D2 && !t.arrayed && !t.multisampled
        );
        Ok((flat_2d && self.width(*size)? == 2).then(|| vec![0, 1]))
    }

    /// Whether `v` is a plain 32-bit float constant whose every scalar is at least
    /// [`FRAMEBUFFER_BOUND`].
    fn at_least_bound(&self, v: Value) -> BResult<bool> {
        let m = self.m;
        if !self.is_f32(v)? {
            return Ok(false);
        }
        let ValueOrigin::Const(c) = m.value(v)?.origin else {
            return Ok(false);
        };
        let at_least = |bits: u64| {
            u32::try_from(bits).is_ok_and(|bits| f32::from_bits(bits) >= FRAMEBUFFER_BOUND)
        };
        Ok(match &m.consts().get(c)?.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => at_least(bits.0),
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts
                .iter()
                .all(|part| m.consts().scalar_bits(*part).is_some_and(|b| at_least(b.0))),
            _ => false,
        })
    }

    /// Whether condition `cond` is false wherever its operand is a pixel's coordinate: Metal's
    /// saturating conversion's NaN tests (`lower::arith::nan`'s `is_nan`: `OpIsNan`, or
    /// `(bits & 0x7fffffff) > 0x7f800000`) and its test `x >= limit` for a limit of at least
    /// [`FRAMEBUFFER_BOUND`], of a value whose every lane is a [`Lane::Pixel`].
    fn never_at_a_pixel(&self, cond: Value, depth: u32, memo: &mut Memo) -> BResult<bool> {
        let x = match self.def(cond) {
            Some(Inst::Unary { op: UnOp::IsNan, a }) => *a,
            Some(Inst::Binary {
                op: BinOp::FOrdGreaterThanEqual,
                a,
                b,
                ..
            }) if self.at_least_bound(*b)? => *a,
            Some(Inst::Binary {
                op: BinOp::UGreaterThan,
                a: magnitude,
                b: infinity,
                ..
            }) if self.all_scalars(*infinity, 0x7f80_0000)? => {
                let Some(Inst::Binary {
                    op: BinOp::BitwiseAnd,
                    a: bits,
                    b: mask,
                    ..
                }) = self.def(*magnitude)
                else {
                    return Ok(false);
                };
                if !self.all_scalars(*mask, 0x7fff_ffff)? {
                    return Ok(false);
                }
                let Some(Inst::Convert {
                    op: ConvOp::Bitcast,
                    a,
                    ..
                }) = self.def(*bits)
                else {
                    return Ok(false);
                };
                if !self.is_f32(*a)? {
                    return Ok(false);
                }
                *a
            }
            _ => return Ok(false),
        };
        let lanes = self.lanes(x, depth, memo)?;
        Ok(lanes.iter().all(|lane| matches!(lane, Lane::Pixel(_))))
    }

    /// Whether `v` is a plain 32-bit constant whose every scalar is `bits`.
    fn all_scalars(&self, v: Value, bits: u32) -> BResult<bool> {
        let m = self.m;
        let ValueOrigin::Const(c) = m.value(v)?.origin else {
            return Ok(false);
        };
        let want = u64::from(bits);
        Ok(match &m.consts().get(c)?.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => bits.0 == want,
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts
                .iter()
                .all(|part| m.consts().scalar_bits(*part).is_some_and(|b| b.0 == want)),
            _ => false,
        })
    }

    /// Whether `v` is a plain constant whose every scalar is zero.
    fn all_zero(&self, v: Value) -> BResult<bool> {
        let ValueOrigin::Const(c) = self.m.value(v)?.origin else {
            return Ok(false);
        };
        self.const_zero(c)
    }

    fn const_zero(&self, c: crate::spirv::ids::ConstId) -> BResult<bool> {
        Ok(match &self.m.consts().get(c)?.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => bits.0 == 0,
            ConstKind::Plain(Constant::Null { .. }) => true,
            ConstKind::Plain(Constant::Composite { parts, .. }) => {
                for part in parts.iter() {
                    if !self.const_zero(*part)? {
                        return Ok(false);
                    }
                }
                true
            }
            _ => false,
        })
    }

    /// The number of lanes of `v`'s type: a vector's length, 1 for anything else.
    fn width(&self, v: Value) -> BResult<usize> {
        let m = self.m;
        Ok(match m.types().get(m.value(v)?.ty)? {
            Type::Vector { len, .. } => usize::from(*len),
            _ => 1,
        })
    }

    /// Whether `v` is an integer scalar or vector of at least 16 bits a lane.
    fn wide_int(&self, v: Value) -> BResult<bool> {
        let m = self.m;
        let ty = match m.types().get(m.value(v)?.ty)? {
            Type::Vector { elem, .. } => *elem,
            _ => m.value(v)?.ty,
        };
        Ok(matches!(m.types().get(ty)?, Type::Int { width, .. } if *width >= 16))
    }

    /// The lanes of `v` (the module docs).
    fn lanes(&self, v: Value, depth: u32, memo: &mut Memo) -> BResult<Vec<Lane>> {
        if let Some(known) = memo.get(&(self.slot, v)) {
            return Ok(known.clone());
        }
        let width = self.width(v)?;
        let other = vec![Lane::Other; width];
        if depth > MOST_STEPS {
            return Ok(other);
        }
        let m = self.m;
        let lanes = match m.value(v)?.origin {
            ValueOrigin::Const(c) => match &m.consts().get(c)?.kind {
                ConstKind::Plain(Constant::Scalar { bits, .. })
                    if bits.0 == u64::from(0.5f32.to_bits()) && self.is_f32(v)? =>
                {
                    vec![Lane::Half]
                }
                ConstKind::Plain(Constant::Composite { parts, .. }) => {
                    let half = u64::from(0.5f32.to_bits());
                    let mut lanes = Vec::with_capacity(parts.len());
                    for part in parts.iter() {
                        let lane = match m.consts().scalar_bits(*part) {
                            Some(bits) if bits.0 == half && self.is_f32(v)? => Lane::Half,
                            _ => Lane::Other,
                        };
                        lanes.push(lane);
                    }
                    lanes
                }
                _ => other,
            },
            ValueOrigin::Inst(_) => match self.def(v) {
                Some(inst) => self.inst_lanes(inst, width, depth, memo)?,
                None => other,
            },
            _ => other,
        };
        let lanes = if lanes.len() == width {
            lanes
        } else {
            vec![Lane::Other; width]
        };
        memo.insert((self.slot, v), lanes.clone());
        Ok(lanes)
    }

    /// Whether `v`'s scalars are 32-bit floats.
    fn is_f32(&self, v: Value) -> BResult<bool> {
        let m = self.m;
        let ty = match m.types().get(m.value(v)?.ty)? {
            Type::Vector { elem, .. } => *elem,
            _ => m.value(v)?.ty,
        };
        Ok(matches!(m.types().get(ty)?, Type::Float { width: 32 }))
    }

    /// The lanes of the result of `inst`, of `width` lanes.
    fn inst_lanes(
        &self,
        inst: &Inst,
        width: usize,
        depth: u32,
        memo: &mut Memo,
    ) -> BResult<Vec<Lane>> {
        let other = vec![Lane::Other; width];
        let next = depth + 1;
        Ok(match inst {
            Inst::Load { ptr, .. } => match self.pixel_pointer(*ptr)? {
                Some(lanes) => lanes,
                None => other,
            },
            Inst::Unary {
                op: UnOp::CopyObject,
                a,
            } => self.lanes(*a, next, memo)?,
            // Metal's saturating conversion's core (`lower::arith::math`'s `ToInt`):
            // `ConvertFToU(NClamp(x, 0, largest float below 2^32))`, the pixel's index (0 for a
            // helper invocation's left of or above the framebuffer). A bare `OpConvertFToU` is
            // undefined at such a helper's negative coordinate: it leaves the slot out.
            Inst::Convert {
                op: ConvOp::ConvertFToU,
                a,
                ..
            } => match self.def(*a) {
                Some(Inst::Glsl {
                    op: GlslOp::NClamp,
                    args,
                    ..
                }) => match args.as_ref() {
                    [x, lo, hi] if self.all_zero(*lo)? && self.at_least_bound(*hi)? => {
                        let lanes = self.lanes(*x, next, memo)?;
                        lanes
                            .into_iter()
                            .map(|lane| match lane {
                                Lane::Pixel(axis) => Lane::Index(axis),
                                _ => Lane::Other,
                            })
                            .collect()
                    }
                    _ => other,
                },
                _ => other,
            },
            Inst::Convert {
                op: ConvOp::UConvert | ConvOp::SConvert | ConvOp::Bitcast,
                a,
                ..
            } => {
                // Integers to integers of at least 16 bits a lane, one lane for one: a pixel
                // index is below 2^15 (Vulkan's largest framebuffer is far smaller).
                if self.wide_int(*a)? && self.width(*a)? == width {
                    let lanes = self.lanes(*a, next, memo)?;
                    lanes
                        .into_iter()
                        .map(|lane| match lane {
                            Lane::Index(axis) => Lane::Index(axis),
                            Lane::Clamped(axis) => Lane::Clamped(axis),
                            _ => Lane::Other,
                        })
                        .collect()
                } else {
                    other
                }
            }
            Inst::CompositeExtract { composite, indices } => match indices.as_ref() {
                [i] => {
                    let lanes = self.lanes(*composite, next, memo)?;
                    vec![lanes.get(*i as usize).copied().unwrap_or(Lane::Other)]
                }
                _ => other,
            },
            Inst::VectorShuffle { a, b, components } => {
                let mut both = self.lanes(*a, next, memo)?;
                both.extend(self.lanes(*b, next, memo)?);
                components
                    .iter()
                    .map(|&c| both.get(c as usize).copied().unwrap_or(Lane::Other))
                    .collect()
            }
            Inst::CompositeConstruct { parts } => {
                let mut lanes = Vec::with_capacity(width);
                for part in parts.iter() {
                    lanes.extend(self.lanes(*part, next, memo)?);
                }
                lanes
            }
            Inst::Glsl {
                op: GlslOp::Floor,
                args,
                ..
            } => match args.as_ref() {
                [a] => {
                    let lanes = self.lanes(*a, next, memo)?;
                    lanes
                        .into_iter()
                        .map(|lane| match lane {
                            Lane::Pixel(axis) => Lane::Floor(axis),
                            _ => Lane::Other,
                        })
                        .collect()
                }
                _ => other,
            },
            // The saturating conversion's selects of 0 for a NaN and of the maximum at or above
            // `2^32`, neither of which a pixel's coordinate is.
            Inst::Select { cond, f, .. } => match self.never_at_a_pixel(*cond, next, memo)? {
                true => self.lanes(*f, next, memo)?,
                false => other,
            },
            // The shader's clamp of the coordinate below the image's size (`lower::texture`'s
            // `clamp_below`).
            Inst::Glsl {
                op: GlslOp::UMin,
                args,
                ..
            } => match args.as_ref() {
                [a, b] => match self.size_less_one(*b)? {
                    Some(axes) if axes.len() == width => {
                        let lanes = self.lanes(*a, next, memo)?;
                        lanes
                            .into_iter()
                            .zip(axes)
                            .map(|(lane, axis)| match lane {
                                Lane::Index(i) | Lane::Clamped(i) if i == axis => Lane::Clamped(i),
                                _ => Lane::Other,
                            })
                            .collect()
                    }
                    _ => other,
                },
                _ => other,
            },
            Inst::Binary {
                op: BinOp::FAdd,
                a,
                b,
                ..
            } => {
                let (a, b) = (self.lanes(*a, next, memo)?, self.lanes(*b, next, memo)?);
                a.into_iter()
                    .zip(b)
                    .map(|pair| match pair {
                        (Lane::Floor(axis), Lane::Half) | (Lane::Half, Lane::Floor(axis)) => {
                            Lane::Pixel(axis)
                        }
                        _ => Lane::Other,
                    })
                    .collect()
            }
            _ => other,
        })
    }

    /// The lanes a load through `ptr` gives where it points at `FragCoord` (its x and y, or one
    /// of them through an access chain of a constant index); `None` for any other pointer.
    fn pixel_pointer(&self, ptr: Value) -> BResult<Option<Vec<Lane>>> {
        if self.frag_coord.contains(&ptr) {
            return Ok(Some(vec![
                Lane::Pixel(0),
                Lane::Pixel(1),
                Lane::Other,
                Lane::Other,
            ]));
        }
        let Some(Inst::AccessChain { base, indices, .. }) = self.def(ptr) else {
            return Ok(None);
        };
        if !self.frag_coord.contains(base) {
            return Ok(None);
        }
        let [index] = indices.as_ref() else {
            return Ok(None);
        };
        let ValueOrigin::Const(c) = self.m.value(*index)?.origin else {
            return Ok(None);
        };
        Ok(Some(vec![match self.m.consts().scalar_bits(c) {
            Some(bits) if bits.0 == 0 => Lane::Pixel(0),
            Some(bits) if bits.0 == 1 => Lane::Pixel(1),
            _ => Lane::Other,
        }]))
    }
}
