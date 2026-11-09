//! Which levels of its bound view each texture slot of a module reads
//! (`pipeline::LevelReads`).
//!
//! A runtime that serves a draw's read of a texture its own render pass renders at one level
//! (the runtime's `texture/attached_reads.rs`) can bind the texture's own image through a view of
//! fewer levels where the draw reads none of the attached ones, in place of a copy. That needs
//! the levels each read reaches, counted from the view's first level, which this decides from the
//! module's instructions alone:
//!
//! - **A sample** (`OpImageSample*`, any of its forms) reads the levels its level of detail
//!   selects, which its sampler clamps into `[minLod, maxLod]` (Vulkan's LOD Operation), so the
//!   fact holds it by sampler: the slot's binding's own (a combined image sampler's, or a
//!   separate image's pairing), and whether its level of detail is a constant `f32` the
//!   instruction names (explicit, without gradients) or the module's to compute (implicit,
//!   biased, from gradients, or an explicit one that is no constant). A sample with a minimum
//!   level of detail operand may select any level ("undefined" where it exceeds `maxLod`).
//! - **A gather** reads the view's first level ("4 texels in the base level of the image view",
//!   Texel Gathering).
//! - **A fetch or a size query** reads the level it names: a constant, or the lowering's
//!   containment of one, `UMin(c, levels - 1)` of the slot's own level count (`lower::texture`'s
//!   `clamp_below`). A level-count query whose every use is that containment, or the in-range
//!   test `c < levels` of a constant `c`, answers alike through any view of more than `c` levels.
//!   A fetch without a level reads level 0.
//! - **Anything else** reaches any level: a fetch or size query at a level the module computes,
//!   a level-count query used otherwise, a level-of-detail query (which answers by the view's
//!   level count), a storage access or texel pointer, and any use of the image the walk does not
//!   follow (a call, a store, a phi or select of it, the entry's returned value).
//!
//! What carries a slot's image on is followed as `own_pixel` follows it: a load, `OpSampledImage`,
//! `OpImage` or copy whose result is an image, a sampled image or a pointer to one. A slot's
//! binding of each kind is a seed: the texture's own (no sampler: a sample through it reaches any
//! level), a combined image sampler's, or a separate image's, whose sampler is its pairing's.

use std::collections::{BTreeMap, HashMap};

use super::own_pixel::{carried_on, holds_image};
use crate::budget::Meter;
use crate::error::Error;
use crate::pipeline::interface::{LevelReads, SampledLevels};
use crate::pipeline::sym::{ResourceKind, SamplerRef};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConstKind, Constant, ExecModel, GlobalKind, GlslOp, ImageQuery, Inst, Terminator, Type,
    TypedModule, ValueOrigin,
};

/// No seed: a value that holds no slot's image.
const NONE: u32 = u32::MAX;

#[cfg(test)]
mod tests;

/// What the walk found of one slot's reads.
#[derive(Default)]
struct Found {
    any: bool,
    named: Option<u32>,
    /// By sampler: the highest constant explicit level of detail, and whether one is computed.
    sampled: BTreeMap<SamplerRef, (Option<f32>, bool)>,
}

impl Found {
    fn name(&mut self, level: u32) {
        self.named = Some(self.named.map_or(level, |n| n.max(level)));
    }

    fn sample(&mut self, sampler: SamplerRef, lod: Option<f32>) {
        let (highest, computed) = self.sampled.entry(sampler).or_default();
        match lod {
            Some(c) => *highest = Some(highest.map_or(c, |h: f32| h.max(c))),
            None => *computed = true,
        }
    }
}

/// The levels each texture slot of module `m` reads (the module docs), one entry a slot, sorted by
/// slot; empty for a kernel.
///
/// One pass lists the instructions that carry an image on, and a worklist from the slots'
/// variables finds every value that holds one, as `own_pixel_textures` does. Where the module
/// queries a slot's level count, a pass finds the containment's `levels - 1` and its `UMin` of a
/// constant. A last pass judges every use. The table of values and the list are charged to
/// `meter` while held.
pub fn level_reads(m: &TypedModule, meter: &mut Meter) -> Result<Vec<LevelReads>, Error> {
    if m.entry().model == ExecModel::GLCompute {
        return Ok(Vec::new());
    }
    // The slots' variables, each with its seed: its slot and the sampler it samples through.
    let mut seeds: Vec<(u32, Option<SamplerRef>)> = Vec::new();
    let mut found: BTreeMap<u32, Found> = BTreeMap::new();
    let mut work: Vec<(Value, u32)> = Vec::new();
    for g in m.globals() {
        let GlobalKind::Resource(slot) = &g.kind else {
            continue;
        };
        let seed = match slot.kind {
            ResourceKind::Texture => Some((slot.index, None, false)),
            ResourceKind::CombinedTextureSampler(cs) => Some((slot.index, Some(cs.sampler), false)),
            ResourceKind::SeparateTexture(_) => slot
                .separate_texture_of()
                .map(|(texture, cs)| (texture, Some(cs.sampler), false)),
            // Planes a conversion reads as one texel: no level of them is a view's.
            ResourceKind::YCbCrPlanes(_) => Some((slot.index, None, true)),
            _ => None,
        };
        if let Some((texture, sampler, any)) = seed {
            found.entry(texture).or_default().any |= any;
            let index = u32::try_from(seeds.len()).unwrap_or(NONE);
            seeds.push((texture, sampler));
            work.push((g.value, index));
        }
    }
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let table = 4 * m.values().len();
    meter.charge(table)?;
    let mut carried = vec![NONE; m.values().len()];
    for &(v, seed) in &work {
        if let Some(c) = carried.get_mut(v.index()) {
            *c = seed;
        }
    }
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
    while let Some((v, seed)) = work.pop() {
        let start = carriers.partition_point(|&(from, _)| from < v);
        let from_v = carriers
            .iter()
            .skip(start)
            .take_while(|&&(from, _)| from == v);
        for &(_, r) in from_v {
            if let Some(c) = carried.get_mut(r.index()).filter(|c| **c == NONE) {
                *c = seed;
                work.push((r, seed));
            }
        }
    }
    drop(carriers);
    meter.release(listed);
    let seed_of = |v: Value| {
        let seed = *carried.get(v.index())?;
        seeds.get(seed as usize).copied()
    };
    let contained = Containment::of(m, &seed_of)?;
    for b in m.blocks() {
        // A phi or a return of the image reaches what the walk does not follow.
        let merged = b
            .phis
            .iter()
            .flat_map(|phi| phi.incoming.iter().map(|(v, _)| *v));
        let returned = match &b.term {
            Terminator::ReturnValue(v) => Some(*v),
            _ => None,
        };
        for (slot, _) in merged.chain(returned).filter_map(seed_of) {
            found.entry(slot).or_default().any = true;
        }
        for n in &b.insts {
            let mut uses: Vec<Value> = Vec::new();
            n.inst.for_each_value(|v| uses.push(v));
            for v in uses {
                if let Some((slot, sampler)) = seed_of(v) {
                    let f = found.entry(slot).or_default();
                    judge(m, &contained, (n.result, &n.inst), (v, slot, sampler), f)?;
                } else if let Some(&slot) = contained.levels.get(&v) {
                    // A level count used but by the containment of a constant level.
                    let f = found.entry(slot).or_default();
                    match level_count_use(m, &n.inst, v)? {
                        LevelCountUse::Top => {}
                        LevelCountUse::Compared(c) => f.name(c),
                        LevelCountUse::Other => f.any = true,
                    }
                } else if let Some(&slot) = contained.tops.get(&v) {
                    let f = found.entry(slot).or_default();
                    match contained.clamps_by_use(m, &n.inst, v)? {
                        Some(c) => f.name(c),
                        None => f.any = true,
                    }
                }
            }
        }
    }
    meter.release(table);
    Ok(found
        .into_iter()
        .map(|(slot, f)| LevelReads {
            slot,
            any: f.any,
            named: f.named,
            sampled: f
                .sampled
                .into_iter()
                .map(|(sampler, (highest, computed))| SampledLevels {
                    sampler,
                    highest_lod_bits: highest.map(f32::to_bits),
                    computed,
                })
                .collect(),
        })
        .collect())
}

/// The lowering's containment of a constant level by a slot's level count (the module docs):
/// each level-count query's result, each `levels - 1` of one, and each `UMin` of a constant and
/// such a `levels - 1`, by value, with the slot and (for the `UMin`) the constant.
struct Containment {
    levels: HashMap<Value, u32>,
    tops: HashMap<Value, u32>,
    clamps: HashMap<Value, (u32, u32)>,
}

impl Containment {
    fn of(
        m: &TypedModule,
        seed_of: &impl Fn(Value) -> Option<(u32, Option<SamplerRef>)>,
    ) -> BResult<Containment> {
        let mut c = Containment {
            levels: HashMap::new(),
            tops: HashMap::new(),
            clamps: HashMap::new(),
        };
        let insts = || m.blocks().iter().flat_map(|b| &b.insts);
        for n in insts() {
            if let (
                Some(r),
                Inst::ImageQuery {
                    query: ImageQuery::Levels,
                    image,
                },
            ) = (n.result, &n.inst)
            {
                if let Some((slot, _)) = seed_of(*image) {
                    c.levels.insert(r, slot);
                }
            }
        }
        if c.levels.is_empty() {
            return Ok(c);
        }
        for n in insts() {
            if let (
                Some(r),
                Inst::Binary {
                    op: BinOp::ISub,
                    a,
                    b,
                    ..
                },
            ) = (n.result, &n.inst)
            {
                if let (Some(&slot), Some(1)) = (c.levels.get(a), constant_u32(m, *b)?) {
                    c.tops.insert(r, slot);
                }
            }
        }
        for n in insts() {
            let (
                Some(r),
                Inst::Glsl {
                    op: GlslOp::UMin,
                    args,
                    ..
                },
            ) = (n.result, &n.inst)
            else {
                continue;
            };
            let [x, y] = args[..] else {
                continue;
            };
            for (top, level) in [(x, y), (y, x)] {
                if let (Some(&slot), Some(level)) = (c.tops.get(&top), constant_u32(m, level)?) {
                    c.clamps.insert(r, (slot, level));
                }
            }
        }
        Ok(c)
    }

    /// The constant a use of `top` (a `levels - 1`) clamps: `inst` is a `UMin` of it and a
    /// constant. `None` for any other use.
    fn clamps_by_use(&self, m: &TypedModule, inst: &Inst, top: Value) -> BResult<Option<u32>> {
        let Inst::Glsl {
            op: GlslOp::UMin,
            args,
            ..
        } = inst
        else {
            return Ok(None);
        };
        let [x, y] = args[..] else {
            return Ok(None);
        };
        match (x == top, y == top) {
            (true, false) => constant_u32(m, y),
            (false, true) => constant_u32(m, x),
            _ => Ok(None),
        }
    }

    /// The constant level value `v` is: a constant, or a containment of one ([`Containment`]) of
    /// slot `slot`'s level count.
    fn level(&self, m: &TypedModule, v: Value, slot: u32) -> BResult<Option<u32>> {
        if let Some(c) = constant_u32(m, v)? {
            return Ok(Some(c));
        }
        Ok(self
            .clamps
            .get(&v)
            .filter(|&&(of, _)| of == slot)
            .map(|&(_, c)| c))
    }
}

/// How an instruction uses a level-count query's result ([`level_count_use`]).
enum LevelCountUse {
    /// The containment's `levels - 1`, whose uses are judged apart.
    Top,
    /// The in-range test `c < levels` of the constant `c`.
    Compared(u32),
    /// Any other use.
    Other,
}

/// How `inst` uses `levels`, a level-count query's result: as the containment's `levels - 1`
/// (`ISub` of the constant 1), as the in-range test `c < levels` of a constant, or otherwise.
fn level_count_use(m: &TypedModule, inst: &Inst, levels: Value) -> BResult<LevelCountUse> {
    Ok(match inst {
        Inst::Binary {
            op: BinOp::ISub,
            a,
            b,
            ..
        } if *a == levels && *b != levels && constant_u32(m, *b)? == Some(1) => LevelCountUse::Top,
        Inst::Binary {
            op: BinOp::ULessThan,
            a,
            b,
            ..
        } if *b == levels && *a != levels => match constant_u32(m, *a)? {
            Some(c) => LevelCountUse::Compared(c),
            None => LevelCountUse::Other,
        },
        _ => LevelCountUse::Other,
    })
}

/// Judge `inst`'s use (its result `result`) of `v`, which holds slot `slot`'s image sampled
/// through `sampler` where it is sampled, into `f`.
fn judge(
    m: &TypedModule,
    contained: &Containment,
    (result, inst): (Option<Value>, &Inst),
    (v, slot, sampler): (Value, u32, Option<SamplerRef>),
    f: &mut Found,
) -> BResult<()> {
    match inst {
        // What carries the image on reads nothing.
        inst if carried_on(inst) == Some(v) && result.is_some() => {}
        Inst::ImageQuery { query, image } if *image == v => match query {
            ImageQuery::Size | ImageQuery::Samples => {}
            ImageQuery::SizeLod(l) => match contained.level(m, *l, slot)? {
                Some(c) => f.name(c),
                None => f.any = true,
            },
            // Its uses are judged as the level count's (`level_reads`).
            ImageQuery::Levels if result.is_some_and(|r| contained.levels.contains_key(&r)) => {}
            _ => f.any = true,
        },
        Inst::ImageFetch { image, ops, .. } if *image == v => match ops.lod {
            None => f.name(0),
            Some(l) => match contained.level(m, l, slot)? {
                Some(c) => f.name(c),
                None => f.any = true,
            },
        },
        Inst::ImageGather { sampled_image, .. } | Inst::ImageDrefGather { sampled_image, .. }
            if *sampled_image == v =>
        {
            f.name(0)
        }
        Inst::ImageSample {
            form,
            sampled_image,
            ops,
            ..
        } if *sampled_image == v => match sampler {
            Some(sampler) if ops.min_lod.is_none() => {
                let lod = match (form.implicit, ops.lod, ops.grad) {
                    (false, Some(l), None) => constant_f32(m, l)?,
                    _ => None,
                };
                f.sample(sampler, lod);
            }
            _ => f.any = true,
        },
        _ => f.any = true,
    }
    Ok(())
}

/// The value of `v` where it is a plain 32-bit integer constant.
fn constant_u32(m: &TypedModule, v: Value) -> BResult<Option<u32>> {
    scalar_bits(m, v, |ty| matches!(ty, Type::Int { width: 32, .. }))
}

/// The value of `v` where it is a plain `f32` constant.
fn constant_f32(m: &TypedModule, v: Value) -> BResult<Option<f32>> {
    Ok(scalar_bits(m, v, |ty| matches!(ty, Type::Float { width: 32 }))?.map(f32::from_bits))
}

/// The bits of `v` where it is a plain scalar constant of a type `of` accepts.
fn scalar_bits(m: &TypedModule, v: Value, of: impl Fn(&Type) -> bool) -> BResult<Option<u32>> {
    let def = m.value(v)?;
    let ValueOrigin::Const(c) = def.origin else {
        return Ok(None);
    };
    if !of(m.types().get(def.ty)?) {
        return Ok(None);
    }
    Ok(match &m.consts().get(c)?.kind {
        ConstKind::Plain(Constant::Scalar { bits, .. }) => u32::try_from(bits.0).ok(),
        _ => None,
    })
}
