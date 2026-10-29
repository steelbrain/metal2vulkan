//! Selection of the sampling operations: the sampler operand, the level of detail per stage,
//! operand legality per profile, and the texture's use.

use crate::air::abi::{AddressMode, Coordinates, Filter, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape, LodMode, TexOp, TexRole, Texture};
use crate::air::ir::{BinOp, Const, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::device::profile::CubeGradientRoute;
use crate::device::requirements::{VkFeature, VkLimit};
use crate::error::{
    Capability, Error, Feature, InternalError, InternalKind, LowerFamily, Unsupported,
};
use crate::lower::access::handle::HandleTarget;
use crate::lower::select::{any_of, NoteOn, NoteTest, SelectCx};
use crate::lower::texture::{
    texel_conv, AbsentArgument, ImageRef, NullArm, Prelude, Switch, TexelClass, TexelConv,
    TexelForm, TextureStrategy,
};
use crate::memory::handles::Candidate;
use crate::metal::sampler::{SamplerUse, SamplerWord};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{ImageAccess, SamplerRef, SlotClass, SlotRef, Stage};

use super::{
    BiasReach, BindingKey, CubeFootprint, HeapKey, HeapSamplerKey, Lod, MinLevel, Offset, Pixel,
    SampleOp, Sampling, Source, Unmapped,
};

fn refuse() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Sampler)).into()
}

/// A bicubic form no Metal reading establishes.
fn bicubic_refused() -> Error {
    Unsupported::not_implemented(Feature::SamplerBicubicFilter).into()
}

/// The operands of a sampling call, by role.
struct Operands<'a> {
    t: &'a Texture,
    args: &'a [Value],
}

impl Operands<'_> {
    fn get(&self, role: TexRole) -> Option<Value> {
        let i = self.t.layout().index(role)?;
        self.args.get(i).copied()
    }
}

/// The value id of an operand (constants included: emission materializes them).
fn id(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("a sampling operand out of the value range"))
}

pub(super) fn konst<'a>(cx: &'a SelectCx<'_>, v: Value) -> Option<&'a Const> {
    let Value::Const(c) = v else { return None };
    cx.entry.module.consts().get(c).map(|k| &k.value)
}

/// How nonzero fragment bias `v`, sampled through `sampler` on a texture of `plane` coordinate
/// components, reaches the level. Through a sampler without a bias of its own, the `Bias` operand
/// as given, the device's sampler's reading of it. Else (a constexpr sampler with a bias of its
/// own) by Metal's `B` of the bias: through an anisotropic sampler of a 2D plane, the axis
/// gradients; otherwise a constant `B` decides at compile time, by its `f32` sum with the
/// sampler's own bias as the device forms it, and a dynamic bias per invocation. A NaN constant
/// stays on the `Bias` operand, as before.
fn bias_reach(
    cx: &SelectCx<'_>,
    v: Value,
    sampler: SamplerArg,
    plane: u8,
) -> Result<BiasReach, Error> {
    if bias_to_metal(sampler)? {
        return Ok(BiasReach::Metal);
    }
    let max_bias_bits = cx.profile.fields().limits.max_sampler_lod_bias_bits;
    let own = own_bias(sampler)?;
    // A sampler with a bias of its own is a constexpr one, whose state states its anisotropy.
    let SamplerArg::Static(SamplerRef::Constexpr(words), _) = sampler else {
        return Err(cx.invariant("a runtime sampler's own bias"));
    };
    let anisotropic = super::state_of(words)?.max_anisotropy > 1;
    let either = BiasReach::Either {
        max_bias_bits,
        sampler_bias_bits: own.to_bits(),
    };
    Ok(match (anisotropic, plane == 2) {
        (true, true) => BiasReach::Axes,
        _ => match konst(cx, v) {
            Some(Const::Float(bits)) => {
                let sum = super::emit::metal_bias_of(f32::from_bits(*bits as u32)) + own;
                if sum.abs() > f32::from_bits(max_bias_bits) {
                    BiasReach::Beyond
                } else {
                    BiasReach::Within
                }
            }
            _ => either,
        },
    })
}

/// The bias of `sampler`'s own: a constexpr sampler's `lod_bias`. A runtime sampler, bound or an
/// argument buffer's, has none (an `MTLSamplerDescriptor` states none).
fn own_bias(sampler: SamplerArg) -> Result<f32, Error> {
    Ok(match sampler {
        SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
            super::state_of(words)?.lod_bias.to_f32()
        }
        SamplerArg::Static(SamplerRef::Slot(_), _) | SamplerArg::Heap(_) => 0.0,
    })
}

/// Whether fragment bias sampled through `sampler` goes to the device's sampler whole (on
/// MoltenVK, `VkProperty::MetalSampler`, the `Bias` operand reaches Metal's `bias()` as Metal
/// applies it): the sampler has no bias of its own, whose sum with the operand's no measurement
/// has read.
fn bias_to_metal(sampler: SamplerArg) -> Result<bool, Error> {
    Ok(own_bias(sampler)? == 0.0)
}

/// Whether a rule-2 sample's hoisted derivatives give Metal's own sample: a sample of a 2D color
/// texture, not an array, without a bias, a `min_lod_clamp` or a nonzero offset, through samplers
/// whose fragment bias goes to the device's sampler whole ([`bias_to_metal`]: samplers without a
/// bias of their own). The device hands the
/// derivatives to Metal's `gradient2d`, and Metal's implicit sample of a texture chosen per
/// fragment is its sample through `gradient2d` of the coordinate's own derivatives, bit for bit,
/// through constexpr samplers and runtime sampler states of five filter settings, anisotropic among
/// them; through the runtime on MoltenVK the translation gives Metal's words on every quad.
fn hoist_is_metals(
    cx: &SelectCx<'_>,
    h: &super::Hoist,
    ops: &Operands<'_>,
    samplers: &[Candidate],
) -> Result<bool, Error> {
    let shape = ops.t.image;
    let plain_2d = shape.dim == ImageDim::D2 && !shape.array && !shape.depth && !shape.ms;
    let TexOp::Sample { offset, .. } = ops.t.op else {
        return Ok(false);
    };
    // An offset the call does not apply, or a constant zero.
    let unshifted = !offset
        || ops.get(TexRole::Offset).is_some_and(|v| {
            metal_const_offset(cx, v, 2).is_some_and(|k| k.iter().all(|x| *x == 0))
        });
    if h.bias.is_some() || !plain_2d || !unshifted || clamp_value(cx, ops)?.is_some() {
        return Ok(false);
    }
    for &c in samplers {
        if c != Candidate::Absent && !bias_to_metal(sampler_of(cx, c)?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `v` as a float constant that is zero (`+0.0`, `-0.0`, or `zeroinitializer`).
fn is_zero_float(cx: &SelectCx<'_>, v: Value) -> bool {
    match konst(cx, v) {
        Some(Const::Null) => true,
        // An `f32` constant's bit pattern; zero with either sign.
        Some(Const::Float(bits)) => bits & 0x7fff_ffff == 0,
        _ => false,
    }
}

/// The sampler a sampling reads: a static one (a runtime slot's or a constexpr sampler, with its
/// coordinates), or an argument buffer's, by its heap handle's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SamplerArg {
    Static(SamplerRef, Coordinates),
    Heap(ValueId),
}

/// The sampler a static sampler candidate names: a runtime `[[sampler(n)]]` slot (normalized, or
/// pixel by its `SamplerCoordinates` specialization) or a constexpr sampler.
fn sampler_of(cx: &SelectCx<'_>, c: Candidate) -> Result<SamplerArg, Error> {
    match c {
        Candidate::Sampler(n) => Ok(SamplerArg::Static(
            SamplerRef::Slot(n),
            Coordinates::Normalized,
        )),
        Candidate::Constexpr { global, .. } => cx
            .entry
            .static_samplers
            .iter()
            .find(|(g, _)| *g == global)
            .ok_or_else(|| cx.invariant("a constexpr sampler candidate without its state"))
            .and_then(|(_, state)| {
                // The state as Metal reads it, or its refusal.
                let state = super::constexpr::lowered(state)?;
                Ok(SamplerArg::Static(
                    SamplerRef::Constexpr(super::state_words(&state)),
                    state.coordinates,
                ))
            }),
        Candidate::Null | Candidate::Texture(_) => {
            Err(cx.invariant("a sampler operand with a texture candidate"))
        }
        // Its sampling is the absent read before any sampler is read.
        Candidate::Absent => Err(cx.invariant("an absent sampler read as a sampler")),
    }
}

/// Refuse an LOD query outside a fragment function where Metal's answer was not measured: in a
/// stage other than a kernel or a vertex function, or of a texture other than a 2D, 2D-array, 3D,
/// or cube one of `float` texels.
fn query_level_measured(cx: &SelectCx<'_>, shape: ImageShape, integer: bool) -> Result<(), Error> {
    let stage = matches!(cx.stage, Stage::Kernel | Stage::Vertex);
    let texture = !shape.depth
        && !shape.ms
        && !integer
        && match shape.dim {
            ImageDim::D2 | ImageDim::D3 => true,
            ImageDim::Cube => !shape.array,
            ImageDim::D1 | ImageDim::Buffer => false,
        };
    if stage && texture {
        Ok(())
    } else {
        Err(Unsupported::not_implemented(Feature::ImplicitLodOutsideFragment).into())
    }
}

/// The level a clamped LOD query outside a fragment function answers through `stat` before the
/// texture's last level bounds it: a constexpr sampler's from its state, a runtime slot's by its
/// `SamplerMinLevel` specialization. An argument buffer's sampler states its clamp in no table the
/// function reads, so its query is not implemented.
fn query_min_level(
    cx: &mut SelectCx<'_>,
    shape: ImageShape,
    integer: bool,
    stat: Option<(SamplerRef, Coordinates)>,
) -> Result<MinLevel, Error> {
    query_level_measured(cx, shape, integer)?;
    match stat {
        Some((SamplerRef::Constexpr(words), _)) => Ok(MinLevel::Const(
            super::constexpr::query_min_level(&super::state_of(words)?)?,
        )),
        Some((SamplerRef::Slot(n), _)) => match cx.families.sampler.slots.get_mut(&n) {
            Some(s) => {
                s.min_level = true;
                Ok(MinLevel::Slot(n))
            }
            None => Err(cx.invariant("an unplanned sampler slot")),
        },
        None => Err(Unsupported::not_implemented(Feature::ImplicitLodOutsideFragment).into()),
    }
}

/// A sampling call: the instruction, its decoded intrinsic, and its arguments.
#[derive(Clone, Copy)]
struct Call<'a> {
    at: InstId,
    inst: &'a Inst,
    t: &'a Texture,
    args: &'a [Value],
}

/// The candidates of a target: the one candidate, a switch's non-null candidates with the
/// handle it dispatches on, or none for the constant null handle.
fn split(target: HandleTarget) -> (Option<ValueId>, Vec<Candidate>) {
    match target {
        HandleTarget::One(c) => (None, vec![c]),
        HandleTarget::Switch { handle, cands } => (Some(handle), cands),
        HandleTarget::Null => (None, Vec::new()),
    }
}

/// Select sampling call `inst` (sample, gather, or LOD query) decoded as `t`, on texture
/// `texture`. A texture or sampler with several static candidates is a switch-at-use over every
/// candidate combination. A tangled use cannot sit in a switch-at-use: an implicit-LOD sample
/// becomes an explicit-gradient sample whose gradients (scaled by `2^bias`) are hoisted before the
/// switch (`PrecisionNote::GradFromBias`), and an LOD query, which has no gradient form, evaluates
/// every candidate's query before the switch, at the call's own position with the call's
/// coordinate, and the switch selects one: exact, as each is the query the call makes on that
/// candidate. Past [`super::LOD_QUERY_CANDIDATES`] combinations the query is
/// `DivergentImplicitLod`, not implemented.
pub fn select_sampling(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    t: &Texture,
    args: &[Value],
    texture: HandleTarget,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let call = Call { at, inst, t, args };
    let ops = Operands { t, args };
    let smp = ops
        .get(TexRole::Sampler)
        .ok_or_else(|| cx.invariant("a sampling call without a sampler"))?;
    // A sampling of the absent texture is the absent read, whatever samples it.
    if texture == HandleTarget::One(Candidate::Absent) {
        return AbsentArgument::select(cx, at, inst, t);
    }
    // A static texture sampled through an argument buffer's sampler would pair the slot's own
    // image with the heap sampler; it is not implemented.
    if crate::lower::access::handle::heap_sampler_target(cx, smp)?.is_some() {
        return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into());
    }
    let sampler = crate::lower::access::handle::sampler_target(cx, smp, m)?;
    // A sampling through the absent sampler is the absent read too.
    if sampler == HandleTarget::One(Candidate::Absent) {
        return AbsentArgument::select(cx, at, inst, t);
    }
    if matches!(texture, HandleTarget::Null) {
        // A sample of the null texture (a `null`, `undef`, or `poison` handle; an absent texture
        // argument's observed reads are absent reads) is the null arm alone, zero, as any read of
        // it; it reads no image and has no tangled instruction.
        return Ok(TextureStrategy::Switch(Box::new(Switch {
            handles: Vec::new(),
            arms: Vec::new(),
            null: NullArm::Zero(cx.lowered_repr(Value::Inst(at), inst.ty)?),
            prelude: Prelude::Nothing,
        })));
    }
    if matches!(sampler, HandleTarget::Null) {
        // A texture that may be present sampled through the null sampler (a `null`, `undef`, or
        // `poison` handle; an absent sampler argument's observed samples are absent reads): Metal
        // defines no result and none is measured.
        return Err(refuse());
    }
    if !texture.switched() && !sampler.switched() {
        let (HandleTarget::One(Candidate::Texture(index)), HandleTarget::One(s)) =
            (texture, sampler)
        else {
            return Err(cx.invariant("a sampling operand of the wrong class"));
        };
        let s = sampler_of(cx, s)?;
        return select_slot(cx, call, index, s, None, m);
    }
    let (tex_handle, textures) = split(texture);
    let (smp_handle, samplers) = split(sampler);
    let combinations = textures.len().saturating_mul(samplers.len());
    // A tangled use: the fragment's implicit-LOD sample, or an LOD query.
    let fragment = cx.stage == Stage::Fragment;
    let mut queries = false;
    let hoist = match t.op {
        // Every candidate's query, at the call's own position with the call's coordinate,
        // before the switch, which selects one: bounded, as every fragment runs each.
        TexOp::CalculateLod { .. } if fragment => {
            if combinations > super::LOD_QUERY_CANDIDATES {
                return Err(Unsupported::not_implemented(Feature::DivergentImplicitLod).into());
            }
            queries = true;
            None
        }
        TexOp::Sample {
            lod: LodMode::Bias, ..
        } if fragment => {
            let coord = ops
                .get(TexRole::Coord)
                .ok_or_else(|| cx.invariant("a sampling call without a coordinate"))?;
            let bias = match ops.get(TexRole::LodValue) {
                Some(v) if !is_zero_float(cx, v) => Some(id(cx, v)?),
                _ => None,
            };
            Some(super::Hoist {
                coord: id(cx, coord)?,
                lanes: plane(t.image.dim),
                bias,
            })
        }
        _ => None,
    };
    let mut arms = Vec::with_capacity(combinations);
    for tc in &textures {
        let index = match *tc {
            Candidate::Texture(index) => Some(index),
            // The absent texture's arms are absent reads.
            Candidate::Absent => None,
            _ => return Err(cx.invariant("a texture operand with a sampler candidate")),
        };
        for sc in &samplers {
            let one = match (index, sc) {
                (Some(index), sc) if *sc != Candidate::Absent => {
                    let s = sampler_of(cx, *sc)?;
                    select_slot(cx, call, index, s, Some(hoist.is_some()), m)?
                }
                // The absent texture, or the absent sampler.
                _ => AbsentArgument::select(cx, at, inst, t)?,
            };
            let mut lits = Vec::with_capacity(2);
            if tex_handle.is_some() {
                lits.push(tc.handle());
            }
            if smp_handle.is_some() {
                lits.push(sc.handle());
            }
            arms.push((lits, one));
        }
    }
    // The gradients are hoisted only for arms that sample with them: an arm through a nearest
    // pixel sampler fetches and needs none.
    let hoist = hoist.filter(|_| {
        arms.iter().any(|(_, s)| {
            matches!(s, TextureStrategy::Sampling(x) if matches!(
                x.op,
                SampleOp::Sample {
                    lod: Lod::Hoisted,
                    ..
                }
            ))
        })
    });
    // The hoisted derivatives are the whole bias's, but Metal's own sample where the device samples
    // through Metal's sampler and the sample takes no bias, which the module then requires.
    if let Some(h) = &hoist {
        if hoist_is_metals(cx, h, &ops, &samplers)? {
            cx.families.sampler.metal_sampler = true;
        } else {
            cx.precision.insert(PrecisionNote::GradFromBias);
        }
    }
    let prelude = match hoist {
        Some(h) => Prelude::Gradients(h),
        None if queries => Prelude::Arms,
        None => Prelude::Nothing,
    };
    Ok(TextureStrategy::Switch(Box::new(Switch {
        handles: tex_handle.into_iter().chain(smp_handle).collect(),
        arms,
        null: NullArm::Zero(cx.lowered_repr(Value::Inst(at), inst.ty)?),
        prelude,
    })))
}

/// Select sampling call `inst` (sample, gather, or LOD query) decoded as `t` on element `handle`
/// of heap group `group` (a texture loaded from an argument buffer): the heap element through a
/// separate heap sampler. The sampler is one constexpr sampler, or one runtime `[[sampler(n)]]`,
/// whose heap sampler binding the binder writes with the sampler bound at the slot, or a sampler
/// loaded from an argument buffer, whose element of the sampler array of the use the binder writes
/// with it; several sampler candidates beside a heap image are not implemented
/// (`ArgumentBufferTextures`), and the null sampler has no defined result.
pub fn select_heap_sampling(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    t: &Texture,
    args: &[Value],
    group: u32,
    handle: ValueId,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let call = Call { at, inst, t, args };
    let ops = Operands { t, args };
    let smp = ops
        .get(TexRole::Sampler)
        .ok_or_else(|| cx.invariant("a sampling call without a sampler"))?;
    let s = match crate::lower::access::handle::heap_sampler_target(cx, smp)? {
        Some(h) => SamplerArg::Heap(h),
        None => match crate::lower::access::handle::sampler_target(cx, smp, m)? {
            // A heap texture sampled through the absent sampler: the absent read.
            HandleTarget::One(Candidate::Absent) => {
                return AbsentArgument::select(cx, at, inst, t);
            }
            HandleTarget::One(c) => sampler_of(cx, c)?,
            HandleTarget::Null => return Err(refuse()),
            HandleTarget::Switch { .. } => {
                return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into())
            }
        },
    };
    let image = ImageRef::Heap { group, handle };
    let mut x = select_one(cx, call, image, s, None, m)?;
    x.heap_raw = select_heap_raw(cx, call, (group, handle), s)?;
    Ok(TextureStrategy::Sampling(Box::new(x)))
}

/// The emulation of heap sampling call `call` of element `handle` of heap group `group` through
/// `sampler` that answers a texture the heap holds by its raw texels alone as Metal does
/// (`heap_raw`), where the call's texture type may hold one: a non-multisampled color texture
/// of a float class (a `float` or `half` result's, or a level query's of a group no access
/// states as an integer one), whose float class holds such a texture's neutral (`bind::heap`).
/// `None` for another type, and where the emulation is not built ([`heap_raw_sample`]). Where
/// such a type's sampling is not answered for every such texture (`HeapRaw::whole`), or is
/// through a runtime sampler slot, whose bound sampler decides it, the stage states
/// `RuntimeFact::HeapRawSamples` (naming the slot for the latter).
fn select_heap_raw(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    (group, handle): (u32, ValueId),
    sampler: SamplerArg,
) -> Result<Option<super::heap_raw::HeapRaw>, Error> {
    let shape = call.t.image;
    let float = match call.t.texel {
        Some(_) => texel(call.t, 4)?.1 == TexelClass::Float,
        None => cx
            .families
            .texture
            .heap
            .get(&group)
            .and_then(|h| h.texel)
            .is_none_or(|c| c == TexelClass::Float),
    };
    if shape.ms || shape.depth || shape.dim == ImageDim::Buffer || !float {
        return Ok(None);
    }
    let raw = heap_raw_sample(cx, call, (group, handle), sampler)?;
    // The neutral wherever such a texture is reached, or where the runtime sampler slot's bound
    // sampler states what the emulation does not answer.
    let reads_neutral = match &raw {
        Some(r) if r.whole() => match r.state {
            super::heap_raw::State::Constant { .. } => None,
            super::heap_raw::State::Slot { sampler } => Some(Some(sampler)),
        },
        _ => Some(None),
    };
    if let Some(sampler) = reads_neutral {
        cx.facts
            .insert(crate::pipeline::facts::RuntimeFact::HeapRawSamples {
                stage: cx.stage,
                sampler,
            });
    }
    Ok(raw)
}

/// The emulation of [`select_heap_raw`]'s call, where it is built as a slot's extended-range
/// sample is ([`select_extended`]): a 2D texture's sample or gradient sample without
/// comparison, or gather of a component without comparison, with a `float` or `half` result, a
/// constant texel offset or none, no `min_lod_clamp`, and no observed residency, through a
/// constexpr sampler with one filter for minification and magnification, no mip filter, and no
/// anisotropy, or through a runtime sampler slot (not one of the plan's cubic variant), whose
/// word the emulation tests. `None` for every other call, an argument buffer's sampler's
/// among them (whose listed states the plan does not check against its textures). It
/// plans the group's unsigned integer class (`HeapPlan::reads_raw`), the heap's decode routines,
/// the handle's class index, and the exact routine where it may filter.
fn heap_raw_sample(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    (group, handle): (u32, ValueId),
    sampler: SamplerArg,
) -> Result<Option<super::heap_raw::HeapRaw>, Error> {
    use super::bicubic::TapAddress;
    use super::heap_raw::{HeapRaw, State};
    let Call { at, t, args, .. } = call;
    let ops = Operands { t, args };
    let shape = t.image;
    let (has_offset, gather) = match t.op {
        TexOp::Sample {
            compare: false,
            offset,
            ..
        }
        | TexOp::SampleGrad {
            compare: false,
            offset,
        } => (offset, None),
        TexOp::Gather {
            component,
            compare: false,
            offset,
        } => {
            use crate::air::intrinsics::Component as C;
            let component = match component {
                None | Some(C::X) => 0,
                Some(C::Y) => 1,
                Some(C::Z) => 2,
                Some(C::W) => 3,
            };
            (offset, Some(component))
        }
        _ => return Ok(None),
    };
    if shape.dim != ImageDim::D2 || shape.array {
        return Ok(None);
    }
    if clamp_value(cx, &ops)?.is_some() {
        return Ok(None);
    }
    let offset = match fetch_offset(cx, &ops, has_offset, plane(shape.dim))? {
        None => None,
        Some(Offset::Const(k)) => Some(k),
        Some(Offset::Dynamic(..)) => return Ok(None),
    };
    let half = match texel(t, 4)?.0.conv {
        TexelConv::F16 => true,
        TexelConv::F32 => false,
        _ => return Ok(None),
    };
    let state = match sampler {
        SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
            let s = super::state_of(words)?;
            let linear = match (s.min_filter, s.mag_filter) {
                (Filter::Linear, Filter::Linear) => true,
                (Filter::Nearest, Filter::Nearest) => false,
                _ => return Ok(None),
            };
            if s.mip_filter != MipFilter::None || s.max_anisotropy > 1 {
                return Ok(None);
            }
            let address = [s.address_s, s.address_t]
                .into_iter()
                .map(|m| match m {
                    AddressMode::ClampToEdge => TapAddress::Edge,
                    AddressMode::Repeat => TapAddress::Repeat,
                    AddressMode::MirroredRepeat => TapAddress::Mirror,
                    AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
                })
                .collect();
            State::Constant {
                linear: linear && gather.is_none(),
                address,
                border: s.border_color,
                pixel: s.coordinates == Coordinates::Pixel,
            }
        }
        SamplerArg::Static(SamplerRef::Slot(n), _) => {
            if cubic_slot(cx, n)?.is_some() {
                return Ok(None);
            }
            let Some(s) = cx.families.sampler.slots.get_mut(&n) else {
                return Err(cx.invariant("an unplanned sampler slot"));
            };
            s.linear = true;
            State::Slot { sampler: n }
        }
        // An argument buffer's sampler's state is not checked against its textures at bind.
        SamplerArg::Heap(_) => return Ok(None),
    };
    if crate::lower::texture::residency(cx, at, ImageRef::Heap { group, handle })? {
        return Ok(None);
    }
    // An offset on an axis that may repeat or mirror is placed by Metal's rule on the element's
    // own base-level size, where Metal reads a view's parent's: a heap element's access reads no
    // slot's `OffsetBaseAxes` (the residue).
    let wraps = |a: usize| match &state {
        State::Constant { address, .. } => address
            .get(a)
            .is_some_and(|m| matches!(m, TapAddress::Repeat | TapAddress::Mirror)),
        State::Slot { .. } => true,
    };
    let moved = offset.as_deref().unwrap_or_default();
    if moved.iter().enumerate().any(|(a, k)| *k != 0 && wraps(a)) {
        cx.precision.insert(PrecisionNote::ViewBaseTexelOffset);
    }
    if let Some(h) = cx.families.texture.heap.get_mut(&group) {
        h.reads_raw = true;
    }
    cx.families.texture.heap_raw.insert(false);
    if half {
        cx.families.texture.heap_raw.insert(true);
    }
    cx.families.access.heap_class_index = true;
    if gather.is_none() && !matches!(state, State::Constant { linear: false, .. }) {
        cx.families
            .sampler
            .bicubic_routines
            .insert(super::bicubic::routines::Routine::Quad);
    }
    Ok(Some(HeapRaw {
        state,
        gather,
        offset,
        half,
    }))
}

/// The texel form of a sampling result: `lanes` components of the named texel.
fn texel(t: &Texture, lanes: u8) -> Result<(TexelForm, TexelClass), Error> {
    let x = t.texel.ok_or_else(refuse)?;
    let (conv, class) = texel_conv(x.ty.elem, x.sign)?;
    Ok((TexelForm { conv, lanes }, class))
}

/// [`texel`] of a sampling of a slot whose variant binds a texture of class `cross`, where it
/// does (`texture::cross_class`): that class's own form (`texture::native_form`), with the
/// declared class, whose type the slot's interface keeps. A declared result whose answer is not
/// measured is `NotImplemented(Format)`; the slot's compile states no
/// `RuntimeFact::CrossClassReads` for it, so the binder asks for no such variant.
fn texel_in(
    t: &Texture,
    cross: Option<TexelClass>,
    lanes: u8,
) -> Result<(TexelForm, TexelClass), Error> {
    let (form, class) = texel(t, lanes)?;
    match cross {
        None => Ok((form, class)),
        Some(bound) => match crate::lower::texture::native_form(form, bound) {
            Some(native) => Ok((native, class)),
            None => Err(Unsupported::not_implemented(Feature::Format).into()),
        },
    }
}

/// The plane coordinate count of a sampled `dim` (a cube samples by a direction).
fn plane(dim: ImageDim) -> u8 {
    match dim {
        ImageDim::D1 | ImageDim::Buffer => 1,
        ImageDim::D2 => 2,
        ImageDim::D3 | ImageDim::Cube => 3,
    }
}

/// A texel offset: absent, zero (dropped), a constant within the device's range (`gather`
/// selects the gather range), or a dynamic offset ([`dynamic_offset`]). A constant is the value
/// Metal applies for it ([`super::metal_offset`]). A cube takes none: Metal's cube `sample` and
/// `gather` have no offset parameter and SPIR-V admits no offset on a cube image, so a set flag
/// with a nonzero or dynamic offset there (malformed AIR) is `NotRepresentable`.
fn offset(
    cx: &mut SelectCx<'_>,
    ops: &Operands<'_>,
    present: bool,
    lanes: u8,
    gather: bool,
) -> Result<Option<Offset>, Error> {
    let Some(v) = ops.get(TexRole::Offset).filter(|_| present) else {
        return Ok(None);
    };
    let k = metal_const_offset(cx, v, lanes);
    if k.as_ref().is_some_and(|k| k.iter().all(|x| *x == 0)) {
        return Ok(None);
    }
    if ops.t.image.dim == ImageDim::Cube {
        return Err(Unsupported::not_representable(Feature::TexelOffset).into());
    }
    let Some(k) = k else {
        return dynamic_offset(cx, v, lanes, gather).map(Some);
    };
    in_device_range(cx, &k, gather)?;
    Ok(Some(Offset::Const(k)))
}

/// A dynamic texel offset: the `Offset` image operand, which needs `shaderImageGatherExtended`, and
/// on a sample also `maintenance8` (VUID-RuntimeSpirv-Offset-10213). Without them the access takes
/// the offset's arms ([`super::OffsetArms`]): a switch over the values Metal reads on its dynamic
/// lanes, each arm the access with its value as `ConstOffset`, which Vulkan 1.0 admits on every
/// sample and gather. A function whose armed accesses take more than
/// [`super::OFFSET_ARMS_PER_FUNCTION`] arms together is refused as without the arms: a gather
/// `GatherExtended` and a sample `TexelOffset`, naming the missing features. Emission reads the
/// value as Metal does ([`super::metal_offset`]), which lies in Metal's range
/// [`super::METAL_TEXEL_OFFSET`], and Vulkan requires the runtime value, and a constant, within the
/// device's range (VUID-RuntimeSpirv-OpImageSample-06435/06436, gathers 06376/06377), so the
/// device's range must contain all of Metal's. A nearest pixel sampler's fetch adds the offset to
/// the texel coordinate instead (`fetch_offset`).
fn dynamic_offset(
    cx: &mut SelectCx<'_>,
    v: Value,
    lanes: u8,
    gather: bool,
) -> Result<Offset, Error> {
    let images = &cx.profile.fields().images;
    let missing: Vec<Capability> = [
        (
            gather || images.dynamic_sample_offset,
            VkFeature::Maintenance8,
        ),
        (images.gather_extended, VkFeature::ShaderImageGatherExtended),
    ]
    .into_iter()
    .filter(|(enabled, _)| !enabled)
    .map(|(_, f)| Capability::Feature(f))
    .collect();
    let (lo, hi) = super::METAL_TEXEL_OFFSET;
    in_device_range(cx, &[lo, hi], gather)?;
    let value = id(cx, v)?;
    if missing.is_empty() {
        return Ok(Offset::Dynamic(value, None));
    }
    let arms = super::OffsetArms {
        lanes: offset_lanes(cx, v, lanes),
    };
    let total = cx.families.sampler.offset_arms.saturating_add(arms.arms());
    if total > super::OFFSET_ARMS_PER_FUNCTION {
        let feature = if gather {
            Feature::GatherExtended
        } else {
            Feature::TexelOffset
        };
        return Err(Unsupported::needs_any_of(feature, vec![missing]).into());
    }
    cx.families.sampler.offset_arms = total;
    Ok(Offset::Dynamic(value, Some(arms)))
}

/// How many instructions [`offset_lane`] reads back from one lane of an offset: a chain that sets
/// each lane of a 3D offset, shuffles it and adds constants to it, and no more.
const OFFSET_LANE_STEPS: usize = 16;

/// Each of the `lanes` lanes of dynamic offset `v` as Metal reads it ([`super::metal_offset`])
/// where the IR states it constant, else `None`.
fn offset_lanes(cx: &SelectCx<'_>, v: Value, lanes: u8) -> Vec<Option<i32>> {
    (0..u32::from(lanes))
        .map(|j| {
            let mut steps = OFFSET_LANE_STEPS;
            offset_lane(cx, v, lanes, j, &mut steps).map(super::metal_offset)
        })
        .collect()
}

/// Lane `lane` of `v`, a vector of `n` lanes (a scalar when `n` is 1), where the IR states it
/// constant: a constant's lane, a lane that inserts and shuffles carry from one, or a lane-wise
/// `add`, `sub`, `mul`, `and`, `or` or `xor` of two such lanes, reading at most `steps` more
/// instructions. A shuffle's poison lane is its first source's lane 0, the value the arithmetic
/// family gives it (`lower::arith::chunk`); integer operations wrap at 32 bits, as the lowered
/// operations do, which refines an overflow's poison.
fn offset_lane(cx: &SelectCx<'_>, v: Value, n: u8, lane: u32, steps: &mut usize) -> Option<i32> {
    let Value::Inst(i) = v else {
        return const_lane(cx, v, n, lane);
    };
    *steps = steps.checked_sub(1)?;
    match &cx.body.insts.get(i)?.kind {
        InstKind::InsertElement { vec, elt, idx } if n > 1 => {
            let Const::Int(k) = konst(cx, *idx)? else {
                return None;
            };
            if *k == u64::from(lane) {
                offset_lane(cx, *elt, 1, 0, steps)
            } else {
                offset_lane(cx, *vec, n, lane, steps)
            }
        }
        InstKind::ShuffleVector { a, b, mask } => {
            let m = mask.get(usize::try_from(lane).ok()?)?.unwrap_or(0);
            // Both sources have one vector type.
            let from_a = match cx.entry.module.types().get(cx.value_ty(*a)?)? {
                crate::air::ir::Type::Vector { len, .. } => *len,
                _ => return None,
            };
            let n = u8::try_from(from_a).ok()?;
            if m < from_a {
                offset_lane(cx, *a, n, m, steps)
            } else {
                offset_lane(cx, *b, n, m - from_a, steps)
            }
        }
        InstKind::BinOp { op, a, b, .. } => {
            let fold: fn(i32, i32) -> i32 = match op {
                BinOp::Add => i32::wrapping_add,
                BinOp::Sub => i32::wrapping_sub,
                BinOp::Mul => i32::wrapping_mul,
                BinOp::And => |x, y| x & y,
                BinOp::Or => |x, y| x | y,
                BinOp::Xor => |x, y| x ^ y,
                _ => return None,
            };
            let x = offset_lane(cx, *a, n, lane, steps)?;
            Some(fold(x, offset_lane(cx, *b, n, lane, steps)?))
        }
        _ => None,
    }
}

/// Lane `lane` of constant `v`, a vector of `n` lanes (a scalar when `n` is 1), when it is an
/// integer (zero for `zeroinitializer`), else `None`: an `undef` or `poison` lane among others.
fn const_lane(cx: &SelectCx<'_>, v: Value, n: u8, lane: u32) -> Option<i32> {
    // Integers are zero-extended bit patterns: the low 32 bits are the value.
    let int = |c: &Const| match c {
        Const::Null => Some(0),
        Const::Int(u) => Some(*u as u32 as i32),
        _ => None,
    };
    match konst(cx, v)? {
        Const::Null => Some(0),
        Const::Aggregate(parts) => {
            let part = parts.get(usize::try_from(lane).ok()?)?;
            int(&cx.entry.module.consts().get(*part)?.value)
        }
        c if n == 1 => int(c),
        _ => None,
    }
}

/// Every offset component in `k` lies in the device's texel offset range (`gather`: the gather
/// range), else `Unsupported{TexelOffset}` naming the first limit it breaks.
fn in_device_range(cx: &SelectCx<'_>, k: &[i32], gather: bool) -> Result<(), Error> {
    let limits = &cx.profile.fields().limits;
    let (min, max, lo, hi) = if gather {
        (
            limits.min_texel_gather_offset,
            limits.max_texel_gather_offset,
            VkLimit::MinTexelGatherOffset,
            VkLimit::MaxTexelGatherOffset,
        )
    } else {
        (
            limits.min_texel_offset,
            limits.max_texel_offset,
            VkLimit::MinTexelOffset,
            VkLimit::MaxTexelOffset,
        )
    };
    for x in k {
        let (limit, required, available) = if *x < min {
            (
                lo,
                u64::from(x.unsigned_abs()),
                u64::from(min.unsigned_abs()),
            )
        } else if i64::from(*x) > i64::from(max) {
            (hi, u64::from(x.unsigned_abs()), u64::from(max))
        } else {
            continue;
        };
        return Err(Unsupported::needs(
            Feature::TexelOffset,
            Capability::Limit {
                limit,
                required,
                available,
            },
        )
        .into());
    }
    Ok(())
}

/// The offset of a pixel fetch: absent, zero (dropped), a constant, or dynamic, each the value
/// Metal applies ([`super::metal_offset`]). The fetch adds it to the integer texel coordinate
/// itself, so neither the device's offset range nor `Offset`'s gather-only rule applies.
fn fetch_offset(
    cx: &SelectCx<'_>,
    ops: &Operands<'_>,
    present: bool,
    lanes: u8,
) -> Result<Option<Offset>, Error> {
    let Some(v) = ops.get(TexRole::Offset).filter(|_| present) else {
        return Ok(None);
    };
    Ok(match metal_const_offset(cx, v, lanes) {
        Some(k) if k.iter().all(|x| *x == 0) => None,
        Some(k) => Some(Offset::Const(k)),
        None => Some(Offset::Dynamic(id(cx, v)?, None)),
    })
}

/// The offset Metal applies for constant offset `v` of `lanes` components
/// ([`super::metal_offset`] per component), or `None` when the IR does not state each lane
/// constant ([`offset_lanes`]).
fn metal_const_offset(cx: &SelectCx<'_>, v: Value, lanes: u8) -> Option<Vec<i32>> {
    offset_lanes(cx, v, lanes).into_iter().collect()
}

/// The reference value of a comparison (`compare`), else `None`.
fn compare_ref(
    cx: &SelectCx<'_>,
    ops: &Operands<'_>,
    compare: bool,
) -> Result<Option<ValueId>, Error> {
    if !compare {
        return Ok(None);
    }
    let r = ops
        .get(TexRole::CompareRef)
        .ok_or_else(|| cx.invariant("a comparison without a reference"))?;
    Ok(Some(id(cx, r)?))
}

/// A nonzero `min_lod_clamp`.
fn clamp_value(cx: &SelectCx<'_>, ops: &Operands<'_>) -> Result<Option<ValueId>, Error> {
    match ops.get(TexRole::MinLodClamp) {
        Some(v) if !is_zero_float(cx, v) => Ok(Some(id(cx, v)?)),
        _ => Ok(None),
    }
}

/// A nonzero `min_lod_clamp` as the `MinLod` image operand, which needs `shaderResourceMinLod`.
fn min_lod(cx: &SelectCx<'_>, clamp: Option<ValueId>) -> Result<Option<ValueId>, Error> {
    let Some(v) = clamp else {
        return Ok(None);
    };
    if !cx.profile.fields().images.min_lod {
        return Err(Unsupported::needs(
            Feature::MinLodClamp,
            Capability::Feature(VkFeature::ShaderResourceMinLod),
        )
        .into());
    }
    Ok(Some(v))
}

/// Whether constexpr sampler `r` finds its texel on a texture of dimension `dim` exactly by a
/// fetch: pixel coordinates (Metal then requires no mip filter and equal minification and
/// magnification filters), nearest filtering, no anisotropy, and clamp-to-edge addressing on every
/// plane axis. An `integer` texture is never filtered, so its sampler's filters read as nearest
/// whatever they say (`metal::sampler::vk_sampler`). A pixel sampler that clamps to zero or to a
/// border is not: its out-of-range texel (the zero's alpha, the border) depends on the bound pixel
/// format.
fn nearest_fetch(r: SamplerRef, dim: ImageDim, integer: bool) -> Result<bool, Error> {
    let SamplerRef::Constexpr(words) = r else {
        return Ok(false);
    };
    let s = super::state_of(words)?;
    let axes = match dim {
        ImageDim::D1 => 1,
        ImageDim::D2 => 2,
        ImageDim::D3 => 3,
        ImageDim::Cube | ImageDim::Buffer => return Ok(false),
    };
    let edge = [s.address_s, s.address_t, s.address_r]
        .iter()
        .take(axes)
        .all(|m| *m == AddressMode::ClampToEdge);
    let nearest = integer || (s.min_filter == Filter::Nearest && s.mag_filter == Filter::Nearest);
    Ok(s.coordinates == Coordinates::Pixel
        && nearest
        && s.mip_filter == MipFilter::None
        && s.max_anisotropy <= 1
        && edge)
}

/// Select sampling call `call` of texture slot `index` through static sampler `sampler`: an
/// extended-range sample where the slot's variant binds such a format ([`select_extended`]), else
/// [`select_one`]'s sampling. `arm` is set for an arm of a switch-at-use, whether it reads the
/// gradients its switch hoisted ([`select_one`]).
fn select_slot(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    index: u32,
    sampler: SamplerArg,
    arm: Option<bool>,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    // A constexpr sampler samples the slot with its address modes as the bound texture's map
    // sends them (`LoweringInputs::address_maps`): its binding's immutable sampler and every
    // emulation that reads its modes take the mapped state.
    let sampler = match sampler {
        SamplerArg::Static(SamplerRef::Constexpr(words), coordinates) => {
            match crate::lower::texture::constexpr_address_map(cx, index, words)? {
                Some(map) => SamplerArg::Static(
                    SamplerRef::Constexpr(super::constexpr::mapped_words(words, &map)?),
                    coordinates,
                ),
                None => sampler,
            }
        }
        _ => sampler,
    };
    if let Some(sample) = select_in_place(cx, call, index, &sampler, arm)? {
        return Ok(TextureStrategy::InPlaceSample(Box::new(sample)));
    }
    match crate::lower::texture::extended_range(cx, ImageRef::Slot(index))? {
        Some(form) => select_extended(cx, call, (index, form), sampler, m),
        None => select_one(cx, call, ImageRef::Slot(index), sampler, arm, m)
            .map(|x| TextureStrategy::Sampling(Box::new(x))),
    }
}

/// Observe sampling call `call` of texture slot `index` through `sampler` for the slot's in-place
/// form (`slot::InPlaceReads`), and where the slot's variant reads a two-plane YCbCr texture's
/// planes in place (`LoweringInputs::ycbcr_in_place`) select the form's sample
/// ([`in_place_sample`]); else `None`, and the slot's other forms sample it. A call the form does
/// not read keeps the slot from it, so its variant never meets one.
fn select_in_place(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    index: u32,
    sampler: &SamplerArg,
    arm: Option<bool>,
) -> Result<Option<super::in_place::Sample>, Error> {
    let variant = crate::lower::texture::ycbcr_in_place(cx, index)?;
    let Some((sample, samplers)) = in_place_sample(cx, call, index, sampler, arm)? else {
        crate::lower::texture::not_in_place(cx, index);
        return Ok(None);
    };
    let Some(s) = cx.families.texture.slots.get_mut(&index) else {
        return Err(cx.invariant("an unplanned texture slot"));
    };
    s.in_place.samplers |= samplers;
    s.in_place.device_blocked |= !device_takes(&sample);
    if !variant {
        return Ok(None);
    }
    // The slot's planes are fetched, as a read fetches texels.
    s.observe(call.t.image, Some(TexelClass::Float))?;
    s.uses.read = true;
    s.offset_base |= sample.offset_base;
    // A runtime sampler's word states the filter, the coordinates, the address modes and the
    // border color.
    if let super::extended::State::Slot { sampler: n } = sample.state {
        let Some(r) = cx.families.sampler.slots.get_mut(&n) else {
            return Err(cx.invariant("an unplanned sampler slot"));
        };
        r.linear = true;
    }
    if !matches!(
        sample.state,
        super::extended::State::Constant { linear: false, .. }
    ) {
        cx.families
            .sampler
            .bicubic_routines
            .insert(super::bicubic::routines::Routine::Quad);
    }
    Ok(Some(sample))
}

/// Whether the device's own sampler Y′CbCr conversion of a planes image takes in-place sample `s`
/// (`LoweringInputs::ycbcr_planes_sampled`, `super::in_place`'s **The device's sampler**): a
/// constexpr sampler that addresses no border on either plane axis (a 4:2:0 texture samples every
/// other mode as clamp-to-edge, as that sampler does), or a runtime sampler, whose bound word the
/// binder holds to the same (`bind::plan::slots::planes_word`); and no texel offset Metal places
/// by a parent's size.
fn device_takes(s: &super::in_place::Sample) -> bool {
    use super::bicubic::TapAddress;
    let border = match &s.state {
        super::extended::State::Constant { address, .. } => {
            address.iter().any(|m| matches!(m, TapAddress::Border))
        }
        super::extended::State::Slot { .. } => false,
    };
    !border && !s.offset_base
}

/// Sampling call `call` of texture slot `index` through `sampler` as the in-place form reads it
/// (`in_place`), with the runtime sampler slots it reads (bit `n` `[[sampler(n)]]`); `None` where
/// the form does not: anything but a 2D texture's sample or gradient sample without comparison,
/// with a `float` or `half` result, a constant texel offset or none, no `min_lod_clamp`, and no
/// status word observed, through a constexpr sampler of one filter for minification and
/// magnification (nearest or linear), no anisotropy and the weighted-average reduction, or a
/// runtime sampler slot, whose bound word the binder holds to the same
/// (`bind::plan::slots::in_place_word`); and not an arm of a switch-at-use. Metal's sample reads
/// the texture's one level whatever the mip filter, the level, the bias or the gradients, so the
/// form reads none of them.
fn in_place_sample(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    index: u32,
    sampler: &SamplerArg,
    arm: Option<bool>,
) -> Result<Option<(super::in_place::Sample, u32)>, Error> {
    use super::bicubic::TapAddress;
    use super::extended::State;
    let Call { at, inst, t, args } = call;
    let ops = Operands { t, args };
    let shape = t.image;
    if arm.is_some() || shape.dim != ImageDim::D2 || shape.array || shape.ms || shape.depth {
        return Ok(None);
    }
    let has_offset = match t.op {
        TexOp::Sample {
            compare: false,
            offset,
            ..
        }
        | TexOp::SampleGrad {
            compare: false,
            offset,
        } => offset,
        _ => return Ok(None),
    };
    if clamp_value(cx, &ops)?.is_some() || crate::lower::texture::observes_status(cx, at) {
        return Ok(None);
    }
    let offset = match fetch_offset(cx, &ops, has_offset, 2)? {
        None => None,
        Some(Offset::Const(k)) => Some(k),
        Some(Offset::Dynamic(..)) => return Ok(None),
    };
    let Ok((form, _)) = texel(t, 4) else {
        return Ok(None);
    };
    let half = match form.conv {
        TexelConv::F16 => true,
        TexelConv::F32 => false,
        _ => return Ok(None),
    };
    let (state, samplers) = match *sampler {
        SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
            let s = super::state_of(words)?;
            let linear = match (s.min_filter, s.mag_filter) {
                (Filter::Linear, Filter::Linear) => true,
                (Filter::Nearest, Filter::Nearest) => false,
                _ => return Ok(None),
            };
            if s.max_anisotropy > 1 || s.reduction != crate::air::abi::Reduction::WeightedAverage {
                return Ok(None);
            }
            let address = [s.address_s, s.address_t]
                .into_iter()
                .map(|m| match m {
                    AddressMode::ClampToEdge => TapAddress::Edge,
                    AddressMode::Repeat => TapAddress::Repeat,
                    AddressMode::MirroredRepeat => TapAddress::Mirror,
                    AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
                })
                .collect();
            let state = State::Constant {
                linear,
                address,
                border: s.border_color,
                pixel: s.coordinates == Coordinates::Pixel,
            };
            (state, 0)
        }
        SamplerArg::Static(SamplerRef::Slot(n), _) if n < 32 => {
            (State::Slot { sampler: n }, 1 << n)
        }
        _ => return Ok(None),
    };
    // An offset on an axis that may repeat or mirror is placed by Metal's rule, which reads a
    // view's parent's base-level size: the slot's `OffsetBaseAxes` states where it differs.
    let offset_base = offset.as_deref().is_some_and(|o| {
        o.iter().enumerate().any(|(a, k)| {
            *k != 0
                && match &state {
                    State::Constant { address, .. } => address
                        .get(a)
                        .is_some_and(|m| matches!(m, TapAddress::Repeat | TapAddress::Mirror)),
                    State::Slot { .. } => true,
                }
        })
    });
    let coord = ops
        .get(TexRole::Coord)
        .ok_or_else(|| cx.invariant("a sampling call without a coordinate"))?;
    let sample = super::in_place::Sample {
        texture: index,
        state,
        offset,
        offset_base,
        coord: id(cx, coord)?,
        half,
        result: cx.lowered_repr(Value::Inst(at), inst.ty)?,
    };
    Ok(Some((sample, samplers)))
}

/// Select sampling call `call` of texture slot `index`, whose variant binds the extended-range
/// format `form`, through `sampler`: Metal's nearest or linear filter of the decoded texels,
/// emulated by fetches of the raw texels on every device (`extended`). Built: a 2D or 2D array
/// texture's sample or gradient sample without comparison (and a `BGR10_XR` or `BGRA10_XR`
/// form's, or `_sRGB` twin's, 3D texture's under no mip filter or a nearest one), with a `float` or `half` result, a
/// constant texel offset or none, no `min_lod_clamp`, and no observed residency, through a
/// constexpr sampler with one filter for minification and magnification (nearest or linear), no
/// anisotropy, and no mip filter, a nearest one, or a linear one of a form whose levels' blend is
/// built (`ExtendedRange::blends_levels`), or through a runtime sampler slot, whose bound sampler
/// the binder holds to the same (`ResourcePlanner::pair`). Its level is Metal's rule's
/// (`extended::Level`): an explicit level or a kernel's bias exactly, a fragment function's
/// implicit level and explicit gradients by Vulkan's level of detail, recorded
/// (`PrecisionNote::DeviceSampleLevel`). A bfloat form's sample is nearest under either filter, as
/// Metal's, and its gather of a component is built, the taps of the linear footprint; so is another
/// form's gather, but a two-plane form's and a padded form's `half` one, at the base level; and a
/// swizzled texture of the slot's variant (`LoweringInputs::swizzled_storage`) swizzles the decoded
/// texels. A `BGR10_XR`, `BGRA10_XR` or two-plane form's (or `_sRGB` twin's) 2D sample through a
/// bicubic sampler, constexpr or a runtime one whose bound word states `Cubic`
/// (`LoweringInputs::cubic_samplers`), is Metal's bicubic filter of the decoded texels
/// (`extended`), under the bicubic filter's checks (`NotImplemented(SamplerBicubicFilter)`:
/// `bicubic::check`), and refused there for a bfloat or padded form, a gather, a 2D array, and a
/// swizzled texture (not measured). Every other sampling of the slot is `NotImplemented(Format)`:
/// a two-plane form's gather, or a padded form's `half` one, an LOD query, a comparison, a 1D or
/// cube texture, another form's 3D texture, an anisotropic or mixed filter, a linear mip filter of
/// another form or of a 3D texture, a constexpr sampler's level bias of its own or empty clamp,
/// and a dynamic offset.
fn select_extended(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    (index, form): (u32, crate::metal::format::ExtendedRange),
    sampler: SamplerArg,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    use super::bicubic::TapAddress;
    use super::extended::{Level, LevelAt, Sample, State};
    let Call { at, inst, t, args } = call;
    let ops = Operands { t, args };
    let shape = t.image;
    let refused = || -> Error { Unsupported::not_implemented(Feature::Format).into() };
    let access = cx
        .families
        .texture
        .slots
        .get(&index)
        .ok_or_else(|| cx.invariant("an unplanned texture slot"))?
        .access;
    // Metal samples only `access::sample` textures.
    if access != crate::air::abi::Access::Sample {
        return Err(refuse());
    }
    let fragment = cx.stage == Stage::Fragment;
    // Where a mip filter's level comes from: an explicit level or a kernel's bias (none, level 0),
    // or a fragment function's implicit level shifted by its bias, or explicit gradients.
    let (has_offset, gather, at_level) = match t.op {
        TexOp::Sample {
            compare: false,
            offset,
            lod: LodMode::Level,
        } => {
            let v = ops
                .get(TexRole::LodValue)
                .ok_or_else(|| cx.invariant("a level sample without a level"))?;
            (offset, None, LevelAt::Explicit(Some(id(cx, v)?)))
        }
        TexOp::Sample {
            compare: false,
            offset,
            lod: LodMode::Bias,
        } => {
            let bias = match ops.get(TexRole::LodValue) {
                Some(v) if !is_zero_float(cx, v) => Some(id(cx, v)?),
                _ => None,
            };
            let at = match fragment {
                true => LevelAt::Derived { bias, grad: None },
                false => LevelAt::Explicit(bias),
            };
            (offset, None, at)
        }
        TexOp::SampleGrad {
            compare: false,
            offset,
        } => {
            let (dx, dy) = match (ops.get(TexRole::Dpdx), ops.get(TexRole::Dpdy)) {
                (Some(dx), Some(dy)) => (id(cx, dx)?, id(cx, dy)?),
                _ => return Err(cx.invariant("a gradient sample without gradients")),
            };
            let at = LevelAt::Derived {
                bias: None,
                grad: Some((dx, dy)),
            };
            (offset, None, at)
        }
        // A bfloat texture's gather of a component, its taps' reads. So is an extended-range or
        // padded one's, whose `half` result is refused below.
        TexOp::Gather {
            component,
            compare: false,
            offset,
        } if !matches!(form, crate::metal::format::ExtendedRange::Planes { .. }) => {
            use crate::air::intrinsics::Component as C;
            let component = match component {
                None | Some(C::X) => 0,
                Some(C::Y) => 1,
                Some(C::Z) => 2,
                Some(C::W) => 3,
            };
            (offset, Some(component), LevelAt::Explicit(None))
        }
        _ => return Err(refused()),
    };
    // A 2D, 2D array or 3D texture's (a 3D one's samples alone: Metal has no 3D gather).
    let volume = shape.dim == ImageDim::D3;
    if !matches!(shape.dim, ImageDim::D2 | ImageDim::D3) || shape.ms || shape.depth {
        return Err(refused());
    }
    if clamp_value(cx, &ops)?.is_some() {
        return Err(refused());
    }
    let offset = match fetch_offset(cx, &ops, has_offset, plane(shape.dim))? {
        None => None,
        Some(Offset::Const(k)) => Some(k),
        Some(Offset::Dynamic(..)) => return Err(refused()),
    };
    let (texel_form, class) = texel(t, 4)?;
    let half = match texel_form.conv {
        TexelConv::F16 => true,
        TexelConv::F32 => false,
        _ => return Err(refused()),
    };
    // A `BGR10_XR` or `BGRA10_XR` form's (or `_sRGB` twin's) `half` gather rounds each tap as its
    // nearest sample does, with ties away from zero (every lane of 8,192 gathers of each form's 2D
    // and 2D array textures, an Apple M3 Max under macOS 27.0.1). A padded form's is not measured:
    // whether its taps round as its reads (to nearest even) or as its nearest samples.
    let measured = matches!(
        form,
        crate::metal::format::ExtendedRange::Bgr10 { .. }
            | crate::metal::format::ExtendedRange::Bgra10 { .. }
    );
    if gather.is_some() && half && !form.bfloat() && !measured {
        return Err(refused());
    }
    // A 3D texture's samples are measured for those forms alone (the XR mip probe: every lane of
    // 8,192 nearest and linear samples of each form's 3D texture of five levels under a nearest
    // mip filter, `float` and `half`).
    if volume && (shape.array || gather.is_some() || !measured) {
        return Err(refused());
    }
    // Metal's bicubic filter: a constexpr bicubic sampler's, or a runtime sampler's whose bound
    // word states it, under its slot's cubic-sampler variant (whose fact the function states).
    let cubic = match sampler {
        SamplerArg::Static(SamplerRef::Slot(n), _) => cubic_slot(cx, n)?,
        SamplerArg::Static(r, _) => super::bicubic::state(r)?,
        SamplerArg::Heap(_) => None,
    };
    let swizzle = crate::lower::texture::swizzled_storage(cx, ImageRef::Slot(index))?;
    let cubic = match cubic {
        None => None,
        // Measured on the six `BGR10_XR`, `BGRA10_XR` and two-plane forms' 2D samples alone.
        Some(_)
            if form.bfloat()
                || form.padded()
                || gather.is_some()
                || swizzle
                || shape.array
                || volume =>
        {
            return Err(bicubic_refused());
        }
        Some(s) => {
            super::bicubic::check(&s, shape.dim, (false, false, false), true)?;
            Some(super::bicubic::plan(&s, index, shape.dim, offset.clone())?)
        }
    };
    // The level a sample reads by its sampler's mip filter (`extended::Level`): a gather and a
    // bicubic sample read the base level, as Metal's do.
    let mut level = Level::Base;
    let state = match &cubic {
        // The bicubic filter's addressing, which places a constant offset (below).
        Some(b) => State::Constant {
            linear: false,
            address: b.address.clone(),
            border: b.border,
            pixel: b.pixel,
        },
        None => match sampler {
            SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
                let s = super::state_of(words)?;
                let linear = match (s.min_filter, s.mag_filter) {
                    (Filter::Linear, Filter::Linear) => true,
                    (Filter::Nearest, Filter::Nearest) => false,
                    _ => return Err(refused()),
                };
                // Metal's linear filter of a bfloat texture reads the nearest texel,
                // and a gather follows no filter.
                let linear = linear && !form.bfloat() && gather.is_none();
                if s.max_anisotropy > 1 {
                    return Err(refused());
                }
                if gather.is_none() && s.mip_filter != MipFilter::None {
                    let Some(stated) = super::extended::Stated::of(&s, form) else {
                        return Err(refused());
                    };
                    // A 3D texture's blend of two levels is not modelled (the XR mip probe's 3D
                    // samples under a linear mip filter fit no rule tried).
                    if volume && stated.linear {
                        return Err(refused());
                    }
                    // A level from derivatives or gradients is Vulkan's level of detail.
                    if matches!(at_level, LevelAt::Derived { .. }) {
                        cx.precision.insert(PrecisionNote::DeviceSampleLevel);
                    }
                    level = Level::Stated {
                        stated,
                        at: at_level,
                    };
                }
                let axes = if volume { 3 } else { 2 };
                let address = [s.address_s, s.address_t, s.address_r]
                    .into_iter()
                    .take(axes)
                    .map(|m| match m {
                        AddressMode::ClampToEdge => TapAddress::Edge,
                        AddressMode::Repeat => TapAddress::Repeat,
                        AddressMode::MirroredRepeat => TapAddress::Mirror,
                        AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
                    })
                    .collect();
                State::Constant {
                    linear,
                    address,
                    border: s.border_color,
                    pixel: s.coordinates == Coordinates::Pixel,
                }
            }
            // The function reads the word's filter, coordinates, address modes, and border color;
            // a sample its mip filter and the levels its clamp leaves, and under a linear mip
            // filter the slot's `SamplerMinLevel` specialization.
            SamplerArg::Static(SamplerRef::Slot(n), _) => {
                let Some(s) = cx.families.sampler.slots.get_mut(&n) else {
                    return Err(cx.invariant("an unplanned sampler slot"));
                };
                s.linear = true;
                if gather.is_none() {
                    s.fetch_levels = true;
                    // A 3D texture's linear mip filter is the binder's refusal.
                    s.min_level |= form.blends_levels() && !volume;
                    level = Level::Word {
                        sampler: n,
                        at: at_level,
                    };
                    // A level from derivatives or gradients is Vulkan's level of detail, where
                    // the word states a mip filter.
                    if matches!(at_level, LevelAt::Derived { .. }) {
                        let on = NoteOn::Sampler(n);
                        let when = any_of([NoteTest::differ(on, SamplerWord::MIP_FILTER, 0)]);
                        cx.note_when(PrecisionNote::DeviceSampleLevel, when, m)?;
                    }
                }
                State::Slot { sampler: n }
            }
            SamplerArg::Heap(_) => return Err(refused()),
        },
    };
    if crate::lower::texture::residency(cx, at, ImageRef::Slot(index))? {
        return Err(refused());
    }
    // A swizzled texture of the slot's variant: the sample swizzles its decoded texel, a border
    // texel included, and a gather of a component gathers the component its swizzle names, as
    // plain Metal does.
    if swizzle {
        crate::lower::texture::swizzles(cx, index)?;
    }
    // An offset on an axis that may repeat or mirror is placed by Metal's rule, which reads a
    // view's parent's base-level size: the slot's `OffsetBaseAxes` states where it differs.
    let offset_base = offset.as_deref().is_some_and(|o| {
        o.iter().enumerate().any(|(a, k)| {
            *k != 0
                && match &state {
                    State::Constant { address, .. } => address
                        .get(a)
                        .is_some_and(|m| matches!(m, TapAddress::Repeat | TapAddress::Mirror)),
                    State::Slot { .. } => true,
                }
        })
    });
    // The slot's raw texels are fetched, as a read fetches them.
    if let Some(s) = cx.families.texture.slots.get_mut(&index) {
        s.observe(shape, Some(class))?;
        s.uses.read = true;
        s.offset_base |= offset_base;
    }
    cx.families.arith.converts(form);
    let filters = !form.bfloat() && gather.is_none();
    if filters && !matches!(state, State::Constant { linear: false, .. }) {
        // A 2D linear filter of a form with grid steps sums its taps' steps; a 3D one, and a
        // padded form's, their decoded values.
        let steps = crate::lower::arith::extended_range::has_steps(form);
        let routine = match (volume, steps) {
            (true, _) => super::bicubic::routines::Routine::Block,
            (false, true) => super::bicubic::routines::Routine::QuadSteps,
            (false, false) => super::bicubic::routines::Routine::Quad,
        };
        cx.families.sampler.bicubic_routines.insert(routine);
    }
    // A bicubic sample takes the 2D routines, and a level below four texels on either axis the
    // linear ones; its sum near a rounding tie is not Metal's datapath's.
    if cubic.is_some() {
        cx.families.sampler.bicubic_routines.extend([
            super::bicubic::routines::Routine::Plane,
            super::bicubic::routines::Routine::Quad,
        ]);
        cx.precision.insert(PrecisionNote::BicubicFilterArithmetic);
    }
    // A linear mip filter's blend of two levels (not a 3D texture's).
    let blends = match level {
        Level::Base => false,
        Level::Stated { stated, .. } => stated.linear,
        Level::Word { .. } => form.blends_levels() && !volume,
    };
    if blends {
        cx.families
            .sampler
            .bicubic_routines
            .insert(super::bicubic::routines::Routine::Levels);
    }
    let coord = ops
        .get(TexRole::Coord)
        .ok_or_else(|| cx.invariant("a sampling call without a coordinate"))?;
    let layer = match shape.array {
        true => {
            let l = ops
                .get(TexRole::Layer)
                .ok_or_else(|| cx.invariant("an array sample without its layer"))?;
            Some(id(cx, l)?)
        }
        false => None,
    };
    let sample = Sample {
        texture: index,
        form,
        state,
        cubic,
        gather,
        swizzle,
        offset,
        offset_base,
        coord: id(cx, coord)?,
        layer,
        volume,
        level,
        half,
        result: cx.lowered_repr(Value::Inst(at), inst.ty)?,
        routine: cx
            .profile
            .fields()
            .workarounds
            .contains(&crate::device::profile::DriverWorkaround::DontInlineKeepsCalls),
    };
    // Its routine, counted once in the module (`routine::Extended`).
    if sample.routine {
        super::routine::note_extended(&mut cx.families.sampler, &sample);
    }
    Ok(TextureStrategy::ExtendedSample(Box::new(sample)))
}

/// Select sampling call `inst` (sample, gather, or LOD query) of `image` (a texture slot, or a
/// heap element) through sampler `sampler` (a static one, or an argument buffer's heap sampler),
/// decoded as `t`. `arm` is set for an arm of a switch-at-use: with `Some(true)`, a fragment
/// implicit-LOD sample reads the gradients its switch hoisted, so that the arms share one emulation
/// (SEMANTICS 1012, `Sampling::shared_shape`).
fn select_one(
    cx: &mut SelectCx<'_>,
    call: Call<'_>,
    image: ImageRef,
    sampler: SamplerArg,
    arm: Option<bool>,
    m: &mut Meter,
) -> Result<Sampling, Error> {
    let hoisted = arm == Some(true);
    let Call { at, inst, t, args } = call;
    let ops = Operands { t, args };
    let shape = t.image;
    // Metal samples only `access::sample` textures; a heap group of sampled members holds them.
    let sampled = match image {
        ImageRef::Slot(index) => {
            cx.families
                .texture
                .slots
                .get(&index)
                .ok_or_else(|| cx.invariant("an unplanned texture slot"))?
                .access
                == crate::air::abi::Access::Sample
        }
        ImageRef::Heap { group, .. } => {
            cx.families
                .texture
                .heap
                .get(&group)
                .ok_or_else(|| cx.invariant("an unplanned heap group"))?
                .access
                == ImageAccess::Sampled
        }
    };
    if !sampled {
        return Err(refuse());
    }
    // The static sampler, when it is one; an argument buffer's has no constexpr state.
    let stat = match sampler {
        SamplerArg::Static(r, c) => Some((r, c)),
        SamplerArg::Heap(_) => None,
    };
    if let Some(w) = ops.get(TexRole::DepthWord) {
        // The depth word's only observed value is 1; any other is not modelled.
        if !matches!(konst(cx, w), Some(Const::Int(1))) {
            return Err(refuse());
        }
    }
    if shape.dim == ImageDim::Cube && shape.array && !cx.profile.fields().images.cube_array {
        return Err(Unsupported::needs(
            Feature::CubeArray,
            Capability::Feature(VkFeature::ImageCubeArray),
        )
        .into());
    }
    let fragment = cx.stage == Stage::Fragment;
    let lanes = plane(shape.dim);
    // A nearest pixel sampler's sample is an exact fetch of the base level: its level, bias,
    // gradients, and minimum-LOD clamp select nothing (Metal's pixel samplers have no mip filter),
    // so it is not tangled. The texture's numeric type (an LOD query has no texel and never
    // fetches): that of the texture of another class the slot's variant binds, whose texels the
    // sampling takes in its own class (`texel_in`).
    let cross = crate::lower::texture::cross_class(cx, image)?;
    let integer = texel(t, 4).is_ok_and(|(_, class)| cross.unwrap_or(class) != TexelClass::Float);
    // Metal's bicubic filter: every operation through such a sampler passes its checks. A runtime
    // sampler's is its slot's under the cubic-sampler variant.
    let bicubic = match stat {
        Some((SamplerRef::Slot(n), _)) => cubic_slot(cx, n)?,
        Some((r, _)) => super::bicubic::state(r)?,
        None => None,
    };
    if let Some(b) = &bicubic {
        let sample = matches!(
            t.op,
            TexOp::Sample { compare: false, .. } | TexOp::SampleGrad { compare: false, .. }
        );
        let kind = (shape.depth, integer, image.heap());
        super::bicubic::check(b, shape.dim, kind, sample)?;
    }
    let fetch = match stat {
        Some((r, _)) => nearest_fetch(r, shape.dim, integer)?,
        None => false,
    };
    let (op, form, class, tangled) = match t.op {
        // A bicubic sample is fetches of the base level: its level, bias, gradients, and fragment
        // derivatives select nothing, so it is not tangled. A `min_lod_clamp` has no Metal
        // reference (Apple's device compiler fails to build the pipeline), and a dynamic offset is
        // not measured.
        TexOp::Sample {
            compare: false,
            offset: has_offset,
            ..
        }
        | TexOp::SampleGrad {
            compare: false,
            offset: has_offset,
        } if bicubic.is_some() => {
            let (Some(state), ImageRef::Slot(index)) = (&bicubic, image) else {
                return Err(bicubic_refused());
            };
            if clamp_value(cx, &ops)?.is_some() {
                return Err(bicubic_refused());
            }
            let offset = match fetch_offset(cx, &ops, has_offset, lanes)? {
                None => None,
                Some(Offset::Const(k)) => Some(k),
                Some(Offset::Dynamic(..)) => return Err(bicubic_refused()),
            };
            let b = super::bicubic::plan(state, index, shape.dim, offset)?;
            let (form, class) = texel_in(t, cross, 4)?;
            (SampleOp::Bicubic(b), Some(form), Some(class), false)
        }
        // An LOD query through a bicubic sampler is not measured. (A comparison never reaches
        // here: AIR compares only depth textures, which `bicubic::check` refuses.)
        TexOp::CalculateLod { .. } if bicubic.is_some() => {
            return Err(bicubic_refused());
        }
        TexOp::Sample {
            compare: false,
            offset: has_offset,
            ..
        }
        | TexOp::SampleGrad {
            compare: false,
            offset: has_offset,
        } if fetch => {
            let off = fetch_offset(cx, &ops, has_offset, lanes)?;
            let (form, class) = texel_in(t, cross, if shape.depth { 1 } else { 4 })?;
            (
                SampleOp::PixelFetch { offset: off },
                Some(form),
                Some(class),
                false,
            )
        }
        TexOp::Sample {
            compare,
            lod,
            offset: has_offset,
        } => {
            let clamp = clamp_value(cx, &ops)?;
            let value = ops.get(TexRole::LodValue);
            let lod = match lod {
                LodMode::Level => {
                    if clamp.is_some() {
                        // SPIR-V admits `MinLod` only with an implicit level or gradients.
                        return Err(Unsupported::not_implemented(Feature::MinLodClamp).into());
                    }
                    let v = value.ok_or_else(|| cx.invariant("a level sample without a level"))?;
                    Lod::Level(Some(id(cx, v)?))
                }
                // The gradients (and the bias) are the switch-at-use's, hoisted.
                LodMode::Bias if fragment && hoisted => Lod::Hoisted,
                // Metal applies the whole bias; past the device's limit, where Vulkan clamps a
                // `Bias` operand, the sample takes scaled derivatives.
                LodMode::Bias if fragment => {
                    let (bias, reach) = match value {
                        Some(v) if !is_zero_float(cx, v) => {
                            let reach = bias_reach(cx, v, sampler, lanes)?;
                            // Metal's bias is its own fixed-point reading, which Metal's sampler
                            // forms from the `Bias` operand at any bias: the module requires the
                            // device.
                            if reach == BiasReach::Metal {
                                cx.families.sampler.metal_sampler = true;
                            }
                            (Some(id(cx, v)?), reach)
                        }
                        _ => (None, BiasReach::Within),
                    };
                    if !matches!(reach, BiasReach::Within | BiasReach::Metal) {
                        cx.precision.insert(PrecisionNote::GradFromBias);
                    }
                    Lod::Implicit { bias, reach }
                }
                // Outside a fragment function a sample without a level or gradients reads Metal's
                // default level 0 (MSL 6.13), which its bias shifts and its clamp bounds below:
                // an explicit level.
                LodMode::Bias => {
                    let bias = match value {
                        Some(v) if !is_zero_float(cx, v) => Some(id(cx, v)?),
                        _ => None,
                    };
                    if bias.is_none() && clamp.is_none() {
                        Lod::Level(None)
                    } else {
                        Lod::Base { bias, min: clamp }
                    }
                }
            };
            // `MinLod` rides an implicit level or gradients only.
            let clamp = match lod {
                Lod::Implicit { .. } | Lod::Hoisted => min_lod(cx, clamp)?,
                _ => None,
            };
            let dref = compare_ref(cx, &ops, compare)?;
            let off = offset(cx, &ops, has_offset, lanes, false)?;
            // An armed offset's switch holds no implicit level: the sample takes the fragment's
            // derivatives in its own position as gradients, Metal's bias scaling them
            // (`emit::sample_operands`).
            if matches!(off, Some(Offset::Dynamic(_, Some(_))))
                && matches!(lod, Lod::Implicit { .. })
            {
                cx.precision.insert(PrecisionNote::GradFromBias);
            }
            let (form, class) = texel_in(t, cross, if shape.depth { 1 } else { 4 })?;
            let tangled = matches!(lod, Lod::Implicit { .. } | Lod::Hoisted);
            (
                SampleOp::Sample {
                    lod,
                    dref,
                    min_lod: clamp,
                    offset: off,
                },
                Some(form),
                Some(class),
                tangled,
            )
        }
        TexOp::SampleGrad {
            compare,
            offset: has_offset,
        } => {
            let clamp = min_lod(cx, clamp_value(cx, &ops)?)?;
            let (dx, dy) = match (ops.get(TexRole::Dpdx), ops.get(TexRole::Dpdy)) {
                (Some(dx), Some(dy)) => (id(cx, dx)?, id(cx, dy)?),
                _ => return Err(cx.invariant("a gradient sample without gradients")),
            };
            let dref = compare_ref(cx, &ops, compare)?;
            let off = offset(cx, &ops, has_offset, lanes, false)?;
            let (form, class) = texel_in(t, cross, if shape.depth { 1 } else { 4 })?;
            // A cube's gradients are sampled with Metal's footprint, a near-equivalent: an
            // isotropic sampler's as its length, an anisotropic one's as its face gradients, a
            // constexpr sampler's by its state and a runtime or argument buffer's by its word.
            // Where the driver hands the stage's gradients to Metal's `gradientcube` as given, the
            // function hands Metal's own, exactly; through MoltenVK's graphics fixup, the form
            // that fixup hands Metal on the stage's identity.
            let lod = if shape.dim == ImageDim::Cube {
                let anisotropic = match sampler {
                    SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
                        Some(super::state_of(words)?.max_anisotropy > 1)
                    }
                    SamplerArg::Static(SamplerRef::Slot(_), _) | SamplerArg::Heap(_) => None,
                };
                // A route the device's description does not state is Vulkan's rule: the device's
                // own reading of the form handed to it (`PrecisionNote::CubeGradientFootprint`).
                let route = cx.profile.fields().cube_gradients.route(cx.stage);
                let form = match (route, anisotropic) {
                    (CubeGradientRoute::Metal, _) => CubeFootprint::Metal,
                    (CubeGradientRoute::AppleProjected, _) => CubeFootprint::Projected,
                    (CubeGradientRoute::Vulkan, Some(false)) => CubeFootprint::Length,
                    (CubeGradientRoute::Vulkan, Some(true)) => CubeFootprint::Face,
                    (CubeGradientRoute::Vulkan, None) => CubeFootprint::Word,
                };
                Lod::CubeGrad(dx, dy, form)
            } else {
                Lod::Grad(dx, dy)
            };
            (
                SampleOp::Sample {
                    lod,
                    dref,
                    min_lod: clamp,
                    offset: off,
                },
                Some(form),
                Some(class),
                false,
            )
        }
        TexOp::Gather {
            component,
            compare,
            offset: has_offset,
        } => {
            use crate::air::intrinsics::Component as C;
            let component = match component {
                None | Some(C::X) => 0,
                Some(C::Y) => 1,
                Some(C::Z) => 2,
                Some(C::W) => 3,
            };
            let dref = compare_ref(cx, &ops, compare)?;
            let off = offset(cx, &ops, has_offset, lanes, true)?;
            let (form, class) = texel_in(t, cross, 4)?;
            (
                SampleOp::Gather {
                    component,
                    dref,
                    offset: off,
                },
                Some(form),
                Some(class),
                false,
            )
        }
        TexOp::CalculateLod { clamped } if fragment => {
            (SampleOp::Lod { clamped }, None, None, true)
        }
        // MSL leaves `calculate_*_lod` undefined outside a fragment function (MSL 6.13); Metal's
        // answer was measured, and is the query's own level.
        TexOp::CalculateLod { clamped } => {
            let min = if clamped {
                Some(query_min_level(cx, shape, integer, stat)?)
            } else {
                query_level_measured(cx, shape, integer)?;
                None
            };
            (SampleOp::QueryLevel { min }, None, None, false)
        }
        _ => return Err(cx.invariant("a non-sampling texture operation routed to sampling")),
    };
    let mut op = op;
    let coord = ops
        .get(TexRole::Coord)
        .ok_or_else(|| cx.invariant("a sampling call without a coordinate"))?;
    let layer = match ops.get(TexRole::Layer) {
        Some(v) => Some(id(cx, v)?),
        None => None,
    };
    let compare = matches!(
        op,
        SampleOp::Sample { dref: Some(_), .. } | SampleOp::Gather { dref: Some(_), .. }
    );
    // Pixel coordinates are scaled in the shader: always for a constexpr pixel sampler, and behind
    // the slot's `SamplerCoordinates` specialization for a runtime sampler. Metal reads a cube's
    // coordinate as a direction through a pixel sampler too, as through a normalized one, so a
    // cube's sampler is never scaled.
    let cube = shape.dim == ImageDim::Cube;
    // A cube's level of detail from the fragment's derivatives (an implicit-level sample, a
    // hoisted one, or an LOD query) is the device's, taken of the direction with its ties broken
    // as Metal breaks them: a near-equivalent, as an explicit gradient's is.
    if cube && tangled {
        cx.precision.insert(PrecisionNote::CubeGradientFootprint);
    }
    // An argument buffer's sampler is scaled behind its handle's pixel-coordinate flag.
    let pixel = match sampler {
        _ if matches!(
            op,
            SampleOp::PixelFetch { .. } | SampleOp::Bicubic(_) | SampleOp::QueryLevel { .. }
        ) =>
        {
            Pixel::None
        }
        SamplerArg::Heap(h) if !cube => Pixel::Heap(h),
        SamplerArg::Static(SamplerRef::Slot(n), _) if !cube => Pixel::Spec(n),
        SamplerArg::Static(SamplerRef::Constexpr(_), Coordinates::Pixel) if !cube => Pixel::Always,
        _ => Pixel::None,
    };
    // A `half` texel's NaN is the device's decode, a bicubic sample's taps included: where the
    // bound format may hold one.
    let half = form.is_some_and(|f| f.conv == TexelConv::F16);
    if half {
        crate::lower::texture::half_texel_nans(cx, image, shape, m)?;
    }
    // A declared result whose answer from a texture of another class is not measured.
    if let Ok((declared, _)) = texel(t, if shape.depth { 1 } else { 4 }) {
        crate::lower::texture::mark_result(cx, image, declared);
    }
    // The texture's use: its one shape and texel class, sampled, with comparison.
    let textures = &mut cx.families.texture;
    match image {
        ImageRef::Slot(index) => {
            if let Some(s) = textures.slots.get_mut(&index) {
                s.observe(shape, class)?;
                s.uses.sample = true;
                s.uses.compare |= compare;
            }
        }
        ImageRef::Heap { group, .. } => {
            if let Some(h) = textures.heap.get_mut(&group) {
                h.observe(shape, class)?;
                h.uses.sample = true;
                h.uses.compare |= compare;
            }
        }
    }
    if let SamplerArg::Static(SamplerRef::Slot(n), _) = sampler {
        let Some(s) = cx.families.sampler.slots.get_mut(&n) else {
            return Err(Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "an unplanned sampler slot",
            )));
        };
        s.compare |= compare;
        s.coordinates |= pixel != Pixel::None;
        // A cube's gradients whose footprint the word chooses read its anisotropy.
        s.anisotropy |= matches!(
            op,
            SampleOp::Sample {
                lod: Lod::CubeGrad(_, _, CubeFootprint::Word),
                ..
            }
        );
    }
    if let SampleOp::Bicubic(b) = &op {
        cx.families.sampler.bicubic.insert(b.texture);
        // A 2D sample's level below four texels on an axis takes the linear 2D routines.
        let routines: &[_] = if shape.dim == ImageDim::D1 {
            &[super::bicubic::routines::Routine::Line]
        } else {
            &[
                super::bicubic::routines::Routine::Plane,
                super::bicubic::routines::Routine::Quad,
            ]
        };
        cx.families
            .sampler
            .bicubic_routines
            .extend(routines.iter().copied());
        cx.precision.insert(PrecisionNote::BicubicFilterArithmetic);
    }
    // An observed residency; a bicubic sample's fetches have no sparse form, so its slot takes
    // none, and an LOD query has no status word.
    let observed = !matches!(op, SampleOp::Lod { .. } | SampleOp::QueryLevel { .. })
        && crate::lower::texture::residency(cx, at, image)?;
    let fetches = matches!(op, SampleOp::Bicubic(_));
    let sparse_images = cx
        .profile
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderResourceResidency);
    // Metal's footprint where its measured rule decides it, a constexpr sampler's, on a device
    // without residency code of its own (no `shaderResourceResidency`), where the slot's tile map
    // answers the residency. A device with it answers a sample's or gather's residency itself,
    // its `OpImageSparse*` word (`PrecisionNote::SparseResidency`).
    let footprint = match (observed, image, sampler) {
        (true, ImageRef::Slot(_), SamplerArg::Static(SamplerRef::Constexpr(words), _))
            if !sparse_images =>
        {
            let state = super::constexpr::state_of(words)?;
            super::footprint::of(&state, &op, shape, integer)
        }
        _ => None,
    };
    // A tile map answers a pixel sampler's fetch, whose texel the shader names, and a sample or
    // gather with Metal's footprint; every other sample's or gather's footprint is the device
    // filter's.
    let named = matches!(op, SampleOp::PixelFetch { .. }) || footprint.is_some();
    // Where the profile enables `shaderResourceResidency`, a sparse texture is a sparse image: a
    // sample's, gather's, or pixel fetch's texels in a block of it that holds no mapped tile read
    // the device's non-resident value under either arm (zero only under
    // `residencyNonResidentStrict`, and RADV's zero alpha where Metal reads a missing alpha as 1),
    // and a sample's or gather's footprint and word on the sparse arm are the device's. There
    // every access of a slot's color texture sets the lanes Metal reads as constants (a channel the
    // format lacks, through the swizzle), whether the function reads its residency or not, and a
    // pixel fetch whose residency it reads zeroes its texel where its residency says it is
    // unmapped, Metal's value; a sample's or gather's lanes that are no constants, and its word,
    // stay the device's. A bicubic sample's fetches, and a level query, take neither. Elsewhere the
    // texture's image holds Metal's unmapped value and its tile map answers exactly the texels
    // Metal counts.
    let texel_op = !matches!(op, SampleOp::Lod { .. } | SampleOp::QueryLevel { .. });
    let unmapped = match image {
        ImageRef::Slot(index) if sparse_images && texel_op && !fetches && !shape.depth => {
            let named_texels = matches!(op, SampleOp::PixelFetch { .. });
            if let Some(s) = cx.families.texture.slots.get_mut(&index) {
                s.constant_lanes = true;
                let slot = s.slot;
                cx.facts
                    .insert(crate::pipeline::facts::RuntimeFact::ConstantLanes { slot });
            }
            if observed && named_texels {
                Unmapped::Named
            } else {
                Unmapped::Constants
            }
        }
        _ => Unmapped::Device,
    };
    if observed && !fetches && sparse_images && unmapped != Unmapped::Named {
        cx.precision.insert(PrecisionNote::SparseResidency);
    }
    if let (true, ImageRef::Slot(index)) = (observed, image) {
        if let Some(s) = cx.families.texture.slots.get_mut(&index) {
            s.uses.plain_residency |= fetches;
            s.uses.untiled_residency |= !named;
        }
    }
    let residency = observed && !fetches;
    // Every sample, gather and level query goes to the device's sampler: Metal's own on MoltenVK
    // (`VkProperty::MetalSampler`), whose decode, filters, levels and comparisons are Metal's, and
    // the device's own elsewhere (`VkProperty::DeviceSampling`), whose answers the module records
    // (a level query's level clamped by the device into the sampler's clamp, the descriptor's).
    if matches!(
        op,
        SampleOp::Sample { .. } | SampleOp::Gather { .. } | SampleOp::Lod { .. }
    ) {
        cx.families.sampler.device_sampling = true;
    }
    // An 8-bit sRGB texture the device reads through its own sRGB view, but under the slot's twin
    // variant (a bound format the device cannot sample, held as its UNORM twin), whose view reads
    // the codes: what the device reads (one texel's codes, its filter of them, a gather's, a
    // bicubic sample's taps) is decoded by the device's transfer (`raw`). An LOD query reads no
    // texel.
    let raw = match form {
        Some(f) if matches!(f.conv, TexelConv::F16 | TexelConv::F32) => {
            crate::lower::texture::srgb_texels(cx, image, shape, m)?
        }
        _ => false,
    };
    // A filtered sample's note, where Metal's own sampler filters the slot and a two-plane YCbCr
    // format bound there takes another filter than Metal's: the clauses test the bound format
    // alone, which the runtime reads at each draw, whatever a runtime sampler's word states (no key
    // carries it there).
    let by = stat.map(|(r, _)| r);
    if let Some((note, when)) = super::linear::filter_note(cx, &op, form, (by, image))? {
        cx.note_when(note, when, m)?;
    }
    // A pixel sampler's coordinate reaches the device's sampler over the level-0 size, whose
    // texels at a texel edge or a filter step's edge are the device's arithmetic's on that
    // coordinate (`pixel_note`). An argument buffer's sampler records a note of its own, which the
    // runtime announces only while a bound state can need it.
    let when = pixel_note(pixel);
    let note = match pixel {
        Pixel::Heap(_) => PrecisionNote::ArgumentPixelSamplerScaling,
        _ => PrecisionNote::PixelSamplerScaling,
    };
    cx.note_when(note, when, m)?;
    if let SampleOp::Bicubic(b) = &mut op {
        b.raw = raw;
    }
    // Whether the access reads one texel of one level, whose `half` or `float` texel a packed or
    // normalized class converts from its codes; a runtime sampler's word decides it, which the
    // slot's specialization then reads.
    let one_texel = one_texel(&op, sampler, shape.dim == ImageDim::Cube)?;
    // On a device that samples through Metal's own texture unit, the `half` routine reads an
    // access's lanes only for a YCbCr field's one texel (`texture::half`): no sampler word decides
    // them. A slot's texture of a YCbCr format on the grid `8192` (`TexelFormat::YCBCR_8192`) then
    // rounds a sample of its one texel to nearest even, where Metal rounds it with ties away,
    // noted where it is bound.
    let one_texel = match one_texel {
        crate::lower::texture::OneTexel::Word(_)
            if half && !raw && !crate::lower::texture::device_texels(cx) =>
        {
            if let ImageRef::Slot(index) = image {
                use crate::metal::texel_format::TexelFormat;
                let ties = NoteTest::equal(
                    NoteOn::Bound(index),
                    TexelFormat::YCBCR_8192,
                    TexelFormat::YCBCR_8192,
                );
                cx.note_when(PrecisionNote::HalfTexelConversion, vec![[ties].into()], m)?;
            }
            crate::lower::texture::OneTexel::Never
        }
        other => other,
    };
    // A bicubic sample rounds its `half` result exactly, keeping denormals, and a raw texel's
    // decode rounds its own; another `half` access converts its texel: the device's where the device's texel is
    // the answer, else by the bound format's class (a heap color texture's by the class its
    // handle carries: the module's routine, called with the class read from the handle).
    let device = crate::lower::texture::device_texels(cx);
    if !raw && half && !matches!(op, SampleOp::Bicubic(_)) {
        if image.heap() && !shape.depth && !device {
            cx.families.texture.heap_half = true;
            cx.families.access.heap_class = true;
        }
        let sample = !matches!(op, SampleOp::Gather { .. });
        crate::lower::texture::half_texels(cx, image, shape, sample, device, m)?
    }
    // A `float` texel that may be one texel's is the device's conversion of its codes: not an sRGB
    // raw texel's (decoded by the device's transfer, `SrgbTexelDecode`), a bicubic sample's, or an LOD query's; nor a depth
    // comparison's, whose result is no texel.
    let comparison = matches!(
        op,
        SampleOp::Sample { dref: Some(_), .. } | SampleOp::Gather { dref: Some(_), .. }
    );
    if form.is_some_and(|f| f.conv == TexelConv::F32)
        && !raw
        && !comparison
        && !matches!(
            op,
            SampleOp::Bicubic(_) | SampleOp::Lod { .. } | SampleOp::QueryLevel { .. }
        )
    {
        crate::lower::texture::float_texel_notes(cx, image, shape, one_texel);
    }
    // The conversion's lanes read a runtime sampler's filter fields where its word decides: a
    // `half` texel's on every profile.
    let half_converts = half && !raw && !matches!(op, SampleOp::Bicubic(_)) && !device;
    if let (crate::lower::texture::OneTexel::Word(n), true, ImageRef::Slot(_)) =
        (one_texel, half_converts, image)
    {
        if !shape.depth {
            if let Some(s) = cx.families.sampler.slots.get_mut(&n) {
                s.texels = true;
            }
        }
    }
    let source = match (image, sampler) {
        (ImageRef::Slot(texture), SamplerArg::Static(sampler, _)) => {
            let key = BindingKey {
                texture,
                sampler,
                compare,
            };
            cx.families.sampler.combined.insert(key);
            Source::Combined(key)
        }
        (ImageRef::Heap { group, handle }, SamplerArg::Static(sampler, _)) => {
            let key = HeapKey {
                group,
                sampler,
                compare,
            };
            cx.families.sampler.heap.insert(key);
            Source::Heap { handle, key }
        }
        // An argument buffer's sampler is read from the sampler array of its use.
        (ImageRef::Heap { group, handle }, SamplerArg::Heap(sampler)) => {
            let key = HeapSamplerKey { group, compare };
            cx.families.sampler.heap_samplers.insert(key);
            cubic_reach(cx);
            Source::HeapSampler {
                handle,
                sampler,
                key,
            }
        }
        (ImageRef::Slot(_), SamplerArg::Heap(_)) => {
            return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into())
        }
    };
    // A depth comparison, the device's, reads the texture's depth image, which a device without
    // `VK_EXT_depth_range_unrestricted` may hold clamped.
    if shape.depth && compare {
        crate::lower::texture::depth_comparison(cx);
    }
    // A texel offset the module places by Metal's rule on an axis that may repeat or mirror
    // reads the base-level size of a view's parent: a slot's access through the slot's
    // `OffsetBaseAxes` specialization, which states where the parent's and the bound view's
    // differ; a heap element's reads the bound view's alone (the residue). A bicubic filter's
    // footprint is the one the module places so.
    let offset_base = places_wrapped_offset(&op);
    let offset_base = match image {
        ImageRef::Slot(index) if offset_base => {
            if let Some(s) = cx.families.texture.slots.get_mut(&index) {
                s.offset_base = true;
            }
            true
        }
        ImageRef::Heap { .. } if offset_base => {
            cx.precision.insert(PrecisionNote::ViewBaseTexelOffset);
            false
        }
        _ => false,
    };
    // A packed normalized sampled slot of the variant (`LoweringInputs::swizzled_storage`, on a
    // device without Metal's own sampler), whose view takes no swizzle: the sampling swizzles
    // the texel its class converts, and a gather of a component gathers the component the
    // swizzle names, as plain Metal does.
    let swizzle = match image {
        ImageRef::Slot(index)
            if texel_op && !shape.depth && crate::lower::texture::swizzled_storage(cx, image)? =>
        {
            if form.is_none_or(|f| f.lanes != 4 || matches!(f.conv, TexelConv::U64Words { .. })) {
                return Err(Unsupported::not_implemented(Feature::TextureSwizzle).into());
            }
            crate::lower::texture::swizzles(cx, index)?;
            true
        }
        _ => false,
    };
    // Metal's anisotropic filter emulated by the device's samples at explicit levels, where it
    // reaches the sample; the device's anisotropic filter, recorded, where it does not.
    let anisotropic = super::anisotropic::select(
        cx,
        &op,
        (stat.map(|(r, _)| r), image, shape),
        (integer, observed, cross.is_some(), swizzle),
        SamplerUse::new(compare, integer),
        m,
    )?;
    // A cube's explicit gradients handed to the device in a form other than Metal's own: the
    // footprint's note, but where a constexpr sampler's emulation answers every word of the
    // sample (a runtime sampler's answers only the words its variant names).
    if let SampleOp::Sample {
        lod: Lod::CubeGrad(_, _, form),
        ..
    } = &op
    {
        let emulated =
            anisotropic.is_some_and(|a| matches!(a.degree, super::anisotropic::Degree::Const(_)));
        if *form != CubeFootprint::Metal && !emulated {
            cx.precision.insert(PrecisionNote::CubeGradientFootprint);
        }
    }
    let result = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    // A texture of another class: its own result, handed back as the declared one.
    let (result, declared) = match (cross, form) {
        (Some(_), Some(native)) => {
            let conv = texel(t, native.lanes)?.0.conv;
            let own = super::native_result(&result, native.conv)
                .ok_or_else(|| cx.invariant("a sampling result without a texel"))?;
            (own, Some(super::DeclaredResult { conv, result }))
        }
        _ => (result, None),
    };
    let value = id(cx, Value::Inst(at))?;
    if tangled {
        cx.tangled.insert(value);
    }
    let clamp_reference = clamps_reference(cx, comparison, sampler)?;
    let sampling = Sampling {
        source,
        op,
        coord: id(cx, coord)?,
        lanes,
        layer,
        texel: form,
        depth: shape.depth,
        result,
        pixel,
        residency,
        footprint,
        cube,
        one_texel,
        half_texel: half_converts && !shape.depth,
        unmapped,
        heap_raw: None,
        declared,
        swizzle,
        offset_base,
        anisotropic,
        raw,
        clamp_reference,
    };
    Ok(sampling)
}

/// Whether a fragment function's depth comparison (where `compares`, the device's or the
/// shader's) through `sampler` clamps its reference, as the pipeline's private
/// `fragmentDepthCompareClampMask` says (`LoweringInputs::depth_compare_clamp_mask`; probes on an
/// Apple M3 Max, macOS 27.0.1): bit `s` clamps a
/// comparison through `[[sampler(s)]]`, whichever texture slot it compares, and bit 31 one
/// through any constexpr sampler; the reference is clamped to `[0, 1]`, a NaN taken as 0, for
/// `sample_compare` and `gather_compare` alike. Each comparison's sampler is recorded
/// (`StageIo::depth_compares`); one through an argument buffer's sampler, which no bit names, is
/// refused under a mask other than 0.
fn clamps_reference(
    cx: &mut SelectCx<'_>,
    compares: bool,
    sampler: SamplerArg,
) -> Result<bool, Error> {
    use crate::pipeline::stage_io::DepthCompares;
    if cx.stage != Stage::Fragment || !compares {
        return Ok(false);
    }
    let bit = match sampler {
        SamplerArg::Static(SamplerRef::Slot(n), _) if n < 16 => 1u32 << n,
        SamplerArg::Static(SamplerRef::Slot(_), _) => {
            return Err(cx.invariant("a sampler slot past Metal's 16"))
        }
        SamplerArg::Static(SamplerRef::Constexpr(_), _) => DepthCompares::CONSTEXPR,
        SamplerArg::Heap(_) => DepthCompares::ARGUMENT,
    };
    let mask = cx.inputs.depth_compare_clamp_mask;
    if bit == DepthCompares::ARGUMENT && mask != 0 {
        return Err(Unsupported::not_implemented(Feature::PrivateRenderState(
            crate::error::PrivateRenderField::FragmentDepthCompareClamp,
        ))
        .into());
    }
    let compares = &mut cx.families.stage_io.interface.depth_compares;
    compares.samplers |= bit;
    let clamps = mask & bit != 0;
    if clamps {
        compares.clamped |= bit;
    }
    Ok(clamps)
}

/// Where a sampling through a sampler of pixel coordinates `pixel` hands the device a coordinate
/// scaled by the texture's level-0 size, whose texels the device's sampler decides
/// (`PrecisionNote::PixelSamplerScaling`): the clauses of the arms that do (empty: none).
///
/// - A constexpr pixel sampler: always (its exact fetch, `SampleOp::PixelFetch`, reaches here with
///   [`Pixel::None`]).
/// - A runtime sampler slot: where its word states pixel coordinates.
/// - An argument buffer's sampler ([`PrecisionNote::ArgumentPixelSamplerScaling`]): always, since
///   its word, which the specialized module cannot read, may state pixel coordinates; the runtime
///   announces it only while a state of pixel coordinates lives.
fn pixel_note(pixel: Pixel) -> Vec<crate::lower::select::NoteClause> {
    use crate::lower::select::NoteClause;
    match pixel {
        Pixel::None => Vec::new(),
        Pixel::Always | Pixel::Heap(_) => vec![NoteClause::new()],
        Pixel::Spec(n) => {
            let coordinates =
                NoteTest::differ(NoteOn::Sampler(n), SamplerWord::PIXEL_COORDINATES, 0);
            vec![[coordinates].into()]
        }
    }
}

/// Whether sampling `op` through `sampler` reads one texel of one level: a 2D gather and a pixel
/// fetch always, a sample through a constexpr sampler whose state reads one (`linear::unfiltered`),
/// a runtime or argument buffer's sampler's where its word states one; a cube gather (`cube`: its
/// corner texels are made up), a bicubic sample, and an LOD query never.
fn one_texel(
    op: &SampleOp,
    sampler: SamplerArg,
    cube: bool,
) -> Result<crate::lower::texture::OneTexel, Error> {
    use crate::lower::texture::OneTexel;
    Ok(match op {
        SampleOp::Gather { .. } if cube => OneTexel::Never,
        SampleOp::Gather { .. } | SampleOp::PixelFetch { .. } => OneTexel::Always,
        SampleOp::Sample { .. } => match sampler {
            SamplerArg::Static(SamplerRef::Constexpr(words), _) => {
                match super::linear::unfiltered(&super::state_of(words)?) {
                    true => OneTexel::Always,
                    false => OneTexel::Never,
                }
            }
            SamplerArg::Static(SamplerRef::Slot(n), _) => OneTexel::Word(n),
            SamplerArg::Heap(h) => OneTexel::Heap(h),
        },
        SampleOp::Bicubic(_) | SampleOp::Lod { .. } | SampleOp::QueryLevel { .. } => {
            OneTexel::Never
        }
    })
}

/// Whether the module places a texel offset of `op` by Metal's rule (`position::offset_dropped`) on
/// an axis that may repeat or mirror: a bicubic filter's footprint, on a constexpr sampler's
/// repeating or mirrored axis.
fn places_wrapped_offset(op: &SampleOp) -> bool {
    use super::bicubic::TapAddress;
    let wraps = |a: &TapAddress| matches!(a, TapAddress::Repeat | TapAddress::Mirror);
    let moved = |o: &[i32]| o.iter().any(|o| *o != 0);
    match op {
        SampleOp::Bicubic(b) => {
            b.offset.as_deref().is_some_and(moved) && b.address.iter().any(wraps)
        }
        _ => false,
    }
}

/// An operation through an argument buffer's sampler tests the sampler's table entry for a cubic
/// filter where the stage can mark the command buffer's `ExecutionStatus`
/// (`RuntimeFact::CubicArgumentSampler`): Metal samples through a cubic sampler with its bicubic
/// filter, which the operation does not emulate on a heap texture, so where it reaches one it marks
/// the buffer and its command buffer fails; a run that never reaches one is Metal's. Where the
/// stage cannot mark it, the planner refuses a listed cubic sampler beside the stage. A fragment
/// stage never marks it: the mark is a store, and a fragment function holding one loses the
/// device's hidden-surface removal and early depth test although the store never runs (an M4: 16
/// hidden layers cost 13.5 times what they cost without it), so the planner refuses a listed cubic
/// sampler beside a fragment stage.
fn cubic_reach(cx: &mut SelectCx<'_>) {
    if cx.stage == Stage::Fragment {
        return;
    }
    if cx.families.sampler.cubic_reach || crate::lower::access::absent::status_writable(cx) {
        cx.families.sampler.cubic_reach = true;
        cx.facts
            .insert(crate::pipeline::facts::RuntimeFact::CubicArgumentSampler { stage: cx.stage });
    }
}

/// The slot of runtime sampler parameter index `index` of the stage.
/// The bicubic state of the runtime sampler at slot `n` where the compile's cubic-sampler variant
/// names the slot (`LoweringInputs::cubic_samplers`), else `None`. Every texture operation
/// through a runtime sampler slot records `RuntimeFact::CubicSampler`, so the binder asks for the
/// variant wherever the bound sampler's min or mag filter is `Cubic`: Metal samples through such
/// a sampler with its bicubic filter, which the variant emulates as a constexpr
/// bicubic sampler's (`bicubic::cubic_state`).
fn cubic_slot(
    cx: &mut SelectCx<'_>,
    n: u32,
) -> Result<Option<crate::air::abi::SamplerState>, Error> {
    let slot = sampler_slot(cx.stage, n);
    cx.facts
        .insert(crate::pipeline::facts::RuntimeFact::CubicSampler { slot });
    cx.inputs
        .cubic_samplers
        .get(&slot)
        .map(|w| super::bicubic::cubic_state(*w))
        .transpose()
}

pub(super) fn sampler_slot(stage: Stage, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class: SlotClass::Sampler,
        index,
    }
}
