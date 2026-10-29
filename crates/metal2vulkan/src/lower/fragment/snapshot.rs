//! Reads of a texture through a snapshot (`LoweringInputs::snapshot_textures`).
//!
//! A draw that reads a texture `T` its own subpass renders, at the fragment's own pixel alone
//! (`StageInterface::snapshot_textures`), runs twice with one pipeline (`PushField::SnapshotPhase`,
//! the binder's `bind::graphics`). The first draw, every color write mask 0 and the phase 1, copies
//! each non-helper fragment's texel of `T` into the slot's snapshot `S` (`ResourceKind::
//! SnapshotTexels`) and returns; the second, the pipeline's own masks and the phase 0, copies
//! nothing and runs the function, whose every read of `T` reads `S`, or nothing in a helper
//! invocation. The entry opens, before the body:
//!
//! ```text
//! helper ← this invocation is a helper
//! OpBeginInvocationInterlockEXT
//! if phase ≠ 0 && !helper { for each slot: S[p] ← bits(fetch T[p], level 0) }   p = uint(FragCoord.xy)
//! OpEndInvocationInterlockEXT
//! if phase ≠ 0 { return }
//! ```
//!
//! **Why one draw is not enough.** The function's read of `T[p]` must happen before the same
//! draw's color write of `T[p]`. Vulkan states only an execution order there: "Most operations in
//! the fragment shader are not performed in rasterization order, with exceptions called out in the
//! following sections" (Fragment Shading), and blending and color writes execute in rasterization
//! order (Rasterization Order). Its memory model relates a shader instruction to another only
//! through program order, the interlock's synchronizes-with ("If A is an
//! OpEndInvocationInterlockEXT instruction and B is an OpBeginInvocationInterlockEXT instruction,
//! then A synchronizes-with B if the agent that executes A is ordered before the agent that executes
//! B"), and execution dependencies ("If there is an execution dependency between two operations A
//! and B, then the operation in the first synchronization scope system-synchronizes-with the
//! operation in the second synchronization scope"), none of which reaches the fixed-function color
//! write. So no single-draw form, interlocked or not, orders the read before the write; a form that
//! "simplifies" this back into one draw relies on behaviour the specification does not state.
//!
//! **Why two draws are exact.** Each link rests on a sentence of the specification:
//!
//! - *The copy reads `T` as the draws before left it.* The consumer records a by-region barrier
//!   from the color attachment writes to fragment-shader reads before the first draw, with `T`'s
//!   attachment in the feedback-loop layout: "A vkCmdPipelineBarrier inside a subpass can
//!   synchronize prior attachment writes in the subpass with subsequent non-attachment reads if
//!   feedback loop is enabled for the attachment", "In that case the overlap is defined to occur at
//!   a per-pixel granularity", and each region "covers a single (x,y,layer) position". The copy
//!   reads `T` at its own pixel alone.
//! - *Nothing writes `T` during the first draw.* Every color write mask is 0: "VK_COLOR_COMPONENT_R_BIT
//!   specifies that the R value is written to the color attachment for the appropriate sample.
//!   Otherwise, the value in memory is unmodified" (Color Write Mask), and likewise for G, B and A.
//!   The function writes no memory (the interface fact), and the first draw runs nothing of it past
//!   the copy, so the outputs it leaves unwritten reach nothing.
//! - *The two draws shade the same fragments.* They differ only in the write masks, dynamic state of
//!   one pipeline: "Rule 2 Changes to the following state values have no side effects ... Write
//!   masks (color, depth, stencil)", and "Corollary 1 Fragment generation is invariant with respect
//!   to the state values listed in Rule 2" (Invariance). A second pipeline would be another state
//!   vector, which those rules do not cover; the side-effect relaxation lifts only "rule 1,
//!   corollary 2, and rule 4". Each fragment of the first draw runs the copy: "Otherwise, at least
//!   one fragment shader must be executed" (Fragment Shading), which covers each pixel the second
//!   draw's fragments read; no fragment operation before shading discards one (no depth or stencil
//!   attachment, and the same scissor).
//! - *Fragments that cover one pixel in the first draw write `S[p]` in order.* The entry declares
//!   `PixelInterlockOrderedEXT`: "any interlocked operations in a fragment shader must happen before
//!   interlocked operations in fragment shader invocations that execute later in rasterization order
//!   and cover at least one sample in the same pixel", the End/Begin synchronizes-with above, and
//!   `S` is decorated `Coherent`, whose per-instruction availability and visibility the interlock
//!   needs ("shaders using fragment shader interlock will declare the relevant resources as
//!   coherent", Memory Model). Every such write stores the same bits, since `T` does not change.
//! - *The second draw reads `S` after the first wrote it.* The consumer records a by-region barrier
//!   from the first draw's fragment-shader writes to the second draw's fragment-shader reads, which
//!   also orders the copy's read of `T` before the second draw's color writes.
//! - *No other access races.* "Data races can only occur due to instructions that are actually
//!   executed" (Memory Model): the second draw executes no copy, and the first no read of `S`. A
//!   helper invocation copies nothing and reads nothing; its codes of the slot are 0, which reaches
//!   no fragment's result, since no value read from the slot reaches a derivative, a quad or
//!   subgroup operation or a branch, nor an integer divisor or a dynamic component index, whose
//!   behavior SPIR-V leaves undefined at some values (`spirv::requirements::snapshot_textures`):
//!   "Stores and atomics performed by helper invocations must not have any effect on memory except
//!   for the Function, Private and Output storage classes" (Helper Invocations).
//!
//! **The snapshot's form** (`SnapshotForm`, the binder's choice by the slot's bound view): the
//! copy stores the texel's codes, which it takes from the fetched value, and a read decodes them.
//!
//! - `Bits`: the fetched `vec4`'s 32 bits a lane in an `R32G32B32A32_UINT` image, which come back
//!   bit for bit, whatever the texel.
//! - A normalized form: each lane's code `c = RoundEven(clamp(v) * D)` (`D = 2^b - 1`, a signed
//!   form's `2^(b-1) - 1` and its clamp `[-1, 1]`), in an `R8G8B8A8_UINT` or `R16G16B16A16_UINT`
//!   image, two's complement where signed. The fetched value of a normalized code `c` is "f =
//!   c / (2^b - 1)" (a signed one's "f = max(c / (2^(b-1) - 1), -1.0)", Fixed-Point Data
//!   Conversions), so `v * D` lies within far less than half of one of `c`, and "OpFMul ...
//!   Correctly rounded" with `RoundEven` exact (Precision and Operation of SPIR-V Instructions)
//!   give `c` back; a signed form's `-2^(b-1)`, whose value is `-1` as the code above it's is, comes
//!   back as that code. A read decodes `c` by the device's arithmetic, `c` times the `f32` nearest
//!   `1 / D` (a signed code's at least `-1`), within an `f32` rounding or two of `c / D`
//!   (`PrecisionNote::DevicePackConversion`; Metal reads `c / D` correctly rounded).
//! - `Half`: each lane's half-float code (`narrow::to_bits`: the device's narrowing, every half
//!   value's bits but a NaN's and, where the device may flush a 16-bit denormal, a denormal's,
//!   each recorded, `narrow::encodes_bits`; a code loaded unchanged keeps its bits), in an
//!   `R16G16B16A16_UINT` image; a read widens it back as a `half` read
//!   from memory widens (`narrow::from_bits`): the device's widening on every device, every code's
//!   value but a NaN's bits and, where the device may flush a 16-bit denormal, a denormal, each
//!   recorded (`narrow::widens_bits`).
//! - An integer form: each lane's code (the fetched integer lane, a code of the format), its low 8
//!   or 16 bits in an `R8G8B8A8_UINT` or `R16G16B16A16_UINT` image; a read extends it, with its
//!   sign where signed.
//!
//! A narrower format's channels the texel lacks are the fetch's constants (0, or 1 in alpha),
//! whose codes the form holds too, and a swizzled view's lanes are channels or those constants.
//!
//! **The view the copy fetches through** is the one the read it stands for would have fetched, and
//! the one the binder names the form by: the slot's own image.

use std::collections::BTreeMap;

use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{splat, ty};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::lower::sync::barrier::BarrierModel;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SlotClass, SnapshotForm, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, GlslOp, ImageOperands, ImageType,
    ImageUsage, MemAccess, StorageClass, TexelMemory, Type as SType, UnOp,
};

/// An upper estimate of the words the entry's block adds: the helper's copy, the phase's test,
/// the interlock, the two selections and the return.
const BLOCK_WORDS: u64 = 80;
/// An upper estimate of the words one slot adds: its snapshot's variable, type and decorations, the
/// pixel, the copy's fetch, encode and write, and at a read the selection, the read and its decode
/// (a signed normalized form's, the longest: the sign's extension, the product and the bound).
const SLOT_WORDS: u64 = 400;

/// The fragment texture slots function `cx` reads through a snapshot, from its lowering inputs:
/// the stage interface's `FragCoord`, the phase's push field and each snapshot's storage image are
/// planned for them. A function with another interlock or a loop over its covered samples has none
/// (the binder asks for neither), and a device without `fragmentShaderPixelInterlock` refuses them.
pub(super) fn finish(
    cx: &mut SelectCx<'_>,
    interlocked: bool,
    looped: bool,
) -> Result<BTreeMap<u32, SnapshotForm>, Error> {
    let slots: BTreeMap<u32, SnapshotForm> = cx
        .inputs
        .snapshot_textures
        .iter()
        .filter(|(s, _)| s.stage == Stage::Fragment && s.class == SlotClass::Texture)
        .map(|(s, form)| (s.index, *form))
        .collect();
    if slots.is_empty() || cx.stage != Stage::Fragment {
        return Ok(BTreeMap::new());
    }
    if interlocked || looped {
        return Err(cx.invariant(
            "a snapshot read in a function with another critical section or a sample loop",
        ));
    }
    if !cx.profile.fields().fragment.pixel_interlock {
        return Err(Unsupported::needs(
            Feature::RasterOrderGroups,
            Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
        )
        .into());
    }
    // `S` is `Coherent`, a `GLSL450` decoration; a storage image beside the `GLSL450` form of a
    // coherent access, or under the Vulkan model, is not implemented (`sync::memory_model`).
    let model = &cx.families.memory_model;
    if model.model == BarrierModel::Vulkan || model.coherent_resources {
        return Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::Fragment)).into());
    }
    cx.families.fragment.fetch_coord = Some(stage_io::frag_coord(cx));
    cx.push.insert(PushField::SnapshotPhase);
    // A normalized form's read decodes its codes by the device's arithmetic.
    if slots.values().any(|f| normalized(*f)) {
        cx.precision.insert(PrecisionNote::DevicePackConversion);
    }
    let n = u32::try_from(slots.len()).unwrap_or(u32::MAX);
    cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(n);
    let words = BLOCK_WORDS.saturating_add(SLOT_WORDS.saturating_mul(u64::from(n)));
    stage_io::add_interface_words(cx, words);
    Ok(slots)
}

/// The interlock mode of a function with snapshot slots `slots`.
pub(super) fn mode(slots: &BTreeMap<u32, SnapshotForm>) -> Option<ExecMode> {
    (!slots.is_empty()).then_some(ExecMode::PixelInterlockOrdered)
}

/// The module-scope objects of the snapshot reads.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// Each slot's snapshot, `Coherent`.
    pub images: BTreeMap<u32, GlobalId>,
    /// The `Private bool` the entry stores whether the invocation is a helper in.
    pub helper: Option<GlobalId>,
}

/// Declare each of `slots`' snapshot, an image of its form's format, and the helper's copy.
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    slots: &BTreeMap<u32, SnapshotForm>,
) -> BResult<Declared> {
    let mut d = Declared::default();
    if slots.is_empty() {
        return Ok(d);
    }
    let sampled_type = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    for (&index, &form) in slots {
        let format = match form.code() {
            None => crate::spv::ImageFormat::Rgba32ui,
            Some((8, _)) => crate::spv::ImageFormat::Rgba8ui,
            Some(_) => crate::spv::ImageFormat::Rgba16ui,
        };
        let ty = mb.ty(SType::Image(ImageType {
            sampled_type,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Storage,
            format,
        }))?;
        let slot = ResourceSlot {
            stage: Stage::Fragment,
            kind: ResourceKind::SnapshotTexels(index),
            index,
        };
        let g = mb.resource(slot, StorageClass::UniformConstant, ty)?;
        mb.coherent(g)?;
        d.images.insert(index, g);
    }
    let bool_ = mb.ty(SType::Bool)?;
    let no = mb.constant(Constant::Scalar {
        ty: bool_,
        bits: ConstBits(0),
    })?;
    d.helper = Some(mb.plain_global(StorageClass::Private, bool_, Some(no))?);
    Ok(d)
}

/// The image variable the copy of slot `index` fetches from, with its image type: the slot's
/// image.
fn source(cx: &EmitCx<'_>, index: u32) -> Option<(GlobalId, TypeId)> {
    cx.decl.families.texture.images.get(&index).copied()
}

/// Whether `form`'s codes hold every lane an image of sampled type `sampled` returns: a float
/// lane's every form but an integer one's, an integer lane's [`SnapshotForm::Bits`] or an integer
/// form of its signedness. The binder names a form of the slot's class (`SnapshotForm::of`), so a
/// pairing this rejects is an invariant failure.
fn holds(fb: &mut FunctionBuilder<'_>, form: SnapshotForm, sampled: TypeId) -> BResult<bool> {
    let integer = matches!(
        form,
        SnapshotForm::Uint8 | SnapshotForm::Uint16 | SnapshotForm::Sint8 | SnapshotForm::Sint16
    );
    let signed_form = matches!(form, SnapshotForm::Sint8 | SnapshotForm::Sint16);
    Ok(match fb.module().types().get(sampled)? {
        SType::Float { width: 32 } => !integer,
        SType::Int { width: 32, signed } => {
            form == SnapshotForm::Bits || (integer && *signed == signed_form)
        }
        _ => false,
    })
}

/// The sampled type of image type `ty`, or `None` where `ty` is no image.
fn sampled_type(fb: &mut FunctionBuilder<'_>, ty: TypeId) -> BResult<Option<TypeId>> {
    Ok(match fb.module().types().get(ty)? {
        SType::Image(i) => Some(i.sampled_type),
        _ => None,
    })
}

/// One slot's copy ([`emit_entry`]): the image the slot's reads fetch, the slot's snapshot, and
/// the form its codes are in.
#[derive(Clone, Copy, Debug)]
struct SlotCopy {
    image: GlobalId,
    snapshot: GlobalId,
    form: SnapshotForm,
}

/// The texel of image variable `g` at `pixel`, level 0: the plain fetch the slot's reads make.
fn fetch(fb: &mut FunctionBuilder<'_>, g: GlobalId, pixel: SValue) -> BResult<SValue> {
    let ptr = fb.module().global_value(g)?;
    let img = fb.load(ptr, MemAccess::default())?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let ops = ImageOperands {
        lod: Some(zero),
        ..ImageOperands::default()
    };
    fb.fetch(img, pixel, &ops)
}

/// The codes of `form` of the texel image variable `g` returns at `pixel` ([`encode`]).
fn fetched_codes(
    fb: &mut FunctionBuilder<'_>,
    g: GlobalId,
    form: SnapshotForm,
    pixel: SValue,
) -> BResult<SValue> {
    let texel = fetch(fb, g, pixel)?;
    encode(fb, form, texel)
}

/// Store codes `bits` in snapshot variable `g` at `pixel`.
fn store(fb: &mut FunctionBuilder<'_>, g: GlobalId, pixel: SValue, bits: SValue) -> BResult<()> {
    let ptr = fb.module().global_value(g)?;
    let snapshot = fb.load(ptr, MemAccess::default())?;
    fb.write(
        snapshot,
        pixel,
        bits,
        &ImageOperands::default(),
        TexelMemory::PRIVATE,
    )
}

/// The entry's block before the body ([module docs](self)), in the function's first block.
pub(super) fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let fp = &cx.plan.families.fragment;
    if fp.snapshot.is_empty() {
        return Ok(());
    }
    let decl = &cx.decl.families.fragment.snapshot;
    let Some(helper_copy) = decl.helper else {
        return Err(cx.fail("a snapshot read without its helper's copy"));
    };
    let mut copies: Vec<SlotCopy> = Vec::with_capacity(fp.snapshot.len());
    for (&index, &form) in &fp.snapshot {
        let (Some(&s), Some((t, ty))) = (decl.images.get(&index), source(cx, index)) else {
            return Err(cx.fail(format!("snapshot slot {index} is not declared")));
        };
        // The form's codes must hold the lanes the copy's fetch returns: the binder names a form
        // of the slot's class, so a float code form over an integer source (or the other way) is
        // an invariant failure, not a value for `encode` to clamp or bitcast its way through.
        let Some(sampled) = sampled_type(fb, ty)? else {
            return Err(cx.fail(format!("snapshot slot {index} is copied from no image")));
        };
        if !holds(fb, form, sampled)? {
            return Err(cx.fail(format!(
                "the copy of snapshot slot {index} fetches lanes its form {form:?} does not hold"
            )));
        }
        copies.push(SlotCopy {
            image: t,
            snapshot: s,
            form,
        });
    }
    let Some(coord) = fp.fetch_coord else {
        return Err(cx.fail("a snapshot read without its FragCoord"));
    };
    let frag = stage_io::object_global(cx, coord)?;
    // At entry no discard has run: a helper is the device's.
    let helper = if fp.demotes() {
        fb.is_helper_invocation()?
    } else {
        let Some(g) = cx.decl.families.fragment.helper else {
            return Err(cx.fail("a snapshot read without the helper input"));
        };
        let p = fb.module().global_value(g)?;
        fb.load(p, MemAccess::default())?
    };
    let p = fb.module().global_value(helper_copy)?;
    fb.store(p, helper, MemAccess::default())?;
    let phase = cx.push_field(fb, PushField::SnapshotPhase)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let taking = fb.binary(BinOp::INotEqual, phase, zero)?;
    fb.begin_interlock()?;
    let other = fb.unary(UnOp::LogicalNot, helper)?;
    let copy = fb.binary(BinOp::LogicalAnd, taking, other)?;
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let pixel = pixel(fb, frag)?;
        for &SlotCopy {
            image,
            snapshot,
            form,
        } in &copies
        {
            let bits = fetched_codes(fb, image, form, pixel)?;
            store(fb, snapshot, pixel, bits)?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), copy, arm, None)?;
    fb.end_interlock()?;
    // The draw that takes the snapshot runs nothing else: its write masks are all 0, and the
    // function writes no memory.
    let done: ArmFn<'_, ()> = Box::new(|fb, _| {
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), taking, done, None)?;
    Ok(())
}

/// `uvec2(uint(FragCoord.x), uint(FragCoord.y))` from `FragCoord` variable `frag`: a non-helper
/// fragment's coordinates lie in its pixel `[p, p + 1)`, at or above 0.
fn pixel(fb: &mut FunctionBuilder<'_>, frag: GlobalId) -> BResult<SValue> {
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, MemAccess::default())?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = fb.module().ty(SType::Vector { elem: u32_, len: 2 })?;
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    fb.composite_construct(uvec2, &[x, y])
}

/// The `uvec4` of 32-bit lanes `texel`'s codes in `form` ([module docs](self)), for the
/// snapshot's write: each lane's bits for [`SnapshotForm::Bits`], else each lane's code, its low
/// 8 or 16 bits.
pub(crate) fn encode(
    fb: &mut FunctionBuilder<'_>,
    form: SnapshotForm,
    texel: SValue,
) -> BResult<SValue> {
    let u = lanes(Held::Int {
        bits: 32,
        container: 32,
    });
    let uvec4 = ty(fb, u)?;
    let Some((width, signed)) = form.code() else {
        return match fb.value_ty(texel)? == uvec4 {
            true => Ok(texel),
            false => fb.convert(ConvOp::Bitcast, texel, uvec4),
        };
    };
    let codes = match form {
        SnapshotForm::Unorm8
        | SnapshotForm::Unorm16
        | SnapshotForm::Snorm8
        | SnapshotForm::Snorm16 => {
            let f = lanes(Held::Float(FloatFmt::F32));
            let low = match signed {
                true => (-1.0f32).to_bits(),
                false => 0,
            };
            let (low, high) = (
                splat(fb, f, u64::from(low))?,
                splat(fb, f, u64::from(1.0f32.to_bits()))?,
            );
            let clamped = fb.glsl(GlslOp::FClamp, &[texel, low, high])?;
            let d = splat(fb, f, u64::from((scale(width, signed) as f32).to_bits()))?;
            let scaled = fb.binary(BinOp::FMul, clamped, d)?;
            let code = fb.glsl(GlslOp::RoundEven, &[scaled])?;
            match signed {
                true => {
                    let ivec4 = signed_lanes(fb)?;
                    let c = fb.convert(ConvOp::ConvertFToS, code, ivec4)?;
                    fb.convert(ConvOp::Bitcast, c, uvec4)?
                }
                false => fb.convert(ConvOp::ConvertFToU, code, uvec4)?,
            }
        }
        SnapshotForm::Half => narrow::to_bits(fb, lanes(Held::Float(FloatFmt::Half)), texel)?,
        _ => match fb.value_ty(texel)? == uvec4 {
            true => texel,
            false => fb.convert(ConvOp::Bitcast, texel, uvec4)?,
        },
    };
    mask_codes(fb, form, codes)
}

/// The `uvec4` of codes `codes` (32-bit lanes) kept to `form`'s code width: the snapshot's image
/// holds that many bits a lane, so a lane of a wider value loses the rest.
fn mask_codes(fb: &mut FunctionBuilder<'_>, form: SnapshotForm, codes: SValue) -> BResult<SValue> {
    let u = lanes(Held::Int {
        bits: 32,
        container: 32,
    });
    let uvec4 = ty(fb, u)?;
    let codes = match fb.value_ty(codes)? == uvec4 {
        true => codes,
        false => fb.convert(ConvOp::Bitcast, codes, uvec4)?,
    };
    let Some((width, _)) = form.code() else {
        return Ok(codes);
    };
    let mask = splat(fb, u, (1u64 << width) - 1)?;
    fb.binary(BinOp::BitwiseAnd, codes, mask)
}

/// The value of type `texel` the codes `codes` (a `uvec4` read from the snapshot) of `form` stand
/// for ([module docs](self)): the inverse of [`encode`] on every value it encodes.
pub(crate) fn decode(
    fb: &mut FunctionBuilder<'_>,
    form: SnapshotForm,
    codes: SValue,
    texel: TypeId,
) -> BResult<SValue> {
    let u = lanes(Held::Int {
        bits: 32,
        container: 32,
    });
    let Some((width, signed)) = form.code() else {
        return match fb.value_ty(codes)? == texel {
            true => Ok(codes),
            false => fb.convert(ConvOp::Bitcast, codes, texel),
        };
    };
    // A signed code's lanes, sign-extended from their low `width` bits.
    let extended = |fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        let ivec4 = signed_lanes(fb)?;
        let shift = splat(fb, u, u64::from(32 - width))?;
        let up = fb.binary(BinOp::ShiftLeftLogical, codes, shift)?;
        let up = fb.convert(ConvOp::Bitcast, up, ivec4)?;
        fb.binary(BinOp::ShiftRightArithmetic, up, shift)
    };
    // A normalized code `c` times the `f32` nearest `1 / D`.
    let f = lanes(Held::Float(FloatFmt::F32));
    let reciprocal = |fb: &mut FunctionBuilder<'_>, signed: bool| -> BResult<SValue> {
        let r = 1.0 / scale(width, signed) as f32;
        splat(fb, f, u64::from(r.to_bits()))
    };
    let value = match form {
        SnapshotForm::Unorm8 | SnapshotForm::Unorm16 => {
            let vec4 = ty(fb, f)?;
            let c = fb.convert(ConvOp::ConvertUToF, codes, vec4)?;
            let r = reciprocal(fb, false)?;
            fb.binary(BinOp::FMul, c, r)?
        }
        SnapshotForm::Snorm8 | SnapshotForm::Snorm16 => {
            let n = extended(fb)?;
            let vec4 = ty(fb, f)?;
            let c = fb.convert(ConvOp::ConvertSToF, n, vec4)?;
            let r = reciprocal(fb, true)?;
            let v = fb.binary(BinOp::FMul, c, r)?;
            // `-2^(b-1)` reads `-1`, as the code above it.
            let low = splat(fb, f, u64::from((-1.0f32).to_bits()))?;
            fb.glsl(GlslOp::FMax, &[v, low])?
        }
        SnapshotForm::Half => narrow::from_bits(fb, lanes(Held::Float(FloatFmt::Half)), codes)?,
        _ => match signed {
            true => extended(fb)?,
            false => codes,
        },
    };
    match fb.value_ty(value)? == texel {
        true => Ok(value),
        false => fb.convert(ConvOp::Bitcast, value, texel),
    }
}

/// Whether `form` is a normalized form, whose read decodes its codes by the device's arithmetic.
fn normalized(form: SnapshotForm) -> bool {
    matches!(
        form,
        SnapshotForm::Unorm8 | SnapshotForm::Unorm16 | SnapshotForm::Snorm8 | SnapshotForm::Snorm16
    )
}

/// The scale `D` of a normalized code of `width` bits: `2^b - 1`, or `2^(b-1) - 1` where signed.
fn scale(width: u8, signed: bool) -> u32 {
    match signed {
        true => (1u32 << (width - 1)) - 1,
        false => (1u32 << width) - 1,
    }
}

/// The four-lane form of `held`.
fn lanes(held: Held) -> Form {
    Form { held, lanes: 4 }
}

/// The type `ivec4`.
fn signed_lanes(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    fb.module().ty(SType::Vector { elem: i32_, len: 4 })
}

/// What a read of a snapshot slot needs, made in the open block before the read: the snapshot's
/// image, whether the invocation is no helper, and the type the slot's fetch returns.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotRead {
    snapshot: SValue,
    other: SValue,
    texel: TypeId,
    form: SnapshotForm,
}

/// The snapshot read standing for a fetch of texture slot `index` from image `img` (loaded from
/// the image the copy fetches from), or `None` where the function reads the slot
/// directly. The read must be the fetch of a single-sampled 2D level the copy makes: anything
/// else is an invariant failure, since the slot's interface fact admits no other.
pub(crate) fn prepare(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: u32,
    img: SValue,
    plain_fetch: bool,
) -> BResult<Option<SnapshotRead>> {
    let Some(&form) = cx.plan.families.fragment.snapshot.get(&index) else {
        return Ok(None);
    };
    if !plain_fetch {
        return Err(cx.fail(format!(
            "a read of snapshot slot {index} that is no plain fetch"
        )));
    }
    let ty = fb.value_ty(img)?;
    let image = match fb.module().types().get(ty)? {
        SType::Image(i) => *i,
        _ => return Err(cx.fail(format!("snapshot slot {index} read from no image"))),
    };
    if image.dim != Dim::D2 || image.arrayed || image.multisampled {
        return Err(cx.fail(format!("snapshot slot {index} is no single 2D image")));
    }
    if !holds(fb, form, image.sampled_type)? {
        return Err(cx.fail(format!(
            "snapshot slot {index} returns lanes its form {form:?} does not hold"
        )));
    }
    let texel = fb.module().ty(SType::Vector {
        elem: image.sampled_type,
        len: 4,
    })?;
    let decl = &cx.decl.families.fragment.snapshot;
    let (Some(&s), Some(helper)) = (decl.images.get(&index), decl.helper) else {
        return Err(cx.fail(format!("snapshot slot {index} is not declared")));
    };
    let ptr = fb.module().global_value(s)?;
    let snapshot = fb.load(ptr, MemAccess::default())?;
    let ptr = fb.module().global_value(helper)?;
    let h = fb.load(ptr, MemAccess::default())?;
    let other = fb.unary(UnOp::LogicalNot, h)?;
    Ok(Some(SnapshotRead {
        snapshot,
        other,
        texel,
        form,
    }))
}

/// The texel at `coord` a read of `r` returns: the snapshot's codes decoded as the slot's lanes
/// ([`decode`]), in a non-helper invocation, and 0 in a helper, which reads nothing.
pub(crate) fn read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: SnapshotRead,
    coord: SValue,
) -> BResult<SValue> {
    let texel = r.texel;
    let snapshot = r.snapshot;
    let form = r.form;
    let read: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let codes = fb.read(
            snapshot,
            coord,
            &ImageOperands::default(),
            TexelMemory::PRIVATE,
        )?;
        let value = decode(fb, form, codes, texel)?;
        Ok(ArmEnd::FallThrough(vec![value]))
    });
    let nothing: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let zero = fb.module().constant(Constant::Null { ty: texel })?;
        let zero = fb.module().const_value(zero)?;
        Ok(ArmEnd::FallThrough(vec![zero]))
    });
    let merged = fb.if_(&mut (), r.other, read, Some(nothing))?;
    match merged.0.as_slice() {
        [v] => Ok(*v),
        _ => Err(cx.fail("a snapshot read without its merged texel")),
    }
}
