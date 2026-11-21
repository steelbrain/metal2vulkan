//! A fragment function's `[[stencil]]` output without `VK_EXT_shader_stencil_export`: the
//! shader-side stencil.
//!
//! **Metal's semantics** (measured). The output's low 8 bits `r` are the stencil reference of both
//! faces. The stencil test `(r & readMask) cmp (s & readMask)` against the stored value `s` runs
//! before the depth test, and the face's operation writes `s' = (s & ~writeMask) | (op(s, r) &
//! writeMask)`, per sample in primitive order. A fragment that fails is dropped; a discarded one
//! changes nothing. `[[early_fragment_tests]]` does not compile beside the output, so the tests
//! always follow the function.
//!
//! **The form.** Vulkan's fixed-function stencil test takes one reference per draw, so no state
//! can give each fragment its own. Where the device has the ordered pixel interlock
//! ([`device_form`]), the function applies the test itself: its body runs inside the interlock
//! (`fragment::wrap_entry`), and at each return, unless a discard was recorded, it reads the
//! pixel's value from a coherent `R8Uint` shadow of the stencil (`SyntheticBinding::ShadowStencil`,
//! texel `(x, y)` for pixel `(x, y)`), takes the face's state from `PushField::ShadowStencilState`
//! by `FrontFacing`, applies Metal's comparison and the stencil-failure or pass operation under the
//! masks, writes the shadow, and records a failing fragment as discarded, which the entry kills
//! after the critical section. The ordered interlock makes the read, the operation, and the write
//! of each fragment of a pixel one step in primitive order, across the draws of a subpass; the
//! device-scope coherence of every access inside the critical section (`sync::memory_model`) makes
//! each step's write visible to the next. The fixed-function stencil test is off; the depth test
//! stays fixed-function after the function, so the module never declares early tests. The consumer
//! keeps the shadow (`GraphicsState::shadow_stencil`).
//!
//! **Multisampled** (compiled for a pipeline of more than one sample,
//! `LoweringInputs::multisampled`). Metal tests and writes each covered sample of the fragment
//! against the one reference. The shadow is then a multisampled storage image of the pipeline's
//! count (sample `s` of texel `(x, y)` is sample `s` of pixel `(x, y)`), and the step runs once per
//! sample of the invocation's `SampleMask` input: every covered sample of a pixel-rate invocation,
//! the one sample of an invocation under sample shading (a function that writes memory or reads
//! `[[sample_id]]`). The function writes the samples that pass to the `SampleMask` output, so a
//! failing sample gets no color or depth, and a fragment no sample of which passes is recorded as
//! discarded. The form reads only the invocation's own coverage, never Metal's `[[sample_mask]]`
//! (the pixel's), so the stage interface states the output and not the input.
//!
//! **Exact only where the stencil write cannot depend on the depth result**: the binder refuses a
//! state with a depth attachment, a depth comparison other than `Always`, and a face whose
//! depth-failure operation differs from its pass operation; link refuses alpha to coverage. A
//! returned sample mask and an imageblock beside the output are not implemented.

use crate::air::abi::OutputRole;
use crate::air::ir::TypeId as AirTypeId;
use crate::device::requirements::{FormatUsage, VkExtension, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::arith::Form;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{self, Returned};
use crate::lower::sync::memory_model;
use crate::metal::format::{without_format_gaps, StorageUse};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, Dim, GlslOp, ImageOperands, ImageType, ImageUsage, MemAccess, StorageClass,
    Type as SType, UnOp,
};
use crate::vk;

/// The shadow's format: Metal's stencil values are 8-bit.
pub(crate) const SHADOW_FORMAT: vk::Format = vk::Format::R8_UINT;

/// Bit 31 of a face's word: the pass has a stencil attachment to test against.
pub(crate) const ATTACHED_BIT: u32 = 1 << 31;

/// The word of one face of `PushField::ShadowStencilState` (see its docs): Metal's compare
/// function `compare` and operations `stencil_failure` and `pass` by their raw values, the read
/// and write masks' low 8 bits, and [`ATTACHED_BIT`] when `attached`.
pub(crate) fn face_word(
    compare: u32,
    stencil_failure: u32,
    pass: u32,
    read_mask: u32,
    write_mask: u32,
    attached: bool,
) -> u32 {
    (compare & 7)
        | (stencil_failure & 7) << 3
        | (pass & 7) << 6
        | (read_mask & 0xFF) << 9
        | (write_mask & 0xFF) << 17
        | if attached { ATTACHED_BIT } else { 0 }
}

/// The shader-side stencil's plan: the output's field of the returned value and its form, and
/// how the module declares the shadow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowStencil {
    /// The returned value's field that holds the output (0 for a returned value itself).
    pub field: u32,
    pub form: Form,
    /// The declared SPIR-V image format: `R8ui`, or `Unknown` for a formatless declaration.
    pub declared: crate::spv::ImageFormat,
    /// The stage interface's `FrontFacing` object, which selects the face's state.
    pub front_facing: usize,
    /// The multisampled form's coverage objects (see the module docs); `None` for one sample.
    pub samples: Option<SampleMasks>,
}

/// The stage interface's objects of the multisampled form: the `SampleMask` input it tests and
/// the `SampleMask` output it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleMasks {
    pub covered: usize,
    pub kept: usize,
}

/// The declaration of the shadow on `p`, or what the device lacks for the form (see the module
/// docs): the ordered pixel interlock, fragment stores, the shadow's format as a storage image,
/// declared `R8ui` where the device has the extended storage formats, else without a format
/// where it grants that access, and, `multisampled`, multisampled storage images. The refusal names
/// both ways to translate the output: the extension, or what the form lacks (the extended storage
/// formats for the declaration, as the interlocked framebuffer fetch names them for a typed
/// format).
pub(crate) fn device_form(
    p: &DeviceProfile,
    multisampled: bool,
) -> Result<crate::spv::ImageFormat, Unsupported> {
    let f = p.fields();
    let mut missing = Vec::new();
    if !super::fetch::interlock_forms(p) {
        missing.push(Capability::Feature(VkFeature::FragmentShaderPixelInterlock));
    }
    if !f.leaf.fragment_stores_and_atomics {
        missing.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    if multisampled && !f.images.storage_multisample {
        missing.push(Capability::Feature(
            VkFeature::ShaderStorageImageMultisample,
        ));
    }
    if !p.supports_format(SHADOW_FORMAT, FormatUsage::StorageImage) {
        missing.push(Capability::Format {
            format: SHADOW_FORMAT,
            usage: FormatUsage::StorageImage,
        });
    }
    let use_ = StorageUse::texture(true, true);
    let declared = if f.images.storage_extended_formats {
        Some(crate::spv::ImageFormat::R8ui)
    } else if without_format_gaps(SHADOW_FORMAT, use_, p).is_empty() {
        Some(crate::spv::ImageFormat::Unknown)
    } else {
        // `R8ui` is an extended format: the typed declaration needs only the feature.
        missing.push(Capability::Feature(
            VkFeature::ShaderStorageImageExtendedFormats,
        ));
        None
    };
    match declared {
        Some(d) if missing.is_empty() => Ok(d),
        _ => Err(Unsupported::needs_any_of(
            Feature::StencilExport,
            vec![
                vec![Capability::Extension(VkExtension::ExtShaderStencilExport)],
                missing,
            ],
        )),
    }
}

fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::StencilExport).into()
}

/// Plan the `[[stencil]]` output of AIR type `ty`, held in field `field` of the returned value,
/// as the shader-side stencil (`stage_io` plans the entry's other outputs; the caller checked the
/// device has no stencil export).
pub fn select_output(cx: &mut SelectCx<'_>, field: u32, ty: AirTypeId) -> Result<(), Error> {
    let multisampled = cx.inputs.multisampled;
    let declared = device_form(cx.profile, multisampled)?;
    let form = stage_io::int_scalar_output(cx, ty)?;
    // A returned sample mask would drop samples after the function's write.
    let sample_mask = cx
        .entry
        .ep
        .outputs
        .iter()
        .any(|o| matches!(o.role, OutputRole::SampleMask));
    if sample_mask {
        return Err(not_implemented());
    }
    if cx.families.fragment.stencil.is_some() {
        return Err(cx.invariant("a second stencil output"));
    }
    let coord = stage_io::frag_coord(cx);
    let front_facing = stage_io::front_facing(cx);
    let samples = multisampled.then(|| {
        let (covered, kept) = stage_io::sample_masks(cx);
        SampleMasks { covered, kept }
    });
    let fp = &mut cx.families.fragment;
    fp.fetch_coord = Some(coord);
    fp.stencil = Some(ShadowStencil {
        field,
        form,
        declared,
        front_facing,
        samples,
    });
    cx.facts.insert(RuntimeFact::ShadowStencil {
        formatless: declared == crate::spv::ImageFormat::Unknown,
        multisampled,
    });
    cx.push.insert(PushField::ShadowStencilState);
    cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(1);
    Ok(())
}

/// Check the plan once every output and parameter is planned: an imageblock beside the output
/// is not implemented (what Metal's stencil test does to its writes is not read).
pub(super) fn finish(cx: &SelectCx<'_>) -> Result<(), Error> {
    let fp = &cx.families.fragment;
    if fp.stencil.is_some() && fp.imageblock.is_some() {
        return Err(not_implemented());
    }
    Ok(())
}

/// An upper estimate of the words the form adds: the shadow and its type, and at each of `rets`
/// returns the guard, the state's fields, the comparison and operation selections, and the read
/// and write (with the multisampled form's loop over the samples, its variables, and the mask's
/// writes).
pub(super) fn words(cx: &SelectCx<'_>, rets: u64) -> u64 {
    const OBJECT: u64 = 40;
    const RETURN: u64 = 480;
    const SAMPLES: u64 = 160;
    let Some(plan) = cx.families.fragment.stencil else {
        return 0;
    };
    let per_return = if plan.samples.is_some() {
        RETURN + SAMPLES
    } else {
        RETURN
    };
    OBJECT.saturating_add(rets.saturating_mul(per_return))
}

/// Declare the shadow: a 2D `uint` storage image of the planned declaration, multisampled for the
/// multisampled form.
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    plan: Option<&ShadowStencil>,
) -> BResult<Option<GlobalId>> {
    let Some(plan) = plan else {
        return Ok(None);
    };
    let sampled_type = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: plan.samples.is_some(),
        usage: ImageUsage::Storage,
        format: plan.declared,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::ShadowStencil),
        index: 0,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
        .map(Some)
}

/// A `u32` constant.
fn u32_value(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// The bits `[offset, offset + width)` of `word`.
pub(super) fn field_of(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    offset: u32,
    width: u32,
) -> BResult<SValue> {
    let o = u32_value(fb, offset)?;
    let w = u32_value(fb, width)?;
    fb.bit_field_extract(false, word, o, w)
}

/// `choices[code]` for a `code` in `0..8`, as a chain of selections.
fn select_by(fb: &mut FunctionBuilder<'_>, code: SValue, choices: [SValue; 8]) -> BResult<SValue> {
    let mut out = choices[7];
    for (i, &choice) in choices.iter().enumerate().take(7).rev() {
        let k = u32_value(fb, i as u32)?;
        let is = fb.binary(BinOp::IEqual, code, k)?;
        out = fb.select(is, choice, out)?;
    }
    Ok(out)
}

/// Whether `a cmp b` under Metal's compare function `code` (`MTLCompareFunction`: never, less,
/// equal, less-equal, greater, not-equal, greater-equal, always), with `a` the masked reference
/// and `b` the masked stored value.
pub(super) fn compare(
    fb: &mut FunctionBuilder<'_>,
    code: SValue,
    a: SValue,
    b: SValue,
) -> BResult<SValue> {
    let never = super::bool_const(fb, false)?;
    let always = super::bool_const(fb, true)?;
    let less = fb.binary(BinOp::ULessThan, a, b)?;
    let equal = fb.binary(BinOp::IEqual, a, b)?;
    let less_equal = fb.binary(BinOp::ULessThanEqual, a, b)?;
    let greater = fb.binary(BinOp::UGreaterThan, a, b)?;
    let not_equal = fb.binary(BinOp::INotEqual, a, b)?;
    let greater_equal = fb.binary(BinOp::UGreaterThanEqual, a, b)?;
    select_by(
        fb,
        code,
        [
            never,
            less,
            equal,
            less_equal,
            greater,
            not_equal,
            greater_equal,
            always,
        ],
    )
}

/// The 8-bit value Metal's stencil operation `code` (`MTLStencilOperation`: keep, zero, replace,
/// increment-clamp, decrement-clamp, invert, increment-wrap, decrement-wrap) makes of stored
/// value `s` with reference `r`.
pub(super) fn operation(
    fb: &mut FunctionBuilder<'_>,
    code: SValue,
    s: SValue,
    r: SValue,
) -> BResult<SValue> {
    let zero = u32_value(fb, 0)?;
    let one = u32_value(fb, 1)?;
    let max = u32_value(fb, 0xFF)?;
    let up = fb.binary(BinOp::IAdd, s, one)?;
    let down = fb.binary(BinOp::ISub, s, one)?;
    let at_max = fb.binary(BinOp::IEqual, s, max)?;
    let at_zero = fb.binary(BinOp::IEqual, s, zero)?;
    let increment_clamp = fb.select(at_max, max, up)?;
    let decrement_clamp = fb.select(at_zero, zero, down)?;
    let invert = fb.binary(BinOp::BitwiseXor, s, max)?;
    let increment_wrap = fb.binary(BinOp::BitwiseAnd, up, max)?;
    let decrement_wrap = fb.binary(BinOp::BitwiseAnd, down, max)?;
    select_by(
        fb,
        code,
        [
            s,
            zero,
            r,
            increment_clamp,
            decrement_clamp,
            invert,
            increment_wrap,
            decrement_wrap,
        ],
    )
}

/// The values one step of the test reads: the face's word, the reference, the pixel, the
/// shadow, and the shadow's texel coherence.
#[derive(Clone, Copy)]
struct Step {
    word: SValue,
    reference: SValue,
    pixel: SValue,
    image: SValue,
    mem: crate::spirv::ir::TexelMemory,
}

/// Apply Metal's stencil test and the face's operation to the shadow's texel at `step.pixel`
/// (its sample `sample` in the multisampled form): read it, compare under the read mask, write the
/// stencil-failure or pass operation's value under the write mask. Whether the test passed.
fn apply(fb: &mut FunctionBuilder<'_>, step: Step, sample: Option<SValue>) -> BResult<SValue> {
    let Step {
        word,
        reference,
        pixel,
        image,
        mem,
    } = step;
    let operands = ImageOperands {
        sample,
        ..ImageOperands::default()
    };
    let texel = fb.read(image, pixel, &operands, mem)?;
    let stored = fb.composite_extract(texel, &[0])?;
    let compare_code = field_of(fb, word, 0, 3)?;
    let fail_code = field_of(fb, word, 3, 3)?;
    let pass_code = field_of(fb, word, 6, 3)?;
    let read_mask = field_of(fb, word, 9, 8)?;
    let write_mask = field_of(fb, word, 17, 8)?;
    let a = fb.binary(BinOp::BitwiseAnd, reference, read_mask)?;
    let b = fb.binary(BinOp::BitwiseAnd, stored, read_mask)?;
    let passed = compare(fb, compare_code, a, b)?;
    let code = fb.select(passed, pass_code, fail_code)?;
    let written = operation(fb, code, stored, reference)?;
    let max = u32_value(fb, 0xFF)?;
    let keep_mask = fb.binary(BinOp::BitwiseXor, write_mask, max)?;
    let kept_bits = fb.binary(BinOp::BitwiseAnd, stored, keep_mask)?;
    let new_bits = fb.binary(BinOp::BitwiseAnd, written, write_mask)?;
    let value = fb.binary(BinOp::BitwiseOr, kept_bits, new_bits)?;
    let u32_ = u32_type(fb)?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    let zero = u32_value(fb, 0)?;
    let texel = fb.composite_construct(uvec4, &[value, zero, zero, zero])?;
    fb.write(image, pixel, texel, &operands, mem)?;
    Ok(passed)
}

/// The `u32` type.
fn u32_type(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

/// A pointer to element 0 of the stage interface's array object `i` (a `SampleMask`).
fn element0(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, i: usize) -> BResult<SValue> {
    let g = stage_io::object_global(cx, i)?;
    let base = fb.module().global_value(g)?;
    let zero = u32_value(fb, 0)?;
    fb.access_chain(base, &[zero])
}

/// The multisampled form's step over the samples of `covered` (see the module docs): the mask
/// of those that pass.
fn apply_per_sample(fb: &mut FunctionBuilder<'_>, step: Step, covered: SValue) -> BResult<SValue> {
    let u32_ = u32_type(fb)?;
    let zero_c = fb.u32_const(0)?;
    let left_var = fb.local_var(u32_, Some(zero_c))?;
    let left = fb.var_ptr(left_var)?;
    let kept_var = fb.local_var(u32_, Some(zero_c))?;
    let kept = fb.var_ptr(kept_var)?;
    fb.store(left, covered, MemAccess::default())?;
    let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
        let rest = fb.load(left, MemAccess::default())?;
        let zero = u32_value(fb, 0)?;
        let done = fb.binary(BinOp::IEqual, rest, zero)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, exit, None)?;
        // The lowest sample left, then the others.
        let sample = fb.glsl(GlslOp::FindILsb, &[rest])?;
        let one = u32_value(fb, 1)?;
        let below = fb.binary(BinOp::ISub, rest, one)?;
        let others = fb.binary(BinOp::BitwiseAnd, rest, below)?;
        fb.store(left, others, MemAccess::default())?;
        let passed = apply(fb, step, Some(sample))?;
        let bit = fb.binary(BinOp::ShiftLeftLogical, one, sample)?;
        let bit = fb.select(passed, bit, zero)?;
        let so_far = fb.load(kept, MemAccess::default())?;
        let so_far = fb.binary(BinOp::BitwiseOr, so_far, bit)?;
        fb.store(kept, so_far, MemAccess::default())?;
        Ok(())
    });
    fb.loop_(&mut (), LoopSpec::default(), body)?;
    fb.load(kept, MemAccess::default())
}

/// At a return of returned value `v`: unless a discard was recorded, apply Metal's stencil test
/// and operation of the fragment's face to the shadow at its pixel (at each sample of the
/// invocation in the multisampled form) with the output as the reference, and record a failing
/// fragment as discarded (see the module docs). The multisampled form first keeps every covered
/// sample in the `SampleMask` output, then the samples that pass. Called before the other writes
/// at the return, which a recorded discard skips.
pub(super) fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
) -> BResult<()> {
    let Some(plan) = cx.plan.families.fragment.stencil else {
        return Ok(());
    };
    let Some(v) = v else {
        return Err(cx.fail("a stencil output without a returned value"));
    };
    let Some(shadow) = cx.decl.families.fragment.shadow_stencil else {
        return Err(cx.fail("the shadow stencil is not declared"));
    };
    let Some(discarded) = cx.decl.families.fragment.discarded else {
        return Err(cx.fail("a shader-side stencil without its discard predicate"));
    };
    let x = cx.operand(fb, v)?;
    let output = match cx.plan.families.stage_io.returned {
        Returned::Struct => fb.composite_extract(x, &[plan.field])?,
        Returned::Value => x,
        Returned::Void => return Err(cx.fail("a stencil output of a void entry")),
    };
    let max = u32_value(fb, 0xFF)?;
    let reference = fb.binary(BinOp::BitwiseAnd, output, max)?;
    let state = cx.push_field(fb, PushField::ShadowStencilState)?;
    let facing = stage_io::object_global(cx, plan.front_facing)?;
    let pixel = super::fetch::pixel(cx, fb)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let masks = match plan.samples {
        // The coverage word, from the entry's copy inside an interlocked body.
        Some(m) => Some((
            stage_io::coverage_in(cx, fb, m.covered)?,
            element0(cx, fb, m.kept)?,
        )),
        None => None,
    };
    let facing = fb.module().global_value(facing)?;
    let front = fb.load(facing, MemAccess::default())?;
    let front_word = fb.composite_extract(state, &[0])?;
    let back_word = fb.composite_extract(state, &[1])?;
    let word = fb.select(front, front_word, back_word)?;
    let attached_bit = u32_value(fb, ATTACHED_BIT)?;
    let attached = fb.binary(BinOp::BitwiseAnd, word, attached_bit)?;
    let zero = u32_value(fb, 0)?;
    let attached = fb.binary(BinOp::INotEqual, attached, zero)?;
    let p = fb.module().global_value(discarded)?;
    let was = fb.load(p, MemAccess::default())?;
    let kept = fb.unary(UnOp::LogicalNot, was)?;
    let test = fb.binary(BinOp::LogicalAnd, kept, attached)?;
    // Every covered sample stays unless the test drops it.
    let covered = match masks {
        Some((c, kept)) => {
            fb.store(kept, c, MemAccess::default())?;
            Some((c, kept))
        }
        None => None,
    };
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let image = fb.module().global_value(shadow)?;
        let image = fb.load(image, MemAccess::default())?;
        let step = Step {
            word,
            reference,
            pixel,
            image,
            mem,
        };
        // A failing fragment is dropped: the entry kills it after the critical section, and
        // the return's other writes skip it. In the multisampled form, a fragment of which
        // some samples pass keeps those.
        let failed = match covered {
            None => {
                let passed = apply(fb, step, None)?;
                fb.unary(UnOp::LogicalNot, passed)?
            }
            Some((covered, kept)) => {
                let passed = apply_per_sample(fb, step, covered)?;
                fb.store(kept, passed, MemAccess::default())?;
                let zero = u32_value(fb, 0)?;
                fb.binary(BinOp::IEqual, passed, zero)?
            }
        };
        let p = fb.module().global_value(discarded)?;
        fb.store(p, failed, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), test, arm, None)?;
    Ok(())
}

#[cfg(test)]
mod tests;
