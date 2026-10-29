//! Framebuffer fetch: `[[color(n)]]` fragment inputs read through input attachments.
//!
//! Each color index a function reads is one `SubpassData` image, declared as the descriptor
//! symbol `ResourceSlot{Fragment, InputAttachment(n), n}` (serialization decorates it
//! `InputAttachmentIndex n`, and link lists it in `GraphicsState::input_attachments`). A read is
//! `OpImageRead` at offset `(0, 0)`, the calling fragment's own texel, taken to the parameter's
//! lanes and brought to its lowered representation like a stage input: the image's sampled type is
//! `float` for float colors (a `half` is rounded to its format) and `uint` for integer colors
//! (masked to the AIR width). Integer colors of a signed format read the same bits through a `uint`
//! image; like the `uint` color outputs, link must check the attachment format's class against the
//! declared one.
//!
//! **Ordering**: Metal orders a fetch after earlier overlapping fragments of the draw.
//! Vulkan orders an input-attachment read so only with rasterization-order color attachment
//! access. Without it, a device with `fragmentShaderPixelInterlock` reads (and writes) the
//! attachment as a storage image inside the ordered interlock where the pipeline's state lets
//! that be exact (`fetch`). Everywhere else the function is refused, naming what either ordered
//! form needs (`Unsupported{FramebufferFetch, ...}`; the storage form's own reason where it alone
//! was blocked): an unordered read could see the attachment before an earlier overlapping
//! fragment's write, which Metal never shows. The function states
//! `RuntimeFact::FramebufferFetch{color}` for every index it reads as an input attachment, so
//! link can set the pipeline's rasterization-order flags and the render pass can provide the
//! self-dependency.
//!
//! **Per sample**. Metal runs a function that reads a color attachment of a multisampled pass once
//! per covered sample, reading its own sample (measured). Compiled for such a pipeline
//! (`LoweringInputs::multisampled`), each input attachment is a multisampled `SubpassData` image
//! (`MS` 1, as Vulkan requires for an attachment of more than one sample), read at the invocation's
//! `SampleId` (`fetch::plan_sample`), which runs the function once per covered sample under sample
//! shading; the function's color outputs then write only that sample, and a returned sample mask
//! and alpha to coverage stay the fixed function's, as at one sample. Rasterization-order
//! attachment access orders each sample's reads and writes in primitive order. Link's sample
//! invocations read the function's pixel inputs where Metal does or refuse them, and refuse a stage
//! compiled for another sample count.
//!
//! **An extended-range attachment** (`metal::format::ExtendedRange`) holds Metal's raw texels in
//! its raw unsigned format: its input attachment is of `uint` texels, and a read decodes the texel
//! to Metal's tile value (`arith::extended_range::tile_value`), as the storage form's does. The
//! interface records the attachment (`StageIo::extended_range_colors`), so link holds a stage
//! compiled for it to the pipeline's attachments. Where the pipeline blends such an attachment the
//! function writes, the write blends over the same input attachment (`input_blend`), which a
//! `[[color(n)]]` read of the attachment shares ([`input_texel`]).
//!
//! **An 8-bit sRGB attachment** (`RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and the rest) reads through
//! its input attachment the device's decode of its bytes, Vulkan's sRGB EOTF within the device's
//! precision, where Metal decodes a color byte to its twelve-bit step over 4095. On a device that
//! decodes through Metal's own texture unit (`VkProperty::MetalSampler`) that is Metal's decode,
//! and the module requires the property; elsewhere the interface records each `float` color it
//! reads so (`StageIo::decoded_fetches`), and a render pipeline whose attachment there has such a
//! format records `PrecisionNote::SrgbTexelDecode` (`color_writes::pipeline_writes`). The
//! interlocked storage form reads the attachment's bytes, no sRGB view of them, and decodes them
//! by Metal's table itself (`fetch`).
//!
//! **An `A8Unorm` attachment whose image holds Metal's alpha in red** (`R8_UNORM` on a device
//! without `A8_UNORM`, `LoweringInputs::alpha_in_red_colors`) reads `(r, 0, 0, 1)` through its
//! input attachment, which the read turns into Metal's `(0, 0, 0, r)` ([`alpha_in_red`]); the
//! interface records the color (`StageIo::alpha_in_red_colors`), which link holds the pipeline's
//! to.
//!
//! A color input in a raster order group is read like any other: the family runs the whole body
//! inside the interlock, and the read's ordering against other fragments' color writes is still the
//! ordering policy's (the interlock orders shader invocations, not the attachment writes that
//! blending performs after them).
//!
//! Refused: `bool`, 64-bit, and aggregate colors.

use std::collections::BTreeMap;

use crate::air::abi::{AbiInt, OutputRole, ParamBinding, ParamRole};

use crate::air::ir::{ArgIdx, TypeId as AirTypeId, Value};
use crate::device::requirements::VkFeature;
use crate::error::{Attachment, Capability, DescriptorError, Error, Feature, Unsupported};
use crate::lower::arith::legal::{form_on, refuse_int64_interface};
use crate::lower::arith::{extended_range, narrow, ops, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::format::ExtendedRange;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{
    Constant, Dim, ImageType, ImageUsage, MemAccess, StorageClass, TexelMemory, Type as SType,
};

use super::not_lowered;

/// One used color input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorInput {
    /// The LLVM parameter index.
    pub param: u32,
    /// The color attachment index, which is also the input attachment index.
    pub color: u32,
    /// The parameter's form: a scalar or 2–4 lane vector of 32-bit held integers or floats.
    pub form: Form,
    /// The attachment's extended-range form (`LoweringInputs::extended_range_colors`), whose
    /// image holds Metal's raw texels: an input attachment of the raw format's `uint`s, which
    /// the read decodes to Metal's tile value. `None` for a format held directly.
    pub extended: Option<ExtendedRange>,
    /// The attachment's image holds Metal's alpha in red (`LoweringInputs::alpha_in_red_colors`,
    /// `A8Unorm` in `R8_UNORM`): a read, `(r, 0, 0, 1)` from the image, is Metal's `(0, 0, 0, r)`.
    pub alpha_red: bool,
}

/// An upper estimate of the words one input attachment declares (type, variable, decorations).
const OBJECT_WORDS: u64 = 32;
/// An upper estimate of the words one read adds (load, the sample's load, image read, lane
/// extraction, mask or rounding).
const READ_WORDS: u64 = 32;

/// The form of a color value of AIR type `ty`, read or written: a scalar or 2 to 4 lanes of
/// 32-bit held integers or floats. A 64-bit one is not Metal's (`refuse_int64_interface`);
/// `bool`, other widths, and aggregates are not lowered.
pub(super) fn form(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Form, Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let form = form_on(cx.entry, cx.profile, ty)?;
    let held32 = match form.held {
        Held::Int { container, .. } => container == 32,
        Held::Float(f) => f.container() == 32,
        Held::Bool | Held::Wide { .. } => false,
    };
    if !held32 || !(1..=4).contains(&form.lanes) {
        return Err(not_lowered());
    }
    Ok(form)
}

/// The [`form`] of color input parameter `b`.
pub(super) fn param_form(cx: &SelectCx<'_>, b: &ParamBinding) -> Result<Form, Error> {
    let ty = cx
        .value_ty(Value::Arg(ArgIdx::new(b.param)))
        .ok_or_else(|| cx.invariant("a color input without a type"))?;
    form(cx, ty)
}

/// Plan color input `b` (a `ColorInput` role) of the entry.
pub(super) fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<ColorInput, Error> {
    let ParamRole::ColorInput { index, .. } = &b.role else {
        return Err(cx.invariant("a non-color parameter planned as a color input"));
    };
    if cx.stage != Stage::Fragment {
        return Err(cx.invariant("a color input outside a fragment function"));
    }
    let frag = cx.profile.fields().fragment;
    let AbiInt::Const(color) = *index else {
        return Err(not_lowered());
    };
    let form = param_form(cx, b)?;
    let extended = cx.inputs.extended_range_colors.get(&color).copied();
    // A word attachment's input reads the word. Where the interlocked storage form took the
    // attachment, its read decodes a packed 16-bit word, Metal's value, and refuses the other two
    // kinds (`fetch::select_read`); the input attachment form, whose attachment is the word
    // itself, decodes none.
    if cx.inputs.word_colors.contains_key(&color) && !super::fetch::is_storage(cx, color) {
        let refused = Unsupported::not_implemented(Feature::Format);
        return Err(refused.at_attachment(Attachment::Color(color)).into());
    }
    // A fetch of a bfloat attachment is unmeasured (its draws are built; what a fetch reads of the
    // `half`s Metal's tile holds is not).
    if extended.is_some_and(|e| e.form.bfloat()) {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    let input = ColorInput {
        param: b.param,
        color,
        form,
        extended: extended.map(|e| e.form),
        alpha_red: cx.inputs.alpha_in_red_colors.contains(&color),
    };
    // The interlocked storage form `fetch::plan` chose.
    if super::fetch::is_storage(cx, color) {
        super::fetch::select_read(cx, input)?;
        return Ok(input);
    }
    // On a device with the pixel interlock `fetch::plan` refused a blocked attachment already;
    // here neither ordered form exists.
    if !frag.rasterization_order_color_access {
        return Err(unordered(cx, color));
    }
    // Two reads of one attachment must agree on its numeric class (one image declaration).
    let known = &cx.families.fragment.colors;
    if known
        .iter()
        .any(|c| c.color == color && c.form.is_float() != form.is_float())
    {
        return Err(not_lowered());
    }
    let blended = cx
        .families
        .fragment
        .input_blends
        .get(&color)
        .is_some_and(|ib| ib.blend.is_some());
    if !known.iter().any(|c| c.color == color) && !blended {
        cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
        cx.facts.insert(RuntimeFact::FramebufferFetch { color });
    }
    // An extended-range attachment's input reads its raw texels, decoded to the tile value, a
    // float (an integer read is Metal's pipeline error). The interface records the attachment,
    // which link holds the pipeline's to (`color_writes::check`).
    if let Some(e) = extended {
        if !form.is_float() {
            return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
            }));
        }
        // A padded 10-bit attachment's tile value is not measured.
        if e.form.padded() {
            let refused = Unsupported::not_implemented(Feature::FramebufferFetch);
            return Err(refused.at_attachment(Attachment::Color(color)).into());
        }
        cx.families.arith.converts(e.form);
        cx.families
            .stage_io
            .interface
            .extended_range_colors
            .insert(color, e);
    }
    // A `float` color the device's input attachment decodes, an 8-bit sRGB format's through its
    // sRGB EOTF: Metal's decode on a device that decodes through Metal's own texture unit, which
    // the module then requires; elsewhere the interface records the color, whose pipeline notes the
    // decode where the attachment is such a format (`color_writes::pipeline_writes`).
    if form.is_float() && extended.is_none() && !input.alpha_red {
        let metal = cx
            .profile
            .fields()
            .enabled
            .properties
            .contains(&crate::device::requirements::VkProperty::MetalSampler);
        match metal {
            true => cx.families.sampler.metal_sampler = true,
            false => {
                cx.families.stage_io.interface.decoded_fetches.insert(color);
            }
        }
    }
    // An attachment whose image holds Metal's alpha in red reads the red as Metal's alpha: a float
    // (an integer read is Metal's pipeline error). The interface records the attachment, which
    // link holds the pipeline's to (`color_writes::check`).
    if input.alpha_red {
        if !form.is_float() {
            return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
            }));
        }
        cx.families
            .stage_io
            .interface
            .alpha_in_red_colors
            .insert(color);
    }
    super::fetch::plan_sample(cx)?;
    Ok(input)
}

/// The refusal of a read of color attachment `color` that no ordered form serves on the device:
/// either rasterization-order access, or the pixel interlock (with fragment stores where the
/// function writes the attachment) for the storage form.
fn unordered(cx: &SelectCx<'_>, color: u32) -> Error {
    let f = cx.profile.fields();
    let writes = cx.entry.ep.outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                index: 0,
                ..
            } if a == color
        )
    });
    let mut interlock = Vec::new();
    if !f.fragment.pixel_interlock {
        interlock.push(Capability::Feature(VkFeature::FragmentShaderPixelInterlock));
    }
    if writes && !f.leaf.fragment_stores_and_atomics {
        interlock.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    Unsupported::needs_any_of(
        Feature::FramebufferFetch,
        vec![
            vec![Capability::Feature(
                VkFeature::RasterizationOrderColorAttachmentAccess,
            )],
            interlock,
        ],
    )
    .into()
}

/// How many uses the body makes of the planned color inputs.
pub(super) fn uses(cx: &SelectCx<'_>, colors: &[ColorInput]) -> u64 {
    let mut uses = 0u64;
    for inst in cx.body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                if colors.iter().any(|c| c.param == a.raw()) {
                    uses += 1;
                }
            }
        });
    }
    uses
}

/// The words the planned input attachments add: one object per attachment and one read per use
/// (the storage form's are `fetch::words`).
pub(super) fn words(cx: &SelectCx<'_>, colors: &[ColorInput]) -> u64 {
    let uses = uses(cx, colors);
    // An extended-range attachment's read decodes its raw texel.
    let extended: Vec<ColorInput> = colors
        .iter()
        .filter(|c| c.extended.is_some())
        .copied()
        .collect();
    let decodes = self::uses(cx, &extended);
    // An alpha-in-red attachment's read moves its red to alpha: a shuffle and its zero.
    let red: Vec<ColorInput> = colors.iter().filter(|c| c.alpha_red).copied().collect();
    let moves = self::uses(cx, &red);
    let mut attachments: Vec<u32> = colors.iter().map(|c| c.color).collect();
    attachments.sort_unstable();
    attachments.dedup();
    OBJECT_WORDS
        .saturating_mul(attachments.len() as u64)
        .saturating_add(READ_WORDS.saturating_mul(uses))
        .saturating_add(u64::from(extended_range::DECODE_WORDS + 120).saturating_mul(decodes))
        .saturating_add(ALPHA_IN_RED_WORDS.saturating_mul(moves))
}

/// An upper estimate of the words [`alpha_in_red`] adds: the shuffle, and the zero vector's
/// constant and type.
pub(super) const ALPHA_IN_RED_WORDS: u64 = 16;

/// Declare one input attachment per color index the plan reads, multisampled for a per-sample
/// function (`multisampled`), or its storage image in the interlocked storage form (`fetch`), which
/// holds the samples as its view states; the storage image of every other attachment in that form
/// (one the function only writes); the input attachment of every extended-range attachment whose
/// blend the function computes over it (`input_blend`); and that of every word attachment whose
/// word the function merges into the texel's (`word_input`), of `uint` texels.
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    colors: &[ColorInput],
    (storage, blends, merges): (
        &BTreeMap<u32, super::fetch::StorageColor>,
        &BTreeMap<u32, super::InputBlend>,
        &BTreeMap<u32, u32>,
    ),
    multisampled: bool,
) -> BResult<BTreeMap<u32, GlobalId>> {
    let mut out = BTreeMap::new();
    for c in colors {
        if out.contains_key(&c.color) {
            continue;
        }
        if let Some(sc) = storage.get(&c.color) {
            out.insert(c.color, super::fetch::declare(mb, c.color, sc)?);
            continue;
        }
        // An extended-range attachment's image holds the raw format's `uint` texels.
        let float = c.form.is_float() && c.extended.is_none();
        out.insert(c.color, input_attachment(mb, c.color, float, multisampled)?);
    }
    // A storage-form attachment the function writes without reading it: an extended-range
    // attachment whose blend the shader computes from the texel.
    for (&color, sc) in storage {
        if let std::collections::btree_map::Entry::Vacant(e) = out.entry(color) {
            e.insert(super::fetch::declare(mb, color, sc)?);
        }
    }
    let blended = blends
        .iter()
        .filter(|(_, ib)| ib.blend.is_some())
        .map(|(&c, _)| c);
    for color in blended.chain(merges.keys().copied()) {
        if let std::collections::btree_map::Entry::Vacant(e) = out.entry(color) {
            e.insert(input_attachment(mb, color, false, multisampled)?);
        }
    }
    Ok(out)
}

/// The input attachment of color `color`, of `float` texels or (not `float`) `uint` ones,
/// multisampled for a per-sample function.
fn input_attachment(
    mb: &mut ModuleBuilder,
    color: u32,
    float: bool,
    multisampled: bool,
) -> BResult<GlobalId> {
    let sampled_type = match float {
        true => mb.ty(SType::Float { width: 32 })?,
        false => mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?,
    };
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::SubpassData,
        depth: false,
        arrayed: false,
        multisampled,
        usage: ImageUsage::Storage,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::InputAttachment(color),
        index: color,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
}

/// The value of color input `c` in its lowered representation, read at each use (the use may
/// sit in a construct whose values do not dominate the rest of the block), at the invocation's
/// sample for a per-sample function.
pub(super) fn value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: ColorInput,
) -> BResult<SValue> {
    if cx.plan.families.fragment.fetch.contains_key(&c.color) {
        return super::fetch::value(cx, fb, c);
    }
    let texel = input_texel(cx, fb, c.color)?;
    let texel = match c.extended {
        Some(form) => extended_range::tile_value(fb, form, texel)?,
        // The image's red, Metal's alpha: `(0, 0, 0, r)`.
        None if c.alpha_red => alpha_in_red(fb, texel)?,
        None => texel,
    };
    let x = match c.form.lanes {
        4 => texel,
        1 => fb.composite_extract(texel, &[0])?,
        n => {
            let lanes: Vec<u32> = (0..u32::from(n)).collect();
            fb.vector_shuffle(texel, texel, &lanes)?
        }
    };
    let x = ops::mask(fb, c.form, x)?;
    narrow::round(fb, c.form, x)
}

/// Metal's texel of an `A8Unorm` attachment whose image holds Metal's alpha in red, from the
/// image's `texel` (a `vec4`, `(r, 0, 0, 1)`): `(0, 0, 0, r)`, as Metal reads such an attachment
/// (probe `a8`'s uses on an M3 Max under macOS 27.0.1: a fetch of the byte 153 is
/// `(0, 0, 0, 0.6)`, a scalar or two-lane fetch its leading lanes).
pub(super) fn alpha_in_red(fb: &mut FunctionBuilder<'_>, texel: SValue) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let vec4 = fb.module().ty(SType::Vector { elem: f32_, len: 4 })?;
    let zero = fb.module().constant(Constant::Null { ty: vec4 })?;
    let zero = fb.module().const_value(zero)?;
    fb.vector_shuffle(zero, texel, &[0, 1, 2, 4])
}

/// The texel of color `color`'s input attachment, as the image reads it: at the calling
/// fragment's own texel, at the invocation's sample for a per-sample function.
pub(super) fn input_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
) -> BResult<SValue> {
    let Some(g) = cx.decl.families.fragment.attachments.get(&color).copied() else {
        return Err(cx.fail(format!("input attachment {color} is not declared")));
    };
    let p = fb.module().global_value(g)?;
    let img = fb.load(p, MemAccess::default())?;
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let ivec2 = fb.module().ty(SType::Vector { elem: i32_, len: 2 })?;
    let origin = fb.module().constant(Constant::Null { ty: ivec2 })?;
    let origin = fb.module().const_value(origin)?;
    let ops = super::fetch::operands(cx, fb)?;
    fb.read(img, origin, &ops, TexelMemory::PRIVATE)
}
