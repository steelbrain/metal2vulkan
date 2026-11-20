//! A fragment function that runs its draw's depth test itself, inside the ordered pixel
//! interlock, against a stand-in of the depth attachment.
//!
//! **Where.** A function that writes a color through the interlocked storage form stores inside the
//! shader, before Vulkan's tests. Where its draws' tests may drop a fragment and it cannot declare
//! early tests (`RuntimeFact::TestedStorageWrites`: it discards under tests that write, writes
//! other memory, or runs on a driver that drops an early-tests shader's image writes at more than
//! one sample), a store before the tests keeps colors Metal's tests drop. Compiled for a draw whose
//! depth test alone can drop a fragment (`LoweringInputs::shader_tests`, the binder's
//! `FragmentTests::Depth`), the function runs that test at its return instead, next to its color
//! store, as Metal tests after the function:
//!
//! - the consumer keeps the depth attachment in the subpass under `Always` without writes, and
//!   holds its depth in a stand-in (`GraphicsState::shader_tests`, `SyntheticBinding::DepthStandIn`):
//!   an `R32_SFLOAT` storage image of the pass's size, texel `(x, y)` for pixel `(x, y)`, copied
//!   from the attachment before the draws and back after them;
//! - at each return, unless a discard was recorded, the function takes the fragment's depth,
//!   clamps it as Metal does by the draw's depth clip mode (to `[0, 1]` in clip mode, to the
//!   viewport's range `[min(near, far), max(near, far)]` in clamp mode; a `-0` as `+0`), compares
//!   it with the stand-in's texel by the draw's compare function, writes it there where it passes
//!   and the draw writes depth, and records a failing fragment as discarded, so the storage form's
//!   color store skips it and the entry kills it after the critical section;
//! - the draw's depth state is `PushField::ShaderTestsState` (see its docs).
//!
//! The ordered interlock makes each fragment's test, depth write and color store one step in
//! primitive order across the draws of a subpass, so the next fragment tests against the depth
//! the last one left: Metal's order. The body runs whole, every memory write included, for every
//! fragment, as Metal runs it before its tests.
//!
//! **The fragment's depth**. At one sample it is `FragCoord.z`: the device's own depth, which its
//! fixed-function test compares, with the draw's depth bias, at the pixel's one sample (on the M4,
//! Metal's `[[position]].z`, equal bit for bit to the depth Metal tests once clamped; on ANV, RADV
//! and NVIDIA the depth the device's own draw stores).
//!
//! **More than one sample.** The function runs once per covered sample (its `SampleId`) and tests
//! its sample of a stand-in holding each: a multisampled storage image, or, on a device without
//! one, sample planes (a 2D array, layer `s` for sample `s`). A device whose `FragCoord.z` there is
//! its sample's, the depth its own test stores at that sample (RADV, measured,
//! `VkProperty::SampleFragCoordDepth`), tests `FragCoord.z` as at one sample. A device whose
//! `FragCoord.z` is the pixel centre's there while each sample stores its own (MoltenVK, measured,
//! `VkProperty::CenterFragCoordDepth`) has the function compute each sample's depth: its vertex
//! function writes the position's window depth `z / w` to a no-perspective varying
//! (`IoRole::WindowDepth`, `LoweringInputs::window_depth_varying`), and the function takes
//! `v(sample) · (far − near) + near`, plus `FragCoord.z` less `v(centre) · (far − near) + near`,
//! which carries the draw's depth bias, each operation rounded on its own: `v(sample)` read through
//! `InterpolateAtSample` at the invocation's sample, `v(centre)` read as it is, which that device
//! reads at the pixel centre. On the M4 that is the depth MoltenVK stores at each sample in the
//! flat, sloped and steep cases, under a constant bias, and where the clamp decides; under
//! perspective, a slope-scaled bias, or a viewport range other than `[0, 1]` it is within 2 ulp.
//! The module records `PrecisionNote::ComputedSampleDepth`, which no draw's
//! state can rule out at compile time. Both properties were measured facts, which no query
//! produces, so on every device a function that does not return its depth is refused at more than
//! one sample, as its own kind (`Feature::MultisampledShaderDepthTest`).
//!
//! **A returned depth**. A function that returns `[[depth(...)]]` tests the depth it returns at
//! that return, in place of `FragCoord.z`, as Metal tests it: a positive NaN fails the test,
//! `Always` included, which the return's store of the depth output records as discarded before the
//! test (`stage_io::clamp_depth`); any other value with its sign bit set is +0 (a negative NaN
//! stays 0 in clamp mode, past the range); a value past 1 is 1; then, in clamp mode, the viewport's
//! range clamps it. It takes no depth bias, and is the same at every sample, so it needs neither of
//! the per-sample depth sources above.
//!
//! **The stencil test**. Compiled for a draw whose stencil test can drop a fragment or write too
//! (`LoweringInputs::shader_stencil`, the binder's `FragmentTests::Stencil`, in a pass with a depth
//! and a stencil attachment), the function runs it first, as Metal does, against a stand-in of the
//! stencil (`SyntheticBinding::StencilStandIn`): an `R8_UINT` storage image of the pass's size,
//! texel `(x, y)` for pixel `(x, y)`, each sample at more than one (multisampled; sample planes are
//! not implemented). At each return, unless a discard was recorded, the face's state (`FrontFacing`
//! selects the front or back word and reference of `PushField::ShaderStencilState`) compares the
//! reference with the stand-in's value under the read mask; the face's stencil-failure operation
//! applies where that fails, its depth-failure operation where the depth test fails, and its pass
//! operation where both pass, under the write mask, as `s' = (s & !mask) | (op(s, r) & mask)` with
//! Metal's clamps and wraps (measured: Vulkan's rule); and only a fragment that passes both tests
//! writes its depth and stores its colors. The stand-in's read, operation and write are one step of
//! the critical section, as the depth's are.
//!
//! **A returned stencil and sample mask**. A function that returns `[[stencil]]` tests with the
//! output's low 8 bits as both faces' reference, in place of the draw's (measured), its stencil
//! output stored nowhere else: the device's stencil test is off for such draws (`stage_io`), and
//! where their stencil test does nothing the output changes nothing. A function that returns
//! `[[sample_mask]]` keeps a fragment only where the mask holds the invocation's sample's bit (bit
//! 0 at one sample; a per-sample invocation's other bits do nothing): at each return, before the
//! tests, a fragment without it is recorded as discarded, so it runs no test, writes neither
//! stand-in, and stores no color, as Metal drops the samples outside the mask (measured on an M4).
//! Beside a returned stencil Metal still runs the stencil test for such a sample, with the returned
//! reference, and fails its depth test whatever the draw's compare function: the stencil-failure
//! operation where the stencil test fails, else the depth-failure operation, no depth write and no
//! color. So does the function, where it runs the stencil test.

use crate::device::requirements::{FormatUsage, VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{Attachment, Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{self, Returned};
use crate::lower::sync::memory_model;
use crate::metal::format::{spirv_image_format, without_format_gaps, StorageUse};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, GlslOp, ImageOperands, ImageType, ImageUsage,
    MemAccess, StorageClass, Type as SType, UnOp,
};
use crate::vk;

use super::fetch::StorageSamples;
use super::stencil::{self, SHADOW_FORMAT};

/// The depth stand-in's format: Metal's depth values are 32-bit floats.
pub(crate) const STAND_IN_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

/// Bit 3 of the state's word: the draw writes depth.
const WRITE_BIT: u32 = 1 << 3;
/// Bit 4 of the state's word: the draw's depth clip mode is `MTLDepthClipModeClamp`.
const CLAMP_BIT: u32 = 1 << 4;

/// The first word of `PushField::ShaderTestsState` (see its docs): Metal's depth compare function
/// `compare` by its raw value, [`WRITE_BIT`] where the draw writes depth, and [`CLAMP_BIT`] in
/// depth clamp mode.
pub(crate) fn depth_test_word(compare: u32, write: bool, clamp: bool) -> u32 {
    (compare & 7) | if write { WRITE_BIT } else { 0 } | if clamp { CLAMP_BIT } else { 0 }
}

/// The word of one face of `PushField::ShaderStencilState` (see its docs): the shader-side
/// stencil's face word (`stencil::face_word`, its attached bit clear) of Metal's compare function
/// `compare`, operations `stencil_failure` and `pass`, and masks `read_mask` and `write_mask`, with
/// the depth-failure operation `depth_failure` in bits 25 to 27.
pub(crate) fn stencil_test_word(
    compare: u32,
    stencil_failure: u32,
    depth_failure: u32,
    pass: u32,
    read_mask: u32,
    write_mask: u32,
) -> u32 {
    stencil::face_word(compare, stencil_failure, pass, read_mask, write_mask, false)
        | (depth_failure & 7) << 25
}

/// The in-shader depth test's form on a device: how the module declares the stand-in and holds
/// its samples, and where the fragment's depth comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DepthForm {
    /// The declared SPIR-V image format: `R32f`, or `Unknown` for a formatless declaration.
    pub declared: crate::spv::ImageFormat,
    /// How the stand-in holds the pipeline's samples.
    pub samples: StorageSamples,
    /// Each sample's depth is computed from the vertex function's window depth (see the module
    /// docs), which the vertex function then writes (`LoweringInputs::window_depth_varying`);
    /// otherwise the depth is `FragCoord.z`.
    pub computed: bool,
}

/// The in-shader depth test's plan: how the module declares the stand-in and holds its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthTest {
    /// The declared SPIR-V image format: `R32f`, or `Unknown` for a formatless declaration.
    pub declared: crate::spv::ImageFormat,
    /// How the stand-in holds the pipeline's samples.
    pub samples: StorageSamples,
    /// The stage interface's window-depth input object (`IoRole::WindowDepth`), from which each
    /// sample's depth is computed; `None` where the depth is `FragCoord.z` or returned.
    pub window: Option<usize>,
    /// The returned value's field that holds the function's `[[depth(...)]]` output (0 for a
    /// returned value itself), which the test takes in place of `FragCoord.z`; `None` where the
    /// function returns no depth.
    pub returned: Option<u32>,
    /// The stencil test the function runs first (see the module docs); `None` where the draws it
    /// serves have a stencil test that does nothing.
    pub stencil: Option<StencilTest>,
    /// The returned value's field that holds the function's `[[sample_mask]]` output, whose bit of
    /// the invocation's sample (bit 0 at one sample) keeps the fragment before the tests; `None`
    /// where the function returns no sample mask.
    pub mask: Option<u32>,
}

/// The in-shader stencil test's plan: how the module declares the stencil's stand-in (of the
/// depth test's samples, which are then one or multisampled), and the face it tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilTest {
    /// The declared SPIR-V image format: `R8ui`, or `Unknown` for a formatless declaration.
    pub declared: crate::spv::ImageFormat,
    /// The stage interface's `FrontFacing` object, which selects the face's state.
    pub front_facing: usize,
    /// The returned value's field that holds the function's `[[stencil]]` output, whose low 8
    /// bits are both faces' reference in place of the draw's; `None` where the function returns
    /// no stencil.
    pub returned: Option<u32>,
}

fn not_implemented() -> Unsupported {
    Unsupported::not_implemented(Feature::FramebufferFetch)
}

/// The in-shader depth test's form on `p` for a pipeline of more than one sample where
/// `multisampled`, or what the device lacks for it (see the module docs): the ordered pixel
/// interlock, fragment stores, the stand-in's format as a storage image, declared `R32f` (a base
/// storage format) or without a format where the device grants that access; at more than one sample
/// `sampleRateShading`, and each sample's depth: `FragCoord.z` where the device is measured to give
/// its sample's (`VkProperty::SampleFragCoordDepth`), else computed where it is measured to give
/// the centre's (`VkProperty::CenterFragCoordDepth`), which also takes the interpolation functions
/// (on a portability-subset device `shaderSampleRateInterpolationFunctions`); elsewhere not
/// implemented. A function that returns its depth (`returned`) tests that, the same at every
/// sample, and needs no per-sample depth source.
pub(crate) fn depth_test_form(
    p: &DeviceProfile,
    multisampled: bool,
    returned: bool,
) -> Result<DepthForm, Unsupported> {
    let f = p.fields();
    let measured = |property| f.enabled.properties.contains(&property);
    let computed = match multisampled {
        false => false,
        true if returned => false,
        true if measured(VkProperty::SampleFragCoordDepth) => false,
        true if measured(VkProperty::CenterFragCoordDepth) => true,
        true => {
            return Err(Unsupported::not_implemented(
                Feature::MultisampledShaderDepthTest,
            ))
        }
    };
    let mut missing = Vec::new();
    if multisampled && !f.fragment.sample_rate_shading {
        missing.push(Capability::Feature(VkFeature::SampleRateShading));
    }
    if computed {
        let interpolation = VkFeature::ShaderSampleRateInterpolationFunctions;
        if f.enabled.portability_withholds(interpolation) {
            missing.push(Capability::Feature(interpolation));
        }
    }
    if !f.fragment.pixel_interlock {
        missing.push(Capability::Feature(VkFeature::FragmentShaderPixelInterlock));
    }
    if !f.leaf.fragment_stores_and_atomics {
        missing.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    if !p.supports_format(STAND_IN_FORMAT, FormatUsage::StorageImage) {
        missing.push(Capability::Format {
            format: STAND_IN_FORMAT,
            usage: FormatUsage::StorageImage,
        });
    }
    let typed = spirv_image_format(STAND_IN_FORMAT).filter(|d| {
        !crate::spirv::requirements::extended_format(*d) || f.images.storage_extended_formats
    });
    let declared = match typed {
        Some(d) => Some(d),
        None if without_format_gaps(STAND_IN_FORMAT, StorageUse::texture(true, true), p)
            .is_empty() =>
        {
            Some(crate::spv::ImageFormat::Unknown)
        }
        None => {
            missing.push(Capability::Feature(
                VkFeature::ShaderStorageImageExtendedFormats,
            ));
            None
        }
    };
    let samples = match (multisampled, f.images.storage_multisample) {
        (false, _) => StorageSamples::One,
        (true, true) => StorageSamples::Multisampled,
        (true, false) => StorageSamples::Planes,
    };
    match declared {
        Some(declared) if missing.is_empty() => Ok(DepthForm {
            declared,
            samples,
            computed,
        }),
        _ => Err(Unsupported::needs_any_of(
            Feature::FramebufferFetch,
            vec![missing],
        )),
    }
}

/// The in-shader stencil test's declaration of the stencil's stand-in on `p` for a pipeline of
/// more than one sample where `multisampled`, or what it lacks (see the module docs); the
/// interlock, the fragment stores and the per-sample invocation are the depth test's
/// ([`depth_test_form`]). The stand-in's format as a storage image, declared `R8ui` where the
/// device has the extended storage formats, else without a format where it grants that access; at
/// more than one sample a multisampled storage image (sample planes, on a device without one, are
/// not implemented), and `maintenance8`, whose depth/stencil and color copies move the samples
/// between the stencil aspect and the stand-in (as the shader-side stencil's shadow's).
pub(crate) fn stencil_test_form(
    p: &DeviceProfile,
    multisampled: bool,
) -> Result<crate::spv::ImageFormat, Unsupported> {
    let f = p.fields();
    if multisampled && !f.images.storage_multisample {
        return Err(not_implemented());
    }
    let mut missing = Vec::new();
    if multisampled && !f.enabled.features.contains(&VkFeature::Maintenance8) {
        missing.push(Capability::Feature(VkFeature::Maintenance8));
    }
    if !p.supports_format(SHADOW_FORMAT, FormatUsage::StorageImage) {
        missing.push(Capability::Format {
            format: SHADOW_FORMAT,
            usage: FormatUsage::StorageImage,
        });
    }
    let declared = if f.images.storage_extended_formats {
        Some(crate::spv::ImageFormat::R8ui)
    } else if without_format_gaps(SHADOW_FORMAT, StorageUse::texture(true, true), p).is_empty() {
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
            Feature::FramebufferFetch,
            vec![missing],
        )),
    }
}

/// The fields of a function's returned value that hold its exports, which the in-shader tests
/// take (see the module docs): `[[depth(...)]]`, `[[stencil]]` and `[[sample_mask]]`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Exports {
    pub depth: Option<u32>,
    pub stencil: Option<u32>,
    pub mask: Option<u32>,
}

/// Plan the in-shader depth test of a function that stores a color before the tests, compiled
/// for a draw whose depth test alone can drop a fragment (`LoweringInputs::shader_tests`), which
/// returns its depth, stencil and sample mask in the fields of its returned value `exports` names
/// (see the module docs). A refusal is placed at the depth attachment. Compiled for a draw whose
/// stencil test acts too (`LoweringInputs::shader_stencil`), plan the stencil test as well; a
/// refusal of its form is placed at the stencil attachment.
pub(super) fn plan(cx: &mut SelectCx<'_>, exports: Exports) -> Result<(), Error> {
    let at_depth = |u: Unsupported| Error::from(u).at_attachment(Attachment::Depth);
    let returned = exports.depth;
    let form = depth_test_form(cx.profile, cx.inputs.multisampled, returned.is_some())
        .map_err(at_depth)?;
    let coord = stage_io::frag_coord(cx);
    // At more than one sample the function runs once per sample and tests its own (`SampleId`).
    super::fetch::plan_sample(cx)?;
    // Each sample's depth from the vertex function's window depth, where `FragCoord.z` is the
    // pixel centre's; else `FragCoord.z`.
    let window = match form.computed {
        true => {
            cx.precision.insert(PrecisionNote::ComputedSampleDepth);
            Some(stage_io::window_depth_input(cx)?)
        }
        false => None,
    };
    let stencil = match cx.inputs.shader_stencil {
        true => {
            let declared = stencil_test_form(cx.profile, cx.inputs.multisampled)
                .map_err(|u| Error::from(u).at_attachment(Attachment::Stencil))?;
            let front_facing = stage_io::front_facing(cx);
            cx.facts.insert(RuntimeFact::ShaderStencilTest {
                formatless: declared == crate::spv::ImageFormat::Unknown,
            });
            cx.push.insert(PushField::ShaderStencilState);
            cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(1);
            Some(StencilTest {
                declared,
                front_facing,
                returned: exports.stencil,
            })
        }
        false => None,
    };
    let fp = &mut cx.families.fragment;
    fp.fetch_coord = Some(coord);
    fp.depth_test = Some(DepthTest {
        declared: form.declared,
        samples: form.samples,
        window,
        returned,
        stencil,
        mask: exports.mask,
    });
    cx.facts.insert(RuntimeFact::ShaderTests {
        formatless: form.declared == crate::spv::ImageFormat::Unknown,
        multisampled: cx.inputs.multisampled,
        planes: form.samples == StorageSamples::Planes,
    });
    cx.push.insert(PushField::ShaderTestsState);
    cx.descriptors.storage_images = cx.descriptors.storage_images.saturating_add(1);
    Ok(())
}

/// An upper estimate of the words the test adds: the stand-in and its type, and at each of
/// `rets` returns the guard, the state's fields, the depth (computed per sample, with its two
/// reads of the window depth, or returned, with Metal's rule for its sign, NaN and range) and
/// its clamp, the comparison's selections, the read, and the write; with the stencil test, its
/// stand-in and type, and at each return the face's state, the comparison's and operations'
/// selections, the read, and the write; and with a returned sample mask, at each return its bit's
/// test and the drop.
pub(super) fn words(cx: &SelectCx<'_>, rets: u64) -> u64 {
    const OBJECT: u64 = 40;
    const RETURN: u64 = 400;
    const COMPUTED: u64 = 120;
    const RETURNED: u64 = 160;
    const STENCIL: u64 = 520;
    const MASK: u64 = 80;
    let Some(plan) = cx.families.fragment.depth_test else {
        return 0;
    };
    let per_return = match (plan.window, plan.returned) {
        (Some(_), _) => RETURN + COMPUTED,
        (None, Some(_)) => RETURN + RETURNED,
        (None, None) => RETURN,
    };
    let (objects, per_return) = match plan.stencil {
        Some(_) => (2 * OBJECT, per_return + STENCIL),
        None => (OBJECT, per_return),
    };
    let per_return = match plan.mask {
        Some(_) => per_return + MASK,
        None => per_return,
    };
    objects.saturating_add(rets.saturating_mul(per_return))
}

/// Declare the stand-in: a 2D `float` storage image of the planned declaration, multisampled or
/// arrayed as it holds the samples ([`StorageSamples`]).
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    plan: Option<&DepthTest>,
) -> BResult<Option<GlobalId>> {
    let Some(plan) = plan else {
        return Ok(None);
    };
    let sampled_type = mb.ty(SType::Float { width: 32 })?;
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: plan.samples == StorageSamples::Planes,
        multisampled: plan.samples == StorageSamples::Multisampled,
        usage: ImageUsage::Storage,
        format: plan.declared,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::DepthStandIn),
        index: 0,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
        .map(Some)
}

/// Declare the stencil's stand-in, where the test runs the stencil test: a 2D `uint` storage
/// image of the planned declaration, multisampled where the depth's stand-in is.
pub(super) fn declare_stencil(
    mb: &mut ModuleBuilder,
    plan: Option<&DepthTest>,
) -> BResult<Option<GlobalId>> {
    let Some((plan, stencil)) = plan.and_then(|p| Some((p, p.stencil?))) else {
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
        multisampled: plan.samples == StorageSamples::Multisampled,
        usage: ImageUsage::Storage,
        format: stencil.declared,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::StencilStandIn),
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

/// An `f32` constant.
fn f32_value(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}

/// Whether `a cmp b` under Metal's compare function `code` (`MTLCompareFunction`: never, less,
/// equal, less-equal, greater, not-equal, greater-equal, always), with `a` the fragment's clamped
/// depth and `b` the stand-in's: neither is a NaN.
fn compare(fb: &mut FunctionBuilder<'_>, code: SValue, a: SValue, b: SValue) -> BResult<SValue> {
    let never = super::bool_const(fb, false)?;
    let always = super::bool_const(fb, true)?;
    let less = fb.binary(BinOp::FOrdLessThan, a, b)?;
    let equal = fb.binary(BinOp::FOrdEqual, a, b)?;
    let less_equal = fb.binary(BinOp::FOrdLessThanEqual, a, b)?;
    let greater = fb.binary(BinOp::FOrdGreaterThan, a, b)?;
    let not_equal = fb.binary(BinOp::FOrdNotEqual, a, b)?;
    let greater_equal = fb.binary(BinOp::FOrdGreaterThanEqual, a, b)?;
    let choices = [
        never,
        less,
        equal,
        less_equal,
        greater,
        not_equal,
        greater_equal,
        always,
    ];
    let mut out = choices[7];
    for (i, &choice) in choices.iter().enumerate().take(7).rev() {
        let k = u32_value(fb, i as u32)?;
        let is = fb.binary(BinOp::IEqual, code, k)?;
        out = fb.select(is, choice, out)?;
    }
    Ok(out)
}

/// Whether bit `bit` of `word` is set.
fn bit_set(fb: &mut FunctionBuilder<'_>, word: SValue, bit: u32) -> BResult<SValue> {
    let b = u32_value(fb, bit)?;
    let masked = fb.binary(BinOp::BitwiseAnd, word, b)?;
    let zero = u32_value(fb, 0)?;
    fb.binary(BinOp::INotEqual, masked, zero)
}

/// Depth `z` clamped as Metal clamps the depth it tests and stores, under the state's word `word`
/// and its depth range `near`, `far`: to the viewport's range `[min(near, far), max(near, far)]` in
/// clamp mode, to `[0, 1]` in clip mode; a `-0` as `+0`, on the bits, which no float optimization
/// folds away.
fn clamped(
    fb: &mut FunctionBuilder<'_>,
    z: SValue,
    word: SValue,
    (near, far): (SValue, SValue),
) -> BResult<SValue> {
    let clamp_mode = bit_set(fb, word, CLAMP_BIT)?;
    let low = fb.glsl(GlslOp::FMin, &[near, far])?;
    let high = fb.glsl(GlslOp::FMax, &[near, far])?;
    let zero = f32_value(fb, 0.0)?;
    let one = f32_value(fb, 1.0)?;
    let low = fb.select(clamp_mode, low, zero)?;
    let high = fb.select(clamp_mode, high, one)?;
    let z = fb.glsl(GlslOp::FClamp, &[z, low, high])?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let bits = fb.convert(ConvOp::Bitcast, z, u32_)?;
    let negative_zero = u32_value(fb, 0x8000_0000)?;
    let is_negative_zero = fb.binary(BinOp::IEqual, bits, negative_zero)?;
    let zero_bits = u32_value(fb, 0)?;
    let bits = fb.select(is_negative_zero, zero_bits, bits)?;
    fb.convert(ConvOp::Bitcast, bits, f32_)
}

/// A returned depth `z` as Metal tests and stores it, under the state's word `word` and its depth
/// range `near`, `far`: a value with its sign bit set is +0 (a negative NaN stays 0 in clamp mode,
/// past the range); a value past 1, `+inf` included, is 1; then, in clamp mode, the range
/// `[min(near, far), max(near, far)]` clamps it, a `-0` as `+0`. The sign and the unit clamp are
/// decided on the bits, so a positive denormal is kept. A positive NaN, which fails Metal's test
/// whatever the compare function, never reaches it: the return's store of the depth output recorded
/// it as discarded (`stage_io::clamp_depth`).
fn returned_depth(
    fb: &mut FunctionBuilder<'_>,
    z: SValue,
    word: SValue,
    (near, far): (SValue, SValue),
) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let bits = fb.convert(ConvOp::Bitcast, z, u32_)?;
    let magnitude_mask = u32_value(fb, 0x7FFF_FFFF)?;
    let magnitude = fb.binary(BinOp::BitwiseAnd, bits, magnitude_mask)?;
    let infinity = u32_value(fb, 0x7F80_0000)?;
    let nan = fb.binary(BinOp::UGreaterThan, magnitude, infinity)?;
    let sign_bit = u32_value(fb, 0x8000_0000)?;
    let negative = fb.binary(BinOp::UGreaterThanEqual, bits, sign_bit)?;
    let negative_nan = fb.binary(BinOp::LogicalAnd, nan, negative)?;
    let one = u32_value(fb, 0x3F80_0000)?;
    let past_one = fb.binary(BinOp::UGreaterThan, bits, one)?;
    let unit = fb.select(past_one, one, bits)?;
    let zero = u32_value(fb, 0)?;
    let unit = fb.select(negative, zero, unit)?;
    let unit = fb.convert(ConvOp::Bitcast, unit, f32_)?;
    let clamp_mode = bit_set(fb, word, CLAMP_BIT)?;
    let low = fb.glsl(GlslOp::FMin, &[near, far])?;
    let high = fb.glsl(GlslOp::FMax, &[near, far])?;
    let ranged = fb.glsl(GlslOp::FClamp, &[unit, low, high])?;
    let not_nan = fb.unary(UnOp::LogicalNot, negative_nan)?;
    let ranges = fb.binary(BinOp::LogicalAnd, clamp_mode, not_nan)?;
    let z = fb.select(ranges, ranged, unit)?;
    let bits = fb.convert(ConvOp::Bitcast, z, u32_)?;
    let is_negative_zero = fb.binary(BinOp::IEqual, bits, sign_bit)?;
    let bits = fb.select(is_negative_zero, zero, bits)?;
    fb.convert(ConvOp::Bitcast, bits, f32_)
}

/// At a return of value `v`: unless a discard was recorded, test the fragment's clamped depth (its
/// returned one where it returns depth) against the stand-in's texel at its pixel by the draw's
/// compare function, run the stencil test where the plan has one (its comparison, then the
/// operation the two tests' results select, written to the stencil's stand-in), write the depth
/// where both pass and the draw writes depth, and record a failing fragment as discarded (see the
/// module docs). Called before the storage form's color store, which a recorded discard skips.
pub(super) fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
) -> BResult<()> {
    let Some(plan) = cx.plan.families.fragment.depth_test else {
        return Ok(());
    };
    let Some(stand_in) = cx.decl.families.fragment.depth_stand_in else {
        return Err(cx.fail("the depth stand-in is not declared"));
    };
    let Some(discarded) = cx.decl.families.fragment.discarded else {
        return Err(cx.fail("an in-shader depth test without its discard predicate"));
    };
    let Some(coord) = cx.plan.families.fragment.fetch_coord else {
        return Err(cx.fail("an in-shader depth test without its FragCoord"));
    };
    let state = cx.push_field(fb, PushField::ShaderTestsState)?;
    let (pixel, operands) = texel(cx, fb, plan.samples)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let frag = stage_io::object_global(cx, coord)?;
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, MemAccess::default())?;
    let depth = fb.composite_extract(frag, &[2])?;
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let word = fb.composite_extract(state, &[0, 0])?;
    let near = fb.composite_extract(state, &[0, 1])?;
    let near = fb.convert(ConvOp::Bitcast, near, f32_)?;
    let far = fb.composite_extract(state, &[0, 2])?;
    let far = fb.convert(ConvOp::Bitcast, far, f32_)?;
    let returned = match plan.returned {
        Some(field) => Some(exported(cx, fb, v, field)?),
        None => None,
    };
    let depth = match (plan.window, returned) {
        (_, Some(returned)) => returned,
        (Some(window), None) => sample_depth(cx, fb, window, depth, (near, far))?,
        (None, None) => depth,
    };
    // The stencil test's stand-in, the face's word and reference.
    let stencil = match plan.stencil {
        Some(s) => {
            let Some(image) = cx.decl.families.fragment.stencil_stand_in else {
                return Err(cx.fail("the stencil stand-in is not declared"));
            };
            let state = cx.push_field(fb, PushField::ShaderStencilState)?;
            let facing = stage_io::object_global(cx, s.front_facing)?;
            let facing = fb.module().global_value(facing)?;
            let front = fb.load(facing, MemAccess::default())?;
            let front_word = fb.composite_extract(state, &[0, 0])?;
            let back_word = fb.composite_extract(state, &[0, 1])?;
            let face = fb.select(front, front_word, back_word)?;
            // A returned stencil's low 8 bits are both faces' reference, in place of the draw's.
            let reference = match s.returned {
                Some(field) => exported(cx, fb, v, field)?,
                None => {
                    let references = fb.composite_extract(state, &[0, 2])?;
                    let zero = u32_value(fb, 0)?;
                    let eight = u32_value(fb, 8)?;
                    let shift = fb.select(front, zero, eight)?;
                    fb.binary(BinOp::ShiftRightLogical, references, shift)?
                }
            };
            let max = u32_value(fb, 0xFF)?;
            let reference = fb.binary(BinOp::BitwiseAnd, reference, max)?;
            Some(StencilStep {
                image,
                face,
                reference,
            })
        }
        None => None,
    };
    let p = fb.module().global_value(discarded)?;
    let was = fb.load(p, MemAccess::default())?;
    // A returned sample mask without the invocation's sample's bit (bit 0 at one sample) drops
    // the fragment before the tests, as Metal's coverage does: no test, no write, no store.
    // Beside a returned stencil the stencil test still runs for it, its depth test failing
    // (measured): `uncovered` reaches the tests.
    let returned_stencil = plan.stencil.is_some_and(|t| t.returned.is_some());
    let (was, uncovered) = match plan.mask {
        Some(field) => {
            let mask = exported(cx, fb, v, field)?;
            let sample = match plan.samples {
                StorageSamples::One => u32_value(fb, 0)?,
                StorageSamples::Multisampled | StorageSamples::Planes => {
                    match super::fetch::sample(cx, fb)? {
                        Some(sample) => sample,
                        None => return Err(cx.fail("a per-sample mask without its SampleId")),
                    }
                }
            };
            let bit = fb.binary(BinOp::ShiftRightLogical, mask, sample)?;
            let one = u32_value(fb, 1)?;
            let bit = fb.binary(BinOp::BitwiseAnd, bit, one)?;
            let zero = u32_value(fb, 0)?;
            let uncovered = fb.binary(BinOp::IEqual, bit, zero)?;
            match returned_stencil {
                true => (was, Some(uncovered)),
                false => {
                    let dropped = fb.binary(BinOp::LogicalOr, was, uncovered)?;
                    fb.store(p, dropped, MemAccess::default())?;
                    (dropped, None)
                }
            }
        }
        None => (was, None),
    };
    let kept = fb.unary(UnOp::LogicalNot, was)?;
    let is_returned = returned.is_some();
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let z = match is_returned {
            true => returned_depth(fb, depth, word, (near, far))?,
            false => clamped(fb, depth, word, (near, far))?,
        };
        let image = fb.module().global_value(stand_in)?;
        let image = fb.load(image, MemAccess::default())?;
        let texel = fb.read(image, pixel, &operands, mem)?;
        let stored = fb.composite_extract(texel, &[0])?;
        let seven = u32_value(fb, 7)?;
        let code = fb.binary(BinOp::BitwiseAnd, word, seven)?;
        let passed = compare(fb, code, z, stored)?;
        // Outside a returned mask, beside a returned stencil, the depth test fails.
        let passed = match uncovered {
            Some(uncovered) => {
                let covered = fb.unary(UnOp::LogicalNot, uncovered)?;
                fb.binary(BinOp::LogicalAnd, passed, covered)?
            }
            None => passed,
        };
        // The stencil test, first in Metal's order: its operation follows both results.
        let passed = match stencil {
            Some(step) => stencil_test(fb, step, (pixel, &operands, mem), passed)?,
            None => passed,
        };
        let writes = bit_set(fb, word, WRITE_BIT)?;
        let store = fb.binary(BinOp::LogicalAnd, passed, writes)?;
        let write: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let f32_ = fb.module().ty(SType::Float { width: 32 })?;
            let vec4 = fb.module().ty(SType::Vector { elem: f32_, len: 4 })?;
            let zero = f32_value(fb, 0.0)?;
            let texel = fb.composite_construct(vec4, &[z, zero, zero, zero])?;
            fb.write(image, pixel, texel, &operands, mem)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), store, write, None)?;
        // A failing fragment is dropped: the entry kills it after the critical section, and
        // the return's color store skips it.
        let failed = fb.unary(UnOp::LogicalNot, passed)?;
        let p = fb.module().global_value(discarded)?;
        fb.store(p, failed, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), kept, arm, None)?;
    Ok(())
}

/// Field `field` of the returned value `v` (the value itself where the entry returns one
/// output): an export the in-shader tests take.
fn exported(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
    field: u32,
) -> BResult<SValue> {
    let Some(v) = v else {
        return Err(cx.fail("a tested export without a returned value"));
    };
    let x = cx.operand(fb, v)?;
    match cx.plan.families.stage_io.returned {
        Returned::Struct => fb.composite_extract(x, &[field]),
        Returned::Value => Ok(x),
        Returned::Void => Err(cx.fail("a tested export of a void entry")),
    }
}

/// What the stencil test of one return reads: the stencil's stand-in (its global), and the face's
/// word and 8-bit reference.
#[derive(Clone, Copy)]
struct StencilStep {
    image: GlobalId,
    face: SValue,
    reference: SValue,
}

/// Metal's stencil test of the fragment at its texel `at` (the depth's stand-in's texel, image
/// operands and coherence: one sample, or the invocation's sample of a multisampled stand-in)
/// under `step`, after the depth test, which passed where `depth_passed` (see the module docs):
/// read the stand-in's value, compare the reference with it under the read mask, write the
/// stencil-failure, depth-failure or pass operation's value under the write mask. Whether both
/// tests passed.
fn stencil_test(
    fb: &mut FunctionBuilder<'_>,
    step: StencilStep,
    (pixel, operands, mem): (SValue, &ImageOperands, crate::spirv::ir::TexelMemory),
    depth_passed: SValue,
) -> BResult<SValue> {
    let StencilStep {
        image,
        face,
        reference,
    } = step;
    let image = fb.module().global_value(image)?;
    let image = fb.load(image, MemAccess::default())?;
    let texel = fb.read(image, pixel, operands, mem)?;
    let stored = fb.composite_extract(texel, &[0])?;
    let compare_code = stencil::field_of(fb, face, 0, 3)?;
    let fail_code = stencil::field_of(fb, face, 3, 3)?;
    let pass_code = stencil::field_of(fb, face, 6, 3)?;
    let read_mask = stencil::field_of(fb, face, 9, 8)?;
    let write_mask = stencil::field_of(fb, face, 17, 8)?;
    let depth_fail_code = stencil::field_of(fb, face, 25, 3)?;
    let a = fb.binary(BinOp::BitwiseAnd, reference, read_mask)?;
    let b = fb.binary(BinOp::BitwiseAnd, stored, read_mask)?;
    let stencil_passed = stencil::compare(fb, compare_code, a, b)?;
    let on_pass = fb.select(depth_passed, pass_code, depth_fail_code)?;
    let code = fb.select(stencil_passed, on_pass, fail_code)?;
    let written = stencil::operation(fb, code, stored, reference)?;
    let max = u32_value(fb, 0xFF)?;
    let keep_mask = fb.binary(BinOp::BitwiseXor, write_mask, max)?;
    let kept_bits = fb.binary(BinOp::BitwiseAnd, stored, keep_mask)?;
    let new_bits = fb.binary(BinOp::BitwiseAnd, written, write_mask)?;
    let value = fb.binary(BinOp::BitwiseOr, kept_bits, new_bits)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    let zero = u32_value(fb, 0)?;
    let texel = fb.composite_construct(uvec4, &[value, zero, zero, zero])?;
    fb.write(image, pixel, texel, operands, mem)?;
    fb.binary(BinOp::LogicalAnd, stencil_passed, depth_passed)
}

/// The stand-in's texel of the invocation's sample held as `samples`, and the access's image
/// operands: the pixel alone at one sample; with the `Sample` operand of its `SampleId` in a
/// multisampled image; and for planes the pixel and its `SampleId` as the layer.
fn texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    samples: StorageSamples,
) -> BResult<(SValue, ImageOperands)> {
    let pixel = super::fetch::pixel(cx, fb)?;
    if samples == StorageSamples::One {
        return Ok((pixel, ImageOperands::default()));
    }
    let Some(sample) = super::fetch::sample(cx, fb)? else {
        return Err(cx.fail("a per-sample depth test without its SampleId"));
    };
    if samples == StorageSamples::Multisampled {
        let operands = ImageOperands {
            sample: Some(sample),
            ..ImageOperands::default()
        };
        return Ok((pixel, operands));
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = fb.module().ty(SType::Vector { elem: u32_, len: 3 })?;
    let x = fb.composite_extract(pixel, &[0])?;
    let y = fb.composite_extract(pixel, &[1])?;
    let at = fb.composite_construct(uvec3, &[x, y, sample])?;
    Ok((at, ImageOperands::default()))
}

/// The invocation's sample's depth computed from the window-depth input object `window` and the
/// pixel centre's `FragCoord.z` `centre` under the viewport's depth range `near`, `far` (see the
/// module docs): `(v(sample) · (far − near) + near) + (centre − (v(centre) · (far − near) +
/// near))`, each operation rounded on its own.
fn sample_depth(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    window: usize,
    centre: SValue,
    (near, far): (SValue, SValue),
) -> BResult<SValue> {
    let Some(sample) = super::fetch::sample(cx, fb)? else {
        return Err(cx.fail("a computed sample depth without its SampleId"));
    };
    let var = stage_io::object_global(cx, window)?;
    let var = fb.module().global_value(var)?;
    let at_sample = fb.glsl(GlslOp::InterpolateAtSample, &[var, sample])?;
    let at_centre = fb.load(var, MemAccess::default())?;
    let range = fb.binary_no_contraction(BinOp::FSub, far, near)?;
    let window_depth = |fb: &mut FunctionBuilder<'_>, v: SValue| -> BResult<SValue> {
        let scaled = fb.binary_no_contraction(BinOp::FMul, v, range)?;
        fb.binary_no_contraction(BinOp::FAdd, scaled, near)
    };
    let at_sample = window_depth(fb, at_sample)?;
    let at_centre = window_depth(fb, at_centre)?;
    let carried = fb.binary_no_contraction(BinOp::FSub, centre, at_centre)?;
    fb.binary_no_contraction(BinOp::FAdd, at_sample, carried)
}
