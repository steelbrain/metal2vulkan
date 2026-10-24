//! Selection of the stage interface: which Vulkan object each vertex or fragment input and output
//! becomes, checked against the profile.

use crate::air::abi::{
    AbiInt, Builtin, DepthQualifier, Interpolation as AirInterp, OutputBinding, OutputRole,
    ParamBinding, ParamRole, PatchInput, Sampling as AirSampling, Stage as AirStage,
};
use crate::air::ir::{ArgIdx, Type, TypeId as AirTypeId, Value};
use crate::device::profile::DriverWorkaround;
use crate::device::requirements::{VkExtension, VkFeature, VkLimit, VkProperty};
use crate::error::{Attachment, Capability, DriverDeviation, Error, Feature, Unsupported};
use crate::lower::arith::legal::{form_on, refuse_bfloat_color, refuse_int64_interface};
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{
    fetch, not_lowered, AmplifiedOutput, Compensation, DepthClamp, IoObject, MergedRead,
    MergedValue, PlannedOutput, PlannedParam, Returned, RotatedSecond, Source, StageIoPlan,
};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::{
    CaptureForm, FactorLane, RotatedBlendForm, VertexFetch, ViewportCompensation,
};
use crate::pipeline::stage_io::{
    DepthMode, Interpolation, IoNumeric, IoRole, IoType, IoVariable, Sampling, StageBuiltin,
};
use crate::pipeline::sym::{AirUserKey, PushField, Stage, VaryingKey};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::TypeId;
use crate::spirv::ir::{
    ConstBits, Constant, ExecMode, Interpolation as SInterp, IoDecor, IoDir, Sampling as SSampling,
    Type as SType,
};
use crate::spirv::requirements::builtin_enablers;

use crate::spv::BuiltIn as B;

/// The data type of a builtin variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    U32,
    F32,
    Bool,
    /// A `float` vector of 2–4 lanes.
    FVec(u8),
    /// A `uint` array of this length.
    U32Array(u32),
    /// A `float` array of this length.
    F32Array(u32),
}

impl Shape {
    /// The SPIR-V type of the shape.
    pub fn spirv_type(self, mb: &mut ModuleBuilder) -> BResult<TypeId> {
        let u32_ = mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let f32_ = mb.ty(SType::Float { width: 32 })?;
        let array = |mb: &mut ModuleBuilder, elem: TypeId, n: u32| -> BResult<TypeId> {
            let len = mb.constant(Constant::Scalar {
                ty: u32_,
                bits: ConstBits(u64::from(n)),
            })?;
            mb.ty(SType::Array {
                elem,
                len,
                stride: None,
            })
        };
        match self {
            Shape::U32 => Ok(u32_),
            Shape::F32 => Ok(f32_),
            Shape::Bool => mb.ty(SType::Bool),
            Shape::FVec(n) => mb.ty(SType::Vector { elem: f32_, len: n }),
            Shape::U32Array(n) => array(mb, u32_, n),
            Shape::F32Array(n) => array(mb, f32_, n),
        }
    }
}

/// Where in its object an AIR value lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    /// The whole variable.
    Whole,
    /// Element 0 of an array variable (`SampleMask`, a one-element `ClipDistance`).
    Element0,
    /// An aggregate of this many members, one per array element (`ClipDistance`).
    Spread(u32),
    /// A read of this many (1 to 3) leading lanes of Metal's three barycentric weights, of a
    /// variable whose first two lanes are the first two weights of the primitive as drawn
    /// (`BaryCoordKHR`, or the `float2` of `IoRole::Barycentric`): `1 - (x + y)` of the first
    /// two is Metal's third weight (see `input_builtin`). Where `rotated`
    /// (`LoweringInputs::rotated_weights`), the primitive is a triangle drawn with Metal's last
    /// vertex first, `(c, a, b)` for Metal's `(a, b, c)`, so Metal's first two weights are the
    /// variable's second lane and `1 - (x + y)` of its first two.
    Weights { lanes: u8, rotated: bool },
}

/// How an AIR value (a parameter or an output part) reaches its interface object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Link {
    /// The object index in `StageIoPlan::objects`.
    pub object: usize,
    pub place: Place,
    /// The value's form (the element's, for `Place::Spread`).
    pub form: Form,
}

/// Record `PrecisionNote::DevicePointSize` for a function that writes a point size on a device
/// whose `pointSizeRange` ends below Metal's largest point: a size between the two draws at the
/// device's bound.
pub(crate) fn note_device_point_size(cx: &mut SelectCx<'_>) {
    let largest = super::METAL_POINT_SIZE_LIMIT as u32;
    if cx.profile.fields().limits.max_point_size < largest {
        cx.precision
            .insert(crate::pipeline::artifact::PrecisionNote::DevicePointSize);
    }
}

fn needs(f: Feature, v: VkFeature) -> Error {
    Unsupported::needs(f, Capability::Feature(v)).into()
}

fn limit(f: Feature, limit: VkLimit, required: u64, available: u32) -> Error {
    Unsupported::needs(
        f,
        Capability::Limit {
            limit,
            required,
            available: u64::from(available),
        },
    )
    .into()
}

/// The form of an interface value of AIR type `ty`: a scalar or 2–4 lane vector of 32-bit held
/// integers or 32-bit held floats (narrow ones widened). `bool`, 64-bit, and aggregate values are
/// not interface values here.
pub(super) fn io_form(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Form, Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    match f.held {
        Held::Int { container: 32, .. } => Ok(f),
        Held::Float(ff) if ff.container() == 32 => Ok(f),
        _ => Err(not_lowered()),
    }
}

/// The form of a varying (a vertex output or a fragment input) of AIR type `ty`: an
/// [`io_form`], or a `bool` scalar or vector, which the interface carries as a flat `uint` 0 or
/// 1 and each end converts.
fn varying_form(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Form, Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    match f.held {
        Held::Bool => Ok(f),
        _ => io_form(cx, ty),
    }
}

/// The contract type of a user variable of form `f`.
pub(super) fn io_type(f: Form) -> IoType {
    IoType {
        numeric: if f.is_float() {
            IoNumeric::Float
        } else {
            IoNumeric::Uint
        },
        components: f.lanes,
        source_bits: f.bits(),
        bits: 32,
    }
}

/// The contract type of a color output of form `f` to `attachment`: [`io_type`], except that an
/// integer output to a signed-integer attachment (`LoweringInputs::signed_color_attachments`) is
/// declared `int` and stored by `OpBitcast`, the same 32 bits. A narrower integer output to one is
/// not lowered: which extension Metal applies to fill the attachment's lanes is not established.
fn color_type(cx: &SelectCx<'_>, attachment: u32, f: Form) -> Result<IoType, Error> {
    let ty = io_type(f);
    if f.is_float() || !cx.inputs.signed_color_attachments.contains(&attachment) {
        return Ok(ty);
    }
    if ty.source_bits != 32 {
        return Err(not_lowered());
    }
    Ok(IoType {
        numeric: IoNumeric::Sint,
        ..ty
    })
}

fn user(
    sp: &mut StageIoPlan,
    key: VaryingKey,
    dir: IoDir,
    ty: IoType,
    deco: IoDecor,
) -> Result<usize, Error> {
    let o = IoObject::User {
        key: key.clone(),
        dir,
        numeric: ty.numeric,
        components: ty.components,
        deco,
        half: ty.bits == 16,
    };
    // One key per direction: a second, different declaration of the same key is not a Metal
    // interface.
    let clash = sp.objects.iter().any(|x| {
        matches!(x, IoObject::User { key: k, dir: d, .. } if *k == key && *d == dir) && *x != o
    });
    if clash {
        return Err(not_lowered());
    }
    Ok(sp.object(o))
}

pub(super) fn builtin(
    sp: &mut StageIoPlan,
    b: crate::spv::BuiltIn,
    dir: IoDir,
    shape: Shape,
    contract: StageBuiltin,
) -> usize {
    builtin_through(sp, b, dir, shape, contract, None)
}

/// [`builtin`]'s object without its contract builtin on the stage interface: a family's own read
/// of a builtin whose contract would say something else (the shader-side stencil's coverage).
pub(super) fn builtin_object(
    sp: &mut StageIoPlan,
    b: crate::spv::BuiltIn,
    dir: IoDir,
    shape: Shape,
) -> usize {
    sp.object(IoObject::Builtin {
        builtin: b,
        dir,
        shape,
        enabler: None,
    })
}

/// [`builtin`], declared through enabling capability `enabler` when the builtin has
/// alternatives ([`builtin_enablers`]).
fn builtin_through(
    sp: &mut StageIoPlan,
    b: crate::spv::BuiltIn,
    dir: IoDir,
    shape: Shape,
    contract: StageBuiltin,
    enabler: Option<crate::spv::Capability>,
) -> usize {
    if !sp.interface.builtins.contains(&contract) {
        sp.interface.builtins.push(contract);
        sp.interface.builtins.sort();
    }
    sp.object(IoObject::Builtin {
        builtin: b,
        dir,
        shape,
        enabler,
    })
}

/// A scalar `u32` builtin read into an integer of at most 32 bits.
fn int_scalar(f: Form) -> Result<(), Error> {
    match (f.held, f.lanes) {
        (Held::Int { container: 32, .. }, 1) => Ok(()),
        _ => Err(not_lowered()),
    }
}

/// The [`io_form`] of AIR type `ty`, which must be [`int_scalar`].
pub(super) fn int_scalar_form(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Form, Error> {
    let f = io_form(cx, ty)?;
    int_scalar(f)?;
    Ok(f)
}

/// A `float`-lane builtin read into or written from a float of `lanes` lanes.
fn float_lanes(f: Form, lanes: u8) -> Result<(), Error> {
    match f.held {
        Held::Float(ff) if ff.container() == 32 && f.lanes == lanes => Ok(()),
        _ => Err(not_lowered()),
    }
}

pub(super) fn param_ty(cx: &SelectCx<'_>, b: &ParamBinding) -> Result<AirTypeId, Error> {
    cx.value_ty(Value::Arg(ArgIdx::new(b.param)))
        .ok_or_else(|| cx.invariant("a stage-IO parameter without a type"))
}

/// The decorations of a fragment input interpolated as `i`, and its contract form. Integers are
/// never interpolated (Vulkan requires `Flat` on them, and Metal integer inputs are flat). In a
/// function looping over its covered samples a sample-qualified input is not decorated `Sample`,
/// which would turn sample shading on: each pass interpolates it at its sample.
fn fragment_input_deco(
    cx: &SelectCx<'_>,
    i: AirInterp,
    f: Form,
) -> Result<(IoDecor, Interpolation), Error> {
    // Link may pack varyings into shared locations: the module relocates `Component`.
    let mut deco = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    let i = if f.is_float() { i } else { AirInterp::Flat };
    let contract = match i {
        AirInterp::Flat => {
            deco.interpolation = SInterp::Flat;
            Interpolation::Flat
        }
        AirInterp::Smooth {
            sampling,
            perspective,
        } => {
            if !perspective {
                deco.interpolation = SInterp::NoPerspective;
            }
            let s = match sampling {
                AirSampling::Center => Sampling::Center,
                AirSampling::Centroid => {
                    deco.sampling = SSampling::Centroid;
                    Sampling::Centroid
                }
                // `InterpolateAtSample` needs what the loop's plan checked.
                AirSampling::Sample if crate::lower::fragment::loops_samples(cx) => {
                    Sampling::Sample
                }
                AirSampling::Sample => {
                    if !cx.profile.fields().fragment.sample_rate_shading {
                        return Err(needs(
                            Feature::SampleRateShading,
                            VkFeature::SampleRateShading,
                        ));
                    }
                    deco.sampling = SSampling::Sample;
                    Sampling::Sample
                }
            };
            Interpolation::Smooth {
                sampling: s,
                perspective,
            }
        }
    };
    Ok((deco, contract))
}

/// Whether the profile can read an input at the pixel center through `GLSL.std.450
/// InterpolateAtOffset` at offset 0: the `InterpolationFunction` capability needs
/// `sampleRateShading`, and a portability-subset device `shaderSampleRateInterpolationFunctions`
/// besides (`VUID-RuntimeSpirv-shaderSampleRateInterpolationFunctions-06325`), and the device must
/// read the offset from the center as GLSL.std.450 states
/// (`VkProperty::InterpolateAtOffsetCentered`; MoltenVK does not).
fn interpolation_functions(cx: &SelectCx<'_>) -> bool {
    let p = cx.profile.fields();
    p.fragment.sample_rate_shading
        && !p
            .enabled
            .portability_withholds(VkFeature::ShaderSampleRateInterpolationFunctions)
        && p.enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
}

/// Whether the device reads a center-sampled varying of a multisampled pipeline at the pixel
/// center in every fragment invocation, a sample-rate one's included, as Metal does
/// (`VkProperty::MetalCenterVaryings`, the measured `DriverDeviation::MetalCenterVaryings`): the
/// varying's read as it is is then its pixel form.
fn device_reads_center_varyings_at_pixel(cx: &SelectCx<'_>) -> bool {
    cx.profile
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalCenterVaryings)
}

/// Whether the entry may read lane 2 or 3 (z or w) of vector parameter `param`: anything but an
/// extract of a constant lane 0 or 1, or a shuffle that takes only those lanes of it.
fn reads_depth_lanes(cx: &SelectCx<'_>, param: u32) -> bool {
    use crate::air::ir::{Const, InstKind};
    let arg = Value::Arg(ArgIdx::new(param));
    let xy = |idx: Value| match idx {
        Value::Const(c) => cx
            .entry
            .module
            .consts()
            .get(c)
            .is_some_and(|c| matches!(c.value, Const::Int(i) if i < 2)),
        _ => false,
    };
    cx.body.insts.values().any(|inst| match &inst.kind {
        InstKind::ExtractElement { vec, idx } if *vec == arg => !xy(*idx),
        InstKind::ShuffleVector { a, b, mask } if *a == arg || *b == arg => {
            mask.iter().flatten().any(|&lane| {
                let (from_a, k) = if lane < 4 {
                    (true, lane)
                } else {
                    (false, lane - 4)
                };
                let of_arg = if from_a { *a == arg } else { *b == arg };
                of_arg && k >= 2
            })
        }
        kind => {
            let mut used = false;
            kind.for_each_value(|v| used |= v == arg);
            used
        }
    })
}

/// Plan a used stage-IO parameter of a vertex or fragment entry.
///
/// Under `LoweringInputs::multisampled` a fragment function reads what Metal reads at the pixel
/// center in every invocation there, whatever samples the invocation shades: `[[position]]`'s x and
/// y, and a center-sampled float varying where the profile has interpolation functions whose
/// `InterpolateAtOffset` reads from the center ([`Source::PixelCenter`]). That holds for a
/// pixel-rate function too: Vulkan evaluates its undecorated input at an implementation-defined
/// position in the pixel. A device that reads such a varying at the pixel center itself, as Metal
/// does (`VkProperty::MetalCenterVaryings`), reads it as it is, and that read is its pixel form.
pub fn param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let ty = param_ty(cx, b)?;
    let frag = cx.profile.fields().fragment;
    let stage = cx.stage;
    let mut at_pixel = false;
    let mut at_loop_sample = false;
    let mut uniform = false;
    let mut quad_primitive = false;
    let mut widened = false;
    let link = match (&b.role, stage) {
        // A merged draw's vertex reads its corner's command's draw values.
        (
            ParamRole::Builtin(
                bi @ (Builtin::VertexId
                | Builtin::InstanceId
                | Builtin::BaseVertex
                | Builtin::BaseInstance),
            ),
            Stage::Vertex,
        ) if cx.inputs.merged_draws => {
            let value = match bi {
                Builtin::VertexId => MergedValue::Vertex,
                Builtin::InstanceId => MergedValue::Instance,
                Builtin::BaseVertex => MergedValue::BaseVertex,
                _ => MergedValue::BaseInstance,
            };
            return merged_param(cx, b, ty, value);
        }
        // A merged draw counts its primitives across its commands, where Metal counts each
        // command's.
        (ParamRole::Builtin(Builtin::PrimitiveId), Stage::Fragment) if cx.inputs.merged_draws => {
            return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
        }
        (ParamRole::VertexInput { location }, Stage::Vertex) => {
            let AbiInt::Const(index) = location.index else {
                return Err(not_lowered());
            };
            let entry = cx
                .inputs
                .vertex_fetch
                .as_ref()
                .and_then(|p| p.attributes.get(&index))
                .copied();
            let pulled = entry.filter(|f| matches!(f, VertexFetch::Pulled { .. }));
            let f = io_form(cx, ty)?;
            // The indexed capture form's and the de-indexed form's draws have no indices, so
            // fixed-function input would fetch at the stream position: their projection pulls
            // every attribute, and a hand-built one that does not is refused.
            if pulled.is_none() && captured_index_type(cx).is_some() {
                return Err(Unsupported::not_implemented(indices_feature(cx)).into());
            }
            if let Some(fetch) = pulled {
                // Fetched in the shader from its source buffer, no interface variable.
                let read = fetch::select(cx, index, fetch, f)?;
                cx.families.stage_io.params.push(PlannedParam {
                    param: b.param,
                    source: Source::Pulled(read),
                });
                return Ok(());
            }
            // The device's fixed-function input widens a narrow-float format with a widening of
            // its own, which carries the field's bits up into the `float` payload; Metal's fetch
            // answers its canonical NaN. A `float` parameter's load is canonicalized
            // (`Source::WidenedAttribute`); a `half` one keeps the fetched code's own bits, as
            // Metal's `half` attribute does, so it is loaded as it is.
            let narrow_float = matches!(
                entry,
                Some(VertexFetch::FixedFunction { narrow_float: true })
            );
            widened = narrow_float && f.float_fmt() == Some(FloatFmt::F32);
            let key = VaryingKey::attribute(index);
            let sp = &mut cx.families.stage_io;
            let object = user(
                sp,
                key.clone(),
                IoDir::Input,
                io_type(f),
                IoDecor::default(),
            )?;
            push_var(
                &mut sp.interface.inputs,
                IoVariable {
                    key,
                    role: IoRole::Attribute {
                        index,
                        narrow_float,
                    },
                    ty: io_type(f),
                },
            );
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        (ParamRole::FragmentInput(v), Stage::Fragment) => {
            let key = VaryingKey::user(&v.key.0);
            // An input the pipeline's linking drops that no vertex output writes: zero, and no
            // interface variable, which Vulkan would require the vertex stage to write.
            if cx.inputs.absent_varyings.contains(&key) {
                let f = form_on(cx.entry, cx.profile, ty)?;
                cx.families.stage_io.params.push(PlannedParam {
                    param: b.param,
                    source: Source::Absent(f),
                });
                return Ok(());
            }
            let f = varying_form(cx, ty)?;
            // A varying its mesh function writes per primitive holds the primitive's value: read
            // flat, whatever interpolation the function states.
            let interpolation = if read_per_primitive(cx, &key) {
                AirInterp::Flat
            } else if read_uniform(cx, &key) {
                // One its vertex function writes the same at every vertex of a primitive: read
                // flat, as Metal's interpolation returns such a value (`Source::Uniform`).
                uniform = true;
                AirInterp::Flat
            } else {
                v.interpolation
            };
            let (deco, interp) = fragment_input_deco(cx, interpolation, f)?;
            note_device_interpolation(cx, interp);
            at_loop_sample = crate::lower::fragment::loops_samples(cx)
                && matches!(
                    interp,
                    Interpolation::Smooth {
                        sampling: Sampling::Sample,
                        ..
                    }
                );
            let center = cx.inputs.multisampled
                && matches!(
                    interp,
                    Interpolation::Smooth {
                        sampling: Sampling::Center,
                        ..
                    }
                );
            // Where the device reads a center varying at the pixel center in every invocation,
            // as Metal does, the read as it is is the pixel form; elsewhere it is
            // `InterpolateAtOffset` at offset 0 where that reads from the center.
            let by_device = center && device_reads_center_varyings_at_pixel(cx);
            at_pixel = center && !by_device && interpolation_functions(cx);
            let sp = &mut cx.families.stage_io;
            sp.interface.varyings_at_pixel |= at_pixel || by_device;
            sp.center_varyings_by_device |= by_device;
            let object = user(sp, key.clone(), IoDir::Input, io_type(f), deco)?;
            push_var(
                &mut sp.interface.inputs,
                IoVariable {
                    key,
                    role: IoRole::Varying {
                        user: AirUserKey(v.key.0.clone()),
                        interpolation: Some(interp),
                    },
                    ty: io_type(f),
                },
            );
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        // Drawn one amplification at a time, a stage reads its amplification from the push word the
        // binder writes before each draw, not `ViewIndex`. A mesh function always does: a mesh
        // pipeline draws one amplification at a time, without multiview.
        (ParamRole::Builtin(Builtin::AmplificationId), _)
            if cx.inputs.amplification_draws.is_some() || stage == Stage::Mesh =>
        {
            return pushed(cx, b, ty, PushField::AmplificationId);
        }
        // A vertex function that writes its render target slice is never drawn amplified as
        // views: the binder draws such a pipeline amplified only one amplification at a time, so
        // outside that variant its amplification id is 0 in every draw (Metal's without
        // amplification). It does not read `ViewIndex`, for which SPIRV-Cross (MoltenVK) gives a
        // vertex function a second `[[render_target_array_index]]` output that Metal refuses beside
        // the function's own.
        (ParamRole::Builtin(Builtin::AmplificationId), Stage::Vertex) if writes_layer(cx) => {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            cx.families.stage_io.params.push(PlannedParam {
                param: b.param,
                source: Source::Unamplified(f),
            });
            return Ok(());
        }
        // The sample of the loop's pass over the covered samples: no `SampleId`, which would turn
        // sample shading on.
        (ParamRole::Builtin(Builtin::SampleId), Stage::Fragment)
            if crate::lower::fragment::loops_samples(cx) =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            cx.families.stage_io.params.push(PlannedParam {
                param: b.param,
                source: Source::LoopSample(f),
            });
            return Ok(());
        }
        // Vulkan has no builtin for the count: the binder specializes it.
        (ParamRole::Builtin(Builtin::AmplificationCount), _) => {
            let f = amplification_count_form(cx, ty)?;
            cx.families.stage_io.params.push(PlannedParam {
                param: b.param,
                source: Source::AmplificationCount(f),
            });
            return Ok(());
        }
        // Compiled for a vertex function that writes the index, a fragment function reads it as
        // written from the raw varying: that function's `ViewportIndex` holds 0 for an index at
        // or past the draw's viewport count, and Metal's fragment function reads the index the
        // vertex function wrote.
        (ParamRole::Builtin(Builtin::ViewportArrayIndex), Stage::Fragment)
            if cx.inputs.viewport_index_varying =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            read_per_primitive(cx, &VaryingKey::viewport_index());
            let object = raw_index(&mut cx.families.stage_io, RawIndex::Viewport, IoDir::Input)?;
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        // Compiled for a vertex function that writes the index, a fragment function reads it as
        // written from the raw varying: that function's `Layer` holds 0 for an index at or past
        // the pass's length, and its variant for a pass of one layer writes no `Layer`, while
        // Metal's fragment function reads the index the vertex function wrote.
        (ParamRole::Builtin(Builtin::RenderTargetArrayIndex), Stage::Fragment)
            if cx.inputs.render_target_array_index_varying =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            read_per_primitive(cx, &VaryingKey::render_target_array_index());
            let object = raw_index(&mut cx.families.stage_io, RawIndex::Layer, IoDir::Input)?;
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        // Metal's `[[primitive_id]]` is the flat varying the pipeline's vertex function writes
        // (`IoRole::PrimitiveId`) on a device that cannot declare a fragment `PrimitiveId`
        // (neither `geometryShader` nor `tessellationShader`), where the vertex function computes
        // the primitive each vertex provokes, and under `LoweringInputs::primitive_id_varying`
        // behind an emulated post-tessellation vertex function, which writes its patch's position:
        // Metal's primitive id behind the tessellator, while Vulkan's `PrimitiveId` without a
        // tessellation stage counts the draw's triangles.
        (ParamRole::Builtin(Builtin::PrimitiveId), Stage::Fragment)
            if !declarable(cx, B::PrimitiveId) || cx.inputs.primitive_id_varying =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            let object = primitive_id(&mut cx.families.stage_io, IoDir::Input)?;
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        // Drawn one amplification at a time beside a vertex function that writes no such index, a
        // fragment function reads the amplification's offset as the index, as Metal adds it to the
        // index the vertex function writes, 0 (a Metal probe on the M3 Max and an M4).
        (ParamRole::Builtin(Builtin::RenderTargetArrayIndex), Stage::Fragment)
            if cx.inputs.amplification_draws.is_some() =>
        {
            return pushed(cx, b, ty, PushField::AmplificationLayerOffset);
        }
        (ParamRole::Builtin(Builtin::ViewportArrayIndex), Stage::Fragment)
            if cx.inputs.amplification_draws.is_some() =>
        {
            return pushed(cx, b, ty, PushField::AmplificationViewportOffset);
        }
        // Beside a vertex function that writes no such index, on a device that cannot declare the
        // builtin, a fragment function reads 0, which the builtin would hold and Metal's function
        // reads there (Metal probes on the M3 Max, macOS 27.0.1: in a pass of four layers with
        // four viewports, and of no layers with one).
        (ParamRole::Builtin(Builtin::RenderTargetArrayIndex), Stage::Fragment)
            if !declarable(cx, B::Layer) =>
        {
            return zero_index(cx, b, ty, StageBuiltin::Layer);
        }
        (ParamRole::Builtin(Builtin::ViewportArrayIndex), Stage::Fragment)
            if !declarable(cx, B::ViewportIndex) =>
        {
            return zero_index(cx, b, ty, StageBuiltin::ViewportIndex);
        }
        // The indexed capture form draws without indices: its vertex is the one its index names,
        // and its base vertex the layout's, whatever the profile's draw parameters.
        (ParamRole::Builtin(Builtin::VertexId), Stage::Vertex)
            if captured_index_type(cx).is_some() =>
        {
            return captured_param(cx, b, ty, captured_index_type(cx));
        }
        (ParamRole::Builtin(Builtin::BaseVertex), Stage::Vertex)
            if captured_index_type(cx).is_some() =>
        {
            return captured_param(cx, b, ty, None);
        }
        // The draw's base vertex and base instance without `shaderDrawParameters`: the lanes of
        // `PushField::BaseVertexInstance`, which the binder writes for each draw.
        (ParamRole::Builtin(Builtin::BaseVertex), Stage::Vertex)
            if !cx.profile.fields().draw_parameters =>
        {
            return draw_base(cx, b, ty, DrawBase::Vertex);
        }
        (ParamRole::Builtin(Builtin::BaseInstance), Stage::Vertex)
            if !cx.profile.fields().draw_parameters =>
        {
            return draw_base(cx, b, ty, DrawBase::Instance);
        }
        // Without `fragmentShaderBarycentric`, the weights are the `float2` varying of their form
        // that the pipeline's synthesized geometry stage writes (`IoRole::Barycentric`,
        // `link::geometry`): `(1, 0)`, `(0, 1)` and `(0, 0)` at the primitive's vertices in its
        // order, whose interpolation is Metal's first two weights, read at the pixel center as a
        // center varying is.
        (ParamRole::Builtin(Builtin::BarycentricCoord(i)), Stage::Fragment)
            if !frag.barycentric =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            let (perspective, place) = barycentric_read(*i, f, cx.inputs.rotated_weights)?;
            if !frag.geometry_shader {
                return Err(Unsupported::needs_any_of(
                    Feature::Barycentrics,
                    vec![
                        vec![Capability::Feature(VkFeature::FragmentShaderBarycentric)],
                        vec![Capability::Feature(VkFeature::GeometryShader)],
                    ],
                )
                .into());
            }
            let interp = Interpolation::Smooth {
                sampling: Sampling::Center,
                perspective,
            };
            note_device_interpolation(cx, interp);
            let center = cx.inputs.multisampled;
            let by_device = center && device_reads_center_varyings_at_pixel(cx);
            at_pixel = center && !by_device && interpolation_functions(cx);
            let sp = &mut cx.families.stage_io;
            sp.interface.varyings_at_pixel |= at_pixel || by_device;
            sp.center_varyings_by_device |= by_device;
            let object = barycentric_weights(sp, perspective)?;
            note_rotated_weights(cx, place);
            Link {
                object,
                place,
                form: f,
            }
        }
        (ParamRole::Builtin(bi), _) => {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            let (spirv, shape, place, contract, enabler) = input_builtin(cx, *bi, f, frag)?;
            // `[[sample_mask, post_depth_coverage]]`: `SampleMask` read after the early depth and
            // stencil tests, which the entry declares (the AIR decode requires them), through
            // `VK_EXT_post_depth_coverage`.
            if *bi == Builtin::PostDepthCoverage {
                let ext = VkExtension::ExtPostDepthCoverage;
                if !cx.profile.fields().enabled.extensions.contains(&ext) {
                    return Err(Unsupported::needs(
                        Feature::PostDepthCoverage,
                        Capability::Extension(ext),
                    )
                    .into());
                }
                let modes = &mut cx.families.stage_io.modes;
                if !modes.contains(&ExecMode::PostDepthCoverage) {
                    modes.push(ExecMode::PostDepthCoverage);
                }
            }
            if matches!(
                contract,
                StageBuiltin::BaseVertex | StageBuiltin::BaseInstance
            ) {
                cx.facts.insert(RuntimeFact::BaseVertexInstance);
            }
            // A draw of Metal's private quads lists two triangles a quad: the fragment function
            // reads the quad's, Vulkan's `PrimitiveId` shifted right by one
            // (`LoweringInputs::quad_primitive_ids`).
            if contract == StageBuiltin::PrimitiveId && cx.inputs.quad_primitive_ids {
                quad_primitive = true;
                cx.facts.insert(RuntimeFact::QuadPrimitiveIds);
            }
            note_rotated_weights(cx, place);
            if contract == StageBuiltin::FragCoord {
                at_pixel = cx.inputs.multisampled;
                let depth = reads_depth_lanes(cx, b.param);
                cx.families.stage_io.interface.position_depth |= depth;
            }
            let object = builtin_through(
                &mut cx.families.stage_io,
                spirv,
                IoDir::Input,
                shape,
                contract,
                enabler,
            );
            Link {
                object,
                place,
                form: f,
            }
        }
        (ParamRole::PatchInput(PatchInput::ControlPoints { .. } | PatchInput::Patch { .. }), _) => {
            return Err(Unsupported::not_implemented(Feature::Tessellation).into())
        }
        _ => return Err(cx.invariant("a stage-IO parameter of another stage")),
    };
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: if widened {
            Source::WidenedAttribute(link)
        } else if at_loop_sample {
            Source::AtLoopSample(link)
        } else if at_pixel {
            Source::PixelCenter(link)
        } else if uniform {
            Source::Uniform(link)
        } else if quad_primitive {
            Source::QuadPrimitive(link)
        } else {
            Source::Io(link)
        },
    });
    Ok(())
}

/// An index a vertex function compiled under `LoweringInputs::amplification_draws` writes
/// though it writes no such index of its own.
#[derive(Clone, Copy)]
enum AmplifiedIndex {
    /// `Layer`, the amplification's render target slice offset.
    Layer,
    /// `ViewportIndex`, the amplification's viewport offset.
    Viewport,
}

/// Plan the zero a fragment function compiled under `LoweringInputs::zeroed_colors` writes to
/// color attachment `color` of pixel format `format`, which its Metal pipeline lacks: an output
/// variable at that color of the format's numeric class, four lanes, stored zero at every return
/// (`StageIoPlan::zeroed`). Its interface lists it as the color's output, so link gives the
/// attachment a write of every channel. Zero is every format's zero code, so no lane rounds; the
/// map of the attachment's color writes is recorded as a written attachment's is. A function
/// that writes or reads `color` itself is not lowered (what Metal does there is unmeasured).
fn zeroed_output(
    cx: &mut SelectCx<'_>,
    outputs: &[OutputBinding],
    color: u32,
    format: crate::metal::PixelFormat,
) -> Result<(), Error> {
    let writes = outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                ..
            } if a == color
        )
    });
    let reads = cx.entry.ep.params.iter().any(|b| {
        matches!(
            b.role,
            ParamRole::ColorInput {
                index: AbiInt::Const(a),
                ..
            } if a == color
        )
    });
    if writes || reads {
        return Err(not_lowered());
    }
    // What Metal stores into an extended-range attachment the pipeline lacks is unmeasured (its
    // tile holds decoded values, not the raw texel's bytes).
    if crate::metal::format::ExtendedRange::of(format).is_some() {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    cx.facts.insert(RuntimeFact::ZeroedColor { color, format });
    let limits = &cx.profile.fields().limits;
    if color >= limits.max_fragment_output_attachments {
        return Err(limit(
            Feature::InterfaceComponents,
            VkLimit::MaxFragmentOutputAttachments,
            u64::from(color) + 1,
            limits.max_fragment_output_attachments,
        ));
    }
    use crate::pipeline::sym::SampledType;
    let (numeric, held) = match crate::metal::format::sampled_type(format) {
        Some(SampledType::Float) => (
            IoNumeric::Float,
            Held::Float(crate::lower::arith::FloatFmt::F32),
        ),
        Some(SampledType::Uint) => (
            IoNumeric::Uint,
            Held::Int {
                bits: 32,
                container: 32,
            },
        ),
        Some(SampledType::Sint) => (
            IoNumeric::Sint,
            Held::Int {
                bits: 32,
                container: 32,
            },
        ),
        // A 64-bit color format (`RG32Uint` read as one `ulong`) is no attachment's.
        Some(SampledType::Uint64) | None => return Err(not_lowered()),
    };
    let form = Form { held, lanes: 4 };
    let io = IoType {
        numeric,
        components: 4,
        source_bits: 32,
        bits: 32,
    };
    if let Some(rules) = cx.inputs.color_writes.get(&color).copied() {
        cx.families
            .stage_io
            .interface
            .color_writes
            .insert(color, rules);
    }
    let key = VaryingKey::color(color, 0);
    let sp = &mut cx.families.stage_io;
    let object = user(sp, key.clone(), IoDir::Output, io, IoDecor::default())?;
    push_var(
        &mut sp.interface.outputs,
        IoVariable {
            key,
            role: IoRole::Color {
                attachment: color,
                index: 0,
            },
            ty: io,
        },
    );
    sp.zeroed.push(Link {
        object,
        place: Place::Whole,
        form,
    });
    cx.descriptors.color_attachments = cx.descriptors.color_attachments.saturating_add(1);
    Ok(())
}

/// Plan output `index` of a vertex function compiled under `LoweringInputs::amplification_draws`
/// that writes no such index: stored as the amplification's offset below its count push field
/// ([`AmplifiedOutput`]) where [`index_store`] stores it, to its builtin or its selected index.
/// Metal renders the amplification through the slice or viewport its offset names, the first at or
/// past the count (a Metal probe on the M3 Max and an M4).
fn amplified_output(cx: &mut SelectCx<'_>, index: AmplifiedIndex) -> Result<(), Error> {
    let (raw, spirv, contract, offset, below) = match index {
        AmplifiedIndex::Layer => (
            RawIndex::Layer,
            B::Layer,
            StageBuiltin::Layer,
            PushField::AmplificationLayerOffset,
            PushField::RenderTargetArrayLength,
        ),
        AmplifiedIndex::Viewport => (
            RawIndex::Viewport,
            B::ViewportIndex,
            StageBuiltin::ViewportIndex,
            PushField::AmplificationViewportOffset,
            PushField::ViewportCount,
        ),
    };
    let store = index_store(cx, raw, true, true)?;
    if store == IndexStore::Nowhere {
        return Ok(());
    }
    if matches!(index, AmplifiedIndex::Viewport) {
        cx.facts.insert(RuntimeFact::ViewportCount);
    }
    cx.push.insert(offset);
    cx.push.insert(below);
    let sp = &mut cx.families.stage_io;
    let object = match store {
        IndexStore::Selected => selected_index(sp, raw)?,
        _ => builtin(sp, spirv, IoDir::Output, Shape::U32, contract),
    };
    sp.amplified.push(AmplifiedOutput {
        link: Link {
            object,
            place: Place::Whole,
            form: Form {
                held: Held::Int {
                    bits: 32,
                    container: 32,
                },
                lanes: 1,
            },
        },
        offset,
        below,
    });
    Ok(())
}

/// Plan builtin parameter `b` of AIR type `ty`, a scalar integer, as `u32` push field `field`
/// ([`Source::Pushed`]).
fn pushed(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    ty: AirTypeId,
    field: PushField,
) -> Result<(), Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(f)?;
    cx.push.insert(field);
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::Pushed { field, form: f },
    });
    Ok(())
}

/// Which of the draw's bases a `[[base_vertex]]` or `[[base_instance]]` parameter reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawBase {
    /// The base vertex: lane 0 of `PushField::BaseVertexInstance`.
    Vertex,
    /// The base instance: lane 1.
    Instance,
}

impl DrawBase {
    /// The lane of `PushField::BaseVertexInstance` that holds it.
    pub fn lane(self) -> u32 {
        match self {
            DrawBase::Vertex => 0,
            DrawBase::Instance => 1,
        }
    }
}

/// Plan parameter `b` of AIR type `ty` as the draw's `base`, read from its lane of
/// `PushField::BaseVertexInstance` (the binder writes the draw's Vulkan base vertex, its first
/// vertex or vertex offset, and its first instance, and reads an indirect draw's arguments on the
/// host to do so), into an integer of the parameter's form.
pub(crate) fn draw_base(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    ty: AirTypeId,
    base: DrawBase,
) -> Result<(), Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(f)?;
    cx.push.insert(PushField::BaseVertexInstance);
    cx.facts.insert(RuntimeFact::BaseVertexInstance);
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::DrawBase { base, form: f },
    });
    Ok(())
}

/// Whether the entry writes `[[render_target_array_index]]` (a vertex function choosing its
/// render target slice, which makes its pipelines declare `Layer`).
fn writes_layer(cx: &SelectCx<'_>) -> bool {
    cx.entry
        .ep
        .outputs
        .iter()
        .any(|o| matches!(o.role, OutputRole::RenderTargetArrayIndex))
}

/// Whether a fragment function's first output of form `form` to the rotated attachment
/// `attachment`, whose blend takes the form `blend` (`LoweringInputs::rotated_blends`), writes
/// Metal's blue premultiplied by Metal's alpha (`RotatedBlendForm::Premultiplied`), recording
/// the form in the function's interface. Its blend reads Metal's source alpha from a second
/// output (`LoweringInputs::rotated_factors`'s lane 3), else the inputs are no pipeline's. Apple's
/// blend multiplies an `f32` source exactly as the function does; another float format rounds the
/// product apart, and a blend in none of the measured forms (`RotatedBlendForm::Unmeasured`) may
/// round apart: each records `PrecisionNote::RotatedBlendRounding`.
fn rotated_blend_form(
    cx: &mut SelectCx<'_>,
    attachment: u32,
    blend: RotatedBlendForm,
    form: Form,
) -> Result<bool, Error> {
    let premultiplied = match blend {
        RotatedBlendForm::Premultiplied => {
            let lanes = cx.inputs.rotated_factors.get(&attachment);
            if lanes.is_none_or(|l| l[3] != FactorLane::Output(3)) {
                return Err(not_lowered());
            }
            if form.float_fmt() != Some(FloatFmt::F32) {
                cx.precision.insert(PrecisionNote::RotatedBlendRounding);
            }
            true
        }
        RotatedBlendForm::Unmeasured => {
            cx.precision.insert(PrecisionNote::RotatedBlendRounding);
            false
        }
    };
    let sp = &mut cx.families.stage_io;
    sp.interface.rotated_blends.insert(attachment, blend);
    Ok(premultiplied)
}

/// The second output a fragment function writes beside its first output, the returned field
/// `field` of form `form`, to the rotated attachment `attachment`
/// (`LoweringInputs::rotated_factors`): an output variable of dual-source index 1 there, of the
/// first output's type, which needs `dualSrcBlend` and an attachment below
/// `maxFragmentDualSrcAttachments`.
fn rotated_second(
    cx: &mut SelectCx<'_>,
    attachment: u32,
    (field, form): (u32, Form),
    lanes: [FactorLane; 4],
) -> Result<RotatedSecond, Error> {
    let frag = cx.profile.fields().fragment;
    let limits = cx.profile.fields().limits;
    if !frag.dual_src_blend {
        return Err(needs(Feature::DualSourceBlending, VkFeature::DualSrcBlend));
    }
    if attachment >= limits.max_fragment_dual_src_attachments {
        return Err(limit(
            Feature::DualSourceBlending,
            VkLimit::MaxFragmentDualSrcAttachments,
            u64::from(attachment) + 1,
            limits.max_fragment_dual_src_attachments,
        ));
    }
    // A lane of a channel the first output lacks is no input `rotated_factors` derives.
    if lanes
        .iter()
        .any(|l| matches!(l, FactorLane::Output(m) if *m >= form.lanes))
    {
        return Err(not_lowered());
    }
    let io = color_type(cx, attachment, form)?;
    let key = VaryingKey::color(attachment, 1);
    let deco = IoDecor {
        index: Some(1),
        ..IoDecor::default()
    };
    let sp = &mut cx.families.stage_io;
    let object = user(sp, key.clone(), IoDir::Output, io, deco)?;
    push_var(
        &mut sp.interface.outputs,
        IoVariable {
            key,
            role: IoRole::Color {
                attachment,
                index: 1,
            },
            ty: io,
        },
    );
    sp.interface.rotated_factors.insert(attachment, lanes);
    cx.descriptors.color_attachments = cx.descriptors.color_attachments.saturating_add(1);
    Ok(RotatedSecond {
        field,
        link: Link {
            object,
            place: Place::Whole,
            form,
        },
        lanes,
    })
}

fn push_var(list: &mut Vec<IoVariable>, v: IoVariable) {
    if !list.contains(&v) {
        list.push(v);
    }
}

/// Whether the fragment function reads the varying `key` per primitive: its mesh render
/// pipeline's mesh function writes it per primitive (`LoweringInputs::primitive_varyings`). The
/// stage interface lists it (`StageIo::primitive_inputs`), and the declaration decorates it
/// `PerPrimitiveEXT`, or, on a device that emulates mesh functions, reads it flat alone
/// (`StageIoPlan::flat_primitive_inputs`).
fn read_per_primitive(cx: &mut SelectCx<'_>, key: &VaryingKey) -> bool {
    if !cx.inputs.primitive_varyings.contains(key) {
        return false;
    }
    cx.families.stage_io.flat_primitive_inputs = crate::lower::mesh::emulates(cx.profile);
    let keys = &mut cx.families.stage_io.interface.primitive_inputs;
    if let Err(at) = keys.binary_search(key) {
        keys.insert(at, key.clone());
    }
    true
}

/// Whether the fragment function reads the varying `key` flat because its pipeline's vertex
/// function writes it the same at every vertex of a primitive
/// (`LoweringInputs::uniform_varyings`), listing it among `StageIo::uniform_inputs` if so.
fn read_uniform(cx: &mut SelectCx<'_>, key: &VaryingKey) -> bool {
    if !cx.inputs.uniform_varyings.contains(key) {
        return false;
    }
    let keys = &mut cx.families.stage_io.interface.uniform_inputs;
    if let Err(at) = keys.binary_search(key) {
        keys.insert(at, key.clone());
    }
    true
}

/// An index a vertex function writes as written to a raw varying for its fragment function.
#[derive(Clone, Copy)]
enum RawIndex {
    /// `[[viewport_array_index]]` (`IoRole::ViewportIndex`).
    Viewport,
    /// `[[render_target_array_index]]` (`IoRole::RenderTargetArrayIndex`).
    Layer,
}

/// Where a vertex function stores the index it selects its primitive's render target slice or
/// viewport with ([`index_store`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexStore {
    /// To its builtin, `Layer` or `ViewportIndex`.
    Builtin,
    /// To its selected index (`IoRole::SelectedLayer`, `IoRole::SelectedViewport`), which the
    /// pipeline's synthesized geometry stage stores to the builtin from the primitive's first
    /// vertex (`pipeline::link::geometry`).
    Selected,
    /// Nowhere: a viewport index on a device without `multiViewport`, whose draws have one
    /// viewport (the runtime refuses more), so Metal's index selects viewport 0 whatever it is,
    /// the viewport Vulkan uses where no stage writes `ViewportIndex`.
    Nowhere,
}

/// Where a vertex function on `cx`'s profile stores index `index`, which it writes the same at
/// every vertex of a primitive where `uniform` ([`uniform_index`]): to its selected index where
/// the function stores another there ([`StageIoPlan::indices_selected`]), and otherwise where
/// [`own_index_store`] puts it.
fn index_store(
    cx: &SelectCx<'_>,
    index: RawIndex,
    uniform: bool,
    amplification: bool,
) -> Result<IndexStore, Error> {
    match own_index_store(cx, index, uniform, amplification)? {
        IndexStore::Builtin if cx.families.stage_io.indices_selected => Ok(IndexStore::Selected),
        store => Ok(store),
    }
}

/// Where a vertex function on `cx`'s profile stores index `index` taken alone. Vulkan takes a
/// primitive's `Layer` and `ViewportIndex` only where every vertex of it stores the same, Metal
/// the primitive's first vertex's whatever the others store. So: to its builtin where the profile
/// enables `shaderOutputLayer` (`shaderOutputViewportIndex` for a viewport index) and the function
/// writes the index the same at every vertex of a primitive (`uniform`), or the profile takes
/// `DriverDeviation::MetalFirstVertexIndices`; nowhere, a viewport index on a profile without
/// `multiViewport`; else to its selected index where the profile enables `geometryShader`.
/// Otherwise refused, as `VertexAmplification` for an index the function stores from the
/// amplification's offset alone (`amplification`), else as `LayeredRendering`: one written alike
/// needing the output feature or `geometryShader`, one not needing `geometryShader`.
fn own_index_store(
    cx: &SelectCx<'_>,
    index: RawIndex,
    uniform: bool,
    amplification: bool,
) -> Result<IndexStore, Error> {
    let f = cx.profile.fields();
    let (native, output) = match index {
        RawIndex::Layer => (f.fragment.output_layer, VkFeature::ShaderOutputLayer),
        RawIndex::Viewport => (
            f.fragment.output_viewport_index,
            VkFeature::ShaderOutputViewportIndex,
        ),
    };
    let first_vertex = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalFirstVertexIndices);
    if native && (uniform || first_vertex) {
        return Ok(IndexStore::Builtin);
    }
    if matches!(index, RawIndex::Viewport)
        && !f.enabled.features.contains(&VkFeature::MultiViewport)
    {
        return Ok(IndexStore::Nowhere);
    }
    if f.fragment.geometry_shader {
        return Ok(IndexStore::Selected);
    }
    let geometry = vec![Capability::Feature(VkFeature::GeometryShader)];
    let needs = match uniform {
        true => vec![vec![Capability::Feature(output)], geometry],
        false => vec![geometry],
    };
    Err(match amplification {
        true => Unsupported::needs_any_of(Feature::VertexAmplification, needs),
        false => Unsupported::needs_any_of(Feature::LayeredRendering, needs),
    }
    .into())
}

/// Whether vertex function `cx` writes index `index` the same at every vertex of a primitive
/// (its raw varying is among `StageIo::uniform_outputs`).
fn uniform_index(cx: &SelectCx<'_>, index: RawIndex) -> bool {
    let key = match index {
        RawIndex::Layer => VaryingKey::render_target_array_index(),
        RawIndex::Viewport => VaryingKey::viewport_index(),
    };
    (cx.families.stage_io.interface.uniform_outputs)
        .binary_search(&key)
        .is_ok()
}

/// Whether vertex function `cx`, with outputs `outputs`, drawn under `amplified` (`None` where it
/// returns nothing), stores one of the slice and viewport indices it stores to its selected index
/// ([`own_index_store`]). The pipeline's geometry stage then stores both builtins, and a vertex
/// function's own would not reach the rasterizer past it, so it stores every one there
/// ([`StageIoPlan::indices_selected`]). An index it writes and stores is uniform as written; one
/// it stores from the amplification's offset alone is uniform; a slice it writes in a pass of one
/// layer is stored nowhere. A refusal is left to the output that stores the index.
fn indices_selected(
    cx: &SelectCx<'_>,
    outputs: &[OutputBinding],
    amplified: Option<crate::pipeline::lower_inputs::AmplificationDraws>,
) -> bool {
    let writes = |role: fn(&OutputRole) -> bool| outputs.iter().any(|o| role(&o.role));
    let layer = writes(|r| matches!(r, OutputRole::RenderTargetArrayIndex));
    let viewport = writes(|r| matches!(r, OutputRole::ViewportArrayIndex));
    let dropped = cx.inputs.dropped_vertex_layer || amplified.is_some_and(|d| !d.layered);
    let stored = [
        (
            RawIndex::Layer,
            if layer {
                !dropped
            } else {
                amplified.is_some_and(|d| d.layered)
            },
            !layer || uniform_index(cx, RawIndex::Layer),
        ),
        (
            RawIndex::Viewport,
            viewport || amplified.is_some_and(|d| d.viewports),
            !viewport || uniform_index(cx, RawIndex::Viewport),
        ),
    ];
    stored.into_iter().any(|(index, stored, uniform)| {
        stored
            && matches!(
                own_index_store(cx, index, uniform, false),
                Ok(IndexStore::Selected)
            )
    })
}

/// The selected index's flat `uint` vertex output (`IoRole::SelectedLayer`,
/// `IoRole::SelectedViewport`), which the pipeline's synthesized geometry stage reads.
fn selected_index(sp: &mut StageIoPlan, index: RawIndex) -> Result<usize, Error> {
    let (key, role) = match index {
        RawIndex::Viewport => (VaryingKey::selected_viewport(), IoRole::SelectedViewport),
        RawIndex::Layer => (VaryingKey::selected_layer(), IoRole::SelectedLayer),
    };
    let ty = IoType {
        numeric: IoNumeric::Uint,
        components: 1,
        source_bits: 32,
        bits: 32,
    };
    let deco = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    push_var(
        &mut sp.interface.outputs,
        IoVariable {
            key: key.clone(),
            role,
            ty,
        },
    );
    user(sp, key, IoDir::Output, ty, deco)
}

/// Whether the profile can declare fragment input builtin `b`: one of its enabling
/// capabilities' features ([`builtin_enablers`]) is enabled.
fn declarable(cx: &SelectCx<'_>, b: crate::spv::BuiltIn) -> bool {
    let enabled = &cx.profile.fields().enabled.features;
    builtin_enablers(b, IoDir::Input)
        .iter()
        .any(|(_, f)| enabled.contains(f))
}

/// Plan fragment parameter `b` of AIR type `ty`, a scalar integer, reading index `builtin` as 0
/// (`Source::Absent`) on a profile that cannot declare it, listing it among
/// `StageIo::zero_indices`.
fn zero_index(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    ty: AirTypeId,
    builtin: StageBuiltin,
) -> Result<(), Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(f)?;
    let zeros = &mut cx.families.stage_io.interface.zero_indices;
    if let Err(at) = zeros.binary_search(&builtin) {
        zeros.insert(at, builtin);
    }
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::Absent(f),
    });
    Ok(())
}

/// The window depth's `float` variable in direction `dir` (`IoRole::WindowDepth`): a vertex
/// function's output, or the no-perspective input of a fragment function that computes each
/// sample's depth from it.
pub(super) fn window_depth(sp: &mut StageIoPlan, dir: IoDir) -> Result<usize, Error> {
    let key = VaryingKey::window_depth();
    let ty = IoType {
        numeric: IoNumeric::Float,
        components: 1,
        source_bits: 32,
        bits: 32,
    };
    let mut deco = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    let list = match dir {
        IoDir::Input => {
            deco.interpolation = SInterp::NoPerspective;
            &mut sp.interface.inputs
        }
        IoDir::Output => &mut sp.interface.outputs,
    };
    push_var(
        list,
        IoVariable {
            key: key.clone(),
            role: IoRole::WindowDepth,
            ty,
        },
    );
    user(sp, key, dir, ty, deco)
}

/// The primitive ID's flat `uint` variable in direction `dir` (`IoRole::PrimitiveId`): the input a
/// fragment function reads as `[[primitive_id]]` on a device without a fragment `PrimitiveId` or
/// under `LoweringInputs::primitive_id_varying`, or the output of a vertex function compiled under
/// that field.
pub(super) fn primitive_id(sp: &mut StageIoPlan, dir: IoDir) -> Result<usize, Error> {
    flat_uint(sp, VaryingKey::primitive_id(), IoRole::PrimitiveId, dir)
}

/// Where a vertex function reads its draw's first vertex, Vulkan's base vertex of a draw without
/// indices: the `BaseVertex` builtin with `shaderDrawParameters`, lane 0 of
/// `PushField::BaseVertexInstance` without (as for `[[base_vertex]]`, whose fact it states).
fn first_vertex(cx: &mut SelectCx<'_>) -> super::FirstVertex {
    cx.facts.insert(RuntimeFact::BaseVertexInstance);
    if !cx.profile.fields().draw_parameters {
        cx.push.insert(PushField::BaseVertexInstance);
        return super::FirstVertex::Push;
    }
    let index = builtin(
        &mut cx.families.stage_io,
        B::BaseVertex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::BaseVertex,
    );
    super::FirstVertex::Builtin { index }
}

/// Plan a vertex function's primitive ID (`LoweringInputs::primitive_id_varying`,
/// [`super::PrimitiveIdStore`]): its varying, the vertex index it divides, the draw's first
/// vertex (the `BaseVertex` builtin with `shaderDrawParameters`, lane 0 of
/// `PushField::BaseVertexInstance` without, as for `[[base_vertex]]`), and
/// `PushField::VerticesPerPrimitive`. A post-tessellation vertex function's vertices are the
/// tessellator's: one the device emulates writes its patch's position to the varying instead
/// (`lower::tessellation::emulated`), and one drawn through the device's tessellation stages,
/// whose primitive IDs are not built, is `NotImplemented(Tessellation)`, as for a fragment
/// function that reads `[[primitive_id]]` behind one (`link::tessellation`). A form
/// that reads its draw's indices at their address ([`captured_index_type`]) draws them as
/// `index count` vertices from vertex 0: its first vertex is 0
/// ([`super::FirstVertex::Zero`]). The private list form draws its list from its first vertex,
/// as a draw without indices does.
fn primitive_id_output(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) }) {
        return match crate::lower::tessellation::emulates(cx.profile) {
            true => Ok(()),
            false => Err(Unsupported::not_implemented(Feature::Tessellation).into()),
        };
    }
    cx.push.insert(PushField::VerticesPerPrimitive);
    let first = match captured_index_type(cx).is_some() && !listed(cx) {
        true => super::FirstVertex::Zero,
        false => first_vertex(cx),
    };
    let sp = &mut cx.families.stage_io;
    let vertex = builtin(
        sp,
        B::VertexIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::VertexIndex,
    );
    let object = primitive_id(sp, IoDir::Output)?;
    sp.primitive_id = Some(super::PrimitiveIdStore {
        vertex,
        first,
        object,
    });
    sp.words = sp.words.saturating_add(super::PRIMITIVE_ID_WORDS);
    Ok(())
}

/// The barycentric weights' `float2` input of form `perspective` (`IoRole::Barycentric`): a
/// center-sampled varying, perspective-correct or not, which the pipeline's synthesized geometry
/// stage writes.
fn barycentric_weights(sp: &mut StageIoPlan, perspective: bool) -> Result<usize, Error> {
    let key = VaryingKey::barycentric(perspective);
    let ty = IoType {
        numeric: IoNumeric::Float,
        components: 2,
        source_bits: 32,
        bits: 32,
    };
    let deco = IoDecor {
        interpolation: if perspective {
            SInterp::Smooth
        } else {
            SInterp::NoPerspective
        },
        ..IoDecor::default()
    };
    push_var(
        &mut sp.interface.inputs,
        IoVariable {
            key: key.clone(),
            role: IoRole::Barycentric { perspective },
            ty,
        },
    );
    user(sp, key, IoDir::Input, ty, deco)
}

/// The perspective and place of a `[[barycentric_coord]]` read of interpolation `i` into form
/// `f`, of a triangle drawn with Metal's last vertex first where `rotated`
/// (`LoweringInputs::rotated_weights`). Metal declares the weights at the pixel center (MSL
/// 5.2.3.4) and reads them there under every qualifier: a `centroid` or `sample` one reads the
/// center's (Metal probes on the Apple M3 Max, macOS 27.0.1, at one sample and at four). A
/// `float` or `float2` reads the leading weights, a `float3` all three ([`Place::Weights`]).
fn barycentric_read(i: AirInterp, f: Form, rotated: bool) -> Result<(bool, Place), Error> {
    let AirInterp::Smooth { perspective, .. } = i else {
        return Err(not_lowered());
    };
    let place = match f.held {
        Held::Float(ff) if ff.container() == 32 && (1..=3).contains(&f.lanes) => Place::Weights {
            lanes: f.lanes,
            rotated,
        },
        _ => return Err(not_lowered()),
    };
    Ok((perspective, place))
}

/// Records that the fragment function reads `[[barycentric_coord]]` through `place` of a triangle
/// drawn with Metal's last vertex first (`LoweringInputs::rotated_weights`), which its plans
/// state (`RuntimeFact::RotatedWeights`).
fn note_rotated_weights(cx: &mut SelectCx<'_>, place: Place) {
    if let Place::Weights { rotated: true, .. } = place {
        cx.facts.insert(RuntimeFact::RotatedWeights);
    }
}

/// Records that the device interpolates a fragment input of interpolation `interp`: a smooth one
/// takes Metal's answer only where the device is measured to interpolate as Metal does (the module
/// then requires that), elsewhere the device's, which the module records.
fn note_device_interpolation(cx: &mut SelectCx<'_>, interp: Interpolation) {
    if matches!(interp, Interpolation::Smooth { .. }) {
        if cx
            .profile
            .takes_driver_deviation(DriverDeviation::MetalInterpolation)
        {
            cx.families.stage_io.interpolation_by_metal = true;
        } else {
            cx.precision.insert(PrecisionNote::DeviceInterpolation);
        }
    }
}

/// The raw index's flat `uint` variable in direction `dir`: a vertex function's output, or the
/// input of a fragment function compiled under `LoweringInputs::viewport_index_varying` or
/// `LoweringInputs::render_target_array_index_varying`. Its type is fixed, so a `ushort` index on
/// either side links with a `uint` one on the other: the vertex function stores the index
/// zero-extended and the fragment function masks it to its width.
fn raw_index(sp: &mut StageIoPlan, index: RawIndex, dir: IoDir) -> Result<usize, Error> {
    let (key, role) = match index {
        RawIndex::Viewport => (VaryingKey::viewport_index(), IoRole::ViewportIndex),
        RawIndex::Layer => (
            VaryingKey::render_target_array_index(),
            IoRole::RenderTargetArrayIndex,
        ),
    };
    flat_uint(sp, key, role, dir)
}

/// The flat `uint` variable of `key` in role `role` in direction `dir`.
fn flat_uint(
    sp: &mut StageIoPlan,
    key: VaryingKey,
    role: IoRole,
    dir: IoDir,
) -> Result<usize, Error> {
    let ty = IoType {
        numeric: IoNumeric::Uint,
        components: 1,
        source_bits: 32,
        bits: 32,
    };
    let mut deco = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    let list = match dir {
        IoDir::Input => {
            deco.interpolation = SInterp::Flat;
            &mut sp.interface.inputs
        }
        IoDir::Output => &mut sp.interface.outputs,
    };
    push_var(
        list,
        IoVariable {
            key: key.clone(),
            role,
            ty,
        },
    );
    user(sp, key, dir, ty, deco)
}

/// An input builtin's Vulkan builtin, its shape, the place of the value, the contract entry, and
/// the enabling capability chosen among the builtin's alternatives.
type InputBuiltin = (
    crate::spv::BuiltIn,
    Shape,
    Place,
    StageBuiltin,
    Option<crate::spv::Capability>,
);

/// The enabling capability of fragment input builtin `b`: the first of its alternatives
/// ([`builtin_enablers`]) whose feature the profile enables, or the refusal as `feature` needing
/// any one of those features.
fn enabler(
    cx: &SelectCx<'_>,
    b: crate::spv::BuiltIn,
    feature: Feature,
) -> Result<Option<crate::spv::Capability>, Error> {
    let alternatives = builtin_enablers(b, IoDir::Input);
    let enabled = &cx.profile.fields().enabled.features;
    match alternatives.iter().find(|(_, f)| enabled.contains(f)) {
        Some(&(cap, _)) => Ok(Some(cap)),
        None => Err(Unsupported::needs_any_of(
            feature,
            alternatives
                .iter()
                .map(|&(_, f)| vec![Capability::Feature(f)])
                .collect(),
        )
        .into()),
    }
}

/// Check that the profile can declare `ViewIndex`, Metal's `[[amplification_id]]`: the `MultiView`
/// capability needs `multiview` (mandatory since Vulkan 1.1). The builtin has no alternative
/// enabling capabilities.
pub(crate) fn view_index_enabled(
    cx: &SelectCx<'_>,
) -> Result<Option<crate::spv::Capability>, Error> {
    if cx
        .profile
        .fields()
        .enabled
        .features
        .contains(&VkFeature::Multiview)
    {
        Ok(None)
    } else {
        Err(needs(Feature::VertexAmplification, VkFeature::Multiview))
    }
}

/// The form of `[[amplification_count]]` of AIR type `ty`: a scalar integer of at most 32 bits,
/// which reads the `SpecKey::AmplificationCount` constant (`RuntimeFact::AmplificationCount`).
pub(crate) fn amplification_count_form(
    cx: &mut SelectCx<'_>,
    ty: AirTypeId,
) -> Result<Form, Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(f)?;
    cx.facts.insert(RuntimeFact::AmplificationCount);
    Ok(f)
}

/// The Vulkan builtin an input builtin reads (see [`InputBuiltin`]), after checking the
/// parameter's form and the profile.
fn input_builtin(
    cx: &SelectCx<'_>,
    bi: Builtin,
    f: Form,
    frag: crate::device::profile::FragmentCaps,
) -> Result<InputBuiltin, Error> {
    use Builtin as Bi;
    let vertex = cx.stage == Stage::Vertex;
    Ok(match bi {
        Bi::VertexId if vertex => {
            int_scalar(f)?;
            (
                B::VertexIndex,
                Shape::U32,
                Place::Whole,
                StageBuiltin::VertexIndex,
                None,
            )
        }
        Bi::InstanceId if vertex => {
            int_scalar(f)?;
            (
                B::InstanceIndex,
                Shape::U32,
                Place::Whole,
                StageBuiltin::InstanceIndex,
                None,
            )
        }
        // With `shaderDrawParameters` (the push lanes otherwise, `draw_base`): Vulkan's
        // `BaseVertex` is a draw's first vertex without an index buffer and its vertex offset
        // with one, Metal's `vertexStart` and `baseVertex`, and `BaseInstance` its first
        // instance, Metal's `baseInstance`.
        Bi::BaseVertex if vertex => {
            int_scalar(f)?;
            (
                B::BaseVertex,
                Shape::U32,
                Place::Whole,
                StageBuiltin::BaseVertex,
                None,
            )
        }
        Bi::BaseInstance if vertex => {
            int_scalar(f)?;
            (
                B::BaseInstance,
                Shape::U32,
                Place::Whole,
                StageBuiltin::BaseInstance,
                None,
            )
        }
        // The view a multiview draw runs for is the amplification: the binder's view mask makes
        // view `a` amplification `a`, and without multiview both are 0.
        Bi::AmplificationId => {
            int_scalar(f)?;
            (
                B::ViewIndex,
                Shape::U32,
                Place::Whole,
                StageBuiltin::ViewIndex,
                view_index_enabled(cx)?,
            )
        }
        Bi::AmplificationCount => return Err(cx.invariant("an amplification count as a builtin")),
        Bi::PositionInPatch | Bi::PatchId => {
            return Err(Unsupported::not_implemented(Feature::Tessellation).into())
        }
        Bi::Position(i) if !vertex => {
            // Metal's fragment position is the pixel-center, non-perspective position; another
            // sampling qualifier is not modelled.
            match i {
                None
                | Some(AirInterp::Smooth {
                    sampling: AirSampling::Center,
                    ..
                }) => {}
                Some(_) => return Err(not_lowered()),
            }
            float_lanes(f, 4)?;
            (
                B::FragCoord,
                Shape::FVec(4),
                Place::Whole,
                StageBuiltin::FragCoord,
                None,
            )
        }
        Bi::FrontFacing if !vertex => {
            if f.held != Held::Bool || f.lanes != 1 {
                return Err(not_lowered());
            }
            (
                B::FrontFacing,
                Shape::Bool,
                Place::Whole,
                StageBuiltin::FrontFacing,
                None,
            )
        }
        Bi::PointCoord if !vertex => {
            float_lanes(f, 2)?;
            (
                B::PointCoord,
                Shape::FVec(2),
                Place::Whole,
                StageBuiltin::PointCoord,
                None,
            )
        }
        Bi::SampleId if !vertex => {
            int_scalar(f)?;
            if !frag.sample_rate_shading {
                return Err(needs(
                    Feature::SampleRateShading,
                    VkFeature::SampleRateShading,
                ));
            }
            (
                B::SampleId,
                Shape::U32,
                Place::Whole,
                StageBuiltin::SampleId,
                None,
            )
        }
        // The coverage after the early tests reads the same builtin under `PostDepthCoverage`
        // (the entry's mode, `ParamRole::Builtin` in `plan`).
        Bi::SampleMaskIn | Bi::PostDepthCoverage if !vertex => {
            int_scalar(f)?;
            (
                B::SampleMask,
                Shape::U32Array(1),
                Place::Element0,
                StageBuiltin::SampleMaskIn,
                None,
            )
        }
        Bi::PrimitiveId if !vertex => {
            int_scalar(f)?;
            (
                B::PrimitiveId,
                Shape::U32,
                Place::Whole,
                StageBuiltin::PrimitiveId,
                enabler(cx, B::PrimitiveId, Feature::PrimitiveId)?,
            )
        }
        Bi::RenderTargetArrayIndex if !vertex => {
            int_scalar(f)?;
            (
                B::Layer,
                Shape::U32,
                Place::Whole,
                StageBuiltin::Layer,
                enabler(cx, B::Layer, Feature::LayeredRendering)?,
            )
        }
        Bi::ViewportArrayIndex if !vertex => {
            int_scalar(f)?;
            (
                B::ViewportIndex,
                Shape::U32,
                Place::Whole,
                StageBuiltin::ViewportIndex,
                enabler(cx, B::ViewportIndex, Feature::LayeredRendering)?,
            )
        }
        Bi::BarycentricCoord(i) if !vertex => {
            // Metal declares the weights only at the pixel center, perspective-correct or not
            // (MSL 5.2.3.4); a `float` or `float2` reads the leading weights (the rest are
            // ignored), a `float3` all three. Metal's third weight is `1 - (x + y)` of the first
            // two in `float`, at every pixel and sample of triangles and lines, perspective or
            // not, where its first two are the interpolation of a varying that holds `(1, 0)`,
            // `(0, 1)` and `(0, 0)` at the primitive's vertices (Metal probes on the Apple M3 Max,
            // macOS 27.0.1: 574 of 574 pixels at one sample, 2,341 of 2,341 samples at four, 86
            // of 86 line pixels). `BaryCoordKHR`'s first two lanes are that interpolation (every
            // pixel of every topology on the RTX 5080 Laptop GPU), its third the device's own
            // (there 42 of 574 pixels `1 - (x + y)`, and a line's exactly 0), so a `float3` read
            // takes the first two and computes the third ([`Place::Weights`]). Both orders are
            // the primitive's vertex order, Vulkan's first-vertex order, which is Metal's; a
            // triangle drawn with Metal's last vertex first takes them rotated back
            // (`LoweringInputs::rotated_weights`).
            let (perspective, place) = barycentric_read(i, f, cx.inputs.rotated_weights)?;
            // `param` reads the weights from a varying without `fragmentShaderBarycentric`.
            if !frag.barycentric {
                return Err(cx.invariant("a barycentric builtin without the feature"));
            }
            // A driver that rejects a shader reading both forms refuses the second.
            let defect = DriverWorkaround::BothBarycentricFormsRejected;
            let other = StageBuiltin::BaryCoord {
                perspective: !perspective,
            };
            if cx.families.stage_io.interface.builtins.contains(&other)
                && cx.profile.fields().workarounds.contains(&defect)
            {
                return Err(Unsupported::driver_defect(Feature::Barycentrics, defect).into());
            }
            let b = if perspective {
                B::BaryCoordKHR
            } else {
                B::BaryCoordNoPerspKHR
            };
            (
                b,
                Shape::FVec(3),
                place,
                StageBuiltin::BaryCoord { perspective },
                None,
            )
        }
        _ => return Err(not_lowered()),
    })
}

/// The shape of the entry's returned value and the AIR type of each output in output order: the
/// returned value itself, or the returned struct's field the output names ([`OutputBinding::field`];
/// a gated-off output's field has no output).
pub(crate) fn output_types(cx: &SelectCx<'_>) -> Result<(Returned, Vec<AirTypeId>), Error> {
    let e = cx.entry;
    let func = e
        .module
        .func(e.func)
        .ok_or_else(|| cx.invariant("the entry function is missing"))?;
    let types = e.module.types();
    let Some(Type::Function { ret, .. }) = types.get(func.ty) else {
        return Err(cx.invariant("the entry has no function type"));
    };
    let (shape, fields): (Returned, &[AirTypeId]) = match types.get(*ret) {
        Some(Type::Void) => (Returned::Void, &[]),
        // The ABI decoder checked one output per field (or one for a non-struct value).
        Some(Type::Struct { fields, .. }) => (Returned::Struct, fields),
        Some(_) => (Returned::Value, std::slice::from_ref(ret)),
        None => return Err(cx.invariant("the entry returns an unknown type")),
    };
    let mut tys = Vec::with_capacity(e.ep.outputs.len());
    let mut last = None;
    for o in &e.ep.outputs {
        if last.is_some_and(|l| o.field <= l) {
            return Err(cx.invariant("the outputs are not in field order"));
        }
        last = Some(o.field);
        let ty = fields
            .get(o.field as usize)
            .copied()
            .ok_or_else(|| cx.invariant("an output names no field of the returned value"))?;
        tys.push(ty);
    }
    Ok((shape, tys))
}

/// Plan the entry's outputs and execution modes (vertex and fragment). A post-tessellation
/// vertex function's outputs are a vertex function's; its inputs and patch domain are
/// `lower::tessellation`'s.
pub fn entry(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.inputs.merged_draws && cx.stage == Stage::Vertex {
        merged(cx)?;
    }
    let (returned, tys) = output_types(cx)?;
    cx.families.stage_io.returned = returned;
    // A fragment function of one of its pipeline's color groups (`LoweringInputs::color_group`),
    // its colors renumbered before planning (`pipeline::compile::color_group::entry`): stated for
    // link, and, where the pipeline takes more than one group, its per-fragment outputs invariant
    // across the groups' variants, and the note of what is not.
    if cx.stage == Stage::Fragment {
        if let Some(group) = cx.inputs.color_group.clone() {
            if group.groups() > 1 {
                cx.families.stage_io.invariant_fragment_outputs = true;
                cx.precision.insert(PrecisionNote::ColorAttachmentGroups);
            }
            cx.families.stage_io.interface.color_group = Some(group);
        }
    }
    // A fragment function compiled for a multisampled pipeline reads `[[position]]` at the pixel
    // center.
    cx.families.stage_io.interface.multisampled =
        cx.stage == Stage::Fragment && cx.inputs.multisampled;
    let outputs = cx.entry.ep.outputs.clone();
    // Drawn one amplification at a time, a vertex function adds the amplification's offsets to
    // the indices it writes, and writes those it does not where the draw needs them.
    let amplified = cx
        .inputs
        .amplification_draws
        .filter(|_| cx.stage == Stage::Vertex);
    if let Some(d) = amplified {
        cx.facts.insert(RuntimeFact::AmplificationDraws {
            layered: d.layered,
            viewports: d.viewports,
        });
    }
    let offset = |field| amplified.map(|_| field);
    if cx.stage == Stage::Vertex {
        cx.families.stage_io.indices_selected = indices_selected(
            cx,
            &outputs,
            amplified.filter(|_| returned != Returned::Void),
        );
    }
    // The pipeline's colors the pass lacks, whose writes plain Metal drops: stated whether or
    // not the function writes them, so the plan settles on the variant.
    let dropped = match cx.stage {
        Stage::Fragment => cx.inputs.dropped_colors.clone(),
        _ => std::collections::BTreeSet::new(),
    };
    for &color in &dropped {
        cx.facts.insert(RuntimeFact::DroppedColor { color });
    }
    // The colors the render pipeline has no attachment for, whose writes plain Metal discards:
    // stored nowhere, as a dropped color's, and listed on the interface for link.
    let unattached = match cx.stage {
        Stage::Fragment => cx.inputs.unattached_colors.clone(),
        _ => std::collections::BTreeSet::new(),
    };
    let mut point_size = false;
    // The first outputs to rotated attachments whose blend reads a second output the rotation
    // needs: the attachment, the returned field and its form.
    let mut seconds = Vec::new();
    for (o, ty) in outputs.iter().zip(tys) {
        if matches!(o.role, OutputRole::PointSize) {
            point_size = true;
        }
        // The private alpha test reads color 0's alpha as the function returns it, whatever the
        // pipeline stores of it.
        if let (Stage::Fragment, Some(function)) = (cx.stage, cx.inputs.alpha_test) {
            if let OutputRole::RenderTarget {
                attachment: AbiInt::Const(0),
                index: 0,
                ..
            } = o.role
            {
                let f = io_form(cx, ty)?;
                if !f.is_float() || f.lanes != 4 {
                    return Err(alpha_test_refused());
                }
                cx.families.stage_io.alpha_test = Some(super::AlphaTest {
                    field: o.field,
                    function,
                });
            }
        }
        // A `bfloat` color output is not Metal's (its compiler rejects the return type), whether
        // or not the pass or pipeline has the attachment.
        if cx.stage == Stage::Fragment && matches!(o.role, OutputRole::RenderTarget { .. }) {
            refuse_bfloat_color(cx.entry, ty)?;
            // A `half` output to an attachment the pipeline has: Metal may carry its last
            // operation at `float` precision into the blend and the format's conversion, where
            // the function rounds it to `half` first. An attachment whose channels take the
            // output at `float` precision where Metal does is recorded, and the arith family's plan
            // decides its note (`arith::output_precision`); without the attachment's entry (a
            // function compiled without the map among them) the output keeps the `half` and the
            // note, and links all the same.
            if let OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                ..
            } = o.role
            {
                let lanes = cx.inputs.half_output_lanes.get(&color).copied();
                match lanes.filter(|_| half_color(cx.entry, ty)) {
                    Some(lanes) => {
                        let sp = &mut cx.families.stage_io;
                        sp.interface.half_output_lanes.insert(color, lanes);
                    }
                    None if half_color(cx.entry, ty) && !unattached.contains(&color) => {
                        crate::lower::arith::output_precision::note_half_output(cx, color);
                    }
                    None => {}
                }
                // A `float` output of a module without fast math: the pipeline decides by the
                // attachment's blend whether Metal stores it otherwise (`float_blend_colors`).
                if cx.entry.compile.fast_math == Some(false)
                    && float_color(cx.entry, ty)
                    && !unattached.contains(&color)
                {
                    let colors = &mut cx.families.stage_io.interface.float_blend_colors;
                    if let Err(at) = colors.binary_search(&color) {
                        colors.insert(at, color);
                    }
                }
            }
        }
        if matches!(o.role, OutputRole::RenderTargetArrayIndex) && cx.stage == Stage::Vertex {
            // The slice a vertex function chooses selects one only in a pass of at least two
            // layers; in a pass of one layer every index renders slice 0, and the variant for it
            // does not write `Layer`. In every variant the raw varying holds the index as written,
            // which Metal's fragment function reads. Drawn one amplification at a time, a pass of
            // one layer writes no `Layer` either.
            let dropped = cx.inputs.dropped_vertex_layer || amplified.is_some_and(|d| !d.layered);
            cx.facts.insert(RuntimeFact::RenderTargetArrayLength {
                layer_dropped: dropped,
            });
            if dropped {
                let f = io_form(cx, ty)?;
                int_scalar(f)?;
                if amplified.is_some() {
                    cx.push.insert(PushField::AmplificationLayerOffset);
                }
                let sp = &mut cx.families.stage_io;
                let object = raw_index(sp, RawIndex::Layer, IoDir::Output)?;
                sp.outputs.push(PlannedOutput {
                    field: o.field,
                    link: Link {
                        object,
                        place: Place::Whole,
                        form: f,
                    },
                    below: None,
                    color: None,
                    rotated: false,
                    alpha_red: false,
                    premultiplied: false,
                    offset: offset(PushField::AmplificationLayerOffset),
                });
                continue;
            }
        }
        // The tessellator emits only triangles, so a post-tessellation vertex function's point
        // size sizes nothing and dropping it is exact: stored nowhere, as a gated-off output, its
        // type checked as a vertex function's point size is.
        if matches!(o.role, OutputRole::PointSize)
            && matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) })
        {
            float_lanes(io_form(cx, ty)?, 1)?;
            continue;
        }
        // An imageblock output is written to the pass's imageblock storage, not an interface
        // variable.
        if let OutputRole::ImageblockData(ib) = &o.role {
            if cx.stage != Stage::Fragment {
                return Err(cx.invariant("an imageblock output outside a fragment function"));
            }
            crate::lower::fragment::select_imageblock_output(cx, o.field, ib, o.ty.align, ty)?;
            continue;
        }
        // Compiled for draws whose fragment function runs their depth and stencil tests itself
        // (`fragment::depth_test`), the device's stencil test is off: a stencil output is the
        // in-shader stencil test's reference where the draws' stencil test acts, and changes
        // nothing where it does not. Stored nowhere, on any device, so no shader-side stencil.
        if matches!(o.role, OutputRole::Stencil)
            && cx.stage == Stage::Fragment
            && cx.inputs.shader_tests
        {
            int_scalar_form(cx, ty)?;
            continue;
        }
        // A stencil output on a device without stencil export is the shader-side stencil,
        // written to the stencil's shadow, not an interface variable.
        if matches!(o.role, OutputRole::Stencil)
            && cx.stage == Stage::Fragment
            && !cx.profile.fields().fragment.stencil_export
        {
            crate::lower::fragment::select_stencil_output(cx, o.field, ty)?;
            continue;
        }
        // An output to a color the pass lacks is stored nowhere, as plain Metal drops it: no
        // output variable, so the pipeline writes no location without an attachment. So is one to a
        // color the pipeline has no attachment for.
        if let OutputRole::RenderTarget { attachment, .. } = &o.role {
            if !dropped.is_empty() || !unattached.is_empty() {
                let AbiInt::Const(color) = attachment else {
                    return Err(not_lowered());
                };
                if dropped.contains(color) {
                    continue;
                }
                if unattached.contains(color) {
                    let listed = &mut cx.families.stage_io.interface.unattached_colors;
                    if let Err(at) = listed.binary_search(color) {
                        listed.insert(at, *color);
                    }
                    continue;
                }
            }
        }
        // An output to an attachment the function reads in the interlocked storage form is
        // written to its storage image, not an interface variable.
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index: 0,
            ..
        } = o.role
        {
            if cx.stage == Stage::Fragment
                && crate::lower::fragment::writes_through_storage(cx, color)
            {
                crate::lower::fragment::select_storage_output(cx, o.field, color, ty)
                    .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
                continue;
            }
        }
        // A second output of the function's own to an attachment whose blend reads the one the
        // rotation needs (`LoweringInputs::rotated_factors`) is stored nowhere: Metal's blend
        // there reads no second output.
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index: 1,
            ..
        } = o.role
        {
            if cx.stage == Stage::Fragment && cx.inputs.rotated_factors.contains_key(&color) {
                continue;
            }
            // The second source of an extended-range or generic blend the shader computes in the
            // storage form, which the blend reads from the returned value.
            if cx.stage == Stage::Fragment
                && crate::lower::fragment::blends_second_through_storage(cx, color)
            {
                crate::lower::fragment::select_storage_second_output(cx, o.field, color, ty)
                    .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
                continue;
            }
            // The second source of an extended-range blend the function computes over the
            // attachment's input attachment, which the blend reads from the returned value.
            if cx.stage == Stage::Fragment
                && crate::lower::fragment::blends_second_through_inputs(cx, color)
            {
                crate::lower::fragment::select_input_second_output(cx, o.field, color, ty)
                    .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
                continue;
            }
            // Every other write of an extended-range attachment (the storage form's of none of its
            // channels or unblended, or the raw texel's through the interface) reads no second
            // output, as Metal's write there reads none: it is stored nowhere, where the
            // interface's raw texel refused it `Format`.
            if cx.stage == Stage::Fragment && cx.inputs.extended_range_colors.contains_key(&color) {
                continue;
            }
            // A word attachment's write is unblended: a second output there is stored nowhere,
            // as Metal's unblended write reads none.
            if cx.stage == Stage::Fragment && cx.inputs.word_colors.contains_key(&color) {
                continue;
            }
        }
        // On a device without `multiViewport` a draw has one viewport, so the index selects
        // viewport 0 whatever it is: no `ViewportIndex` is stored, and the raw varying holds the
        // index as written, which Metal's fragment function reads. Drawn one amplification at a
        // time, it holds the index plus the amplification's offset.
        if matches!(o.role, OutputRole::ViewportArrayIndex)
            && cx.stage == Stage::Vertex
            && index_store(
                cx,
                RawIndex::Viewport,
                uniform_index(cx, RawIndex::Viewport),
                false,
            )? == IndexStore::Nowhere
        {
            let f = io_form(cx, ty)?;
            int_scalar(f)?;
            if amplified.is_some() {
                cx.push.insert(PushField::AmplificationViewportOffset);
            }
            let sp = &mut cx.families.stage_io;
            let object = raw_index(sp, RawIndex::Viewport, IoDir::Output)?;
            sp.outputs.push(PlannedOutput {
                field: o.field,
                link: Link {
                    object,
                    place: Place::Whole,
                    form: f,
                },
                below: None,
                color: None,
                rotated: false,
                alpha_red: false,
                premultiplied: false,
                offset: offset(PushField::AmplificationViewportOffset),
            });
            continue;
        }
        // A color output's refusal is tied to its attachment.
        let link = output(cx, o, ty).map_err(|e| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                ..
            } => e.at_attachment(Attachment::Color(color)),
            _ => e,
        })?;
        // A vertex function's viewport index selects a viewport only below the draw's viewport
        // count: `ViewportIndex` holds 0 at or past it, where Metal draws through viewport 0. Its
        // render target slice index selects a layer only below the pass's length: `Layer` holds 0
        // at or past it, where Metal renders the pass's first layer. For each the raw varying holds
        // the index as written, which Metal's fragment function reads. Drawn one amplification at a
        // time, each holds the index plus the amplification's offset, and the raw varying that sum,
        // which Metal's fragment function reads.
        let raw = match o.role {
            OutputRole::ViewportArrayIndex if cx.stage == Stage::Vertex => {
                cx.facts.insert(RuntimeFact::ViewportCount);
                Some((
                    RawIndex::Viewport,
                    PushField::ViewportCount,
                    PushField::AmplificationViewportOffset,
                ))
            }
            OutputRole::RenderTargetArrayIndex if cx.stage == Stage::Vertex => Some((
                RawIndex::Layer,
                PushField::RenderTargetArrayLength,
                PushField::AmplificationLayerOffset,
            )),
            _ => None,
        };
        // A color output of dual-source index 0 rounds to the code Metal stores where the
        // attachment's format converts it.
        let color = match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if cx.stage == Stage::Fragment => {
                super::color_write::plan(cx, color, link.form);
                Some(color)
            }
            _ => None,
        };
        // An output to an attachment held with its channels rotated stores them rotated; one to an
        // attachment holding Metal's alpha in red holds that alpha in the image's alpha lane too,
        // where alpha to coverage reads it.
        let mut premultiplied = false;
        let mut alpha_red = false;
        let rotated = match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(attachment),
                index,
                ..
            } if cx.stage == Stage::Fragment
                && cx.inputs.rotated_colors.contains(&attachment)
                && link.form.float_fmt().is_some() =>
            {
                if link.form.lanes != 4 {
                    let refused = Unsupported::not_implemented(Feature::Format);
                    return Err(refused.at_attachment(Attachment::Color(attachment)).into());
                }
                let sp = &mut cx.families.stage_io;
                sp.interface.rotated_colors.insert(attachment);
                if cx.inputs.alpha_in_red_colors.contains(&attachment) {
                    sp.interface.alpha_in_red_colors.insert(attachment);
                    alpha_red = true;
                }
                if let (0, Some(&lanes)) = (index, cx.inputs.rotated_factors.get(&attachment)) {
                    seconds.push((attachment, o.field, link.form, lanes));
                }
                if let (0, Some(&form)) = (index, cx.inputs.rotated_blends.get(&attachment)) {
                    premultiplied = rotated_blend_form(cx, attachment, form, link.form)
                        .map_err(|e| e.at_attachment(Attachment::Color(attachment)))?;
                }
                true
            }
            _ => false,
        };
        let added = raw.and_then(|(_, _, field)| offset(field));
        cx.families.stage_io.outputs.push(PlannedOutput {
            field: o.field,
            link,
            below: raw.map(|(_, count, _)| count),
            color,
            rotated,
            alpha_red,
            premultiplied,
            offset: added,
        });
        if let Some((index, count, _)) = raw {
            cx.push.insert(count);
            if let Some(field) = added {
                cx.push.insert(field);
            }
            let sp = &mut cx.families.stage_io;
            let object = raw_index(sp, index, IoDir::Output)?;
            sp.outputs.push(PlannedOutput {
                field: o.field,
                link: Link { object, ..link },
                below: None,
                color: None,
                rotated: false,
                alpha_red: false,
                premultiplied: false,
                offset: added,
            });
        }
    }
    for (attachment, field, form, lanes) in seconds {
        let second = rotated_second(cx, attachment, (field, form), lanes)
            .map_err(|e| e.at_attachment(Attachment::Color(attachment)))?;
        cx.families.stage_io.rotated_seconds.push(second);
    }
    // The indices a draw of one amplification stores though the function writes none: `Layer`
    // in a pass of at least two layers, and `ViewportIndex` under mappings that offset the
    // viewport, each the amplification's offset held below its count as an index the function
    // wrote would be. A function without outputs rasterizes nothing.
    if let Some(d) = amplified.filter(|_| returned != Returned::Void) {
        let writes = |role: fn(&OutputRole) -> bool| outputs.iter().any(|o| role(&o.role));
        if d.layered && !writes(|r| matches!(r, OutputRole::RenderTargetArrayIndex)) {
            amplified_output(cx, AmplifiedIndex::Layer)?;
        }
        if d.viewports && !writes(|r| matches!(r, OutputRole::ViewportArrayIndex)) {
            amplified_output(cx, AmplifiedIndex::Viewport)?;
        }
    }
    // A viewport Vulkan's cannot be: the position compensated for it.
    if let Some(c) = cx
        .inputs
        .viewport_compensation
        .filter(|_| cx.stage == Stage::Vertex)
    {
        compensate(cx, c)?;
    }
    // The window depth of the position the function stores, for a fragment function that
    // computes each sample's depth from it. A function that writes no position rasterizes
    // nothing.
    if cx.inputs.window_depth_varying && cx.stage == Stage::Vertex {
        let sp = &mut cx.families.stage_io;
        let position = sp.objects.iter().position(|o| {
            matches!(
                o,
                IoObject::Builtin {
                    builtin: B::Position,
                    dir: IoDir::Output,
                    ..
                }
            )
        });
        if let Some(position) = position {
            let object = window_depth(sp, IoDir::Output)?;
            sp.window_depth = Some(super::WindowDepth { position, object });
            sp.words = sp.words.saturating_add(super::WINDOW_DEPTH_WORDS);
        }
    }
    // The capture form: each vertex's clip position and point size, written to its record.
    if let (Some(form), Stage::Vertex) = (cx.inputs.captured_positions, cx.stage) {
        capture(cx, form)?;
    }
    // The primitive ID a fragment function reads from its varying (an emulated post-tessellation
    // vertex function's is its patch's position, `lower::tessellation::emulated`).
    if cx.inputs.primitive_id_varying && cx.stage == Stage::Vertex {
        primitive_id_output(cx)?;
    }
    // The pass's colors the pipeline lacks, which plain Metal zeroes where a draw covers them.
    if cx.stage == Stage::Fragment {
        let zeroed = cx.inputs.zeroed_colors.clone();
        for (color, format) in zeroed {
            zeroed_output(cx, &outputs, color, format)
                .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
        }
    }
    match cx.entry.ep.stage {
        // The tessellator emits triangles: no point size to default.
        AirStage::Vertex { patch: Some(_) } => {}
        // A vertex function without a point-size output writes 1.0, one without outputs (a
        // rasterization-disabled function) too: the module does not know the pipeline's
        // topology, and a point-list pipeline's vertex stage must write `PointSize` without
        // `maintenance5`, rasterizer discard included
        // (`VUID-VkGraphicsPipelineCreateInfo-topology-08773`).
        AirStage::Vertex { patch: None } => {
            if !point_size {
                let sp = &mut cx.families.stage_io;
                let i = builtin(
                    sp,
                    B::PointSize,
                    IoDir::Output,
                    Shape::F32,
                    StageBuiltin::PointSize { default: true },
                );
                sp.point_size_default = Some(i);
            }
        }
        AirStage::Fragment {
            early_fragment_tests,
        } => {
            let sp = &mut cx.families.stage_io;
            sp.modes.insert(0, ExecMode::OriginUpperLeft);
            sp.interface.early_fragment_tests = early_fragment_tests;
            if early_fragment_tests {
                sp.modes.push(ExecMode::EarlyFragmentTests);
            }
        }
        AirStage::Kernel { .. } | AirStage::Object { .. } | AirStage::Mesh { .. } => {}
    }
    Ok(())
}

/// Plan a vertex function's capture form `form` (`LoweringInputs::captured_positions`): the
/// record of its `Position` output and its own `PointSize` output ([`super::Capture`]), the
/// record's address and layout push fields, the vertex and instance indices, the indexed form's
/// index read ([`captured_index`], whatever else the function reads: its record's restart flag
/// and a strip's restart test need it), and
/// `RuntimeFact::CapturedPositions`, which every function compiled in the form states, so its
/// plans state the variant. A function that writes no position rasterizes nothing and plans no
/// record. Refused [`Feature::TileReach`]: a post-tessellation vertex function (its vertices are
/// the tessellator's, not the draw's), and a device without device addresses or without the
/// vertex stage's stores, naming what it needs.
fn capture(cx: &mut SelectCx<'_>, form: CaptureForm) -> Result<(), Error> {
    if matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) }) {
        return Err(Unsupported::not_implemented(Feature::TileReach).into());
    }
    cx.facts.insert(RuntimeFact::CapturedPositions { form });
    // The indexed form reads its index whatever else it reads: a strip's restart index runs no
    // body.
    if let Some(ty) = form.index_type() {
        captured_index(cx, ty)?;
    }
    let output = |o: &IoObject, b| matches!(o, IoObject::Builtin { builtin, dir: IoDir::Output, .. } if *builtin == b);
    let sp = &cx.families.stage_io;
    let Some(position) = sp.objects.iter().position(|o| output(o, B::Position)) else {
        return Ok(());
    };
    // The function's own point size: the default one the vertex stage writes is planned later,
    // and the record takes 1.0 for it.
    let point_size = sp
        .outputs
        .iter()
        .map(|o| o.link.object)
        .find(|&i| sp.objects.get(i).is_some_and(|o| output(o, B::PointSize)));
    refuse_unaddressed(cx)?;
    let f = cx.profile.fields();
    let enabled = &f.enabled.features;
    if let Some(stores) =
        crate::lower::access::dynamic::stage_stores(Stage::Vertex).find(|s| !enabled.contains(s))
    {
        return Err(Unsupported::needs(Feature::TileReach, Capability::Feature(stores)).into());
    }
    cx.push.insert(PushField::CaptureAddress);
    cx.push.insert(PushField::CaptureLayout);
    let sp = &mut cx.families.stage_io;
    let vertex = builtin(
        sp,
        B::VertexIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::VertexIndex,
    );
    let instance = builtin(
        sp,
        B::InstanceIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::InstanceIndex,
    );
    sp.capture = Some(super::Capture {
        position,
        point_size,
        vertex,
        instance,
    });
    sp.words = sp.words.saturating_add(super::CAPTURE_WORDS);
    Ok(())
}

/// The index type of the draw whose indices a vertex function reads at their address: the
/// indexed capture form's (`CaptureForm::Indexed`), else the private list form's `u32`
/// (`LoweringInputs::private_list`), else the primitive ID's de-indexed form's
/// (`LoweringInputs::primitive_id_indices`, under `primitive_id_varying`); `None` for any other
/// function.
pub(super) fn captured_index_type(cx: &SelectCx<'_>) -> Option<crate::metal::IndexType> {
    match (cx.stage, cx.inputs.captured_positions) {
        (Stage::Vertex, Some(form)) => form.index_type(),
        (Stage::Vertex, None) if cx.inputs.private_list => Some(crate::metal::IndexType::UInt32),
        (Stage::Vertex, None) => cx
            .inputs
            .primitive_id_indices
            .filter(|_| cx.inputs.primitive_id_varying),
        _ => None,
    }
}

/// Whether a vertex function runs in the private list form (`LoweringInputs::private_list`): its
/// list's positions from the draw's first vertex, Metal's base vertex in the word before.
pub(super) fn listed(cx: &SelectCx<'_>) -> bool {
    cx.stage == Stage::Vertex && cx.inputs.captured_positions.is_none() && cx.inputs.private_list
}

/// The feature a form that reads a draw's indices or records at their addresses serves: tile
/// reach for the capture form, Metal's private primitive types for the private list form, the
/// primitive ID for its de-indexed form.
fn indices_feature(cx: &SelectCx<'_>) -> Feature {
    match cx.inputs.captured_positions {
        Some(_) => Feature::TileReach,
        None if listed(cx) => Feature::PrivatePrimitives,
        None => Feature::PrimitiveId,
    }
}

/// Refuse the capture form or the de-indexed form ([`indices_feature`]) on a device without
/// device addresses, naming what it needs: the records and the indices are reached by address.
fn refuse_unaddressed(cx: &SelectCx<'_>) -> Result<(), Error> {
    if cx.profile.fields().buffer_access != crate::device::profile::BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            indices_feature(cx),
            vec![cx.profile.addressed_needs()],
        )
        .into());
    }
    Ok(())
}

/// Plan the index read of type `ty` of the indexed capture form, the de-indexed form or the
/// private list form once ([`super::CapturedIndex`]): its address push field, the capture and
/// de-indexed forms' layout, the private list form's first vertex (as the primitive ID's,
/// [`first_vertex`]) and its fact, and the vertex index builtin. Refused ([`indices_feature`]) on
/// a device without device addresses.
pub(super) fn captured_index(
    cx: &mut SelectCx<'_>,
    ty: crate::metal::IndexType,
) -> Result<(), Error> {
    if cx.families.stage_io.captured_index.is_some() {
        return Ok(());
    }
    refuse_unaddressed(cx)?;
    cx.push.insert(PushField::CaptureIndexAddress);
    let first = match listed(cx) {
        true => {
            cx.facts.insert(RuntimeFact::PrivateList);
            Some(first_vertex(cx))
        }
        false => {
            cx.push.insert(PushField::CaptureLayout);
            None
        }
    };
    let sp = &mut cx.families.stage_io;
    let vertex = builtin(
        sp,
        B::VertexIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::VertexIndex,
    );
    sp.captured_index = Some(super::CapturedIndex { vertex, ty, first });
    sp.words = sp.words.saturating_add(super::CAPTURED_INDEX_WORDS);
    Ok(())
}

/// Plan builtin parameter `b` of AIR type `ty`, a scalar integer, of a vertex function in the
/// indexed capture form of index type `index` (`None` for `[[base_vertex]]`): `[[vertex_id]]` is
/// the vertex its index names ([`Source::CapturedVertex`]), `[[base_vertex]]` the layout's base
/// vertex ([`Source::CapturedBaseVertex`]).
fn captured_param(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    ty: AirTypeId,
    index: Option<crate::metal::IndexType>,
) -> Result<(), Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(f)?;
    let source = match index {
        Some(index) => {
            captured_index(cx, index)?;
            Source::CapturedVertex(f)
        }
        // The private list form's base vertex lies before its list, found from its first vertex.
        None if listed(cx) => {
            captured_index(cx, crate::metal::IndexType::UInt32)?;
            Source::CapturedBaseVertex(f)
        }
        None => {
            cx.push.insert(PushField::CaptureLayout);
            Source::CapturedBaseVertex(f)
        }
    };
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source,
    });
    Ok(())
}

/// Plan a merged draw's vertex function (`LoweringInputs::merged_draws`, [`MergedRead`]): its
/// index builtins, the header's address, and its fact. Refused
/// (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`): one compiled without
/// `replay_bindings`, or under an input that reads the draw's own vertex or instance indices or
/// draws it more than once (the primitive ID's varying or de-indexed form, the private list form,
/// the capture form, amplifications drawn apart, tessellation).
fn merged(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let i = cx.inputs;
    if !i.replay_bindings
        || i.primitive_id_varying
        || i.primitive_id_indices.is_some()
        || i.private_list
        || i.captured_positions.is_some()
        || i.amplification_draws.is_some()
        || i.tessellation.is_some()
    {
        return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
    }
    let sp = &mut cx.families.stage_io;
    let vertex = builtin(
        sp,
        B::VertexIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::VertexIndex,
    );
    let instance = builtin(
        sp,
        B::InstanceIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::InstanceIndex,
    );
    sp.merged = Some(MergedRead { vertex, instance });
    sp.words = sp.words.saturating_add(super::MERGED_WORDS);
    cx.push.insert(PushField::MergedDrawAddress);
    cx.facts.insert(RuntimeFact::MergedDraws);
    Ok(())
}

/// Plan builtin parameter `b` of AIR type `ty`, a scalar integer, of a merged draw's vertex
/// function: the `value` of its corner's command ([`Source::Merged`]).
fn merged_param(
    cx: &mut SelectCx<'_>,
    b: &ParamBinding,
    ty: AirTypeId,
    value: MergedValue,
) -> Result<(), Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let form = form_on(cx.entry, cx.profile, ty)?;
    int_scalar(form)?;
    cx.families.stage_io.params.push(PlannedParam {
        param: b.param,
        source: Source::Merged { value, form },
    });
    Ok(())
}

/// The push fields of a function compiled under `LoweringInputs::merged_draws`, once every family
/// planned its own: a vertex function reads its command's slot table from the header
/// ([`MergedRead`]), so `RootTableAddress` is no push field of it; a fragment function reads no
/// command's buffer, which a merged draw does not keep per command, and one that does is
/// `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`.
pub(crate) fn finish_merged(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if !cx.inputs.merged_draws {
        return Ok(());
    }
    match cx.stage {
        Stage::Vertex => {
            cx.push.remove(&PushField::RootTableAddress);
        }
        Stage::Fragment if cx.push.contains(&PushField::RootTableAddress) => {
            return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
        }
        _ => {}
    }
    Ok(())
}

/// Plan a vertex function's viewport compensation `c` (`LoweringInputs::viewport_compensation`):
/// its `Position` is stored compensated by `PushField::ViewportCompensation`, and under
/// [`ViewportCompensation::depth`] two `ClipDistance` elements follow the function's own, which
/// Metal's near and far planes clip at. Under [`ViewportCompensation::point_guard`] the position
/// is scaled by `PushField::PointGuard` too, and four more elements follow, Metal's guard band
/// for points. Under [`ViewportCompensation::vertex_clamp`] the compensated depth is clamped by
/// `PushField::VertexDepthClamp` before the interpolation. A function that writes no position
/// rasterizes nothing and is compensated in nothing. Refused: the clip distances on a device
/// without `shaderClipDistance`, or past its `maxClipDistances` or
/// `maxCombinedClipAndCullDistances`.
fn compensate(cx: &mut SelectCx<'_>, c: ViewportCompensation) -> Result<(), Error> {
    cx.facts.insert(RuntimeFact::ViewportCompensation {
        depth: c.depth,
        point_guard: c.point_guard,
        mirrors: c.mirrors,
        vertex_clamp: c.vertex_clamp,
    });
    let is = |o: &IoObject, b| matches!(o, IoObject::Builtin { builtin, dir: IoDir::Output, .. } if *builtin == b);
    let sp = &cx.families.stage_io;
    let Some(position) = sp.objects.iter().position(|o| is(o, B::Position)) else {
        return Ok(());
    };
    cx.push.insert(PushField::ViewportCompensation);
    if c.point_guard {
        cx.push.insert(PushField::PointGuard);
    }
    if c.mirrors {
        cx.push.insert(PushField::ViewportMirrors);
    }
    // Each vertex's depth clamped, which only a compensated depth takes.
    if c.vertex_clamp && c.depth {
        cx.push.insert(PushField::VertexDepthClamp);
    }
    let added = 2 * u32::from(c.depth) + 4 * u32::from(c.point_guard);
    let clips = match added {
        0 => None,
        _ => {
            if !cx.profile.fields().fragment.clip_distance {
                return Err(needs(Feature::ClipDistance, VkFeature::ShaderClipDistance));
            }
            let limits = cx.profile.fields().limits;
            let sp = &mut cx.families.stage_io;
            let own = sp.objects.iter().position(|o| is(o, B::ClipDistance));
            let first = match own.and_then(|i| sp.objects.get(i)) {
                Some(IoObject::Builtin {
                    shape: Shape::F32Array(n),
                    ..
                }) => *n,
                Some(_) => return Err(not_lowered()),
                None => 0,
            };
            let count = first + added;
            for (lim, max) in [
                (VkLimit::MaxClipDistances, limits.max_clip_distances),
                (
                    VkLimit::MaxCombinedClipAndCullDistances,
                    limits.max_combined_clip_and_cull_distances,
                ),
            ] {
                if count > max {
                    return Err(limit(Feature::ClipDistance, lim, u64::from(count), max));
                }
            }
            let object = match own {
                // The function's own clip distances, two elements longer.
                Some(i) => {
                    if let Some(IoObject::Builtin { shape, .. }) = sp.objects.get_mut(i) {
                        *shape = Shape::F32Array(count);
                    }
                    let builtins = &mut sp.interface.builtins;
                    builtins.retain(|b| !matches!(b, StageBuiltin::ClipDistance { .. }));
                    builtins.push(StageBuiltin::ClipDistance { count });
                    builtins.sort();
                    i
                }
                None => builtin(
                    sp,
                    B::ClipDistance,
                    IoDir::Output,
                    Shape::F32Array(count),
                    StageBuiltin::ClipDistance { count },
                ),
            };
            Some((object, first))
        }
    };
    let clip = clips.filter(|_| c.depth);
    let guard = clips
        .filter(|_| c.point_guard)
        .map(|(object, first)| (object, first + 2 * u32::from(c.depth)));
    cx.families.stage_io.compensation = Some(Compensation {
        position,
        clip,
        guard,
        mirrors: c.mirrors,
        vertex_clamp: c.vertex_clamp && c.depth,
    });
    Ok(())
}

/// Plan one output of AIR type `ty`.
fn output(cx: &mut SelectCx<'_>, o: &OutputBinding, ty: AirTypeId) -> Result<Link, Error> {
    // A shared output (`air.shared`) has the same value for every amplification; each view
    // computes it again, so it is stored like any other.
    let vertex = cx.stage == Stage::Vertex;
    let frag = cx.profile.fields().fragment;
    let limits = cx.profile.fields().limits;
    let out = |cx: &mut SelectCx<'_>, b, shape, place, form, contract| -> Link {
        let object = builtin(&mut cx.families.stage_io, b, IoDir::Output, shape, contract);
        Link {
            object,
            place,
            form,
        }
    };
    Ok(match &o.role {
        // Decorated `Invariant` whether or not Metal's position is `[[invariant]]`: a draw's
        // captured positions (the tile reach's capture variant) are then its drawn ones.
        OutputRole::Position { .. } if vertex => {
            let f = io_form(cx, ty)?;
            float_lanes(f, 4)?;
            let link = out(
                cx,
                B::Position,
                Shape::FVec(4),
                Place::Whole,
                f,
                StageBuiltin::Position { invariant: true },
            );
            cx.families.stage_io.invariant = Some(link.object);
            link
        }
        OutputRole::VertexOutput(k) if vertex => {
            let f = varying_form(cx, ty)?;
            let key = VaryingKey::user(&k.0);
            let sp = &mut cx.families.stage_io;
            let deco = IoDecor {
                packed: true,
                ..IoDecor::default()
            };
            let object = user(sp, key.clone(), IoDir::Output, io_type(f), deco)?;
            push_var(
                &mut sp.interface.outputs,
                IoVariable {
                    key,
                    role: IoRole::Varying {
                        user: AirUserKey(k.0.clone()),
                        interpolation: None,
                    },
                    ty: io_type(f),
                },
            );
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        // `entry` drops a post-tessellation vertex function's point size before it plans an
        // output: an evaluation shader would write `PointSize` only with the tessellation
        // point-size feature, for triangles it never sizes.
        OutputRole::PointSize
            if matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) }) =>
        {
            return Err(cx.invariant("a post-tessellation point size reached output planning"))
        }
        OutputRole::PointSize if vertex => {
            let f = io_form(cx, ty)?;
            float_lanes(f, 1)?;
            note_device_point_size(cx);
            out(
                cx,
                B::PointSize,
                Shape::F32,
                Place::Whole,
                f,
                StageBuiltin::PointSize { default: false },
            )
        }
        OutputRole::ClipDistance { array_size } if vertex => {
            if !frag.clip_distance {
                return Err(needs(Feature::ClipDistance, VkFeature::ShaderClipDistance));
            }
            let types = cx.entry.module.types();
            let (elem, count, place) = match types.get(ty) {
                Some(Type::Array { elem, len }) => {
                    let n = u32::try_from(*len).map_err(|_| not_lowered())?;
                    (*elem, n, Place::Spread(n))
                }
                _ => (ty, 1, Place::Element0),
            };
            if array_size.is_some_and(|a| a != count) || count == 0 {
                return Err(cx.invariant("a clip distance array of another size"));
            }
            if count > limits.max_clip_distances {
                return Err(limit(
                    Feature::ClipDistance,
                    VkLimit::MaxClipDistances,
                    u64::from(count),
                    limits.max_clip_distances,
                ));
            }
            let f = io_form(cx, elem)?;
            float_lanes(f, 1)?;
            out(
                cx,
                B::ClipDistance,
                Shape::F32Array(count),
                place,
                f,
                StageBuiltin::ClipDistance { count },
            )
        }
        OutputRole::RenderTargetArrayIndex | OutputRole::ViewportArrayIndex if vertex => {
            let (index, spirv, contract) = match o.role {
                OutputRole::RenderTargetArrayIndex => {
                    (RawIndex::Layer, B::Layer, StageBuiltin::Layer)
                }
                _ => (
                    RawIndex::Viewport,
                    B::ViewportIndex,
                    StageBuiltin::ViewportIndex,
                ),
            };
            let uniform = uniform_index(cx, index);
            let store = index_store(cx, index, uniform, false)?;
            let f = io_form(cx, ty)?;
            int_scalar(f)?;
            match store {
                IndexStore::Builtin => {
                    // Differing among a primitive's vertices, stored to the builtin only on a
                    // device that takes the first vertex's as Metal does.
                    cx.families.stage_io.first_vertex_indices_by_device |= !uniform;
                    out(cx, spirv, Shape::U32, Place::Whole, f, contract)
                }
                IndexStore::Selected => Link {
                    object: selected_index(&mut cx.families.stage_io, index)?,
                    place: Place::Whole,
                    form: f,
                },
                IndexStore::Nowhere => {
                    return Err(cx.invariant("a viewport index stored nowhere as an output"))
                }
            }
        }
        // A raster order group on the output is the fragment family's: it runs the whole body,
        // these stores included, inside the interlock.
        OutputRole::RenderTarget {
            attachment, index, ..
        } if !vertex => {
            let AbiInt::Const(attachment) = *attachment else {
                return Err(not_lowered());
            };
            let index = *index;
            if index > 1 {
                return Err(cx.invariant("a dual-source index above 1"));
            }
            if index == 1 {
                if !frag.dual_src_blend {
                    return Err(needs(Feature::DualSourceBlending, VkFeature::DualSrcBlend));
                }
                if attachment >= limits.max_fragment_dual_src_attachments {
                    return Err(limit(
                        Feature::DualSourceBlending,
                        VkLimit::MaxFragmentDualSrcAttachments,
                        u64::from(attachment) + 1,
                        limits.max_fragment_dual_src_attachments,
                    ));
                }
            }
            if attachment >= limits.max_fragment_output_attachments {
                return Err(limit(
                    Feature::InterfaceComponents,
                    VkLimit::MaxFragmentOutputAttachments,
                    u64::from(attachment) + 1,
                    limits.max_fragment_output_attachments,
                ));
            }
            let f = io_form(cx, ty)?;
            let io = color_type(cx, attachment, f)?;
            // Written as a half's bits through 16-bit lanes (`color_write::half_output`).
            let io = match index == 0 && super::color_write::half_output(cx, attachment, f) {
                true => IoType { bits: 16, ..io },
                false => io,
            };
            // An extended-range attachment's raw texel, through a variable of its raw components,
            // and a word attachment's word; the interface keeps the Metal output's type.
            let declared = match super::word_color::plan(cx, attachment, index, f)? {
                Some(word) => word,
                None => super::extended_color::plan(cx, attachment, index, f)?.unwrap_or(io),
            };
            let key = VaryingKey::color(attachment, index);
            let deco = IoDecor {
                index: (index == 1).then_some(1),
                ..IoDecor::default()
            };
            let sp = &mut cx.families.stage_io;
            let object = user(sp, key.clone(), IoDir::Output, declared, deco)?;
            push_var(
                &mut sp.interface.outputs,
                IoVariable {
                    key,
                    role: IoRole::Color { attachment, index },
                    ty: io,
                },
            );
            cx.descriptors.color_attachments = cx.descriptors.color_attachments.saturating_add(1);
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        OutputRole::Depth(q) if !vertex => {
            let f = io_form(cx, ty)?;
            float_lanes(f, 1)?;
            let (mode, extra) = match q {
                DepthQualifier::Any => (DepthMode::Any, None),
                DepthQualifier::Greater => (DepthMode::Greater, Some(ExecMode::DepthGreater)),
                DepthQualifier::Less => (DepthMode::Less, Some(ExecMode::DepthLess)),
            };
            // A depth rounded to Metal's code can land below the fragment's own, or above it, so
            // the function promises neither.
            let codes = depth16_codes(cx);
            let sp = &mut cx.families.stage_io;
            sp.modes.push(ExecMode::DepthReplacing);
            sp.modes.extend(extra.filter(|_| !codes));
            let link = out(
                cx,
                B::FragDepth,
                Shape::F32,
                Place::Whole,
                f,
                StageBuiltin::FragDepth(mode),
            );
            let range = cx.inputs.depth_clamp_range;
            if range {
                cx.facts.insert(RuntimeFact::DepthClampRange);
                cx.push.insert(PushField::DepthClampRange);
            }
            if codes {
                cx.facts.insert(RuntimeFact::Depth16Codes { biased: false });
            }
            cx.families.stage_io.depth_clamp = Some(DepthClamp {
                object: link.object,
                clamps: clamps_fragment_depth(cx),
                range,
                codes,
            });
            cx.families.fragment.depth_output = true;
            link
        }
        // A device without stencil export runs the shader-side stencil (`entry`).
        OutputRole::Stencil if !vertex && frag.stencil_export => {
            let f = io_form(cx, ty)?;
            int_scalar(f)?;
            cx.families
                .stage_io
                .modes
                .push(ExecMode::StencilRefReplacing);
            out(
                cx,
                B::FragStencilRefEXT,
                Shape::U32,
                Place::Whole,
                f,
                StageBuiltin::StencilRef,
            )
        }
        OutputRole::SampleMask if !vertex => {
            let f = io_form(cx, ty)?;
            int_scalar(f)?;
            out(
                cx,
                B::SampleMask,
                Shape::U32Array(1),
                Place::Element0,
                f,
                StageBuiltin::SampleMaskOut,
            )
        }
        OutputRole::ImageblockData(_) => {
            return Err(cx.invariant("an imageblock output planned as an interface object"))
        }
        _ => return Err(cx.invariant("a stage output of another stage")),
    })
}

/// Words per interface object, per parameter use, and per stored output element.
const OBJECT_WORDS: u64 = 32;
const USE_WORDS: u64 = 24;
const STORE_WORDS: u64 = 16;
/// The words a use of a [`Source::PixelCenter`] parameter adds to a load: `[[position]]`'s floor,
/// add, shuffle, and their constants; a varying's `InterpolateAtOffset` and its offset.
const PIXEL_WORDS: u64 = 24;
/// The words a use of a [`Source::Uniform`] parameter adds to a load: Metal's reading of an
/// interpolated constant, nine integer instructions on the bits and their constants.
const UNIFORM_WORDS: u64 = 64;
/// The words a use of a [`Source::QuadPrimitive`] parameter adds to a load: the shift and its
/// constant.
const QUAD_PRIMITIVE_WORDS: u64 = 8;
/// The words a use of a [`Source::WidenedAttribute`] parameter adds to a load: the NaN test on the
/// bits, the select, and their constants.
const CANONICAL_NAN_WORDS: u64 = 32;

/// The per-stage interface limits and the size estimate.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage == Stage::Kernel {
        // A kernel's stage inputs: the region it depends on, and their size.
        if cx.families.stage_io.region {
            cx.facts.insert(RuntimeFact::StageInRegion);
        }
        cx.families.stage_io.words = param_words(cx);
        return Ok(());
    }
    // Buffer writes and atomics in a vertex or fragment function need the stage's store feature
    // (the requirement derivation adds it for every such store; refuse it here with its name).
    // Vulkan's stage store features name no task or mesh stage, and the emulation runs object
    // and mesh functions as compute entries: their stores need none.
    let leaf = &cx.profile.fields().leaf;
    let store = match cx.stage {
        Stage::Vertex => Some((
            leaf.vertex_stores_and_atomics,
            VkFeature::VertexPipelineStoresAndAtomics,
        )),
        Stage::Fragment => Some((
            leaf.fragment_stores_and_atomics,
            VkFeature::FragmentStoresAndAtomics,
        )),
        Stage::Kernel | Stage::Object | Stage::Mesh => None,
    };
    let writes = cx.memory.slots.iter().any(|s| s.access.write || s.atomic);
    if let Some((false, feature)) = store.filter(|_| writes) {
        return Err(needs(Feature::StageStoresAndAtomics, feature));
    }
    // The primitive ID's de-indexed form runs the vertex function once per index of the draw,
    // where Metal shades each vertex the indices name as its GPU shades it: a function that
    // writes memory would write once per index. The private list form runs it once per list
    // position, as Metal runs its private types.
    let deindexed = cx.stage == Stage::Vertex
        && cx.inputs.captured_positions.is_none()
        && captured_index_type(cx).is_some()
        && !listed(cx);
    if deindexed && writes {
        return Err(Unsupported::not_implemented(Feature::PrimitiveId).into());
    }
    let l = cx.profile.fields().limits;
    let io = &cx.families.stage_io.interface;
    // Link numbers attributes densely and packs varyings into four-component locations when
    // needed: a stage whose varyings exceed the component limits can never link.
    let ins = io.inputs.len() as u64;
    let components = |vars: &[crate::pipeline::stage_io::IoVariable]| -> u64 {
        vars.iter()
            .filter(|v| v.role.is_varying())
            .map(|v| u64::from(v.ty.components))
            .sum()
    };
    // An emulated post-tessellation vertex function is a plain vertex function.
    let evaluation = matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) })
        && !crate::lower::tessellation::emulates(cx.profile);
    let checks = match cx.stage {
        // A post-tessellation vertex function's outputs are an evaluation shader's; it has no
        // input variable of this family's.
        Stage::Vertex if evaluation => [
            (
                ins,
                VkLimit::MaxVertexInputAttributes,
                l.max_vertex_input_attributes,
            ),
            (
                components(&io.outputs),
                VkLimit::MaxTessellationEvaluationOutputComponents,
                l.max_tessellation_evaluation_output_components,
            ),
        ],
        Stage::Vertex => [
            (
                ins,
                VkLimit::MaxVertexInputAttributes,
                l.max_vertex_input_attributes,
            ),
            (
                components(&io.outputs),
                VkLimit::MaxVertexOutputComponents,
                l.max_vertex_output_components,
            ),
        ],
        _ => [
            (
                components(&io.inputs),
                VkLimit::MaxFragmentInputComponents,
                l.max_fragment_input_components,
            ),
            (0, VkLimit::MaxFragmentOutputAttachments, 0),
        ],
    };
    for (n, lim, max) in checks {
        if n > u64::from(max) {
            return Err(limit(Feature::InterfaceComponents, lim, n, max));
        }
    }
    // The render pipeline's private alpha test, which each return runs on color 0: Metal's
    // compiler fails a function that writes no color 0 under the test.
    if let (Stage::Fragment, Some(function)) = (cx.stage, cx.inputs.alpha_test) {
        if cx.families.stage_io.alpha_test.is_none() {
            return Err(alpha_test_refused());
        }
        if function != crate::metal::CompareFunction::Never {
            cx.push.insert(PushField::AlphaTestReference);
        }
        cx.families.stage_io.interface.alpha_test = Some(function);
    }
    // The render pipeline's private masks a fragment function applies itself: its `SampleMask`
    // output, the function's own where it writes one, which each return ANDs the pixel's mask
    // into or stores it as; and the `FragCoord` input whose pixel picks the mask, where the masks
    // differ between the places of a quad.
    if let (Stage::Fragment, Some(masks)) = (cx.stage, cx.inputs.sample_mask) {
        let sp = &mut cx.families.stage_io;
        let object = builtin(
            sp,
            B::SampleMask,
            IoDir::Output,
            Shape::U32Array(1),
            StageBuiltin::SampleMaskOut,
        );
        let coord = (!crate::pipeline::private_state::uniform(&masks))
            .then(|| builtin_object(sp, B::FragCoord, IoDir::Input, Shape::FVec(4)));
        sp.private_mask = Some(super::PrivateMask {
            object,
            masks,
            coord,
        });
        sp.interface.sample_mask = Some(masks);
    }
    // A fragment function that returns no depth under `LoweringInputs::depth16_codes` stores the
    // depth Metal stores for the fragment at entry, its Vulkan bias recorded as zero: objects of
    // its own, not the function's interface, which writes no depth. One that reads its position's
    // depth keeps the device's conversion: whether Metal's position takes the draw's bias is
    // unmeasured, and without the Vulkan bias it would not. Under a bias of known side
    // (`LoweringInputs::depth16_bias_raises`) it declares the side (`depth16::bias_side_mode`).
    if cx.stage == Stage::Fragment
        && depth16_codes(cx)
        && !cx.families.fragment.depth_output
        && !cx.families.stage_io.interface.position_depth
    {
        let raises = cx.inputs.depth16_bias_raises;
        let sp = &mut cx.families.stage_io;
        let depth = builtin_object(sp, B::FragDepth, IoDir::Output, Shape::F32);
        let coord = builtin_object(sp, B::FragCoord, IoDir::Input, Shape::FVec(4));
        sp.modes.push(ExecMode::DepthReplacing);
        sp.modes.extend(raises.map(super::depth16::bias_side_mode));
        sp.depth16 = Some(super::Depth16Store {
            depth,
            coord,
            raises,
        });
        cx.facts.insert(RuntimeFact::Depth16Codes { biased: true });
        cx.push.insert(PushField::DepthBias);
    }
    // Size: every object, the parameters' words, and a store per output element; a count's
    // compare and select (and its constant 0) are a store's words more, and so is an
    // amplification's offset added; an amplified output is a store, an add and a count's.
    let params = param_words(cx);
    let sp = &mut cx.families.stage_io;
    let stores: u64 = sp
        .outputs
        .iter()
        .map(|o| {
            let elements = match o.link.place {
                Place::Spread(n) => u64::from(n),
                _ => 1,
            };
            elements
                + super::POINT_SIZE_WORDS * u64::from(super::point_size_object(&sp.objects, o.link.object))
                + u64::from(o.below.is_some())
                + u64::from(o.offset.is_some())
                + u64::from(o.rotated)
                // Metal's blue premultiplied: two extracts, each lane's NaN test, select and
                // clamp, the product (an `Fma` with `-0.0` where the builder writes one so), its
                // insert, and their constants.
                + 12 * u64::from(o.premultiplied)
        })
        .sum::<u64>()
        // A rotation's second output: its field, two shuffles and a store.
        + 4 * sp.rotated_seconds.len() as u64
        + 3 * sp.amplified.len() as u64
        + u64::from(sp.point_size_default.is_some())
        // The private sample mask's AND and store.
        + 2 * u64::from(sp.private_mask.is_some())
        // The alpha test: the alpha's extracts, its clamp and code, the reference, the
        // comparison and the kill.
        + 40 * u64::from(sp.alpha_test.is_some())
        + compensation_stores(sp.compensation)
        + sp.depth_clamp.map_or(0, |d| match (d.range, d.clamps || d.codes) {
            // The `[0, 1]` selects, the range's two clamps and the negative NaN's select.
            (true, _) => 14,
            (false, true) => 5,
            (false, false) => 2,
        });
    // Each return stores every output.
    let rets = cx
        .body
        .insts
        .values()
        .filter(|i| matches!(i.kind, crate::air::ir::InstKind::Ret(_)))
        .count() as u64;
    // Each return also rounds every color output that rounds, writes the halves' bits, and encodes
    // the extended-range colors' texels.
    let rounding = super::color_write::words(sp.rounded.values().copied())
        .saturating_add(
            crate::lower::arith::color_write::HALF_BITS_WORDS
                .saturating_mul(sp.halves.len() as u64),
        )
        .saturating_add(super::extended_color::WORDS.saturating_mul(sp.extended.len() as u64))
        .saturating_add(
            (sp.words_written.values())
                .map(|&w| super::word_color::encode_words(w))
                .fold(0u64, u64::saturating_add),
        )
        // A returned depth's code.
        .saturating_add(match sp.depth_clamp.is_some_and(|d| d.codes) {
            true => super::depth16::CODE_WORDS,
            false => 0,
        });
    sp.words = (sp.objects.len() as u64)
        .saturating_mul(OBJECT_WORDS)
        .saturating_add(params)
        .saturating_add(
            stores
                .saturating_add(1)
                .saturating_mul(rets)
                .saturating_mul(STORE_WORDS),
        )
        .saturating_add(rounding.saturating_mul(rets))
        // The depth stored at entry.
        .saturating_add(match sp.depth16 {
            Some(d) if d.raises.is_some() => {
                super::depth16::STORE_WORDS + super::depth16::SIDE_WORDS
            }
            Some(_) => super::depth16::STORE_WORDS,
            None => 0,
        });
    Ok(())
}

/// Whether a fragment function stores Metal's `Depth16Unorm` codes itself
/// (`LoweringInputs::depth16_codes`): compiled under the input, it neither declares nor takes
/// early fragment tests (the storage form's, `fetch_early_tests`) nor runs its draw's tests itself,
/// where the device's tests take its depth before it runs.
fn depth16_codes(cx: &SelectCx<'_>) -> bool {
    let early = matches!(
        cx.entry.ep.stage,
        AirStage::Fragment {
            early_fragment_tests: true
        }
    );
    cx.stage == Stage::Fragment
        && cx.inputs.depth16_codes
        && !early
        && !cx.families.fragment.fetch_early_tests
        && !cx.inputs.shader_tests
}

/// Whether AIR type `ty` of entry `e` is `half` or a vector of it: a `half` color output
/// (`PrecisionNote::HalfOutputPrecision`, or `HalfOutputFastMathBlend` under fast math).
fn half_color(e: &crate::air::Entry, ty: AirTypeId) -> bool {
    let types = e.module.types();
    let scalar = match types.get(ty) {
        Some(Type::Vector { elem, .. }) => types.get(*elem),
        other => other,
    };
    matches!(scalar, Some(Type::Half))
}

/// The refusal of a private alpha test the translation has not built (`PrivateRenderField`'s
/// `AlphaTest`): a fragment function under the test that writes no color 0, or one that is not
/// four `float` or `half` lanes.
fn alpha_test_refused() -> Error {
    Error::Unsupported(Unsupported::not_implemented(Feature::PrivateRenderState(
        crate::error::PrivateRenderField::AlphaTest,
    )))
}

/// Whether `ty` is a `float` color output: a scalar or vector of AIR's `float`.
fn float_color(e: &crate::air::Entry, ty: AirTypeId) -> bool {
    let types = e.module.types();
    let scalar = match types.get(ty) {
        Some(Type::Vector { elem, .. }) => types.get(*elem),
        other => other,
    };
    matches!(scalar, Some(Type::Float))
}

/// The stores' worth of words a compensated position adds to each return: the field's lanes and
/// `x`'s product, under the depth's compensation its sum of products and the two clip distances'
/// selects and stores, and under the point guard band its two lanes and products, the band, and the
/// four clip distances' sums and stores.
fn compensation_stores(c: Option<Compensation>) -> u64 {
    let Some(c) = c else { return 0 };
    let depth = if c.clip.is_some() { 9 } else { 0 };
    let guard = if c.guard.is_some() { 13 } else { 0 };
    3 + depth + guard
}

/// Whether a fragment function's depth output is clamped to `[0, 1]` as Metal stores it
/// (`DepthClamp::clamps`; five stores' words with a positive NaN's kill, two for the kill
/// alone): where the profile enables `VK_EXT_depth_range_unrestricted`, under which Vulkan no
/// longer clamps it (RADV, NVIDIA and x86 lavapipe store 1.5 and -0.5 unclamped, measured).
/// Without the extension Vulkan clamps it to `[0, 1]` itself.
fn clamps_fragment_depth(cx: &SelectCx<'_>) -> bool {
    cx.profile
        .fields()
        .enabled
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
}

/// The words of the used stage-IO parameters: a load per use of an interface parameter (or of a
/// kernel's stage-in region), with its pixel-center form for a [`Source::PixelCenter`] one, and one
/// fetch per pulled attribute at entry (`fetch::PulledRead::words`), whose uses read the held
/// value.
fn param_words(cx: &SelectCx<'_>) -> u64 {
    let mut fetch_words = 0u64;
    let mut loaded: Vec<(u32, u64)> = Vec::new();
    for p in &cx.families.stage_io.params {
        match &p.source {
            Source::Io(_)
            | Source::AmplificationCount(_)
            | Source::Unamplified(_)
            | Source::Absent(_)
            | Source::Pushed { .. }
            | Source::DrawBase { .. }
            | Source::StageInRegion { .. }
            | Source::CapturedVertex(_)
            | Source::CapturedBaseVertex(_)
            | Source::Merged { .. } => loaded.push((p.param, USE_WORDS)),
            Source::QuadPrimitive(_) => loaded.push((p.param, USE_WORDS + QUAD_PRIMITIVE_WORDS)),
            Source::PixelCenter(_) | Source::LoopSample(_) | Source::AtLoopSample(_) => {
                loaded.push((p.param, USE_WORDS + PIXEL_WORDS))
            }
            Source::Uniform(_) => loaded.push((p.param, USE_WORDS + UNIFORM_WORDS)),
            Source::WidenedAttribute(_) => loaded.push((p.param, USE_WORDS + CANONICAL_NAN_WORDS)),
            Source::Pulled(r) => fetch_words = fetch_words.saturating_add(u64::from(r.words())),
        }
    }
    let mut use_words = 0u64;
    for inst in cx.body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                if let Some(&(_, words)) = loaded.iter().find(|(p, _)| *p == a.raw()) {
                    use_words = use_words.saturating_add(words);
                }
            }
        });
    }
    fetch_words.saturating_add(use_words)
}
