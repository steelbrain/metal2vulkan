//! Selection of the stage interface: which Vulkan object each vertex or fragment input and output
//! becomes, checked against the profile.

use std::collections::BTreeSet;

use crate::air::abi::{
    AbiInt, Builtin, DepthQualifier, Interpolation as AirInterp, OutputBinding, OutputRole,
    ParamBinding, ParamRole, PatchInput, Sampling as AirSampling, Stage as AirStage, Varying,
};
use crate::air::intrinsics::{AirIntrinsic, Fragment as AirFragment, InterpolateAt};
use crate::air::ir::{ArgIdx, Callee, InstKind, Type, TypeId as AirTypeId, Value};
use crate::device::requirements::{VkExtension, VkFeature, VkLimit, VkProperty};
use crate::error::{Attachment, Capability, Error, Feature, Unsupported};
use crate::lower::arith::legal::{form_on, refuse_bfloat_color, refuse_int64_interface};
use crate::lower::arith::{Form, Held};
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{
    fetch, not_lowered, AmplifiedOutput, DepthClamp, IoObject, MergedRead, MergedValue,
    PlannedOutput, PlannedParam, Returned, RotatedSecond, Source, StageIoPlan,
};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::{FactorLane, VertexFetch};
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
    /// (`BaryCoordKHR`): `1 - (x + y)` of the first
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
                // Planned by `interpolant`, never as a varying read by value.
                AirSampling::Interpolant => {
                    return Err(cx.invariant("an interpolant planned as a varying"))
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
    cx.profile.interpolation_function_needs().is_empty()
        && cx
            .profile
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
}

/// Plan interpolant parameter `b` (Metal's pull-model interpolation): a float varying of the
/// interpolant's form and perspective, declared without `Centroid` or `Sample`, which only its
/// interpolation functions read (`lower::fragment`'s `Interpolate`), each where it says.
/// `interpolate_at_center` reads it as a center-sampled varying is read: under
/// `LoweringInputs::multisampled` through its pixel-center form where the device has one
/// (`InterpolateAtOffset` at offset 0, [`Source::PixelCenter`]'s, or the read as it is on a device
/// that reads such a varying at the pixel center itself), else as it is. The other functions are
/// `GLSL.std.450`'s `InterpolateAtCentroid`, `InterpolateAtSample` and `InterpolateAtOffset`,
/// which the `InterpolationFunction` capability declares, enabled by `sampleRateShading` (and on
/// a portability-subset device `shaderSampleRateInterpolationFunctions`): without them the
/// function needs them. An interpolant no vertex output writes is zero at every read
/// ([`Source::Absent`]). One used any other way ([`interpolant_reads`]), or one its mesh function
/// writes per primitive, is not lowered (`Feature::PullModelInterpolation`).
fn interpolant(cx: &mut SelectCx<'_>, b: &ParamBinding, v: &Varying) -> Result<(), Error> {
    let AirInterp::Smooth { perspective, .. } = v.interpolation else {
        return Err(cx.invariant("an interpolant without its perspective"));
    };
    let (reads, ty) = interpolant_reads(cx, b.param)?;
    let key = VaryingKey::user(&v.key.0);
    if cx.inputs.absent_varyings.contains(&key) {
        let f = form_on(cx.entry, cx.profile, ty)?;
        cx.families.stage_io.params.push(PlannedParam {
            param: b.param,
            source: Source::Absent(f),
        });
        return Ok(());
    }
    if read_per_primitive(cx, &key) {
        return Err(Unsupported::not_implemented(Feature::PullModelInterpolation).into());
    }
    if reads.iter().any(|at| *at != InterpolateAt::Center) {
        let lacks = cx.profile.interpolation_function_needs();
        if !lacks.is_empty() {
            return Err(
                Unsupported::needs_any_of(Feature::PullModelInterpolation, vec![lacks]).into(),
            );
        }
    }
    let f = varying_form(cx, ty)?;
    if !f.is_float() {
        return Err(cx.invariant("an interpolant of no float type"));
    }
    let mut deco = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    if !perspective {
        deco.interpolation = SInterp::NoPerspective;
    }
    let center = reads.contains(&InterpolateAt::Center);
    let interp = Interpolation::Smooth {
        sampling: Sampling::Interpolant {
            center,
            centroid: reads.contains(&InterpolateAt::Centroid),
            sample: reads.contains(&InterpolateAt::Sample),
        },
        perspective,
    };
    note_device_interpolation(cx, interp);
    if perspective
        && (reads.contains(&InterpolateAt::Sample) || reads.contains(&InterpolateAt::Offset))
    {
        note_extrapolated_read(cx);
    }
    // The center read's pixel form, as a center varying's.
    let at_center = center && cx.inputs.multisampled;
    let at_pixel = at_center && interpolation_functions(cx);
    let sp = &mut cx.families.stage_io;
    sp.interface.varyings_at_pixel |= at_pixel;
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
    sp.params.push(PlannedParam {
        param: b.param,
        source: Source::Interpolant {
            link: Link {
                object,
                place: Place::Whole,
                form: f,
            },
            at_pixel,
        },
    });
    Ok(())
}

/// Where the interpolation functions read interpolant parameter `param`, and the type they
/// return, the interpolant's: every use of the parameter must be the interpolant operand of an
/// `air.interpolate_*` call, all of one result type. Any other use (a `select` or `phi` choosing
/// the interpolant at run time, a store, a call, another operand) is not lowered
/// (`Feature::PullModelInterpolation`). A sample or offset read outside the function's first
/// block, or after a discard there, where some of the quad's invocations may have left the control
/// flow, is not lowered on any device (`Feature::DivergentInterpolation`).
pub(super) fn interpolant_reads(
    cx: &SelectCx<'_>,
    param: u32,
) -> Result<(BTreeSet<InterpolateAt>, AirTypeId), Error> {
    let arg = Value::Arg(ArgIdx::new(param));
    let refused =
        || -> Error { Unsupported::not_implemented(Feature::PullModelInterpolation).into() };
    let first_block_end = cx.body.blocks.values().next().map(|b| b.end);
    let mut discarded = false;
    let mut reads = BTreeSet::new();
    let mut ty = None;
    for (id, inst) in cx.body.insts.iter() {
        let read = match &inst.kind {
            InstKind::Call(call) => match &call.callee {
                Callee::Air {
                    op: AirIntrinsic::Interpolate(i),
                    ..
                } => Some((i, call.args.first() == Some(&arg))),
                Callee::Air {
                    op: AirIntrinsic::Fragment(AirFragment::Discard),
                    ..
                } => {
                    discarded |= first_block_end.is_some_and(|end| id < end);
                    None
                }
                _ => None,
            },
            _ => None,
        };
        let mut uses = 0usize;
        inst.kind.for_each_value(|v| uses += usize::from(v == arg));
        if uses == 0 {
            continue;
        }
        let Some((i, true)) = read else {
            return Err(refused());
        };
        if uses != 1 {
            return Err(refused());
        }
        if matches!(i.at, InterpolateAt::Sample | InterpolateAt::Offset)
            && (discarded || first_block_end.is_none_or(|end| id >= end))
        {
            return Err(Unsupported::not_implemented(Feature::DivergentInterpolation).into());
        }
        if ty.is_some_and(|t| t != inst.ty) {
            return Err(refused());
        }
        ty = Some(inst.ty);
        reads.insert(i.at);
    }
    match ty {
        Some(ty) => Ok((reads, ty)),
        None => Err(cx.invariant("an interpolant planned without a read")),
    }
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
/// position in the pixel.
pub fn param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let ty = param_ty(cx, b)?;
    let frag = cx.profile.fields().fragment;
    let stage = cx.stage;
    let mut at_pixel = false;
    let mut at_loop_sample = false;
    let mut uniform = false;
    let mut quad_primitive = false;
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
            // The de-indexed and private list forms' draws have no indices, so fixed-function
            // input would fetch at the stream position: their projection pulls every attribute,
            // and a hand-built one that does not is refused.
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
            // Fixed-function input: the format is link-time state, and so is what the device's
            // conversion of it may answer otherwise than Metal's fetch, which link notes on the
            // pipeline's vertex stage (`pipeline::link::vertex::fixed_function_notes`). A signed
            // format's variable is declared `int` (`VertexFetch::SignedFixedFunction`), its bits
            // read by `OpBitcast`.
            let mut ty = io_type(f);
            if entry == Some(VertexFetch::SignedFixedFunction) {
                if f.is_float() {
                    return Err(not_lowered());
                }
                ty.numeric = IoNumeric::Sint;
            }
            let key = VaryingKey::attribute(index);
            let sp = &mut cx.families.stage_io;
            let object = user(sp, key.clone(), IoDir::Input, ty, IoDecor::default())?;
            push_var(
                &mut sp.interface.inputs,
                IoVariable {
                    key,
                    role: IoRole::Attribute { index },
                    ty,
                },
            );
            Link {
                object,
                place: Place::Whole,
                form: f,
            }
        }
        (ParamRole::FragmentInput(v), Stage::Fragment) if v.interpolation.is_interpolant() => {
            return interpolant(cx, b, v);
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
            // Each pass reads it through `InterpolateAtSample` at its sample.
            if at_loop_sample
                && matches!(
                    interp,
                    Interpolation::Smooth {
                        perspective: true,
                        ..
                    }
                )
            {
                note_extrapolated_read(cx);
            }
            let center = cx.inputs.multisampled
                && matches!(
                    interp,
                    Interpolation::Smooth {
                        sampling: Sampling::Center,
                        ..
                    }
                );
            // A center varying's pixel form is `InterpolateAtOffset` at offset 0 where that reads
            // from the center.
            at_pixel = center && interpolation_functions(cx);
            let sp = &mut cx.families.stage_io;
            sp.interface.varyings_at_pixel |= at_pixel;
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
        // behind a vertex function compiled under it.
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
        // Drawn a rate-mapped pass's regions at a time, the vertex function's `ViewportIndex` holds
        // its region's draw index, where Metal's, which writes no index, renders through viewport
        // 0: a fragment function reads 0, which Metal's reads there.
        (ParamRole::Builtin(Builtin::ViewportArrayIndex), Stage::Fragment)
            if cx.inputs.region_viewports =>
        {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            int_scalar(f)?;
            cx.families.stage_io.params.push(PlannedParam {
                param: b.param,
                source: Source::Absent(f),
            });
            return Ok(());
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
        // The de-indexed and private list forms draw without indices: the vertex is the one its
        // index names, and its base vertex the layout's (the list's), whatever the profile's draw
        // parameters.
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
        // Without `fragmentShaderBarycentric` the weights have no form: the device's builtin is
        // the one way to read them.
        (ParamRole::Builtin(Builtin::BarycentricCoord(_)), Stage::Fragment)
            if !frag.barycentric =>
        {
            return Err(Unsupported::needs(
                Feature::Barycentrics,
                Capability::Feature(VkFeature::FragmentShaderBarycentric),
            )
            .into());
        }
        // `[[sample_mask, post_depth_coverage]]`: Vulkan's one form, `SampleMask` read under
        // `VK_EXT_post_depth_coverage`, is an extension no runtime enables, and no other form is
        // built.
        (ParamRole::Builtin(Builtin::PostDepthCoverage), _) => {
            return Err(Unsupported::not_implemented(Feature::PostDepthCoverage).into());
        }
        (ParamRole::Builtin(bi), _) => {
            refuse_int64_interface(cx.entry, ty)?;
            let f = form_on(cx.entry, cx.profile, ty)?;
            let (spirv, shape, place, contract, enabler) = input_builtin(cx, *bi, f, frag)?;
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
        source: if at_loop_sample {
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
/// map of the attachment's color writes is recorded as a written attachment's is. What the
/// function writes to `color` itself is stored nowhere: plain Metal stores the zero there all the
/// same (a Metal probe). A function that declares a color input at `color`, read or not, is no
/// pipeline's: plain Metal refuses its pipeline whether or not the body reads the input
/// (`pipeline::compile`'s `color_inputs_attached`), so compiled alone under such inputs it is not
/// lowered.
fn zeroed_output(
    cx: &mut SelectCx<'_>,
    color: u32,
    format: crate::metal::PixelFormat,
) -> Result<(), Error> {
    let reads = cx.entry.ep.params.iter().any(|b| {
        matches!(
            b.role,
            ParamRole::ColorInput {
                index: AbiInt::Const(a),
                ..
            } if a == color
        )
    });
    if reads {
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
/// ([`AmplifiedOutput`]) where [`index_store`] stores it, to its builtin.
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
    let object = builtin(sp, spirv, IoDir::Output, Shape::U32, contract);
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

/// Plan the `ViewportIndex` of a vertex function with outputs `outputs`, returning `returned`,
/// compiled under `LoweringInputs::region_viewports`: its `DrawIndex`, "the zero-based index of
/// the draw" of a `vkCmdDrawMultiEXT` or `vkCmdDrawMultiIndexedEXT` (the specification's
/// `DrawIndex`), and so its region's among the command's ([`super::RegionViewport`]). `DrawIndex`
/// is dynamically uniform, so every vertex of a primitive stores the same index, as Vulkan's
/// `ViewportIndex` asks. A merged draw's vertex function (`merged_draws`) takes it too: its corner
/// is its `VertexIndex` and `InstanceIndex`'s, which a multi-draw's draws each give from their own
/// first vertex. So does a draw replay's of one draw per command (`replay_bindings`), whose
/// buffers are in the slot table at its push: its draws' indices are their own. A function that
/// writes its own viewport or slice index, and one under another field that moves its indices or
/// position (`amplification_draws`), or that runs behind the
/// tessellator, is `NotImplemented(RasterizationRateMap)`: the runtime draws those region by
/// region. A device that cannot store `ViewportIndex` from a vertex function,
/// read `DrawIndex` or take several viewports refuses it, naming the three features. A function
/// without outputs rasterizes nothing and stores no index.
fn region_viewport(
    cx: &mut SelectCx<'_>,
    outputs: &[OutputBinding],
    returned: Returned,
) -> Result<(), Error> {
    let i = &cx.inputs;
    let indexed = outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::ViewportArrayIndex | OutputRole::RenderTargetArrayIndex
        )
    });
    let tessellated = matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) });
    if indexed || tessellated || i.amplification_draws.is_some() {
        return Err(Unsupported::not_implemented(Feature::RasterizationRateMap).into());
    }
    let f = cx.profile.fields();
    let multi_viewport = f.enabled.features.contains(&VkFeature::MultiViewport);
    if !(f.fragment.output_viewport_index && f.draw_parameters && multi_viewport) {
        return Err(Unsupported::needs_any_of(
            Feature::RasterizationRateMap,
            vec![vec![
                Capability::Feature(VkFeature::ShaderOutputViewportIndex),
                Capability::Feature(VkFeature::ShaderDrawParameters),
                Capability::Feature(VkFeature::MultiViewport),
            ]],
        )
        .into());
    }
    cx.facts.insert(RuntimeFact::RegionViewports);
    if returned == Returned::Void {
        return Ok(());
    }
    let sp = &mut cx.families.stage_io;
    let draw_index = builtin(
        sp,
        B::DrawIndex,
        IoDir::Input,
        Shape::U32,
        StageBuiltin::DrawIndex,
    );
    // Not the contract's `ViewportIndex`: the function selects no viewport of the encoder's
    // (`GraphicsState::writes_viewport_index` stays unset), its draw's index selecting its
    // region's.
    let viewport = builtin_object(sp, B::ViewportIndex, IoDir::Output, Shape::U32);
    sp.region_viewport = Some(super::RegionViewport {
        draw_index,
        viewport,
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
/// `PerPrimitiveEXT`.
fn read_per_primitive(cx: &mut SelectCx<'_>, key: &VaryingKey) -> bool {
    if !cx.inputs.primitive_varyings.contains(key) {
        return false;
    }
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
    /// Nowhere: a viewport index on a device without `multiViewport`, whose draws have one
    /// viewport (the runtime refuses more), so Metal's index selects viewport 0 whatever it is,
    /// the viewport Vulkan uses where no stage writes `ViewportIndex`.
    Nowhere,
}

/// Where a vertex function on `cx`'s profile stores index `index`, which it writes the same at
/// every vertex of a primitive where `uniform` ([`uniform_index`]): to its builtin where the
/// profile enables `shaderOutputLayer` (`shaderOutputViewportIndex` for a viewport index) and
/// the function writes the index alike, and nowhere, a viewport index on a profile without
/// `multiViewport`. Otherwise refused, as `VertexAmplification` for an index the function stores
/// from the amplification's offset alone (`amplification`), else as `LayeredRendering`: one
/// written alike needing the output feature, one not shown alike as not implemented on any
/// device.
///
/// An index not shown alike is refused because Metal gives it no answer to match: "You must
/// return the same value of `[[render_target_array_index]]` for every vertex in a primitive. If
/// values differ, the behavior and value passed to the fragment function are undefined." (the
/// Metal Shading Language specification, and the same of `[[viewport_array_index]]`). It is not
/// stored to its builtin as written because Vulkan asks a stage that writes `Layer` to "write the
/// same value to Layer for all vertices of a given primitive"
/// (VUID-VkPipelineShaderStageCreateInfo-stage-02596), and the same of `ViewportIndex`
/// (VUID-VkPipelineShaderStageCreateInfo-stage-02597), which the function's structure does not
/// show of it.
fn index_store(
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
    if native && uniform {
        return Ok(IndexStore::Builtin);
    }
    if matches!(index, RawIndex::Viewport)
        && !f.enabled.features.contains(&VkFeature::MultiViewport)
    {
        return Ok(IndexStore::Nowhere);
    }
    // An index stored from the amplification's offset alone is the same at every vertex.
    if amplification {
        return Err(
            Unsupported::needs(Feature::VertexAmplification, Capability::Feature(output)).into(),
        );
    }
    Err(match uniform {
        true => Unsupported::needs(Feature::LayeredRendering, Capability::Feature(output)),
        false => Unsupported::not_implemented(Feature::LayeredRendering),
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
/// tessellator's, whose primitive IDs are not built: `NotImplemented(Tessellation)`, as for a
/// fragment function that reads `[[primitive_id]]` behind one (`link::tessellation`). A form
/// that reads its draw's indices at their address ([`captured_index_type`]) draws them as
/// `index count` vertices from vertex 0: its first vertex is 0
/// ([`super::FirstVertex::Zero`]). The private list form draws its list from its first vertex,
/// as a draw without indices does.
fn primitive_id_output(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) }) {
        return Err(Unsupported::not_implemented(Feature::Tessellation).into());
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
    Ok(())
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
/// takes the device's answer, which the module records.
fn note_device_interpolation(cx: &mut SelectCx<'_>, interp: Interpolation) {
    if matches!(interp, Interpolation::Smooth { .. }) {
        cx.precision.insert(PrecisionNote::DeviceInterpolation);
    }
}

/// Records that the fragment function reads a perspective input at a sample or an offset other
/// than the pixel center (`InterpolateAtSample`, `InterpolateAtOffset`): a device may answer such
/// a read with the pixel center's value extrapolated along the quad's derivatives (RADV does), its
/// own interpolation, which the module records on every device. The pixel-center form of a center
/// read (`InterpolateAtOffset` at offset 0) moves nothing and records nothing.
fn note_extrapolated_read(cx: &mut SelectCx<'_>) {
    cx.precision
        .insert(PrecisionNote::ExtrapolatedInterpolation);
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
        // A `bool` argument, or an integer one an older front end's ABI passes it in (`i8`,
        // truncated to `i1` where read): the face as 1 or 0 (`to_lowered`).
        Bi::FrontFacing if !vertex => {
            if f.held != Held::Bool || f.lanes != 1 {
                int_scalar(f)?;
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
        Bi::SampleMaskIn if !vertex => {
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
            // `param` refuses the weights without `fragmentShaderBarycentric`.
            if !frag.barycentric {
                return Err(cx.invariant("a barycentric builtin without the feature"));
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
        // A viewport or slice index the vertex function never writes is stored as 0 wherever it
        // is stored.
        let unwritten = cx.stage == Stage::Vertex
            && matches!(
                o.role,
                OutputRole::RenderTargetArrayIndex | OutputRole::ViewportArrayIndex
            )
            && super::returns_undefined(cx.entry, o.field);
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
            // the function rounds it to `half` first (`arith::output_precision`).
            if let OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                ..
            } = o.role
            {
                if half_color(cx.entry, ty) && !unattached.contains(&color) {
                    crate::lower::arith::output_precision::note_half_output(cx, color);
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
            // layers. Drawn one amplification at a time, a pass of one layer writes no `Layer`. In
            // every variant the raw varying holds the index as written, which Metal's fragment
            // function reads.
            let dropped = amplified.is_some_and(|d| !d.layered);
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
                    offset: offset(PushField::AmplificationLayerOffset),
                    unwritten,
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
        // An imageblock output is not implemented.
        if let OutputRole::ImageblockData(_) = &o.role {
            return Err(Unsupported::not_implemented(Feature::Imageblocks).into());
        }
        // A stencil output needs stencil export.
        if matches!(o.role, OutputRole::Stencil)
            && cx.stage == Stage::Fragment
            && !cx.profile.fields().fragment.stencil_export
        {
            return Err(Unsupported::needs(
                Feature::StencilExport,
                Capability::Extension(VkExtension::ExtShaderStencilExport),
            )
            .into());
        }
        // An output to a color the pass lacks is stored nowhere, as plain Metal drops it: no
        // output variable, so the pipeline writes no location without an attachment. So is one to a
        // color the pipeline has no attachment for, and one to a color of the pass the pipeline
        // lacks, where plain Metal stores zero bytes whatever the function writes (a Metal probe):
        // the zero `zeroed_output` writes there is the color's output.
        if let OutputRole::RenderTarget { attachment, .. } = &o.role {
            if !dropped.is_empty() || !unattached.is_empty() || !cx.inputs.zeroed_colors.is_empty()
            {
                let AbiInt::Const(color) = attachment else {
                    return Err(not_lowered());
                };
                if dropped.contains(color) || cx.inputs.zeroed_colors.contains_key(color) {
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
            // A write of an extended-range attachment (the raw texel's through the interface)
            // reads no second output, as Metal's unblended write there reads none: it is stored
            // nowhere, where the interface's raw texel refused it `Format`.
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
                offset: offset(PushField::AmplificationViewportOffset),
                unwritten,
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
                let nan_free = super::returns_no_nan(cx, o.field);
                super::color_write::plan(cx, color, link.form, nan_free);
                Some(color)
            }
            _ => None,
        };
        // An output to an attachment held with its channels rotated stores them rotated; one to an
        // attachment holding Metal's alpha in red holds that alpha in the image's alpha lane too,
        // where alpha to coverage reads it.
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
                // A blend not exact by construction may round apart from Metal's
                // (`LoweringInputs::rotated_blends`).
                if index == 0 && cx.inputs.rotated_blends.contains(&attachment) {
                    cx.precision.insert(PrecisionNote::RotatedBlendRounding);
                    let sp = &mut cx.families.stage_io;
                    sp.interface.rotated_blends.insert(attachment);
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
            offset: added,
            unwritten,
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
                offset: added,
                unwritten,
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
    if cx.inputs.region_viewports && cx.stage == Stage::Vertex {
        region_viewport(cx, &outputs, returned)?;
    }
    // The primitive ID a fragment function reads from its varying.
    if cx.inputs.primitive_id_varying && cx.stage == Stage::Vertex {
        primitive_id_output(cx)?;
    }
    if cx.inputs.indirect_range && cx.stage == Stage::Vertex {
        indirect_range(cx)?;
    }
    // The pass's colors the pipeline lacks, which plain Metal zeroes where a draw covers them.
    if cx.stage == Stage::Fragment {
        let zeroed = cx.inputs.zeroed_colors.clone();
        for (color, format) in zeroed {
            zeroed_output(cx, color, format)
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

/// The index type of the draw whose indices a vertex function reads at their address: the
/// private list form's `u32` (`LoweringInputs::private_list`), else the primitive ID's de-indexed
/// form's (`LoweringInputs::primitive_id_indices`, under `primitive_id_varying`); `None` for any
/// other function.
pub(super) fn captured_index_type(cx: &SelectCx<'_>) -> Option<crate::metal::IndexType> {
    match cx.stage {
        Stage::Vertex if cx.inputs.private_list => Some(crate::metal::IndexType::UInt32),
        Stage::Vertex => cx
            .inputs
            .primitive_id_indices
            .filter(|_| cx.inputs.primitive_id_varying),
        _ => None,
    }
}

/// Whether a vertex function runs in the private list form (`LoweringInputs::private_list`): its
/// list's positions from the draw's first vertex, Metal's base vertex in the word before.
pub(super) fn listed(cx: &SelectCx<'_>) -> bool {
    cx.stage == Stage::Vertex && cx.inputs.private_list
}

/// The feature a form that reads a draw's indices at their address serves: Metal's private
/// primitive types for the private list form, the primitive ID for its de-indexed form.
fn indices_feature(cx: &SelectCx<'_>) -> Feature {
    match listed(cx) {
        true => Feature::PrivatePrimitives,
        false => Feature::PrimitiveId,
    }
}

/// Plan the execution range test of a vertex function compiled under
/// `LoweringInputs::indirect_range` ([`super::StageIoPlan::indirect_range`]): its range's address
/// and its command, pushed, and its `Position` output, where it has one. The range is read by
/// address: a device without device addresses refuses it.
fn indirect_range(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let f = cx.profile.fields();
    if f.buffer_access != crate::device::profile::BufferAccess::Addressed || !f.leaf.int64 {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![cx.profile.addressed_needs()],
        )
        .into());
    }
    cx.push.insert(PushField::IndirectRange);
    cx.push.insert(PushField::IndirectSequence);
    cx.push.insert(PushField::IndirectBase);
    let sp = &mut cx.families.stage_io;
    sp.indirect_range = Some(sp.invariant);
    Ok(())
}

/// Refuse the de-indexed or private list form ([`indices_feature`]) on a device without device
/// addresses, naming what it needs: the indices are reached by address.
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

/// Plan the index read of type `ty` of the de-indexed form or the private list form once
/// ([`super::CapturedIndex`]): its address push field, the de-indexed form's layout, the private
/// list form's first vertex (as the primitive ID's,
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
    Ok(())
}

/// Plan builtin parameter `b` of AIR type `ty`, a scalar integer, of a vertex function in the
/// de-indexed or private list form of index type `index` (`None` for `[[base_vertex]]`):
/// `[[vertex_id]]` is
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
/// amplifications drawn apart).
fn merged(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let i = cx.inputs;
    if !i.replay_bindings
        || i.primitive_id_varying
        || i.primitive_id_indices.is_some()
        || i.private_list
        || i.amplification_draws.is_some()
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
    let command = flat_uint(
        sp,
        VaryingKey::merged_command(),
        IoRole::MergedCommand,
        IoDir::Output,
    )?;
    sp.merged = Some(MergedRead {
        vertex,
        instance,
        command,
    });
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
/// ([`MergedRead`]), and a fragment function that reads a command's buffers its command's
/// fragment slot table, at its flat `IoRole::MergedCommand` input (`StageIoPlan::merged_fragment`),
/// so `RootTableAddress` is no push field of either; the fragment function's is the header's
/// address, `PushField::MergedDrawAddress`.
pub(crate) fn finish_merged(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if !cx.inputs.merged_draws {
        return Ok(());
    }
    match cx.stage {
        Stage::Vertex => {
            cx.push.remove(&PushField::RootTableAddress);
        }
        Stage::Fragment if cx.push.remove(&PushField::RootTableAddress) => {
            cx.push.insert(PushField::MergedDrawAddress);
            let sp = &mut cx.families.stage_io;
            let object = flat_uint(
                sp,
                VaryingKey::merged_command(),
                IoRole::MergedCommand,
                IoDir::Input,
            )?;
            sp.merged_fragment = Some(object);
        }
        _ => {}
    }
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
        // Decorated `Invariant` whether or not Metal's position is `[[invariant]]`: a draw run
        // through more than one variant of its pipeline (each color group's) rasterizes the same
        // positions in each.
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
                IndexStore::Builtin => out(cx, spirv, Shape::U32, Place::Whole, f, contract),
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
            let sp = &mut cx.families.stage_io;
            sp.modes.push(ExecMode::DepthReplacing);
            sp.modes.extend(extra);
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
            cx.families.stage_io.depth_clamp = Some(DepthClamp {
                object: link.object,
                range,
            });
            cx.families.fragment.depth_output = true;
            link
        }
        // A device without stencil export refuses the output (`entry`).
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

/// The per-stage interface limits.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage == Stage::Kernel {
        // A kernel's stage inputs: the region it depends on, and their size.
        if cx.families.stage_io.region {
            cx.facts.insert(RuntimeFact::StageInRegion);
        }
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
    let deindexed = cx.stage == Stage::Vertex && captured_index_type(cx).is_some() && !listed(cx);
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
    let evaluation = matches!(cx.entry.ep.stage, AirStage::Vertex { patch: Some(_) });
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
    Ok(())
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
