//! `lower::tessellation`: a post-tessellation vertex function as a Vulkan tessellation
//! evaluation shader, or, on a device without `tessellationShader`, as a vertex shader that draws
//! the tessellator's triangles itself.
//!
//! A vertex function with `[[patch(triangle|quad, N)]]` runs once per vertex the fixed-function
//! tessellator generates, like a Vulkan `TessellationEvaluation` entry, so the module is one,
//! with the patch type as its `Triangles`/`Quads` execution mode. Spacing and vertex order are
//! render-pipeline state in Metal (the descriptor's partition mode and winding), so the pipeline's
//! control stage states them; this module does not. The function keeps its Metal stage: it binds
//! through the vertex argument table and push window, and its outputs are a vertex function's
//! (`lower::stage_io`, which gives it no default point size and drops a `[[point_size]]` output:
//! the tessellator emits triangles, which no point size sizes).
//!
//! Its inputs (this family's; `select::entry_param_family` routes them here):
//!
//! - `[[position_in_patch]]` is `TessCoord` (its first two lanes for a quad): Metal's triangle
//!   barycentrics and quad coordinates are Vulkan's, lane for lane.
//! - `[[patch_id]]` and `[[instance_id]]` are `uint` patch inputs at the fixed locations
//!   `TessellationStage::PATCH_INDEX_LOCATION` and `TessellationStage::INSTANCE_LOCATION`,
//!   which the pipeline's control stage writes (Metal's patch index, `patchStart` included, and
//!   its instance index, the base instance included): an evaluation shader has no
//!   `InstanceIndex`, and its `PrimitiveId` is not Metal's patch index once instances or a patch
//!   start are involved. A `ushort` parameter holds the input's low 16 bits zero-extended.
//! - `[[attribute(n)]]` patch inputs (`air.patch_input`) are pulled: fetched once in the function's
//!   first block from their source buffer at element `patch_id` for a `PerPatch` layout,
//!   `instance_id` for a `PerInstance` layout of step rate 1, or no element for a `Constant` one,
//!   through the same source-buffer slots and decoding as a vertex function's pulled attributes
//!   (`stage_io::fetch`, `access::pulled`).
//! - Control points (`patch_control_point<T>`, `air.patch_control_point_input`): the body calls
//!   the entry's control-point function (a declaration whose body Metal synthesizes from the
//!   vertex descriptor) with a control-point index `k` and the input. Each call becomes the fetch
//!   of the struct it returns, at the call as Metal's synthesized body reads it: member `j` is
//!   field `j`'s attribute at control point `k`'s element of its `PerPatchControlPoint` layout,
//!   which the pipeline's control stage hands the stage as patch inputs from
//!   `TessellationStage::CONTROL_POINT_LOCATION`: `patch_id · N + k`, or the control-point index
//!   buffer's index there. An index `k` outside `[0, N)`, undefined in Metal, reads control point
//!   `N - 1`'s element. A member no instruction reads, and a field Metal marks `air.arg_unused`, is
//!   zero and fetches nothing.
//! - `[[base_instance]]` is the second lane of `PushField::BaseVertexInstance`, which the binder
//!   writes with the draw's base instance: an evaluation shader has no `BaseInstance`. An indirect
//!   draw of patches then needs its arguments on the host, which the runtime does not read yet
//!   (`NotImplemented(IndirectDrawArguments)`). Emulated (below), it is the draw's work buffer's,
//!   Metal's arguments for an indirect draw, and needs nothing of the host.
//! - `[[amplification_id]]` is `ViewIndex` and `[[amplification_count]]` the
//!   `SpecKey::AmplificationCount` constant, as in a vertex function. A multiview pipeline with
//!   tessellation stages also needs `multiviewTessellationShader`, the pipeline's to require.
//!
//! # Without `tessellationShader`
//!
//! On a device without `tessellationShader` ([`emulates`]) the function is a plain `Vertex` entry
//! that draws the tessellator's triangles itself, and its render pipeline has no control or
//! evaluation stage (`GraphicsState::emulated_tessellation`). It bakes in the render pipeline
//! descriptor's tessellation state (`LoweringInputs::tessellation`, Metal's defaults without
//! one), whose partition, winding and maximum factor decide its points ([`tessellator`]).
//!
//! Each patch draw first runs the tessellation scan (`bind::tessellation_scan`), which processes
//! each patch's factors into the draw's records, one per instance and patch, and writes its
//! chunks' indirect draws ([`work`]). Chunk `k` (`PushField::TessellationChunk`) draws the draw's
//! triangles `[k · C, (k + 1) · C)`, three vertices each, as a `TRIANGLE_LIST` from vertex 0 of
//! one instance. Before the function's body, the entry's wrapper ([`wrap_entry`]), for the vertex
//! `VertexIndex` names:
//!
//! - takes triangle `t = VertexIndex / 3` of the chunk and its corner `VertexIndex % 3`;
//! - finds the record that holds it by bisection over the chunk's records, from the chunk's first
//!   record through the next chunk's (or the draw's last), each keyed by its first triangle less
//!   the chunk's modulo `2^32`, exact within a chunk; among records of one key the last, which is
//!   the one that draws (the ones before it draw nothing);
//! - decodes the record's triangle `t - key` to its corner's point with the record's processed
//!   factors and tile word (`tessellator::emit_decode`), and places it in the patch's domain
//!   (`emit_place`): Metal's `[[position_in_patch]]`;
//! - and from record `r = i · patchCount + p` hands the function, through `Private` variables,
//!   what the native control stage hands the evaluation stage: `[[patch_id]]`, the patch index
//!   buffer's index at the position `patchStart + p` where the draw binds one, else the position
//!   (`link::tessellation::control`); `[[instance_id]]`, `i + baseInstance`; `[[base_instance]]`;
//!   and each control point's element, `patch_id · N + k` or the control-point index buffer's
//!   index there. Patch inputs and control points are then pulled as above.
//!
//! Under `LoweringInputs::primitive_id_varying` (its pipeline's fragment function reads
//! `[[primitive_id]]`) the function also writes the patch's position `patchStart + p` to the
//! primitive ID's flat `uint` varying (`IoRole::PrimitiveId`), which that fragment function reads
//! as its `[[primitive_id]]`: Metal's primitive id behind the tessellator, as measured on the M3
//! Max. A plain vertex function writes the same varying from another source on a device that
//! cannot declare a fragment `PrimitiveId` (`lower::stage_io`); this one never takes that form,
//! on any device. Vulkan's flat shading gives a fragment the provoking
//! vertex's value ("Flat shading a vertex output attribute means to assign all vertices of the
//! primitive the same value for that output. The output values assigned are those of the
//! provoking vertex of the primitive.", Fixed-Function Vertex Post-Processing, Flat Shading); every
//! vertex of a triangle writes its record's position, so any provoking vertex gives it. Behind an
//! evaluation stage a fragment `[[primitive_id]]` stays refused (`NotImplemented(Tessellation)`).
//!
//! Every attribute is pulled, so the lowering inputs must project each one it reads
//! (`VertexFetch::Pulled`); without a projection the compile needs the vertex descriptor
//! (`NeedsInput(VertexDescriptor)`), since an evaluation shader has no fixed-function input.
//!
//! Refused, `NotImplemented(Tessellation)`: other builtins, an attribute outside the patch input,
//! a control point read from a function that does not state its
//! control-point count, a per-instance patch input of step rate above 1, a step function the
//! attribute's kind does not name, and simdgroup operations (their subgroup support is the
//! evaluation stage's, which the profile does not model).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{
    AbiInt, Builtin, ControlPointField, ParamBinding, ParamRole, PatchInput, PatchKind,
    Stage as AirStage,
};
use crate::air::ir::{
    ArgIdx, Body, Callee, Inst, InstId, InstKind, Type, TypeId as AirTypeId, Value,
};
use crate::air::Entry;
use crate::device::subgroup::SimdClass;
use crate::device::DeviceProfile;
use crate::error::{DescriptorError, Error, Feature, LowerFamily, RuntimeInput, Unsupported};
use crate::lower::access::{pulled, BufferReach};
use crate::lower::arith::legal::{form_on, refuse_int64_interface};
use crate::lower::arith::{self, ops, FloatFmt, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::stage_io;
use crate::lower::stage_io::fetch::{self, Encoding, PulledRead, Step, Stride};
use crate::lower::strategy::FamilyStrategy;
use crate::memory::contain::{self, Check};
use crate::memory::leaf::k32;
use crate::metal::VertexStepFunction;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{PatchDomain, TessellationStage};
use crate::pipeline::lower_inputs::VertexFetch;
use crate::pipeline::stage_io::{IoNumeric, IoType, PulledInput, StageBuiltin};
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ExecMode, GlslOp, IoDir, MemAccess, Type as SType,
};

mod emulated;
pub mod factors;
pub mod tessellator;
pub mod work;

pub use emulated::{wrap_entry, Emulation};

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Tessellation)).into()
}

fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::Tessellation).into()
}

fn invariant(detail: &str) -> Error {
    Error::Internal(crate::error::InternalError::new(
        crate::error::InternalKind::Invariant,
        detail,
    ))
}

/// The element index a pulled patch attribute reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Element {
    /// The patch index (`PerPatch`).
    Patch,
    /// The instance index (`PerInstance`, step rate 1).
    Instance,
    /// `patch · N + k` for the control-point index `k` of the call (`PerPatchControlPoint`).
    ControlPoint,
    /// None: the attribute's offset alone (`Constant`).
    None,
}

/// One pulled patch or control-point attribute read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PatchRead {
    pub attribute: u32,
    pub encoding: Encoding,
    /// The Metal buffer index of the source buffer.
    pub buffer: u32,
    /// The attribute's byte offset in an element.
    pub offset: u32,
    pub element: Element,
    /// The value's form.
    pub form: Form,
    pub check: Check,
}

impl PatchRead {
    /// An upper estimate of the SPIR-V words the read emits: a vertex function's pulled read of the
    /// same format, plus the element's index arithmetic.
    pub fn words(&self) -> u32 {
        let same = PulledRead {
            attribute: self.attribute,
            encoding: self.encoding,
            buffer: self.buffer,
            offset: self.offset,
            step: Step::Constant,
            stride: Stride::Push(PushField::VertexBufferStride(self.buffer)),
            form: self.form,
            check: self.check,
        };
        same.words().saturating_add(32)
    }
}

/// Where a used parameter of the evaluation entry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// `TessCoord`, its first `lanes` lanes.
    Coord { lanes: u8 },
    /// The patch-index patch input, into an integer of `form` (a `ushort` holds its low 16 bits
    /// zero-extended).
    PatchIndex { form: Form },
    /// The instance patch input, into an integer of `form`.
    Instance { form: Form },
    /// `[[base_instance]]`, the draw's base instance, lane 1 of `PushField::BaseVertexInstance`,
    /// into an integer of `form`: an evaluation shader has no `BaseInstance`.
    BaseInstance { form: Form },
    /// `ViewIndex`, Metal's `[[amplification_id]]`, into an integer of `form`.
    ViewIndex { form: Form },
    /// `[[amplification_count]]`, the `SpecKey::AmplificationCount` constant.
    AmplificationCount { form: Form },
    /// A patch attribute, fetched at entry and held.
    Pulled(PatchRead),
    /// The control-point input: only its function's calls read it.
    ControlPoints,
}

/// One used parameter of the evaluation entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvalParam {
    /// The LLVM parameter index.
    pub param: u32,
    pub source: Source,
}

/// Which members of a control-point call's result the body reads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Reads {
    /// Only these members, each through an `extractvalue`.
    Members(BTreeSet<u32>),
    /// The whole struct (another use of the call's result).
    All,
}

/// The plan of a post-tessellation vertex function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalPlan {
    pub domain: PatchDomain,
    /// `[[patch(_, N)]]` (0 when unstated).
    pub control_points: u32,
    /// Whether the module declares `TessCoord`, the patch-index input, the instance input,
    /// `ViewIndex`, and the control points' elements.
    pub coord: bool,
    pub patch: bool,
    pub instance: bool,
    pub view: bool,
    pub elements: bool,
    /// Whether an emulated entry reads `[[base_instance]]` (from the draw's work buffer).
    pub base_instance: bool,
    /// `Some` on a device without `tessellationShader`: the entry is a plain `Vertex` entry
    /// behind [`wrap_entry`] (the module docs).
    pub emulated: Option<Emulation>,
    /// Every used parameter this family owns.
    pub params: Vec<EvalParam>,
    /// The members each control-point call's result has read, by call.
    reads: BTreeMap<InstId, Reads>,
    /// An upper estimate of the words the interface adds.
    pub words: u64,
}

/// The tessellation part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TessellationPlan {
    /// `Some` for a post-tessellation vertex function.
    pub eval: Option<EvalPlan>,
}

/// The tessellation strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TessellationStrategy {
    /// A control-point function call: the struct it returns, one read per member the body reads
    /// (`None`: a zero member). `ty` is the struct's lowered representation.
    ControlPoint {
        members: Box<[Option<PatchRead>]>,
        ty: DemoteTy,
    },
}

impl FamilyStrategy for TessellationStrategy {
    /// A control-point call is qualified by the reach of the source buffers it reads
    /// (`access::BufferReach`): the vertex stage's buffer slots `pulled_read` plans.
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str> {
        match self {
            TessellationStrategy::ControlPoint { members, .. } => members
                .iter()
                .flatten()
                .map(|r| {
                    plans
                        .access
                        .source_reach(fetch::slot(Stage::Vertex, r.buffer))
                })
                .fold(BufferReach::NONE, BufferReach::union)
                .qualify("tessellation.control_point"),
        }
    }

    fn words(&self) -> u32 {
        match self {
            TessellationStrategy::ControlPoint { members, .. } => members
                .iter()
                .map(|m| m.map_or(8, |r| r.words()))
                .fold(32u32, u32::saturating_add),
        }
    }
}

/// The patch of `e` when it is a post-tessellation vertex function, with its control-point count;
/// `None` for `[[patch(kind)]]`, which states none.
fn patch_of(e: &Entry) -> Option<(PatchDomain, Option<u32>)> {
    match e.ep.stage {
        AirStage::Vertex { patch: Some(p) } => Some((
            match p.kind {
                PatchKind::Triangle => PatchDomain::Triangle,
                PatchKind::Quad => PatchDomain::Quad,
            },
            p.control_points,
        )),
        _ => None,
    }
}

/// Whether this family owns parameter role `role` of entry `e`: every builtin and stage input of
/// a post-tessellation vertex function.
pub fn owns_param(e: &Entry, role: &ParamRole) -> bool {
    patch_of(e).is_some()
        && matches!(
            role,
            ParamRole::Builtin(_) | ParamRole::PatchInput(_) | ParamRole::VertexInput { .. }
        )
}

/// Whether `f` is one of the entry's control-point functions (`air.patch_control_point_function`).
pub fn is_control_point_fetch(e: &Entry, f: crate::air::ir::FuncId) -> bool {
    e.ep.is_control_point_function(f)
}

/// The evaluation interface of the plan's module, for the artifact (`StageInterface`).
pub fn interface(plan: &Plan) -> Option<TessellationStage> {
    plan.families.tessellation.eval.as_ref().map(|e| {
        let stage = TessellationStage::new(e.domain, e.control_points);
        match e.emulated {
            Some(em) => stage.emulated(em.desc),
            None => stage,
        }
    })
}

/// Each control-point call of `body` (of entry `e`) with the members of its result the body
/// reads: one pass over the body, and one more when it calls a control-point function.
fn control_point_reads(e: &Entry, body: &Body) -> BTreeMap<InstId, Reads> {
    let mut reads: BTreeMap<InstId, Reads> = BTreeMap::new();
    for (id, inst) in body.insts.iter() {
        if let InstKind::Call(call) = &inst.kind {
            if let Callee::Direct(f) = call.callee {
                if is_control_point_fetch(e, f) {
                    reads
                        .entry(id)
                        .or_insert_with(|| Reads::Members(BTreeSet::new()));
                }
            }
        }
    }
    if reads.is_empty() {
        return reads;
    }
    for inst in body.insts.values() {
        match &inst.kind {
            InstKind::ExtractValue {
                agg: Value::Inst(c),
                indices,
            } => {
                if let (Some(Reads::Members(m)), Some(k)) = (reads.get_mut(c), indices.first()) {
                    m.insert(*k);
                }
            }
            kind => kind.for_each_value(|v| {
                if let Value::Inst(c) = v {
                    if let Some(r) = reads.get_mut(&c) {
                        *r = Reads::All;
                    }
                }
            }),
        }
    }
    reads
}

/// The Metal attribute indices the specialized post-tessellation vertex function `e` reads, with
/// the numeric class it reads each as: its used patch inputs and every control-point field some
/// call's result has read and Metal does not mark unused (the attributes [`select_param`] and
/// [`select`] fetch). Empty for any other function.
pub fn attributes_read(e: &Entry) -> BTreeMap<u32, IoNumeric> {
    let mut out = BTreeMap::new();
    if patch_of(e).is_none() {
        return out;
    }
    let Some(func) = e.module.func(e.func) else {
        return out;
    };
    let Some(body) = func.body.as_ref() else {
        return out;
    };
    let types = e.module.types();
    let numeric = |ty: AirTypeId| {
        if arith::legal::form_of(types, ty).is_ok_and(|f| f.is_float()) {
            IoNumeric::Float
        } else {
            IoNumeric::Uint
        }
    };
    let mut used = BTreeSet::new();
    for inst in body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                used.insert(a.raw());
            }
        });
    }
    for b in &e.ep.params {
        let ParamRole::PatchInput(PatchInput::Patch { location }) = &b.role else {
            continue;
        };
        if let (AbiInt::Const(a), true) = (location.index, used.contains(&b.param)) {
            if let Some(ty) = e.module.value_ty(e.func, Value::Arg(ArgIdx::new(b.param))) {
                out.insert(a, numeric(ty));
            }
        }
    }
    for (id, reads) in control_point_reads(e, body) {
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        let Some(Value::Arg(input)) = call.args.get(1) else {
            continue;
        };
        let Some(Type::Struct {
            fields: members, ..
        }) = types.get(inst.ty)
        else {
            continue;
        };
        let fields = e.ep.params.iter().find_map(|b| match &b.role {
            ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. })
                if b.param == input.raw() =>
            {
                Some(fields)
            }
            _ => None,
        });
        for (k, (field, member)) in fields.into_iter().flatten().zip(members.iter()).enumerate() {
            let read = match &reads {
                Reads::All => true,
                Reads::Members(m) => u32::try_from(k).is_ok_and(|k| m.contains(&k)),
            };
            if !read || field.unused || field.gated_off {
                continue;
            }
            if let AbiInt::Const(a) = field.location.index {
                out.insert(a, numeric(*member));
            }
        }
    }
    out
}

/// Plan a post-tessellation vertex function: check the profile and record, for each control-point
/// call, the members of its result the body reads. A patch that states no count
/// (`[[patch(kind)]]`, a patch of the draw's count) is not implemented: the control stage fetches
/// each patch's `N` control points with `N` compiled in, and no pipeline is keyed on a draw's count.
pub fn plan_entry(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let Some((domain, control_points)) = patch_of(cx.entry) else {
        return Ok(());
    };
    let Some(control_points) = control_points else {
        return Err(Unsupported::not_implemented(Feature::Tessellation).into());
    };
    // Without `tessellationShader` the function is a plain vertex function behind the
    // tessellator's emulation (the module docs).
    let emulated = match emulates(cx.profile) {
        true => Some(emulated::plan(cx, domain)?),
        false => None,
    };
    let reads = control_point_reads(cx.entry, cx.body);
    cx.families.tessellation.eval = Some(EvalPlan {
        domain,
        control_points,
        coord: false,
        patch: false,
        instance: false,
        view: false,
        elements: false,
        base_instance: false,
        emulated,
        params: Vec::new(),
        reads,
        words: 0,
    });
    Ok(())
}

/// Whether profile `p` runs post-tessellation vertex functions as plain vertex functions behind
/// the tessellator's emulation (the module docs): it does not enable `tessellationShader`.
pub fn emulates(p: &DeviceProfile) -> bool {
    !p.fields().fragment.tessellation_shader
}

fn eval<'c>(cx: &'c mut SelectCx<'_>) -> Result<&'c mut EvalPlan, Error> {
    cx.families
        .tessellation
        .eval
        .as_mut()
        .ok_or_else(|| invariant("a tessellation input of an entry without a patch"))
}

/// The form of an interface value of AIR type `ty`: a scalar or 2-4-lane vector held in 32 bits
/// (as `stage_io` declares interface values).
fn io_form(cx: &SelectCx<'_>, ty: AirTypeId) -> Result<Form, Error> {
    refuse_int64_interface(cx.entry, ty)?;
    let f = form_on(cx.entry, cx.profile, ty)?;
    match f.held {
        Held::Int { container: 32, .. } => Ok(f),
        Held::Float(ff) if ff.container() == 32 => Ok(f),
        _ => Err(not_lowered()),
    }
}

/// The projection entry of attribute `attribute`: the lowering inputs must project it pulled.
fn projected(cx: &SelectCx<'_>, attribute: u32) -> Result<VertexFetch, Error> {
    let Some(projection) = cx.inputs.vertex_fetch.as_ref() else {
        return Err(Error::NeedsInput(RuntimeInput::VertexDescriptor {
            attribute,
        }));
    };
    projection
        .attributes
        .get(&attribute)
        .copied()
        .ok_or(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute,
        }))
}

/// What kind of attribute a read is for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Patch,
    ControlPoint,
}

/// Plan the pulled read of attribute `attribute` of kind `kind` into a value of form `form`: its
/// element and the push fields, facts, and source buffer it needs (as `stage_io::fetch` plans a
/// vertex function's pulled attribute).
fn pulled_read(
    cx: &mut SelectCx<'_>,
    attribute: u32,
    kind: Kind,
    form: Form,
) -> Result<PatchRead, Error> {
    let fetch = projected(cx, attribute)?;
    let VertexFetch::Pulled {
        format,
        buffer_index: buffer,
        offset,
        step_function,
        step_rate,
    } = fetch
    else {
        // A fixed-function projection of an attribute nothing but the shader can fetch.
        return Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute,
        }));
    };
    let encoding = fetch::encoding(format)?;
    if encoding.is_float() != form.is_float() || form.is_bool() {
        return Err(not_lowered());
    }
    fetch::decode_notes(cx, encoding, form);
    let element = match (kind, step_function) {
        (Kind::ControlPoint, VertexStepFunction::PerPatchControlPoint) => Element::ControlPoint,
        (Kind::Patch, VertexStepFunction::PerPatch) => Element::Patch,
        (Kind::Patch, VertexStepFunction::PerInstance) => match step_rate {
            0 => {
                return Err(Error::InvalidDescriptor(DescriptorError::VertexStepRate {
                    step_rate,
                }))
            }
            1 => Element::Instance,
            _ => return Err(not_implemented()),
        },
        (Kind::Patch, VertexStepFunction::Constant) => Element::None,
        _ => return Err(not_implemented()),
    };
    if element != Element::None {
        cx.push.insert(PushField::VertexBufferStride(buffer));
        cx.facts
            .insert(RuntimeFact::VertexBufferStride { layout: buffer });
    }
    cx.push.insert(PushField::SlotBindOffset(fetch::slot(
        Stage::Vertex,
        buffer,
    )));
    cx.facts
        .insert(RuntimeFact::VertexFetch { attr: attribute });
    // The source buffers are the vertex function's pulled-attribute buffers: the access family
    // chooses their slots, and link checks the projection against this record.
    let sp = &mut cx.families.stage_io;
    sp.buffers.insert(buffer);
    if let Err(at) = sp
        .interface
        .pulled
        .binary_search_by_key(&attribute, |p| p.index)
    {
        let ty = IoType {
            numeric: if form.is_float() {
                IoNumeric::Float
            } else {
                IoNumeric::Uint
            },
            components: form.lanes,
            source_bits: form.bits(),
        };
        sp.interface
            .pulled
            .insert(at, PulledInput::new(attribute, ty, fetch));
    }
    let check = contain::encoding(cx.profile);
    let e = eval(cx)?;
    match element {
        Element::Patch => e.patch = true,
        Element::ControlPoint => e.elements = true,
        Element::Instance => e.instance = true,
        Element::None => {}
    }
    Ok(PatchRead {
        attribute,
        encoding,
        buffer,
        offset,
        element,
        form,
        check,
    })
}

/// A scalar `uint` read into an integer of at most 32 bits: its form.
fn int_scalar(f: Form) -> Result<Form, Error> {
    match (f.held, f.lanes) {
        (Held::Int { container: 32, .. }, 1) => Ok(f),
        _ => Err(not_lowered()),
    }
}

/// Plan a used builtin or patch input `b` of a post-tessellation vertex function.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let ty = cx
        .value_ty(Value::Arg(ArgIdx::new(b.param)))
        .ok_or_else(|| invariant("a tessellation parameter without a type"))?;
    let domain = eval(cx)?.domain;
    let source = match &b.role {
        ParamRole::Builtin(Builtin::PositionInPatch) => {
            let f = form_on(cx.entry, cx.profile, ty)?;
            let lanes = match domain {
                PatchDomain::Triangle => 3,
                PatchDomain::Quad => 2,
            };
            if f.held != Held::Float(FloatFmt::F32) || f.lanes != lanes {
                return Err(not_lowered());
            }
            eval(cx)?.coord = true;
            Source::Coord { lanes }
        }
        ParamRole::Builtin(Builtin::PatchId) => {
            refuse_int64_interface(cx.entry, ty)?;
            let form = int_scalar(form_on(cx.entry, cx.profile, ty)?)?;
            eval(cx)?.patch = true;
            Source::PatchIndex { form }
        }
        ParamRole::Builtin(Builtin::InstanceId) => {
            refuse_int64_interface(cx.entry, ty)?;
            let form = int_scalar(form_on(cx.entry, cx.profile, ty)?)?;
            eval(cx)?.instance = true;
            Source::Instance { form }
        }
        // The binder writes a draw of patches' base instance into the push field's second lane
        // (and reads an indirect draw's arguments on the host to do so).
        // An emulated entry reads it from the draw's work buffer, Metal's arguments for an
        // indirect draw.
        ParamRole::Builtin(Builtin::BaseInstance) if eval(cx)?.emulated.is_some() => {
            refuse_int64_interface(cx.entry, ty)?;
            let form = int_scalar(form_on(cx.entry, cx.profile, ty)?)?;
            eval(cx)?.base_instance = true;
            Source::BaseInstance { form }
        }
        ParamRole::Builtin(Builtin::BaseInstance) => {
            refuse_int64_interface(cx.entry, ty)?;
            let form = int_scalar(form_on(cx.entry, cx.profile, ty)?)?;
            cx.push.insert(PushField::BaseVertexInstance);
            cx.facts.insert(RuntimeFact::BaseVertexInstance);
            Source::BaseInstance { form }
        }
        // A tessellation pipeline is drawn amplified as views only: the binder does not draw its
        // patches one amplification at a time.
        ParamRole::Builtin(Builtin::AmplificationId) if cx.inputs.amplification_draws.is_some() => {
            return Err(Unsupported::not_implemented(Feature::VertexAmplification).into())
        }
        ParamRole::Builtin(Builtin::AmplificationId) => {
            refuse_int64_interface(cx.entry, ty)?;
            let form = int_scalar(form_on(cx.entry, cx.profile, ty)?)?;
            stage_io::view_index_enabled(cx)?;
            let contract = &mut cx.families.stage_io.interface.builtins;
            if !contract.contains(&StageBuiltin::ViewIndex) {
                contract.push(StageBuiltin::ViewIndex);
                contract.sort();
            }
            eval(cx)?.view = true;
            Source::ViewIndex { form }
        }
        ParamRole::Builtin(Builtin::AmplificationCount) => Source::AmplificationCount {
            form: stage_io::amplification_count_form(cx, ty)?,
        },
        ParamRole::Builtin(_) => return Err(not_implemented()),
        ParamRole::PatchInput(PatchInput::Patch { location }) => {
            let AbiInt::Const(attribute) = location.index else {
                return Err(not_lowered());
            };
            let f = io_form(cx, ty)?;
            Source::Pulled(pulled_read(cx, attribute, Kind::Patch, f)?)
        }
        ParamRole::PatchInput(PatchInput::ControlPoints { .. }) => Source::ControlPoints,
        // An attribute outside the patch input: not a post-tessellation input Metal defines.
        ParamRole::VertexInput { .. } => return Err(not_implemented()),
        _ => return Err(invariant("a parameter tessellation does not own")),
    };
    eval(cx)?.params.push(EvalParam {
        param: b.param,
        source,
    });
    Ok(())
}

/// Select a call to the entry's control-point function (routed here by `select::route`).
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
) -> Result<TessellationStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(invariant("a tessellation instruction that is not a call"));
    };
    let Callee::Direct(f) = call.callee else {
        return Err(invariant("a control-point fetch that is not a direct call"));
    };
    let [index, Value::Arg(input)] = call.args.as_ref() else {
        return Err(invariant("a control-point fetch of another shape"));
    };
    let fields: Vec<ControlPointField> = cx
        .entry
        .ep
        .params
        .iter()
        .find_map(|b| match &b.role {
            ParamRole::PatchInput(PatchInput::ControlPoints { function, fields })
                if b.param == input.raw() && *function == f =>
            {
                Some(fields.clone())
            }
            _ => None,
        })
        .ok_or_else(|| invariant("a control-point fetch of another input"))?;
    let index_ty = cx
        .value_ty(*index)
        .ok_or_else(|| invariant("a control-point index without a type"))?;
    int_scalar(form_on(cx.entry, cx.profile, index_ty)?)?;
    let members: Vec<AirTypeId> = match cx.entry.module.types().get(inst.ty) {
        Some(Type::Struct { fields: m, .. }) if m.len() == fields.len() => m.to_vec(),
        _ => return Err(invariant("a control-point struct of another shape")),
    };
    let ty = arith::lowered_type(cx.entry, cx.profile, inst.ty)?;
    let reads = eval(cx)?
        .reads
        .get(&at)
        .cloned()
        .ok_or_else(|| invariant("an unrecorded control-point call"))?;
    let mut out = Vec::with_capacity(members.len());
    for (k, (field, member)) in fields.iter().zip(members).enumerate() {
        let read = match &reads {
            Reads::All => true,
            Reads::Members(m) => u32::try_from(k).is_ok_and(|k| m.contains(&k)),
        };
        // A field Metal marks unused (`air.arg_unused`) no instruction's result depends on, and
        // its pipeline fetches nothing for it (a descriptor may omit its attribute): zero, as an
        // unread member, also when the struct is used whole. A field its function-constant
        // predicate removes is not declared (MSL 5.8.1.3): Metal fetches nothing for it and
        // defines no value for its read, which reads zero, as any absent argument passed by
        // value does.
        if !read || field.unused || field.gated_off {
            out.push(None);
            continue;
        }
        let AbiInt::Const(attribute) = field.location.index else {
            return Err(not_lowered());
        };
        let form = io_form(cx, member)?;
        out.push(Some(pulled_read(cx, attribute, Kind::ControlPoint, form)?));
    }
    // The control stage hands each of `N` control points' elements: without `N` there is none.
    if out.iter().any(Option::is_some) && eval(cx)?.control_points == 0 {
        return Err(not_implemented());
    }
    Ok(TessellationStrategy::ControlPoint {
        members: out.into_boxed_slice(),
        ty,
    })
}

/// Words per interface object and per use of a loaded input.
const OBJECT_WORDS: u64 = 32;
const USE_WORDS: u64 = 24;

/// Complete the plan: refuse what the evaluation stage does not model, and estimate the words
/// of its interface.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let Some(e) = cx.families.tessellation.eval.as_ref() else {
        return Ok(());
    };
    if cx.simd != SimdClass::None {
        return Err(not_implemented());
    }
    if let Some(em) = e.emulated {
        let reads = (e.patch, e.elements);
        emulated::finish(cx, reads, &em);
    }
    let Some(e) = cx.families.tessellation.eval.as_ref() else {
        return Ok(());
    };
    let objects = u64::from(e.coord)
        + u64::from(e.patch)
        + u64::from(e.instance)
        + u64::from(e.view)
        + u64::from(e.elements);
    let mut words = objects.saturating_mul(OBJECT_WORDS);
    let mut loaded = Vec::new();
    for p in &e.params {
        match p.source {
            Source::Pulled(r) => words = words.saturating_add(u64::from(r.words())),
            Source::Coord { .. }
            | Source::PatchIndex { .. }
            | Source::Instance { .. }
            | Source::BaseInstance { .. }
            | Source::ViewIndex { .. }
            | Source::AmplificationCount { .. } => loaded.push(p.param),
            Source::ControlPoints => {}
        }
    }
    for inst in cx.body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                if loaded.contains(&a.raw()) {
                    words = words.saturating_add(USE_WORDS);
                }
            }
        });
    }
    if let Some(e) = cx.families.tessellation.eval.as_mut() {
        e.words = words;
    }
    Ok(())
}

/// The words the evaluation interface adds to the size estimate.
pub fn estimate_words(plan: &TessellationPlan) -> u64 {
    plan.eval.as_ref().map_or(0, |e| e.words)
}

/// The module-scope objects of an evaluation entry.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    pub coord: Option<GlobalId>,
    pub patch: Option<GlobalId>,
    pub instance: Option<GlobalId>,
    pub view: Option<GlobalId>,
    pub elements: Option<GlobalId>,
    /// An emulated entry's `[[base_instance]]`, a `Private` variable its wrapper writes.
    pub base_instance: Option<GlobalId>,
    /// An emulated entry's `VertexIndex`, work buffer, records, and patch index and
    /// control-point index buffers.
    pub vertex_index: Option<GlobalId>,
    pub work: Option<GlobalId>,
    pub records: Option<GlobalId>,
    pub patch_indices: Option<GlobalId>,
    pub control_point_indices: Option<GlobalId>,
}

/// Declare the evaluation entry's inputs and add its patch domain to `modes`.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let Some(e) = plan.families.tessellation.eval.as_ref() else {
        return Ok(Declared::default());
    };
    let mut d = Declared::default();
    if let Some(em) = &e.emulated {
        emulated::declare(mb, e, em, &mut d)?;
        if e.view {
            let u32_ = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            d.view = Some(mb.builtin(crate::spv::BuiltIn::ViewIndex, IoDir::Input, u32_)?);
        }
        return Ok(d);
    }
    modes.push(match e.domain {
        PatchDomain::Triangle => ExecMode::Triangles,
        PatchDomain::Quad => ExecMode::Quads,
    });
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    if e.coord {
        let f32_ = mb.ty(SType::Float { width: 32 })?;
        let v3 = mb.ty(SType::Vector { elem: f32_, len: 3 })?;
        d.coord = Some(mb.builtin(crate::spv::BuiltIn::TessCoord, IoDir::Input, v3)?);
    }
    if e.patch {
        let at = TessellationStage::PATCH_INDEX_LOCATION;
        d.patch = Some(mb.fixed_io(at, IoDir::Input, u32_, true)?);
    }
    if e.instance {
        let at = TessellationStage::INSTANCE_LOCATION;
        d.instance = Some(mb.fixed_io(at, IoDir::Input, u32_, true)?);
    }
    if e.view {
        d.view = Some(mb.builtin(crate::spv::BuiltIn::ViewIndex, IoDir::Input, u32_)?);
    }
    if e.elements {
        let at = TessellationStage::CONTROL_POINT_LOCATION;
        let ty = element_array(mb, e.control_points)?;
        d.elements = Some(mb.fixed_io(at, IoDir::Input, ty, true)?);
    }
    Ok(d)
}

/// The `uvec4[ceil(n / 4)]` that holds `n` control points' elements
/// (`TessellationStage::CONTROL_POINT_LOCATION`).
pub fn element_array(mb: &mut ModuleBuilder, n: u32) -> BResult<TypeId> {
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let v4 = mb.ty(SType::Vector { elem: u32_, len: 4 })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(n.div_ceil(4))),
    })?;
    mb.ty(SType::Array {
        elem: v4,
        len,
        stride: None,
    })
}

/// Load the input `g` in the open block.
fn load(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, g: Option<GlobalId>) -> BResult<SValue> {
    let Some(g) = g else {
        return Err(cx.fail("a tessellation input that is not declared"));
    };
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default())
}

/// The value of read `r` at element `element` (`None` for a constant step), fetched in the open
/// block.
fn fetch_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: &PatchRead,
    element: Option<SValue>,
) -> BResult<SValue> {
    let element = match element {
        Some(e) => Some((
            e,
            cx.push_field(fb, PushField::VertexBufferStride(r.buffer))?,
        )),
        None => None,
    };
    let m = fetch::displacement(fb, element, r.offset)?;
    let read = pulled::prepare(cx, fb, r.buffer, m, r.encoding.bytes(), r.check)?;
    fetch::fetch_from(fb, read, r.encoding, r.form)
}

/// Fetch every pulled patch input in the function's first block, before the walk emits any
/// instruction of the body, and hold each value for all its uses (as `stage_io::emit_entry` fetches
/// a vertex function's attributes).
pub fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let Some(e) = cx.plan.families.tessellation.eval.as_ref() else {
        return Ok(());
    };
    let pulled: Vec<(u32, PatchRead)> = e
        .params
        .iter()
        .filter_map(|p| match p.source {
            Source::Pulled(r) => Some((p.param, r)),
            _ => None,
        })
        .collect();
    for (param, r) in pulled {
        let Some(v) = cx.plan.values.id(Value::Arg(ArgIdx::new(param))) else {
            return Err(cx.fail(format!("patch input {param} is out of range")));
        };
        let d = &cx.decl.families.tessellation;
        let (patch, instance) = (d.patch, d.instance);
        let element = match r.element {
            Element::Patch => Some(load(cx, fb, patch)?),
            Element::Instance => Some(load(cx, fb, instance)?),
            Element::None => None,
            Element::ControlPoint => {
                return Err(cx.fail("a control-point read planned as a patch input"))
            }
        };
        let x = fetch_at(cx, fb, &r, element)?;
        cx.hold(v, x);
    }
    Ok(())
}

/// The value of builtin `b` of the evaluation entry in its lowered representation, loaded at
/// each use. A patch input's value is the one [`emit_entry`] holds, so it never reaches here.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    let source = cx
        .plan
        .families
        .tessellation
        .eval
        .as_ref()
        .and_then(|e| e.params.iter().find(|p| p.param == b.param))
        .map(|p| p.source);
    let d = &cx.decl.families.tessellation;
    let (coord, patch, instance, view) = (d.coord, d.patch, d.instance, d.view);
    match source {
        Some(Source::Coord { lanes }) => {
            let x = load(cx, fb, coord)?;
            if lanes == 3 {
                Ok(x)
            } else {
                fb.vector_shuffle(x, x, &[0, 1])
            }
        }
        // The inputs are 32-bit: a narrower parameter holds their low bits zero-extended.
        Some(Source::PatchIndex { form }) => {
            let x = load(cx, fb, patch)?;
            ops::mask(fb, form, x)
        }
        Some(Source::Instance { form }) => {
            let x = load(cx, fb, instance)?;
            ops::mask(fb, form, x)
        }
        Some(Source::BaseInstance { form }) => {
            let x = match cx.decl.families.tessellation.base_instance {
                Some(g) => load(cx, fb, Some(g))?,
                None => {
                    let v = cx.push_field(fb, PushField::BaseVertexInstance)?;
                    fb.composite_extract(v, &[stage_io::DrawBase::Instance.lane()])?
                }
            };
            ops::mask(fb, form, x)
        }
        Some(Source::ViewIndex { form }) => {
            let x = load(cx, fb, view)?;
            ops::mask(fb, form, x)
        }
        Some(Source::AmplificationCount { form }) => stage_io::amplification_count(fb, form),
        Some(Source::Pulled(_)) => {
            Err(cx.fail(format!("patch input {} was not fetched at entry", b.param)))
        }
        Some(Source::ControlPoints) => Err(cx.fail("a control-point input read as a value")),
        None => Err(cx.fail(format!("parameter {} has no tessellation value", b.param))),
    }
}

/// Emit instruction `id` under strategy `s`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    id: InstId,
    s: &TessellationStrategy,
) -> BResult<()> {
    let TessellationStrategy::ControlPoint { members, ty } = s;
    let Some(v) = cx.plan.values.inst(id) else {
        return Err(cx.fail(format!("{id:?} is out of range")));
    };
    let inst = cx.body.insts.get(id);
    let index = match inst.map(|i| &i.kind) {
        Some(InstKind::Call(call)) => call.args.first().copied(),
        _ => None,
    };
    let Some(index) = index.and_then(|i| cx.plan.values.id(i)) else {
        return Err(cx.fail(format!("{id:?} is not a control-point call")));
    };
    let n = cx
        .plan
        .families
        .tessellation
        .eval
        .as_ref()
        .map_or(0, |e| e.control_points);
    let DemoteTy::Composite(parts) = ty else {
        return Err(cx.fail("a control-point struct that is not a composite"));
    };
    // Control point `k`'s element, as the control stage hands it: component `k % 4` of element
    // `k / 4`, `k` at most `N - 1` (the module docs).
    let element = if members.iter().any(Option::is_some) {
        let k = cx.operand(fb, index)?;
        let last = k32(fb, n.saturating_sub(1))?;
        let k = fb.glsl(GlslOp::UMin, &[k, last])?;
        let two = k32(fb, 2)?;
        let quad = fb.binary(BinOp::ShiftRightLogical, k, two)?;
        let three = k32(fb, 3)?;
        let lane = fb.binary(BinOp::BitwiseAnd, k, three)?;
        let Some(elements) = cx.decl.families.tessellation.elements else {
            return Err(cx.fail("a control-point read without the elements input"));
        };
        let base = fb.module().global_value(elements)?;
        let ptr = fb.access_chain(base, &[quad, lane])?;
        Some(fb.load(ptr, MemAccess::default())?)
    } else {
        None
    };
    let mut values = Vec::with_capacity(members.len());
    for (m, part) in members.iter().zip(parts.iter()) {
        let x = match (m, element) {
            (Some(r), Some(e)) => fetch_at(cx, fb, r, Some(e))?,
            _ => {
                let Some(t) = arith::spirv_type(fb.module(), part)? else {
                    return Err(cx.fail("a control-point member without a lowered type"));
                };
                let z = fb.module().constant(Constant::Null { ty: t })?;
                fb.module().const_value(z)?
            }
        };
        values.push(x);
    }
    let Some(t) = arith::spirv_type(fb.module(), ty)? else {
        return Err(cx.fail("a control-point struct without a lowered type"));
    };
    let x = fb.composite_construct(t, &values)?;
    cx.define(v, x);
    Ok(())
}
