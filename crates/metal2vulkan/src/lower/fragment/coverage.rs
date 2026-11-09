//! `[[sample_mask]]` in a fragment function that Metal runs once per covered sample.
//!
//! Metal runs a fragment function of a multisampled pipeline once per covered sample when it
//! reads `[[sample_id]]` or a sample-qualified varying, or writes memory, and hands every one of
//! those runs the pixel's whole coverage as `[[sample_mask]]`, each run its own sample index and
//! its sample-qualified varyings at its sample, and everything else at the pixel (an M3 Max and an
//! M4, measured). A Vulkan invocation under sample shading sees only its own sample's bit of the
//! coverage, and nothing gathers a pixel's sample invocations, so such a function, compiled for a
//! multisampled pipeline (`LoweringInputs::multisampled`), runs once per pixel instead, without
//! sample shading, and its body runs once per covered sample, called from the entry [`wrap`]
//! builds:
//!
//! ```text
//! main:  rest = SampleMask[0]; kept = 0
//!        loop { if rest == 0 { break }
//!               sample = FindILsb(rest)          // the Private `loop_sample`
//!               call body
//!               kept |= SampleMaskOut[0] & (1 << sample)   // where the stage writes it
//!               rest &= rest - 1 }
//!        SampleMaskOut[0] = kept                  // where the stage writes it
//!        return
//! ```
//!
//! In the body `[[sample_id]]` reads `loop_sample` (`stage_io::Source::LoopSample`), and a
//! sample-qualified varying is interpolated at it through `InterpolateAtSample`
//! (`stage_io::Source::AtLoopSample`), declared without `Sample`, which would turn sample shading
//! on; every other input is read at the pixel, as Metal reads it in each run.
//!
//! The helper invocations of a partly covered quad read an empty coverage and leave the loop at
//! once. Where the driver's interpolation at a sample then reads garbage
//! (`DriverWorkaround::InterpolationNeedsTheWholeQuad`, RADV), the entry reads each
//! sample-qualified varying at every sample a pipeline of the device may have before the loop,
//! where the whole quad runs, into a `Private` array (`stage_io::read_at_every_sample`), and each
//! pass reads its sample's element. That driver answers a perspective varying's read at a sample
//! with the pixel center's value extrapolated along the quad's derivatives
//! (`DriverWorkaround::PerspectiveReadsExtrapolated`), its own interpolation, which the module
//! records (`PrecisionNote::ExtrapolatedInterpolation`).
//!
//! Metal writes each covered sample's outputs (color, depth, stencil) from that sample's own run,
//! and keeps the sample only where the run's `[[sample_mask]]` output has the sample's bit (an M3
//! Max under macOS 27.0.1, our probe `co`: the runs go lowest sample first). One pixel-rate
//! invocation writes one value per output, so a looping function's outputs are the last pass's,
//! written to the samples Metal's runs keep: where the stage writes `SampleMask` (the function's
//! own output, or the pipeline's private masks), the loop ORs each pass's word ANDed with its
//! sample's bit, and stores that union once the loop ends; elsewhere every covered sample is
//! written, as on Metal. Equal to Metal's where every pass computes the same outputs, and
//! recorded as `PrecisionNote::LoopedSampleOutputs`: Vulkan hands an invocation under sample
//! shading only its own sample's bit of the coverage, so no form runs each sample's pass in its
//! own invocation and reads the pixel's coverage.
//!
//! A function the loop does not fit is refused ([`CoverageLoopRefusal`]): by link in a pipeline of
//! more than one sample, from the reason its interface records (`StageIo::coverage_loop`), or,
//! compiled for a multisampled pipeline once its body showed the reason, by the compile layer.
//!
//! Metal's rasterizer splits a primitive it clips into pieces whose coverages it hands to
//! separate runs, which no Vulkan rasterizer reproduces; the loop reads the pixel's coverage of the
//! whole primitive, as today's pixel-rate read does. A function compiled for a multisampled
//! pipeline that reads `[[sample_mask]]`, looped or not, records
//! `PrecisionNote::ClippedCoverageSplit` on a device whose rasterizer is not Apple's
//! (`DriverWorkaround::ClippedCoverageNotMetals`).
//!
//! MoltenVK on Apple's GPU runs a shader that writes memory once per covered sample, as Metal
//! does, sample shading or not (`DriverWorkaround::MemoryWritingFragmentsRunPerSample`), so the
//! loop of a function that writes memory would repeat each pass there. Such a function is compiled
//! as written there instead (`CoverageLoop::DriverRunsPerSample`), each of the driver's runs
//! reading the pixel's coverage, its sample index and its sample-qualified varyings, with the
//! pipeline's sample shading off; except where MoltenVK's SPIRV-Cross masks the coverage with the
//! run's sample bit (a module with `SampleRateShading` that reads `FragCoord`), where it is refused
//! as the driver's defect.

use crate::air::abi::{
    Builtin, Interpolation as AirInterp, ParamRole, Sampling as AirSampling, Stage as AirStage,
};
use crate::air::ir::{ArgIdx, Body, Value};
use crate::device::requirements::VkFeature;
use crate::device::DriverWorkaround;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::lower::arith::legal::form_on;
use crate::lower::emit::FamilyDecls;
use crate::lower::select::{Plan, SelectCx};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::stage_io::{CoverageLoop, CoverageLoopRefusal};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::constructs::{ArmEnd, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId};
use crate::spirv::ir::{BinOp, ConstBits, Constant, GlslOp, MemAccess, StorageClass, Type};

/// The fragment family's plan for a function that reads `[[sample_mask]]` and that Metal runs
/// once per covered sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoveragePlan {
    /// Compiled for a multisampled pipeline, the body runs once per covered sample from the
    /// entry's loop (`wrap`).
    pub looped: bool,
    /// Why the function cannot loop, as far as its interface shows before its body is selected:
    /// its early tests, its attachments, or the device's interpolation functions.
    pub refusal: Option<CoverageLoopRefusal>,
    /// The function writes memory, which a driver with
    /// `DriverWorkaround::MemoryWritingFragmentsRunPerSample` repeats per covered sample.
    pub writes: bool,
    /// Such a driver runs the function once per covered sample itself, each run reading the
    /// pixel's coverage, so it is compiled as written, without the loop
    /// (`CoverageLoop::DriverRunsPerSample`).
    pub driver_runs: bool,
    /// Where the looping function reads a sample-qualified varying and the driver's interpolation
    /// needs the whole quad (`DriverWorkaround::InterpolationNeedsTheWholeQuad`): the most samples
    /// a pipeline of the device may have, at each of which the entry reads every such varying
    /// before the loop.
    pub samples_before_loop: Option<u32>,
}

/// An upper estimate of the words the looping entry adds around the body function: its function
/// type, the entry function, the loop's blocks and merges, the mask's load, the bit's extraction,
/// the call, the `rest` variable's loads and stores, and the union of the kept samples.
pub(super) const LOOP_WORDS: u64 = 144;

fn invariant(msg: &'static str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, msg))
}

/// Whether the body reads argument `param`.
fn arg_used(body: &Body, param: u32) -> bool {
    let arg = Value::Arg(ArgIdx::new(param));
    body.insts.values().any(|inst| {
        let mut used = false;
        inst.kind.for_each_value(|v| used |= v == arg);
        used
    })
}

/// Whether used parameter role `role` of AIR argument `param` is a sample-qualified float varying
/// (an integer or `bool` varying is flat whatever it is qualified with).
fn sample_varying(cx: &SelectCx<'_>, param: u32, role: &ParamRole) -> bool {
    let ParamRole::FragmentInput(v) = role else {
        return false;
    };
    // An input the linking drops is read as zero, at no sample.
    let key = crate::pipeline::sym::VaryingKey::user(&v.key.0);
    if cx.inputs.absent_varyings.contains(&key) {
        return false;
    }
    // One its vertex function writes the same at every vertex of a primitive is read flat, at
    // no sample: every sample of the pixel reads the same value.
    if cx.inputs.uniform_varyings.contains(&key) {
        return false;
    }
    if !matches!(
        v.interpolation,
        AirInterp::Smooth {
            sampling: AirSampling::Sample,
            ..
        }
    ) {
        return false;
    }
    cx.value_ty(Value::Arg(ArgIdx::new(param)))
        .and_then(|ty| form_on(cx.entry, cx.profile, ty).ok())
        .is_some_and(|f| f.is_float())
}

/// Plan the loop over the covered samples before the stage interface (`fragment::plan_entry`):
/// a fragment function that reads `[[sample_mask]]` and runs once per covered sample on Metal (it
/// reads `[[sample_id]]` or a sample-qualified varying, or has a memory side effect) loops,
/// compiled for a multisampled pipeline, unless its interface already shows why it cannot.
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment {
        return Ok(());
    }
    let e = cx.entry;
    let body = cx.body;
    let used = |role: &dyn Fn(&ParamRole) -> bool| {
        e.ep.params
            .iter()
            .any(|b| role(&b.role) && arg_used(body, b.param))
    };
    if !used(&|r| {
        matches!(
            r,
            ParamRole::Builtin(Builtin::SampleMaskIn | Builtin::PostDepthCoverage)
        )
    }) {
        return Ok(());
    }
    // Off Apple's rasterizer, a multisampled read of the coverage meets no split of a clipped
    // primitive, where Metal's can (the module docs).
    if cx.inputs.multisampled
        && cx
            .profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::ClippedCoverageNotMetals)
    {
        cx.precision.insert(PrecisionNote::ClippedCoverageSplit);
    }
    let sample_id = used(&|r| matches!(r, ParamRole::Builtin(Builtin::SampleId)));
    let sample_varying =
        e.ep.params
            .iter()
            .any(|b| sample_varying(cx, b.param, &b.role) && arg_used(body, b.param));
    let writes = body
        .insts
        .iter()
        .any(|(id, inst)| super::discard::side_effect(cx, id, inst));
    if !(sample_id || sample_varying || writes) {
        // Metal runs it once per pixel, which reads the pixel's coverage as Vulkan's does.
        return Ok(());
    }
    let p = cx.profile.fields();
    let interpolation = p.fragment.sample_rate_shading
        && !p
            .enabled
            .portability_withholds(VkFeature::ShaderSampleRateInterpolationFunctions);
    // A driver that runs a memory writer once per covered sample hands each run the pixel's
    // coverage, unless its shader compiler masks the coverage with the run's sample bit: MoltenVK
    // 1.4.1's SPIRV-Cross (`adec7acb`) does where the module declares `SampleRateShading` (it
    // reads the sample index or a sample-qualified varying) and reads `FragCoord`.
    let position = used(&|r| matches!(r, ParamRole::Builtin(Builtin::Position(_))));
    let driver_runs = writes
        && p.workarounds
            .contains(&DriverWorkaround::MemoryWritingFragmentsRunPerSample)
        && !((sample_id || sample_varying) && position);
    let attachments = super::rog::declared(e)
        || used(&|r| matches!(r, ParamRole::ColorInput { .. } | ParamRole::Imageblock(_)));
    let refusal = if matches!(
        e.ep.stage,
        AirStage::Fragment {
            early_fragment_tests: true
        }
    ) {
        Some(CoverageLoopRefusal::EarlyFragmentTests)
    } else if attachments {
        Some(CoverageLoopRefusal::Attachments)
    } else if sample_varying && !interpolation && !driver_runs {
        Some(CoverageLoopRefusal::InterpolationFunctions)
    } else {
        None
    };
    let looped = cx.inputs.multisampled && refusal.is_none() && !driver_runs;
    // Each pass's outputs land on every sample the passes keep, the last pass's (the module docs).
    if looped && !e.ep.outputs.is_empty() {
        cx.precision.insert(PrecisionNote::LoopedSampleOutputs);
    }
    let whole_quad = p
        .workarounds
        .contains(&DriverWorkaround::InterpolationNeedsTheWholeQuad);
    let samples_before_loop =
        (looped && sample_varying && whole_quad).then(|| most_samples(&p.limits));
    if samples_before_loop.is_some() {
        // The entry reads the varyings at the pipeline's samples, by its count.
        cx.facts.insert(RuntimeFact::SampleCount);
    }
    cx.families.fragment.coverage = Some(CoveragePlan {
        looped,
        refusal,
        writes,
        driver_runs,
        samples_before_loop,
    });
    Ok(())
}

/// The most samples a pipeline of a device with `limits` may rasterize: the largest of the
/// framebuffer sample counts it supports, with attachments of any kind or none.
fn most_samples(limits: &crate::device::Limits) -> u32 {
    let counts = [
        limits.framebuffer_color_sample_counts,
        limits.framebuffer_integer_color_sample_counts,
        limits.framebuffer_depth_sample_counts,
        limits.framebuffer_stencil_sample_counts,
        limits.framebuffer_no_attachments_sample_counts,
    ]
    .iter()
    .fold(0u32, |all, c| all | c.as_raw());
    match counts.checked_ilog2() {
        Some(top) => 1 << top,
        None => 1,
    }
}

/// Complete the plan once every instruction is selected (`fragment::finish`): a discard, or a
/// derivative, implicit-LOD sample, quad or simdgroup operation, or barrier, refuses the loop. A
/// looping function is refused here; any other records in its interface how link treats it
/// (`StageIo::coverage_loop`). On a driver that runs such a shader once per covered sample
/// (`MemoryWritingFragmentsRunPerSample`) a function that writes memory runs as written, or, where
/// that driver would mask its coverage, is refused as its defect: the loop would repeat each pass
/// per covered sample there.
pub(super) fn finish(cx: &mut SelectCx<'_>) -> Result<bool, Error> {
    let Some(plan) = cx.families.fragment.coverage else {
        return Ok(false);
    };
    let late = if !cx.families.fragment.discards.is_empty() {
        Some(CoverageLoopRefusal::Discard)
    } else if cx
        .body
        .insts
        .iter()
        .any(|(id, inst)| super::discard::quad_operation(cx, id, inst))
    {
        Some(CoverageLoopRefusal::QuadOperations)
    } else {
        None
    };
    if plan.looped {
        if let Some(r) = late {
            return Err(r.error());
        }
        let per_sample = DriverWorkaround::MemoryWritingFragmentsRunPerSample;
        if plan.writes && cx.profile.fields().workarounds.contains(&per_sample) {
            return Err(
                Unsupported::driver_defect(Feature::PixelCoverageAtSampleRate, per_sample).into(),
            );
        }
    }
    cx.families.stage_io.interface.coverage_loop = match (plan.looped, plan.refusal.or(late)) {
        (true, _) => CoverageLoop::Looped,
        (false, None) if plan.driver_runs => CoverageLoop::DriverRunsPerSample,
        (false, None) => CoverageLoop::Unlooped,
        (false, Some(r)) => CoverageLoop::Refused(r),
    };
    Ok(plan.looped)
}

/// Declare the `Private u32` the loop hands each pass its sample index in.
pub(super) fn declare(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<Option<GlobalId>> {
    if !plan.families.fragment.coverage.is_some_and(|c| c.looped) {
        return Ok(None);
    }
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let zero = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(0),
    })?;
    mb.plain_global(StorageClass::Private, u32_, Some(zero))
        .map(Some)
}

/// The entry of a looping function ([module docs](self)): a new function that runs `body` once
/// per set bit of `SampleMask[0]`, the pixel's coverage, with that bit's index in `loop_sample`.
pub(super) fn wrap(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    decl: &FamilyDecls,
    body: FuncId,
) -> Result<FuncId, Error> {
    let Some(sample) = decl.fragment.loop_sample else {
        return Err(invariant(
            "a looping fragment function without its loop sample",
        ));
    };
    let void = mb.ty(Type::Void)?;
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let mut fb = mb.function(void)?;
    // The coverage, also stored in the copy the body reads it from.
    let covered = super::load_coverage(&mut fb, plan, decl)?
        .ok_or_else(|| invariant("a looping fragment function without its coverage input"))?;
    // Where the driver's interpolation needs the whole quad, the sample-qualified varyings at
    // every sample, before the quad's helpers leave the loop.
    crate::lower::stage_io::read_at_every_sample(&mut fb, plan, decl)?;
    let kept = match super::sample_mask_output(plan) {
        Some(i) => Some(decl.stage_io.globals.get(i).copied().ok_or_else(|| {
            invariant("a looping fragment function's sample mask output without its variable")
        })?),
        None => None,
    };
    emit_loop(&mut fb, u32_, covered, sample, body, kept)?;
    Ok(fb.finish()?)
}

/// `rest = covered; loop { if rest == 0 { break } sample = FindILsb(rest); call body;
/// rest &= rest - 1 }; return`, in `fb`'s open block, `covered` the loaded `SampleMask[0]`; where
/// the stage writes `SampleMask` (`out`, its variable), each pass's word ANDed with its sample's
/// bit is ORed into the word stored there once the loop ends.
fn emit_loop(
    fb: &mut FunctionBuilder<'_>,
    u32_: crate::spirv::ids::TypeId,
    covered: crate::spirv::ids::Value,
    sample: GlobalId,
    body: FuncId,
    out: Option<GlobalId>,
) -> BResult<()> {
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    let rest = fb.local_var(u32_, None)?;
    let rest = fb.var_ptr(rest)?;
    fb.store(rest, covered, MemAccess::default())?;
    // The output word and the union of the samples the passes keep.
    let out = match out {
        Some(g) => {
            let base = fb.module().global_value(g)?;
            let word = fb.access_chain(base, &[zero])?;
            let kept = fb.local_var(u32_, None)?;
            let kept = fb.var_ptr(kept)?;
            fb.store(kept, zero, MemAccess::default())?;
            Some((word, kept))
        }
        None => None,
    };
    let sample = fb.module().global_value(sample)?;
    fb.loop_(
        &mut (),
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                let left = fb.load(rest, MemAccess::default())?;
                let done = fb.binary(BinOp::IEqual, left, zero)?;
                fb.if_(
                    ctx,
                    done,
                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let s = fb.glsl(GlslOp::FindILsb, &[left])?;
                fb.store(sample, s, MemAccess::default())?;
                fb.call(body, &[])?;
                let below = fb.binary(BinOp::ISub, left, one)?;
                if let Some((word, kept)) = out {
                    let bit = fb.binary(BinOp::ShiftLeftLogical, one, s)?;
                    let written = fb.load(word, MemAccess::default())?;
                    let written = fb.binary(BinOp::BitwiseAnd, written, bit)?;
                    let union = fb.load(kept, MemAccess::default())?;
                    let union = fb.binary(BinOp::BitwiseOr, union, written)?;
                    fb.store(kept, union, MemAccess::default())?;
                }
                let next = fb.binary(BinOp::BitwiseAnd, left, below)?;
                fb.store(rest, next, MemAccess::default())
            },
        ),
    )?;
    if let Some((word, kept)) = out {
        let union = fb.load(kept, MemAccess::default())?;
        fb.store(word, union, MemAccess::default())?;
    }
    fb.ret(None)
}
