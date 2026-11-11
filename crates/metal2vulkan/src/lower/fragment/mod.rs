//! `lower::fragment`: discard, fragment-only intrinsics, raster order groups, framebuffer fetch.
//!
//! - **Discard** (`air.discard_fragment`): each site's form is decided in [`finish`] by
//!   `discard::plan` (demote, predication, or a terminator, from the language version, forward
//!   reachability, and the profile). A predicated site sets the `Private bool discarded` this
//!   family declares, and [`emit_return`] kills a discarded invocation before `OpReturn`.
//! - **Framebuffer fetch** (`[[color(n)]]` inputs): each color index is one input attachment
//!   (`color`), read at every use; on a device that orders the read only through the pixel
//!   interlock, a storage image of the attachment read and written inside the interlock where the
//!   pipeline's state lets that be exact (`fetch`). Compiled for a multisampled pipeline, either
//!   form reads (and writes) the invocation's own sample at `SampleId`, which runs the function
//!   once per covered sample, as Metal does. On a device that orders the read through
//!   rasterization-order input attachments, a blend into an extended-range attachment is computed
//!   over its input attachment (`input_blend`).
//! - **Raster order groups** (`rog`): a function that names a group runs its body as a function
//!   called from `main` between `Begin/EndInvocationInterlockEXT` ([`wrap_entry`], the
//!   entry-wrapper hook of `lower::emit`); inside it a discard only records `discarded`, and `main`
//!   kills after the critical section.
//! - **Imageblocks** (`imageblock`): an `[[imageblock_data]]` input is read from the pass's
//!   imageblock storage image at the pixel in the function's first block ([`emit_entry`]), an
//!   output written at the return ([`emit_return`]), both inside the interlock critical section,
//!   which a function that reaches the imageblock always has.
//! - **Sample count** (`get_num_samples`): Metal's header spells `get_num_samples()` as the call
//!   with operand 0, the pipeline's rasterization sample count. It reads the `SpecKey::SampleCount`
//!   specialization constant (`RuntimeFact::SampleCount`, default 1), which the binder sets to the
//!   pipeline's `rasterizationSamples`. A call with another operand is refused: no other operand's
//!   meaning is established.
//! - **Shader-side stencil** (`stencil`): on a device without stencil export, a `[[stencil]]`
//!   output is tested and written by the function itself against a shadow of the stencil, inside
//!   the interlock critical section, at each return; a failing fragment is recorded as discarded
//!   and killed after the critical section.
//! - **The depth test in the shader** (`depth_test`): a function that stores a color in the storage
//!   form before its draws' tests, compiled for a draw whose depth test alone can drop a fragment,
//!   tests and writes its depth itself against a stand-in of the depth attachment, inside the
//!   interlock critical section, at each return; a failing fragment is recorded as discarded and
//!   killed after the critical section. Compiled for a draw whose stencil test acts too, it runs
//!   that first, against a stand-in of the stencil attachment.
//! - **Coverage at sample rate** (`coverage`): a function that reads `[[sample_mask]]` and that
//!   Metal runs once per covered sample runs, compiled for a multisampled pipeline, once per pixel,
//!   and its body once per covered sample from a loop over the pixel's coverage ([`wrap_entry`]),
//!   where its side effects are its memory writes alone.
//! - **The coverage in a body function**: where the entry wraps the body (an interlock or the
//!   coverage loop) and the body reads the `SampleMask` input, the entry stores the input's word in
//!   a `Private u32` before it calls the body, and the body reads that copy
//!   (`stage_io::coverage_in`). SPIRV-Cross (MoltenVK) declares the input a scalar in the entry
//!   but passes it to a function as an array, which Metal refuses.
//! - **Snapshot reads** (`snapshot`): a function compiled to read textures through snapshots
//!   (`LoweringInputs::snapshot_textures`) opens with an ordered pixel interlock's critical section
//!   in which, in the draw that takes the snapshot, each non-helper fragment copies its pixel's
//!   texel, and that draw returns after it; in the draw that reads it, each read of the slot reads
//!   the snapshot, and a helper reads 0. The body runs outside the critical section.
//! - **Pull-model interpolation** (`interpolate`): each `interpolate_at_*` call of an
//!   `interpolant` reads the interpolant's variable where it says (`stage_io::interpolate`); the
//!   interpolant itself is planned by `lower::stage_io`.
//! - Refused: the `uint2` overloads of `map_screen_to_physical_coordinates` and its inverse
//!   (canon compiles the `float2` ones to the rate map's decoder; no reading of the
//!   physical-to-screen `uint2` one is established).

mod color;
mod coverage;
mod depth_test;
mod discard;
pub(crate) mod fetch;
mod imageblock;
mod input_blend;
mod input_shadow;
mod interpolate;
mod logic;
mod rog;
mod snapshot;
mod stencil;
mod word_input;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, ParamBinding, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, Fragment};
use crate::air::ir::{Callee, Const, Inst, InstId, InstKind, Value};
use crate::budget::Meter;
use crate::device::profile::DriverWorkaround;
use crate::device::subgroup::OpBits;
use crate::error::{Attachment, Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::legal::{form_on, Form};
use crate::lower::arith::ops;
use crate::lower::emit::{EmitCx, FamilyDecls};
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::stage_io;
use crate::lower::strategy::FamilyStrategy;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{SpecKey, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, KillForm};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ExecMode, IoDir, MemAccess, QuadDirection, StorageClass,
    Type as SType, UnOp,
};

pub use color::ColorInput;
pub use coverage::CoveragePlan;
pub use depth_test::DepthTest;
pub(crate) use depth_test::{
    depth_test_form, depth_test_word, stencil_test_form, stencil_test_word, STAND_IN_FORMAT,
};
pub use discard::DiscardForm;
pub(crate) use fetch::{
    compiled_write_mask, extended_range_blend, half_float_format, inputs_order, interlock_orders,
    kept_channels, storage_form, ColorUse, StorageSamples,
};
pub use fetch::{
    select_output as select_storage_output, select_second_output as select_storage_second_output,
    StorageColor,
};
pub use imageblock::{select_output as select_imageblock_output, ImageblockPlan};
pub(crate) use input_blend::{rotated_texel as rotated_input_texel, texel as input_blend_texel};
pub use input_blend::{
    select_second_output as select_input_second_output, InputBlend, RotatedInputBlend,
};
pub use input_shadow::InputShadow;
pub(crate) use input_shadow::{shadows as input_shadow, store as input_shadow_store};
pub use logic::LogicInput;
pub(crate) use logic::{channels as logic_channels, input_operated as logic_input_operated};
#[cfg(test)]
pub(crate) use snapshot::{decode as snapshot_decode, encode as snapshot_encode};
pub(crate) use snapshot::{prepare as snapshot_prepare, read as snapshot_read};
pub(crate) use stencil::{face_word, SHADOW_FORMAT};
pub use stencil::{select_output as select_stencil_output, ShadowStencil};
pub(crate) use word_input::merge as word_input_merge;

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Fragment)).into()
}

/// The fragment strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentStrategy {
    /// `air.discard_fragment`; its form is the plan's (`FragmentPlan::discards`), decided once
    /// every instruction is selected.
    Discard,
    /// `get_num_samples()`: the `SpecKey::SampleCount` constant in the result's form.
    NumSamples(Form),
    /// `quad_is_helper_thread()` and `simd_is_helper_thread()`: in a fragment function
    /// `OpIsHelperInvocationEXT` where a discard demotes, else the `HelperInvocation` input, or a
    /// predicated discard's `discarded`; `false` in every other stage, as Metal answers there.
    IsHelperThread,
    /// An interpolation function of an `interpolant` (`stage_io::interpolate`).
    Interpolate(crate::lower::stage_io::InterpolantRead),
}

impl FamilyStrategy for FragmentStrategy {
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            FragmentStrategy::Discard if plans.fragment.deferred_discards => {
                "fragment.discard.deferred"
            }
            FragmentStrategy::Discard => "fragment.discard",
            FragmentStrategy::NumSamples(_) => "fragment.num_samples",
            FragmentStrategy::IsHelperThread => "fragment.is_helper_thread",
            FragmentStrategy::Interpolate(_) => "fragment.interpolate",
        })
    }

    /// The largest form: a selection construct around a kill (header branch, merge, and the kill
    /// block), or the store of the predicate; the sample count's specialization constant, its
    /// `SpecId` decoration, and a mask; the helper input's variable, decoration and interface
    /// entry, its load, the predicate's load and the `or`.
    fn words(&self) -> u32 {
        match self {
            FragmentStrategy::Discard => 24,
            FragmentStrategy::NumSamples(_) => 16,
            FragmentStrategy::IsHelperThread => 32,
            // An offset's move and clamp, and their constants, or a sample's mask; the read is
            // the parameter's use (`stage_io`).
            FragmentStrategy::Interpolate(_) => 32,
        }
    }
}

/// The fragment family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FragmentPlan {
    /// Each discard site's form, by instruction; filled by [`finish`].
    pub discards: BTreeMap<InstId, DiscardForm>,
    /// The non-quad subgroup operations before which a quad whose every invocation is discarded
    /// or a helper exits, as Metal drops such a quad from the simdgroup: every site is then
    /// predicated, or, where a side effect follows one, every site demotes. Filled by [`finish`].
    pub quad_exits: BTreeSet<InstId>,
    /// The sites that would discard at once are predicated against the driver's store-dropping
    /// defect (`DriverWorkaround::DiscardDropsSimdgroupStores`), so the census names the
    /// function's discards `fragment.discard.deferred`. Filled by [`finish`].
    pub deferred_discards: bool,
    /// The stores, atomics, image writes and copies a predicated site reaches: each runs only
    /// where the invocation was not discarded ([`not_discarded`]). Filled by [`finish`].
    pub guarded: BTreeSet<InstId>,
    /// The profile has `shaderTerminateInvocation`: kills are `OpTerminateInvocation`.
    pub terminate_invocation: bool,
    /// The function asks whether its thread is a helper (`IsHelperThread`).
    pub helper_queries: bool,
    /// Every used color input (framebuffer fetch), in parameter order.
    pub colors: Vec<ColorInput>,
    /// The function is in a raster order group: its body runs inside this interlock mode's
    /// critical section, called from the entry [`wrap_entry`] builds.
    pub interlock: Option<ExecMode>,
    /// The function's views of the render pass's imageblock.
    pub imageblock: Option<ImageblockPlan>,
    /// The color attachments the function reads in the interlocked storage form, by index
    /// (`fetch`).
    pub fetch: BTreeMap<u32, StorageColor>,
    /// The extended-range attachments whose blend the function computes over their input
    /// attachments, on a device that orders fetch through them (`input_blend`), by index.
    pub input_blends: BTreeMap<u32, InputBlend>,
    /// The attachments held with their channels rotated whose blend the function computes over
    /// their input attachments, on a device that orders fetch through them (`input_blend`), by
    /// index.
    pub rotated_blends: BTreeMap<u32, RotatedInputBlend>,
    /// The word attachments whose word the function merges into the texel's over their input
    /// attachments, on a device that orders fetch through them (`word_input`), by index: the bits
    /// of the fields each write keeps from the texel.
    pub word_merges: BTreeMap<u32, u32>,
    /// The attachments whose private logic operation the function computes over their input
    /// attachments, on a device that orders fetch through them (`logic`), by index.
    pub logic_inputs: BTreeMap<u32, LogicInput>,
    /// The extended-range attachments whose `half` shadow the draw's instance holds, on a device
    /// that orders fetch through input attachments (`input_shadow`), by index.
    pub input_shadows: BTreeMap<u32, InputShadow>,
    /// The stage interface's `FragCoord` object, which gives the pixel of the storage form and
    /// of the shader-side stencil.
    pub fetch_coord: Option<usize>,
    /// The module declares early fragment tests for the storage form, which the function itself
    /// does not.
    pub fetch_early_tests: bool,
    /// The stage interface's `SampleId` object, at which a function compiled for a multisampled
    /// pipeline (`LoweringInputs::multisampled`) reads each color it fetches, in either form, and
    /// writes each storage-form color: Metal runs such a function once per covered sample and
    /// reads its own sample.
    pub fetch_sample: Option<usize>,
    /// The field of the returned value holding the function's `[[sample_mask]]` output, which
    /// gates each storage-form write of a per-sample function on the invocation's own sample bit,
    /// and of a function at one sample on bit 0.
    pub fetch_mask: Option<u32>,
    /// The function's `[[stencil]]` output is a shader-side stencil (`stencil`).
    pub stencil: Option<ShadowStencil>,
    /// The function runs its draw's depth test itself against a stand-in of the depth
    /// attachment (`depth_test`).
    pub depth_test: Option<DepthTest>,
    /// The function reads `[[sample_mask]]` and runs once per covered sample on Metal: its loop
    /// over the covered samples, or why it has none (`coverage`).
    pub coverage: Option<CoveragePlan>,
    /// The function returns a `[[depth(...)]]` output (`StageIoPlan::depth_clamp`). Inside an
    /// interlocked body each return records a positive NaN depth as discarded.
    pub depth_output: bool,
    /// The texture slots the function reads through a snapshot (`snapshot`,
    /// `LoweringInputs::snapshot_textures`), by Metal index, each with its snapshot's form.
    /// Filled by [`finish`].
    pub snapshot: BTreeMap<u32, crate::pipeline::sym::SnapshotForm>,
}

impl FragmentPlan {
    /// Some site is predicated, so the entry declares `discarded` and kills at its returns.
    pub fn predicated(&self) -> bool {
        self.discards.values().any(|f| *f == DiscardForm::Predicate)
    }

    /// Some site demotes, so a helper query is `OpIsHelperInvocationEXT`: the `HelperInvocation`
    /// input need not say a demoted invocation is a helper.
    fn demotes(&self) -> bool {
        self.discards.values().any(|f| *f == DiscardForm::Demote)
    }

    /// The function reads the `HelperInvocation` input: before a whole-quad exit of predicated
    /// sites, or for a helper query where no site demotes.
    fn reads_helper_input(&self) -> bool {
        (self.predicated() && !self.quad_exits.is_empty())
            || ((self.helper_queries || !self.snapshot.is_empty()) && !self.demotes())
    }

    /// The function declares `discarded`: a predicated site, inside an interlocked body a
    /// terminate-form site (which records the discard and returns to the entry) or a depth output
    /// (whose returns record a positive NaN), a shader-side stencil, or an in-shader depth test,
    /// which records a failing fragment.
    pub fn records_discards(&self) -> bool {
        self.predicated()
            || (self.interlock.is_some()
                && (self.depth_output
                    || self.discards.values().any(|f| *f == DiscardForm::Terminate)))
            || self.stencil.is_some()
            || self.depth_test.is_some()
    }

    fn kill_form(&self) -> KillForm {
        if self.terminate_invocation {
            KillForm::TerminateInvocation
        } else {
            KillForm::Kill
        }
    }

    /// How a return of the entry kills an invocation whose outputs Metal drops, where it can:
    /// `None` inside an interlocked body, whose caller kills after the critical section, and in a
    /// function looping over its covered samples, where a kill would drop every sample's outputs.
    pub(crate) fn return_kill(&self) -> Option<KillForm> {
        let looped = self.coverage.is_some_and(|c| c.looped);
        (self.interlock.is_none() && !looped).then(|| self.kill_form())
    }
}

/// An upper estimate of the words `if discarded { kill }` adds at one return.
const RETURN_KILL_WORDS: u64 = 16;
/// An upper estimate of the words one whole-quad exit adds: two loads, an `or`, three quad swaps
/// and their direction constants, three `and`s, and `if all { kill }`.
const QUAD_EXIT_WORDS: u64 = 48;
/// An upper estimate of the words the interlocked entry adds around the body function: its
/// function type, the entry function, `Begin`, the call, `End`, the kill, and the return.
const WRAPPER_WORDS: u64 = 48;
/// An upper estimate of the words the guard of one side effect adds ([`not_discarded`]): the load of
/// `discarded` and its negation, the selection's merge, branch, three labels and two branches,
/// and for a result its phi and zero.
const GUARD_WORDS: u64 = 40;

/// Whether the function runs a quad or simdgroup operation whose answer depends on which of
/// its group's lanes are active (`GroupOp::reads_active_lanes`).
fn counts_active_lanes(cx: &SelectCx<'_>) -> bool {
    cx.body.insts.values().any(|inst| match &inst.kind {
        InstKind::Call(call) => matches!(
            &call.callee,
            Callee::Air {
                op: AirIntrinsic::Group(g),
                ..
            } if g.op.reads_active_lanes()
        ),
        _ => false,
    })
}

/// How many returns the function's body has.
fn returns(cx: &SelectCx<'_>) -> u64 {
    cx.body
        .insts
        .values()
        .filter(|i| matches!(i.kind, InstKind::Ret(_)))
        .count() as u64
}

/// Complete the fragment family's plan once every instruction and parameter is selected: the
/// interlock of a function in a raster order group, the form of each discard site, and the words
/// the input attachments, the predicated returns, and the interlocked entry add to the stage
/// interface's size estimate.
pub fn finish(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    // On a device that shades a triangle it clips as several, a quad a seam crosses runs once
    // per piece: an operation over the active lanes sees each piece's.
    if cx
        .profile
        .fields()
        .workarounds
        .contains(&DriverWorkaround::ClippedTrianglesSplitPixelFragments)
        && counts_active_lanes(cx)
    {
        cx.precision.insert(PrecisionNote::QuadsSplitByClipping);
    }
    let looped = coverage::finish(cx)?;
    imageblock::finish(cx)?;
    fetch::finish(cx)?;
    stencil::finish(cx)?;
    // An imageblock, an attachment in the storage form, the shadow of a shader-side stencil and
    // an extended-range attachment's shadow over its input attachment are read and written inside
    // the critical section whether or not the function names a raster order group.
    let shadow = cx.families.fragment.stencil.is_some();
    let input_shadows = !cx.families.fragment.input_shadows.is_empty();
    // A function looping over its covered samples runs no interlocked body (`wrap_entry`).
    if input_shadows && looped {
        return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into());
    }
    let storage = !cx.families.fragment.fetch.is_empty() || shadow || input_shadows;
    let interlock = if rog::declared(cx) || cx.families.fragment.imageblock.is_some() || storage {
        Some(rog::mode(cx)?)
    } else {
        None
    };
    let snapshot = snapshot::finish(cx, interlock.is_some(), looped)?;
    let sites: Vec<InstId> = cx.families.fragment.discards.keys().copied().collect();
    let discard::DiscardPlan {
        forms,
        quad_exits,
        counted_before,
        deferred,
        exposed,
        guarded,
    } = discard::plan(cx, &sites, interlock.is_some(), m)?;
    if counted_before {
        cx.precision
            .insert(PrecisionNote::QuadsCountedBeforeDiscard);
    }
    if exposed {
        cx.precision
            .insert(PrecisionNote::DiscardDropsSimdgroupStores);
    }
    if !quad_exits.is_empty() {
        // The exits' quad swaps are quad operations of the simd family's class.
        let simd = &mut cx.families.simd;
        simd.direct_ops = simd.direct_ops.union(OpBits::QUAD);
        simd.cluster_ops = simd.cluster_ops.union(OpBits::QUAD);
        simd.groups = true;
    }
    // Inside an interlocked body the returns do not kill: `main` kills once, after `End`.
    let predicated = interlock.is_none() && forms.values().any(|f| *f == DiscardForm::Predicate);
    // The returns, counted only where something is emitted at each of them.
    let rets = if predicated || cx.families.fragment.imageblock.is_some() || storage {
        returns(cx)
    } else {
        0
    };
    let killing_rets = if predicated { rets } else { 0 };
    let wrapper = if interlock.is_some() {
        WRAPPER_WORDS
    } else if looped {
        coverage::LOOP_WORDS
    } else {
        0
    };
    // The imageblock output is written at every return.
    let imageblock = cx
        .families
        .fragment
        .imageblock
        .as_ref()
        .map_or(0, |p| p.words_estimate(rets));
    let words = color::words(cx, &cx.families.fragment.colors)
        .saturating_add(fetch::words(
            cx,
            color::uses(cx, &cx.families.fragment.colors),
            rets,
        ))
        .saturating_add(stencil::words(cx, rets))
        .saturating_add(input_blend::words(cx, returns(cx)))
        .saturating_add(input_shadow::words(
            cx,
            color::uses(cx, &cx.families.fragment.colors),
            returns(cx),
        ))
        .saturating_add(word_input::words(cx, returns(cx)))
        .saturating_add(logic::words(cx, returns(cx)))
        .saturating_add(depth_test::words(cx, rets))
        .saturating_add(killing_rets.saturating_mul(RETURN_KILL_WORDS))
        .saturating_add((quad_exits.len() as u64).saturating_mul(QUAD_EXIT_WORDS))
        .saturating_add((guarded.len() as u64).saturating_mul(GUARD_WORDS))
        .saturating_add(wrapper)
        .saturating_add(imageblock);
    stage_io::add_interface_words(cx, words);
    // The requirement is the feature alone: a Vulkan 1.3 device has it in core, and below 1.3
    // the description lists it only with its extension.
    let terminate = cx.profile.fields().fragment.terminate_invocation;
    let fp = &mut cx.families.fragment;
    fp.discards = forms;
    fp.quad_exits = quad_exits;
    fp.deferred_discards = deferred;
    fp.guarded = guarded;
    fp.terminate_invocation = terminate;
    fp.interlock = interlock;
    fp.snapshot = snapshot;
    Ok(())
}

/// The fragment family's module-scope objects.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The `Private bool discarded` of predicated discards, and of every discard inside an
    /// interlocked body.
    pub discarded: Option<GlobalId>,
    /// The `HelperInvocation` input the whole-quad exits and helper queries where no discard
    /// demotes read.
    pub helper: Option<GlobalId>,
    /// The input attachment of each color index read, or its storage image in the interlocked
    /// storage form.
    pub attachments: BTreeMap<u32, GlobalId>,
    /// The render pass's imageblock storage image.
    pub imageblock: Option<GlobalId>,
    /// The shadow of a shader-side stencil.
    pub shadow_stencil: Option<GlobalId>,
    /// The stand-in of the depth attachment an in-shader depth test reads and writes.
    pub depth_stand_in: Option<GlobalId>,
    /// The stand-in of the stencil attachment an in-shader stencil test reads and writes.
    pub stencil_stand_in: Option<GlobalId>,
    /// The `half` shadow of each extended-range attachment the storage form reads and writes
    /// against it (`fetch::StorageColor::shadowed`), or that a function on a device that fetches
    /// through input attachments reads and writes beside them (`input_shadow`), by color index.
    pub extended_shadows: BTreeMap<u32, GlobalId>,
    /// The `Private u32` holding the sample of the loop's pass over the covered samples.
    pub loop_sample: Option<GlobalId>,
    /// The `Private u32` copy of the `SampleMask` input's word, which the entry stores before it
    /// calls the body it wraps, and the body reads for every read of the input.
    pub coverage_copy: Option<GlobalId>,
    /// The snapshots the function reads textures through, and the helper's copy.
    pub snapshot: snapshot::Declared,
}

/// Declare the fragment family's module-scope objects and modes: the discard predicate, the
/// input attachments, and the interlock mode of a function in a raster order group.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let fp = &plan.families.fragment;
    let mut d = Declared::default();
    if let Some(m) = fp.interlock.or(snapshot::mode(&fp.snapshot)) {
        modes.push(m);
    }
    // The storage form's writes, and the imageblock's, are Metal's under early tests.
    let imageblock_early = fp.imageblock.as_ref().is_some_and(|p| p.early_tests);
    if fp.fetch_early_tests || imageblock_early {
        modes.push(ExecMode::EarlyFragmentTests);
    }
    // The function's own early tests, where the imageblock's write tests late instead.
    if fp.imageblock.as_ref().is_some_and(|p| p.late_tests) {
        modes.retain(|m| *m != ExecMode::EarlyFragmentTests);
    }
    if fp.records_discards() {
        let bool_ = mb.ty(SType::Bool)?;
        let no = mb.constant(Constant::Scalar {
            ty: bool_,
            bits: ConstBits(0),
        })?;
        d.discarded = Some(mb.plain_global(StorageClass::Private, bool_, Some(no))?);
    }
    if fp.reads_helper_input() {
        let bool_ = mb.ty(SType::Bool)?;
        d.helper = Some(mb.builtin(crate::spv::BuiltIn::HelperInvocation, IoDir::Input, bool_)?);
    }
    d.attachments = color::declare(
        mb,
        &fp.colors,
        (&fp.fetch, &fp.input_blends, &fp.word_merges),
        (&fp.rotated_blends, &fp.logic_inputs),
        fp.fetch_sample.is_some(),
    )?;
    for (&color, sc) in &fp.fetch {
        if sc.shadowed {
            d.extended_shadows
                .insert(color, fetch::declare_shadow(mb, color, sc.view.samples)?);
        }
    }
    for (&color, shadow) in &fp.input_shadows {
        d.extended_shadows
            .insert(color, fetch::declare_shadow(mb, color, shadow.samples)?);
    }
    d.imageblock = imageblock::declare(mb, fp.imageblock.as_ref())?;
    d.shadow_stencil = stencil::declare(mb, fp.stencil.as_ref())?;
    d.depth_stand_in = depth_test::declare(mb, fp.depth_test.as_ref())?;
    d.stencil_stand_in = depth_test::declare_stencil(mb, fp.depth_test.as_ref())?;
    d.loop_sample = coverage::declare(mb, plan)?;
    d.snapshot = snapshot::declare(mb, &fp.snapshot)?;
    if wraps_body(fp) && sample_mask_input(plan).is_some() {
        let u32_ = mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let zero = mb.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(0),
        })?;
        d.coverage_copy = Some(mb.plain_global(StorageClass::Private, u32_, Some(zero))?);
    }
    Ok(d)
}

/// Whether the entry is a new function that calls the body ([`wrap_entry`]): an interlocked
/// body, or one looping over the covered samples.
fn wraps_body(fp: &FragmentPlan) -> bool {
    fp.interlock.is_some() || fp.coverage.is_some_and(|c| c.looped)
}

/// The index of the stage interface's `SampleMask` input object, if the function reads it.
pub(crate) fn sample_mask_input(plan: &Plan) -> Option<usize> {
    plan.families.stage_io.objects.iter().position(|o| {
        matches!(
            o,
            stage_io::IoObject::Builtin {
                builtin: crate::spv::BuiltIn::SampleMask,
                dir: IoDir::Input,
                ..
            }
        )
    })
}

/// The index of the stage's `SampleMask` output object, where it has one: the function's sample
/// mask output, or the one the pipeline's private masks write.
pub(super) fn sample_mask_output(plan: &Plan) -> Option<usize> {
    plan.families.stage_io.objects.iter().position(|o| {
        matches!(
            o,
            stage_io::IoObject::Builtin {
                builtin: crate::spv::BuiltIn::SampleMask,
                dir: IoDir::Output,
                ..
            }
        )
    })
}

/// `SampleMask[0]`, the invocation's coverage word, loaded from the input variable in the open
/// block of the entry `fb` builds, and stored in the body's copy where it has one.
pub(super) fn load_coverage(
    fb: &mut FunctionBuilder<'_>,
    plan: &Plan,
    decls: &FamilyDecls,
) -> Result<Option<SValue>, Error> {
    let Some(i) = sample_mask_input(plan) else {
        return Ok(None);
    };
    let Some(g) = decls.stage_io.globals.get(i).copied() else {
        return Err(Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::Invariant,
            "the coverage input without its variable",
        )));
    };
    let base = fb.module().global_value(g)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let word = fb.access_chain(base, &[zero])?;
    let covered = fb.load(word, MemAccess::default())?;
    if let Some(copy) = decls.fragment.coverage_copy {
        let p = fb.module().global_value(copy)?;
        fb.store(p, covered, MemAccess::default())?;
    }
    Ok(Some(covered))
}

/// Plan a used parameter this family owns (`select::param_family`): a color input, or an
/// imageblock input.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    if let ParamRole::Imageblock(ib) = &b.role {
        return imageblock::select_input(cx, b.param, ib, b.ty.align);
    }
    let c = color::select_param(cx, b).map_err(|e| match b.role {
        ParamRole::ColorInput {
            index: AbiInt::Const(color),
            ..
        } => e.at_attachment(Attachment::Color(color)),
        _ => e,
    })?;
    cx.families.fragment.colors.push(c);
    Ok(())
}

/// Plan the entry's framebuffer-fetch forms before its stage interface: an output to an attachment
/// in the storage form is no interface variable. Then the loop over the covered samples, which
/// decides how the stage interface reads `[[sample_id]]` and the sample-qualified varyings.
pub fn plan_entry(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    fetch::plan(cx)?;
    input_blend::plan(cx)?;
    input_shadow::plan(cx)?;
    word_input::plan(cx)?;
    logic::plan(cx)?;
    coverage::plan(cx)
}

/// Whether the function's body runs once per covered sample from the entry's loop: its
/// `[[sample_id]]` is the pass's sample, and a sample-qualified varying is interpolated there.
pub fn loops_samples(cx: &SelectCx<'_>) -> bool {
    cx.families.fragment.coverage.is_some_and(|c| c.looped)
}

/// Whether the function reads color attachment `color` in the interlocked storage form, so an
/// output to it is written there, not to an interface variable.
pub fn writes_through_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    fetch::is_storage(cx, color)
}

/// Whether the function writes color attachment `color` in the storage form under Metal's
/// 16-bit float, 8-bit unorm or 8-bit sRGB blend, which the shader computes, and that blend is
/// Metal's exactly: the attachment stores Metal's value.
pub fn blends_half_in_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    fetch::blends_in_shader(cx, color, true)
}

/// Whether the function writes color attachment `color` in the storage form under a write mask
/// that names none of its format's channels: the write stores nothing.
pub fn stores_nothing_in_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    (cx.families.fragment.fetch.get(&color)).is_some_and(|sc| sc.writes && sc.view.stores_nothing())
}

/// Whether the function writes (or reads) color attachment `color` in the storage form, which the
/// pipeline's color attachments do not hold.
pub fn written_in_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    cx.families.fragment.fetch.contains_key(&color)
}

/// Whether the function writes color attachment `color` in the storage form under Metal's
/// 16-bit float blend, which the shader computes, exactly or not: the write stores a `half` in
/// every channel, whose conversion by the device is exact.
pub fn blends_sixteen_in_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    fetch::blends_half(cx, color)
}

/// Whether the function writes color attachment `color` in the storage form under a blend the
/// shader computes of a format held directly (Metal's 16-bit float, 8-bit unorm or 8-bit sRGB
/// blend), exactly or not: the write stores the value the blend gives, which no conversion of the
/// device's changes (an 8-bit unorm channel's code is rounded by Metal's rule in the shader, an
/// sRGB one's encoded by Metal's table).
pub fn blends_in_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    fetch::blends_in_shader(cx, color, false)
}

/// Whether the function's second (index-1) output to `color` is the second source of the
/// extended-range blend the shader computes in the storage form, outside the stage interface.
pub fn blends_second_through_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    fetch::blends_second(cx, color)
}

/// The extended-range write of attachment `color` the shader computes, in the storage form
/// (`fetch`) or over the input attachment (`input_blend`): the pipeline's blend there (`None`
/// for an unblended write and for a write that stores none of the format's channels), the
/// attachment's form, and whether the write stores none of its channels. `None` where the
/// function writes the attachment otherwise (or not at all).
pub(crate) fn shader_blend(
    cx: &SelectCx<'_>,
    color: u32,
) -> Option<(
    Option<crate::pipeline::lower_inputs::BlendEquation>,
    crate::metal::format::ExtendedRange,
    bool,
)> {
    let fp = &cx.families.fragment;
    if let Some(sc) = fp.fetch.get(&color) {
        return sc
            .view
            .extended
            .map(|form| (sc.view.blend, form, sc.view.stores_nothing()));
    }
    fp.input_blends.get(&color).map(|ib| {
        let blends = ib.blend.is_some();
        (
            blends.then_some(ib.equation).flatten(),
            ib.form,
            !blends && !ib.merges(),
        )
    })
}

/// Whether the function's second (index-1) output to `color` is the second source of the
/// extended-range blend it computes over the attachment's input attachment, outside the stage
/// interface.
pub fn blends_second_through_inputs(cx: &SelectCx<'_>, color: u32) -> bool {
    input_blend::blends_second(cx, color)
}

/// Whether the function writes an output the returned value holds outside the stage interface:
/// an imageblock output, an output to an attachment in the storage form, or a shader-side stencil.
pub fn writes_outside_interface(cx: &SelectCx<'_>) -> bool {
    let fp = &cx.families.fragment;
    fp.imageblock.as_ref().is_some_and(|p| p.output.is_some())
        || fp.fetch.values().any(|sc| sc.writes)
        || fp.stencil.is_some()
}

/// The work before the body, in the function's first block: read the imageblock input, place the
/// fragment's tile in the pass's tile memory, copy the textures read through snapshots
/// (`snapshot`), and read the texel of each storage-form color the function reads
/// (`EmitCx::fetched`), which every use of the color then takes.
pub fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    imageblock::emit_entry(cx, fb)?;
    crate::lower::access::tile_memory::emit_entry(cx, fb)?;
    snapshot::emit_entry(cx, fb)?;
    let fp = &cx.plan.families.fragment;
    let read: BTreeSet<u32> = fp
        .colors
        .iter()
        .map(|c| c.color)
        .filter(|color| fp.fetch.contains_key(color))
        .collect();
    for color in read {
        let texel = fetch::texel(cx, fb, color)?;
        cx.fetched.push((color, texel));
    }
    Ok(())
}

/// The value of parameter `b` in its lowered representation, read by an instruction or a
/// terminator: a color input's texel (`color::value`).
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    let planned = cx
        .plan
        .families
        .fragment
        .colors
        .iter()
        .find(|c| c.param == b.param)
        .copied();
    match planned {
        Some(c) => color::value(cx, fb, c),
        None => Err(cx.fail(format!("parameter {} has no fragment value", b.param))),
    }
}

/// A bool constant.
fn bool_const(fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    fb.module().const_value(c)
}

/// `if c { kill }` in the open block, which stays open after the construct.
pub(crate) fn kill_if(fb: &mut FunctionBuilder<'_>, c: SValue, form: KillForm) -> BResult<()> {
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        fb.kill(form)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// The whole-quad exit before non-quad subgroup operation `at`, where the plan places one: a quad
/// whose four invocations are each discarded or a helper ends, so it is no lane of the operation,
/// as Metal drops such a quad from the simdgroup. The operation runs in a block every invocation
/// executes together, so each quad's invocations are all present. Under predication an invocation
/// is out where its predicate or the `HelperInvocation` input is set; where the sites demote, where
/// `OpIsHelperInvocationEXT` answers it is a helper, as a demoted invocation is.
pub fn emit_quad_exit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
) -> BResult<()> {
    let fp = &cx.plan.families.fragment;
    if !fp.quad_exits.contains(&at) {
        return Ok(());
    }
    let kill = fp.kill_form();
    let out = match fp.predicated() {
        true => {
            let (Some(discarded), Some(helper)) = (
                cx.decl.families.fragment.discarded,
                cx.decl.families.fragment.helper,
            ) else {
                return Err(cx.fail("a whole-quad exit without its predicate and helper input"));
            };
            let p = fb.module().global_value(discarded)?;
            let d = fb.load(p, MemAccess::default())?;
            let p = fb.module().global_value(helper)?;
            let h = fb.load(p, MemAccess::default())?;
            fb.binary(BinOp::LogicalOr, d, h)?
        }
        false => fb.is_helper_invocation()?,
    };
    let mut all = out;
    for dir in [
        QuadDirection::Horizontal,
        QuadDirection::Vertical,
        QuadDirection::Diagonal,
    ] {
        let other = fb.group_quad_swap(out, dir)?;
        all = fb.binary(BinOp::LogicalAnd, all, other)?;
    }
    kill_if(fb, all, kill)
}

/// Whether the calling invocation is a helper, Metal's `quad_is_helper_thread()` and
/// `simd_is_helper_thread()`. In a fragment function where a discard demotes,
/// `OpIsHelperInvocationEXT`, which a demoted invocation answers too; else the `HelperInvocation`
/// input, or'd with a predicated discard's `discarded` (a predicated invocation is no Vulkan
/// helper, and Metal's discarded thread is one). A terminating discard leaves no invocation to ask.
/// No thread of another stage is a helper (measured).
fn is_helper(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let fp = &cx.plan.families.fragment;
    if !fp.helper_queries {
        return bool_const(fb, false);
    }
    if fp.demotes() {
        return fb.is_helper_invocation();
    }
    let Some(helper) = cx.decl.families.fragment.helper else {
        return Err(cx.fail("a helper query without the helper input"));
    };
    let p = fb.module().global_value(helper)?;
    let h = fb.load(p, MemAccess::default())?;
    if !fp.predicated() {
        return Ok(h);
    }
    let Some(discarded) = cx.decl.families.fragment.discarded else {
        return Err(cx.fail("a helper query without the predicated discard's predicate"));
    };
    let p = fb.module().global_value(discarded)?;
    let d = fb.load(p, MemAccess::default())?;
    fb.binary(BinOp::LogicalOr, h, d)
}

/// The condition of a side effect a predicated site reaches (`FragmentPlan::guarded`): the
/// invocation was not discarded, `!discarded`. `lower::emit` emits each such side effect under
/// it, with an atomic's result zero where it does not run.
pub fn not_discarded(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let Some(g) = cx.decl.families.fragment.discarded else {
        return Err(cx.fail("a guarded side effect without the predicated discard's predicate"));
    };
    let p = fb.module().global_value(g)?;
    let d = fb.load(p, MemAccess::default())?;
    fb.unary(UnOp::LogicalNot, d)
}

/// Emit the entry's return terminator after `stage_io::emit_return` stored the outputs: a
/// shader-side stencil tests and writes its shadow, recording a failing fragment as discarded, and
/// an in-shader depth test its stand-in; the outputs of returned value `v` to attachments in the
/// storage form and its imageblock output are written, then a predicated discard kills a discarded
/// invocation, which drops its outputs. An interlocked body only returns; its caller kills after
/// the critical section.
pub fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
) -> BResult<()> {
    stencil::emit_return(cx, fb, v)?;
    depth_test::emit_return(cx, fb, v)?;
    fetch::emit_return(cx, fb, v)?;
    imageblock::emit_return(cx, fb, v)?;
    let fp = &cx.plan.families.fragment;
    if fp.interlock.is_none() && fp.predicated() {
        let Some(g) = cx.decl.families.fragment.discarded else {
            return Err(cx.fail("a predicated discard without its predicate"));
        };
        let p = fb.module().global_value(g)?;
        let d = fb.load(p, MemAccess::default())?;
        kill_if(fb, d, cx.plan.families.fragment.kill_form())?;
    }
    fb.ret(None)
}

/// The entry-wrapper hook of `lower::emit`: the entry function of finished function `body`. That is
/// `body` itself, except in a raster order group, where it is a new function that runs `body`
/// inside the interlock critical section and kills a discarded invocation after it: `Begin; call
/// body; End; if discarded { kill }; return`; and for a function looping over its covered samples,
/// where it is a new function that runs `body` once per covered sample (`coverage`).
pub fn wrap_entry(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    decls: &FamilyDecls,
    body: FuncId,
) -> Result<FuncId, Error> {
    let fp = &plan.families.fragment;
    let decl = &decls.fragment;
    let looped = fp.coverage.is_some_and(|c| c.looped);
    match (fp.interlock, looped) {
        (None, false) => return Ok(body),
        (None, true) => return coverage::wrap(mb, plan, decls, body),
        (Some(_), true) => {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "an interlocked function looping over its covered samples",
            )))
        }
        (Some(_), false) => {}
    }
    let discarded = match (fp.records_discards(), decl.discarded) {
        (true, Some(g)) => Some(g),
        (false, _) => None,
        (true, None) => {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "an interlocked discard without its predicate",
            )))
        }
    };
    let kill = fp.kill_form();
    let void = mb.ty(SType::Void)?;
    let mut fb = mb.function(void)?;
    // The body reads the coverage from the entry's copy.
    if decl.coverage_copy.is_some() {
        load_coverage(&mut fb, plan, decls)?;
    }
    fb.begin_interlock()?;
    fb.call(body, &[])?;
    fb.end_interlock()?;
    if let Some(g) = discarded {
        let p = fb.module().global_value(g)?;
        let d = fb.load(p, MemAccess::default())?;
        kill_if(&mut fb, d, kill)?;
    }
    fb.ret(None)?;
    Ok(fb.finish()?)
}

/// Select the strategy of instruction `inst` of this family.
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<FragmentStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(cx.invariant("a non-call routed to the fragment family"));
    };
    let op = match &call.callee {
        Callee::Air {
            op: AirIntrinsic::Fragment(op),
            ..
        } => op,
        Callee::Air {
            op: AirIntrinsic::Interpolate(i),
            ..
        } => return interpolate::select(cx, i, &call.args),
        _ => return Err(cx.invariant("a call routed to the fragment family is no fragment op")),
    };
    match op {
        Fragment::Discard if cx.stage == Stage::Fragment => {
            // A placeholder until `finish` decides the site's form.
            cx.families
                .fragment
                .discards
                .insert(at, DiscardForm::Terminate);
            Ok(FragmentStrategy::Discard)
        }
        Fragment::Discard => Err(not_lowered()),
        // Canon compiled every `float2` map; the `uint2` ones reach here.
        Fragment::MapCoordinates { .. } => {
            Err(Unsupported::not_implemented(Feature::RasterizationRateMap).into())
        }
        Fragment::NumSamples if cx.stage == Stage::Fragment => num_samples(cx, inst, &call.args),
        Fragment::NumSamples => Err(not_lowered()),
        Fragment::IsHelperThread => {
            if cx.stage == Stage::Fragment {
                cx.families.fragment.helper_queries = true;
            }
            Ok(FragmentStrategy::IsHelperThread)
        }
    }
}

/// `get_num_samples()`: the call with operand 0, as Metal's header spells it, is the pipeline's
/// rasterization sample count, read from the `SpecKey::SampleCount` constant. Another operand has
/// no established meaning and is refused.
fn num_samples(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    args: &[Value],
) -> Result<FragmentStrategy, Error> {
    let zero = match args {
        [Value::Const(c)] => matches!(
            cx.entry.module.consts().get(*c).map(|k| &k.value),
            Some(Const::Int(0) | Const::Null)
        ),
        _ => false,
    };
    if !zero {
        return Err(not_lowered());
    }
    let form = form_on(cx.entry, cx.profile, inst.ty)?;
    if !form.is_int() || form.lanes != 1 {
        return Err(cx.invariant("get_num_samples does not return a scalar integer"));
    }
    cx.facts.insert(RuntimeFact::SampleCount);
    Ok(FragmentStrategy::NumSamples(form))
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &FragmentStrategy,
) -> BResult<()> {
    match s {
        FragmentStrategy::Interpolate(read) => {
            let Some(result) = cx.plan.values.inst(at) else {
                return Err(cx.fail(format!("{at:?} is out of range")));
            };
            let x = crate::lower::stage_io::interpolate(cx, fb, read)?;
            cx.define(result, x);
            Ok(())
        }
        FragmentStrategy::IsHelperThread => {
            let Some(result) = cx.plan.values.inst(at) else {
                return Err(cx.fail(format!("{at:?} is out of range")));
            };
            let x = is_helper(cx, fb)?;
            cx.define(result, x);
            Ok(())
        }
        FragmentStrategy::NumSamples(form) => {
            let Some(result) = cx.plan.values.inst(at) else {
                return Err(cx.fail(format!("{at:?} is out of range")));
            };
            let u32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let c = fb.module().spec(SpecKey::SampleCount, u32_, ConstBits(1))?;
            let x = fb.module().const_value(c)?;
            let x = ops::mask(fb, *form, x)?;
            cx.define(result, x);
            Ok(())
        }
        FragmentStrategy::Discard => {
            let fp = &cx.plan.families.fragment;
            let kill = fp.kill_form();
            let Some(form) = fp.discards.get(&at).copied() else {
                return Err(cx.fail(format!("discard {at:?} has no planned form")));
            };
            match form {
                DiscardForm::Demote => fb.demote(),
                // The rest of the block stays structurally reachable through the construct's
                // false edge, so the kill needs no new CFG node from the structurizer.
                DiscardForm::Terminate if fp.interlock.is_none() => {
                    let yes = bool_const(fb, true)?;
                    kill_if(fb, yes, kill)
                }
                // No terminator may run inside the critical section: record the discard and
                // return to the entry, which kills after `End`.
                DiscardForm::Terminate => {
                    let Some(g) = cx.decl.families.fragment.discarded else {
                        return Err(cx.fail("an interlocked discard without its predicate"));
                    };
                    let yes = bool_const(fb, true)?;
                    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                        let p = fb.module().global_value(g)?;
                        let yes = bool_const(fb, true)?;
                        fb.store(p, yes, MemAccess::default())?;
                        fb.ret(None)?;
                        Ok(ArmEnd::Exits)
                    });
                    fb.if_(&mut (), yes, arm, None)?;
                    Ok(())
                }
                DiscardForm::Predicate => {
                    let Some(g) = cx.decl.families.fragment.discarded else {
                        return Err(cx.fail("a predicated discard without its predicate"));
                    };
                    let p = fb.module().global_value(g)?;
                    let yes = bool_const(fb, true)?;
                    fb.store(p, yes, MemAccess::default())
                }
            }
        }
    }
}
