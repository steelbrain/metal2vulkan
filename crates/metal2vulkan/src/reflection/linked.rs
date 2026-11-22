//! Which stages of a render or mesh pipeline use a direct texture or sampler argument once Metal
//! has linked the pipeline: Metal refuses a pipeline state that supports indirect command buffers
//! where one does.
//!
//! - **An object function** uses one where its pipeline's reflection says it does: a used binding
//!   of type texture or sampler ([`Binding::is_used`](super::Binding::is_used)), decided from the
//!   arguments' roles and use alone (An argument whose reflection Metal's answer has not been
//!   measured for no longer refuses the answer).
//! - **A fragment function** (It used the reflection's answer) uses one where an instruction its
//!   pipeline's linking keeps reads it: what reaches a side effect, a depth, stencil or sample mask
//!   output, or a color output its attachment writes ([`fragment_output`]), whether or not the
//!   pipeline rasterizes. A color output to an attachment the pipeline lacks or under a write mask
//!   of no channel is dropped, as Metal drops it (on an M4 under macOS 27.0.1).
//! - **A vertex or mesh function** uses one where an instruction its pipeline's linking keeps
//!   reads it (`live.rs`, the measured rule): what reaches the position and the other built-in
//!   outputs, a varying the fragment function reads, a side effect, or a branch, switch or loop
//!   any of those depend on. A varying no fragment input reads (one the fragment function
//!   declares and never reads, one it does not declare, or any where the pipeline has no fragment
//!   function or does not rasterize) is dropped with everything only it needs, though Metal's
//!   reflection still lists its texture as used (on an M4 under macOS 27.0.1).
//! - **What a fragment function reads** is a varying a kept instruction of its own reads, under
//!   the fragment function's own rule.
//! - **Proven or not** (a use only an unmeasured rule keeps must not read as Metal's refusal). The
//!   rules that keep beyond Metal where its answer is unmeasured ([`LinkedUseGap`]) each only add
//!   to what is kept, so the analysis runs first with all of them off: a use still kept is one
//!   Metal's measured rule keeps (proven, `gap: None`). Else, where it runs again with them on and
//!   the use is kept, the gap named is the first rule that keeps it alone, or, where none does, the
//!   first of the rules left once each that is not needed is dropped in turn. Each run is linear in
//!   the functions; a proven use takes one, an unused one two, and an unproven one at most ten.

use std::collections::BTreeSet;

use super::live::{forward_reach, kept_parameters, MeshBuiltin, MeshSlots, Reach, Returned, Roots};
use super::stage_direct;
use crate::air;
use crate::air::abi::{AbiInt, MeshFieldRole, OutputRole, ParamRole, UserKey};
use crate::air::ir::Type;
use crate::budget::Meter;
use crate::error::{Error, Feature, Unsupported};
use crate::metal::PixelFormat;
use crate::pipeline::compile;
use crate::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, Stage, StageEntries, StageSource,
};

/// A stage whose function uses a direct texture or sampler argument once its pipeline is linked,
/// with whether Metal's measured rule proves the use: what
/// `RenderPipelineDescriptor::stages_using_direct_arguments` and its mesh form answer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectArgumentUse {
    /// The stage.
    pub stage: Stage,
    /// `None` where a rule measured on Metal keeps the use, so Metal refuses the state with
    /// indirect command buffers; else the rule kept beyond Metal, where its answer is unmeasured,
    /// that keeps it: Metal may make the state.
    pub gap: Option<LinkedUseGap>,
}

/// A rule of the linked use that keeps beyond Metal where Metal's answer is unmeasured: a use only
/// such a rule keeps is not proven to be one Metal refuses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinkedUseGap {
    /// A block of the function reaches no exit (an infinite loop), and every branch is kept.
    EndlessLoop,
    /// The texture or sampler reaches what is kept only through the function's own stack memory
    /// (an `alloca`): a load keeps every write into its allocation, an allocation whose address
    /// goes elsewhere keeps them all, and no measured row passed a value through memory.
    StackMemory,
    /// A mesh function's write of a built-in output field its function constants gate off is
    /// kept.
    GatedOffMeshBuiltin,
    /// A fragment function's color output that its attachment writes only in part is kept, and
    /// so what it reads: under a write mask of some of the format's channels but not all, or
    /// [`ColorWriteMask::UNSPECIALIZED`](crate::metal::ColorWriteMask::UNSPECIALIZED), into a
    /// format of fewer than four channels, or a dual-source second color under blending that
    /// reads no second source. Metal keeps a color channel by channel (a texel
    /// that reaches only a channel the mask leaves out is dropped); the analysis does not.
    UnmeasuredColorWrite,
}

/// The gaps in the order a gap is named.
const GAPS: [LinkedUseGap; 4] = [
    LinkedUseGap::EndlessLoop,
    LinkedUseGap::StackMemory,
    LinkedUseGap::GatedOffMeshBuiltin,
    LinkedUseGap::UnmeasuredColorWrite,
];

/// The rules kept beyond Metal that apply to a run, a bit per [`LinkedUseGap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rules(u8);

impl Rules {
    pub(crate) const ALL: Rules = Rules(0b1111);
    pub(crate) const NONE: Rules = Rules(0);

    fn bit(gap: LinkedUseGap) -> u8 {
        1 << (gap as u8)
    }

    fn only(gap: LinkedUseGap) -> Rules {
        Rules(Rules::bit(gap))
    }

    fn without(self, gap: LinkedUseGap) -> Rules {
        Rules(self.0 & !Rules::bit(gap))
    }

    /// Whether the rule of `gap` applies.
    pub(crate) fn keeps(self, gap: LinkedUseGap) -> bool {
        self.0 & Rules::bit(gap) != 0
    }
}

/// The function of stage `source`, which a linked-use question decodes: a stage the compile
/// layer already compiled kept no AIR, and is refused.
fn function(source: &StageSource) -> Result<&FunctionDescriptor, Error> {
    match source {
        StageSource::Function(fd) => Ok(fd),
        StageSource::Compiled(_) => Err(Error::Unsupported(Unsupported::not_implemented(
            Feature::Reflection,
        ))),
    }
}

/// Whether a fragment function's output of role `role` is kept in a pipeline of color
/// attachments `attachments`: `Some` as measured on Metal, `None` where unmeasured (kept only
/// under `UnmeasuredColorWrite`). A color output to an attachment the pipeline lacks (none, or of
/// no format) is dropped, as is one under a write mask of no channel the attachment's format
/// holds, or a dual-source second color without blending (on an M4 under macOS 27.0.1; the
/// consumer's `R8Unorm` under a mask of alpha alone); one whose attachment writes every channel of
/// four is kept, a second color too where blending reads a second source. A depth, stencil or
/// sample mask output is kept, the stencil one without a stencil attachment too, and any other
/// output.
fn fragment_output(
    role: &OutputRole,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
) -> Option<bool> {
    use crate::metal::{BlendFactor as F, ColorWriteMask as M};
    let OutputRole::RenderTarget {
        attachment, index, ..
    } = role
    else {
        return Some(true);
    };
    let AbiInt::Const(n) = attachment else {
        return None;
    };
    let Some(a) = attachments
        .get(*n as usize)
        .and_then(Option::as_ref)
        .filter(|a| a.pixel_format != PixelFormat::Invalid)
    else {
        return Some(false);
    };
    if *index != 0 {
        let factors = [
            a.source_rgb_blend_factor,
            a.destination_rgb_blend_factor,
            a.source_alpha_blend_factor,
            a.destination_alpha_blend_factor,
        ];
        let second = factors.iter().any(|f| {
            matches!(
                f,
                F::Source1Color
                    | F::OneMinusSource1Color
                    | F::Source1Alpha
                    | F::OneMinusSource1Alpha
            )
        });
        match (a.blending_enabled, second) {
            (false, _) => return Some(false),
            (true, false) => return None,
            (true, true) => {}
        }
    }
    let mask = a.write_mask;
    if mask.contains(M::UNSPECIALIZED) {
        return None;
    }
    // The mask writes only the channels the format holds: one holding none of them writes
    // nothing, as a mask of none (an `R8Unorm` attachment under a mask of alpha alone, made on
    // an M3 Max under macOS 26.6.2).
    let held = crate::metal::format::channels(a.pixel_format);
    let written = match held {
        Some(held) => [M::RED, M::GREEN, M::BLUE, M::ALPHA]
            .into_iter()
            .zip(held)
            .filter(|&(_, h)| h)
            .fold(M::NONE, |m, (c, _)| m.union(c))
            .intersection(mask),
        None => mask.intersection(M::ALL),
    };
    if written.is_empty() {
        return Some(false);
    }
    match held == Some([true; 4]) && mask.contains(M::ALL) {
        true => Some(true),
        false => None,
    }
}

/// What fragment function `entry`'s `ret` demands in a pipeline of color attachments
/// `attachments` under `rules`: its outputs [`fragment_output`] keeps.
fn fragment_returned(
    entry: &air::Entry,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
    rules: Rules,
) -> Returned {
    let kept: BTreeSet<u32> = entry
        .ep
        .outputs
        .iter()
        .filter(|o| {
            fragment_output(&o.role, attachments)
                .unwrap_or_else(|| rules.keeps(LinkedUseGap::UnmeasuredColorWrite))
        })
        .map(|o| o.field)
        .collect();
    returned(entry, kept)
}

/// The user keys of fragment function `entry`'s stage inputs its pipeline's linking drops, in a
/// pipeline of color attachments `attachments`, which a compile may read as zero: those no kept
/// instruction reads with every rule kept beyond Metal on, so Metal's linking drops them too. Each
/// is checked forward ([`forward_reach`], the same roots): a value of theirs that reaches a kept
/// output, effect, branch, loop or stack write is an internal error, the two analyses
/// disagreeing, rather than a zero fed into kept work.
pub(crate) fn dropped_fragment_inputs(
    entry: &air::Entry,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
    m: &mut Meter,
) -> Result<BTreeSet<UserKey>, Error> {
    let rules = Rules::ALL;
    let roots = Roots {
        returned: fragment_returned(entry, attachments, rules),
        mesh: None,
        rules,
    };
    let kept = kept_parameters(entry, &roots, m)?;
    let mut params = BTreeSet::new();
    let mut dropped = BTreeSet::new();
    for p in &entry.ep.params {
        let ParamRole::FragmentInput(varying) = &p.role else {
            continue;
        };
        if !kept.get(p.param as usize).copied().unwrap_or(true) {
            params.insert(p.param);
            dropped.insert(varying.key.clone());
        }
    }
    // One pass checks them all: any reaching what is kept is the disagreement.
    if !params.is_empty() && forward_reach(entry, &params, &roots, m)? == Reach::Kept {
        return Err(Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::Invariant,
            "linked use: a stage input the linking drops reaches what it keeps".to_string(),
        )));
    }
    Ok(dropped)
}

/// The user keys of the varyings fragment function `entry` reads in a pipeline of color
/// attachments `attachments` under `rules`: its inputs a kept instruction reads, with its effects
/// and the outputs [`fragment_output`] keeps kept.
fn fragment_reads(
    entry: &air::Entry,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
    rules: Rules,
    m: &mut Meter,
) -> Result<BTreeSet<UserKey>, Error> {
    let roots = Roots {
        returned: fragment_returned(entry, attachments, rules),
        mesh: None,
        rules,
    };
    let kept = kept_parameters(entry, &roots, m)?;
    Ok(entry
        .ep
        .params
        .iter()
        .filter(|p| kept.get(p.param as usize).copied().unwrap_or(false))
        .filter_map(|p| match &p.role {
            ParamRole::FragmentInput(varying) => Some(varying.key.clone()),
            _ => None,
        })
        .collect())
}

/// Whether fragment function `fragment` uses a direct texture or sampler once linked in a
/// pipeline of color attachments `attachments`, its calls nesting `depth` deep, as
/// [`linked_use`] answers.
fn fragment_use(
    fragment: &FunctionDescriptor,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
    depth: u32,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<Option<Option<LinkedUseGap>>, Error> {
    compile::stage_entry(fragment, depth, entries, m, |entry, m| {
        let roots = |_: &BTreeSet<UserKey>, rules| Roots {
            returned: fragment_returned(entry, attachments, rules),
            mesh: None,
            rules,
        };
        linked_use(None, entry, roots, m)
    })
}

/// What `entry`'s `ret` demands where its outputs of fields `kept` are kept.
fn returned(entry: &air::Entry, kept: BTreeSet<u32>) -> Returned {
    let returns_struct = entry
        .module
        .func(entry.func)
        .and_then(|f| match entry.module.types().get(f.ty) {
            Some(Type::Function { ret, .. }) => entry.module.types().get(*ret),
            _ => None,
        })
        .is_some_and(|ret| matches!(ret, Type::Struct { .. }));
    match (kept.is_empty(), returns_struct) {
        (true, _) => Returned::Nothing,
        (false, true) => Returned::Fields(kept),
        (false, false) => Returned::All,
    }
}

/// The fragment function a stage's linked use reads through: its entry and the pipeline's color
/// attachments.
type Reader<'a> = Option<(&'a air::Entry, &'a [Option<ColorAttachmentDescriptor>; 8])>;

/// Whether `entry` uses a direct texture or sampler parameter once linked beside fragment
/// function `reader` (the module docs): `None` where it does not, else the gap that keeps the
/// use, `None` where it is proven. `roots` makes the roots of `entry` where the fragment stage
/// reads the varyings it is given, under the rules given.
fn linked_use(
    reader: Reader<'_>,
    entry: &air::Entry,
    roots: impl Fn(&BTreeSet<UserKey>, Rules) -> Roots,
    m: &mut Meter,
) -> Result<Option<Option<LinkedUseGap>>, Error> {
    let uses = |rules: Rules, m: &mut Meter| -> Result<bool, Error> {
        let read = match reader {
            Some((fragment, attachments)) => fragment_reads(fragment, attachments, rules, m)?,
            None => BTreeSet::new(),
        };
        reads_direct(entry, &roots(&read, rules), m)
    };
    proof(uses, m)
}

/// Whether `uses` holds with the rules kept beyond Metal on, and if so whether with them all off
/// (the module docs): `None` where it does not hold, else the gap that makes it hold, `None`
/// where it holds proven.
fn proof(
    uses: impl Fn(Rules, &mut Meter) -> Result<bool, Error>,
    m: &mut Meter,
) -> Result<Option<Option<LinkedUseGap>>, Error> {
    if uses(Rules::NONE, m)? {
        return Ok(Some(None));
    }
    if !uses(Rules::ALL, m)? {
        return Ok(None);
    }
    for gap in GAPS {
        if uses(Rules::only(gap), m)? {
            return Ok(Some(Some(gap)));
        }
    }
    let mut needed = Rules::ALL;
    for gap in GAPS {
        let fewer = needed.without(gap);
        if uses(fewer, m)? {
            needed = fewer;
        }
    }
    let gap = GAPS
        .into_iter()
        .find(|&gap| needed.keeps(gap))
        .ok_or_else(|| {
            Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "linked use: a use no rule keeps".to_string(),
            ))
        })?;
    Ok(Some(Some(gap)))
}

/// Runs `f` with fragment function `fragment`'s entry, its calls nesting `depth` deep, or with
/// none.
fn with_fragment<T>(
    fragment: Option<&FunctionDescriptor>,
    depth: u32,
    entries: &StageEntries,
    m: &mut Meter,
    f: impl FnOnce(Option<&air::Entry>, &mut Meter) -> Result<T, Error>,
) -> Result<T, Error> {
    match fragment {
        Some(fd) => compile::stage_entry(fd, depth, entries, m, |entry, m| f(Some(entry), m)),
        None => f(None, m),
    }
}

/// Whether a direct texture or sampler parameter of `entry` is read by an instruction kept under
/// `roots`.
fn reads_direct(entry: &air::Entry, roots: &Roots, m: &mut Meter) -> Result<bool, Error> {
    let kept = kept_parameters(entry, roots, m)?;
    Ok(entry.ep.params.iter().any(|p| {
        matches!(p.role, ParamRole::Texture(_) | ParamRole::Sampler { .. })
            && kept.get(p.param as usize).copied().unwrap_or(false)
    }))
}

/// What vertex function `entry`'s `ret` demands where the fragment stage reads the varyings
/// `read`: its built-in outputs, and the varyings of those keys.
fn vertex_returned(entry: &air::Entry, read: &BTreeSet<UserKey>) -> Returned {
    let kept: BTreeSet<u32> = entry
        .ep
        .outputs
        .iter()
        .filter(|o| match &o.role {
            OutputRole::VertexOutput(key) => read.contains(key),
            _ => true,
        })
        .map(|o| o.field)
        .collect();
    returned(entry, kept)
}

/// The data slots of mesh function `entry` whose writes its pipeline keeps where the fragment
/// stage reads the varyings `read`.
fn mesh_slots(entry: &air::Entry, read: &BTreeSet<UserKey>) -> MeshSlots {
    let mut slots = MeshSlots {
        vertex: BTreeSet::new(),
        primitive: BTreeSet::new(),
        gated_off: BTreeSet::new(),
    };
    for p in &entry.ep.params {
        let ParamRole::Mesh(outputs) = &p.role else {
            continue;
        };
        for field in outputs.vertex.iter().chain(&outputs.primitive) {
            if field.gated_off {
                slots.gated_off.extend(MeshBuiltin::of_field(&field.role));
                continue;
            }
            match &field.role {
                MeshFieldRole::VertexData { slot, key } if read.contains(key) => {
                    slots.vertex.insert(u64::from(*slot));
                }
                MeshFieldRole::PrimitiveData { slot, key } if read.contains(key) => {
                    slots.primitive.insert(u64::from(*slot));
                }
                _ => {}
            }
        }
    }
    slots
}

/// The stage inputs of `fragment` (its calls nesting `depth` deep) its linking keeps in a
/// pipeline of color attachments `attachments` whose user key is not among `written` (the vertex
/// or mesh function's outputs): proven where kept with every rule kept beyond Metal off, else
/// with the rule that keeps them ([`proof`]); `None` where there are none.
fn unwritten_inputs(
    fragment: &FunctionDescriptor,
    depth: u32,
    attachments: &[Option<ColorAttachmentDescriptor>; 8],
    written: &BTreeSet<UserKey>,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<Option<UnwrittenFragmentInputs>, Error> {
    compile::stage_entry(fragment, depth, entries, m, |entry, m| {
        let unwritten = |rules: Rules, m: &mut Meter| -> Result<Vec<String>, Error> {
            let read = fragment_reads(entry, attachments, rules, m)?;
            Ok(entry
                .ep
                .params
                .iter()
                .filter_map(|p| match &p.role {
                    ParamRole::FragmentInput(varying)
                        if read.contains(&varying.key) && !written.contains(&varying.key) =>
                    {
                        Some(p.name.as_ref().map_or_else(String::new, |n| n.to_string()))
                    }
                    _ => None,
                })
                .collect())
        };
        let gap = proof(|rules, m| Ok(!unwritten(rules, m)?.is_empty()), m)?;
        Ok(match gap {
            None => None,
            Some(None) => Some(UnwrittenFragmentInputs {
                names: unwritten(Rules::NONE, m)?,
                gap: None,
            }),
            Some(gap) => Some(UnwrittenFragmentInputs {
                names: unwritten(Rules::ALL, m)?,
                gap,
            }),
        })
    })
}

/// The user keys of the per-vertex and per-primitive data mesh function `entry` writes: its
/// output type's data fields its function constants do not gate off.
fn mesh_written(entry: &air::Entry) -> BTreeSet<UserKey> {
    let mut written = BTreeSet::new();
    for p in &entry.ep.params {
        let ParamRole::Mesh(outputs) = &p.role else {
            continue;
        };
        for field in outputs.vertex.iter().chain(&outputs.primitive) {
            match &field.role {
                MeshFieldRole::VertexData { key, .. }
                | MeshFieldRole::PrimitiveData { key, .. }
                    if !field.gated_off =>
                {
                    written.insert(key.clone());
                }
                _ => {}
            }
        }
    }
    written
}

impl RenderPipelineDescriptor {
    /// The stages of the pipeline this descriptor describes whose functions use a direct
    /// texture or sampler argument once Metal links the pipeline, in pipeline order: what
    /// `newRenderPipelineStateWithDescriptor:` refuses with `supportIndirectCommandBuffers`. The
    /// fragment function uses one where an instruction the linking keeps reads it: what reaches a
    /// side effect, a depth, stencil or sample mask output, or a color output its attachment
    /// writes; the vertex function where an instruction the linking keeps reads it: what reaches
    /// the position or another built-in output, a varying the fragment function reads, a side
    /// effect, or a branch, switch or loop those depend on (the rules are
    /// [`reflection`](crate::reflection)'s module `linked`). A varying no fragment input reads is
    /// dropped, as are those of a pipeline without a fragment function or with rasterization
    /// disabled. Each use names the rule kept beyond Metal that keeps it where Metal's measured
    /// rule does not ([`DirectArgumentUse::gap`]): Metal's answer is then unproven. It decodes and
    /// specializes the functions' AIR under `budget`; each function only where it has a used
    /// texture or sampler argument, decided from the arguments' roles and use alone (the fragment
    /// function's reads for the vertex function's), and only where `entries` holds no entry of it.
    ///
    /// Refused with the decode's and specialization's own errors, and where the vertex function
    /// needs deciding and a stage is one the compile layer already compiled
    /// (`StageSource::Compiled`, which kept no AIR): `Unsupported` with `Feature::Reflection`.
    /// An argument whose reflection Metal's answer has not been measured for, which
    /// [`reflection`](Self::reflection) refuses, decides nothing here (It refused the answer).
    pub fn stages_using_direct_arguments(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<Vec<DirectArgumentUse>, Error> {
        let mut m = Meter::new(budget);
        let mut uses = Vec::new();
        if stage_direct(&self.vertex, self.max_vertex_call_stack_depth, &mut m)? {
            let fragment = match (&self.fragment, self.rasterization_enabled) {
                (Some(fragment), true) => Some(function(fragment)?),
                _ => None,
            };
            let vertex = function(&self.vertex)?;
            let attachments = &self.color_attachments;
            let (fragment_depth, vertex_depth) = (
                self.max_fragment_call_stack_depth,
                self.max_vertex_call_stack_depth,
            );
            let used = with_fragment(fragment, fragment_depth, entries, &mut m, |fragment, m| {
                compile::stage_entry(vertex, vertex_depth, entries, m, |entry, m| {
                    let reader = fragment.map(|f| (f, attachments));
                    let roots = |read: &BTreeSet<UserKey>, rules| Roots {
                        returned: vertex_returned(entry, read),
                        mesh: None,
                        rules,
                    };
                    linked_use(reader, entry, roots, m)
                })
            })?;
            if let Some(gap) = used {
                uses.push(DirectArgumentUse {
                    stage: Stage::Vertex,
                    gap,
                });
            }
        }
        if let Some(fragment) = &self.fragment {
            let depth = self.max_fragment_call_stack_depth;
            if stage_direct(fragment, depth, &mut m)? {
                let used = fragment_use(
                    function(fragment)?,
                    &self.color_attachments,
                    depth,
                    entries,
                    &mut m,
                )?;
                if let Some(gap) = used {
                    uses.push(DirectArgumentUse {
                        stage: Stage::Fragment,
                        gap,
                    });
                }
            }
        }
        Ok(uses)
    }
}

impl RenderPipelineDescriptor {
    /// Whether the vertex function returns a value (its return type is not `void`), which
    /// `newRenderPipelineStateWithDescriptor:` refuses with rasterization disabled, with or
    /// without `supportIndirectCommandBuffers` (on an M4 under macOS 27.0.1). It decodes the vertex
    /// function's AIR under `budget`, unless `entries` holds the answer an earlier ask read: the
    /// vertex stage's descriptor alone decides it, so a consumer that gives one object to every
    /// creation decodes for this once however many states it makes of the function
    /// ([`StageEntries`]).
    ///
    /// Refused where the vertex function is one the compile layer already compiled
    /// (`StageSource::Compiled`, which kept no AIR: `Unsupported` with `Feature::Reflection`), and
    /// as its decode refuses.
    pub fn vertex_returns_value(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<bool, Error> {
        let mut m = Meter::new(budget);
        compile::stage_returns_value(function(&self.vertex)?, entries, &mut m)
    }
}

/// The stage inputs of a render pipeline's fragment function that its linking keeps and its
/// vertex function does not write: what `newRenderPipelineStateWithDescriptor:` refuses
/// ("Fragment input(s) ... not written by vertex shader"), as
/// [`RenderPipelineDescriptor::unwritten_fragment_inputs`] answers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnwrittenFragmentInputs {
    /// The inputs' names (`air.arg_name`, the stage-in struct's member names), in declaration
    /// order: those kept by a rule measured on Metal where `gap` is `None`, else those kept with
    /// every rule on.
    pub names: Vec<String>,
    /// `None` where a rule measured on Metal keeps them, so Metal refuses the state; else the rule
    /// kept beyond Metal that keeps one: Metal may make the state.
    pub gap: Option<LinkedUseGap>,
}

/// A render pipeline descriptor's answers to the linked-use questions a creation asks before its
/// compile, each `None` where it was not asked or not answered: what a consumer keeps under
/// [`Device::queried_render_descriptor_key`](crate::device::Device::queried_render_descriptor_key)
/// so that a later creation of the same descriptor, in this process or another, asks none of them
/// again.
/// [`LinkedUseAnswers::to_bytes`] encodes it under that key for one device, and
/// [`LinkedUseAnswers::from_bytes_for_key`] takes it back.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinkedUseAnswers {
    /// [`RenderPipelineDescriptor::vertex_returns_value`]'s answer.
    pub vertex_returns_value: Option<bool>,
    /// [`RenderPipelineDescriptor::fragment_writes_depth_unattached`]'s answer.
    pub fragment_writes_depth_unattached: Option<bool>,
    /// [`RenderPipelineDescriptor::stages_using_direct_arguments`]'s answer.
    pub stages_using_direct_arguments: Option<Vec<DirectArgumentUse>>,
    /// [`RenderPipelineDescriptor::unwritten_fragment_inputs`]'s answer.
    pub unwritten_fragment_inputs: Option<Option<UnwrittenFragmentInputs>>,
}

impl RenderPipelineDescriptor {
    /// The fragment function's stage inputs its linking keeps (a varying a kept instruction of
    /// its own reads, under its rule: its effects, its depth, stencil and sample mask outputs and
    /// the colors its attachments write) whose user key the vertex function writes no output of,
    /// whether or not the pipeline rasterizes (Metal links the fragment function either way, on an
    /// M4 under macOS 27.0.1); `None` where there are none or no fragment function. It decodes and
    /// specializes the functions' AIR under `budget`, where `entries` holds no entry of them.
    ///
    /// Refused where a stage is one the compile layer already compiled (`StageSource::Compiled`,
    /// which kept no AIR): `Unsupported` with `Feature::Reflection`, and as their decode refuses.
    pub fn unwritten_fragment_inputs(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<Option<UnwrittenFragmentInputs>, Error> {
        let Some(fragment) = &self.fragment else {
            return Ok(None);
        };
        let fragment = function(fragment)?;
        let mut m = Meter::new(budget);
        let written: BTreeSet<UserKey> = compile::stage_entry(
            function(&self.vertex)?,
            self.max_vertex_call_stack_depth,
            entries,
            &mut m,
            |entry, _| {
                Ok(entry
                    .ep
                    .outputs
                    .iter()
                    .filter_map(|o| match &o.role {
                        OutputRole::VertexOutput(key) => Some(key.clone()),
                        _ => None,
                    })
                    .collect())
            },
        )?;
        unwritten_inputs(
            fragment,
            self.max_fragment_call_stack_depth,
            &self.color_attachments,
            &written,
            entries,
            &mut m,
        )
    }

    /// Whether the fragment function writes depth (a `[[depth(...)]]` output) where the pipeline
    /// has no depth attachment, which `newRenderPipelineStateWithDescriptor:` refuses
    /// ("depthAttachmentPixelFormat is not valid and shader writes to depth"), with or without
    /// `supportIndirectCommandBuffers` and before that refusal (on an M4 under macOS 27.0.1). It
    /// decodes and specializes the fragment function's AIR under `budget`, only where the pipeline
    /// has no depth attachment and `entries` holds no entry of it.
    ///
    /// Refused where the fragment function is one the compile layer already compiled, as
    /// [`vertex_returns_value`](Self::vertex_returns_value) refuses.
    pub fn fragment_writes_depth_unattached(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<bool, Error> {
        writes_depth_unattached(
            self.fragment.as_ref(),
            self.depth_format,
            self.max_fragment_call_stack_depth,
            (budget, entries),
        )
    }
}

/// Whether `fragment`, its calls nesting `depth` deep, writes depth (a `[[depth(...)]]` output)
/// where the pipeline's depth attachment `depth_format` is none: a render and a mesh pipeline's
/// answer. It decodes the fragment function's AIR under `budget`, only where there is no depth
/// attachment.
fn writes_depth_unattached(
    fragment: Option<&StageSource>,
    depth_format: Option<PixelFormat>,
    depth: u32,
    (budget, entries): (&crate::Budget, &StageEntries),
) -> Result<bool, Error> {
    let attached = depth_format.is_some_and(|format| format != PixelFormat::Invalid);
    let Some(fragment) = fragment.filter(|_| !attached) else {
        return Ok(false);
    };
    let mut m = Meter::new(budget);
    compile::stage_entry(function(fragment)?, depth, entries, &mut m, |entry, _| {
        Ok(entry
            .ep
            .outputs
            .iter()
            .any(|o| matches!(o.role, OutputRole::Depth(_))))
    })
}

impl MeshPipelineDescriptor {
    /// Whether the fragment function writes depth (a `[[depth(...)]]` output) where the pipeline
    /// has no depth attachment, which `newRenderPipelineStateWithMeshDescriptor:` refuses
    /// ("depthAttachmentPixelFormat is not valid and shader writes to depth"), with or without a
    /// color attachment or `supportIndirectCommandBuffers` and before that refusal, as a render
    /// pipeline's (on an M4 under macOS 27.0.1). Its calls nest Metal's default
    /// depth of 1. It decodes and specializes the fragment function's AIR under `budget`, only
    /// where the pipeline has no depth attachment and `entries` holds no entry of it.
    ///
    /// Refused where the fragment function is one the compile layer already compiled, as
    /// [`RenderPipelineDescriptor::vertex_returns_value`] refuses.
    pub fn fragment_writes_depth_unattached(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<bool, Error> {
        writes_depth_unattached(
            self.fragment.as_ref(),
            self.depth_format,
            1,
            (budget, entries),
        )
    }

    /// The stages of the mesh pipeline this descriptor describes whose functions use a direct
    /// texture or sampler argument once Metal links the pipeline, in pipeline order: what
    /// `newRenderPipelineStateWithMeshDescriptor:` refuses with `supportIndirectCommandBuffers`.
    /// The object function uses one where it has a used texture or sampler argument (decided
    /// from the arguments' roles and use, as its reflection lists them); the fragment function as
    /// [`RenderPipelineDescriptor::stages_using_direct_arguments`] decides it; the mesh function
    /// where an instruction the linking keeps reads it, as that decides a vertex function's: a
    /// write of a vertex or primitive data field no fragment input reads is dropped. Each use
    /// names the rule kept beyond Metal that keeps it where Metal's measured rule does not, as
    /// there. Each stage's calls nest Metal's default depth of 1, as
    /// [`reflection`](Self::reflection) takes them. It decodes and specializes the functions' AIR
    /// under `budget`; each function only where it has a used texture or sampler argument (the
    /// fragment function's reads for the mesh function's), and only where `entries` holds no
    /// entry of it. Refused as `RenderPipelineDescriptor::stages_using_direct_arguments` refuses.
    pub fn stages_using_direct_arguments(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<Vec<DirectArgumentUse>, Error> {
        let mut m = Meter::new(budget);
        let mut uses = Vec::new();
        if let Some(object) = &self.object {
            if stage_direct(object, 1, &mut m)? {
                uses.push(DirectArgumentUse {
                    stage: Stage::Object,
                    gap: None,
                });
            }
        }
        if stage_direct(&self.mesh, 1, &mut m)? {
            let fragment = match (&self.fragment, self.rasterization_enabled) {
                (Some(fragment), true) => Some(function(fragment)?),
                _ => None,
            };
            let mesh = function(&self.mesh)?;
            let attachments = &self.color_attachments;
            let used = with_fragment(fragment, 1, entries, &mut m, |fragment, m| {
                compile::stage_entry(mesh, 1, entries, m, |entry, m| {
                    let reader = fragment.map(|f| (f, attachments));
                    let roots = |read: &BTreeSet<UserKey>, rules| Roots {
                        returned: Returned::Nothing,
                        mesh: Some(mesh_slots(entry, read)),
                        rules,
                    };
                    linked_use(reader, entry, roots, m)
                })
            })?;
            if let Some(gap) = used {
                uses.push(DirectArgumentUse {
                    stage: Stage::Mesh,
                    gap,
                });
            }
        }
        if let Some(fragment) = &self.fragment {
            if stage_direct(fragment, 1, &mut m)? {
                let used = fragment_use(
                    function(fragment)?,
                    &self.color_attachments,
                    1,
                    entries,
                    &mut m,
                )?;
                if let Some(gap) = used {
                    uses.push(DirectArgumentUse {
                        stage: Stage::Fragment,
                        gap,
                    });
                }
            }
        }
        Ok(uses)
    }

    /// The fragment function's stage inputs its linking keeps whose user key the mesh function
    /// writes no per-vertex or per-primitive data of, as
    /// [`RenderPipelineDescriptor::unwritten_fragment_inputs`] answers them for a vertex
    /// function: what `newRenderPipelineStateWithMeshDescriptor:` refuses ("Fragment input(s)
    /// `color` mismatching mesh shader output type(s) or not written by mesh shader", on an M4
    /// under macOS 27.0.1); `None` where there are none or no fragment
    /// function. Each stage's calls nest Metal's default depth of 1, and each is specialized only
    /// where `entries` holds no entry of it. Refused as that refuses.
    pub fn unwritten_fragment_inputs(
        &self,
        budget: &crate::Budget,
        entries: &StageEntries,
    ) -> Result<Option<UnwrittenFragmentInputs>, Error> {
        let Some(fragment) = &self.fragment else {
            return Ok(None);
        };
        let fragment = function(fragment)?;
        let mut m = Meter::new(budget);
        let written =
            compile::stage_entry(function(&self.mesh)?, 1, entries, &mut m, |entry, _| {
                Ok(mesh_written(entry))
            })?;
        unwritten_inputs(
            fragment,
            1,
            &self.color_attachments,
            &written,
            entries,
            &mut m,
        )
    }
}
