//! Render passes of more color attachments than the device draws at once: drawn in color
//! groups ([`ColorGroup`], `metal2vulkan::pipeline::compile::color_group`).
//!
//! A subpass holds at most `maxColorAttachments` color attachments
//! (`VUID-VkSubpassDescription-colorAttachmentCount-00845`), and a fragment shader's outputs lie
//! below `maxFragmentOutputAttachments` locations (`VUID-RuntimeSpirv-Location-06272`); the lesser
//! is a group's width ([`ColorGroup::width_on`]). A pass whose color attachments with a texture
//! number more, or reach an index at or past it, takes the grouping of a pipeline of those
//! attachments: group `g` holds the `g`-th width of them, attachment `colors()[l]` at index `l`.
//!
//! - **One encoder, a pass per group.** The program's encoder is the first group's: its layout
//!   holds that group's colors alone ([`project`]), at the pass's render area, over the pass's own
//!   depth/stencil attachment, and every pipeline state set on it draws as the state's twin of
//!   that group (`GraphicsState::color_group`), whose descriptor holds the group's colors at their
//!   locations as the projected pass does, and whose variants are that group's. A pass of more
//!   groups logs its calls
//!   (`render/replay.rs`), whatever the device's rebinds. As it ends, each later group opens an
//!   encoder over the same core, of its own colors and a twin of the depth and stencil, runs the
//!   log again, and ends its pass: its commands follow the first group's in the encoder's command
//!   buffer, in group order. The encoder's core ends once, so its fences, its uses and what it made
//!   resident are the encoder's, as Metal's.
//! - **Depth and stencil.** The first group renders the pass's attachment, loads it and stores it
//!   as Metal does, and resolves it. Each later group renders a twin made as the first group's pass
//!   ends: an image of the attachment's format, samples, layers and extent
//!   (`attachment_stand_in::stand_in`), loaded as the first group's first instance loads the
//!   attachment: cleared to the same value where it clears, else copied from the attachment after
//!   every step the first group records before its pass, and before that pass. Nothing stores or
//!   resolves a twin. Every draw that writes the depth or uses the stencil runs in every group, so
//!   each twin sees the same tests and writes as the attachment. A `DontCare` load of the depth or
//!   stencil in a pass of more groups loads (`settle_depth_stencil`): Metal's `DontCare` leaves
//!   the aspect's contents undefined, and the texture's own texels are one of its answers, so the
//!   twin's copy holds what the attachment holds (a memoryless one's, which nothing loads, clears
//!   to its clear value). The cost is a load where Metal's tile memory takes none, in such passes
//!   alone.
//! - **Draws.** A draw runs in the first group, and in each later group where its variant of that
//!   group writes a color or it writes the depth or uses the stencil; elsewhere the later group
//!   counts it as drawn (its reach of the attachments' tiles is the pass's) and records nothing. A
//!   decisive call (a pipeline state set, a visibility result mode, a draw) the first group
//!   refused, which the program saw, is refused in each later group too, changing nothing; one it
//!   ran that a later group refuses is a divergence, which fails the encoder's end.
//! - **Once, in the first group.** Visibility results are counted in the first group's pass alone;
//!   a later group draws as the first one does under its mode (one amplification at a time while a
//!   mode is set), without a query. Metal's fences are encoder-level orderings, and the runtime's
//!   are too (`fence.rs`): an update follows all of the encoder's work and a wait goes before all
//!   of it, the later groups' passes included, since they are the encoder's; a later group's
//!   fence calls change nothing. The depth and stencil store actions and resolve targets are the
//!   first group's alone. A color's store action, its options and its resolve target are its
//!   group's: the first group's encoder checks the others as Metal does, on its copy of them.
//! - **Barriers.** A twin's copy is recorded between barriers from everything before and to
//!   everything after (`StandInCopies::record`, `Barrier::everything`): "the first
//!   synchronization scope includes all commands that occur earlier in submission order" and
//!   the second "all commands that occur later in submission order", with the memory accesses of
//!   every stage, so the copy reads what the first group's steps before its pass wrote and that
//!   pass's draws come after it (the Vulkan specification's "Pipeline Barriers" and "Execution
//!   and Memory Dependencies"). The groups' passes and their steps record through the encoder's
//!   core, whose tracker places the barrier each command's uses need after the encoder's earlier
//!   commands, as between any two of its commands. Every image rests in `GENERAL`, which
//!   "supports all types of device access" ("Image Layouts"), as every image of the runtime does.
//! - **Lifetime.** A twin is the runtime's own texture, made through
//!   `Device::new_planned_texture` as every stand-in: it is counted with the device's images and
//!   held by its command buffer until it completes, which "All submitted commands that refer to
//!   image ... must have completed execution" before its destruction asks
//!   (`VUID-vkDestroyImage-image-01000`). It is made at the first group's end, in a pass of more
//!   groups with a depth/stencil attachment alone.
//! - **Refused** (`NotImplemented(ColorAttachmentGroups)`): a drawing state whose grouping is not
//!   the pass's (another set of colors with a pixel format, or a pass the device draws in one
//!   group: plain Metal draws a color without an attachment nowhere and a pass color without a
//!   format as zeroes, which no group variant states); a tile pipeline's state in a grouped pass;
//!   an execution of render commands in a pass of more groups; a depth/stencil attachment the twin
//!   cannot be copied from; and in a pass of more groups a draw that reads what another group
//!   writes: a color of another group (any color but the first group's), the first group's colors
//!   or the depth and stencil where the draw runs in a later group too, a resolve texture, the
//!   visibility result buffer, or memory the draw does not name (a handle table, or a root table
//!   where one of those lies over a buffer's bytes), and a resolve target set after a draw read
//!   its texture. A draw that reads its own group's attachments where it runs in that group alone
//!   reads them as in any pass (`texture/attached_reads.rs`).

use super::extended_shadow::ExtendedShadows;
use super::{
    reads_handles, DrawingState, Opening, RenderLog, RenderLowering, RenderPipelineState,
    RenderStateKind,
};
use crate::attachment_stand_in::{self, StandInCopies};
use crate::binding::CommandUses;
use crate::device::Device;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error, RuntimeGap};
use crate::rate_mapped_pass::RateMappedPass;
use crate::render_pass::{
    AspectTarget, ColorTarget, DepthStencilTarget, PassLayout, RenderPassDescriptor, Target,
};
use crate::render_target::{self, PassShape};
use crate::render_values::{LoadAction, StoreAction, StoreActionOptions};
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::texture::Texture;
use crate::tracker::{Access, UseRange};
use ash::vk;
use metal2vulkan::pipeline::ColorGroup;
use std::cell::RefCell;
use std::collections::BTreeSet;

/// The refusal of what a pass drawn in color groups does not build (the module docs).
pub(super) fn gap() -> Error {
    Error::NotImplemented(RuntimeGap::ColorAttachmentGroups)
}

/// The first group of the pass of `layout` on `device`, where the device draws its colors in
/// groups (the module docs): `None` where every color with a texture lies below the group width.
pub(super) fn grouping(device: &Device, layout: &PassLayout) -> Option<ColorGroup> {
    let width = ColorGroup::width_on(device.compile_device().profile());
    let present = || {
        (0u32..)
            .zip(&layout.colors)
            .filter(|(_, color)| color.is_some())
            .map(|(index, _)| index)
    };
    // Decided before the colors are listed, so a pass the device draws whole lists nothing: its
    // open allocates nothing for the groups it does not take.
    if present().all(|index| index < width) {
        return None;
    }
    ColorGroup::of(present().collect(), width, 0)
}

/// `layout` as `group`'s pass holds it: the group's colors alone, attachment `colors()[l]` at
/// index `l`; everything else as it is, the render area the whole pass's.
pub(super) fn project(layout: &PassLayout, group: &ColorGroup) -> PassLayout {
    let colors = group
        .colors()
        .iter()
        .map(|&index| layout.colors.get(index as usize).cloned().flatten())
        .collect();
    PassLayout {
        colors,
        ..layout.clone()
    }
}

/// `descriptor` as `group`'s pass reads it: the group's color attachments alone, attachment
/// `colors()[l]` at index `l`.
pub(super) fn project_descriptor(
    descriptor: &RenderPassDescriptor,
    group: &ColorGroup,
) -> RenderPassDescriptor {
    let mut out = descriptor.clone();
    out.color_attachments = Default::default();
    for (slot, &index) in out.color_attachments.iter_mut().zip(group.colors()) {
        if let Some(color) = descriptor.color_attachments.get(index as usize) {
            *slot = color.clone();
        }
    }
    out
}

/// Refuse a pass of more groups whose depth/stencil attachment loads an aspect (as `Load`, or a
/// `DontCare` that loads, `settle_depth_stencil`) from an image the twin cannot be copied from:
/// checked before the encoder opens, so the refusal leaves its command buffer as it was.
pub(super) fn check_twin(layout: &PassLayout) -> Result<(), Error> {
    let Some(ds) = &layout.depth_stencil else {
        return Ok(());
    };
    let loads = [ds.depth.as_ref(), ds.stencil.as_ref()]
        .into_iter()
        .flatten()
        .any(|aspect| twin_load(&settled(*aspect)) == LoadAction::Load);
    match loads {
        true => attachment_stand_in::check_copies(&ds.target).map_err(|_| gap()),
        false => Ok(()),
    }
}

/// A pass of more groups: its depth/stencil attachment's `DontCare` loads load, a memoryless
/// one's clear (the module docs).
pub(super) fn settle_depth_stencil(layout: &mut PassLayout) {
    if let Some(ds) = layout.depth_stencil.as_mut() {
        for aspect in [ds.depth.as_mut(), ds.stencil.as_mut()]
            .into_iter()
            .flatten()
        {
            *aspect = settled(*aspect);
        }
    }
}

/// `aspect`'s load as a pass of more groups takes it ([`settle_depth_stencil`]).
fn settled(aspect: AspectTarget) -> AspectTarget {
    let load = match (aspect.load, aspect.memoryless) {
        (LoadAction::DontCare, true) => LoadAction::Clear,
        (LoadAction::DontCare, false) => LoadAction::Load,
        (load, _) => load,
    };
    AspectTarget { load, ..aspect }
}

/// The load of a twin's aspect whose attachment the first group's first instance loads by `aspect`
/// as its steps before the pass leave it: its clear where it clears, else a copy.
fn twin_load(aspect: &AspectTarget) -> LoadAction {
    match aspect.load {
        LoadAction::Clear => LoadAction::Clear,
        _ => LoadAction::Load,
    }
}

/// The colors `l` of a group's variant writes at its locations below the group's `k`, which
/// lie in its pass (a color of another group lands past them).
fn writes_colors(state: &DrawingState, k: usize) -> bool {
    let mask = match k {
        0 => 0,
        k if k >= 32 => u32::MAX,
        k => (1u32 << k) - 1,
    };
    state.base().graphics.color_outputs & mask != 0
}

/// A pass drawn in color groups, as one group's encoder draws it (the module docs).
pub(super) struct Grouped {
    /// The pass's grouping, at the group the encoder draws.
    group: ColorGroup,
    /// The first group's own: what the later groups draw from, and what its checks read. `None`
    /// in a later group's encoder.
    first: Option<Box<First>>,
    /// Whether each of the pass's decisive calls ran (a pipeline state set, a visibility result
    /// mode, a draw), in order: the first group's encoder notes them, a later group's reads them.
    outcomes: Vec<bool>,
    /// The decisive calls a later group's encoder ran so far.
    at: usize,
    /// A later group's decisive call ran otherwise than the first group's.
    diverged: bool,
    /// A later group's visibility result mode is set, which the first group's counts.
    counting: bool,
}

/// What the first group's encoder keeps of a pass drawn in groups.
struct First {
    /// The pass's layout as it opened, every color by Metal's index: each later group's is
    /// projected from it.
    full: PassLayout,
    /// The residency the encoder started with, each later group's start.
    started: Residency,
    /// The pass's colors by Metal's index, which the encoder's color-indexed calls check and set
    /// where another group holds the color, as Metal checks them.
    others: Vec<Option<ColorTarget>>,
    /// The resources the pass's draws read, which a resolve target set later may not be: noted
    /// by a draw's check, which holds the encoder's tables.
    read: RefCell<BTreeSet<ResourceKey>>,
    /// The drawing state the encoder's state is the first group's twin of
    /// (`GraphicsState::color_group`), as the program's last `set_render_pipeline_state` that
    /// took set it: its twins decide which groups a draw runs in ([`Grouped::draws_in`]).
    set: Option<DrawingState>,
}

/// A decisive call's turn in a pass drawn in groups ([`RenderLowering::group_turn`]).
#[derive(Clone, Copy)]
pub(super) enum Turn {
    /// A pass of one group: the call runs as in any pass.
    Alone,
    /// The first group's encoder: the call runs, and its outcome is noted.
    First,
    /// A later group's encoder: the call runs where the first group's ran (`true`).
    Later(bool),
}

impl Grouped {
    /// The first group's encoder of a pass whose layout opened as `full`, its residency
    /// `started`.
    pub(super) fn first(group: ColorGroup, full: PassLayout, started: &Residency) -> Grouped {
        Grouped {
            group,
            first: Some(Box::new(First {
                others: full.colors.clone(),
                full,
                started: started.clone(),
                read: RefCell::new(BTreeSet::new()),
                set: None,
            })),
            outcomes: Vec::new(),
            at: 0,
            diverged: false,
            counting: false,
        }
    }

    /// The group the encoder draws.
    pub(super) fn group(&self) -> &ColorGroup {
        &self.group
    }

    /// Whether the pass takes more than one group.
    fn several(&self) -> bool {
        self.group.groups() > 1
    }

    /// The colors group `group` of the pass holds.
    fn len_of(&self, group: u32) -> usize {
        let width = self.group.width as usize;
        let start = (group as usize).saturating_mul(width);
        self.group.present.len().saturating_sub(start).min(width)
    }

    /// The group of the pass that holds color `index` (Metal's), where one does.
    fn group_of(&self, index: u32) -> Option<u32> {
        let at = self.group.present.iter().position(|&c| c == index)?;
        u32::try_from(at / (self.group.width as usize).max(1)).ok()
    }

    /// Whether a draw of `state` (the first group's) runs in later group `group`: its variant
    /// of the group writes a color there, or the draw writes the depth or uses the stencil
    /// (`depth_stencil`).
    fn draws_in(&self, state: &DrawingState, group: u32, depth_stencil: bool) -> bool {
        depth_stencil
            || state
                .color_group(group)
                .is_none_or(|twin| writes_colors(&twin, self.len_of(group)))
    }
}

/// What the first group's pass leaves the later groups of a pass of more
/// ([`Ended::later_groups`]).
pub(super) struct Later {
    group: ColorGroup,
    full: PassLayout,
    started: Residency,
    /// The depth and stencil the later groups render against, where the pass has an attachment.
    twin: Option<DepthStencilTarget>,
    outcomes: Vec<bool>,
}

impl Grouped {
    /// As the first group's pass is about to record, after every step it records before the pass
    /// (`core`'s command buffer): what the later groups draw, the twin made and loaded from its
    /// depth/stencil attachment as `layout` leaves it (the module docs). `None` in a later
    /// group's encoder and in a pass of one group.
    pub(super) fn later(
        self,
        core: &mut EncoderCore,
        layout: &PassLayout,
    ) -> Result<Option<Box<Later>>, Error> {
        let several = self.several();
        let Grouped {
            group,
            first,
            outcomes,
            ..
        } = self;
        let Some(first) = first.filter(|_| several) else {
            return Ok(None);
        };
        let First { full, started, .. } = *first;
        let twin = match layout.depth_stencil.as_ref().filter(|ds| !ds.hidden) {
            Some(ds) => Some(twin(core, ds, layout.extent)?),
            None => None,
        };
        Ok(Some(Box::new(Later {
            group,
            full,
            started,
            twin,
            outcomes,
        })))
    }
}

/// The twin of the depth/stencil attachment `ds` of a pass of `extent`, made and loaded in
/// `core`'s command buffer (the module docs).
fn twin(
    core: &mut EncoderCore,
    ds: &DepthStencilTarget,
    extent: [u32; 2],
) -> Result<DepthStencilTarget, Error> {
    let aspect = |a: &AspectTarget| AspectTarget {
        load: twin_load(a),
        store: StoreAction::DontCare,
        late_store: false,
        options: StoreActionOptions::NONE,
        memoryless: false,
        clear: a.clear,
        written_nan: false,
    };
    let depth = ds.depth.as_ref().map(aspect);
    let stencil = ds.stencil.as_ref().map(aspect);
    let copied = [
        (depth.as_ref(), vk::ImageAspectFlags::DEPTH),
        (stencil.as_ref(), vk::ImageAspectFlags::STENCIL),
    ]
    .into_iter()
    .filter(|(a, _)| a.is_some_and(|a| a.load == LoadAction::Load))
    .fold(vk::ImageAspectFlags::empty(), |all, (_, flag)| all | flag);
    let target: Target =
        attachment_stand_in::stand_in(core.command_buffer(), &ds.target, !copied.is_empty())?;
    if !copied.is_empty() {
        let [width, height] = extent;
        let pairs = [(&target, &ds.target, [0, 0, width, height], copied)];
        if let Some(copies) = StandInCopies::plan_aspects(&pairs, true)? {
            copies.record(core);
        }
    }
    core.device().shared().counts.color_group_twin();
    Ok(DepthStencilTarget {
        target,
        depth,
        stencil,
        keep_depth: false,
        keep_stencil: false,
        separate: None,
        resolve: None,
        stand_in: None,
        depth_resolve_target: None,
        stencil_resolve_target: None,
        hidden: false,
        ..ds.clone()
    })
}

/// An ended pass of a render encoder: its core, what it made resident, its log, and what its
/// later color groups draw.
pub(super) struct Ended {
    pub(super) core: EncoderCore,
    pub(super) residency: Residency,
    pub(super) log: Option<RenderLog>,
    pub(super) later: Option<Box<Later>>,
}

impl Ended {
    /// The pass's later color groups drawn (the module docs), each over the encoder's `core`
    /// from the first group's `log`: the core they leave. A pass of one group has none
    /// (`later` `None`), and its end calls nothing here.
    pub(super) fn later_groups(
        mut core: EncoderCore,
        later: &Later,
        log: Option<&RenderLog>,
    ) -> Result<EncoderCore, Error> {
        let log = log.ok_or_else(|| internal("a pass of several color groups without its log"))?;
        for group in 1..later.group.groups() {
            let mut encoder = RenderLowering::later_group(core, later, group, log)?;
            log.replay_into(&mut encoder);
            if encoder.group_diverged() {
                return Err(internal(
                    "a later color group's call ran otherwise than the first group's",
                ));
            }
            let ended = encoder.end_pass()?;
            core = ended.core;
            core.device().shared().counts.color_group_pass();
        }
        Ok(core)
    }
}

impl RenderLowering {
    /// The encoder of later group `group` of the pass `later` and `log` describe, over `core`
    /// (the module docs).
    fn later_group(
        core: EncoderCore,
        later: &Later,
        group: u32,
        log: &RenderLog,
    ) -> Result<Box<RenderLowering>, Error> {
        const CALL: &str = "end_encoding";
        let grouping = ColorGroup::of(later.group.present.clone(), later.group.width, group)
            .ok_or_else(|| internal("a later color group of a pass in one"))?;
        let mut layout = project(&later.full, &grouping);
        layout.visibility = None;
        layout.depth_stencil.clone_from(&later.twin);
        let descriptor = project_descriptor(log.descriptor(), &grouping);
        let device = core.device().clone();
        let rate_map = RateMappedPass::of(CALL, &descriptor, &layout, log.screen())?;
        let shape = PassShape::of(&layout)?;
        let compatible = render_target::shared_compatible_pass(&device, &shape)?;
        let draws = core.command_buffer().begin_draws(&compatible)?;
        let extended = ExtendedShadows::of(&descriptor, &layout, rate_map.is_some());
        let opening = Opening::new(
            &descriptor,
            (layout, extended),
            (shape, compatible),
            rate_map,
            None,
            later.started.clone(),
            None,
            Some(Box::new(Grouped {
                group: grouping,
                first: None,
                outcomes: later.outcomes.clone(),
                at: 0,
                diverged: false,
                counting: false,
            })),
        );
        Ok(RenderLowering::assemble(core, draws, opening))
    }

    /// Whether a later group's decisive call ran otherwise than the first group's.
    fn group_diverged(&self) -> bool {
        self.grouped
            .as_ref()
            .is_some_and(|g| g.diverged || g.at != g.outcomes.len())
    }

    /// The turn of a decisive call (a pipeline state set, a visibility result mode, a draw), taken
    /// now (the module docs).
    pub(super) fn group_turn(&mut self) -> Turn {
        let Some(g) = self.grouped.as_mut().filter(|g| g.several()) else {
            return Turn::Alone;
        };
        if g.first.is_some() {
            return Turn::First;
        }
        let ran = match g.outcomes.get(g.at) {
            Some(&ran) => ran,
            None => {
                g.diverged = true;
                false
            }
        };
        g.at += 1;
        Turn::Later(ran)
    }

    /// Note whether a decisive call of `turn` ran (`ran`).
    pub(super) fn group_outcome(&mut self, turn: Turn, ran: bool) {
        let Some(g) = self.grouped.as_mut() else {
            return;
        };
        match turn {
            Turn::Alone => {}
            Turn::First => g.outcomes.push(ran),
            Turn::Later(_) => g.diverged |= !ran,
        }
    }

    /// Whether the encoder is a later group's: a call that is the first group's alone (the depth
    /// and stencil's store actions and resolves, fences) changes nothing here.
    pub(crate) fn later_group_encoder(&self) -> bool {
        self.grouped.as_ref().is_some_and(|g| g.first.is_none())
    }

    /// A later group's visibility result mode, `mode_set` where it is not `Disabled`: the draws
    /// after it draw as the first group's do under it.
    pub(super) fn mirror_visibility(&mut self, mode_set: bool) {
        if let Some(g) = self.grouped.as_mut() {
            g.counting = mode_set;
        }
    }

    /// Whether the pass's draws draw as a visibility query counts them: a query is active, or a
    /// later group's mode is set (the first group's query counts).
    pub(super) fn counts_visibility(&self) -> bool {
        self.visibility.active() || self.grouped.as_ref().is_some_and(|g| g.counting)
    }

    /// The state the encoder draws `state` as in its pass's color group: itself in a pass the
    /// device draws whole; the state's twin of the encoder's group where both group alike.
    /// Refused (the module docs) where they do not, and for a tile pipeline's state in a grouped
    /// pass.
    pub(super) fn grouped_as(
        &self,
        state: RenderPipelineState,
    ) -> Result<RenderPipelineState, Error> {
        let pass = self.grouped.as_ref().map(|g| g.group());
        let own = match state.kind() {
            RenderStateKind::Graphics(graphics) => graphics.base().graphics.color_group.clone(),
            RenderStateKind::Tile(_) => {
                return match pass {
                    Some(_) => Err(gap()),
                    None => Ok(state),
                };
            }
        };
        match (pass, own) {
            (None, None) => Ok(state),
            (Some(pass), Some(own)) if pass.present == own.present && pass.width == own.width => {
                state
                    .color_group(pass.group)
                    .ok_or_else(|| internal("a drawing state without its color group's twin"))
            }
            _ => Err(gap()),
        }
    }

    /// Note `state` as the one the first group's encoder now draws a twin of ([`First::set`]).
    pub(super) fn grouped_set(&mut self, state: &RenderPipelineState) {
        let first = self.grouped.as_mut().and_then(|g| g.first.as_mut());
        if let (Some(first), RenderStateKind::Graphics(state)) = (first, state.kind()) {
            first.set = Some(state.clone());
        }
    }

    /// Color attachment `index` (Metal's) as the encoder's color-indexed calls set it: its group's
    /// at its place in the encoder's layout, another group's on the first group's copy (checked
    /// there as Metal checks it, and set by its own group's encoder), none in a later group's
    /// encoder (whose call changes nothing). Outside a grouped pass, the layout's at `index`.
    pub(super) fn color_slot(&mut self, index: u32) -> Option<&mut Option<ColorTarget>> {
        self.kept_attachments = None;
        let Some(g) = self.grouped.as_mut() else {
            return self.layout.colors.get_mut(index as usize);
        };
        match g.group.colors().contains(&index) {
            true => self.layout.colors.get_mut(g.group.location(index) as usize),
            false => g
                .first
                .as_mut()
                .and_then(|first| first.others.get_mut(index as usize)),
        }
    }

    /// Whether a later group skips the draw now (the module docs): its variant of the group,
    /// set on the encoder, writes no color of it, and it writes no depth and uses no stencil.
    pub(super) fn group_skips(&self) -> bool {
        let Some(g) = self.grouped.as_ref() else {
            return false;
        };
        let Some(RenderStateKind::Graphics(state)) = self.pipeline.as_ref().map(|s| s.kind())
        else {
            return false;
        };
        let depth_stencil = self.state.depth_writes() || self.state.uses_stencil();
        !depth_stencil && !writes_colors(state, g.group.colors().len())
    }

    /// A later group skips the draw, counted.
    pub(super) fn group_skipped(&mut self) {
        self.core
            .device()
            .shared()
            .counts
            .color_group_draw_skipped();
    }

    /// Refuse an execution of render commands in a pass of more groups (the module docs).
    pub(super) fn check_group_execution(&self) -> Result<(), Error> {
        match self.grouped.as_ref().is_some_and(|g| g.several()) {
            true => Err(gap()),
            false => Ok(()),
        }
    }

    /// In the first group's encoder of a pass of more: refuse a draw whose `uses` (its tables'
    /// and residency's), or whose reach of memory it does not name (through `roots`, a root
    /// table, or a handle table, `plan`), read what another group writes (the module docs), and
    /// note what it reads.
    pub(super) fn check_group_reads(
        &self,
        uses: &CommandUses,
        roots: bool,
        plan: &metal2vulkan::bind::BindingPlan,
    ) -> Result<(), Error> {
        let Some(g) = self.grouped.as_ref().filter(|g| g.several()) else {
            return Ok(());
        };
        let Some(first) = g.first.as_ref() else {
            return Ok(());
        };
        let state = first
            .set
            .as_ref()
            .ok_or_else(|| internal("a draw of a grouped pass without its state's twins"))?;
        let depth_stencil = self.state.depth_writes() || self.state.uses_stencil();
        let beyond = (1..g.group.groups()).any(|group| g.draws_in(state, group, depth_stencil));
        let watched = self.watched(beyond)?;
        if reads_handles(plan) {
            return Err(gap());
        }
        if roots && watched.iter().any(|w| w.over_bytes) {
            return Err(gap());
        }
        let reads_watched = uses.iter().any(|used| {
            watched
                .iter()
                .any(|w| used.key == w.key && used.range.overlaps(&w.range))
        });
        if reads_watched {
            return Err(gap());
        }
        if let Some(first) = &g.first {
            first
                .read
                .borrow_mut()
                .extend(uses.iter().map(|used| used.key));
        }
        Ok(())
    }

    /// In the first group's encoder of a pass of more: refuse a resolve target set on `texture`
    /// where a draw of the pass read it (the module docs).
    pub(super) fn check_group_resolve(&self, texture: Option<&Texture>) -> Result<(), Error> {
        let first = self
            .grouped
            .as_ref()
            .filter(|g| g.several())
            .and_then(|g| g.first.as_ref());
        let (Some(first), Some(texture)) = (first, texture) else {
            return Ok(());
        };
        let key = texture.bound_use(Access::NONE, Access::NONE).key;
        match first.read.borrow().contains(&key) {
            true => Err(gap()),
            false => Ok(()),
        }
    }

    /// What a draw of the first group's encoder may not read (the module docs), where it runs in a
    /// later group too (`beyond`).
    fn watched(&self, beyond: bool) -> Result<Vec<Watched>, Error> {
        let mut watched = Vec::new();
        let Some(g) = self.grouped.as_ref() else {
            return Ok(watched);
        };
        let Some(first) = g.first.as_ref() else {
            return Ok(watched);
        };
        // The program's texture of an attachment rendered through another image.
        let own = |color: &ColorTarget| -> Target {
            color
                .stand_in
                .as_ref()
                .map_or(&color.target, |s| &s.own)
                .clone()
        };
        let own_colors = self.layout.colors.iter().flatten();
        let other_colors = (0u32..)
            .zip(&first.others)
            .filter(|&(index, _)| g.group_of(index) != Some(g.group.group))
            .filter_map(|(_, color)| color.as_ref());
        for color in own_colors.clone().chain(other_colors.clone()) {
            if let Some(resolve) = &color.resolve {
                watched.push(Watched::target(
                    resolve.copy_to.as_ref().unwrap_or(&resolve.attachment),
                )?);
            }
            if let Some(copied) = &color.copied_resolve {
                watched.push(Watched::target(copied)?);
            }
        }
        for color in other_colors {
            watched.push(Watched::target(&own(color))?);
        }
        if beyond {
            for color in own_colors {
                watched.push(Watched::target(&own(color))?);
            }
        }
        if let Some(ds) = self.layout.depth_stencil.as_ref().filter(|ds| !ds.hidden) {
            for resolve in [&ds.depth_resolve_target, &ds.stencil_resolve_target]
                .into_iter()
                .flatten()
            {
                watched.push(Watched::texture(&resolve.texture));
            }
            if beyond {
                let separate = ds.separate.iter().flat_map(|s| s.targets());
                let stood = ds.stand_in.iter().map(|s| &s.own);
                for target in std::iter::once(&ds.target).chain(separate).chain(stood) {
                    watched.push(Watched::target(target)?);
                }
            }
        }
        if let Some(buffer) = &self.layout.visibility {
            let (identity, range) = buffer.tracked(0..buffer.length());
            watched.push(Watched {
                key: identity.key,
                range: UseRange::Bytes(range),
                over_bytes: true,
            });
        }
        Ok(watched)
    }
}

/// A resource a draw of a pass drawn in groups may not read ([`RenderLowering::watched`]).
struct Watched {
    key: ResourceKey,
    range: UseRange,
    /// It lies in a buffer's bytes, which a root table may reach.
    over_bytes: bool,
}

impl Watched {
    /// The level and layers of `target` the pass renders or resolves into.
    fn target(target: &Target) -> Result<Watched, Error> {
        let levels = target.level..target.level + 1;
        let layers = target.layer..target.layer + target.layers;
        let used = target
            .texture
            .image_use(levels, layers, Access::NONE, false)?;
        Ok(Watched {
            key: used.key,
            range: used.range,
            over_bytes: target.texture.aliases_buffer(),
        })
    }

    /// The whole of `texture`.
    fn texture(texture: &Texture) -> Watched {
        let used = texture.bound_use(Access::NONE, Access::NONE);
        Watched {
            key: used.key,
            range: used.range,
            over_bytes: texture.aliases_buffer(),
        }
    }
}
