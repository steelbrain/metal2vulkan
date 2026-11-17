//! A render pass's turns of the textures its colors render to their other faces
//! (`texture/faces.rs`), decided as the pass opens, before anything of it names their images; and
//! the counts of the attached reads' copies in that no turn took off
//! (`WorkCounts::attached_read_unswapped_*`).
//!
//! - **At the open** ([`Turns::open`]). A color whose texture has faces turns it where the pass
//!   turns textures at all (it draws its colors in one group, over no rasterization-rate map: each
//!   records the pass's calls again, a later group's or a reset map's, after the turn), and the
//!   color loads `Clear` or `DontCare` and renders the texture itself: level 0 and slice 0, one
//!   layer, single-sampled, resolving nowhere, not memoryless, no other attachment of the pass
//!   the same texture, and the pass failing no command buffer. The turn checks the rest
//!   ([`Texture::turn_face`](crate::Texture)).
//! - **A render area short of the texture.** The pass's render area is its extent from the
//!   origin, which may be short of the texture's (the descriptor's render target size, or a
//!   smaller attachment); the pass's load, store and draws change no texel outside it
//!   (`VkRenderPassBeginInfo::renderArea`: "the effects of attachment load, store and
//!   multisample resolve operations are restricted to the pixels whose x and y coordinates fall
//!   within the render area"), but Metal's `Clear` load of a color the pass stores clears the
//!   area rounded up to whole tiles, or the whole texture (`clear_reach.rs`), which the runtime
//!   clears before the pass. The face turned to holds what the texture held two turns ago, so
//!   before the pass's first instance, ahead of any clear, it takes from the face presented
//!   before the turn the texels outside what the color's clear rewrites ([`AreaSync::rewritten`]:
//!   that reach, or the render area where the color loads otherwise or is not stored), at most
//!   two rectangles beside it and below it, and of those only the ones the faces may not share:
//!   none where that face's copy into the other stands across command buffers
//!   (`shadow_standing.rs`), those inside its stale rectangle where it stands stale there, and
//!   every one where it does not stand or a command of the command buffer before the pass wrote
//!   either face ([`EncoderCore::area_syncs`]), counted
//!   (`WorkCounts::attached_read_swaps_short_area`, `attached_read_swap_area_texels`). The
//!   pass's writes of the face turned to, and the clears before it, change its texels inside the
//!   clear's reach alone ([`AreaSync::written`], `shadow::Within`, `render/shadows.rs`), so the
//!   attached reads' shadow stands stale inside it alone at the command buffer's end, and the
//!   next turn of a pass of the same area takes nothing. A pass that logs its calls for a
//!   rebind (`replay.rs`) turns as any other: a rebind's encoding again of the command buffer's
//!   encoders meets the faces as their first encodings met them (`texture/faces.rs`, **Encoded
//!   again**). A pass encoded again so turns nothing anew; where its first encoding turned a
//!   texture, it takes that turn again ([`Texture::retake_turn`](crate::Texture)), its first
//!   instance reading the face from before the turn with no copy in, as the first encoding's did.
//!   A turned texture's faces' shadows are carried no
//!   further from the command buffer's earlier encoders (`shadow::ShadowCarry`): what they held
//!   is no longer where the truth is. Nor is the kept draw the command buffer's last render
//!   encoder carried (`render/carried.rs`): it bound the face presented before.
//! - **The first instance.** At the pass's first plan of its copies, the first instance's, the
//!   encoder holds each turned texture's attached reads' shadow, the face presented before the
//!   turn, current ([`EncoderCore::hold_turned`]): it holds the texels from before the pass, as
//!   the copy in ahead of the pass's load actions would leave it, so the draws read it with no
//!   copy (`WorkCounts::attached_read_swap_reads`). The pass's writes of the new face then leave
//!   it stale, so a read after a barrier copies in what the draws before it wrote, as before.
//!   Where a barrier before the pass's first draw makes that copy in a clear
//!   (`texture/attached_reads.rs`), the clear goes as before: the face is no longer the truth. A
//!   pass that plans no copy holds nothing.
//! - **The counts** ([`counted`]): each copy in of the attached reads' shadow of a texture a color
//!   of the pass renders, the encoder's first of it by why its pass did not turn the texture, and
//!   every later one (a read after a barrier) and any of a turned texture as `split`.

use crate::clear_reach;
use crate::encoder::EncoderCore;
use crate::render_pass::{ColorTarget, PassLayout, MAX_COLOR_ATTACHMENTS};
use crate::render_values::LoadAction;
use crate::shadow::{CopyCause, ShadowCopies, ShadowCopy, Step};
use crate::texture::faces::Unturned;
use std::sync::Arc;

/// A turn of a texture's faces for a pass whose render area is short of the texture (the
/// module's **A render area short of the texture**).
#[derive(Clone)]
pub(crate) struct AreaSync {
    /// The copies from the face presented before the turn into the face turned to.
    pub(crate) copies: Arc<ShadowCopies>,
    /// The pass's render area's extent, from the origin, and the texture's.
    pub(crate) area: [u32; 2],
    pub(crate) extent: [u32; 2],
    /// The color that renders the texture, by index.
    pub(crate) color: usize,
    /// The color loads `Clear`, as the pass opened (a clear before the pass loads it after).
    clears: bool,
}

impl AreaSync {
    /// The extent from the origin that the pass of `layout` (with a draw call where `called`)
    /// and the clears before it may write of the face turned to: what Metal's `Clear` load of the
    /// color reaches ([`clear_reach::cleared`]), past which a `DontCare` load's clear of the tiles
    /// the draws reach does not go either, and inside which the render area lies.
    pub(crate) fn written(&self, layout: &PassLayout, called: bool) -> [u32; 2] {
        match layout.colors[self.color].as_ref() {
            Some(color) => {
                let target = color.stand_in.as_ref().map_or(&color.target, |s| &s.own);
                clear_reach::cleared(layout, target, called)
            }
            None => self.extent,
        }
    }

    /// The extent from the origin whose texels the color's clear rewrites, whatever the face
    /// turned to held: [`written`](Self::written) where the color loads `Clear` and the pass
    /// stores it (`clear_reach::ReachClears`, or the pass's own clear where it reaches no
    /// further), else the render area, as a pass over the whole texture takes it.
    pub(crate) fn rewritten(&self, layout: &PassLayout, called: bool) -> [u32; 2] {
        let stored = (layout.colors[self.color].as_ref())
            .is_some_and(|color| clear_reach::stores(color.store));
        match self.clears && stored {
            true => self.written(layout, called),
            false => self.area,
        }
    }
}

/// What a pass's open decided of its colors' textures' turns, which its encoder keeps
/// ([`EncoderCore::take_turns`]).
#[derive(Default)]
pub(crate) struct Turns {
    /// Why the pass turns no texture, where it turns none for its own reasons.
    pass: Option<Unturned>,
    /// By color index, for a color whose texture had faces as the pass opened: turned, or why
    /// not. `None` for every other color, whose cause the count works out
    /// ([`unfaced_cause`]).
    colors: [Option<Result<(), Unturned>>; MAX_COLOR_ATTACHMENTS],
    /// The turned textures' attached reads' shadows, which the encoder holds at the pass's first
    /// plan ([`EncoderCore::hold_turned`]).
    ahead: Vec<Arc<ShadowCopies>>,
    /// The turns of textures whose render area the pass's falls short of, whose texels outside
    /// it the pass's first plan copies in ([`EncoderCore::area_syncs`]).
    syncs: Vec<AreaSync>,
    /// The encoder held them.
    held: bool,
}

impl Turns {
    /// The turns of the pass of `layout` whose encoder `core` just opened (the module docs),
    /// `pass` why the pass turns no texture where it turns none for its own reasons, `again`
    /// where the pass is encoded again (`replay.rs`): it takes again the turns its first encoding
    /// took, and no other.
    pub(super) fn open(
        core: &mut EncoderCore,
        layout: &PassLayout,
        pass: Option<Unturned>,
        again: bool,
    ) -> Turns {
        let mut turns = Turns {
            pass,
            ..Turns::default()
        };
        let mut point = None;
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color else {
                continue;
            };
            let texture = &color.target.texture;
            if !texture.has_faces() || index >= MAX_COLOR_ATTACHMENTS {
                continue;
            }
            let checked = match pass {
                Some(why) => Err(why),
                None => color_verdict(layout, index, color),
            };
            let verdict = checked.and_then(|()| {
                let turned = match again {
                    true => texture.retake_turn().ok_or(Unturned::Pass)?,
                    false => {
                        let point = point.get_or_insert_with(|| core.command_buffer().turn_point());
                        texture.turn_face(point)?
                    }
                };
                if let Some(copies) = texture.face_copies() {
                    core.carry_no_further(&copies);
                }
                turns.ahead.push(turned.copies);
                let extent = texture_extent(texture);
                if layout.extent != extent {
                    turns.syncs.push(AreaSync {
                        copies: turned.sync,
                        area: layout.extent,
                        extent,
                        color: index,
                        clears: color.load == LoadAction::Clear,
                    });
                }
                Ok(())
            });
            turns.colors[index] = Some(verdict);
        }
        turns
    }

    /// Whether the pass turned a texture.
    pub(super) fn turned(&self) -> bool {
        !self.ahead.is_empty()
    }

    /// The shadows to hold and the turns whose texels outside the pass's render area to copy
    /// in, at the first call alone; none after.
    pub(crate) fn hold(&mut self) -> (Vec<Arc<ShadowCopies>>, Vec<AreaSync>) {
        match self.held {
            true => (Vec::new(), Vec::new()),
            false => {
                self.held = true;
                (self.ahead.clone(), self.syncs.clone())
            }
        }
    }

    /// The turns of the textures of colors whose render area the pass's falls short of.
    pub(super) fn short_areas(&self) -> impl Iterator<Item = &AreaSync> {
        self.syncs.iter()
    }
}

/// The extent of `texture`'s level 0, as a color attachment's.
fn texture_extent(texture: &crate::Texture) -> [u32; 2] {
    [texture.width(), texture.height()].map(|side| side as u32)
}

/// Why a pass does not turn a color's texture that has faces, for the color itself (the module
/// docs); `Ok` where the turn may go on, the pass's render area the texture's or short of it.
fn color_verdict(layout: &PassLayout, index: usize, color: &ColorTarget) -> Result<(), Unturned> {
    if color.load == LoadAction::Load {
        return Err(Unturned::Load);
    }
    let target = &color.target;
    let texture = &target.texture;
    let extent = texture_extent(texture);
    let itself = (target.level, target.layer, target.layers, target.samples) == (0, 0, 1, 1)
        && layout.layers == 1
        && target.extent == extent
        && layout.extent[0] <= extent[0]
        && layout.extent[1] <= extent[1];
    if !itself {
        return Err(Unturned::Extent);
    }
    if color.resolve.is_some() || color.copied_resolve.is_some() {
        return Err(Unturned::Resolve);
    }
    // No other attachment of the pass renders or resolves into the texture.
    let same = |other: &crate::Texture| other.root_address() == texture.root_address();
    let others = (layout.colors.iter().enumerate())
        .filter(|&(at, _)| at != index)
        .filter_map(|(_, other)| other.as_ref())
        .any(|other| {
            same(&other.target.texture)
                || (other.copied_resolve.as_ref()).is_some_and(|resolve| same(&resolve.texture))
                || (other.resolve.as_ref()).is_some_and(|resolve| {
                    same(&resolve.attachment.texture)
                        || (resolve.copy_to.as_ref()).is_some_and(|to| same(&to.texture))
                })
        });
    let depth = (layout.depth_stencil.as_ref()).is_some_and(|ds| same(&ds.target.texture));
    if others || depth {
        return Err(Unturned::Shared);
    }
    if color.memoryless {
        return Err(Unturned::Memoryless);
    }
    match layout.fails_command_buffer {
        Some(_) => Err(Unturned::Failing),
        None => Ok(()),
    }
}

/// Why a pass did not turn the texture of color `index`, whose texture had no faces as it opened:
/// the pass's own reason, the texture's shape or storage, the color's, else its first attached
/// read, which made its faces. The texture's comes before the color's, so a texture no turn could
/// take counts as such whatever its pass.
fn unfaced_cause(
    turns: &Turns,
    layout: &PassLayout,
    index: usize,
    color: &ColorTarget,
) -> Unturned {
    if let Some(why) = turns.pass {
        return why;
    }
    if let Some(why) = color.target.texture.unfit_for_faces() {
        return why;
    }
    match color_verdict(layout, index, color) {
        Err(why) => why,
        Ok(()) => Unturned::First,
    }
}

/// Count `copies`, planned by `core` before an instance of a pass of `layout`, that copy in the
/// attached reads' shadow of a texture a color of the pass renders (the module docs).
pub(super) fn counted(core: &EncoderCore, layout: &PassLayout, copies: &[ShadowCopy]) {
    let copied_in = |copy: &&ShadowCopy| matches!(copy.step, Step::Into) && copy.clear.is_none();
    if !copies.iter().any(|copy| copied_in(&copy)) {
        return;
    }
    let turns = core.turns();
    let counts = &core.device().shared().counts;
    for (index, color) in layout.colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        let Some(shadow) = color.target.texture.usage_shadow_copies() else {
            continue;
        };
        let verdict = turns.colors.get(index).copied().flatten();
        for copy in copies.iter().filter(copied_in) {
            if copy.copies.shadow_key != shadow.shadow_key {
                continue;
            }
            let cause = match (copy.cause, verdict) {
                (Some(CopyCause::First), Some(Err(why))) => why,
                (Some(CopyCause::First), None) => unfaced_cause(turns, layout, index, color),
                _ => Unturned::Split,
            };
            counts.attached_read_unswapped(cause, copy.texels_copied_in());
        }
    }
}
