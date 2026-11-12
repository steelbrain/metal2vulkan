//! A render pass's turns of the textures its colors render to their other faces
//! (`texture/faces.rs`), decided as the pass opens, before anything of it names their images; and
//! the counts of the attached reads' copies in that no turn took off
//! (`WorkCounts::attached_read_unswapped_*`).
//!
//! - **At the open** ([`Turns::open`]). A color whose texture has faces turns it where the pass
//!   turns textures at all (it encodes no encoder again, draws its colors in one group, over no
//!   rasterization-rate map, logs none of its calls, and no ended encoder of its command buffer
//!   keeps a log a rebind encodes again: each would record commands again that name the face
//!   presented on one side of the turn on the other), and the color loads `Clear` or `DontCare`
//!   and renders the whole texture: level 0 and slice 0, one layer, the render area the texture's
//!   extent, single-sampled, resolving nowhere, no other attachment of the pass the same texture,
//!   and the pass failing no command buffer. The turn checks the rest
//!   ([`Texture::turn_face`](crate::Texture)). A turned texture's faces' shadows are carried no
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

use crate::encoder::EncoderCore;
use crate::render_pass::{ColorTarget, PassLayout, MAX_COLOR_ATTACHMENTS};
use crate::render_values::LoadAction;
use crate::shadow::{CopyCause, ShadowCopies, ShadowCopy, Step};
use crate::texture::faces::Unturned;
use std::sync::Arc;

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
    /// The encoder held them.
    held: bool,
}

impl Turns {
    /// The turns of the pass of `layout` whose encoder `core` just opened (the module docs),
    /// `pass` why the pass turns no texture where it turns none for its own reasons.
    pub(super) fn open(
        core: &mut EncoderCore,
        layout: &PassLayout,
        pass: Option<Unturned>,
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
                let point = point.get_or_insert_with(|| core.command_buffer().turn_point());
                if point.logs_kept {
                    return Err(Unturned::Pass);
                }
                let turned = texture.turn_face(point)?;
                if let Some(copies) = texture.face_copies() {
                    core.carry_no_further(&copies);
                }
                turns.ahead.push(turned.copies);
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

    /// The shadows to hold, at the first call alone; none after.
    pub(crate) fn hold(&mut self) -> Vec<Arc<ShadowCopies>> {
        match self.held {
            true => Vec::new(),
            false => {
                self.held = true;
                self.ahead.clone()
            }
        }
    }
}

/// Why a pass does not turn a color's texture that has faces, for the color itself (the module
/// docs); `Ok` where the turn may go on.
fn color_verdict(layout: &PassLayout, index: usize, color: &ColorTarget) -> Result<(), Unturned> {
    if color.load == LoadAction::Load {
        return Err(Unturned::Load);
    }
    let target = &color.target;
    let texture = &target.texture;
    let extent = [texture.width(), texture.height()].map(|side| side as u32);
    let whole = (target.level, target.layer, target.layers, target.samples) == (0, 0, 1, 1)
        && layout.layers == 1
        && target.extent == extent
        && layout.extent == extent
        && color.resolve.is_none()
        && color.copied_resolve.is_none()
        && !color.memoryless;
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
    match whole && !others && !depth && layout.fails_command_buffer.is_none() {
        true => Ok(()),
        false => Err(Unturned::Area),
    }
}

/// Why a pass did not turn the texture of color `index`, whose texture had no faces as it opened:
/// the pass's own reason, the color's, the texture's shape or storage, else its first attached
/// read, which made its faces.
fn unfaced_cause(
    turns: &Turns,
    layout: &PassLayout,
    index: usize,
    color: &ColorTarget,
) -> Unturned {
    if let Some(why) = turns.pass {
        return why;
    }
    if let Err(why) = color_verdict(layout, index, color) {
        return why;
    }
    color
        .target
        .texture
        .unfit_for_faces()
        .unwrap_or(Unturned::First)
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
