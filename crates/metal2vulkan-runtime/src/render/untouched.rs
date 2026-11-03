//! A render pass that changes no texel records nothing.
//!
//! - **Metal** keeps every attachment's texels in a pass without a draw call that clears nothing:
//!   a `Load` keeps them, a `DontCare` load of
//!   an attachment the pass stores keeps them in a pass without a draw (`clear_reach.rs`, on the M3
//!   Max and an M4), and a store writes back only what the pass's tiles hold, which no draw
//!   touched. A `DontCare` store stores nothing, and resolves are the
//!   only other writes a pass makes. A `Clear` load of a NaN depth stores Metal's NaN word
//!   (`nan_clear.rs`). (A pass that loads and stores a depth and sets a state with depth writes
//!   stores Metal's words without their signs, draw or none, probe `depthstore`; the device's
//!   pass keeps them, announced `signed-depth-test` where the depth may hold a sign.)
//! - **The runtime** ends such a pass without recording it ([`changes_nothing`]): no render pass
//!   instance, load or store, no copy of a shadow or a stand-in, and none of the tracker's uses of
//!   its attachments, so nothing waits on it. A render pass instance would load and store every
//!   attachment over the render area: over a texture whose linear image lies in host pages, its
//!   whole bytes read and written once each. The pass still holds its attachments until its
//!   command buffer completes, as Metal's encoder does, and its encoder's own commands (fence
//!   waits and updates, the residency it asked for) are the encoder's, recorded as ever.
//!   Counted: `WorkCounts::untouched_passes_skipped`.
//! - **Where it applies**: the pass has no draw call (an execution of render commands and a tile
//!   dispatch count as one), no visibility result buffer, one render pass instance that nothing
//!   recorded into, and every attachment (but the runtime's hidden depth) loads with `Load` or
//!   `DontCare` and stores with `Store` or `DontCare` (an unknown store action stores nothing, as
//!   Metal's `DontCare`), resolving into nothing. A `Clear` load clears in a pass without a draw
//!   call, and a resolve writes its texture: such a pass records as before. So does one whose
//!   depth's `Clear` held a NaN (loaded in the
//!   pass, Metal's word written before it, `AspectTarget::written_nan`), and one that
//!   loads separate depth and stencil textures with `DontCare` (`depth_stencil_scratch.rs`),
//!   whose answer in a pass without a draw is not measured (`clear_reach.rs` gives none).
//!   The first group of a pass of several color groups (`render/color_groups.rs`) records as
//!   before too, as its layout holds its own colors alone; each later group is weighed as it ends.

use super::Instance;
use crate::render_pass::PassLayout;
use crate::render_values::{LoadAction, StoreAction};

/// Whether a pass of `layout` changes no texel, nor anything else a later command sees (the
/// module docs): `called` false (no draw call), `instances` one that nothing recorded into, and its
/// attachments' load and store actions as settled (an unknown store made `DontCare`) keeping every
/// texel, no depth written with a NaN before it.
pub(super) fn changes_nothing(layout: &PassLayout, called: bool, instances: &[Instance]) -> bool {
    let keeps = |load: LoadAction, store: StoreAction| {
        matches!(load, LoadAction::Load | LoadAction::DontCare)
            && matches!(store, StoreAction::Store | StoreAction::DontCare)
    };
    // Nothing recorded into its secondary (which a pass that cannot record inline begins as it
    // opens), before it or in its place.
    let unrecorded = match instances {
        [only] => {
            !only.recorded && only.pieces.is_empty() && only.before.is_empty() && !only.inline
        }
        _ => false,
    };
    let colors = layout.colors.iter().flatten().all(|color| {
        keeps(color.load, color.store) && color.resolve.is_none() && color.copied_resolve.is_none()
    });
    let depth_stencil = layout
        .depth_stencil
        .as_ref()
        .filter(|ds| !ds.hidden)
        .is_none_or(|ds| {
            let aspects = [ds.depth.as_ref(), ds.stencil.as_ref()];
            let measured = |load: LoadAction| ds.separate.is_none() || load == LoadAction::Load;
            ds.resolve.is_none()
                && aspects.into_iter().flatten().all(|aspect| {
                    keeps(aspect.load, aspect.store) && measured(aspect.load) && !aspect.written_nan
                })
        });
    !called && unrecorded && layout.visibility.is_none() && colors && depth_stencil
}
