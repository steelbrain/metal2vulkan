//! The bytes of a buffer that textures were made over, which the pooled bound reads
//! (`in_place::POOLED_PITCHED_BYTES`, `texture.rs`'s `pooled_pitched_image`).
//!
//! - **Why.** Past the bound, a tracked texture over a `Private` buffer is a shadow of its linear
//!   cost (`ShadowCause::LinearCost`): a `DRM_FORMAT_MOD_LINEAR` image renders and samples slower
//!   than the shadow's own image, and a shadow whose truth only the runtime's commands write
//!   stands across command buffers, copied in once (`shadow_standing.rs`). That holds for one
//!   texture over the bytes. A shadow is its texture's own: every other texture over the same
//!   bytes (a surface's textures of other formats, made again each frame, its views) makes an
//!   image of its own, whose first use copies every texel in, and a write through any of them is
//!   a write of the bytes that lets go of every other shadow's standing, whose next use copies
//!   every texel in again. A texture over bytes another texture was made over takes the image at
//!   Metal's pitch instead, at any size, as over memory the CPU writes: one image over the bytes
//!   per texture, no copy, the bytes' own aliasing.
//! - **What is kept.** The runs of bytes textures were made over ([`TextureSpans::note`]), each
//!   the texels' first byte to their last, runs that meet joined, at most [`RUNS`] of them: past
//!   that, a new run joins the nearest, so a run may hold bytes no texture was made over. That
//!   errs toward the image in place, which is as exact as the shadow. A texture's making notes its
//!   run once its storage is decided, whatever it took, over any buffer; the placement query reads
//!   the runs and notes nothing.
//! - **The first shadow moves.** A shadow of its linear cost made over bytes no other texture was
//!   made over waits ([`LoneShadow`]): once a later texture is made over its bytes, it moves to the
//!   image at Metal's pitch as a shadow made before its pooled memory's verdicts does
//!   (`texture/storage_form.rs`: at the commit of the device's only uncommitted command buffer,
//!   with no copy), where the device makes that image then. A view's shadow never moves, as a
//!   view's storage never does. One waiting shadow is kept per run at most: a second texture over
//!   it takes the waiting one.

use super::in_place::PendingMove;
use std::any::Any;
use std::ops::Range;
use std::sync::Weak;

/// The most runs a buffer keeps apart (the module docs).
const RUNS: usize = 8;

/// The bytes of a buffer that textures were made over, and the shadows of their linear cost
/// that wait for another texture over their bytes (the module docs).
#[derive(Default)]
pub(super) struct TextureSpans {
    runs: Vec<Range<u64>>,
    lone: Vec<LoneShadow>,
}

/// A shadow of its linear cost made over bytes no other texture was made over: its texture,
/// held weakly and by type erased (`texture.rs`'s `TextureShared`, whose device holds buffers),
/// what it moves with, and its texels' bytes.
pub(super) struct LoneShadow {
    pub(super) texture: Weak<dyn Any + Send + Sync>,
    pub(super) pending: PendingMove,
    bytes: Range<u64>,
}

/// Whether two runs of bytes share one.
fn meet(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

impl TextureSpans {
    /// Whether a texture was made over any of `bytes`.
    pub(super) fn made_over(&self, bytes: &Range<u64>) -> bool {
        self.runs.iter().any(|run| meet(run, bytes))
    }

    /// Note a texture made over `bytes`: the shadows that waited for another texture over them,
    /// which stop waiting here and are the caller's to move (the module docs).
    pub(super) fn note(&mut self, bytes: Range<u64>) -> Vec<LoneShadow> {
        let mut joined = bytes.clone();
        self.runs.retain(|run| {
            let meets = meet(run, &joined);
            if meets {
                joined = joined.start.min(run.start)..joined.end.max(run.end);
            }
            !meets
        });
        if self.runs.len() == RUNS {
            // The run nearest the new one (by the gap between them) takes it.
            let gap = |run: &Range<u64>| match run.end <= joined.start {
                true => joined.start - run.end,
                false => run.start.saturating_sub(joined.end),
            };
            if let Some(near) = (0..self.runs.len()).min_by_key(|&i| gap(&self.runs[i])) {
                let run = self.runs.swap_remove(near);
                joined = joined.start.min(run.start)..joined.end.max(run.end);
            }
        }
        self.runs.push(joined);
        self.lone.retain(|lone| lone.texture.strong_count() > 0);
        self.lone
            .extract_if(.., |lone| meet(&lone.bytes, &bytes))
            .collect()
    }

    /// A shadow of its linear cost was made over `bytes`, which no other texture was made over:
    /// it waits for one (the module docs).
    pub(super) fn wait(
        &mut self,
        bytes: Range<u64>,
        texture: Weak<dyn Any + Send + Sync>,
        pending: PendingMove,
    ) {
        self.lone.retain(|lone| lone.texture.strong_count() > 0);
        self.lone.push(LoneShadow {
            texture,
            pending,
            bytes,
        });
    }
}
