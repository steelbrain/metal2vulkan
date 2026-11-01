//! A render encoder's vertex amplification (`setVertexAmplificationCount:viewMappings:`): the count
//! and the view mappings the draws after it take.
//!
//! The binder takes them as `BindState::vertex_amplification` and plans the draws' views
//! (`bind::ViewSetup`): Vulkan multiview over the pass's layers from the one the first mapping
//! names, or without multiview into that one layer, or, where multiview does not draw them as Metal
//! does, one draw per amplification (`render.rs`). Metal's validation stops a draw whose count is 0
//! or above its pipeline's `maxVertexAmplificationCount`, not the setter (a Metal probe on an M4:
//! "Vertex Amplification Count (3) must be between (inclusive) 1 and the maximum vertex
//! amplification count specified in the pipeline state (2)"), so the encoder holds any count. A
//! draw of a count of 0 draws nothing, as plain Metal draws it (probe 13 `amp0`). One above a
//! pipeline's maximum of 2 or more, up to Metal's 8, draws that many amplifications, as plain Metal
//! draws them (probe 13 `amp3_pipeline2`, a probe `amp_*` on an M4: 3 and 4 on a maximum of 2, 4 on
//! 3, 8 on 2 and on 4 rendered every amplification through its mapping, and `nil` mappings each
//! into the first slice, the last one's texels kept). One above a maximum of 1 is refused: plain
//! Metal stored garbage (probe 13 `amp2_pipeline1`, probe 2 `amp_2_on_1`, `amp_3_on_1`). Metal
//! reads `count` mappings from its pointer; a slice of fewer is refused at the setter, and one of
//! more gives its first `count`. `setVertexAmplificationMode:value:` sets the count and keeps the
//! mappings ([`Amplification::with_count`], `apple_only.rs`).

use super::capabilities::MAX_VERTEX_AMPLIFICATION;
use super::render::DrawProblem;
use metal2vulkan::bind::{VertexAmplification, ViewMapping};

/// `MTLVertexAmplificationViewMapping`: the offsets an amplification adds to the viewport and
/// render target array indices it renders through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VertexAmplificationViewMapping {
    /// `viewportArrayIndexOffset`.
    pub viewport_array_index_offset: u32,
    /// `renderTargetArrayIndexOffset`.
    pub render_target_array_index_offset: u32,
}

impl VertexAmplificationViewMapping {
    /// The mapping of these offsets.
    pub fn new(
        viewport_array_index_offset: u32,
        render_target_array_index_offset: u32,
    ) -> VertexAmplificationViewMapping {
        VertexAmplificationViewMapping {
            viewport_array_index_offset,
            render_target_array_index_offset,
        }
    }
}

/// The encoder's vertex amplification: Metal's default, a count of 1 without mappings, until
/// set.
#[derive(Clone, Debug)]
pub(super) struct Amplification {
    count: u64,
    /// The first `count` mappings; empty for `nil`.
    mappings: Vec<VertexAmplificationViewMapping>,
}

impl Default for Amplification {
    fn default() -> Amplification {
        Amplification {
            count: 1,
            mappings: Vec::new(),
        }
    }
}

impl Amplification {
    /// The amplification of `count` with `mappings` (`None` for `nil`). Refused: fewer mappings
    /// than the count.
    pub(super) fn new(
        count: u64,
        mappings: Option<&[VertexAmplificationViewMapping]>,
    ) -> Result<Amplification, DrawProblem> {
        let mappings = match mappings {
            None => Vec::new(),
            Some(given) => {
                let taken = usize::try_from(count)
                    .ok()
                    .and_then(|count| given.get(..count))
                    .ok_or(DrawProblem::ViewMappings {
                        count,
                        given: given.len() as u64,
                    })?;
                taken.to_vec()
            }
        };
        Ok(Amplification { count, mappings })
    }

    /// The amplification of `count` with these view mappings kept
    /// (`setVertexAmplificationMode:value:`, `apple_only.rs`): `None` where there are mappings
    /// and the count passes them, which Metal's measurements do not reach.
    pub(super) fn with_count(&self, count: u64) -> Option<Amplification> {
        let mappings = match self.mappings.is_empty() {
            true => Vec::new(),
            false => self.mappings.get(..usize::try_from(count).ok()?)?.to_vec(),
        };
        Some(Amplification { count, mappings })
    }

    /// Whether a draw draws nothing: a count of 0 (the module docs).
    pub(super) fn draws_nothing(&self) -> bool {
        self.count == 0
    }

    /// Whether a draw draws more than one amplification.
    pub(super) fn amplifies(&self) -> bool {
        self.count > 1
    }

    /// Whether a mapping moves an amplification to another viewport.
    pub(super) fn offsets_viewports(&self) -> bool {
        self.mappings
            .iter()
            .any(|m| m.viewport_array_index_offset != 0)
    }

    /// The amplification as the binder takes it, for a draw of a pipeline whose
    /// `maxVertexAmplificationCount` is `max`. Refused (the module docs): a count above 1 where
    /// `max` is 1, and one above Metal's 8 (unmeasured); a count of 0 draws nothing before it is
    /// asked for (`draws_nothing`).
    pub(super) fn bind_state(&self, max: u32) -> Result<VertexAmplification, DrawProblem> {
        let count = u32::try_from(self.count)
            .ok()
            .filter(|&count| {
                (1..=max).contains(&count)
                    || (max > 1 && (1..=MAX_VERTEX_AMPLIFICATION).contains(&u64::from(count)))
            })
            .ok_or(DrawProblem::AmplificationCount {
                count: self.count,
                max,
            })?;
        let mappings = self
            .mappings
            .iter()
            .map(|m| {
                ViewMapping::new(
                    m.viewport_array_index_offset,
                    m.render_target_array_index_offset,
                )
            })
            .collect();
        Ok(VertexAmplification::new(count, mappings))
    }
}
