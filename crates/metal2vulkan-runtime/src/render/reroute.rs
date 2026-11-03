//! Recording a render pass instance's draws again, into an instance that leaves more colors out
//! of its subpass, in its place: the start of an extended-range color's `half` shadow where the
//! current instance's draws wrote the color (`render/extended_shadow.rs`).
//!
//! In a pass that may hold such a shadow (`ExtendedShadows::keeps_draws`), on every device, while
//! every recording of the current instance is a draw that can record again (below), each draw
//! keeps the encoder's state at it (`Rerecord`). A shadow's start records the instance's draws
//! again first into an instance that leaves the color out (`record_leaving_out`), which takes the
//! current one's place: each draw through the variant that reads and writes the shadow
//! (`BindState::extended_shadows`).
//!
//! Exact where it happens: a draw whose variant the storage form or the bind layer refuses fails
//! the reroute, which leaves the instance and the pass's layout as they were, and the color keeps
//! its codes. The rerouted instance is the one `next_instance` makes for a draw that fetches every
//! color it leaves out, as if the current instance held nothing: a fetched texture without
//! `STORAGE` renders into its stand-in from that instance on (`attachment_stand_in.rs`), and the
//! pass's first instance reroutes only where each color it leaves out that the pass clears can be
//! cleared before the pass, as a first instance that leaves one out must.
//!
//! A draw records again only where nothing but its own recording depends on when it was
//! recorded: a direct draw of primitives, or of indices read in place, in a single-sampled pass
//! without a rasterization-rate map or an active visibility query, of a vertex and fragment
//! pipeline without tessellation, writing no memory of its own (a color's interlocked storage
//! form and `half` shadow are the product's, `GraphicsState::fragment_writes_memory`), reaching no
//! imageblock, tile memory or shader-side stencil, drawing no views, using no shadowed texture,
//! taking no steps before its instance and no stand-in or hidden depth arriving with it; at most
//! `MAX_DRAWS` per instance. An instance's pieces are then such draws alone: a draw whose pipeline
//! the device's pool was making records into a piece of its own (`render/pending.rs`), and an
//! execution of render commands ends the instance's recording again. What a first recording noted
//! stays noted (its uses, uploads, retained objects and, for a draw whose pipeline was being made,
//! that making, which the command buffer waits for and whose failure fails it: an
//! over-approximation), and the secondary command buffers it recorded into are never executed. A
//! draw recorded again makes its pipeline at the draw, not on the pool, so a reroute that fails
//! leaves no making behind. The kept state holds the encoder's tables and textures, never a Vulkan
//! view or image of its own, so a texture's storage form taking another image after the command
//! buffer commits (`texture/storage_form.rs`) changes nothing it records.

use super::{Assembly, DrawBuffers, Next, RenderLowering};
use crate::amplification::Amplification;
use crate::argument_table::ArgumentTable;
use crate::draw_source::{DrawSource, KeptSource};
use crate::error::Error;
use crate::pipeline::{GraphicsState as DrawingState, RenderPipelineState};
use crate::render_state::EncoderState;
use crate::render_target::PassShape;
use crate::residency::Residency;
use metal2vulkan::bind::{DrawCall, ViewSetup};
use std::sync::Arc;

/// The most draws an instance keeps for recording again: past them its draws record again no
/// more, which bounds what an instance holds.
pub(super) const MAX_DRAWS: usize = 64;

/// A draw recorded into the current instance, and the encoder's state at it: what records it
/// again.
pub(super) struct Rerecord {
    call: &'static str,
    assembly: Assembly,
    draw: DrawCall,
    /// Its indices, read in place.
    index: Option<KeptSource>,
    held: Held,
}

/// The encoder's state a draw reads, as it stood at the draw.
struct Held {
    pipeline: Option<RenderPipelineState>,
    state: EncoderState,
    vertex: ArgumentTable,
    fragment: ArgumentTable,
    residency: Residency,
    amplification: Amplification,
}

impl RenderLowering {
    /// Keep the draw just recorded into the current instance (`call` of `assembly` and `draw`,
    /// its indices from `index`) for recording again where `again` (the module docs) and the
    /// instance keeps fewer than `MAX_DRAWS`; otherwise the instance's draws record again no
    /// more.
    pub(super) fn keep_draw(
        &mut self,
        again: bool,
        (call, assembly, draw): (&'static str, Assembly, DrawCall),
        index: Option<&DrawSource<'_>>,
    ) {
        let index = match index {
            Some(source) => source.kept().map(Some),
            None => Some(None),
        };
        let room = self
            .current
            .rerecord
            .as_ref()
            .is_some_and(|kept| kept.len() < MAX_DRAWS);
        match (again && room, index) {
            (true, Some(index)) => {
                let kept = Rerecord {
                    call,
                    assembly,
                    draw,
                    index,
                    held: self.held(),
                };
                if let Some(draws) = self.current.rerecord.as_mut() {
                    draws.push(kept);
                }
            }
            _ => self.current.rerecord = None,
        }
    }

    /// Record the current instance's draws again into an instance that leaves `colors`
    /// (ascending) out of its subpass, in its place, where its draws can (the module docs):
    /// whether it did. The instance is the one `next_instance` makes for a draw that fetches
    /// them all, its stand-ins included, as if the current one held nothing, a new one even
    /// where its shape is the current one's; nothing where it would follow the current one (the
    /// pass's first instance leaving out a color it cannot clear before the pass, among others)
    /// or the draws cannot record so, which leaves the instance and the pass's layout as they
    /// were.
    pub(super) fn record_leaving_out(
        &mut self,
        colors: Vec<u32>,
        by: &DrawingState,
    ) -> Result<bool, Error> {
        let current = &self.current;
        let kept = current
            .rerecord
            .as_ref()
            .is_some_and(|draws| !draws.is_empty());
        if self.replaying
            || self.rerouting
            || !current.recorded
            || !kept
            || self.layout.fetch_inputs
            || self.layout.samples != 1
            || self.dispatched
            || self.barrier
            || current.shadow
            || current.shader_tests
            || current.imageblock
            || current.tile_memory
            || current.views != ViewSetup::default()
            || !current.reformats.is_empty()
            || !current.before.is_empty()
        {
            return Ok(false);
        }
        let locations = current.shape.locations.clone();
        // `rerouting` has `next_instance` take the current instance's place as an empty one's.
        // Ordered after the current instance's draws: a new instance even where its shape is
        // the current one's (a shadow's start, whose color the current instance may already
        // leave out). The draws recorded again then record into it as their current one.
        self.rerouting = true;
        let made = self
            .next_instance(
                (&colors, Some(by)),
                (false, false),
                ViewSetup::default(),
                true,
                None,
                &locations,
                &[],
            )
            .and_then(|(mut next, stood)| {
                self.begin_new_draws(&mut next)?;
                Ok((next, stood))
            });
        let (instance, stood) = match made {
            Ok((
                Next::New {
                    instance,
                    split: false,
                },
                stood,
            )) => (instance, stood),
            Ok(_) | Err(_) => {
                self.rerouting = false;
                return Ok(false);
            }
        };
        let previous_layout = stood.map(|layout| std::mem::replace(&mut *self.layout, *layout));
        self.kept_attachments = None;
        let mut previous = std::mem::replace(&mut self.current, *instance);
        let mut again = Ok(());
        if let Some(draws) = previous.rerecord.as_mut() {
            for kept in draws.iter_mut() {
                again = self.record_again(kept);
                if again.is_err() {
                    break;
                }
            }
        }
        self.rerouting = false;
        match again {
            Ok(()) => {
                let shape = &self.current.shape;
                if !self.passes.iter().any(|(s, _)| *s == **shape) {
                    let pass = (
                        PassShape::clone(shape),
                        Arc::clone(&self.current.compatible),
                    );
                    self.passes.push(pass);
                }
                // The instance it took the place of is never executed.
                self.end_unrun(&previous);
                Ok(true)
            }
            // A draw that cannot record so leaves the instance and the pass's layout as they
            // were: the rerouted instance is never executed.
            Err(_) => {
                let rerouted = std::mem::replace(&mut self.current, previous);
                self.end_unrun(&rerouted);
                if let Some(layout) = previous_layout {
                    *self.layout = layout;
                    self.kept_attachments = None;
                }
                Ok(false)
            }
        }
    }

    /// Record `kept` again into the current instance, with the encoder's state at it.
    pub(super) fn record_again(&mut self, kept: &mut Rerecord) -> Result<(), Error> {
        let index = kept.index.as_ref().map(KeptSource::source).transpose()?;
        let buffers = DrawBuffers {
            index,
            ..DrawBuffers::default()
        };
        self.swap_held(&mut kept.held);
        let drawn = self.draw_resident(kept.call, kept.assembly, kept.draw, buffers, false);
        self.swap_held(&mut kept.held);
        drawn
    }

    /// The encoder's state a draw reads, as it stands.
    fn held(&self) -> Held {
        Held {
            pipeline: self.pipeline.clone(),
            state: self.state.clone(),
            vertex: self.vertex.clone(),
            fragment: self.fragment.clone(),
            residency: self.residency.clone(),
            amplification: self.amplification.clone(),
        }
    }

    /// Exchange the encoder's state a draw reads with `held`.
    fn swap_held(&mut self, held: &mut Held) {
        std::mem::swap(&mut self.pipeline, &mut held.pipeline);
        std::mem::swap(&mut *self.state, &mut held.state);
        std::mem::swap(&mut self.vertex, &mut held.vertex);
        std::mem::swap(&mut self.fragment, &mut held.fragment);
        std::mem::swap(&mut self.residency, &mut held.residency);
        std::mem::swap(&mut *self.amplification, &mut held.amplification);
    }
}
