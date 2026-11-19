//! The rt side's visible-function-table call, as far as the runtime's public API reaches: the
//! state of the case's stage with its linked functions. The runtime has no function handles and
//! no visible-function tables, so a state it makes answers the next call unavailable.

use super::RtSide;
use crate::case::{Answer, Failure, Handle};
use crate::visible_tables::{LinkedList, Maker, TableOutcome, TableSpec};
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan_rt as rt;

impl RtSide<'_> {
    /// [`crate::case::Side::visible_function_table`]: the runtime's state of `spec.entry` of
    /// `caller`, linking `spec.linked`'s functions of `functions` in the order the spec states,
    /// made as the case's maker makes it: a compute state, or a drawing state whose vertex
    /// function is the entry, with no fragment function and rasterization off. The runtime's
    /// refusal of the state is the answer. A state it makes answers `functionHandleWithFunction:`
    /// (`functionHandleWithFunction:stage:` of a drawing state) unavailable, and a tile stage
    /// answers the tile state unavailable, as the runtime makes none.
    pub(super) fn table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &TableSpec<'_>,
    ) -> Answer<TableOutcome> {
        let on = self.on;
        let translated = |error: metal2vulkan::error::Error| on.translation_failure(&error);
        let defining = self.held_library(functions)?;
        let mut linked = LinkedFunctions::new();
        for (name, list) in spec.linked.functions {
            let function = defining.function(name).map_err(translated)?;
            linked = match list {
                LinkedList::Functions => linked.with_function(function),
                LinkedList::PrivateFunctions => linked.with_private_function(function),
            };
        }
        let entry = (self.held_library(caller)?.function(spec.entry)).map_err(translated)?;
        let stage =
            rt::StageSource::Function(rt::FunctionDescriptor::new(entry).with_linked(linked));
        let next = match spec.maker {
            Maker::Compute => {
                let descriptor = rt::ComputePipelineDescriptor::new(stage);
                (on.device)
                    .new_compute_pipeline_state(
                        &descriptor,
                        |samplers| Ok(on.sampler_ids(samplers)),
                    )
                    .map_err(|error| on.failure(&error))?;
                "functionHandleWithFunction:"
            }
            Maker::DrawingStage(_) => {
                let mut descriptor = rt::RenderPipelineDescriptor::new(stage);
                descriptor.rasterization_enabled = false;
                (on.device)
                    .new_render_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
                    .map_err(|error| on.failure(&error))?;
                "functionHandleWithFunction:stage:"
            }
            Maker::TileStage(_) => "tile_pipeline",
        };
        Err(Failure::unavailable(next))
    }
}
