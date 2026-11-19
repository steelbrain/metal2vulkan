//! The rt side's pipeline state facts: what a compute and a drawing state the runtime made answer
//! of themselves.
//!
//! A state answers the getters the runtime has for it. The runtime keeps no label and no device on
//! a state (both the consumer's own), and it makes no tile render state, so a drawing state has
//! none of the three tile getters (`maxTotalThreadsPerThreadgroup`, a tile threadgroup's,
//! `threadgroupSizeMatchesTileSize` and `requiredThreadsPerTileThreadgroup`): each is a refused
//! field, and the state's other getters compare.

use super::RtSide;
use crate::case::{Answer, Failure, Field, Handle};
use crate::pipeline_facts::{ComputeStateFacts, RenderStateFacts};
use metal2vulkan_rt as rt;

/// `label`: the runtime keeps none on a state.
fn label() -> Field<Option<String>> {
    Err(Failure::unavailable("label"))
}

/// `device`: the runtime keeps none on a state.
fn device() -> Field<bool> {
    Err(Failure::unavailable("device"))
}

/// A tile getter `call` of a drawing state: the runtime makes no tile render state, so it has
/// none.
fn tile<T>(call: &str) -> Field<T> {
    Err(Failure::unavailable(call))
}

impl RtSide<'_> {
    /// [`crate::case::Side::compute_pipeline_facts`].
    pub(super) fn compute_state_facts(
        &self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<ComputeStateFacts> {
        let held = self.held_pipeline(state)?;
        Ok(ComputeStateFacts {
            label: label(),
            device_is_the_cases: device(),
            static_threadgroup_memory_length: u64::from(held.static_threadgroup_memory_length()),
            support_indirect_command_buffers: held.support_indirect_command_buffers(),
            imageblock_memory_length: (held.imageblock_memory_length_for_dimensions(imageblock))
                .map_err(|error| self.on.failure(&error)),
            required_threads_per_threadgroup: held
                .required_threads_per_threadgroup()
                .map(u64::from),
        })
    }

    /// [`crate::case::Side::render_pipeline_facts`] of a drawing state. A length the runtime
    /// answers a typed error for is `None`, as the facts hold it.
    pub(super) fn render_state_facts(
        &self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<RenderStateFacts> {
        let held: &rt::RenderPipelineState = (self.render_pipelines.get(state.0 as usize))
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("render pipeline {}", state.0))
            })?;
        Ok(RenderStateFacts {
            label: label(),
            device_is_the_cases: device(),
            max_total_threads_per_threadgroup: tile("maxTotalThreadsPerThreadgroup"),
            threadgroup_size_matches_tile_size: tile("threadgroupSizeMatchesTileSize"),
            imageblock_sample_length: held.imageblock_sample_length().ok().map(u64::from),
            imageblock_memory_length: held
                .imageblock_memory_length_for_dimensions(imageblock)
                .ok(),
            support_indirect_command_buffers: held.support_indirect_command_buffers(),
            required_threads_per_tile_threadgroup: tile("requiredThreadsPerTileThreadgroup"),
        })
    }
}
