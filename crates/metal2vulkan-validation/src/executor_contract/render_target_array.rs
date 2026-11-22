//! The render pass's render target array length both executors draw a case with: the Metal oracle's
//! `MTLRenderPassDescriptor.renderTargetArrayLength` and the runtime candidate's
//! `RenderPassDescriptor::render_target_array_length`, derived once, so the two agree.

use crate::air::meta::{parse_air_fragment_meta, parse_air_vertex_meta, FragRole, VertOutRole};
use crate::case::Stage;

/// The render target array length of a draw of a `stage` case whose module is `air_ll`: 1 when
/// the stage's metadata declares a `[[render_target_array_index]]` the function writes (a vertex
/// function) or reads (a fragment function), which the Metal oracle renders layered into
/// one-slice array targets, and otherwise 0, Metal's default (a pass without layered rendering).
///
/// This is the rule the oracle has drawn every observation with, read from the module's stage
/// metadata by the product's metadata parser: a role behind a function-constant predicate counts
/// when the module's own defaults enable it, not by the case's constants. Changing it changes the
/// oracle's draws, which is a Metal oracle version (`crate::metal::ORACLE_ABI`), not a candidate
/// change.
pub fn render_target_array_length(stage: Stage, air_ll: &str) -> u32 {
    let layered = match stage {
        Stage::Vertex => parse_air_vertex_meta(air_ll).0.is_some_and(|meta| {
            meta.output_roles
                .iter()
                .any(|role| matches!(role, VertOutRole::RenderTargetArrayIndex))
        }),
        Stage::Fragment => parse_air_fragment_meta(air_ll).0.is_some_and(|meta| {
            meta.roles
                .iter()
                .any(|(_, role)| matches!(role, FragRole::RenderTargetArrayIndex))
        }),
        Stage::Kernel | Stage::Object | Stage::Mesh => false,
    };
    u32::from(layered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let path = metal2vulkan_devtools::fixtures_dir().join(name);
        std::fs::read_to_string(path).expect("public fixture")
    }

    /// A vertex function that chooses its slice draws at length 1; a module without the role,
    /// and a kernel, at 0.
    #[test]
    fn a_function_that_chooses_its_slice_draws_at_length_one() {
        let layer = fixture("vertex_amplification_id_layer.ll");
        assert_eq!(render_target_array_length(Stage::Vertex, &layer), 1);
        assert_eq!(render_target_array_length(Stage::Kernel, &layer), 0);
        let plain = fixture("vertex_varying_triangle.ll");
        assert!(!plain.contains("air.render_target_array_index"));
        assert_eq!(render_target_array_length(Stage::Vertex, &plain), 0);
    }
}
