//! The validation layer's message about a vertex module's subgroup operations on a driver that
//! runs them as Metal does though its `subgroupSupportedStages` leaves the vertex stage out.
//!
//! MoltenVK 1.4.1 hands a vertex function to Metal, which runs it in simdgroups of 32 consecutive
//! vertices; measured on the M3 Max and the M4, its subgroup operations there are Metal's
//! (`DriverDeviation::MetalVertexSimdgroups`). A profile that takes that deviation compiles them,
//! and the module requires `VkProperty::MetalVertexSimdgroups` in place of the vertex stage. The
//! layer checks the stage against `subgroupSupportedStages`, as Vulkan says it must
//! (`VUID-RuntimeSpirv-None-06343`, "Group operations with subgroup scope must not be used if the
//! shader stage is not in subgroupSupportedStages"), and so reports each such shader module. That is
//! no layer defect: the use is outside what the driver advertises, and the deviation is what
//! licenses it.
//!
//! [`shape`] matches exactly that message about the vertex stage alone: the layer's format,
//! "`<opcode>` is used in `<stages>` which is not supported in subgroupSupportedStages
//! (`<stages>`).", read in Vulkan-ValidationLayers v1.4.357 (its library's strings), with the
//! stages `[VK_SHADER_STAGE_VERTEX_BIT]` at `vkCreateShaderModule`, as the lanes saw it, or
//! `VK_SHADER_STAGE_VERTEX_BIT`. The executor's context's sink passes it only where the profile it
//! opened takes the deviation (`platform/device.rs`). Every other subgroup-stage message, and this
//! one anywhere else, still fails.

/// The message's id name.
pub(crate) const ID_NAME: &str = "VUID-RuntimeSpirv-None-06343";

/// What the message says of a vertex module: at `vkCreateShaderModule` the stages of the
/// module's entry points, bracketed (as the lanes saw it on MoltenVK); at
/// pipeline creation the stage alone.
const VERTEX: [&str; 2] = [
    " is used in [VK_SHADER_STAGE_VERTEX_BIT] which is not supported in subgroupSupportedStages (",
    " is used in VK_SHADER_STAGE_VERTEX_BIT which is not supported in subgroupSupportedStages (",
];

/// Whether a message with this id name and text is the layer's report of a vertex module's
/// subgroup operations outside the advertised stages.
pub(crate) fn shape(id_name: &str, text: &str) -> bool {
    id_name == ID_NAME && VERTEX.iter().any(|vertex| text.contains(vertex))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message as the layer gave it for a vertex module on MoltenVK 1.4.1 (layer 1.4.357), of
    /// the stages `stages`.
    fn seen(stages: &str) -> String {
        format!(
            "vkCreateShaderModule(): OpGroupNonUniformIAdd is used in {stages} which is not \
             supported in subgroupSupportedStages (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|\
             VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT).\nThe Vulkan spec \
             states: Group operations with subgroup scope must not be used if the shader stage \
             is not in subgroupSupportedStages (https://docs.vulkan.org/spec/latest/appendices/\
             spirvenv.html#VUID-RuntimeSpirv-None-06343)"
        )
    }

    #[test]
    fn the_vertex_stage_s_message_is_the_shape() {
        assert!(shape(ID_NAME, &seen("[VK_SHADER_STAGE_VERTEX_BIT]")));
        assert!(shape(ID_NAME, &seen("VK_SHADER_STAGE_VERTEX_BIT")));
    }

    #[test]
    fn another_stage_or_message_is_not() {
        for stages in [
            "[VK_SHADER_STAGE_FRAGMENT_BIT]",
            "[VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT]",
            "[VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_GEOMETRY_BIT]",
            "VK_SHADER_STAGE_GEOMETRY_BIT",
        ] {
            assert!(!shape(ID_NAME, &seen(stages)), "{stages}");
        }
        let text = seen("[VK_SHADER_STAGE_VERTEX_BIT]");
        assert!(!shape("VUID-RuntimeSpirv-None-06342", &text));
        assert!(!shape("", &text));
    }
}
