use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// An executor of authored cases: the Metal oracle that qualifies a case, and the Vulkan candidate
/// whose observations are compared against it. The old engine's candidate was retired with the old
/// engine, so a closed authoring-capability census needs every executor to run every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Executor {
    MetalOracle,
    /// The engine's candidate (`candidate::engine`), which runs the shipping product.
    EngineCandidate,
}

impl Executor {
    pub const ALL: [Self; 2] = [Self::MetalOracle, Self::EngineCandidate];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MetalOracle => "metal_oracle",
            Self::EngineCandidate => "engine_candidate",
        }
    }
}

impl fmt::Display for Executor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Whose limit a [`ToolingRequirement`] records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Limit {
    /// This crate's tooling does not yet express, run, or observe the structure: work to model.
    /// Its rows keep the census open.
    Tooling,
    /// Metal itself cannot build a pipeline for the structure: Apple's compiler fails, as measured
    /// on the reference device, so no Metal answer exists for any tooling to produce. It stops
    /// every executor (no case can be qualified), and its rows are classified and closed without a
    /// reference: the census counts them apart and never as unresolved.
    Metal,
}

impl Limit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tooling => "tooling",
            Self::Metal => "metal",
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A concrete gap between harvested AIR structure and the authored execution contract.
///
/// This is the shared vocabulary for classification, schema/executor capability checks, cached
/// census rows, and focused audits. Keeping it typed prevents one component from silently
/// inventing a requirement name that another component cannot select or resolve. Each
/// requirement names the executors it stops ([`ToolingRequirement::stops`]): an authoring
/// requirement (no case can express or observe the structure) stops both, and an executor
/// requirement only the executor that does not implement the family. Each also names whose limit it
/// is ([`ToolingRequirement::limit`]): this crate's tooling, or Metal's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolingRequirement {
    FunctionConstantLiteral,
    VertexAttributeLiteral,
    TessellationAttributeLiteral,
    TessellationSystemInputLiteral,
    FragmentImageblockMemberLiteral,
    KernelStageInputLiteral,
    TextureBufferLiteral,
    StorageTextureFormatLiteral,
    ImplicitImageblockLiteral,
    IndirectCommandBuffer,
    VertexSideEffectObservation,
    FragmentVaryingObservationType,
    FragmentOutputObservationType,
    /// An intersection-function table held in an argument buffer.
    ArgumentBufferIntersectionFunctionTable,
    /// An intersection-function table entry that links an intersection function.
    LinkedIntersectionFunction,
    /// A device-buffer array bound to a vertex or fragment function.
    RenderDeviceBufferArray,
    /// An acceleration structure or intersection-function table bound to a vertex or fragment
    /// function.
    RenderAccelerationStructure,
    /// A depth/stencil attachment of a vertex case.
    VertexDepthStencilAttachment,
    /// Render targets of a rasterization-disabled vertex case.
    RasterizationDisabledRenderTarget,
    /// A coordinate-mapping call reads its rasterization-rate map from somewhere other than
    /// offset 0 of an entry buffer parameter (`executor_contract::rate_map`): the
    /// `rasterization_rate_map` literal places a map only there, so the map stays authored bytes,
    /// which the engine's module decodes as this crate's parameter data.
    RasterizationRateMapPlacement,
    /// A compute kernel that samples a texture at an implicit level with a positive (or NaN)
    /// constant or run-time `min_lod_clamp`, in a form measured to crash Apple's compiler service
    /// building its compute pipeline (`executor_contract::min_lod_clamp`), so no Metal reference
    /// exists for it ([`Limit::Metal`]).
    KernelMinLodClamp,
    /// A stage of a mesh render pipeline that its row or case needs and nothing supplies
    /// (`executor_contract::mesh`): the mesh stage of an object function whose payload the
    /// generated launch observer cannot read, the object stage of a mesh function whose payload no
    /// generated object function writes, or a row's fragment stage where its mesh function's
    /// observable effect is its rasterized output. A case supplies one by naming a partner of its
    /// own module (`MeshDraw`, whose rasterizing draw must name its fragment function), or by
    /// stating the observation or object launch whose generated stages take the partners' places
    /// (`companion::MeshStages`).
    MeshCompanion,
    /// A case gives a 32-bit float depth image a depth outside `[0, 1]`
    /// (`executor_contract::depth_range`): a `Depth32Float` texture's texels or a depth
    /// attachment's initial depth holds a NaN, a negative value other than `-0.0`, or a value
    /// above 1.0, which Metal's texel keeps. Vulkan takes no such value into a depth image without
    /// `VK_EXT_depth_range_unrestricted` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), so on a
    /// device without the extension the engine's candidate cannot put the case's depths where
    /// Metal holds them and stops there; on a device with it, it runs. A case's data on the
    /// executing device raises it, never a source's structure, so no census row carries it.
    DepthOutsideUnitRange,
}

/// Every executor (the requirement leaves no case to author, or none to observe).
const AUTHORING: &[Executor] = &Executor::ALL;
const ENGINE: &[Executor] = &[Executor::EngineCandidate];

impl ToolingRequirement {
    pub const ALL: [Self; 23] = [
        Self::FunctionConstantLiteral,
        Self::VertexAttributeLiteral,
        Self::TessellationAttributeLiteral,
        Self::TessellationSystemInputLiteral,
        Self::FragmentImageblockMemberLiteral,
        Self::KernelStageInputLiteral,
        Self::TextureBufferLiteral,
        Self::StorageTextureFormatLiteral,
        Self::ImplicitImageblockLiteral,
        Self::IndirectCommandBuffer,
        Self::VertexSideEffectObservation,
        Self::FragmentVaryingObservationType,
        Self::FragmentOutputObservationType,
        Self::ArgumentBufferIntersectionFunctionTable,
        Self::LinkedIntersectionFunction,
        Self::RenderDeviceBufferArray,
        Self::RenderAccelerationStructure,
        Self::VertexDepthStencilAttachment,
        Self::RasterizationDisabledRenderTarget,
        Self::RasterizationRateMapPlacement,
        Self::KernelMinLodClamp,
        Self::MeshCompanion,
        Self::DepthOutsideUnitRange,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FunctionConstantLiteral => "function_constant_literal",
            Self::VertexAttributeLiteral => "vertex_attribute_literal",
            Self::TessellationAttributeLiteral => "tessellation_attribute_literal",
            Self::TessellationSystemInputLiteral => "tessellation_system_input_literal",
            Self::FragmentImageblockMemberLiteral => "fragment_imageblock_member_literal",
            Self::KernelStageInputLiteral => "kernel_stage_input_literal",
            Self::TextureBufferLiteral => "texture_buffer_literal",
            Self::StorageTextureFormatLiteral => "storage_texture_format_literal",
            Self::ImplicitImageblockLiteral => "implicit_imageblock_literal",
            Self::IndirectCommandBuffer => "indirect_command_buffer",
            Self::VertexSideEffectObservation => "vertex_side_effect_observation",
            Self::FragmentVaryingObservationType => "fragment_varying_observation_type",
            Self::FragmentOutputObservationType => "fragment_output_observation_type",
            Self::ArgumentBufferIntersectionFunctionTable => {
                "argument_buffer_intersection_function_table"
            }
            Self::LinkedIntersectionFunction => "linked_intersection_function",
            Self::RenderDeviceBufferArray => "render_device_buffer_array",
            Self::RenderAccelerationStructure => "render_acceleration_structure",
            Self::VertexDepthStencilAttachment => "vertex_depth_stencil_attachment",
            Self::RasterizationDisabledRenderTarget => "rasterization_disabled_render_target",
            Self::RasterizationRateMapPlacement => "rasterization_rate_map_placement",
            Self::KernelMinLodClamp => "kernel_min_lod_clamp",
            Self::MeshCompanion => "mesh_companion",
            Self::DepthOutsideUnitRange => "depth_outside_unit_range",
        }
    }

    /// The executors this requirement stops.
    ///
    /// - Every executor: the case schema cannot express the structure (the `*_literal`
    ///   requirements), no declared observation captures the stage's effect (the observation
    ///   requirements), the only effect is encoded indirect commands no case executes
    ///   ([`ToolingRequirement::IndirectCommandBuffer`]), or the Metal oracle cannot build the
    ///   pipeline, so no candidate has a reference to be compared with
    ///   ([`ToolingRequirement::KernelMinLodClamp`]), or a mesh render pipeline lacks a stage
    ///   nothing supplies ([`ToolingRequirement::MeshCompanion`]).
    /// - The engine's candidate: the families its case mapping and render executor do not
    ///   implement yet (`candidate::engine`), each of which the Metal oracle runs, and
    ///   rasterization-rate map data a case can author only as bytes, which hold one
    ///   implementation's private layout where the engine's module decodes this crate's, and a
    ///   depth outside `[0, 1]` a case gives a depth image, which its executor raises only on a
    ///   device without `VK_EXT_depth_range_unrestricted`
    ///   ([`ToolingRequirement::DepthOutsideUnitRange`]).
    ///
    /// The requirements that stopped only the old engine's candidate were retired with it.
    pub const fn stops(self) -> &'static [Executor] {
        match self {
            Self::FunctionConstantLiteral
            | Self::VertexAttributeLiteral
            | Self::TessellationAttributeLiteral
            | Self::TessellationSystemInputLiteral
            | Self::FragmentImageblockMemberLiteral
            | Self::KernelStageInputLiteral
            | Self::TextureBufferLiteral
            | Self::StorageTextureFormatLiteral
            | Self::ImplicitImageblockLiteral
            | Self::IndirectCommandBuffer
            | Self::VertexSideEffectObservation
            | Self::FragmentVaryingObservationType
            | Self::FragmentOutputObservationType
            | Self::KernelMinLodClamp
            | Self::MeshCompanion => AUTHORING,
            Self::ArgumentBufferIntersectionFunctionTable
            | Self::LinkedIntersectionFunction
            | Self::RenderDeviceBufferArray
            | Self::RenderAccelerationStructure
            | Self::VertexDepthStencilAttachment
            | Self::RasterizationDisabledRenderTarget
            | Self::RasterizationRateMapPlacement
            | Self::DepthOutsideUnitRange => ENGINE,
        }
    }

    /// Whose limit this requirement records. Every requirement is this crate's tooling's except
    /// [`ToolingRequirement::KernelMinLodClamp`], which Apple's compiler cannot build a pipeline
    /// for.
    pub const fn limit(self) -> Limit {
        match self {
            Self::KernelMinLodClamp => Limit::Metal,
            _ => Limit::Tooling,
        }
    }

    /// Whether this requirement stops `executor`.
    pub fn stops_executor(self, executor: Executor) -> bool {
        self.stops().contains(&executor)
    }

    /// Whether a row carrying this requirement keeps the authoring-capability census open: it is
    /// a tooling gap ([`Limit::Tooling`]), and every requirement stops an executor the census
    /// closes on. A [`Limit::Metal`] requirement closes its row without a reference: no tooling
    /// can give Metal an answer its compiler cannot build.
    pub fn blocks_closure(self) -> bool {
        self.limit() == Limit::Tooling
    }

    /// The requirements of `requirements` that stop `executor`.
    pub fn stopping(
        requirements: impl IntoIterator<Item = Self>,
        executor: Executor,
    ) -> std::collections::BTreeSet<Self> {
        requirements
            .into_iter()
            .filter(|requirement| requirement.stops_executor(executor))
            .collect()
    }
}

impl FromStr for ToolingRequirement {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|requirement| requirement.as_str() == value)
            .ok_or_else(|| format!("unknown tooling requirement {value:?}"))
    }
}

impl fmt::Display for ToolingRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An effect both executors run exactly but no declared observation can capture.
///
/// Descriptive, never blocking: it is persisted and reported beside [`ToolingRequirement`], never
/// in it. A row carrying one is authorable, and its cases evidence the shader's other outputs only.
/// No report may count the semantics of an unobserved effect as verified: symbol coverage leaves
/// its calls out, and the authoring census prints each effect with [`UnobservedEffect::reason`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnobservedEffect {
    /// The commands a shader encodes into an indirect command buffer. Both executors supply the
    /// buffer under one descriptor policy and fail a run whose encoding faulted, but Metal has no
    /// host readback of encoded commands (its indirect command objects expose setters only). The
    /// one observation is executing the commands, which a case does and a fact of the source alone
    /// cannot claim.
    IndirectCommandContents,
}

impl UnobservedEffect {
    pub const ALL: [Self; 1] = [Self::IndirectCommandContents];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndirectCommandContents => "indirect_command_contents",
        }
    }

    /// Why the effect is not observed and what evidence is owed, for reports.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::IndirectCommandContents => {
                "ICB contents unobserved: no Metal readback of encoded commands; only a case \
                 that executes them observes them"
            }
        }
    }
}

impl FromStr for UnobservedEffect {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|effect| effect.as_str() == value)
            .ok_or_else(|| format!("unknown unobserved effect {value:?}"))
    }
}

impl fmt::Display for UnobservedEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_json_use_the_same_stable_requirement_name() {
        for requirement in ToolingRequirement::ALL {
            assert_eq!(
                serde_json::to_string(&requirement).unwrap(),
                format!("\"{requirement}\"")
            );
            assert_eq!(requirement.to_string().parse(), Ok(requirement));
        }
        for effect in UnobservedEffect::ALL {
            assert_eq!(
                serde_json::to_string(&effect).unwrap(),
                format!("\"{effect}\"")
            );
            assert_eq!(effect.to_string().parse(), Ok(effect));
            assert!(ToolingRequirement::ALL
                .iter()
                .all(|requirement| requirement.as_str() != effect.as_str()));
        }
    }

    /// The mesh render pipeline's requirement: both executors draw one, an object function's
    /// threadgroup memory among it, so only a stage nothing generates stops them
    /// (`mesh_companion`), this crate's gap, never Metal's, which keeps the census open.
    #[test]
    fn mesh_requirements_stop_the_executors_that_do_not_draw_them_yet() {
        let requirement = ToolingRequirement::MeshCompanion;
        assert_eq!(requirement.stops(), Executor::ALL);
        assert_eq!(requirement.limit(), Limit::Tooling, "{requirement}");
        assert!(requirement.blocks_closure(), "{requirement}");
    }

    /// Every requirement stops some executor, and an executor's name is as stable as a
    /// requirement's. A requirement that stops the Metal oracle stops every executor: when Metal
    /// cannot run a case, no candidate has an observation to be compared with. Only a tooling gap
    /// keeps the census open.
    #[test]
    fn every_requirement_stops_an_executor_and_a_metal_stop_is_an_authoring_stop() {
        for requirement in ToolingRequirement::ALL {
            let stops = requirement.stops();
            assert!(!stops.is_empty(), "{requirement}");
            assert!(
                stops.windows(2).all(|pair| pair[0] < pair[1]),
                "{requirement}"
            );
            if requirement.stops_executor(Executor::MetalOracle) {
                assert_eq!(stops, Executor::ALL, "{requirement}");
            }
            // Metal's own limit stops every executor and closes its row; every tooling gap keeps
            // the census open.
            match requirement.limit() {
                Limit::Metal => {
                    assert_eq!(stops, Executor::ALL, "{requirement}");
                    assert!(!requirement.blocks_closure(), "{requirement}");
                }
                Limit::Tooling => assert!(requirement.blocks_closure(), "{requirement}"),
            }
        }
        assert_eq!(
            ToolingRequirement::ALL
                .into_iter()
                .filter(|requirement| requirement.limit() == Limit::Metal)
                .collect::<Vec<_>>(),
            [ToolingRequirement::KernelMinLodClamp]
        );
        for executor in Executor::ALL {
            assert_eq!(
                serde_json::to_string(&executor).unwrap(),
                format!("\"{executor}\"")
            );
        }
        // The old engine's candidate and the requirements that stopped only it are retired: neither
        // name parses.
        assert!(serde_json::from_str::<Executor>("\"old_engine_candidate\"").is_err());
        for retired in ["static_sampler_pixel_address", "tile_grid", "ulong_texture"] {
            assert!(retired.parse::<ToolingRequirement>().is_err(), "{retired}");
        }
    }
}
