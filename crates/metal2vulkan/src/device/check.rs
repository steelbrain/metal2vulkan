//! `requirements ⊆ profile` and profile queries used by lowering.
//!
//! The finalize verifier derives a module's [`Requirements`] from what it contains; this check
//! says whether the profile enabled all of it. A failure after `select` is a translator bug
//! (`Internal(RequirementsExceedProfile)`), because selection only chooses what the profile has.
//! The knowledge is ported from `src/vulkan.rs` (`VulkanTarget::validate`) and
//! `src/vulkan/float_controls.rs`: every feature must be enabled, every property present, and the
//! float-controls independence must admit the modes each entry point declares. SPIR-V
//! capability/extension enablement is not re-derived here: `spirv::requirements` expresses it as
//! the features, extensions, and properties it needs.

use crate::device::derive::{usage_class, UsageClass};
use crate::device::profile::{DeviceProfile, Independence};
use crate::device::requirements::{
    FormatUsage, Requirements, StageCreateFlag, VkFeature, VkLimit, VkProperty,
    WITHOUT_FORMAT_CAPABILITIES,
};
use crate::error::Capability;
use crate::vk;

/// How a limit compares with a requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    /// The device value must be at least the requirement (`max*` limits).
    AtLeast,
    /// The device value must be at most the requirement (alignments, `minSubgroupSize`).
    AtMost,
}

impl DeviceProfile {
    /// Whether the profile enabled everything `req` needs. Returns the first missing capability
    /// in a fixed order (API version, features, extensions, properties, independence, limits,
    /// formats, stage flags). A storage capability without a format is enabled by its feature or
    /// by the formats' own bits (`DeviceProfile::without_format_by_format_bits`), and missing as
    /// its feature.
    pub fn check(&self, req: &Requirements) -> Result<(), Capability> {
        let f = self.fields();
        if let Some(v) = req.api_version {
            if v > f.api_version {
                return Err(Capability::ApiVersion(v));
            }
        }
        if let Some(&x) = req.features.difference(&f.enabled.features).next() {
            return Err(Capability::Feature(x));
        }
        for (capability, feature) in WITHOUT_FORMAT_CAPABILITIES {
            if req.spirv_capabilities.contains(&capability)
                && !f.enabled.features.contains(&feature)
                && !self.without_format_by_format_bits()
            {
                return Err(Capability::Feature(feature));
            }
        }
        if let Some(&x) = req.extensions.difference(&f.enabled.extensions).next() {
            return Err(Capability::Extension(x));
        }
        if let Some(&x) = req.properties.difference(&f.enabled.properties).next() {
            return Err(Capability::Property(x));
        }
        for (required, available, [partial, all]) in [
            (
                req.denorm_independence,
                f.float_controls.denorm_independence,
                [
                    VkProperty::DenormIndependence32BitOnly,
                    VkProperty::DenormIndependenceAll,
                ],
            ),
            (
                req.rounding_independence,
                f.float_controls.rounding_independence,
                [
                    VkProperty::RoundingIndependence32BitOnly,
                    VkProperty::RoundingIndependenceAll,
                ],
            ),
        ] {
            if !available.admits(required) {
                let missing = if required == Independence::All {
                    all
                } else {
                    partial
                };
                return Err(Capability::Property(missing));
            }
        }
        for (&limit, &required) in &req.limits {
            let (available, direction) = self.limit(limit);
            let ok = match direction {
                Direction::AtLeast => available >= required,
                Direction::AtMost => available <= required,
            };
            if !ok {
                return Err(Capability::Limit {
                    limit,
                    required,
                    available,
                });
            }
        }
        for &(format, usage) in &req.formats {
            if !self.supports_format(format, usage) {
                return Err(Capability::Format { format, usage });
            }
        }
        for flag in &req.stage_flags {
            let feature = match flag {
                StageCreateFlag::RequireFullSubgroups => VkFeature::ComputeFullSubgroups,
                StageCreateFlag::AllowVaryingSubgroupSize => VkFeature::SubgroupSizeControl,
            };
            if !f.enabled.features.contains(&feature) {
                return Err(Capability::Feature(feature));
            }
        }
        Ok(())
    }

    /// Whether `format` supports `usage` on this profile.
    pub fn supports_format(&self, format: vk::Format, usage: FormatUsage) -> bool {
        let f = self.fields();
        let table = match usage_class(usage) {
            UsageClass::Vertex => &f.vertex_formats,
            UsageClass::Storage => &f.storage_formats,
            UsageClass::Image => &f.image_formats,
        };
        table
            .formats
            .get(&format)
            .is_some_and(|usages| usages.contains(&usage))
    }

    /// A limit's value, in the unit [`Requirements::limits`] uses, and how requirements compare
    /// with it. Signed offset minima are reported as magnitudes (a requirement of 8 means
    /// "offsets down to -8"); float limits are reported rounded down to an integer.
    fn limit(&self, limit: VkLimit) -> (u64, Direction) {
        use Direction::{AtLeast, AtMost};
        let l = &self.fields().limits;
        let sg = &self.fields().subgroup;
        let up = |v: u32| (u64::from(v), AtLeast);
        let float = |bits: u32| {
            let v = f32::from_bits(bits);
            (if v > 0.0 { v.floor() as u64 } else { 0 }, AtLeast)
        };
        let magnitude = |v: i32| (u64::from(v.unsigned_abs()), AtLeast);
        let float_magnitude = |bits: u32| {
            let v = f32::from_bits(bits).abs();
            (v.floor() as u64, AtLeast)
        };
        match limit {
            VkLimit::MaxComputeWorkGroupSize(axis) => up(l
                .max_compute_work_group_size
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxComputeWorkGroupCount(axis) => up(l
                .max_compute_work_group_count
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxComputeWorkGroupInvocations => up(l.max_compute_work_group_invocations),
            VkLimit::MaxComputeSharedMemorySize => up(l.max_compute_shared_memory_size),
            VkLimit::MaxComputeWorkgroupSubgroups => up(sg.max_compute_workgroup_subgroups),
            // `minSubgroupSize` and `maxSubgroupSize` are the device's, the size range of every
            // stage that can pin a size; a device that can pin none has only its API size.
            VkLimit::MinSubgroupSize => (u64::from(sg.pinnable_sizes().min), AtMost),
            VkLimit::MaxSubgroupSize => up(sg.pinnable_sizes().max),
            VkLimit::MaxPushConstantsSize => up(l.max_push_constants_size),
            VkLimit::MaxPushDescriptors => up(l.max_push_descriptors),
            VkLimit::MaxBoundDescriptorSets => up(l.max_bound_descriptor_sets),
            VkLimit::MaxPerStageDescriptorSamplers => up(l.max_per_stage_descriptor_samplers),
            VkLimit::MaxPerStageDescriptorUniformBuffers => {
                up(l.max_per_stage_descriptor_uniform_buffers)
            }
            VkLimit::MaxPerStageDescriptorStorageBuffers => {
                up(l.max_per_stage_descriptor_storage_buffers)
            }
            VkLimit::MaxPerStageDescriptorSampledImages => {
                up(l.max_per_stage_descriptor_sampled_images)
            }
            VkLimit::MaxPerStageDescriptorStorageImages => {
                up(l.max_per_stage_descriptor_storage_images)
            }
            VkLimit::MaxPerStageDescriptorInputAttachments => {
                up(l.max_per_stage_descriptor_input_attachments)
            }
            VkLimit::MaxPerStageResources => up(l.max_per_stage_resources),
            VkLimit::MaxDescriptorSetSamplers => up(l.max_descriptor_set_samplers),
            VkLimit::MaxDescriptorSetUniformBuffers => up(l.max_descriptor_set_uniform_buffers),
            VkLimit::MaxDescriptorSetStorageBuffers => up(l.max_descriptor_set_storage_buffers),
            VkLimit::MaxDescriptorSetSampledImages => up(l.max_descriptor_set_sampled_images),
            VkLimit::MaxDescriptorSetStorageImages => up(l.max_descriptor_set_storage_images),
            VkLimit::MaxDescriptorSetInputAttachments => up(l.max_descriptor_set_input_attachments),
            VkLimit::MaxDescriptorSetUpdateAfterBindSamplers => {
                up(l.max_descriptor_set_update_after_bind_samplers)
            }
            VkLimit::MaxDescriptorSetUpdateAfterBindSampledImages => {
                up(l.max_descriptor_set_update_after_bind_sampled_images)
            }
            VkLimit::MaxDescriptorSetUpdateAfterBindStorageImages => {
                up(l.max_descriptor_set_update_after_bind_storage_images)
            }
            VkLimit::MaxPerStageDescriptorUpdateAfterBindSamplers => {
                up(l.max_per_stage_descriptor_update_after_bind_samplers)
            }
            VkLimit::MaxPerStageDescriptorUpdateAfterBindSampledImages => {
                up(l.max_per_stage_descriptor_update_after_bind_sampled_images)
            }
            VkLimit::MaxPerStageDescriptorUpdateAfterBindStorageImages => {
                up(l.max_per_stage_descriptor_update_after_bind_storage_images)
            }
            VkLimit::MaxPerStageUpdateAfterBindResources => {
                up(l.max_per_stage_update_after_bind_resources)
            }
            VkLimit::MaxStorageBufferRange => up(l.max_storage_buffer_range),
            VkLimit::MaxUniformBufferRange => up(l.max_uniform_buffer_range),
            VkLimit::MinStorageBufferOffsetAlignment => {
                (l.min_storage_buffer_offset_alignment, AtMost)
            }
            VkLimit::MinUniformBufferOffsetAlignment => {
                (l.min_uniform_buffer_offset_alignment, AtMost)
            }
            VkLimit::MinTexelBufferOffsetAlignment => (l.min_texel_buffer_offset_alignment, AtMost),
            VkLimit::MaxTexelBufferElements => up(l.max_texel_buffer_elements),
            VkLimit::MaxSamplerLodBias => float(l.max_sampler_lod_bias_bits),
            VkLimit::MaxSamplerAnisotropy => float(l.max_sampler_anisotropy_bits),
            VkLimit::MinTexelOffset => magnitude(l.min_texel_offset),
            VkLimit::MaxTexelOffset => up(l.max_texel_offset),
            VkLimit::MinTexelGatherOffset => magnitude(l.min_texel_gather_offset),
            VkLimit::MaxTexelGatherOffset => up(l.max_texel_gather_offset),
            VkLimit::MinInterpolationOffset => float_magnitude(l.min_interpolation_offset_bits),
            VkLimit::MaxInterpolationOffset => float(l.max_interpolation_offset_bits),
            VkLimit::MaxImageDimension1D => up(l.max_image_dimension_1d),
            VkLimit::MaxImageDimension2D => up(l.max_image_dimension_2d),
            VkLimit::MaxImageDimension3D => up(l.max_image_dimension_3d),
            VkLimit::MaxImageDimensionCube => up(l.max_image_dimension_cube),
            VkLimit::MaxImageArrayLayers => up(l.max_image_array_layers),
            VkLimit::MaxVertexInputAttributes => up(l.max_vertex_input_attributes),
            VkLimit::MaxVertexInputBindings => up(l.max_vertex_input_bindings),
            VkLimit::MaxVertexInputAttributeOffset => up(l.max_vertex_input_attribute_offset),
            VkLimit::MaxVertexInputBindingStride => up(l.max_vertex_input_binding_stride),
            VkLimit::MaxVertexOutputComponents => up(l.max_vertex_output_components),
            VkLimit::MaxTessellationGenerationLevel => up(l.max_tessellation_generation_level),
            VkLimit::MaxTessellationEvaluationOutputComponents => {
                up(l.max_tessellation_evaluation_output_components)
            }
            VkLimit::MaxFragmentInputComponents => up(l.max_fragment_input_components),
            VkLimit::MaxFragmentOutputAttachments => up(l.max_fragment_output_attachments),
            VkLimit::MaxFragmentDualSrcAttachments => up(l.max_fragment_dual_src_attachments),
            VkLimit::MaxFragmentCombinedOutputResources => {
                up(l.max_fragment_combined_output_resources)
            }
            VkLimit::MaxColorAttachments => up(l.max_color_attachments),
            VkLimit::MaxClipDistances => up(l.max_clip_distances),
            VkLimit::MaxCullDistances => up(l.max_cull_distances),
            VkLimit::MaxCombinedClipAndCullDistances => up(l.max_combined_clip_and_cull_distances),
            VkLimit::MaxVertexAttribDivisor => up(l.max_vertex_attrib_divisor),
            VkLimit::MaxMultiviewViewCount => up(l.max_multiview_view_count),
            VkLimit::MaxMultiviewInstanceIndex => up(l.max_multiview_instance_index),
            VkLimit::MaxPerStageDescriptorAccelerationStructures => {
                up(l.max_per_stage_descriptor_acceleration_structures)
            }
            VkLimit::MaxDescriptorSetAccelerationStructures => {
                up(l.max_descriptor_set_acceleration_structures)
            }
            VkLimit::MaxFramebufferWidth => up(l.max_framebuffer_width),
            VkLimit::MaxFramebufferHeight => up(l.max_framebuffer_height),
            VkLimit::MaxFramebufferLayers => up(l.max_framebuffer_layers),
            VkLimit::MaxDrawIndexedIndexValue => up(l.max_draw_indexed_index_value),
            VkLimit::MaxTessellationPatchSize => up(l.max_tessellation_patch_size),
            VkLimit::MaxGeometryInputComponents => up(l.max_geometry_input_components),
            VkLimit::MaxGeometryOutputComponents => up(l.max_geometry_output_components),
            VkLimit::MaxGeometryTotalOutputComponents => up(l.max_geometry_total_output_components),
            VkLimit::FunctionStorageBytesPerThread => {
                (self.fields().function_storage.bytes(), AtLeast)
            }
            VkLimit::MaxMeshOutputVertices => up(l.mesh.max_mesh_output_vertices),
            VkLimit::MaxMeshOutputPrimitives => up(l.mesh.max_mesh_output_primitives),
            VkLimit::MaxMeshOutputComponents => up(l.mesh.max_mesh_output_components),
            VkLimit::MaxTaskSharedMemorySize => up(l.mesh.max_task_shared_memory_size),
            VkLimit::MaxMeshSharedMemorySize => up(l.mesh.max_mesh_shared_memory_size),
            VkLimit::MaxMeshWorkGroupCount(axis) => up(l
                .mesh
                .max_mesh_work_group_count
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxMeshWorkGroupTotalCount => up(l.mesh.max_mesh_work_group_total_count),
            VkLimit::MaxMeshWorkGroupSize(axis) => up(l
                .mesh
                .max_mesh_work_group_size
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxMeshWorkGroupInvocations => up(l.mesh.max_mesh_work_group_invocations),
            VkLimit::MaxMeshOutputLayers => up(l.mesh.max_mesh_output_layers),
            VkLimit::MaxTaskWorkGroupCount(axis) => up(l
                .mesh
                .max_task_work_group_count
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxTaskWorkGroupTotalCount => up(l.mesh.max_task_work_group_total_count),
            VkLimit::MaxTaskWorkGroupSize(axis) => up(l
                .mesh
                .max_task_work_group_size
                .get(usize::from(axis))
                .copied()
                .unwrap_or(0)),
            VkLimit::MaxTaskWorkGroupInvocations => up(l.mesh.max_task_work_group_invocations),
            VkLimit::MaxTaskPayloadSize => up(l.mesh.max_task_payload_size),
            VkLimit::MaxTaskPayloadAndSharedMemorySize => {
                up(l.mesh.max_task_payload_and_shared_memory_size)
            }
            VkLimit::MaxMeshPayloadAndSharedMemorySize => {
                up(l.mesh.max_mesh_payload_and_shared_memory_size)
            }
        }
    }

    /// What a pipeline of modules needing `a` and `b` needs: the sets' unions, the stricter
    /// independence, and per limit the stricter value in the limit's direction (the larger for
    /// `max*` limits, the smaller for alignments and minima).
    pub fn union_requirements(&self, a: &Requirements, b: &Requirements) -> Requirements {
        let stricter = |x: Independence, y: Independence| if x.admits(y) { x } else { y };
        let mut out = a.clone();
        out.api_version = a.api_version.max(b.api_version);
        out.spirv_capabilities
            .extend(b.spirv_capabilities.iter().copied());
        out.spirv_extensions
            .extend(b.spirv_extensions.iter().copied());
        out.features.extend(b.features.iter().copied());
        out.extensions.extend(b.extensions.iter().copied());
        out.properties.extend(b.properties.iter().copied());
        out.formats.extend(b.formats.iter().copied());
        out.stage_flags.extend(b.stage_flags.iter().copied());
        out.denorm_independence = stricter(a.denorm_independence, b.denorm_independence);
        out.rounding_independence = stricter(a.rounding_independence, b.rounding_independence);
        for (&limit, &v) in &b.limits {
            let merged = match (out.limits.get(&limit), self.limit(limit).1) {
                (None, _) => v,
                (Some(&w), Direction::AtLeast) => w.max(v),
                (Some(&w), Direction::AtMost) => w.min(v),
            };
            out.limits.insert(limit, merged);
        }
        out
    }
}
