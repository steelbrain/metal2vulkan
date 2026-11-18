//! Render pipeline states: the translator's render pipeline of a descriptor's content, its shader
//! modules, and its `VkPipeline`s, one for each topology class drawn.
//!
//! What a state compiles is its content, never the state: the translator's key of its descriptor
//! (`Device::queried_render_descriptor_key`: every field of the descriptor, each function by its AIR,
//! entry, constant values, linked functions and dynamic libraries, with the device's profile and
//! the translator's semantics version), which reads no function, as a compute state's key reads
//! none. Every state of one content shares one [`RenderCompiled`]: the translation, its shader
//! modules and their `VkPipeline`s. The device's one map of contents (`kernel_map`), which compute
//! states use too, holds each content's, so a state of a content the map holds costs the key and a
//! lookup. A content the map does not hold is decoded, keyed for its compile
//! (`Device::render_variant_key`) and lowered. The cost of keying on the descriptor: two
//! descriptors that differ only in a field their compile does not read are two contents, each
//! compiled.
//!
//! What a `VkPipeline` bakes is what Vulkan 1.3 cannot leave dynamic: the content, and the
//! topology class (point, line or triangle) of the draws it serves. Everything Metal sets on the
//! encoder that Vulkan 1.3 makes dynamic without a feature is dynamic state: viewport, scissor,
//! cull mode, front face, the primitive topology within its class and primitive restart, the
//! depth test, write and compare, the stencil test, operations, masks and reference, depth bias,
//! blend constants and vertex buffer strides. Metal's triangle fill mode and depth clip mode,
//! which Vulkan 1.3 bakes, are their defaults: the render encoder does not set them yet.
//!
//! A descriptor names no topology class unless it sets `inputPrimitiveTopology`, and the replay
//! streams' never do, while nearly every content draws only triangles (527 of the 537 contents
//! gb and b2 draw). So the triangle class's `VkPipeline` is made with the state, where Metal
//! compiles; a point or line class's comes with the draws, made by the first draw of that class.
//! The cost: a content drawn only as lines or points makes a triangle `VkPipeline` it never uses
//! (8 contents of the 537).

use crate::cache::{module_digest, pipeline_key};
use crate::counters::Gauge;
use crate::device::Core;
use crate::error::{Error, Gap, RenderPipelineKind};
use crate::kernel_map::KernelKey;
use crate::pipeline::{layout_words, lower, made_through_cache};
use crate::reaper::{Free, Job};
use ash::vk;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::EnabledCaps;
use metal2vulkan::pipeline::{
    PrecisionNote, RenderPipeline, RenderPipelineDescriptor, RenderPipelineKeys, SpecKey,
    StageEntries, StageSource, VertexStride,
};
use metal2vulkan::Budget;
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::sync::Arc;

/// `MTLRenderPipelineState`.
///
/// A cheap handle: clones share one state. States of one content (the functions, their constant
/// values and linked functions, and the descriptor's fields that change the compile) share its
/// translation and `VkPipeline`s, so a guest that makes one state many times compiles it once.
#[derive(Clone)]
pub struct RenderPipelineState {
    pub(crate) compiled: Arc<RenderCompiled>,
}

/// The state Vulkan 1.3 lets a draw set, which every render `VkPipeline` takes as dynamic.
const DYNAMIC: [vk::DynamicState; 18] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::CULL_MODE,
    vk::DynamicState::FRONT_FACE,
    vk::DynamicState::PRIMITIVE_TOPOLOGY,
    vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
    vk::DynamicState::DEPTH_TEST_ENABLE,
    vk::DynamicState::DEPTH_WRITE_ENABLE,
    vk::DynamicState::DEPTH_COMPARE_OP,
    vk::DynamicState::STENCIL_TEST_ENABLE,
    vk::DynamicState::STENCIL_OP,
    vk::DynamicState::STENCIL_COMPARE_MASK,
    vk::DynamicState::STENCIL_WRITE_MASK,
    vk::DynamicState::STENCIL_REFERENCE,
    vk::DynamicState::DEPTH_BIAS,
    vk::DynamicState::DEPTH_BIAS_ENABLE,
    vk::DynamicState::BLEND_CONSTANTS,
    vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE,
];

/// What one render pipeline content compiles to, shared by every state of it and held by the
/// device's map of contents until the map forgets it.
pub(crate) struct RenderCompiled {
    core: Arc<Core>,
    /// Each stage's module, entry point, stage and specialization.
    stages: Vec<Module>,
    /// The fixed-function state every `VkPipeline` of the content takes.
    fixed: Fixed,
    /// With a cache directory, the digest of everything the content gives a `VkPipeline`'s
    /// creation (its modules' digests and its fixed-function state), which with the topology
    /// class and the specialization's values keys the pipeline's file; else zeros.
    digest: [u8; 32],
    /// The triangle class's `VkPipeline`, made with the content.
    triangles: vk::Pipeline,
    /// The translator's notes of where the pipeline's answers may differ from Metal's.
    notes: Vec<PrecisionNote>,
}

/// One stage's shader module and what a `VkPipeline` of it takes.
struct Module {
    module: vk::ShaderModule,
    entry: CString,
    stage: vk::ShaderStageFlags,
    flags: vk::PipelineShaderStageCreateFlags,
    /// Its specialization constants' entries and their 4-byte values, fixed with the content.
    constants: Vec<vk::SpecializationMapEntry>,
    values: Vec<u8>,
}

/// The fixed-function state of a content's `VkPipeline`s, as Vulkan takes it: the descriptor's
/// vertex layout, attachments, blending and sample count, as the translator states them.
struct Fixed {
    bindings: Vec<vk::VertexInputBindingDescription>,
    attributes: Vec<vk::VertexInputAttributeDescription>,
    discard: bool,
    samples: vk::SampleCountFlags,
    alpha_to_coverage: bool,
    blends: Vec<vk::PipelineColorBlendAttachmentState>,
    /// Each colour attachment's format by Metal index, `UNDEFINED` for an absent one.
    colors: Vec<vk::Format>,
    stencil: vk::Format,
}

impl crate::Device {
    /// `newRenderPipelineStateWithDescriptor:error:`, without reflection.
    ///
    /// A state's content is its descriptor: its functions with their constant values and linked
    /// functions, and every field. Every state of one content shares its translation, its shader
    /// modules and their `VkPipeline`s: a state of a content the device holds (the last 1,024
    /// contents used, compute and render together) costs the translator's key of the descriptor,
    /// which reads no function, and a lookup. For a
    /// content the device does not hold, the translator lowers each function on the calling
    /// thread, which waits for it, within its 20-second budget per lowering, and the content's
    /// triangle-class `VkPipeline` is made here. Where the device has a cache directory
    /// ([`DeviceOptions::pipeline_cache_dir`]), a lowering a device made before is read from it in
    /// place of lowering again, and each `VkPipeline` is made through the pipeline data kept for
    /// it there. A point or line class's `VkPipeline` comes with the draws. The
    /// `completionHandler:` forms are not built (no replay stream calls them).
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    ///
    /// Built: one vertex and at most one fragment function, colour attachments of any format the
    /// translator maps for the device, a stencil attachment, one sample, and every blend state,
    /// write mask and vertex descriptor Vulkan's fixed-function state takes with what device
    /// creation enables; what else the translator states is refused at the creation, by kind.
    ///
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses;
    /// [`Gap::IndirectCommandBuffers`]; [`Gap::RenderPipeline`] for what the runtime does not
    /// build, naming it; [`Error::Vulkan`].
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        refuse(descriptor)?;
        let core = self.core();
        let key = KernelKey {
            pipeline: core.translator.queried_render_descriptor_key(descriptor),
            required: None,
        };
        let compiled = match core.kernels.find_render(&key) {
            Some(compiled) => compiled,
            None => {
                let budget = Budget::standard();
                let entries = StageEntries::new();
                let keys = (core.translator)
                    .render_variant_key(descriptor, &VariantKey::new(), &budget, &entries)
                    .map_err(Error::Translation)?;
                let compiled = RenderCompiled::of(core, descriptor, &keys, &budget, &entries)?;
                (core.kernels).keep_render(key, Arc::new(compiled), &core.counters)
            }
        };
        Ok(RenderPipelineState { compiled })
    }
}

/// What the descriptor asks that the runtime does not build.
fn refuse(d: &RenderPipelineDescriptor) -> Result<(), Error> {
    let kind = if d.support_indirect_command_buffers {
        return Err(Error::Unsupported(Gap::IndirectCommandBuffers));
    } else if d.depth_format.is_some() {
        RenderPipelineKind::DepthAttachment
    } else if d.raster_sample_count != 1 {
        RenderPipelineKind::Multisample
    } else if d.max_vertex_amplification_count != 1 {
        RenderPipelineKind::VertexAmplification
    } else if d.tessellation.is_some() {
        RenderPipelineKind::Tessellation
    } else {
        return Ok(());
    };
    Err(Error::Unsupported(Gap::RenderPipeline { kind }))
}

/// What the translator's pipeline states that the runtime does not build, where it states one;
/// `enabled` is what device creation enabled.
fn unbuilt(pipeline: &RenderPipeline, enabled: &EnabledCaps) -> Option<RenderPipelineKind> {
    let g = &pipeline.graphics;
    let interface = &pipeline.interface;
    let requirements = &pipeline.requirements;
    let synthesized = g.tessellation.is_some()
        || g.emulated_tessellation.is_some()
        || g.line_band.is_some()
        || g.region_fan_out.is_some()
        || g.mesh_records.is_some()
        || pipeline.stages.iter().any(|stage| {
            ![
                metal2vulkan::vk::ShaderStage::VERTEX,
                metal2vulkan::vk::ShaderStage::FRAGMENT,
            ]
            .contains(&stage.stage)
        });
    let numbered = (interface.numbered_samplers.iter()).any(|(_, samplers)| !samplers.is_empty());
    let unenabled = !(requirements.features).is_subset(&enabled.features)
        || !(requirements.extensions).is_subset(&enabled.extensions);
    Some(if synthesized {
        RenderPipelineKind::SynthesizedStages
    } else if !g.input_attachments.is_empty()
        || !g.storage_attachments.is_empty()
        || g.rasterization_order_attachment_access
    {
        RenderPipelineKind::FramebufferFetch
    } else if g.fragment_imageblock.is_some() {
        RenderPipelineKind::Imageblock
    } else if g.shadow_stencil.is_some() {
        RenderPipelineKind::ShaderStencil
    } else if g.alpha_to_one {
        RenderPipelineKind::AlphaToOne
    } else if (pipeline.vertex_input.bindings.iter()).any(|binding| binding.divisor > 1) {
        RenderPipelineKind::StepRate
    } else if numbered {
        RenderPipelineKind::Samplers
    } else if (pipeline.stages.iter()).any(|stage| stage.constant_blob.is_some()) {
        RenderPipelineKind::ConstantData
    } else if !(g.dynamic_states.iter()).all(|state| DYNAMIC.contains(&dynamic(*state))) {
        RenderPipelineKind::DynamicState
    } else if unenabled {
        RenderPipelineKind::DeviceFeature
    } else {
        return None;
    })
}

/// The ash form of a translator dynamic state.
fn dynamic(state: metal2vulkan::vk::DynamicState) -> vk::DynamicState {
    vk::DynamicState::from_raw(state.as_raw())
}

impl Fixed {
    /// The fixed-function state the translator states for `pipeline`.
    fn of(pipeline: &RenderPipeline) -> Fixed {
        let g = &pipeline.graphics;
        let bindings = (pipeline.vertex_input.bindings.iter())
            .map(|binding| vk::VertexInputBindingDescription {
                binding: binding.binding,
                // A stride set per bind is dynamic state (`VERTEX_INPUT_BINDING_STRIDE`).
                stride: match binding.stride {
                    VertexStride::Static(stride) => stride,
                    VertexStride::PerBind => 0,
                },
                input_rate: vk::VertexInputRate::from_raw(binding.input_rate.as_raw()),
            })
            .collect();
        let attributes = (pipeline.vertex_input.attributes.iter())
            .map(|attribute| vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: vk::Format::from_raw(attribute.format.as_raw()),
                offset: attribute.offset,
            })
            .collect();
        let factor =
            |factor: metal2vulkan::vk::BlendFactor| vk::BlendFactor::from_raw(factor.as_raw());
        let op = |op: metal2vulkan::vk::BlendOp| vk::BlendOp::from_raw(op.as_raw());
        let blends = (g.color_blend_attachments().into_iter())
            .map(|(mask, blend)| {
                let state = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::from_raw(mask.as_raw()));
                match blend {
                    Some(blend) => state
                        .blend_enable(true)
                        .src_color_blend_factor(factor(blend.src_color))
                        .dst_color_blend_factor(factor(blend.dst_color))
                        .color_blend_op(op(blend.color_op))
                        .src_alpha_blend_factor(factor(blend.src_alpha))
                        .dst_alpha_blend_factor(factor(blend.dst_alpha))
                        .alpha_blend_op(op(blend.alpha_op)),
                    None => state,
                }
            })
            .collect();
        let colors = (g.color_attachments.iter())
            .map(|attachment| {
                attachment
                    .as_ref()
                    .map_or(vk::Format::UNDEFINED, |attachment| {
                        vk::Format::from_raw(attachment.format.as_raw())
                    })
            })
            .collect();
        Fixed {
            bindings,
            attributes,
            discard: g.rasterizer_discard,
            samples: vk::SampleCountFlags::from_raw(g.rasterization_samples.as_raw()),
            alpha_to_coverage: g.alpha_to_coverage,
            blends,
            colors,
            stencil: g.stencil_format.map_or(vk::Format::UNDEFINED, |format| {
                vk::Format::from_raw(format.as_raw())
            }),
        }
    }

    /// Everything it gives a `VkPipeline`'s creation, with the dynamic states, as words.
    fn words(&self) -> Vec<u32> {
        let raw = |value: i32| value as u32;
        let mut words = vec![self.bindings.len() as u32];
        for b in &self.bindings {
            words.extend([b.binding, b.stride, raw(b.input_rate.as_raw())]);
        }
        words.push(self.attributes.len() as u32);
        for a in &self.attributes {
            words.extend([a.location, a.binding, raw(a.format.as_raw()), a.offset]);
        }
        words.extend([
            u32::from(self.discard),
            self.samples.as_raw(),
            u32::from(self.alpha_to_coverage),
        ]);
        words.push(self.blends.len() as u32);
        for b in &self.blends {
            words.extend([
                b.blend_enable,
                raw(b.src_color_blend_factor.as_raw()),
                raw(b.dst_color_blend_factor.as_raw()),
                raw(b.color_blend_op.as_raw()),
                raw(b.src_alpha_blend_factor.as_raw()),
                raw(b.dst_alpha_blend_factor.as_raw()),
                raw(b.alpha_blend_op.as_raw()),
                b.color_write_mask.as_raw(),
            ]);
        }
        words.push(self.colors.len() as u32);
        words.extend(self.colors.iter().map(|format| raw(format.as_raw())));
        words.push(raw(self.stencil.as_raw()));
        words.extend(DYNAMIC.iter().map(|state| raw(state.as_raw())));
        words
    }
}

impl RenderCompiled {
    /// The translation of `descriptor` under `keys`, its stages lowered through the cache
    /// directory, with its triangle-class `VkPipeline`.
    fn of(
        core: &Arc<Core>,
        descriptor: &RenderPipelineDescriptor,
        keys: &RenderPipelineKeys,
        budget: &Budget,
        entries: &StageEntries,
    ) -> Result<RenderCompiled, Error> {
        let translator = &core.translator;
        let mut lowered = descriptor.clone();
        if let (StageSource::Function(function), Some(inputs)) =
            (&descriptor.vertex, &keys.vertex_inputs)
        {
            let function = lower(core, function, inputs, budget, entries);
            lowered.vertex = StageSource::Compiled(function.map_err(Error::Translation)?);
        }
        // The fragment function's push fields follow the vertex function's in the floor's one
        // range, from the vertex function's end rounded up to 8.
        let base = match &lowered.vertex {
            StageSource::Compiled(vertex) => vertex.fragment_push_base(),
            StageSource::Function(_) => 0,
        };
        if let (Some(StageSource::Function(function)), Some(inputs)) =
            (&descriptor.fragment, &keys.fragment_inputs)
        {
            let mut inputs = inputs.clone();
            inputs.push_base = base;
            let function = lower(core, function, &inputs, budget, entries);
            lowered.fragment = Some(StageSource::Compiled(function.map_err(Error::Translation)?));
        }
        let pipeline =
            (translator.compile_render_pipeline(&lowered, budget)).map_err(Error::Translation)?;
        if let Some(kind) = unbuilt(&pipeline, translator.profile().enabled()) {
            return Err(Error::Unsupported(Gap::RenderPipeline { kind }));
        }
        let samples = descriptor.raster_sample_count;
        let mut stages = Vec::new();
        let mut notes = Vec::new();
        let mut digest = core.caches.on_disk().then(|| {
            let mut hash = Sha256::new();
            hash.update((pipeline.stages.len() as u64).to_le_bytes());
            hash
        });
        for stage in &pipeline.stages {
            let mut constants = Vec::new();
            let mut values = Vec::new();
            for constant in &stage.specialization.constants {
                let value = match (constant.key, constant.size) {
                    (SpecKey::SampleCount, 4) => samples,
                    (SpecKey::AmplificationCount, 4) => 1,
                    _ => {
                        release(core, &stages);
                        return Err(Error::Unsupported(Gap::RenderPipeline {
                            kind: RenderPipelineKind::Specialization,
                        }));
                    }
                };
                constants.push(vk::SpecializationMapEntry {
                    constant_id: constant.id,
                    offset: values.len() as u32,
                    size: 4,
                });
                values.extend(value.to_ne_bytes());
            }
            // The translator's entry points hold no NUL; one that did would name no entry point.
            let Ok(entry) = CString::new(stage.entry_point.as_str()) else {
                release(core, &stages);
                return Err(Error::Unsupported(Gap::RenderPipeline {
                    kind: RenderPipelineKind::SynthesizedStages,
                }));
            };
            let flags = vk::PipelineShaderStageCreateFlags::from_raw(stage.stage_flags.as_raw());
            let kind = vk::ShaderStageFlags::from_raw(stage.stage.as_raw());
            if let Some(hash) = &mut digest {
                let ids: Vec<u32> = constants.iter().map(|entry| entry.constant_id).collect();
                hash.update(kind.as_raw().to_le_bytes());
                hash.update(module_digest(
                    &stage.spirv,
                    entry.as_bytes(),
                    flags,
                    None,
                    &ids,
                    &layout_words(),
                ));
            }
            let info = vk::ShaderModuleCreateInfo::default().code(&stage.spirv);
            // SAFETY: the words are the translator's SPIR-V, for the features device creation
            // enabled.
            let module = match unsafe { core.device.create_shader_module(&info, None) } {
                Ok(module) => module,
                Err(result) => {
                    release(core, &stages);
                    return Err(Error::Vulkan {
                        call: "vkCreateShaderModule",
                        result,
                    });
                }
            };
            stages.push(Module {
                module,
                entry,
                stage: kind,
                flags,
                constants,
                values,
            });
            notes.extend(stage.precision_notes.iter().copied());
        }
        notes.sort_unstable();
        notes.dedup();
        let fixed = Fixed::of(&pipeline);
        let digest: [u8; 32] = digest.map_or([0; 32], |mut hash| {
            for word in fixed.words() {
                hash.update(word.to_le_bytes());
            }
            hash.finalize().into()
        });
        core.counters.rise(Gauge::RenderContents);
        let mut compiled = RenderCompiled {
            core: Arc::clone(core),
            stages,
            fixed,
            digest,
            triangles: vk::Pipeline::null(),
            notes,
        };
        // Made whole before the `VkPipeline`, so dropping it releases what was made.
        compiled.triangles = compiled.make(vk::PrimitiveTopology::TRIANGLE_LIST)?;
        Ok(compiled)
    }

    /// A `VkPipeline` of the content for draws of the class of `topology`, a list topology, made
    /// through the device's pipeline data, rendering dynamically into the descriptor's attachment
    /// formats. A draw sets its own topology within the class as dynamic state.
    fn make(&self, topology: vk::PrimitiveTopology) -> Result<vk::Pipeline, Error> {
        let core = &self.core;
        let fixed = &self.fixed;
        let specializations: Vec<vk::SpecializationInfo<'_>> = (self.stages.iter())
            .map(|stage| {
                vk::SpecializationInfo::default()
                    .map_entries(&stage.constants)
                    .data(&stage.values)
            })
            .collect();
        let stages: Vec<vk::PipelineShaderStageCreateInfo<'_>> = (self.stages.iter())
            .zip(&specializations)
            .map(|(stage, specialization)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .flags(stage.flags)
                    .stage(stage.stage)
                    .module(stage.module)
                    .name(&stage.entry)
                    .specialization_info(specialization)
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&fixed.bindings)
            .vertex_attribute_descriptions(&fixed.attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .rasterizer_discard_enable(fixed.discard)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(fixed.samples)
            .alpha_to_coverage_enable(fixed.alpha_to_coverage);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&fixed.blends);
        let dynamic_states = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&DYNAMIC);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&fixed.colors)
            .stencil_attachment_format(fixed.stencil);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_states)
            .layout(core.layout);
        if !fixed.discard {
            info = info.color_blend_state(&blend);
        }
        // The topology's class and the stages' specialization values, beside the content's
        // digest, key the pipeline's file.
        let key = || {
            let mut values = (topology.as_raw() as u32).to_le_bytes().to_vec();
            for stage in &self.stages {
                values.extend(&stage.values);
            }
            pipeline_key(&self.digest, &values)
        };
        made_through_cache(core, "vkCreateGraphicsPipelines", key, |cache| {
            // SAFETY: the modules, the layout and the cache are the device's; each specialization
            // gives each of its stage's constants a 4-byte value; the formats are the
            // attachments' the translator maps for the device, for the features device creation
            // enabled.
            unsafe { core.device.create_graphics_pipelines(cache, &[info], None) }
        })
    }
}

/// Gives `stages`' modules to the reaper: a creation that failed after making them.
fn release(core: &Core, stages: &[Module]) {
    for stage in stages {
        core.reaper
            .give(Job::Free(Free::ShaderModule(stage.module)));
    }
}

impl RenderPipelineState {
    /// No Metal counterpart: the translator's notes of where this pipeline's answers may differ
    /// from Metal's, of its modules ([`PrecisionNote`]); empty where they match.
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        self.compiled.notes.clone()
    }
}

impl Drop for RenderCompiled {
    fn drop(&mut self) {
        // No state holds it, so no command buffer that used it is left: each kept its state until
        // it retired.
        self.core.counters.fall(Gauge::RenderContents);
        if self.triangles != vk::Pipeline::null() {
            (self.core.reaper).give(Job::Free(Free::Pipeline(self.triangles)));
        }
        release(&self.core, &self.stages);
    }
}
