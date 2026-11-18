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
//! depth test, write and compare, the stencil test, operations, masks and reference, depth bias
//! and blend constants. Vertex buffer strides are the descriptor's, so the content's: they are
//! baked, and a layout whose stride is set per bind (`MTLBufferLayoutStrideDynamic`), which the
//! replay streams' never are, is refused. Metal's triangle fill mode and depth clip mode, which
//! Vulkan 1.3 bakes, are their defaults: the render encoder does not set them yet.
//!
//! What a draw supplies is the content's [`Draws`]: each function's slots with their accesses,
//! its push fields (addresses, resource IDs and indices of the slots, in the floor's one push
//! range, the fragment function's after the vertex function's), and the vertex buffers its
//! fixed-function vertex input reads. A state numbers its functions' samplers as a compute state
//! numbers its kernel's, the vertex function's first.
//!
//! A descriptor names no topology class unless it sets `inputPrimitiveTopology`, and the replay
//! streams' never do, while nearly every content draws only triangles (527 of the 537 contents
//! gb and b2 draw). So the triangle class's `VkPipeline` is made with the state, where Metal
//! compiles; a point or line class's comes with the draws, made by the first draw of that class.
//! The cost: a content drawn only as lines or points makes a triangle `VkPipeline` it never uses
//! (8 contents of the 537). The replay streams draw no points: a draw of points is refused.

use crate::counters::Gauge;
use crate::device::{lock, Core};
use crate::error::{Error, Gap, RenderPipelineKind};
use crate::hazard::Access;
use crate::kernel_map::KernelKey;
use crate::pass::Pass;
use crate::pipeline::{
    access, lower, made_through_cache, number_samplers, of_the_device, texture_access, FieldValue,
    Numbered, PUSH_BYTES,
};
use crate::reaper::{Free, Job};
use crate::times::host_time;
use ash::vk;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::EnabledCaps;
use metal2vulkan::pipeline::{
    BufferKind, CombinedSampler, NumberedSampler, PrecisionNote, PushField, PushLoc,
    RenderPipeline, RenderPipelineDescriptor, RenderPipelineKeys, RuntimeFact, SamplerRef,
    SlotClass, SlotInterface, SlotRef, SpecKey, Stage, StageEntries, StageSource, VertexStride,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

/// `MTLRenderPipelineState`.
///
/// A cheap handle: clones share one state. A command buffer that draws with it keeps it until it
/// retires. States of one content (the functions, their constant values and linked functions, and
/// the descriptor's fields that change the compile) share its translation and `VkPipeline`s, so a
/// guest that makes one state many times compiles it once; each numbers its own samplers.
#[derive(Clone)]
pub struct RenderPipelineState {
    pub(crate) inner: Arc<RenderInner>,
}

pub(crate) struct RenderInner {
    /// What the state's content compiles to, shared with every state of that content.
    pub(crate) compiled: Arc<RenderCompiled>,
    /// The samplers the state numbers and the indices the consumer gave them.
    pub(crate) numbered: Numbered,
}

/// The vertex function's index in a draw's tables, and the fragment function's.
pub(crate) const VERTEX: usize = 0;
pub(crate) const FRAGMENT: usize = 1;

/// What a draw of a content supplies, from the translator.
pub(crate) struct Draws {
    /// Each buffer slot a function uses: its function ([`VERTEX`] or [`FRAGMENT`]), its index and
    /// its access.
    pub(crate) buffers: Vec<(usize, u32, Access)>,
    /// Each texture slot a function uses, likewise.
    pub(crate) textures: Vec<(usize, u32, Access)>,
    /// The fragment function's texture slots it reads only at the fragment's own pixel
    /// (`StageModule::own_pixel_textures`), sorted: what a draw copies of an attachment it reads
    /// through one on a device without the attachment feedback loop (`crate::shadow`).
    pub(crate) own_pixel: Vec<u32>,
    /// Each fixed-function vertex binding: its binding and the vertex buffer index it reads.
    pub(crate) bindings: Vec<(u32, u32)>,
    /// Whether a function reads through an address that is not one of its slots.
    pub(crate) reads_unbound: bool,
    /// Whether a function writes through an address that is not one of its slots.
    pub(crate) writes_unbound: bool,
    /// Whether a function writes memory (`PipelineInterface::writes_memory`).
    pub(crate) writes_memory: bool,
    /// The bytes of the push range the functions read, from 0.
    pub(crate) push_len: u32,
    /// Each push field: its offset, its function's tables, and its value.
    pub(crate) fields: Vec<(u32, usize, FieldValue)>,
    /// Whether a draw must be at base instance 0: a binding has a divisor other than 1 and the
    /// device does not report `supportsNonZeroFirstInstance` (VUID-vkCmdDraw-pNext-09461).
    pub(crate) first_instance_zero: bool,
}

/// The state Vulkan 1.3 lets a draw set, which every render `VkPipeline` takes as dynamic.
const DYNAMIC: [vk::DynamicState; 17] = [
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
];

/// What one render pipeline content compiles to, shared by every state of it and held by the
/// device's map of contents until the map forgets it.
pub(crate) struct RenderCompiled {
    core: Arc<Core>,
    /// Each stage's module, entry point, stage and specialization.
    stages: Vec<Module>,
    /// The fixed-function state every `VkPipeline` of the content takes.
    fixed: Fixed,
    /// The triangle class's `VkPipeline`, made with the content.
    triangles: vk::Pipeline,
    /// The line class's, made by the first draw of lines; null until then.
    lines: Mutex<vk::Pipeline>,
    /// The translator's notes of where the pipeline's answers may differ from Metal's.
    notes: Vec<PrecisionNote>,
    /// What a draw supplies.
    pub(crate) draws: Draws,
    /// The samplers a state numbers, the vertex function's then the fragment function's, each in
    /// Metal's order, and each constexpr sampler a function samples through, by its place there,
    /// with its use.
    numbered: Vec<NumberedSampler>,
    constexprs: Vec<(usize, CombinedSampler)>,
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
    /// Each instance-rate binding's divisor other than 1, a per-instance step rate the translator
    /// states only where device creation enabled the step-rate feature.
    divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR>,
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
    /// write mask and vertex descriptor of static strides Vulkan's fixed-function state takes with
    /// what device creation enables; what else the translator states is refused at the creation,
    /// by kind.
    ///
    /// The state numbers samplers as Metal's pipeline creation does
    /// (`PipelineInterface::numbered_samplers`), as
    /// [`new_compute_pipeline_state`](Self::new_compute_pipeline_state) numbers a kernel's: once
    /// the functions are lowered, `sampler_ids` takes the vertex function's, then the fragment
    /// function's, each in Metal's order, and answers the sampler index the consumer gives each.
    /// Each constexpr sampler a function samples through takes the device sampler of each use it
    /// has at its index before the call returns, and every draw pushes the index.
    ///
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses;
    /// [`Gap::IndirectCommandBuffers`]; [`Gap::RenderPipeline`] for what the runtime does not
    /// build, naming it; [`Refusal::SamplerIds`](crate::Refusal::SamplerIds) where `sampler_ids`
    /// answers another count; [`Refusal::IndicesNotGranted`](crate::Refusal::IndicesNotGranted)
    /// for a constexpr sampler's index;
    /// [`Limit::DeviceSamplers`](crate::Limit::DeviceSamplers); [`Error::Vulkan`].
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
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
                let entries = &core.entries;
                let keys = (core.translator)
                    .render_variant_key(descriptor, &VariantKey::new(), &budget, entries)
                    .map_err(Error::Translation)?;
                let compiled = RenderCompiled::of(core, descriptor, &keys, &budget, entries);
                core.note_entries();
                let compiled = compiled?;
                (core.kernels).keep_render(key, Arc::new(compiled), &core.counters)
            }
        };
        let numbered = number_samplers(
            core,
            compiled.numbered.clone(),
            &compiled.constexprs,
            sampler_ids,
            draw_input,
        )?;
        Ok(RenderPipelineState {
            inner: Arc::new(RenderInner { compiled, numbered }),
        })
    }
}

/// The samplers a state of a content numbers and the constexpr samplers its functions sample
/// through ([`RenderCompiled::numbered`]).
type Numbering = (Vec<NumberedSampler>, Vec<(usize, CombinedSampler)>);

/// [`Gap::RenderPipeline`] of [`RenderPipelineKind::DrawInput`].
fn draw_input() -> Error {
    Error::Unsupported(Gap::RenderPipeline {
        kind: RenderPipelineKind::DrawInput,
    })
}

/// The samplers a state of `pipeline` numbers and the constexpr samplers its functions sample
/// through ([`RenderCompiled::numbered`]), and what a draw of it supplies.
///
/// Err: [`Gap::RenderPipeline`] of [`RenderPipelineKind::DrawInput`] for a pipeline that reads
/// what a draw does not supply: a function table, a table the binder builds, a parameter block,
/// a push field or bind-time fact the encoder does not fill, or an unnormalized constexpr
/// sampler.
fn draws(
    pipeline: &RenderPipeline,
    nonzero_first_instance: bool,
) -> Result<(Draws, Numbering), Error> {
    let translated = &pipeline.interface;
    if translated.needs_pointer_translation
        || translated.needs_handle_translation
        || translated.reaches_command_buffers
        || !pipeline.layout.sets.iter().all(of_the_device)
    {
        return Err(draw_input());
    }
    let table = |stage| match stage {
        Stage::Vertex => Ok(VERTEX),
        Stage::Fragment => Ok(FRAGMENT),
        _ => Err(draw_input()),
    };
    // The slot's address, rounded down to a multiple of its argument's alignment.
    let mut aligned = Vec::new();
    for dependency in &pipeline.dependencies {
        match dependency.fact {
            RuntimeFact::AddressAlignment {
                slot:
                    SlotRef {
                        stage,
                        class: SlotClass::Buffer,
                        index,
                    },
                align,
            } if align.is_power_of_two() => {
                aligned.push((table(stage)?, index, u64::from(align) - 1))
            }
            RuntimeFact::DeviceAddresses { .. } => {}
            // The draw's base vertex and base instance: under the floor's `shaderDrawParameters`
            // the `BaseVertex` and `BaseInstance` built-ins, which are the draw's own Vulkan
            // parameters. The push field the translator places without the feature is a push
            // field the encoder does not fill, refused below.
            RuntimeFact::BaseVertexInstance => {}
            // A colour moved into a storage form, which the runtime never moves.
            RuntimeFact::MovableStorageWrites { .. } => {}
            // `RuntimeFact` is non-exhaustive: a fact this match does not name is one a draw does
            // not supply.
            _ => return Err(draw_input()),
        }
    }
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    for (slot, used) in &translated.slots {
        let at = table(slot.stage)?;
        match (slot.class, used) {
            (SlotClass::Buffer, SlotInterface::Buffer(buffer))
                if buffer.kind == BufferKind::Buffer =>
            {
                buffers.push((at, slot.index, access(buffer.access)));
            }
            (SlotClass::Texture, SlotInterface::Texture(texture)) => {
                textures.push((at, slot.index, texture_access(texture.access)));
            }
            (SlotClass::Sampler, SlotInterface::Sampler(_)) => {}
            _ => return Err(draw_input()),
        }
    }
    // The samplers a state numbers: the vertex function's, then the fragment function's.
    let mut numbered = Vec::new();
    let mut firsts = [0; 2];
    for (stage, samplers) in &translated.numbered_samplers {
        firsts[table(*stage)?] = numbered.len();
        numbered.extend(samplers.iter().copied());
    }
    let mut constexprs = Vec::new();
    let mut fields = Vec::new();
    let mut push_len = 0;
    for placement in &translated.push {
        let at = table(placement.stage)?;
        if placement.argument_table_size != 0 {
            return Err(draw_input());
        }
        if let Some((offset, size)) = placement.push_range {
            push_len = push_len.max(offset + size);
        }
        for placed in &placement.fields {
            let PushLoc::Push { offset } = placed.loc else {
                return Err(draw_input());
            };
            let value = match placed.field {
                PushField::SlotGuestBase(SlotRef {
                    stage,
                    class,
                    index,
                }) => {
                    let of = table(stage)?;
                    let value = match class {
                        SlotClass::Buffer => FieldValue::BufferAddress {
                            slot: index,
                            mask: (aligned.iter())
                                .find(|&&(table, slot, _)| table == of && slot == index)
                                .map_or(0, |&(_, _, mask)| mask),
                        },
                        SlotClass::Texture => FieldValue::TextureId(index),
                        SlotClass::Sampler => FieldValue::SamplerId(index),
                    };
                    fields.push((offset, of, value));
                    continue;
                }
                PushField::TextureIntegerChannels(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index,
                }) => {
                    fields.push((offset, table(stage)?, FieldValue::IntegerChannels(index)));
                    continue;
                }
                PushField::ConstexprSampler(sampler) => {
                    let SamplerRef::Constexpr(words) = sampler.sampler else {
                        return Err(draw_input());
                    };
                    let first = firsts[at];
                    let place = (numbered[first..].iter())
                        .position(|&n| n == NumberedSampler::Constexpr(words))
                        .ok_or_else(draw_input)?
                        + first;
                    if sampler.unnormalized {
                        return Err(draw_input());
                    }
                    constexprs.push((place, sampler));
                    FieldValue::ConstexprSampler(place)
                }
                _ => return Err(draw_input()),
            };
            fields.push((offset, at, value));
        }
    }
    if push_len > PUSH_BYTES {
        return Err(draw_input());
    }
    let bindings = (pipeline.vertex_input.bindings.iter())
        .map(|binding| (binding.binding, binding.buffer_index))
        .collect();
    let divided = (pipeline.vertex_input.bindings.iter()).any(|binding| binding.divisor != 1);
    // Only a fragment stage states them.
    let mut own_pixel: Vec<u32> = (pipeline.stages.iter())
        .flat_map(|stage| stage.own_pixel_textures.iter().copied())
        .collect();
    own_pixel.sort_unstable();
    own_pixel.dedup();
    let draws = Draws {
        buffers,
        textures,
        own_pixel,
        bindings,
        reads_unbound: translated.reads_unbound_resources,
        writes_unbound: translated.writes_unbound_resources,
        writes_memory: translated.writes_memory,
        push_len,
        fields,
        first_instance_zero: divided && !nonzero_first_instance,
    };
    Ok((draws, (numbered, constexprs)))
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
        let divisors = (pipeline.vertex_input.bindings.iter())
            .filter(|binding| binding.divisor != 1)
            .map(|binding| vk::VertexInputBindingDivisorDescriptionKHR {
                binding: binding.binding,
                divisor: binding.divisor,
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
            divisors,
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
        let nonzero_first_instance = translator
            .profile()
            .limits()
            .supports_non_zero_first_instance;
        let (draws, (numbered, constexprs)) = draws(&pipeline, nonzero_first_instance)?;
        let samples = descriptor.raster_sample_count;
        let mut stages = Vec::new();
        let mut notes = Vec::new();
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
        core.counters.rise(Gauge::RenderContents);
        let mut compiled = RenderCompiled {
            core: Arc::clone(core),
            stages,
            fixed,
            triangles: vk::Pipeline::null(),
            lines: Mutex::new(vk::Pipeline::null()),
            notes,
            draws,
            numbered,
            constexprs,
        };
        // Made whole before the `VkPipeline`, so dropping it releases what was made.
        compiled.triangles = compiled.make(vk::PrimitiveTopology::TRIANGLE_LIST)?;
        Ok(compiled)
    }

    /// The `VkPipeline` of the content for draws of `topology`'s class, a line or triangle one:
    /// the line class's made by its first draw, which waits for it, a first-use wait
    /// ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)).
    ///
    /// Err: [`Error::Vulkan`] where the pipeline cannot be made.
    pub(crate) fn pipeline(&self, topology: vk::PrimitiveTopology) -> Result<vk::Pipeline, Error> {
        if topology != vk::PrimitiveTopology::LINE_LIST
            && topology != vk::PrimitiveTopology::LINE_STRIP
        {
            return Ok(self.triangles);
        }
        let mut lines = lock(&self.lines);
        if *lines == vk::Pipeline::null() {
            let began = host_time();
            *lines = self.make(vk::PrimitiveTopology::LINE_LIST)?;
            (self.core.counters).first_use_wait(host_time().saturating_sub(began));
        }
        Ok(*lines)
    }

    /// Whether the content's `VkPipeline`s render into `pass`'s attachments: its colour formats are
    /// the pass's colour attachments', by index, an absent one where the pass has none, and its
    /// stencil format the pass's stencil attachment's, or none without one.
    pub(crate) fn renders_into(&self, pass: &Pass) -> bool {
        let fixed = &self.fixed;
        let format = |t: Option<&crate::pass::Target>| {
            t.map_or(vk::Format::UNDEFINED, |t| t.attachment.format)
        };
        fixed.colors.len() == pass.colors.len()
            && (fixed.colors.iter().zip(&pass.colors)).all(|(&f, t)| f == format(t.as_ref()))
            && fixed.stencil == format(pass.stencil.as_ref())
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
        let mut divided = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
            .vertex_binding_divisors(&fixed.divisors);
        let mut vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&fixed.bindings)
            .vertex_attribute_descriptions(&fixed.attributes);
        // The divisor structure has at least one divisor where it is chained.
        if !fixed.divisors.is_empty() {
            vertex_input = vertex_input.push_next(&mut divided);
        }
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
        // Where the device reads attachments through textures, every pipeline takes the feedback
        // loop's enable as dynamic state too: a state is made before any draw binds its textures.
        let feedback = core.feedback.is_some();
        let states: Vec<vk::DynamicState> = (DYNAMIC.iter().copied())
            .chain(feedback.then_some(vk::DynamicState::ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT))
            .collect();
        let dynamic_states = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
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
        made_through_cache(core, "vkCreateGraphicsPipelines", |cache| {
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
        self.inner.compiled.notes.clone()
    }

    /// Whether two handles are one state.
    pub(crate) fn is(&self, other: &RenderPipelineState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.inner.compiled.core, core)
    }
}

impl Drop for RenderInner {
    fn drop(&mut self) {
        self.numbered.release(&self.compiled.core);
    }
}

impl Drop for RenderCompiled {
    fn drop(&mut self) {
        // No state holds it, so no command buffer that used it is left: each kept its state until
        // it retired.
        self.core.counters.fall(Gauge::RenderContents);
        let lines = *self.lines.get_mut().unwrap_or_else(|e| e.into_inner());
        for pipeline in [self.triangles, lines] {
            if pipeline != vk::Pipeline::null() {
                (self.core.reaper).give(Job::Free(Free::Pipeline(pipeline)));
            }
        }
        release(&self.core, &self.stages);
    }
}

impl crate::kept::Handle for RenderPipelineState {
    fn address(&self) -> usize {
        Arc::as_ptr(&self.inner) as *const () as usize
    }

    unsafe fn uncounted(&self) -> std::mem::ManuallyDrop<RenderPipelineState> {
        // SAFETY: the pointer is a live `Arc`'s; the handle made of it is never dropped, and the
        // caller keeps the object alive past its last use.
        let inner = unsafe { Arc::from_raw(Arc::as_ptr(&self.inner)) };
        std::mem::ManuallyDrop::new(RenderPipelineState { inner })
    }
}
