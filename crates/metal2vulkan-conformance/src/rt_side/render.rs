//! The rt side's render vocabulary: targets, render pipeline states, depth and stencil states, and
//! one command buffer of passes, read-backs, copies and dispatches ([`RenderWork`]).
//!
//! A pass field or a step the runtime has no counterpart of answers `Unavailable` with its Metal
//! name and ends that piece of the work: its encoder is ended with what came before, as Metal keeps
//! a pass's other commands, and the command buffer still commits. A range setter is its slots set
//! one by one, which is what Metal's range call is.

use super::{keep, waited_for, RtSide, WAIT};
use crate::case::{Answer, Failure, Handle, PassOutcome};
use crate::render::{
    AttachmentPlace, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, ScissorBox,
    StageFunction, TargetSpec, ViewportRect,
};
use crate::transcript::{Negative, Value};
use crate::work::Plane;
use metal2vulkan_rt as rt;
use metal2vulkan_runtime::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
    StoreActionOptions, VisibilityResultMode,
};

/// A typed error of the runtime as the failure it records.
fn of_rt(error: rt::Error) -> Failure {
    Failure::of_rt(&error)
}

/// The runtime's load action for a case's.
fn load(action: LoadAction) -> Answer<rt::LoadAction> {
    Ok(match action {
        LoadAction::DontCare => rt::LoadAction::DontCare,
        LoadAction::Load => rt::LoadAction::Load,
        LoadAction::Clear => rt::LoadAction::Clear,
        other => return Err(Failure::named("NoSuchLoadAction", format!("{other:?}"))),
    })
}

/// The runtime's store action for a case's.
fn store(action: StoreAction) -> Answer<rt::StoreAction> {
    Ok(match action {
        StoreAction::DontCare => rt::StoreAction::DontCare,
        StoreAction::Store => rt::StoreAction::Store,
        StoreAction::MultisampleResolve => rt::StoreAction::MultisampleResolve,
        StoreAction::StoreAndMultisampleResolve => rt::StoreAction::StoreAndMultisampleResolve,
        StoreAction::Unknown => rt::StoreAction::Unknown,
        StoreAction::CustomSampleDepthStore => rt::StoreAction::CustomSampleDepthStore,
        other => return Err(Failure::named("NoSuchStoreAction", format!("{other:?}"))),
    })
}

/// An attachment's place, refused where it names what the runtime's attachment has no field
/// for: a depth plane or store action options.
fn plain(place: &AttachmentPlace, options: StoreActionOptions) -> Answer<()> {
    if place.depth_plane != 0 {
        return Err(Failure::unavailable("depthPlane"));
    }
    if options != StoreActionOptions::default() {
        return Err(Failure::unavailable("storeActionOptions"));
    }
    Ok(())
}

/// The runtime's depth resolve filter for a case's.
fn depth_filter(
    filter: MultisampleDepthResolveFilter,
) -> Answer<rt::MultisampleDepthResolveFilter> {
    Ok(match filter {
        MultisampleDepthResolveFilter::Sample0 => rt::MultisampleDepthResolveFilter::Sample0,
        MultisampleDepthResolveFilter::Min => rt::MultisampleDepthResolveFilter::Min,
        MultisampleDepthResolveFilter::Max => rt::MultisampleDepthResolveFilter::Max,
        other => {
            return Err(Failure::named(
                "NoSuchDepthResolveFilter",
                format!("{other:?}"),
            ))
        }
    })
}

/// The runtime's stencil resolve filter for a case's.
fn stencil_filter(
    filter: MultisampleStencilResolveFilter,
) -> Answer<rt::MultisampleStencilResolveFilter> {
    Ok(match filter {
        MultisampleStencilResolveFilter::Sample0 => rt::MultisampleStencilResolveFilter::Sample0,
        MultisampleStencilResolveFilter::DepthResolvedSample => {
            rt::MultisampleStencilResolveFilter::DepthResolvedSample
        }
        other => {
            return Err(Failure::named(
                "NoSuchStencilResolveFilter",
                format!("{other:?}"),
            ))
        }
    })
}

/// The runtime's viewport of the case's.
fn viewport(rect: &ViewportRect) -> rt::Viewport {
    rt::Viewport {
        origin_x: rect.origin[0],
        origin_y: rect.origin[1],
        width: rect.size[0],
        height: rect.size[1],
        znear: rect.depth[0],
        zfar: rect.depth[1],
    }
}

/// The runtime's scissor rectangle of the case's.
fn scissor(rect: &ScissorBox) -> rt::ScissorRect {
    rt::ScissorRect {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

/// The step's variant, which names the Metal call a refusal is of.
fn step_name(step: &RenderStep) -> String {
    let debug = format!("{step:?}");
    let name = debug
        .split(|character: char| !character.is_ascii_alphanumeric())
        .next()
        .unwrap_or("step");
    format!("render step {name}")
}

impl RtSide<'_> {
    pub(super) fn held_sampler(&self, handle: Handle) -> Answer<&rt::SamplerState> {
        (self.samplers.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// `newTextureWithDescriptor:` of a render target: Shared or Private, multisampled or of the
    /// shape the spec states, at a texture index where a shader reaches it. A target over a buffer
    /// is the buffer's `newTextureWithDescriptor:offset:bytesPerRow:` of one level of 2D, whose
    /// storage is the buffer's, as the old runtime's side makes it. A target over a surface's
    /// plane is framebuffer-only and rotated, by Apple's private descriptor setters, which the
    /// runtime has no counterpart of: `setRotation:` where it turns, else `setFramebufferOnly:`.
    pub(super) fn make_target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        if let Some(plane) = &spec.surface {
            return Err(Failure::unavailable(match plane.rotation {
                0 => "setFramebufferOnly:",
                _ => "setRotation:",
            }));
        }
        let mut descriptor = rt::TextureDescriptor::new(
            rt::TextureType::Type2D,
            spec.format,
            spec.size[0],
            spec.size[1],
        );
        let mut usage = rt::TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= rt::TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= rt::TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        if let Some(over) = &spec.over_buffer {
            if let Some(why) = over.unsafe_stride() {
                return Err(Failure::named("WouldEndMetalsProcess", why));
            }
            let index = self.on.texture_index(usage);
            let buffer = self.held(over.buffer)?;
            let texture = (buffer.new_texture(&descriptor, index, over.offset, over.bytes_per_row))
                .map_err(of_rt)?;
            return Ok(self.keep_texture(texture));
        }
        descriptor.storage_mode = match spec.shared {
            true => rt::StorageMode::Shared,
            false => rt::StorageMode::Private,
        };
        // The shape before the samples, so a multisampled array target keeps the array type the
        // shape states.
        if let Some(shape) = &spec.shape {
            descriptor.texture_type = shape.texture_type;
            descriptor.mipmap_level_count = shape.levels;
            descriptor.array_length = shape.array_length;
            descriptor.depth = shape.depth;
        }
        if spec.samples > 1 {
            if spec.shape.is_none() {
                descriptor.texture_type = rt::TextureType::Type2DMultisample;
            }
            descriptor.sample_count = spec.samples;
        }
        let index = (spec.shader_read || spec.shader_write)
            .then(|| self.on.next_index(rt::IndexClass::Texture));
        let texture = (self.on.device.new_texture(&descriptor, index)).map_err(of_rt)?;
        Ok(self.keep_texture(texture))
    }

    /// `newRenderPipelineStateWithDescriptor:error:` of the descriptor the old runtime's side
    /// states for the same spec, but for the label, which the runtime takes nowhere.
    pub(super) fn make_render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let stage = |stage: &StageFunction<'_>| -> Answer<rt::StageSource> {
            let named = stage.library.unwrap_or(library);
            let function = (self.held_library(named)?.function(stage.entry))
                .map_err(|error| self.on.translation_failure(&error))?;
            Ok(rt::StageSource::Function(rt::FunctionDescriptor::new(
                function,
            )))
        };
        let mut descriptor = rt::RenderPipelineDescriptor::new(stage(&spec.vertex)?);
        descriptor.fragment = spec.fragment.as_ref().map(stage).transpose()?;
        let apart = crate::runtime_side::pipeline_fields(spec)?;
        descriptor.color_attachments = apart.color_attachments;
        descriptor.depth_format = apart.depth_format;
        descriptor.stencil_format = apart.stencil_format;
        descriptor.raster_sample_count = apart.raster_sample_count;
        descriptor.rasterization_enabled = apart.rasterization_enabled;
        descriptor.vertex_descriptor = apart.vertex_descriptor;
        descriptor.support_indirect_command_buffers = apart.support_indirect_command_buffers;
        descriptor.tessellation = apart.tessellation;
        descriptor.alpha_to_coverage = apart.alpha_to_coverage;
        descriptor.alpha_to_one = apart.alpha_to_one;
        descriptor.input_primitive_topology = apart.input_primitive_topology;
        let on = self.on;
        let state = (on.device)
            .new_render_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    /// One command buffer of `work` in order, committed and waited for within [`WAIT`].
    pub(super) fn run_render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut refused = Vec::with_capacity(work.len());
        for piece in work {
            let answered = match piece {
                RenderWork::Pass(pass) => self.encode_pass(&command_buffer, pass),
                RenderWork::ReadTexture {
                    texture,
                    into,
                    offset,
                    row_pitch,
                    size,
                    plane,
                    at,
                } => (|| {
                    if *plane != Plane::Whole {
                        return Err(Failure::unavailable(
                            "copyFromTexture:toBuffer: with a blit option",
                        ));
                    }
                    let (texture, into) = (self.held_texture(*texture)?, self.held(*into)?);
                    let mut blit = command_buffer.blit_command_encoder().map_err(of_rt)?;
                    let place = rt::TexturePlace {
                        slice: at.slice,
                        level: at.level,
                        origin: [0, 0, at.depth_plane],
                    };
                    let layout = rt::BufferLayout {
                        offset: *offset,
                        bytes_per_row: u64::from(*row_pitch),
                        bytes_per_image: u64::from(*row_pitch) * u64::from(size[1]),
                    };
                    let copied = blit.copy_from_texture_to_buffer(
                        texture,
                        place,
                        [size[0], size[1], 1],
                        into,
                        layout,
                    );
                    blit.end_encoding();
                    copied.map_err(of_rt)
                })(),
                RenderWork::CopyBuffer {
                    from,
                    from_offset,
                    into,
                    into_offset,
                    length,
                } => (|| {
                    let (from, into) = (self.held(*from)?, self.held(*into)?);
                    let mut blit = command_buffer.blit_command_encoder().map_err(of_rt)?;
                    let copied = blit.copy_from_buffer_to_buffer(
                        from,
                        *from_offset,
                        into,
                        *into_offset,
                        *length,
                    );
                    blit.end_encoding();
                    copied.map_err(of_rt)
                })(),
                RenderWork::Dispatch(dispatch) => (|| {
                    let state =
                        (self.pipelines.get(dispatch.pipeline.0 as usize)).ok_or_else(|| {
                            Failure::named(
                                "NoSuchHandle",
                                format!("compute pipeline {}", dispatch.pipeline.0),
                            )
                        })?;
                    let mut encoder = (command_buffer
                        .compute_command_encoder(rt::DispatchType::Serial))
                    .map_err(of_rt)?;
                    let encoded = (|| {
                        encoder.set_compute_pipeline_state(state).map_err(of_rt)?;
                        for (index, (held, offset)) in dispatch.buffers.iter().enumerate() {
                            (encoder.set_buffer(Some(self.held(*held)?), *offset, index as u32))
                                .map_err(of_rt)?;
                        }
                        for (index, held) in dispatch.textures.iter().enumerate() {
                            (encoder.set_texture(Some(self.held_texture(*held)?), index as u32))
                                .map_err(of_rt)?;
                        }
                        (encoder.dispatch_threads(dispatch.grid, dispatch.threads)).map_err(of_rt)
                    })();
                    encoder.end_encoding();
                    encoded
                })(),
                RenderWork::ComputeCommands { .. }
                | RenderWork::ResetCommands { .. }
                | RenderWork::CopyCommands { .. }
                | RenderWork::OptimizeCommands { .. } => Err(Failure::unavailable(
                    "newIndirectCommandBufferWithDescriptor:",
                )),
            };
            if answered.is_ok() {
                match piece {
                    RenderWork::Dispatch(dispatch) => {
                        self.dispatched.insert(dispatch.pipeline.0 as usize);
                    }
                    RenderWork::Pass(pass) => {
                        self.drawn
                            .extend(pass.steps.iter().filter_map(|step| match step {
                                RenderStep::Pipeline(state) => Some(state.0 as usize),
                                _ => None,
                            }));
                        let notes = pass.steps.iter().filter_map(|step| match step {
                            RenderStep::DepthBias(v) if v.bias != 0.0 || v.slope_scale != 0.0 => {
                                Some(rt::RenderCommandEncoder::DEPTH_BIAS_NOTE)
                            }
                            RenderStep::DepthStencilState(state) => (self.depth_stencils)
                                .get(state.0 as usize)
                                .and_then(rt::DepthStencilState::precision_note),
                            _ => None,
                        });
                        let notes: Vec<rt::PrecisionNote> = notes.collect();
                        self.encoder_notes.extend(notes);
                    }
                    _ => {}
                }
            }
            refused.push(answered.err());
        }
        command_buffer.commit().map_err(of_rt)?;
        let Some(word) = waited_for(&command_buffer, WAIT) else {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        };
        Ok(PassOutcome {
            refused,
            status: Value::unit_variant(word),
            error: (command_buffer.error()).map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    /// One render pass of `pass` encoded into `command_buffer`: its encoder ended with the steps
    /// before the first that was refused, which is the answer.
    fn encode_pass(&self, command_buffer: &rt::CommandBuffer, pass: &RenderPass) -> Answer<()> {
        let mut descriptor = rt::RenderPassDescriptor::default();
        if let Some(size) = pass.size {
            descriptor.render_target_width = size[0];
            descriptor.render_target_height = size[1];
        }
        for (at, color) in pass.colors.iter().enumerate() {
            let Some(held) = color.texture else { continue };
            plain(&color.place, color.options)?;
            let attachment = (descriptor.color_attachments.get_mut(at))
                .ok_or_else(|| Failure::named("NoSuchAttachment", format!("color {at}")))?;
            attachment.texture = Some(self.held_texture(held)?.clone());
            attachment.level = color.place.level;
            attachment.slice = color.place.slice;
            attachment.resolve_texture = (color.resolve)
                .map(|resolve| self.held_texture(resolve).cloned())
                .transpose()?;
            attachment.resolve_level = color.place.resolve_level;
            attachment.resolve_slice = color.place.resolve_slice;
            attachment.resolve_depth_plane = color.place.resolve_depth_plane;
            attachment.load_action = load(color.load)?;
            attachment.store_action = store(color.store)?;
            let [red, green, blue, alpha] = color.clear;
            attachment.clear_color = rt::ClearColor::new(red, green, blue, alpha);
        }
        if let Some(depth) = &pass.depth {
            plain(&depth.place, depth.options)?;
            let attachment = &mut descriptor.depth_attachment;
            attachment.texture = Some(self.held_texture(depth.texture)?.clone());
            attachment.level = depth.place.level;
            attachment.slice = depth.place.slice;
            attachment.resolve_texture = (depth.resolve)
                .map(|resolve| self.held_texture(resolve).cloned())
                .transpose()?;
            attachment.resolve_level = depth.place.resolve_level;
            attachment.resolve_slice = depth.place.resolve_slice;
            attachment.resolve_depth_plane = depth.place.resolve_depth_plane;
            attachment.depth_resolve_filter = depth_filter(depth.filter)?;
            attachment.load_action = load(depth.load)?;
            attachment.store_action = store(depth.store)?;
            attachment.clear_depth = depth.clear;
        }
        if let Some(stencil) = &pass.stencil {
            plain(&stencil.place, stencil.options)?;
            let attachment = &mut descriptor.stencil_attachment;
            attachment.texture = Some(self.held_texture(stencil.texture)?.clone());
            attachment.level = stencil.place.level;
            attachment.slice = stencil.place.slice;
            attachment.resolve_texture = (stencil.resolve)
                .map(|resolve| self.held_texture(resolve).cloned())
                .transpose()?;
            attachment.resolve_level = stencil.place.resolve_level;
            attachment.resolve_slice = stencil.place.resolve_slice;
            attachment.resolve_depth_plane = stencil.place.resolve_depth_plane;
            attachment.stencil_resolve_filter = stencil_filter(stencil.filter)?;
            attachment.load_action = load(stencil.load)?;
            attachment.store_action = store(stencil.store)?;
            attachment.clear_stencil = stencil.clear;
        }
        // `visibilityResultType` is not the runtime's field: it decides only what an enabled mode
        // writes, and the runtime builds `Disabled` alone.
        if let Some(visibility) = pass.visibility {
            descriptor.visibility_result_buffer = Some(self.held(visibility)?.clone());
        }
        if pass.array_length != 0 {
            return Err(Failure::unavailable("renderTargetArrayLength"));
        }
        if !pass.positions.is_empty() {
            return Err(Failure::unavailable("setSamplePositions:count:"));
        }
        let mut encoder = (command_buffer.render_command_encoder(&descriptor)).map_err(of_rt)?;
        let encoded = (pass.steps.iter()).try_for_each(|step| self.step(&mut encoder, step));
        encoder.end_encoding();
        encoded
    }

    /// One step of a pass on `encoder`.
    fn step(&self, encoder: &mut rt::RenderCommandEncoder, step: &RenderStep) -> Answer<()> {
        let buffer = |handle: Handle| self.held(handle);
        let texture = |handle: Handle| self.held_texture(handle);
        let sampler = |handle: Handle| self.held_sampler(handle);
        let indices = |held: Handle, offset: u64, index_type: rt::IndexType| {
            buffer(held).map(|buffer| rt::IndexBuffer {
                buffer,
                offset,
                index_type,
            })
        };
        match step {
            RenderStep::Pipeline(state) => {
                let state = (self.render_pipelines.get(state.0 as usize)).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("render pipeline {}", state.0))
                })?;
                encoder.set_render_pipeline_state(state).map_err(of_rt)
            }
            RenderStep::DepthStencilState(state) => {
                let state = (self.depth_stencils.get(state.0 as usize)).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                })?;
                encoder.set_depth_stencil_state(Some(state)).map_err(of_rt)
            }
            RenderStep::StencilReference(value) => {
                encoder.set_stencil_reference_value(*value);
                Ok(())
            }
            RenderStep::DepthBias(v) => {
                (encoder.set_depth_bias(v.bias, v.slope_scale, v.clamp)).map_err(of_rt)
            }
            RenderStep::FillMode(mode) => encoder.set_triangle_fill_mode(*mode).map_err(of_rt),
            RenderStep::VisibilityResultMode { mode, offset } => {
                let mode = match mode {
                    VisibilityResultMode::Disabled => rt::VisibilityResultMode::Disabled,
                    VisibilityResultMode::Boolean => rt::VisibilityResultMode::Boolean,
                    VisibilityResultMode::Counting => rt::VisibilityResultMode::Counting,
                    _ => return Err(Failure::unavailable("setVisibilityResultMode:offset:")),
                };
                (encoder.set_visibility_result_mode(mode, *offset)).map_err(of_rt)
            }
            RenderStep::DepthClip(mode) => encoder.set_depth_clip_mode(*mode).map_err(of_rt),
            RenderStep::StencilReferences { front, back } => {
                encoder.set_stencil_front_back_reference_value(*front, *back);
                Ok(())
            }
            RenderStep::VertexBuffer {
                buffer: held,
                offset,
                index,
            } => (encoder.set_vertex_buffer(Some(buffer(*held)?), *offset, *index)).map_err(of_rt),
            RenderStep::FragmentBuffer {
                buffer: held,
                offset,
                index,
            } => {
                (encoder.set_fragment_buffer(Some(buffer(*held)?), *offset, *index)).map_err(of_rt)
            }
            RenderStep::VertexBuffers { buffers, first } => (buffers.iter().zip(*first..))
                .try_for_each(|((held, offset), index)| {
                    (encoder.set_vertex_buffer(Some(buffer(*held)?), *offset, index)).map_err(of_rt)
                }),
            RenderStep::FragmentBuffers { buffers, first } => (buffers.iter().zip(*first..))
                .try_for_each(|((held, offset), index)| {
                    (encoder.set_fragment_buffer(Some(buffer(*held)?), *offset, index))
                        .map_err(of_rt)
                }),
            RenderStep::VertexBufferOffset { offset, index } => encoder
                .set_vertex_buffer_offset(*offset, *index)
                .map_err(of_rt),
            RenderStep::FragmentBufferOffset { offset, index } => encoder
                .set_fragment_buffer_offset(*offset, *index)
                .map_err(of_rt),
            RenderStep::VertexBytes { bytes, index } => {
                encoder.set_vertex_bytes(bytes, *index).map_err(of_rt)
            }
            RenderStep::FragmentBytes { bytes, index } => {
                encoder.set_fragment_bytes(bytes, *index).map_err(of_rt)
            }
            RenderStep::VertexTexture {
                texture: held,
                index,
            } => (encoder.set_vertex_texture(Some(texture(*held)?), *index)).map_err(of_rt),
            RenderStep::FragmentTexture {
                texture: held,
                index,
            } => (encoder.set_fragment_texture(Some(texture(*held)?), *index)).map_err(of_rt),
            RenderStep::VertexTextures { textures, first } => (textures.iter().zip(*first..))
                .try_for_each(|(held, index)| {
                    (encoder.set_vertex_texture(Some(texture(*held)?), index)).map_err(of_rt)
                }),
            RenderStep::FragmentTextures { textures, first } => (textures.iter().zip(*first..))
                .try_for_each(|(held, index)| {
                    (encoder.set_fragment_texture(Some(texture(*held)?), index)).map_err(of_rt)
                }),
            RenderStep::VertexSampler {
                sampler: held,
                index,
            } => (encoder.set_vertex_sampler_state(Some(sampler(*held)?), *index)).map_err(of_rt),
            RenderStep::FragmentSampler {
                sampler: held,
                index,
            } => (encoder.set_fragment_sampler_state(Some(sampler(*held)?), *index)).map_err(of_rt),
            RenderStep::VertexSamplers { samplers, first } => (samplers.iter().zip(*first..))
                .try_for_each(|(held, index)| {
                    (encoder.set_vertex_sampler_state(Some(sampler(*held)?), index)).map_err(of_rt)
                }),
            RenderStep::FragmentSamplers { samplers, first } => (samplers.iter().zip(*first..))
                .try_for_each(|(held, index)| {
                    (encoder.set_fragment_sampler_state(Some(sampler(*held)?), index))
                        .map_err(of_rt)
                }),
            RenderStep::Viewport(rect) => encoder.set_viewport(viewport(rect)).map_err(of_rt),
            RenderStep::Viewports(rects) => {
                let viewports: Vec<rt::Viewport> = rects.iter().map(viewport).collect();
                encoder.set_viewports(&viewports).map_err(of_rt)
            }
            RenderStep::Scissor(rect) => {
                encoder.set_scissor_rect(scissor(rect));
                Ok(())
            }
            RenderStep::Scissors(rects) => {
                let rects: Vec<rt::ScissorRect> = rects.iter().map(scissor).collect();
                encoder.set_scissor_rects(&rects).map_err(of_rt)
            }
            RenderStep::Cull(mode) => {
                encoder.set_cull_mode(*mode);
                Ok(())
            }
            RenderStep::FrontFacing(winding) => {
                encoder.set_front_facing_winding(*winding);
                Ok(())
            }
            RenderStep::BlendColor(channels) => {
                let [red, green, blue, alpha] = channels.0;
                encoder.set_blend_color(red, green, blue, alpha);
                Ok(())
            }
            // Metal's short draw selectors are the long ones with its documented defaults.
            RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            }
            | RenderStep::DrawInstanced {
                primitive,
                first,
                count,
                instances,
            } => {
                (encoder.draw_primitives(*primitive, *first, *count, *instances, 0)).map_err(of_rt)
            }
            RenderStep::DrawPlain {
                primitive,
                first,
                count,
            } => (encoder.draw_primitives(*primitive, *first, *count, 1, 0)).map_err(of_rt),
            RenderStep::DrawIndexed {
                primitive,
                count,
                index_type,
                indices: held,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    indices(*held, *offset, *index_type)?,
                    *instances,
                    *base_vertex,
                    *base_instance,
                )
                .map_err(of_rt),
            RenderStep::DrawIndexedPlain {
                primitive,
                count,
                index_type,
                indices: held,
                offset,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    indices(*held, *offset, *index_type)?,
                    1,
                    0,
                    0,
                )
                .map_err(of_rt),
            RenderStep::DrawIndexedInstanced {
                primitive,
                count,
                index_type,
                indices: held,
                offset,
                instances,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    indices(*held, *offset, *index_type)?,
                    *instances,
                    0,
                    0,
                )
                .map_err(of_rt),
            RenderStep::TessellationFactorBuffer {
                buffer: held,
                offset,
                instance_stride,
            } => {
                let held = held.map(buffer).transpose()?;
                (encoder.set_tessellation_factor_buffer(held, *offset, *instance_stride))
                    .map_err(of_rt)
            }
            RenderStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
            } => {
                let indices = match patch_indices {
                    Some(held) => Some((buffer(*held)?, *patch_index_offset)),
                    None => None,
                };
                (encoder.draw_patches(
                    *control_points,
                    *first,
                    *count,
                    indices,
                    *instances,
                    *base_instance,
                ))
                .map_err(of_rt)
            }
            // The scope and the stages only narrow what Metal orders: the runtime's barrier orders
            // every access of the encoder's draws so far before its later ones.
            RenderStep::BarrierScope { .. } => encoder.memory_barrier().map_err(of_rt),
            other => Err(Failure::unavailable(&step_name(other))),
        }
    }
}
