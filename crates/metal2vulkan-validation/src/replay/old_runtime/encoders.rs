//! The calls of command buffers and of their compute, blit and render encoders.

use super::args::{
    arg, data, failed, index, list, metal, narrow, origin, range, refuse, size, uint, variant,
    Fields,
};
use super::{own_buffer, Object, OldRuntime};
use crate::replay::player::{piece, placed_runs, upload, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan::metal::{BarrierScope, DispatchType, IndexType, RenderStages, ResourceUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, BufferContents, ComputeCommandEncoder,
    DrawPrimitiveType, RenderCommandEncoder, ResourceRef, ScissorRect, StoreAction, Texture,
    Viewport,
};
use std::ops::Range;

/// A resource an encoder's call names, `{Buffer: object}` or `{Texture: object}`.
enum Resource {
    Buffer(Buffer),
    Texture(Texture),
}

impl Resource {
    fn as_ref(&self) -> ResourceRef<'_> {
        match self {
            Resource::Buffer(buffer) => ResourceRef::Buffer(buffer),
            Resource::Texture(texture) => ResourceRef::Texture(texture),
        }
    }
}

/// The first index of a `withRange:` call's range, which covers each of its `count` objects.
fn first(call: &Call, at: usize, count: usize) -> Result<u32, PlayError> {
    let Range { start, end } = range(call, arg(call, at)?)?;
    if end - start != count as u64 {
        return Err(refuse(
            call,
            format!("a range of {} for {count}", end - start),
        ));
    }
    narrow(call, start, "the range's location")
}

/// The Metal value of argument `at`.
fn raw<T: TryFrom<u32>>(call: &Call, at: usize) -> Result<T, PlayError>
where
    T::Error: std::fmt::Display,
{
    metal(call, uint(call, at)?)
}

/// Argument `at`, a float, as an `f32`.
fn float(call: &Call, at: usize) -> Result<f32, PlayError> {
    match arg(call, at)?.as_f64() {
        Some(value) => Ok(value as f32),
        None => Err(refuse(call, format!("argument {at} is no float"))),
    }
}

/// A LOD clamp of a list, as an `f32`.
fn clamp(call: &Call, value: &Value) -> Result<f32, PlayError> {
    match value.as_f64() {
        Some(clamp) => Ok(clamp as f32),
        None => Err(refuse(call, format!("{value} is no clamp"))),
    }
}

/// The viewport of `value`, an `MTLViewport`'s fields.
fn viewport(call: &Call, value: &Value) -> Result<Viewport, PlayError> {
    let known = ["originX", "originY", "width", "height", "znear", "zfar"];
    let f = Fields::of(call, value, &known)?;
    let [x, y, width, height, near, far] = known.map(|key| f.required(key, Fields::float));
    Ok(Viewport::new(x?, y?, width?, height?, near?, far?))
}

/// The scissor rectangle of `value`, an `MTLScissorRect`'s fields.
fn scissor(call: &Call, value: &Value) -> Result<ScissorRect, PlayError> {
    let known = ["x", "y", "width", "height"];
    let f = Fields::of(call, value, &known)?;
    let [x, y, width, height] = known.map(|key| f.required(key, Fields::uint));
    Ok(ScissorRect::new(x?, y?, width?, height?))
}

/// `BlitOption` of its raw value.
fn blit_option(call: &Call, raw: u64) -> Result<BlitOption, PlayError> {
    match raw {
        0 => Ok(BlitOption::None),
        1 => Ok(BlitOption::DepthFromDepthStencil),
        2 => Ok(BlitOption::StencilFromDepthStencil),
        4 => Ok(BlitOption::RowLinearPVRTC),
        _ => Err(refuse(call, format!("blit option {raw}"))),
    }
}

impl OldRuntime {
    fn resource(&self, call: &Call, at: usize) -> Result<Resource, PlayError> {
        let (kind, object) = variant(call, arg(call, at)?)?;
        let id = object
            .as_object()
            .ok_or_else(|| refuse(call, format!("{kind} {object}")))?;
        match kind {
            "Buffer" => self.buffer_of(call, id).map(Resource::Buffer),
            "Texture" => self.texture_of(call, id).map(Resource::Texture),
            _ => Err(refuse(call, format!("a resource of kind {kind}"))),
        }
    }

    /// The call of a command buffer, and the encoder it made.
    pub(super) fn command_buffer_call(&self, call: &Call) -> Result<Option<Object>, PlayError> {
        let Some(Object::CommandBuffer(command_buffer)) = self.objects.get(&call.receiver) else {
            unreachable!("a command buffer's call");
        };
        let made = match &*call.selector {
            "computeCommandEncoder" => command_buffer
                .compute_command_encoder()
                .map(Object::Compute),
            "computeCommandEncoderWithDispatchType:" => {
                let dispatch: DispatchType = raw(call, 0)?;
                (command_buffer.compute_command_encoder_with_dispatch_type(dispatch))
                    .map(Object::Compute)
            }
            "computeCommandEncoderWithDescriptor:" => {
                let f = Fields::arg(call, 0, &["dispatchType"])?;
                match f.metal::<DispatchType>("dispatchType")? {
                    None => command_buffer.compute_command_encoder(),
                    Some(dispatch) => {
                        command_buffer.compute_command_encoder_with_dispatch_type(dispatch)
                    }
                }
                .map(Object::Compute)
            }
            "blitCommandEncoder" => (command_buffer.blit_command_encoder())
                .map(|encoder| Object::Blit(Box::new(encoder))),
            "renderCommandEncoderWithDescriptor:" => {
                let id = Self::id(call, 0)?;
                let Object::Pass(pass) = self.get(call, id)? else {
                    return Err(refuse(call, "argument 0 is no render pass descriptor"));
                };
                (command_buffer.render_command_encoder(pass))
                    .map(|encoder| Object::Render(Box::new(encoder)))
            }
            "commit" => return command_buffer.commit().map(|()| None).map_err(failed(call)),
            "encodeSignalEvent:value:" => {
                let event = self.event(call, 0)?;
                let value = uint(call, 1)?;
                return (command_buffer.encode_signal_event(event, value))
                    .map(|()| None)
                    .map_err(failed(call));
            }
            "encodeWaitForEvent:value:" => {
                let event = self.event(call, 0)?;
                let value = uint(call, 1)?;
                return (command_buffer.encode_wait_for_event(event, value))
                    .map(|()| None)
                    .map_err(failed(call));
            }
            _ => return Err(refuse(call, "not played on a command buffer")),
        };
        made.map(Some).map_err(failed(call))
    }

    fn compute(&mut self, id: u64) -> &mut ComputeCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Compute(encoder)) => encoder,
            _ => unreachable!("a compute encoder's call"),
        }
    }

    pub(super) fn compute_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let result = match &*call.selector {
            "endEncoding" => {
                let Some(Object::Compute(encoder)) = self.objects.remove(&id) else {
                    unreachable!("a compute encoder's call");
                };
                encoder.end_encoding()
            }
            "memoryBarrierWithScope:" => {
                let scope: BarrierScope = raw(call, 0)?;
                self.compute(id).memory_barrier_with_scope(scope);
                Ok(())
            }
            "setComputePipelineState:" => {
                let state =
                    self.kind(
                        call,
                        Self::id(call, 0)?,
                        "compute pipeline state",
                        |o| match o {
                            Object::ComputePipeline(state) => Some(state.clone()),
                            _ => None,
                        },
                    )?;
                self.compute(id).set_compute_pipeline_state(&state)
            }
            "setBuffer:offset:atIndex:" => {
                let buffer = self.optional_buffer(call, 0)?;
                let (offset, at) = (uint(call, 1)?, index(call, 2)?);
                self.compute(id).set_buffer(buffer.as_ref(), offset, at)
            }
            "setBuffers:offsets:withRange:" => {
                let buffers = list(call, 0)?.iter();
                let buffers = buffers.map(|buffer| self.optional_buffer_of(call, buffer));
                let buffers = buffers.collect::<Result<Vec<_>, _>>()?;
                let offsets = arg(call, 1)?.as_uints();
                let offsets = offsets.ok_or_else(|| refuse(call, "argument 1 is no offsets"))?;
                let at = first(call, 2, buffers.len())?;
                if offsets.len() != buffers.len() {
                    return Err(refuse(call, "as many offsets as buffers"));
                }
                let buffers: Vec<_> = buffers.iter().map(Option::as_ref).collect();
                self.compute(id).set_buffers(&buffers, &offsets, at)
            }
            "setBytes:length:atIndex:" => {
                let bytes = data(call, 0)?.to_vec();
                if bytes.len() as u64 != uint(call, 1)? {
                    return Err(refuse(call, "data of another length than its length"));
                }
                let at = index(call, 2)?;
                self.compute(id).set_bytes(&bytes, at)
            }
            "setTexture:atIndex:" => {
                let texture = self.optional_texture(call, 0)?;
                let at = index(call, 1)?;
                self.compute(id).set_texture(texture.as_ref(), at)
            }
            "setSamplerStates:withRange:" => {
                let samplers = self.samplers(call, 0)?;
                let at = first(call, 1, samplers.len())?;
                let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
                self.compute(id).set_sampler_states(&samplers, at)
            }
            "setThreadgroupMemoryLength:atIndex:" => {
                let length = index(call, 0)?;
                let at = index(call, 1)?;
                self.compute(id).set_threadgroup_memory_length(length, at)
            }
            "dispatchThreadgroups:threadsPerThreadgroup:" => {
                let groups = size(call, arg(call, 0)?)?;
                let threads = size(call, arg(call, 1)?)?;
                self.compute(id).dispatch_threadgroups(groups, threads)
            }
            "dispatchThreads:threadsPerThreadgroup:" => {
                let grid = size(call, arg(call, 0)?)?;
                let threads = size(call, arg(call, 1)?)?;
                self.compute(id).dispatch_threads(grid, threads)
            }
            "useResource:usage:" => {
                let usage: ResourceUsage = raw(call, 1)?;
                let resource = self.resource(call, 0)?;
                self.compute(id).use_resource(resource.as_ref(), usage)
            }
            "useHeap:" => {
                let heap = self.kind(call, Self::id(call, 0)?, "heap", |object| match object {
                    Object::Heap(heap) | Object::Block(heap, _) => Some(heap.clone()),
                    _ => None,
                })?;
                self.compute(id).use_heap(&heap)
            }
            "updateFence:" => {
                let fence = self.fence(call, 0)?;
                self.compute(id).update_fence(&fence)
            }
            "waitForFence:" => {
                let fence = self.fence(call, 0)?;
                self.compute(id).wait_for_fence(&fence)
            }
            _ => return Err(refuse(call, "not played on a compute encoder")),
        };
        result.map_err(failed(call))
    }

    fn blit(&mut self, id: u64) -> &mut BlitCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Blit(encoder)) => encoder,
            _ => unreachable!("a blit encoder's call"),
        }
    }

    pub(super) fn blit_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let result = match &*call.selector {
            "endEncoding" => {
                let Some(Object::Blit(encoder)) = self.objects.remove(&id) else {
                    unreachable!("a blit encoder's call");
                };
                encoder.end_encoding()
            }
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:" => {
                let (from, to) = (self.buffer(call, 0)?, self.buffer(call, 2)?);
                let (at, to_at, length) = (uint(call, 1)?, uint(call, 3)?, uint(call, 4)?);
                (self.blit(id)).copy_from_buffer_to_buffer(&from, at, &to, to_at, length)
            }
            // The form without options copies as the one with none.
            "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
             toTexture:destinationSlice:destinationLevel:destinationOrigin:options:"
            | "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
               toTexture:destinationSlice:destinationLevel:destinationOrigin:" => {
                let (from, to) = (self.buffer(call, 0)?, self.texture(call, 5)?);
                let option = match call.args.get(9) {
                    Some(_) => blit_option(call, uint(call, 9)?)?,
                    None => BlitOption::None,
                };
                self.blit(id).copy_from_buffer_to_texture(
                    &from,
                    uint(call, 1)?,
                    uint(call, 2)?,
                    uint(call, 3)?,
                    size(call, arg(call, 4)?)?,
                    &to,
                    uint(call, 6)?,
                    uint(call, 7)?,
                    origin(call, arg(call, 8)?)?,
                    option,
                )
            }
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
             destinationOffset:destinationBytesPerRow:destinationBytesPerImage:" => {
                let (from, to) = (self.texture(call, 0)?, self.buffer(call, 5)?);
                self.blit(id).copy_from_texture_to_buffer(
                    &from,
                    uint(call, 1)?,
                    uint(call, 2)?,
                    origin(call, arg(call, 3)?)?,
                    size(call, arg(call, 4)?)?,
                    &to,
                    uint(call, 6)?,
                    uint(call, 7)?,
                    uint(call, 8)?,
                    BlitOption::None,
                )
            }
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
             destinationSlice:destinationLevel:destinationOrigin:" => {
                let (from, to) = (self.texture(call, 0)?, self.texture(call, 5)?);
                self.blit(id).copy_from_texture_to_texture(
                    &from,
                    uint(call, 1)?,
                    uint(call, 2)?,
                    origin(call, arg(call, 3)?)?,
                    size(call, arg(call, 4)?)?,
                    &to,
                    uint(call, 6)?,
                    uint(call, 7)?,
                    origin(call, arg(call, 8)?)?,
                )
            }
            "generateMipmapsForTexture:" => {
                let texture = self.texture(call, 0)?;
                self.blit(id).generate_mipmaps(&texture)
            }
            "fillBuffer:range:value:" => {
                let buffer = self.buffer(call, 0)?;
                let range = range(call, arg(call, 1)?)?;
                let value = u8::try_from(uint(call, 2)?)
                    .map_err(|_| refuse(call, "a fill value past a byte"))?;
                self.blit(id).fill_buffer(&buffer, range, value)
            }
            "synchronizeResource:" => match self.resource(call, 0)? {
                // A stream that records its announced bytes: the GPU's bytes stay out of a
                // buffer's CPU copy, which keeps the stream's for the next `didModifyRange:` to
                // put into the GPU's copy. A check reads the GPU's copy either way
                // (`read_buffer`).
                Resource::Buffer(_) if self.marked => {
                    self.read_backs_left_out += 1;
                    Ok(())
                }
                resource => self.blit(id).synchronize_resource(resource.as_ref()),
            },
            "updateFence:" => {
                let fence = self.fence(call, 0)?;
                self.blit(id).update_fence(&fence)
            }
            "waitForFence:" => {
                let fence = self.fence(call, 0)?;
                self.blit(id).wait_for_fence(&fence)
            }
            "m2v.copyFromGuestRam" | "m2v.copyFromGuestRamAt" => {
                let Some(upload) = upload(call)? else {
                    unreachable!("an upload's selector");
                };
                let to = self.buffer(call, 0)?;
                for (at, entry) in upload.pieces {
                    let bytes = piece(&call.args, entry);
                    if bytes.is_empty() {
                        continue;
                    }
                    let length = bytes.len() as u64;
                    let from = own_buffer(&self.device, length, BufferContents::Copy(bytes));
                    let from = from.map_err(failed(call))?;
                    let copied = self
                        .blit(id)
                        .copy_from_buffer_to_buffer(&from, 0, &to, at, length);
                    copied.map_err(failed(call))?;
                }
                Ok(())
            }
            "m2v.copyToGuestRam" => {
                let from = self.buffer(call, 0)?;
                let runs = placed_runs(arg(call, 1)?).ok_or_else(|| {
                    PlayError::Invalid(format!(
                        "{}: argument 1 is no list of placed runs",
                        call.selector
                    ))
                })?;
                let total = runs
                    .iter()
                    .try_fold(0u64, |sum, &(_, length)| sum.checked_add(length));
                let total =
                    total.ok_or_else(|| PlayError::Invalid("runs past 2^64 bytes".into()))?;
                if total == 0 {
                    return Ok(());
                }
                let host = self.host(call, total)?;
                let blit = self.blit(id);
                let mut into = 0;
                runs.into_iter().try_for_each(|(at, length)| {
                    let copied = blit.copy_from_buffer_to_buffer(&from, at, &host, into, length);
                    into += length;
                    copied
                })
            }
            _ => return Err(refuse(call, "not played on a blit encoder")),
        };
        result.map_err(failed(call))
    }

    /// The host memory `m2v.copyToGuestRam` copies `length` bytes into.
    fn host(&mut self, call: &Call, length: u64) -> Result<Buffer, PlayError> {
        match &self.host {
            Some(host) if host.length() >= length => Ok(host.clone()),
            _ => {
                let made = own_buffer(&self.device, length, BufferContents::Zero);
                let host = made.map_err(failed(call))?;
                self.host = Some(host.clone());
                Ok(host)
            }
        }
    }

    fn render(&mut self, id: u64) -> &mut RenderCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Render(encoder)) => encoder,
            _ => unreachable!("a render encoder's call"),
        }
    }

    pub(super) fn render_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        if call.selector.starts_with("draw") && self.unpiped.contains(&id) {
            self.skips.draws += 1;
            return Ok(());
        }
        let primitive =
            |at: usize| DrawPrimitiveType::from_raw(uint(call, at)?).map_err(failed(call));
        let result = match &*call.selector {
            "endEncoding" => {
                self.unpiped.remove(&id);
                let Some(Object::Render(encoder)) = self.objects.remove(&id) else {
                    unreachable!("a render encoder's call");
                };
                encoder.end_encoding()
            }
            "setRenderPipelineState:" if self.skipped(Self::id(call, 0)?) => {
                self.unpiped.insert(id);
                self.skips.calls += 1;
                return Ok(());
            }
            "setRenderPipelineState:" => {
                self.unpiped.remove(&id);
                let state =
                    self.kind(
                        call,
                        Self::id(call, 0)?,
                        "render pipeline state",
                        |o| match o {
                            Object::RenderPipeline(state) => Some(state.clone()),
                            _ => None,
                        },
                    )?;
                self.render(id).set_render_pipeline_state(&state)
            }
            "setDepthStencilState:" => {
                let state =
                    match arg(call, 0)? {
                        Value::Nil => None,
                        _ => Some(self.kind(
                            call,
                            Self::id(call, 0)?,
                            "depth stencil state",
                            |o| match o {
                                Object::DepthStencil(state) => Some(state.clone()),
                                _ => None,
                            },
                        )?),
                    };
                self.render(id).set_depth_stencil_state(state.as_ref())
            }
            "setVertexBuffer:offset:atIndex:" | "setFragmentBuffer:offset:atIndex:" => {
                let buffer = self.optional_buffer(call, 0)?;
                let (offset, at) = (uint(call, 1)?, index(call, 2)?);
                let encoder = self.render(id);
                match &*call.selector {
                    "setVertexBuffer:offset:atIndex:" => {
                        encoder.set_vertex_buffer(buffer.as_ref(), offset, at)
                    }
                    _ => encoder.set_fragment_buffer(buffer.as_ref(), offset, at),
                }
            }
            "setVertexBufferOffset:atIndex:" => {
                let (offset, at) = (uint(call, 0)?, index(call, 1)?);
                self.render(id).set_vertex_buffer_offset(offset, at)
            }
            "setFragmentBufferOffset:atIndex:" => {
                let (offset, at) = (uint(call, 0)?, index(call, 1)?);
                self.render(id).set_fragment_buffer_offset(offset, at)
            }
            "setFragmentTexture:atIndex:" => {
                let texture = self.optional_texture(call, 0)?;
                let at = index(call, 1)?;
                self.render(id).set_fragment_texture(texture.as_ref(), at)
            }
            "setVertexBytes:length:atIndex:" | "setFragmentBytes:length:atIndex:" => {
                let bytes = data(call, 0)?;
                if bytes.len() as u64 != uint(call, 1)? {
                    return Err(refuse(call, "data of another length than its length"));
                }
                let at = index(call, 2)?;
                match &*call.selector {
                    "setVertexBytes:length:atIndex:" => self.render(id).set_vertex_bytes(bytes, at),
                    _ => self.render(id).set_fragment_bytes(bytes, at),
                }
            }
            "setVertexTexture:atIndex:" => {
                let texture = self.optional_texture(call, 0)?;
                let at = index(call, 1)?;
                self.render(id).set_vertex_texture(texture.as_ref(), at)
            }
            // A sampler of its own slot, and with LOD clamps as a range of one.
            "setVertexSamplerState:atIndex:"
            | "setFragmentSamplerState:atIndex:"
            | "setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:"
            | "setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:" => {
                let sampler = self.sampler(call, arg(call, 0)?)?;
                let clamped = call.args.len() == 4;
                let at = index(call, if clamped { 3 } else { 1 })?;
                let clamps = match clamped {
                    true => Some(([float(call, 1)?], [float(call, 2)?])),
                    false => None,
                };
                let clamps = clamps.as_ref().map(|(min, max)| (&min[..], &max[..]));
                let samplers = [sampler.as_ref()];
                let encoder = self.render(id);
                match call.selector.starts_with("setVertex") {
                    true => encoder.set_vertex_sampler_states(&samplers, clamps, at),
                    false => encoder.set_fragment_sampler_states(&samplers, clamps, at),
                }
            }
            "setVertexSamplerStates:withRange:" => {
                let samplers = self.samplers(call, 0)?;
                let at = first(call, 1, samplers.len())?;
                let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
                self.render(id)
                    .set_vertex_sampler_states(&samplers, None, at)
            }
            "setFragmentSamplerStates:withRange:" => {
                let samplers = self.samplers(call, 0)?;
                let at = first(call, 1, samplers.len())?;
                let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
                self.render(id)
                    .set_fragment_sampler_states(&samplers, None, at)
            }
            "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            | "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:" => {
                let samplers = self.samplers(call, 0)?;
                let clamps = |at: usize| -> Result<Vec<f32>, PlayError> {
                    let clamps = list(call, at)?.iter().map(|value| clamp(call, value));
                    let clamps = clamps.collect::<Result<Vec<_>, _>>()?;
                    match clamps.len() == samplers.len() {
                        true => Ok(clamps),
                        false => Err(refuse(call, "clamps of another count than the samplers")),
                    }
                };
                let (mins, maxes) = (clamps(1)?, clamps(2)?);
                let at = first(call, 3, samplers.len())?;
                let samplers: Vec<_> = samplers.iter().map(Option::as_ref).collect();
                let clamps = Some((&mins[..], &maxes[..]));
                let encoder = self.render(id);
                match call.selector.starts_with("setVertex") {
                    true => encoder.set_vertex_sampler_states(&samplers, clamps, at),
                    false => encoder.set_fragment_sampler_states(&samplers, clamps, at),
                }
            }
            "setColorStoreAction:atIndex:" => {
                let action = StoreAction::from_raw(uint(call, 0)?).map_err(failed(call))?;
                let at = index(call, 1)?;
                self.render(id).set_color_store_action(action, at)
            }
            "setDepthStoreAction:" | "setStencilStoreAction:" => {
                let action = StoreAction::from_raw(uint(call, 0)?).map_err(failed(call))?;
                match &*call.selector {
                    "setDepthStoreAction:" => self.render(id).set_depth_store_action(action),
                    _ => self.render(id).set_stencil_store_action(action),
                }
            }
            "setViewport:" => {
                let viewport = viewport(call, arg(call, 0)?)?;
                self.render(id).set_viewport(viewport)
            }
            "setViewports:count:" => {
                let viewports = (list(call, 0)?.iter())
                    .map(|value| viewport(call, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.render(id).set_viewports(&viewports)
            }
            "setScissorRect:" => {
                let rect = scissor(call, arg(call, 0)?)?;
                self.render(id).set_scissor_rect(rect)
            }
            "setScissorRects:count:" => {
                let rects = (list(call, 0)?.iter())
                    .map(|value| scissor(call, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.render(id).set_scissor_rects(&rects)
            }
            "setCullMode:" => {
                self.render(id).set_cull_mode(raw(call, 0)?);
                Ok(())
            }
            "setFrontFacingWinding:" => {
                self.render(id).set_front_facing_winding(raw(call, 0)?);
                Ok(())
            }
            "setTriangleFillMode:" => self.render(id).set_triangle_fill_mode(raw(call, 0)?),
            "setDepthClipMode:" | "setDepthClipModeSPI:" => {
                self.render(id).set_depth_clip_mode(raw(call, 0)?)
            }
            "setDepthBias:slopeScale:clamp:" => {
                let [bias, slope, clamp] = [float(call, 0)?, float(call, 1)?, float(call, 2)?];
                self.render(id).set_depth_bias(bias, slope, clamp)
            }
            "setBlendColorRed:green:blue:alpha:" => {
                let [red, green, blue] = [float(call, 0)?, float(call, 1)?, float(call, 2)?];
                let alpha = float(call, 3)?;
                self.render(id).set_blend_color(red, green, blue, alpha);
                Ok(())
            }
            "setStencilFrontReferenceValue:backReferenceValue:" => {
                let (front, back) = (index(call, 0)?, index(call, 1)?);
                self.render(id).set_stencil_reference_values(front, back);
                Ok(())
            }
            "setStencilReferenceValue:" => {
                let value = index(call, 0)?;
                self.render(id).set_stencil_reference_value(value);
                Ok(())
            }
            "drawPrimitives:vertexStart:vertexCount:"
            | "drawPrimitives:vertexStart:vertexCount:instanceCount:"
            | "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:" => {
                let kind = primitive(0)?;
                let (start, count) = (index(call, 1)?, index(call, 2)?);
                let instances = match call.args.len() {
                    3 => 1,
                    _ => index(call, 3)?,
                };
                let base_instance = match call.args.len() {
                    5 => index(call, 4)?,
                    _ => 0,
                };
                self.render(id)
                    .draw_primitives(kind, start, count, instances, base_instance)
            }
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
            | "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
               instanceCount:"
            | "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
               instanceCount:baseVertex:baseInstance:" => {
                let kind = primitive(0)?;
                let count = index(call, 1)?;
                let index_type: IndexType = raw(call, 2)?;
                let buffer = self.buffer(call, 3)?;
                let offset = uint(call, 4)?;
                let instances = match call.args.len() {
                    5 => 1,
                    _ => index(call, 5)?,
                };
                let (base_vertex, base_instance) = match call.args.len() {
                    8 => {
                        // `NSInteger`: signed where the recorder saw it negative.
                        let base = match arg(call, 6)? {
                            Value::Sint(base) => i32::try_from(*base).ok(),
                            Value::Uint(base) => i32::try_from(*base).ok(),
                            _ => None,
                        };
                        let base = base.ok_or_else(|| refuse(call, "argument 6 is no i32"))?;
                        (base, index(call, 7)?)
                    }
                    _ => (0, 0),
                };
                self.render(id).draw_indexed_primitives(
                    kind,
                    count,
                    index_type,
                    &buffer,
                    offset,
                    instances,
                    base_vertex,
                    base_instance,
                )
            }
            "updateFence:afterStages:" => {
                let fence = self.fence(call, 0)?;
                let stages: RenderStages = raw(call, 1)?;
                self.render(id).update_fence(&fence, stages)
            }
            "waitForFence:beforeStages:" => {
                let fence = self.fence(call, 0)?;
                let stages: RenderStages = raw(call, 1)?;
                self.render(id).wait_for_fence(&fence, stages)
            }
            "memoryBarrierWithScope:afterStages:beforeStages:" => {
                let scope: BarrierScope = raw(call, 0)?;
                let (after, before): (RenderStages, RenderStages) = (raw(call, 1)?, raw(call, 2)?);
                self.render(id)
                    .memory_barrier_with_scope(scope, after, before);
                Ok(())
            }
            _ => return Err(refuse(call, "not played on a render encoder")),
        };
        result.map_err(failed(call))
    }
}
