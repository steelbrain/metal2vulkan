//! The calls of command buffers and of their compute and blit encoders (a render encoder's are
//! `render`'s). A parallel render encoder is skipped whole, each of its calls counted; a dispatch
//! whose bindings hold a skipped object, and a blit command the runtime does not build or that
//! names a skipped object, are skipped and counted.

use super::super::args::{arg, data, index, list, range, refuse, uint, variant, Fields};
use super::{failed, granule, Held, NewRuntime, Object};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_runtime as rt;
use std::collections::HashSet;

/// A compute encoder, its buffer and texture slots that hold a skipped object, and whether its
/// pipeline state is a skipped one.
pub(super) struct Compute {
    encoder: rt::ComputeCommandEncoder,
    skipped: HashSet<(bool, u32)>,
    skipped_pipeline: bool,
}

/// `[x, y, z]` of a map of `keys`.
fn triple(call: &Call, value: &Value, keys: [&str; 3]) -> Result<[u32; 3], PlayError> {
    let f = Fields::of(call, value, &keys)?;
    let mut out = [0; 3];
    for (key, at) in keys.into_iter().zip(&mut out) {
        *at = f.required(key, Fields::count)?;
    }
    Ok(out)
}

/// `MTLSize`: `{width, height, depth}`.
fn size(call: &Call, value: &Value) -> Result<[u32; 3], PlayError> {
    triple(call, value, ["width", "height", "depth"])
}

/// `MTLOrigin`: `{x, y, z}`.
fn origin(call: &Call, value: &Value) -> Result<[u32; 3], PlayError> {
    triple(call, value, ["x", "y", "z"])
}

/// `MTLRegion`: `{origin, size}`.
pub(super) fn region(call: &Call, value: &Value) -> Result<rt::Region, PlayError> {
    let f = Fields::of(call, value, &["origin", "size"])?;
    let part = |key: &str| f.get(key).ok_or_else(|| refuse(call, format!("no {key}")));
    Ok(rt::Region {
        origin: origin(call, part("origin")?)?,
        size: size(call, part("size")?)?,
    })
}

/// The first index of a `withRange:` call's range, which covers each of its `count` objects.
pub(super) fn first(call: &Call, at: usize, count: usize) -> Result<u32, PlayError> {
    let span = range(call, arg(call, at)?)?;
    if span.end - span.start != count as u64 {
        return Err(refuse(
            call,
            format!("a range of {} for {count}", span.end - span.start),
        ));
    }
    u32::try_from(span.start).map_err(|_| refuse(call, "the range's location past u32"))
}

/// A place in a texture: its slice, level and origin, arguments `at` to `at + 2`.
fn texture_place(call: &Call, at: usize) -> Result<rt::TexturePlace, PlayError> {
    Ok(rt::TexturePlace {
        slice: index(call, at)?,
        level: index(call, at + 1)?,
        origin: origin(call, arg(call, at + 2)?)?,
    })
}

/// A region's layout in a buffer: its offset and strides, arguments `at` to `at + 2`.
fn buffer_layout(call: &Call, at: usize) -> Result<rt::BufferLayout, PlayError> {
    Ok(rt::BufferLayout {
        offset: uint(call, at)?,
        bytes_per_row: uint(call, at + 1)?,
        bytes_per_image: uint(call, at + 2)?,
    })
}

impl NewRuntime {
    /// An event argument: an event's, or a shared event's.
    fn event(&self, call: &Call, at: usize) -> Result<rt::Event, PlayError> {
        match self.get(call, Self::id(call, at)?)? {
            Object::Event(event) => Ok(event.clone()),
            Object::SharedEvent(event) => Ok(event.as_event().clone()),
            _ => Err(refuse(call, format!("argument {at} is no event"))),
        }
    }

    /// The buffer `value` names, or none where it is nil; skipped where it is a skipped object.
    fn optional_buffer(&self, call: &Call, value: &Value) -> Result<Option<rt::Buffer>, PlayError> {
        match value {
            Value::Nil => Ok(None),
            Value::Object(id) => self.buffer_of(call, *id).map(Some),
            _ => Err(refuse(call, format!("{value} is no buffer"))),
        }
    }

    /// The ranges of second copies named since the last commit (`second`), copied from the CPU's
    /// copies into the GPU's in a command buffer of the backend's own on the queue of `call`'s
    /// command buffer, committed just before it, so that it and every command buffer the queue
    /// runs after it read them.
    fn copy_named(&mut self, call: &Call) -> Result<(), PlayError> {
        if !self.seconds.named() {
            return Ok(());
        }
        let queue = self.queue_for(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        let (named, places) = self.seconds.take();
        for named in named {
            for (start, end) in named.ranges {
                let size = end - start;
                (blit.copy_from_buffer_to_buffer(&named.cpu, start, &named.gpu, start, size))
                    .map_err(failed(call))?;
            }
        }
        blit.end_encoding();
        self.commit(&queue, command_buffer, 0)
            .map_err(failed(call))?;
        for place in places {
            self.places.give(place, self.commits, Vec::new());
        }
        Ok(())
    }

    pub(super) fn command_buffer_call(
        &mut self,
        call: &Call,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let Some(Object::CommandBuffer(command_buffer)) = self.objects.get(&call.receiver) else {
            unreachable!("a command buffer's call");
        };
        let command_buffer = command_buffer.clone();
        let encoder = |dispatch: u64| {
            let dispatch = match dispatch {
                0 => rt::DispatchType::Serial,
                1 => rt::DispatchType::Concurrent,
                other => return Err(refuse(call, format!("dispatch type {other}"))),
            };
            let encoder = command_buffer.compute_command_encoder(dispatch);
            let encoder = encoder.map_err(failed(call))?;
            let compute = Compute {
                encoder,
                skipped: HashSet::new(),
                skipped_pipeline: false,
            };
            super::unaddressed(Object::Compute(Box::new(compute)))
        };
        if call.result != 0 && call.selector.contains("CommandEncoder") {
            self.owners.insert(call.result, call.receiver);
        }
        match &*call.selector {
            "computeCommandEncoder" => encoder(0),
            "computeCommandEncoderWithDispatchType:" => encoder(uint(call, 0)?),
            "computeCommandEncoderWithDescriptor:" => {
                let f = Fields::arg(call, 0, &["dispatchType"])?;
                encoder(f.uint("dispatchType")?.unwrap_or(0))
            }
            "blitCommandEncoder" => {
                let encoder = command_buffer.blit_command_encoder();
                let encoder = encoder.map_err(failed(call))?;
                super::unaddressed(Object::Blit(Box::new(encoder)))
            }
            "renderCommandEncoderWithDescriptor:" => self.render_encoder(call, &command_buffer),
            "parallelRenderCommandEncoderWithDescriptor:" => {
                self.skip_encoder("ParallelRenderCommandEncoder".to_owned())
            }
            "commit" => {
                self.surface_fill_owed(call)?;
                self.copy_named(call)?;
                self.guest_ram_fill(call)?;
                self.tiled_take(call)?;
                let queue = self.queue_for(call)?;
                self.commit(&queue, command_buffer, call.receiver)
                    .map_err(failed(call))?;
                self.guest_ram_download(call)?;
                // Its first use committed, a no-copy Managed buffer's GPU copy changes only by
                // `didModifyRange:` from here.
                self.fresh.retain(|_, user| *user != Some(call.receiver));
                self.unwritten.committed(call.receiver);
                self.surfaces.committed(call.receiver);
                self.indices.committed(call.receiver, self.commits);
                if self.users.remove(&call.receiver).is_some() {
                    self.places.ended(call.receiver, self.commits);
                }
                Ok(None)
            }
            "enqueue" => command_buffer
                .enqueue()
                .map(|()| None)
                .map_err(failed(call)),
            "encodeSignalEvent:value:" => {
                let event = self.event(call, 0)?;
                (command_buffer.encode_signal_event(&event, uint(call, 1)?))
                    .map(|()| None)
                    .map_err(failed(call))
            }
            "encodeWaitForEvent:value:" => {
                let event = self.event(call, 0)?;
                (command_buffer.encode_wait_for_event(&event, uint(call, 1)?))
                    .map(|()| None)
                    .map_err(failed(call))
            }
            _ => Err(refuse(call, "not played on a command buffer")),
        }
    }

    fn compute(&mut self, id: u64) -> &mut Compute {
        match self.objects.get_mut(&id) {
            Some(Object::Compute(compute)) => compute,
            _ => unreachable!("a compute encoder's call"),
        }
    }

    /// Binds the texture `value` names at `slot` of the compute encoder `call` is sent to: none
    /// where it is nil; nothing where it is a skipped object, the slot marked skipped.
    fn bind_compute_texture(
        &mut self,
        call: &Call,
        value: &Value,
        slot: u32,
    ) -> Result<Result<(), rt::Error>, PlayError> {
        let texture = match value {
            Value::Nil => None,
            Value::Object(object) if self.skipped(*object) => {
                self.compute(call.receiver).skipped.insert((true, slot));
                return Ok(Ok(()));
            }
            Value::Object(object) => Some(self.texture_of(call, *object)?),
            other => return Err(refuse(call, format!("{other} is no texture"))),
        };
        let compute = self.compute(call.receiver);
        compute.skipped.remove(&(true, slot));
        Ok(compute.encoder.set_texture(texture.as_ref(), slot))
    }

    pub(super) fn compute_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let result = match &*call.selector {
            "endEncoding" => {
                let Some(Object::Compute(compute)) = self.objects.remove(&id) else {
                    unreachable!("a compute encoder's call");
                };
                let compute = *compute;
                compute.encoder.end_encoding();
                Ok(())
            }
            "memoryBarrierWithScope:" => {
                self.compute(id).encoder.memory_barrier();
                Ok(())
            }
            "setComputePipelineState:" => {
                let state = match self.get(call, Self::id(call, 0)?)? {
                    Object::ComputePipeline(state) => Some(state.clone()),
                    Object::Skipped => None,
                    _ => return Err(refuse(call, "argument 0 is no compute pipeline state")),
                };
                (self.unwritten).state(id, state.as_ref().map(|s| s.buffer_accesses()));
                let compute = self.compute(id);
                compute.skipped_pipeline = state.is_none();
                match state {
                    Some(state) => compute.encoder.set_compute_pipeline_state(&state),
                    None => Ok(()),
                }
            }
            "setBuffer:offset:atIndex:" => {
                let buffer = self.optional_buffer(call, arg(call, 0)?)?;
                let (offset, at) = (uint(call, 1)?, index(call, 2)?);
                let compute = self.compute(id);
                compute.skipped.remove(&(false, at));
                compute.encoder.set_buffer(buffer.as_ref(), offset, at)
            }
            "setBufferOffset:atIndex:" => {
                let (offset, at) = (uint(call, 0)?, index(call, 1)?);
                self.compute(id).encoder.set_buffer_offset(offset, at)
            }
            "setBuffers:offsets:withRange:" => {
                let buffers = list(call, 0)?.iter();
                let buffers = buffers.map(|buffer| self.optional_buffer(call, buffer));
                let buffers = buffers.collect::<Result<Vec<_>, _>>()?;
                let offsets = arg(call, 1)?.as_uints();
                let offsets = offsets.ok_or_else(|| refuse(call, "argument 1 is no offsets"))?;
                if offsets.len() != buffers.len() {
                    return Err(refuse(call, "as many offsets as buffers"));
                }
                let at = first(call, 2, buffers.len())?;
                let compute = self.compute(id);
                let mut result = Ok(());
                for (slot, (buffer, offset)) in (at..).zip(buffers.iter().zip(offsets)) {
                    compute.skipped.remove(&(false, slot));
                    result = result.and(compute.encoder.set_buffer(buffer.as_ref(), offset, slot));
                }
                result
            }
            "setBytes:length:atIndex:" => {
                let bytes = data(call, 0)?.to_vec();
                if bytes.len() as u64 != uint(call, 1)? {
                    return Err(refuse(call, "data of another length than its length"));
                }
                let at = index(call, 2)?;
                let compute = self.compute(id);
                compute.skipped.remove(&(false, at));
                compute.encoder.set_bytes(&bytes, at)
            }
            "setTexture:atIndex:" => {
                let at = index(call, 1)?;
                self.bind_compute_texture(call, arg(call, 0)?, at)?
            }
            "setTextures:withRange:" => {
                let textures = list(call, 0)?;
                let at = first(call, 1, textures.len())?;
                let mut result = Ok(());
                for (slot, texture) in (at..).zip(textures) {
                    result = result.and(self.bind_compute_texture(call, texture, slot)?);
                }
                result
            }
            "setSamplerStates:withRange:" => {
                let samplers = list(call, 0)?.iter().map(|value| match value {
                    Value::Nil => Ok(None),
                    Value::Object(object) => match self.get(call, *object)? {
                        Object::Sampler(sampler) => Ok(Some(sampler.clone())),
                        _ => Err(refuse(call, format!("#{object} is no sampler state"))),
                    },
                    other => Err(refuse(call, format!("{other} is no sampler state"))),
                });
                let samplers = samplers.collect::<Result<Vec<_>, _>>()?;
                let at = first(call, 1, samplers.len())?;
                let compute = self.compute(id);
                let mut result = Ok(());
                for (slot, sampler) in (at..).zip(&samplers) {
                    result = result.and(compute.encoder.set_sampler_state(sampler.as_ref(), slot));
                }
                result
            }
            "setThreadgroupMemoryLength:atIndex:" => {
                let (length, at) = (index(call, 0)?, index(call, 1)?);
                self.compute(id)
                    .encoder
                    .set_threadgroup_memory_length(length, at)
            }
            "setImageblockWidth:height:" => {
                let (width, height) = (index(call, 0)?, index(call, 1)?);
                self.compute(id).encoder.set_imageblock_size(width, height)
            }
            "setStageInRegion:" => {
                let at = region(call, arg(call, 0)?)?;
                self.compute(id).encoder.set_stage_in_region(at);
                Ok(())
            }
            "dispatchThreadgroups:threadsPerThreadgroup:"
            | "dispatchThreads:threadsPerThreadgroup:" => {
                let grid = size(call, arg(call, 0)?)?;
                let threads = size(call, arg(call, 1)?)?;
                let compute = self.compute(id);
                if compute.skipped_pipeline || !compute.skipped.is_empty() {
                    self.skips.dispatches += 1;
                    return Ok(());
                }
                let encoder = &mut self.compute(id).encoder;
                match &*call.selector {
                    "dispatchThreads:threadsPerThreadgroup:" => {
                        encoder.dispatch_threads(grid, threads)
                    }
                    _ => encoder.dispatch_threadgroups(grid, threads),
                }
            }
            "useResource:usage:" => {
                let usage = rt::ResourceUsage::READ;
                let usage = match uint(call, 1)? {
                    1 => usage,
                    2 => rt::ResourceUsage::WRITE,
                    3 => usage | rt::ResourceUsage::WRITE,
                    raw => return Err(refuse(call, format!("resource usage {raw}"))),
                };
                match self.resource(call, 0)? {
                    Held::Buffer(buffer) => (self.compute(id).encoder)
                        .use_resource(rt::Resource::Buffer(&buffer), usage),
                    Held::Texture(texture) => (self.compute(id).encoder)
                        .use_resource(rt::Resource::Texture(&texture), usage),
                    Held::Skipped => {
                        self.skips.calls += 1;
                        Ok(())
                    }
                }
            }
            "updateFence:" => {
                let fence = self.fence_of(call, 0)?;
                self.compute(id).encoder.update_fence(&fence)
            }
            "waitForFence:" => {
                let fence = self.fence_of(call, 0)?;
                self.compute(id).encoder.wait_for_fence(&fence)
            }
            "useHeap:" => match self.heap_of(call, 0)? {
                Some(heap) => self.compute(id).encoder.use_heap(heap),
                None => {
                    self.skips.calls += 1;
                    Ok(())
                }
            },
            _ => return Err(refuse(call, "not played on a compute encoder")),
        };
        result.map_err(failed(call))
    }

    /// Buffer `object`'s aside (`aside`): `length` bytes in a Shared place of the play's own,
    /// zeroed, made at its first read-back.
    fn aside(&mut self, call: &Call, object: u64, length: u64) -> Result<rt::Buffer, PlayError> {
        let completed = self.completed();
        let (places, device) = (&mut self.places, &self.device);
        self.asides.read_back(object, length, |length| {
            // The play's own aside: no recorded slack.
            let place = places.take(rt::MemoryClass::Shared, length, granule(None), completed);
            let place = place.map_err(failed(call))?;
            let buffer = device.new_buffer(place.block, place.range.start, length, Some(0));
            Ok((buffer.map_err(failed(call))?, place))
        })
    }

    pub(super) fn blit(&mut self, id: u64) -> &mut rt::BlitCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Blit(encoder)) => encoder,
            _ => unreachable!("a blit encoder's call"),
        }
    }

    pub(super) fn blit_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        if &*call.selector != "endEncoding" && self.skipped_named(call).is_some() {
            self.skips.blits += 1;
            return Ok(());
        }
        let result = match &*call.selector {
            "endEncoding" => {
                let Some(Object::Blit(encoder)) = self.objects.remove(&id) else {
                    unreachable!("a blit encoder's call");
                };
                self.owners.remove(&id);
                (*encoder).end_encoding();
                Ok(())
            }
            "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
             toTexture:destinationSlice:destinationLevel:destinationOrigin:"
            | "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
             toTexture:destinationSlice:destinationLevel:destinationOrigin:options:" => {
                let (from, to) = (
                    self.buffer_of(call, Self::id(call, 0)?)?,
                    self.texture_of(call, Self::id(call, 5)?)?,
                );
                if call.args.get(9).is_some() && uint(call, 9)? != 0 {
                    return Err(refuse(call, "a blit option"));
                }
                let (layout, region) = (buffer_layout(call, 1)?, size(call, arg(call, 4)?)?);
                let place = texture_place(call, 6)?;
                self.blit(id)
                    .copy_from_buffer(&from, layout, region, &to, place)
            }
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
             destinationOffset:destinationBytesPerRow:destinationBytesPerImage:" => {
                let (from, to) = (
                    self.texture_of(call, Self::id(call, 0)?)?,
                    self.buffer_of(call, Self::id(call, 5)?)?,
                );
                let (place, region) = (texture_place(call, 1)?, size(call, arg(call, 4)?)?);
                let layout = buffer_layout(call, 6)?;
                self.blit(id)
                    .copy_from_texture_to_buffer(&from, place, region, &to, layout)
            }
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
             destinationSlice:destinationLevel:destinationOrigin:" => {
                let (from, to) = (
                    self.texture_of(call, Self::id(call, 0)?)?,
                    self.texture_of(call, Self::id(call, 5)?)?,
                );
                let (place, region) = (texture_place(call, 1)?, size(call, arg(call, 4)?)?);
                let to_place = texture_place(call, 6)?;
                self.blit(id)
                    .copy_from_texture(&from, place, region, &to, to_place)
            }
            // A skipped resource is counted above, with the blits that name a skipped object.
            "synchronizeResource:" => {
                let object = variant(call, arg(call, 0)?)?.1.as_object();
                match self.resource(call, 0)? {
                    // A buffer with two copies of a stream that records its announced bytes:
                    // the GPU's copy into the buffer's aside, its whole words (`aside`).
                    Held::Buffer(buffer) if object.is_some_and(|o| self.asides.takes(o)) => {
                        let object = object.unwrap_or_default();
                        let aside = self.aside(call, object, buffer.length())?;
                        let words = buffer.length() / 4 * 4;
                        self.blit(id)
                            .copy_from_buffer_to_buffer(&buffer, 0, &aside, 0, words)
                    }
                    // A buffer with a second copy: the GPU's copy into the CPU's, its whole
                    // words.
                    Held::Buffer(buffer) => {
                        let second = object.and_then(|object| self.seconds.cpu(object));
                        match second.cloned() {
                            Some(cpu) => {
                                let words = buffer.length() / 4 * 4;
                                self.blit(id)
                                    .copy_from_buffer_to_buffer(&buffer, 0, &cpu, 0, words)
                            }
                            None => self
                                .blit(id)
                                .synchronize_resource(rt::Resource::Buffer(&buffer)),
                        }
                    }
                    Held::Texture(texture) => self
                        .blit(id)
                        .synchronize_resource(rt::Resource::Texture(&texture)),
                    Held::Skipped => unreachable!("a blit naming a skipped object is skipped"),
                }
            }
            "updateFence:" => {
                let fence = self.fence_of(call, 0)?;
                self.blit(id).update_fence(&fence)
            }
            "waitForFence:" => {
                let fence = self.fence_of(call, 0)?;
                self.blit(id).wait_for_fence(&fence)
            }
            "fillBuffer:range:value:" => {
                let buffer = self.buffer_of(call, Self::id(call, 0)?)?;
                let span = range(call, arg(call, 1)?)?;
                let value = u8::try_from(uint(call, 2)?)
                    .map_err(|_| refuse(call, "a fill value past a byte"))?;
                self.blit(id).fill_buffer(&buffer, span, value)
            }
            // A layout hint: the contents stay as they are.
            "optimizeContentsForGPUAccess:" => Ok(()),
            // Not built yet: each skipped and counted by its kind.
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:" => {
                let (from, to) = (
                    self.buffer_of(call, Self::id(call, 0)?)?,
                    self.buffer_of(call, Self::id(call, 2)?)?,
                );
                let (from_offset, to_offset) = (uint(call, 1)?, uint(call, 3)?);
                self.blit(id).copy_from_buffer_to_buffer(
                    &from,
                    from_offset,
                    &to,
                    to_offset,
                    uint(call, 4)?,
                )
            }
            "m2v.copyFromGuestRam" | "m2v.copyFromGuestRamAt" => return self.guest_ram_copy(call),
            "m2v.copyToGuestRam" => return self.copy_back(call),
            "synchronizeTexture:slice:level:" => {
                self.skips.texture_syncs += 1;
                Ok(())
            }
            "generateMipmapsForTexture:" => {
                let texture = self.texture_of(call, Self::id(call, 0)?)?;
                self.blit(id).generate_mipmaps(&texture)
            }
            _ => return Err(refuse(call, "not played on a blit encoder")),
        };
        result.map_err(failed(call))
    }
}
