//! The player's backend over the old runtime (`metal2vulkan-runtime`), through its public API on
//! one device. A block is a placement heap with, where its memory is the CPU's too, a buffer over
//! all of it that writes go through; a descriptor is an object its setter records change; a
//! drawable the stream never saw made is a texture of its descriptor. A selector, an argument or
//! a descriptor key this module does not name is refused, naming it.

mod args;
mod descriptors;
mod encoders;

use super::player::{Backend, Counters, Made, PlayError};
use super::stream::Call;
use super::value::Value;
use crate::gpu::wait::{runtime::wait_for_command_buffer, SubmissionBackend};
use crate::gpu::DeviceType;
use args::{arg, data, failed, metal, options, range, refuse, region, uint, Fields};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder,
    ComputePipelineState, DepthStencilState, Device, Error, Event, EventRef, Fence, Function, Heap,
    HeapDescriptor, HeapType, IdPlacement, Library, Origin, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceId, ResourceIdKind, ResourceOptions,
    SamplerState, SharedEvent, Size, StorageMode, Texture, TextureDescriptor, TextureLevelQuery,
};
use std::collections::HashMap;

/// One object of the stream, as the runtime holds it.
enum Object {
    Device,
    Queue(CommandQueue),
    CommandBuffer(CommandBuffer),
    Compute(ComputeCommandEncoder),
    /// Boxed: a blit encoder is some 3 KiB, twenty times most other objects.
    Blit(Box<BlitCommandEncoder>),
    Render(Box<RenderCommandEncoder>),
    Buffer(Buffer),
    /// A block: its heap, and the buffer over all of it where the CPU writes it.
    Block(Heap, Option<Buffer>),
    Heap(Heap),
    Texture(Texture),
    TextureDescriptor(Box<TextureDescriptor>),
    Pass(Box<RenderPassDescriptor>),
    Library(Library),
    Function(Function),
    ComputePipeline(ComputePipelineState),
    RenderPipeline(RenderPipelineState),
    DepthStencil(DepthStencilState),
    Sampler(SamplerState),
    Fence(Fence),
    Event(Event),
    SharedEvent(SharedEvent),
}

/// The old runtime on one device.
pub struct OldRuntime {
    device: Device,
    /// The kind of `device`, which decides what a wait that never ends records.
    backend: SubmissionBackend,
    objects: HashMap<u64, Object>,
    /// The queue of the player's readbacks.
    readback: Option<CommandQueue>,
    /// What `m2v.copyToGuestRam` copies into: the host's memory, which nothing in a replay reads.
    host: Option<Buffer>,
}

/// A made object with no address.
fn unaddressed(object: Object) -> (Object, Made) {
    (object, Made::default())
}

/// A made buffer, at its address.
fn addressed(buffer: Buffer) -> (Object, Made) {
    let made = Made {
        address: Some(buffer.gpu_address().0),
    };
    (Object::Buffer(buffer), made)
}

/// The placement of a resource index, `[index, forced]`: Metal places the object at the index
/// whether or not it is forced.
fn placement(call: &Call, index: &[Value]) -> Result<IdPlacement, PlayError> {
    match index {
        [index, Value::Bool(_)] => match index.as_uint() {
            Some(index) => Ok(IdPlacement::At(ResourceId(index))),
            None => Err(refuse(call, format!("resource index {index}"))),
        },
        _ => Err(refuse(call, "a resource index that is no [index, forced]")),
    }
}

impl OldRuntime {
    /// The backend on `device`, of `device_type`: each wait for a command buffer is bounded
    /// and follows the hang protocol (`gpu::wait`).
    pub fn new(device: Device, device_type: DeviceType) -> Self {
        OldRuntime {
            device,
            backend: SubmissionBackend::vulkan(device_type),
            objects: HashMap::new(),
            readback: None,
            host: None,
        }
    }

    fn get(&self, call: &Call, id: u64) -> Result<&Object, PlayError> {
        self.objects.get(&id).ok_or_else(|| {
            PlayError::Invalid(format!("{}: #{id} is no live object", call.selector))
        })
    }

    /// Object `id` as `pick` takes it, refused where it is no `what`.
    fn kind<T>(
        &self,
        call: &Call,
        id: u64,
        what: &str,
        pick: impl Fn(&Object) -> Option<T>,
    ) -> Result<T, PlayError> {
        pick(self.get(call, id)?).ok_or_else(|| refuse(call, format!("#{id} is no {what}")))
    }

    /// The object argument `at` names.
    fn id(call: &Call, at: usize) -> Result<u64, PlayError> {
        let value = arg(call, at)?;
        value
            .as_object()
            .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not an object")))
    }

    fn buffer_of(&self, call: &Call, id: u64) -> Result<Buffer, PlayError> {
        self.kind(call, id, "buffer", |object| match object {
            Object::Buffer(buffer) => Some(buffer.clone()),
            _ => None,
        })
    }

    fn buffer(&self, call: &Call, at: usize) -> Result<Buffer, PlayError> {
        self.buffer_of(call, Self::id(call, at)?)
    }

    /// The buffer `value` names, or none where it is nil.
    fn optional_buffer_of(&self, call: &Call, value: &Value) -> Result<Option<Buffer>, PlayError> {
        match value {
            Value::Nil => Ok(None),
            Value::Object(id) => self.buffer_of(call, *id).map(Some),
            _ => Err(refuse(call, format!("{value} is no buffer"))),
        }
    }

    fn optional_buffer(&self, call: &Call, at: usize) -> Result<Option<Buffer>, PlayError> {
        self.optional_buffer_of(call, arg(call, at)?)
    }

    fn texture_of(&self, call: &Call, id: u64) -> Result<Texture, PlayError> {
        self.kind(call, id, "texture", |object| match object {
            Object::Texture(texture) => Some(texture.clone()),
            _ => None,
        })
    }

    fn texture(&self, call: &Call, at: usize) -> Result<Texture, PlayError> {
        self.texture_of(call, Self::id(call, at)?)
    }

    fn optional_texture(&self, call: &Call, at: usize) -> Result<Option<Texture>, PlayError> {
        match arg(call, at)? {
            Value::Nil => Ok(None),
            _ => self.texture(call, at).map(Some),
        }
    }

    fn function_of(&self, call: &Call, id: u64) -> Result<Function, PlayError> {
        self.kind(call, id, "function", |object| match object {
            Object::Function(function) => Some(function.clone()),
            _ => None,
        })
    }

    fn fence(&self, call: &Call, at: usize) -> Result<Fence, PlayError> {
        self.kind(call, Self::id(call, at)?, "fence", |object| match object {
            Object::Fence(fence) => Some(fence.clone()),
            _ => None,
        })
    }

    /// The samplers of a list argument, nil for none.
    fn samplers(&self, call: &Call, at: usize) -> Result<Vec<Option<SamplerState>>, PlayError> {
        let sampler = |value: &Value| match value {
            Value::Nil => Ok(None),
            Value::Object(id) => self
                .kind(call, *id, "sampler state", |object| match object {
                    Object::Sampler(sampler) => Some(sampler.clone()),
                    _ => None,
                })
                .map(Some),
            _ => Err(refuse(call, format!("{value} is no sampler state"))),
        };
        args::list(call, at)?.iter().map(sampler).collect()
    }

    fn event<'s>(&'s self, call: &Call, at: usize) -> Result<EventRef<'s>, PlayError> {
        match self.get(call, Self::id(call, at)?)? {
            Object::Event(event) => Ok(EventRef::Event(event)),
            Object::SharedEvent(event) => Ok(EventRef::Shared(event)),
            _ => Err(refuse(call, format!("argument {at} is no event"))),
        }
    }

    /// The texture descriptor object argument `at` names, as it is now.
    fn texture_descriptor(&self, call: &Call, at: usize) -> Result<TextureDescriptor, PlayError> {
        let id = Self::id(call, at)?;
        self.kind(call, id, "texture descriptor", |object| match object {
            Object::TextureDescriptor(descriptor) => Some(**descriptor),
            _ => None,
        })
    }

    /// A call of the platform (receiver 0) or the device, and what it made.
    fn device_call(&self, call: &Call) -> Result<Option<(Object, Made)>, PlayError> {
        let device = &self.device;
        Ok(Some(match &*call.selector {
            "MTLCreateSystemDefaultDevice" => unaddressed(Object::Device),
            "newCommandQueue" => {
                let queue = device.new_command_queue().map_err(failed(call))?;
                unaddressed(Object::Queue(queue))
            }
            "newBufferWithLength:options:" => {
                let options = options(call, uint(call, 1)?)?;
                let buffer = device.new_buffer(uint(call, 0)?, options);
                addressed(buffer.map_err(failed(call))?)
            }
            "newBufferWithBytes:length:options:" => {
                let bytes = data(call, 0)?;
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| bytes.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let options = options(call, uint(call, 2)?)?;
                let buffer = device.new_buffer_with_bytes(bytes, options);
                addressed(buffer.map_err(failed(call))?)
            }
            "newBufferWithDescriptor:" => {
                let (length, options) = descriptors::buffer(call)?;
                addressed(device.new_buffer(length, options).map_err(failed(call))?)
            }
            "m2v.newBlockWithLength:storageMode:" => {
                let length = uint(call, 0)?;
                let mode = StorageMode::from_raw(uint(call, 1)?).map_err(failed(call))?;
                let options = options(call, mode.raw() << 4)?;
                let descriptor = HeapDescriptor::new(HeapType::Placement, length, options);
                let heap = device.new_heap(&descriptor).map_err(failed(call))?;
                let whole = match mode {
                    StorageMode::Shared | StorageMode::Managed => Some(
                        (heap.new_buffer_at_offset(length, options, 0)).map_err(failed(call))?,
                    ),
                    _ => None,
                };
                let made = Made {
                    address: Some(heap.gpu_address().0),
                };
                (Object::Block(heap, whole), made)
            }
            "newHeapWithDescriptor:" => {
                let descriptor = descriptors::heap(call)?;
                let heap = device.new_heap(&descriptor).map_err(failed(call))?;
                let made = Made {
                    address: Some(heap.gpu_address().0),
                };
                (Object::Heap(heap), made)
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                let Object::Block(heap, _) = self.get(call, Self::id(call, 0)?)? else {
                    return Err(refuse(call, "argument 0 is no block"));
                };
                let options = options(call, uint(call, 3)?)?;
                let buffer = heap.new_buffer_at_offset(uint(call, 2)?, options, uint(call, 1)?);
                addressed(buffer.map_err(failed(call))?)
            }
            "newLibraryWithData:error:" => {
                let library = device.new_library_with_data(data(call, 0)?);
                unaddressed(Object::Library(library.map_err(failed(call))?))
            }
            "newComputePipelineStateWithFunction:error:" => {
                let function = self.function_of(call, Self::id(call, 0)?)?;
                let state = device
                    .new_compute_pipeline_state_with_function(
                        &function,
                        &PipelineOptions::default(),
                    )
                    .map_err(failed(call))?;
                unaddressed(Object::ComputePipeline(state))
            }
            "newComputePipelineStateWithDescriptor:options:reflection:error:" => {
                let (descriptor, options) = self.compute_pipeline(call)?;
                let state = device.new_compute_pipeline_state(&descriptor, &options);
                unaddressed(Object::ComputePipeline(state.map_err(failed(call))?))
            }
            "newRenderPipelineStateWithDescriptor:error:" => {
                let (descriptor, options) = self.render_pipeline(call)?;
                let state = device.new_render_pipeline_state(&descriptor, &options);
                unaddressed(Object::RenderPipeline(state.map_err(failed(call))?))
            }
            "newDepthStencilStateWithDescriptor:" => {
                let descriptor = descriptors::depth_stencil(call)?;
                let state = device.new_depth_stencil_state(&descriptor);
                unaddressed(Object::DepthStencil(state.map_err(failed(call))?))
            }
            "newSamplerStateWithDescriptor:" => {
                let (descriptor, options) = descriptors::sampler(call)?;
                let state = device.new_sampler_state_with_options(&descriptor, options);
                unaddressed(Object::Sampler(state.map_err(failed(call))?))
            }
            "newFence" => unaddressed(Object::Fence(device.new_fence())),
            "newTextureWithDescriptor:" => {
                let descriptor = match arg(call, 0)? {
                    // A drawable the stream never saw made: a texture of its descriptor.
                    descriptor @ Value::Map(_) => {
                        let f = Fields::of(call, descriptor, &descriptors::TEXTURE_KEYS)?;
                        let mut made = TextureDescriptor::default();
                        descriptors::texture(call, &f, &mut made)?;
                        made
                    }
                    _ => self.texture_descriptor(call, 0)?,
                };
                let texture = device.new_texture(&descriptor).map_err(failed(call))?;
                unaddressed(Object::Texture(texture))
            }
            "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:" => {
                let format: PixelFormat = metal(call, uint(call, 0)?)?;
                let mipmapped = arg(call, 3)?.as_bool();
                let mipmapped = mipmapped.ok_or_else(|| refuse(call, "argument 3 is no bool"))?;
                let descriptor = TextureDescriptor::texture_2d(
                    format,
                    uint(call, 1)?,
                    uint(call, 2)?,
                    mipmapped,
                );
                unaddressed(Object::TextureDescriptor(Box::new(descriptor)))
            }
            "m2v.textureDescriptor" => {
                let f = Fields::arg(call, 0, &descriptors::TEXTURE_KEYS)?;
                let mut descriptor = TextureDescriptor::default();
                descriptors::texture(call, &f, &mut descriptor)?;
                unaddressed(Object::TextureDescriptor(Box::new(descriptor)))
            }
            "MTLRenderPassDescriptor.new" => {
                unaddressed(Object::Pass(Box::new(RenderPassDescriptor::new())))
            }
            "reserveResourceIndicesForResourceType:indices:indexCount:" => {
                let kind = match uint(call, 0)? {
                    0 => ResourceIdKind::Texture,
                    1 => ResourceIdKind::SamplerState,
                    4 => ResourceIdKind::PipelineState,
                    5 => ResourceIdKind::IndirectCommandBuffer,
                    other => return Err(refuse(call, format!("resource type {other}"))),
                };
                let ids = args::list(call, 1)?
                    .iter()
                    .map(|id| id.as_uint().map(ResourceId));
                let ids: Option<Vec<_>> = ids.collect();
                let ids = ids.ok_or_else(|| refuse(call, "argument 1 is no list of indices"))?;
                device
                    .reserve_resource_ids(kind, &ids)
                    .map_err(failed(call))?;
                return Ok(None);
            }
            "newEvent" => {
                let event = device.new_event().map_err(failed(call))?;
                unaddressed(Object::Event(event))
            }
            "newSharedEvent" => {
                let event = device.new_shared_event().map_err(failed(call))?;
                unaddressed(Object::SharedEvent(event))
            }
            _ => return Err(refuse(call, "not played on a device")),
        }))
    }

    /// A call of buffer `buffer`, and what it made.
    fn buffer_call(
        &self,
        call: &Call,
        buffer: &Buffer,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let texture = match &*call.selector {
            "didModifyRange:" => {
                let range = range(call, arg(call, 0)?)?;
                buffer.did_modify_range(range).map_err(failed(call))?;
                return Ok(None);
            }
            "newTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                buffer.new_texture(&descriptor, uint(call, 1)?, uint(call, 2)?)
            }
            "newTiledTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                buffer.new_tiled_texture(&descriptor, uint(call, 1)?, uint(call, 2)?)
            }
            "newTextureWithDescriptor:iosurface:plane:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                match descriptors::surface_plane(call, descriptor.pixel_format)? {
                    descriptors::Surface::Plane(plane) => {
                        buffer.new_texture_with_iosurface_plane(&descriptor, plane)
                    }
                    descriptors::Surface::None {
                        offset,
                        bytes_per_row,
                    } => buffer.new_texture(&descriptor, offset, bytes_per_row),
                }
            }
            _ => return Err(refuse(call, "not played on a buffer")),
        };
        let texture = texture.map_err(failed(call))?;
        Ok(Some(unaddressed(Object::Texture(texture))))
    }

    /// A call of a heap or a block, and what it made.
    fn heap_call(&self, call: &Call, heap: &Heap) -> Result<(Object, Made), PlayError> {
        match &*call.selector {
            "newBufferWithLength:options:offset:" => {
                let options = options(call, uint(call, 1)?)?;
                let buffer = heap.new_buffer_at_offset(uint(call, 0)?, options, uint(call, 2)?);
                Ok(addressed(buffer.map_err(failed(call))?))
            }
            "newTextureWithDescriptor:offset:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                let texture = heap.new_texture_at_offset(&descriptor, uint(call, 1)?);
                Ok(unaddressed(Object::Texture(texture.map_err(failed(call))?)))
            }
            _ => Err(refuse(call, "not played on a heap")),
        }
    }

    /// A call of texture `texture`, and what it made.
    fn texture_call(
        &self,
        call: &Call,
        texture: &Texture,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let format = |at: usize| metal::<PixelFormat>(call, uint(call, at)?);
        let kind = |at: usize| metal::<TextureType>(call, uint(call, at)?);
        let at = |at: usize| -> Result<IdPlacement, PlayError> {
            Ok(IdPlacement::At(ResourceId(uint(call, at)?)))
        };
        let view = match &*call.selector {
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:" => {
                let replaced = texture.replace_region(
                    region(call, arg(call, 0)?)?,
                    uint(call, 1)?,
                    uint(call, 2)?,
                    data(call, 3)?,
                    uint(call, 4)?,
                    uint(call, 5)?,
                );
                replaced.map_err(failed(call))?;
                return Ok(None);
            }
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:" => {
                let replaced = texture.replace_region(
                    region(call, arg(call, 0)?)?,
                    uint(call, 1)?,
                    0,
                    data(call, 2)?,
                    uint(call, 3)?,
                    0,
                );
                replaced.map_err(failed(call))?;
                return Ok(None);
            }
            "newTextureViewWithPixelFormat:" => texture.new_texture_view(format(0)?),
            "newTextureViewWithPixelFormat:resourceIndex:" => {
                texture.new_texture_view_with_placement(format(0)?, at(1)?)
            }
            "newTextureViewWithPixelFormat:textureType:levels:slices:" => texture
                .new_texture_view_with_levels_slices(
                    format(0)?,
                    kind(1)?,
                    range(call, arg(call, 2)?)?,
                    range(call, arg(call, 3)?)?,
                ),
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:" => texture
                .new_texture_subview_with_placement(
                    format(0)?,
                    kind(1)?,
                    range(call, arg(call, 2)?)?,
                    range(call, arg(call, 3)?)?,
                    None,
                    at(4)?,
                ),
            _ => return Err(refuse(call, "not played on a texture")),
        };
        let view = view.map_err(failed(call))?;
        Ok(Some(unaddressed(Object::Texture(view))))
    }

    /// A setter of the texture descriptor `call` is sent to.
    fn texture_descriptor_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let raw = uint(call, 0)?;
        let Some(Object::TextureDescriptor(descriptor)) = self.objects.get_mut(&call.receiver)
        else {
            unreachable!("a texture descriptor's call");
        };
        match &*call.selector {
            "setUsage:" => descriptor.usage = metal::<TextureUsage>(call, raw)?,
            "setResourceOptions:" => descriptor.resource_options = options(call, raw)?,
            _ => return Err(refuse(call, "not played on a texture descriptor")),
        }
        Ok(())
    }

    /// A call of library `library`: the function it makes.
    fn library_call(&self, call: &Call, library: &Library) -> Result<Function, PlayError> {
        let name = arg(call, 0)?.as_str();
        let name = name.ok_or_else(|| refuse(call, "argument 0 is no name"))?;
        let function = match &*call.selector {
            "newFunctionWithName:" => library.new_function_with_name(name),
            "newFunctionWithName:constantValues:error:" => {
                let constants = descriptors::constants(call, args::list(call, 1)?)?;
                library.new_function_with_name_constant_values(name, &constants)
            }
            "newFunctionWithDescriptor:error:" => {
                let descriptor = descriptors::function(call, name)?;
                library.new_function_with_descriptor(&descriptor)
            }
            _ => return Err(refuse(call, "not played on a library")),
        };
        function.map_err(failed(call))
    }

    /// Read `buffer`'s bytes through a shared buffer the readback queue copies them into.
    fn read_buffer(&mut self, buffer: &Buffer) -> Result<Vec<u8>, PlayError> {
        let length = buffer.length();
        self.read_through(length, |blit, into| {
            blit.copy_from_buffer_to_buffer(buffer, 0, into, 0, length)
        })
    }

    /// Level 0, slice 0 of `texture`, its rows packed.
    fn read_texture(&mut self, texture: &Texture) -> Result<Vec<u8>, PlayError> {
        let (width, height) = (texture.width(), texture.height());
        let query =
            TextureLevelQuery::plane_2d(texture.pixel_format().as_raw().into(), width, height);
        let level = self.device.texture_level_info(&query);
        self.read_through(level.bytes_per_image, |blit, into| {
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(width, height, 1),
                into,
                0,
                level.bytes_per_row,
                level.bytes_per_image,
                BlitOption::None,
            )
        })
    }

    fn read_through(
        &mut self,
        length: u64,
        copy: impl FnOnce(&mut BlitCommandEncoder, &Buffer) -> Result<(), Error>,
    ) -> Result<Vec<u8>, PlayError> {
        let broke = |error: Error| PlayError::Runtime {
            selector: "readback".into(),
            error: error.to_string(),
        };
        let queue = match &self.readback {
            Some(queue) => queue.clone(),
            None => {
                let queue = self.device.new_command_queue().map_err(broke)?;
                self.readback = Some(queue.clone());
                queue
            }
        };
        let shared = ResourceOptions::shared();
        let into = (self.device.new_buffer(length.max(1), shared)).map_err(broke)?;
        let command_buffer = queue.command_buffer().map_err(broke)?;
        let mut blit = command_buffer.blit_command_encoder().map_err(broke)?;
        copy(&mut blit, &into).map_err(broke)?;
        blit.end_encoding().map_err(broke)?;
        command_buffer.commit().map_err(broke)?;
        completed(&command_buffer, self.backend, "readback")?;
        let contents = into.contents().expect("a shared buffer's contents");
        // SAFETY: a shared buffer of at least `length` bytes, which no pending command writes.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
        Ok(bytes.to_vec())
    }
}

/// Wait for `command_buffer`, of `backend`, under the hang protocol (`gpu::wait`), and say how
/// it ended where it did not complete.
fn completed(
    command_buffer: &CommandBuffer,
    backend: SubmissionBackend,
    what: &str,
) -> Result<(), PlayError> {
    let operation = format!("the replay's {what}");
    wait_for_command_buffer(command_buffer, backend, &operation).map_err(|error| {
        PlayError::Runtime {
            selector: what.to_string(),
            error: format!(
                "{error}: {}",
                command_buffer.error_description().unwrap_or_default()
            ),
        }
    })
}

impl Backend for OldRuntime {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        if &*call.selector == "release" {
            self.objects.remove(&call.receiver);
            return Ok(Made::default());
        }
        let receiver = match call.receiver {
            0 => None,
            id => Some(self.get(call, id)?),
        };
        let made = match receiver {
            None | Some(Object::Device) => self.device_call(call)?,
            Some(Object::Library(library)) => {
                let function = self.library_call(call, library)?;
                Some(unaddressed(Object::Function(function)))
            }
            Some(Object::Queue(queue)) => match &*call.selector {
                "commandBuffer" => {
                    let command_buffer = queue.command_buffer().map_err(failed(call))?;
                    Some(unaddressed(Object::CommandBuffer(command_buffer)))
                }
                _ => return Err(refuse(call, "not played on a command queue")),
            },
            Some(Object::CommandBuffer(_)) => self.command_buffer_call(call)?.map(unaddressed),
            Some(Object::Compute(_)) => {
                self.compute_call(call)?;
                None
            }
            Some(Object::Blit(_)) => {
                self.blit_call(call)?;
                None
            }
            Some(Object::Render(_)) => {
                self.render_call(call)?;
                None
            }
            Some(Object::Buffer(buffer)) => self.buffer_call(call, buffer)?,
            Some(Object::Block(heap, _) | Object::Heap(heap)) => Some(self.heap_call(call, heap)?),
            Some(Object::Texture(texture)) => self.texture_call(call, texture)?,
            Some(Object::TextureDescriptor(_)) => {
                self.texture_descriptor_call(call)?;
                None
            }
            Some(Object::Pass(_)) => {
                self.pass_call(call)?;
                None
            }
            Some(Object::SharedEvent(event)) => match &*call.selector {
                "setSignaledValue:" => {
                    event.set_signaled_value(uint(call, 0)?);
                    None
                }
                _ => return Err(refuse(call, "not played on a shared event")),
            },
            Some(_) => return Err(refuse(call, "not played on its receiver")),
        };
        let Some((object, made)) = made else {
            return Ok(Made::default());
        };
        if call.result == 0 {
            return Err(PlayError::Invalid(format!(
                "{} makes no object",
                call.selector
            )));
        }
        self.objects.insert(call.result, object);
        Ok(made)
    }

    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
        let buffer = match self.objects.get(&target) {
            Some(Object::Buffer(buffer)) | Some(Object::Block(_, Some(buffer))) => buffer,
            _ => {
                return Err(PlayError::Invalid(format!(
                    "m2v.write to #{target}, which the CPU cannot write"
                )))
            }
        };
        let fits = offset
            .checked_add(bytes.len() as u64)
            .is_some_and(|end| end <= buffer.length());
        let contents = buffer.contents().filter(|_| fits).ok_or_else(|| {
            PlayError::Invalid(format!("m2v.write past #{target}'s CPU-visible bytes"))
        })?;
        // SAFETY: `[offset, offset + len)` lies in the buffer's mapped bytes, and the player waited
        // for every command buffer that can complete before writing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.as_ptr().add(offset as usize),
                bytes.len(),
            );
        }
        Ok(())
    }

    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        match self.objects.get(&command_buffer) {
            Some(Object::CommandBuffer(waited)) => completed(waited, self.backend, "commit"),
            _ => Err(PlayError::Invalid(format!(
                "#{command_buffer} is no command buffer"
            ))),
        }
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        match self.objects.get(&object) {
            Some(Object::Buffer(buffer)) => {
                let buffer = buffer.clone();
                self.read_buffer(&buffer)
            }
            Some(Object::Texture(texture)) => {
                let texture = texture.clone();
                self.read_texture(&texture)
            }
            _ => Err(PlayError::Invalid(format!(
                "#{object} is no buffer or texture"
            ))),
        }
    }

    fn counters(&mut self) -> Counters {
        let counts = self.device.work_counts();
        let c = &counts.commands_recorded;
        Counters::from([
            ("submissions", counts.queue_submissions),
            ("render_passes", c.cmd_begin_render_pass),
            ("barriers", c.cmd_pipeline_barrier),
            ("dispatches", c.cmd_dispatch + c.cmd_dispatch_indirect),
            (
                "draws",
                c.cmd_draw
                    + c.cmd_draw_indexed
                    + c.cmd_draw_indirect
                    + c.cmd_draw_indexed_indirect
                    + c.cmd_draw_indirect_count
                    + c.cmd_draw_indexed_indirect_count
                    + c.cmd_draw_multi
                    + c.cmd_draw_multi_indexed
                    + c.cmd_draw_mesh_tasks
                    + c.cmd_draw_mesh_tasks_indirect
                    + c.cmd_draw_mesh_tasks_indirect_count,
            ),
            (
                "copies",
                c.cmd_copy_buffer
                    + c.cmd_copy_image
                    + c.cmd_copy_buffer_to_image
                    + c.cmd_copy_image_to_buffer
                    + c.cmd_blit_image
                    + c.cmd_update_buffer
                    + c.cmd_fill_buffer,
            ),
            ("pipelines_made", counts.variant_pipelines_made),
        ])
    }
}
