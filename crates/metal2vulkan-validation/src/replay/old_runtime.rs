//! The player's backend over the old runtime (`metal2vulkan-runtime`), through its public API on
//! one device. A block is a placement heap with, where its memory is the CPU's too, a buffer over
//! all of it that writes go through; a texture is made at its recorded resource ID. A selector,
//! an argument or a descriptor key this file does not name is refused, naming it.

use super::player::{Backend, Counters, Made, PlayError};
use super::stream::Call;
use super::value::Value;
use metal2vulkan::metal::{
    BarrierScope, DispatchType, PixelFormat, ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputeCommandEncoder, ComputePipelineState, Device, Error, Event, EventRef, Function, Heap,
    HeapDescriptor, HeapType, IdPlacement, Library, Origin, PipelineOptions, Region, ResourceId,
    ResourceOptions, ResourceRef, SharedEvent, Size, StorageMode, Texture, TextureDescriptor,
    TextureLevelQuery,
};
use std::collections::HashMap;
use std::fmt::Display;
use std::time::Duration;

/// How long a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// One object of the stream, as the runtime holds it.
enum Object {
    Device,
    Queue(CommandQueue),
    CommandBuffer(CommandBuffer),
    Compute(ComputeCommandEncoder),
    /// Boxed: a blit encoder is some 3 KiB, twenty times any other object.
    Blit(Box<BlitCommandEncoder>),
    Buffer(Buffer),
    /// A block: its heap, and the buffer over all of it where the CPU writes it.
    Block(Heap, Option<Buffer>),
    Texture(Texture),
    Library(Library),
    Function(Function),
    ComputePipeline(ComputePipelineState),
    Event(Event),
    SharedEvent(SharedEvent),
}

/// The old runtime on one device.
pub struct OldRuntime {
    device: Device,
    objects: HashMap<u64, Object>,
    /// The queue of the player's readbacks.
    readback: Option<CommandQueue>,
    /// What `m2v.copyToHost:` copies into: the host's memory, which nothing in a replay reads.
    host: Option<Buffer>,
}

fn refuse(call: &Call, why: impl Into<String>) -> PlayError {
    PlayError::Refused {
        selector: call.selector.to_string(),
        why: why.into(),
    }
}

fn failed<'c>(call: &'c Call) -> impl Fn(Error) -> PlayError + 'c {
    move |error| PlayError::Runtime {
        selector: call.selector.to_string(),
        error: error.to_string(),
    }
}

fn arg(call: &Call, at: usize) -> Result<&Value, PlayError> {
    call.args
        .get(at)
        .ok_or_else(|| refuse(call, format!("no argument {at}")))
}

fn uint(call: &Call, at: usize) -> Result<u64, PlayError> {
    let value = arg(call, at)?;
    value
        .as_uint()
        .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not a uint")))
}

fn index(call: &Call, at: usize) -> Result<u32, PlayError> {
    let value = uint(call, at)?;
    u32::try_from(value).map_err(|_| refuse(call, format!("argument {at}, {value}, past u32")))
}

fn data(call: &Call, at: usize) -> Result<&[u8], PlayError> {
    let value = arg(call, at)?;
    value
        .as_bytes()
        .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not data")))
}

fn size(call: &Call, at: usize) -> Result<Size, PlayError> {
    match arg(call, at)?.as_uints().as_deref() {
        Some(&[w, h, d]) => Ok(Size::new(w, h, d)),
        _ => Err(refuse(call, format!("argument {at} is no [w, h, d]"))),
    }
}

/// A Metal enum or option set of raw value `raw`.
fn metal<T: TryFrom<u32>>(call: &Call, raw: u64) -> Result<T, PlayError>
where
    T::Error: Display,
{
    let narrow = u32::try_from(raw).map_err(|_| refuse(call, format!("raw value {raw}")))?;
    T::try_from(narrow).map_err(|error| refuse(call, format!("raw value {raw}: {error}")))
}

fn options(call: &Call, raw: u64) -> Result<ResourceOptions, PlayError> {
    ResourceOptions::from_raw(raw).map_err(failed(call))
}

impl OldRuntime {
    pub fn new(device: Device) -> Self {
        OldRuntime {
            device,
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

    fn object_arg(&self, call: &Call, at: usize) -> Result<&Object, PlayError> {
        let value = arg(call, at)?;
        let id = value
            .as_object()
            .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not an object")))?;
        self.get(call, id)
    }

    fn buffer(&self, call: &Call, at: usize) -> Result<Buffer, PlayError> {
        match self.object_arg(call, at)? {
            Object::Buffer(buffer) => Ok(buffer.clone()),
            _ => Err(refuse(call, format!("argument {at} is no buffer"))),
        }
    }

    /// The buffer of argument `at`, or none where it is nil.
    fn optional_buffer(&self, call: &Call, at: usize) -> Result<Option<Buffer>, PlayError> {
        match arg(call, at)? {
            Value::Nil => Ok(None),
            _ => self.buffer(call, at).map(Some),
        }
    }

    fn event<'s>(&'s self, call: &Call, at: usize) -> Result<EventRef<'s>, PlayError> {
        match self.object_arg(call, at)? {
            Object::Event(event) => Ok(EventRef::Event(event)),
            Object::SharedEvent(event) => Ok(EventRef::Shared(event)),
            _ => Err(refuse(call, format!("argument {at} is no event"))),
        }
    }

    /// A call whose receiver is the device (or no receiver).
    fn device_call(&self, call: &Call) -> Result<(Object, Made), PlayError> {
        let device = &self.device;
        let addressed = |buffer: Buffer| {
            let made = Made {
                address: Some(buffer.gpu_address().0),
                resource_id: None,
            };
            (Object::Buffer(buffer), made)
        };
        Ok(match &*call.selector {
            "MTLCreateSystemDefaultDevice" => (Object::Device, Made::default()),
            "newCommandQueue" => {
                let queue = device.new_command_queue().map_err(failed(call))?;
                (Object::Queue(queue), Made::default())
            }
            "newBufferWithLength:options:" => {
                let options = options(call, uint(call, 1)?)?;
                addressed(
                    device
                        .new_buffer(uint(call, 0)?, options)
                        .map_err(failed(call))?,
                )
            }
            "newBufferWithBytes:length:options:" => {
                let bytes = data(call, 0)?;
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| bytes.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let options = options(call, uint(call, 2)?)?;
                addressed(
                    device
                        .new_buffer_with_bytes(bytes, options)
                        .map_err(failed(call))?,
                )
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
                    resource_id: None,
                };
                (Object::Block(heap, whole), made)
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                let Object::Block(heap, _) = self.object_arg(call, 0)? else {
                    return Err(refuse(call, "argument 0 is no block"));
                };
                let options = options(call, uint(call, 3)?)?;
                let buffer = heap.new_buffer_at_offset(uint(call, 2)?, options, uint(call, 1)?);
                addressed(buffer.map_err(failed(call))?)
            }
            "newLibraryWithData:error:" => {
                let library = device.new_library_with_data(data(call, 0)?);
                (
                    Object::Library(library.map_err(failed(call))?),
                    Made::default(),
                )
            }
            "newComputePipelineStateWithFunction:error:" => {
                let Object::Function(function) = self.object_arg(call, 0)? else {
                    return Err(refuse(call, "argument 0 is no function"));
                };
                let state = device
                    .new_compute_pipeline_state_with_function(function, &PipelineOptions::default())
                    .map_err(failed(call))?;
                let made = Made {
                    address: None,
                    resource_id: Some(state.gpu_resource_id().0),
                };
                (Object::ComputePipeline(state), made)
            }
            "newTextureWithDescriptor:" => {
                let descriptor = texture_descriptor(call)?;
                let texture = device.new_texture(&descriptor).map_err(failed(call))?;
                let made = Made {
                    address: None,
                    resource_id: Some(texture.gpu_resource_id().0),
                };
                (Object::Texture(texture), made)
            }
            "newEvent" => {
                let event = device.new_event().map_err(failed(call))?;
                (Object::Event(event), Made::default())
            }
            "newSharedEvent" => {
                let event = device.new_shared_event().map_err(failed(call))?;
                (Object::SharedEvent(event), Made::default())
            }
            _ => return Err(refuse(call, "not played on a device")),
        })
    }

    fn compute_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        match &*call.selector {
            "endEncoding" => {
                let Some(Object::Compute(encoder)) = self.objects.remove(&id) else {
                    unreachable!("a compute encoder's call");
                };
                return encoder.end_encoding().map_err(failed(call));
            }
            "memoryBarrierWithScope:" => {
                let scope: BarrierScope = metal(call, uint(call, 0)?)?;
                self.compute(id).memory_barrier_with_scope(scope);
                return Ok(());
            }
            _ => {}
        }
        let result = match &*call.selector {
            "setComputePipelineState:" => {
                let Object::ComputePipeline(state) = self.object_arg(call, 0)? else {
                    return Err(refuse(call, "argument 0 is no compute pipeline state"));
                };
                let state = state.clone();
                self.compute(id).set_compute_pipeline_state(&state)
            }
            "setBuffer:offset:atIndex:" => {
                let buffer = self.optional_buffer(call, 0)?;
                let (offset, at) = (uint(call, 1)?, index(call, 2)?);
                self.compute(id).set_buffer(buffer.as_ref(), offset, at)
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
                let texture = match arg(call, 0)? {
                    Value::Nil => None,
                    _ => match self.object_arg(call, 0)? {
                        Object::Texture(texture) => Some(texture.clone()),
                        _ => return Err(refuse(call, "argument 0 is no texture")),
                    },
                };
                let at = index(call, 1)?;
                self.compute(id).set_texture(texture.as_ref(), at)
            }
            "dispatchThreadgroups:threadsPerThreadgroup:" => {
                let (groups, threads) = (size(call, 0)?, size(call, 1)?);
                self.compute(id).dispatch_threadgroups(groups, threads)
            }
            "dispatchThreads:threadsPerThreadgroup:" => {
                let (grid, threads) = (size(call, 0)?, size(call, 1)?);
                self.compute(id).dispatch_threads(grid, threads)
            }
            "useResource:usage:" => {
                let usage: ResourceUsage = metal(call, uint(call, 1)?)?;
                let resource = match self.object_arg(call, 0)? {
                    Object::Buffer(buffer) => Resource::Buffer(buffer.clone()),
                    Object::Texture(texture) => Resource::Texture(texture.clone()),
                    _ => return Err(refuse(call, "argument 0 is no buffer or texture")),
                };
                self.compute(id).use_resource(resource.as_ref(), usage)
            }
            "useHeap:" => {
                let Object::Block(heap, _) = self.object_arg(call, 0)? else {
                    return Err(refuse(call, "argument 0 is no block"));
                };
                let heap = heap.clone();
                self.compute(id).use_heap(&heap)
            }
            _ => return Err(refuse(call, "not played on a compute encoder")),
        };
        result.map_err(failed(call))
    }

    fn compute(&mut self, id: u64) -> &mut ComputeCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Compute(encoder)) => encoder,
            _ => unreachable!("a compute encoder's call"),
        }
    }

    fn blit(&mut self, id: u64) -> &mut BlitCommandEncoder {
        match self.objects.get_mut(&id) {
            Some(Object::Blit(encoder)) => encoder,
            _ => unreachable!("a blit encoder's call"),
        }
    }

    fn blit_call(&mut self, call: &Call) -> Result<(), PlayError> {
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
            "fillBuffer:range:value:" => {
                let buffer = self.buffer(call, 0)?;
                let Some(&[start, length]) = arg(call, 1)?.as_uints().as_deref() else {
                    return Err(refuse(call, "argument 1 is no [location, length]"));
                };
                let value = u8::try_from(uint(call, 2)?)
                    .map_err(|_| refuse(call, "a fill value past a byte"))?;
                self.blit(id)
                    .fill_buffer(&buffer, start..start + length, value)
            }
            "synchronizeResource:" => {
                let resource = match self.object_arg(call, 0)? {
                    Object::Buffer(buffer) => Resource::Buffer(buffer.clone()),
                    Object::Texture(texture) => Resource::Texture(texture.clone()),
                    _ => return Err(refuse(call, "argument 0 is no buffer or texture")),
                };
                self.blit(id).synchronize_resource(resource.as_ref())
            }
            "m2v.copyBytes:toBuffer:offset:" => {
                let bytes = data(call, 0)?;
                let to = self.buffer(call, 1)?;
                let at = uint(call, 2)?;
                let shared = ResourceOptions::shared();
                let from =
                    (self.device.new_buffer_with_bytes(bytes, shared)).map_err(failed(call))?;
                (self.blit(id)).copy_from_buffer_to_buffer(&from, 0, &to, at, bytes.len() as u64)
            }
            "m2v.copyToHost:" => {
                let from = self.buffer(call, 0)?;
                let (at, length) = (uint(call, 1)?, uint(call, 2)?);
                let host = self.host(call, length)?;
                (self.blit(id)).copy_from_buffer_to_buffer(&from, at, &host, 0, length)
            }
            _ => return Err(refuse(call, "not played on a blit encoder")),
        };
        result.map_err(failed(call))
    }

    /// The host memory `m2v.copyToHost:` copies `length` bytes into.
    fn host(&mut self, call: &Call, length: u64) -> Result<Buffer, PlayError> {
        match &self.host {
            Some(host) if host.length() >= length => Ok(host.clone()),
            _ => {
                let made = self.device.new_buffer(length, ResourceOptions::shared());
                let host = made.map_err(failed(call))?;
                self.host = Some(host.clone());
                Ok(host)
            }
        }
    }

    fn command_buffer_call(&self, call: &Call) -> Result<Option<Object>, PlayError> {
        let Some(Object::CommandBuffer(command_buffer)) = self.objects.get(&call.receiver) else {
            unreachable!("a command buffer's call");
        };
        let made = match &*call.selector {
            "computeCommandEncoder" => command_buffer
                .compute_command_encoder()
                .map(Object::Compute),
            "computeCommandEncoderWithDispatchType:" => {
                let dispatch: DispatchType = metal(call, uint(call, 0)?)?;
                (command_buffer.compute_command_encoder_with_dispatch_type(dispatch))
                    .map(Object::Compute)
            }
            "blitCommandEncoder" => (command_buffer.blit_command_encoder())
                .map(|encoder| Object::Blit(Box::new(encoder))),
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
        let into = (self
            .device
            .new_buffer(length.max(1), ResourceOptions::shared()))
        .map_err(broke)?;
        let command_buffer = queue.command_buffer().map_err(broke)?;
        let mut blit = command_buffer.blit_command_encoder().map_err(broke)?;
        copy(&mut blit, &into).map_err(broke)?;
        blit.end_encoding().map_err(broke)?;
        command_buffer.commit().map_err(broke)?;
        completed(&command_buffer, "readback")?;
        let contents = into.contents().expect("a shared buffer's contents");
        // SAFETY: a shared buffer of at least `length` bytes, which no pending command writes.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
        Ok(bytes.to_vec())
    }
}

/// A resource an encoder's call names.
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

/// Wait for `command_buffer` and say how it ended where it did not complete.
fn completed(command_buffer: &CommandBuffer, what: &str) -> Result<(), PlayError> {
    let runtime = |error: String| PlayError::Runtime {
        selector: what.to_string(),
        error,
    };
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| runtime(format!("not complete after {WAIT:?}")))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(runtime(format!(
            "{status:?}: {}",
            command_buffer.error_description().unwrap_or_default()
        ))),
    }
}

/// The texture descriptor of the map `call` takes, placed at the recorded resource ID.
fn texture_descriptor(call: &Call) -> Result<TextureDescriptor, PlayError> {
    let map = arg(call, 0)?
        .as_map()
        .ok_or_else(|| refuse(call, "argument 0 is no descriptor"))?;
    let key = |name: &str| map.iter().find(|(k, _)| &**k == name).map(|(_, v)| v);
    let number = |name: &str, default: u64| -> Result<u64, PlayError> {
        match key(name) {
            None => Ok(default),
            Some(value) => value
                .as_uint()
                .ok_or_else(|| refuse(call, format!("{name} is {value}, not a uint"))),
        }
    };
    const KNOWN: [&str; 13] = [
        "textureType",
        "pixelFormat",
        "width",
        "height",
        "depth",
        "mipmapLevelCount",
        "sampleCount",
        "arrayLength",
        "resourceOptions",
        "storageMode",
        "cpuCacheMode",
        "hazardTrackingMode",
        "usage",
    ];
    if let Some((unknown, _)) = map.iter().find(|(k, _)| !KNOWN.contains(&&**k)) {
        return Err(refuse(call, format!("the descriptor key {unknown}")));
    }
    let format: PixelFormat = metal(call, number("pixelFormat", 0)?)?;
    let mut d =
        TextureDescriptor::texture_2d(format, number("width", 1)?, number("height", 1)?, false);
    d.texture_type = metal::<TextureType>(call, number("textureType", 2)?)?;
    d.depth = number("depth", 1)?;
    d.mipmap_level_count = number("mipmapLevelCount", 1)?;
    d.sample_count = number("sampleCount", 1)?;
    d.array_length = number("arrayLength", 1)?;
    d.usage = metal::<TextureUsage>(call, number("usage", 1)?)?;
    let raw = number("resourceOptions", 0)?
        | number("cpuCacheMode", 0)?
        | number("storageMode", 0)? << 4
        | number("hazardTrackingMode", 0)? << 8;
    d.resource_options = options(call, raw)?;
    if let Some(id) = call.answer("gpuResourceID").and_then(Value::as_uint) {
        d.placement = IdPlacement::At(ResourceId(id));
    }
    Ok(d)
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
            None | Some(Object::Device) => Some(self.device_call(call)?),
            Some(Object::Library(library)) => match &*call.selector {
                "newFunctionWithName:" => {
                    let name = arg(call, 0)?.as_str();
                    let name = name.ok_or_else(|| refuse(call, "argument 0 is no name"))?;
                    let function = library.new_function_with_name(name).map_err(failed(call))?;
                    Some((Object::Function(function), Made::default()))
                }
                _ => return Err(refuse(call, "not played on a library")),
            },
            Some(Object::Queue(queue)) => match &*call.selector {
                "commandBuffer" => {
                    let command_buffer = queue.command_buffer().map_err(failed(call))?;
                    Some((Object::CommandBuffer(command_buffer), Made::default()))
                }
                _ => return Err(refuse(call, "not played on a command queue")),
            },
            Some(Object::CommandBuffer(_)) => self
                .command_buffer_call(call)?
                .map(|object| (object, Made::default())),
            Some(Object::Compute(_)) => {
                self.compute_call(call)?;
                None
            }
            Some(Object::Blit(_)) => {
                self.blit_call(call)?;
                None
            }
            Some(Object::Buffer(buffer)) => match &*call.selector {
                "didModifyRange:" => {
                    let Some(&[start, length]) = arg(call, 0)?.as_uints().as_deref() else {
                        return Err(refuse(call, "argument 0 is no [location, length]"));
                    };
                    buffer
                        .did_modify_range(start..start + length)
                        .map_err(failed(call))?;
                    None
                }
                _ => return Err(refuse(call, "not played on a buffer")),
            },
            Some(Object::Texture(texture)) => {
                let region = |at: usize| match arg(call, at)?.as_uints().as_deref() {
                    Some(&[x, y, z, w, h, d]) => Ok(Region {
                        origin: Origin::new(x, y, z),
                        size: Size::new(w, h, d),
                    }),
                    _ => Err(refuse(call, format!("argument {at} is no region"))),
                };
                let replaced = match &*call.selector {
                    "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:" => {
                        texture.replace_region(
                            region(0)?,
                            uint(call, 1)?,
                            uint(call, 2)?,
                            data(call, 3)?,
                            uint(call, 4)?,
                            uint(call, 5)?,
                        )
                    }
                    "replaceRegion:mipmapLevel:withBytes:bytesPerRow:" => texture.replace_region(
                        region(0)?,
                        uint(call, 1)?,
                        0,
                        data(call, 2)?,
                        uint(call, 3)?,
                        0,
                    ),
                    _ => return Err(refuse(call, "not played on a texture")),
                };
                replaced.map_err(failed(call))?;
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
            Some(Object::CommandBuffer(waited)) => completed(waited, "commit"),
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
