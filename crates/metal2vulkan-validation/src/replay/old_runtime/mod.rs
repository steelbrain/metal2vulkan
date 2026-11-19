//! The player's backend over the old runtime (`metal2vulkan-runtime`), through its public API on
//! one device. A block is a placement heap with, where its memory is the CPU's too, a buffer over
//! all of it that writes go through; a descriptor is an object its setter records change; a
//! drawable the stream never saw made is a texture of its descriptor. A selector, an argument or
//! a descriptor key this module does not name is refused, naming it.

mod args;
mod descriptors;
mod encoders;

use super::heaps::HeapSplit;
use super::new_runtime::ManagedCopies;
use super::player::{Backend, Counters, Made, PlayError};
use super::stream::Call;
use super::value::Value;
use crate::gpu::wait::{runtime::wait_for_command_buffer, SubmissionBackend};
use crate::gpu::DeviceType;
use args::{arg, data, failed, metal, options, range, refuse, region, uint, Fields};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    AddressPlacement, BlitCommandEncoder, BlitOption, Buffer, BufferContents, BufferDescriptor,
    CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState, DepthStencilState,
    Device, Error, Event, EventRef, Fence, Function, GpuAddress, Heap, HeapDescriptor, HeapType,
    IdPlacement, Library, Origin, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceId, ResourceIdKind, ResourceOptions, SamplerState, SharedEvent,
    Size, StorageMode, Texture, TextureDescriptor, TextureLevelQuery,
};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

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
    /// A heap: the runtime's heap its buffers go in, and, where the player split it (`HeapSplit`)
    /// and the stream places both buffers and textures in it, the heap outside the space its
    /// textures go in. Otherwise its textures go in the first.
    Heap(Heap, Option<Heap>),
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
    /// What no device here builds: a library from source, and what is made from it.
    Skipped,
}

/// What the play skipped, counted.
#[derive(Default)]
struct Skips {
    /// Libraries from source, and objects made from a skipped one or naming one.
    objects: u64,
    /// Draws through a skipped render pipeline state.
    draws: u64,
    /// Every other call on a skipped object or naming one.
    calls: u64,
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
    skips: Skips,
    /// The render encoders whose pipeline state is a skipped one: their draws are skipped.
    unpiped: HashSet<u64>,
    /// Whether the stream's header marks its announced bytes recorded, and the buffers'
    /// `synchronizeResource:` left out of their CPU copies for it.
    marked: bool,
    /// The host's display library (`with_display_library`), which a library from source of no
    /// arguments is made of.
    display: Option<Vec<u8>>,
    read_backs_left_out: u64,
    /// The heap the next call makes and how the player split it (`Backend::split_heap`).
    split: Cell<Option<(u64, HeapSplit)>>,
    /// How the stream's recording device kept Managed buffers, and whether its identity record
    /// said so (else the player's setting, `with_managed`).
    managed: Option<ManagedCopies>,
    identity: bool,
    /// The Managed buffers and blocks a one-copy stream made, played as Shared.
    managed_shared: Cell<u64>,
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

/// Where making `call` puts its object in the runtime's address space (`docs/REPLAY-STREAM.md`,
/// section 5): at its recorded address, the `gpuAddress` it answered, where the stream recorded a
/// range for it; otherwise outside the space, where no recorded address lies (`address.rs` of the
/// runtime: `[2^47, 2^48)`, which a pin never reaches).
fn address_placement(call: &Call) -> Result<AddressPlacement, PlayError> {
    match call.answer("gpuAddress") {
        None => Ok(AddressPlacement::Outside),
        Some(answer) => match answer.as_address() {
            Some(address) => Ok(AddressPlacement::At(GpuAddress(address))),
            None => Err(refuse(call, format!("gpuAddress {answer} is no address"))),
        },
    }
}

/// The error of making `call`'s object at `placement`: a refused recorded address stops the play
/// naming the object (`PlayError::Unpinned`), with no other placement tried.
fn unplaced(call: &Call, placement: AddressPlacement) -> impl Fn(Error) -> PlayError + '_ {
    move |error| match placement {
        AddressPlacement::At(GpuAddress(recorded)) => PlayError::Unpinned {
            object: call.result,
            recorded,
            made: None,
            why: error.to_string(),
        },
        _ => failed(call)(error),
    }
}

/// The buffer `call` makes, `length` bytes of `options` holding `contents`, at its
/// `address_placement`.
fn placed(
    device: &Device,
    call: &Call,
    length: u64,
    options: ResourceOptions,
    contents: BufferContents<'_>,
) -> Result<(Object, Made), PlayError> {
    let at = address_placement(call)?;
    let descriptor = BufferDescriptor::new(length, options, contents).with_placement(at);
    let buffer = device.new_buffer_with_descriptor(descriptor);
    Ok(addressed(buffer.map_err(unplaced(call, at))?))
}

/// `made`, a buffer of `options` that `call` made, holding `bytes` from its first byte, the bytes
/// the stream says it was made with (a descriptor's `contents`, a no-copy buffer's pages): where it
/// plays as the runtime's Managed buffer, its GPU copy starts from them too, as Metal makes it
/// where the device keeps two copies (`docs/REPLAY-STREAM.md`, Managed buffers). The player writes
/// them again once the buffer is made, which changes nothing.
fn made_with(
    call: &Call,
    made: (Object, Made),
    options: ResourceOptions,
    bytes: &[u8],
) -> Result<(Object, Made), PlayError> {
    let (Object::Buffer(buffer), _) = &made else {
        unreachable!("a buffer maker's object");
    };
    if bytes.is_empty() || options.storage_mode() != Ok(StorageMode::Managed) {
        return Ok(made);
    }
    write_into(buffer, 0, bytes).map_err(|why| refuse(call, why))?;
    let end = bytes.len() as u64;
    buffer.did_modify_range(0..end).map_err(failed(call))?;
    Ok(made)
}

/// `bytes` into `buffer`'s CPU-visible bytes at `offset`; why not, where they do not fit.
fn write_into(buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), String> {
    let fits = offset
        .checked_add(bytes.len() as u64)
        .is_some_and(|end| end <= buffer.length());
    let contents = (buffer.contents().filter(|_| fits)).ok_or("bytes past its CPU-visible ones")?;
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

/// A buffer of the player's own (a staging copy, a read-back), `length` bytes of `contents`:
/// outside the space, so that no recorded address can meet it.
pub(super) fn own_buffer(
    device: &Device,
    length: u64,
    contents: BufferContents<'_>,
) -> Result<Buffer, Error> {
    let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents);
    device.new_buffer_with_descriptor(descriptor.with_placement(AddressPlacement::Outside))
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
            skips: Skips::default(),
            unpiped: HashSet::new(),
            marked: false,
            read_backs_left_out: 0,
            display: None,
            split: Cell::new(None),
            managed: None,
            identity: false,
            managed_shared: Cell::new(0),
        }
    }

    /// The backend told how the stream's recording device kept Managed buffers, where the stream
    /// has no identity record that says (`--managed`).
    pub fn with_managed(mut self, copies: ManagedCopies) -> Self {
        self.managed = Some(copies);
        self
    }

    /// The options a buffer or block of `options` is made with: Shared where `options` are
    /// Managed and the stream was recorded with one copy, as the new runtime's backend plays them
    /// (`docs/REPLAY-STREAM.md`, Managed buffers), so a CPU write reaches the GPU as it reached
    /// the recording device's; else as recorded, the runtime's Managed buffer keeping two copies
    /// where the device does.
    fn played(&self, call: &Call, options: ResourceOptions) -> Result<ResourceOptions, PlayError> {
        match (self.managed, options.storage_mode()) {
            (Some(ManagedCopies::One), Ok(StorageMode::Managed)) => {
                self.managed_shared.set(self.managed_shared.get() + 1);
                let storage = 0xf << 4;
                let shared = StorageMode::Shared.raw() << 4;
                args::options(call, options.raw() & !storage | shared)
            }
            _ => Ok(options),
        }
    }

    /// The backend given the host's display library, a metallib (the stream's
    /// `NAME.display.metallib`): the library from source the host makes of its own display
    /// shaders, a `newLibraryWithSource:options:error:` of no arguments, is made of it. Without
    /// it, that library is skipped and counted as any library from source.
    pub fn with_display_library(mut self, metallib: Vec<u8>) -> Self {
        self.display = Some(metallib);
        self
    }

    /// Whether `id` is an object the play skipped.
    fn skipped(&self, id: u64) -> bool {
        matches!(self.objects.get(&id), Some(Object::Skipped))
    }

    /// Whether `value` names an object the play skipped, at any depth.
    fn holds_skipped(&self, value: &Value) -> bool {
        match value {
            Value::Object(id) => self.skipped(*id),
            Value::List(items) => items.iter().any(|item| self.holds_skipped(item)),
            Value::Map(entries) => entries.iter().any(|(_, item)| self.holds_skipped(item)),
            _ => false,
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

    /// The sampler `value` names, nil for none.
    fn sampler(&self, call: &Call, value: &Value) -> Result<Option<SamplerState>, PlayError> {
        match value {
            Value::Nil => Ok(None),
            Value::Object(id) => self
                .kind(call, *id, "sampler state", |object| match object {
                    Object::Sampler(sampler) => Some(sampler.clone()),
                    _ => None,
                })
                .map(Some),
            _ => Err(refuse(call, format!("{value} is no sampler state"))),
        }
    }

    /// The samplers of a list argument, nil for none.
    fn samplers(&self, call: &Call, at: usize) -> Result<Vec<Option<SamplerState>>, PlayError> {
        let list = args::list(call, at)?.iter();
        list.map(|value| self.sampler(call, value)).collect()
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
                let options = self.played(call, options(call, uint(call, 1)?)?)?;
                placed(device, call, uint(call, 0)?, options, BufferContents::Zero)?
            }
            "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                // Played as a buffer of its length: without the read-only option (`buffer`).
                let options = options(call, uint(call, 2)? & !ResourceOptions::PRIVATE_BIT)?;
                let options = self.played(call, options)?;
                let made = placed(device, call, uint(call, 1)?, options, BufferContents::Zero)?;
                // The pages' bytes at its making, where the stream gives them.
                let bytes = arg(call, 0)?.as_bytes().unwrap_or_default();
                made_with(call, made, options, bytes)?
            }
            "newBufferWithBytes:length:options:" => {
                let bytes = data(call, 0)?;
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| bytes.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let options = self.played(call, options(call, uint(call, 2)?)?)?;
                placed(device, call, length, options, BufferContents::Copy(bytes))?
            }
            "newBufferWithDescriptor:" => {
                let (length, options, contents) = descriptors::buffer(call)?;
                let options = self.played(call, options)?;
                let made = placed(device, call, length, options, BufferContents::Zero)?;
                made_with(call, made, options, contents)?
            }
            "m2v.newBlockWithLength:storageMode:" => {
                let length = uint(call, 0)?;
                let mode = StorageMode::from_raw(uint(call, 1)?).map_err(failed(call))?;
                let options = self.played(call, options(call, mode.raw() << 4)?)?;
                let mode = options.storage_mode().map_err(failed(call))?;
                let at = address_placement(call)?;
                let descriptor = HeapDescriptor::new(HeapType::Placement, length, options);
                let descriptor = descriptor.with_placement(at);
                let heap = device.new_heap(&descriptor).map_err(unplaced(call, at))?;
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
                let split = self.split.take().filter(|&(heap, _)| heap == call.result);
                // The heap the buffers go in, at the recorded address and size; the one the
                // textures go in, outside the space at the size the player laid them out at.
                let pinned = || -> Result<Heap, PlayError> {
                    let at = address_placement(call)?;
                    let pinned = descriptor.with_placement(at);
                    device.new_heap(&pinned).map_err(unplaced(call, at))
                };
                let outside = |size: u64| -> Result<Heap, PlayError> {
                    let mut outside = descriptor.with_placement(AddressPlacement::Outside);
                    outside.size = size;
                    device.new_heap(&outside).map_err(failed(call))
                };
                let (heap, textures) = match split.map(|(_, split)| split) {
                    None => (pinned()?, None),
                    Some(HeapSplit {
                        buffers: false,
                        textures,
                    }) => (outside(textures.unwrap_or(descriptor.size))?, None),
                    Some(HeapSplit {
                        buffers: true,
                        textures,
                    }) => (pinned()?, textures.map(outside).transpose()?),
                };
                let made = Made {
                    address: Some(heap.gpu_address().0),
                };
                (Object::Heap(heap, textures), made)
            }
            "newLibraryWithData:error:" => {
                let library = device.new_library_with_data(data(call, 0)?);
                unaddressed(Object::Library(library.map_err(failed(call))?))
            }
            "newLibraryWithSource:options:error:" => match &self.display {
                Some(metallib) if call.args.is_empty() => {
                    let library = device.new_library_with_data(metallib);
                    unaddressed(Object::Library(library.map_err(failed(call))?))
                }
                _ => return Err(refuse(call, "a library from source")),
            },
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
                let descriptor = super::translator::depth_stencil(call)?;
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
                        f.only("outside", Fields::bool, false)?;
                        let mut made = TextureDescriptor::default();
                        descriptors::texture(call, &f, &mut made)?;
                        made
                    }
                    _ => self.texture_descriptor(call, 0)?,
                };
                let texture = device.new_texture(&descriptor).map_err(failed(call))?;
                unaddressed(Object::Texture(texture))
            }
            "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:"
            | "m2v.textureDescriptor" => {
                let descriptor = made_texture_descriptor(call)?;
                unaddressed(Object::TextureDescriptor(Box::new(descriptor)))
            }
            "MTLRenderPassDescriptor.new" | "MTLRenderPassDescriptor.renderPassDescriptor" => {
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
            // Another API's write landed in the buffer: the textures over it read it in place.
            "m2v.externalWriteDone" => return Ok(None),
            "newTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                buffer.new_texture(&descriptor, uint(call, 1)?, uint(call, 2)?)
            }
            "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                let (offset, bytes_per_row) = (uint(call, 1)?, uint(call, 2)?);
                buffer.new_linear_texture(&descriptor, offset, bytes_per_row, uint(call, 3)?)
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

    /// A call of a heap or a block, and what it made: a buffer in `heap`, at its recorded address
    /// where it has one (a refusal stops the play naming it, `PlayError::Unpinned`), a texture in
    /// `textures`.
    fn heap_call(
        &self,
        call: &Call,
        heap: &Heap,
        textures: &Heap,
    ) -> Result<(Object, Made), PlayError> {
        match &*call.selector {
            "m2v.newBufferInBlock:offset:length:options:" => {
                if Self::id(call, 0)? != call.receiver {
                    return Err(refuse(call, "a block other than its receiver"));
                }
                let options = self.played(call, options(call, uint(call, 3)?)?)?;
                let buffer = heap.new_buffer_at_offset(uint(call, 2)?, options, uint(call, 1)?);
                Ok(addressed(buffer.map_err(failed(call))?))
            }
            "newBufferWithLength:options:offset:" => {
                let options = options(call, uint(call, 1)?)?;
                let at = address_placement(call)?;
                let buffer = heap.new_buffer_at_offset(uint(call, 0)?, options, uint(call, 2)?);
                Ok(addressed(buffer.map_err(unplaced(call, at))?))
            }
            "newTextureWithDescriptor:offset:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                let texture = textures.new_texture_at_offset(&descriptor, uint(call, 1)?);
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
        // A view's swizzle, argument 4 of the calls that take one.
        let swizzle = || descriptors::swizzle(&Fields::arg(call, 4, &descriptors::SWIZZLE_KEYS)?);
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
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:" => texture
                .new_texture_view_with_swizzle(
                    format(0)?,
                    kind(1)?,
                    range(call, arg(call, 2)?)?,
                    range(call, arg(call, 3)?)?,
                    swizzle()?,
                ),
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:" => {
                texture.new_texture_subview_with_placement(
                    format(0)?,
                    kind(1)?,
                    range(call, arg(call, 2)?)?,
                    range(call, arg(call, 3)?)?,
                    Some(swizzle()?),
                    at(5)?,
                )
            }
            _ => return Err(refuse(call, "not played on a texture")),
        };
        let view = view.map_err(failed(call))?;
        Ok(Some(unaddressed(Object::Texture(view))))
    }

    /// A setter of the texture descriptor `call` is sent to.
    fn texture_descriptor_call(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(Object::TextureDescriptor(descriptor)) = self.objects.get_mut(&call.receiver)
        else {
            unreachable!("a texture descriptor's call");
        };
        set_texture_descriptor(call, descriptor)
    }

    /// A call of library `library`: the function it makes.
    fn library_call(&self, call: &Call, library: &Library) -> Result<Function, PlayError> {
        let name = arg(call, 0)?.as_str();
        let name = name.ok_or_else(|| refuse(call, "argument 0 is no name"))?;
        let function = match &*call.selector {
            "newFunctionWithName:" => library.new_function_with_name(name),
            "newFunctionWithName:constantValues:error:" => {
                let constants = super::translator::constants(call, args::list(call, 1)?)?;
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
        let into = own_buffer(&self.device, length.max(1), BufferContents::Zero).map_err(broke)?;
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

/// The texture descriptor `call` makes: `m2v.textureDescriptor` or
/// `MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:`.
fn made_texture_descriptor(call: &Call) -> Result<TextureDescriptor, PlayError> {
    match &*call.selector {
        "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:" => {
            let format: PixelFormat = metal(call, uint(call, 0)?)?;
            let mipmapped = arg(call, 3)?.as_bool();
            let mipmapped = mipmapped.ok_or_else(|| refuse(call, "argument 3 is no bool"))?;
            Ok(TextureDescriptor::texture_2d(
                format,
                uint(call, 1)?,
                uint(call, 2)?,
                mipmapped,
            ))
        }
        "m2v.textureDescriptor" => {
            let f = Fields::arg(call, 0, &descriptors::TEXTURE_KEYS)?;
            let mut descriptor = TextureDescriptor::default();
            descriptors::texture(call, &f, &mut descriptor)?;
            Ok(descriptor)
        }
        _ => Err(refuse(call, "makes no texture descriptor")),
    }
}

/// Texture descriptor `descriptor` changed by `call`, one of its setters.
fn set_texture_descriptor(
    call: &Call,
    descriptor: &mut TextureDescriptor,
) -> Result<(), PlayError> {
    let raw = uint(call, 0)?;
    match &*call.selector {
        "setUsage:" => descriptor.usage = metal::<TextureUsage>(call, raw)?,
        "setResourceOptions:" => descriptor.resource_options = options(call, raw)?,
        _ => return Err(refuse(call, "not played on a texture descriptor")),
    }
    Ok(())
}

impl Backend for OldRuntime {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        if &*call.selector == "release" {
            self.objects.remove(&call.receiver);
            return Ok(Made::default());
        }
        if self.skipped(call.receiver) {
            // A call on a skipped object; what it makes is skipped too.
            self.skips.calls += 1;
            if call.result != 0 {
                self.objects.insert(call.result, Object::Skipped);
            }
            return Ok(Made::default());
        }
        let encodes = matches!(
            self.objects.get(&call.receiver),
            Some(
                Object::CommandBuffer(_) | Object::Compute(_) | Object::Blit(_) | Object::Render(_)
            )
        );
        let names_skipped = call.args.iter().any(|value| self.holds_skipped(value));
        // No device here compiles Metal source: such a library is skipped, but for the host's
        // display library where the backend has it, and what is made of a skipped object with it.
        let display = &*call.selector == "newLibraryWithSource:options:error:"
            && call.args.is_empty()
            && self.display.is_some();
        if (&*call.selector == "newLibraryWithSource:options:error:" && !display)
            || (call.result != 0 && !encodes && names_skipped)
        {
            self.skips.objects += 1;
            self.objects.insert(call.result, Object::Skipped);
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
            Some(Object::Block(heap, _)) => Some(self.heap_call(call, heap, heap)?),
            Some(Object::Heap(heap, textures)) => {
                let textures = textures.as_ref().unwrap_or(heap);
                Some(self.heap_call(call, heap, textures)?)
            }
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

    fn heap_size_and_align(
        &mut self,
        call: &Call,
        descriptor: &[Call],
    ) -> Result<(u64, u64), PlayError> {
        let answer = match &*call.selector {
            "newTextureWithDescriptor:offset:" => {
                let (made, set) = descriptor
                    .split_first()
                    .ok_or_else(|| refuse(call, "a texture of no texture descriptor"))?;
                let mut descriptor = made_texture_descriptor(made)?;
                for setter in set {
                    set_texture_descriptor(setter, &mut descriptor)?;
                }
                let answer = self.device.heap_texture_size_and_align(&descriptor);
                answer.map_err(failed(call))?
            }
            "newBufferWithLength:options:offset:" => {
                let options = options(call, uint(call, 1)?)?;
                self.device
                    .heap_buffer_size_and_align(uint(call, 0)?, options)
            }
            _ => return Err(refuse(call, "places nothing in a heap")),
        };
        Ok((answer.size, answer.align))
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
        write_into(buffer, offset, bytes).map_err(|why| {
            PlayError::Invalid(format!(
                "m2v.write past #{target}'s CPU-visible bytes: {why}"
            ))
        })
    }

    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        match self.objects.get(&command_buffer) {
            Some(Object::CommandBuffer(waited)) => completed(waited, self.backend, "commit"),
            _ => Err(PlayError::Invalid(format!(
                "#{command_buffer} is no command buffer"
            ))),
        }
    }

    fn gpu_interval(&mut self, command_buffer: u64) -> Option<(u64, u64)> {
        let Some(Object::CommandBuffer(done)) = self.objects.get(&command_buffer) else {
            return None;
        };
        // Seconds of one host clock, as Metal's.
        let (start, end) = (done.gpu_start_time().ok()?, done.gpu_end_time().ok()?);
        let nanos = |seconds: f64| (seconds * 1e9) as u64;
        (start > 0.0 && end >= start).then(|| (nanos(start), nanos(end)))
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

    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError> {
        let id = match self.objects.get(&object) {
            Some(Object::Texture(texture)) => texture.gpu_resource_id(),
            Some(Object::Sampler(sampler)) => sampler.gpu_resource_id(),
            Some(Object::ComputePipeline(state)) => state.gpu_resource_id(),
            Some(Object::RenderPipeline(state)) => state.gpu_resource_id(),
            // Refused on a device that claims Apple10, where Metal's getter raises.
            Some(Object::DepthStencil(state)) => match state.gpu_resource_id() {
                Ok(id) => id,
                Err(_) => return Ok(None),
            },
            _ => {
                return Err(PlayError::Invalid(format!(
                    "gpuResourceID of #{object}, which has none"
                )))
            }
        };
        Ok(Some(id.0))
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
            (
                "compute_pipeline_create_nanos",
                counts.compute_pipeline_create_nanos,
            ),
            (
                "render_pipeline_create_nanos",
                counts.render_pipeline_create_nanos,
            ),
            ("shader_cache_read_nanos", counts.shader_cache_read_nanos),
            ("shader_cache_write_nanos", counts.shader_cache_write_nanos),
            ("device_memory_allocated", counts.device_memory_allocated),
            (
                "vk_memory_allocations",
                counts.object_churn.vk_memory_allocations,
            ),
            ("vk_memory_freed", counts.object_churn.vk_memory_freed),
            (
                "pooled_memory_blocks_made",
                counts.pooled_memory_blocks_made,
            ),
            (
                "pooled_memory_blocks_freed",
                counts.pooled_memory_blocks_freed,
            ),
            ("skipped_objects", self.skips.objects),
            ("skipped_draws", self.skips.draws),
            ("skipped_calls", self.skips.calls),
        ])
    }

    /// None: the old runtime's backend adds no lines.
    fn lines(&self) -> Vec<String> {
        let rule = match self.marked {
            true => "the stream's (header flag bit 0)",
            false => "the CPU copy's, read-backs included (no header flag)",
        };
        let mut lines = vec![format!(
            "announced bytes: {rule}; {} buffer read-backs left out of the CPU copies",
            self.read_backs_left_out
        )];
        if let Some(copies) = self.managed {
            let from = match self.identity {
                true => "the stream's device identity",
                false => "the player's setting",
            };
            lines.push(match copies {
                ManagedCopies::One => format!(
                    "managed one copy ({from}): {} Managed buffers and blocks played as Shared",
                    self.managed_shared.get()
                ),
                ManagedCopies::Two => format!("managed two copies ({from}): played as recorded"),
            });
        }
        lines
    }

    fn recorded_unified_memory(&mut self, unified: bool) {
        self.identity = true;
        self.managed = Some(match unified {
            true => ManagedCopies::One,
            false => ManagedCopies::Two,
        });
    }

    fn announced_bytes_recorded(&mut self) {
        self.marked = true;
    }

    /// The runtime's live memory blocks: pooled ones with a placement, empty ones kept (pooled or
    /// dedicated), and dedicated ones with their placement.
    fn live(&mut self) -> Counters {
        let counts = self.device.work_counts();
        Counters::from([
            ("memory_blocks", counts.memory_blocks_live),
            ("empty_memory_blocks", counts.empty_memory_blocks_live),
            (
                "dedicated_memory_blocks",
                counts.dedicated_memory_blocks_live,
            ),
        ])
    }

    fn snapshots(&self) -> bool {
        true
    }

    /// The old runtime takes a caller's address (`AddressPlacement::At`): every object with a
    /// recorded range is made at it (`address_placement`), every other one outside the space.
    fn places_at_recorded(&self) -> bool {
        true
    }

    /// The heap's buffers go in a heap at its recorded address and size, its textures in one
    /// outside the space: no address the program holds lies in a texture, and a heap the player
    /// grew for the runtime's texture sizes would hold pages past its recorded span.
    fn split_heap(&mut self, heap: u64, split: HeapSplit) {
        self.split.set(Some((heap, split)));
    }

    fn managed_texture(&self, texture: u64) -> Option<bool> {
        match self.objects.get(&texture) {
            Some(Object::Texture(texture)) => Some(texture.storage_mode() == StorageMode::Managed),
            _ => None,
        }
    }
}
