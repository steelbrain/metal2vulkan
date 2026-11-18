//! The player's backend over the new runtime (`metal2vulkan-rt`), through its public API on one
//! device. The backend is the runtime's consumer: it takes the blocks the stream's own buffers lie
//! in (`places`) and every texture and sampler index, granted at the start. A block of the stream
//! is a block of the runtime, and a heap is a block of its own. What the runtime does not build
//! yet is skipped and counted, never stood in for (`Skips`): render encoders and the states and
//! descriptors only they take, textures over buffers, and the blit commands it lacks; a
//! dispatch or blit that names a skipped object is skipped too. A play with any skip is no gate
//! result. A selector, an argument or a descriptor key this module does not name is refused.

mod encoders;
mod places;

use super::args::{arg, data, metal, option_sets, refuse, uint, variant, Fields};
use super::player::{Backend, Counters, Made, PlayError};
use super::stream::Call;
use super::translator;
use super::value::Value;
use metal2vulkan::pipeline::{LinkedFunction, LinkedFunctions};
use metal2vulkan_rt as rt;
use places::{class_index, Place, Places};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// What the play skipped because the runtime does not build it yet.
#[derive(Default)]
struct Skips {
    /// Encoders the runtime does not build: render encoders.
    encoders: u64,
    /// Objects it does not build: textures over buffers, texture views, render pipeline and
    /// depth-stencil states, render pass descriptors and libraries from source.
    objects: u64,
    /// Dispatches that bind a skipped object.
    dispatches: u64,
    /// Blit commands the runtime does not build, or that name a skipped object.
    blits: u64,
    /// Every other call on a skipped object or naming one, a skipped encoder's among them.
    calls: u64,
}

/// One object of the stream, as the runtime holds it.
enum Object {
    Device,
    Queue(rt::CommandQueue),
    CommandBuffer(rt::CommandBuffer),
    Compute(Box<encoders::Compute>),
    Blit(Box<rt::BlitCommandEncoder>),
    /// A buffer, and its place where the backend placed it.
    Buffer(rt::Buffer, Option<Place>),
    /// A block of the stream (`m2v.newBlockWithLength:storageMode:`) or a heap: its range.
    Block(rt::HeapRange, rt::MemoryClass),
    Texture(rt::Texture),
    TextureDescriptor(Box<rt::TextureDescriptor>),
    Library(rt::Library),
    Function(Box<Function>),
    ComputePipeline(rt::ComputePipelineState),
    Sampler(rt::SamplerState),
    Event(rt::Event),
    SharedEvent(rt::SharedEvent),
    Fence(rt::Fence),
    /// What the runtime does not build yet.
    Skipped,
}

/// A resource argument's object, `{Buffer: object}` or `{Texture: object}`, or a skipped one.
enum Held {
    Buffer(rt::Buffer),
    Texture(rt::Texture),
    Skipped,
}

/// A function: its descriptor, the name `newFunctionWithDescriptor:error:` specialized it
/// under, which a pipeline that links it takes, and its kernel.
#[derive(Clone)]
struct Function {
    descriptor: rt::FunctionDescriptor,
    specialized_name: Option<String>,
    kernel: Kernel,
}

/// A kernel apart from its specializations: the library object it came from and its name.
type Kernel = (u64, Arc<str>);

/// A compute pipeline state's making: its wall time, and whether its kernel was new to the play
/// (no state of another specialization of it made before).
struct Creation {
    nanos: u64,
    new_kernel: bool,
}

/// The new runtime on one device.
pub struct NewRuntime {
    device: rt::Device,
    objects: HashMap<u64, Object>,
    places: Places,
    /// The last index taken of each class: textures, storage-capable textures, samplers.
    last: [u32; 3],
    /// Each constexpr sampler's index, by its state words.
    constexprs: HashMap<[u64; 2], u32>,
    /// The commits so far, and the committed command buffers not yet seen complete, in order.
    commits: u64,
    in_flight: VecDeque<(u64, rt::CommandBuffer)>,
    /// The commits that have completed, every earlier one with them.
    completed: u64,
    skips: Skips,
    /// The compute pipeline states whose `gpuResourceID` the player asked, which the runtime does
    /// not give: a typed gap, counted.
    unanswered_pipeline_ids: u64,
    /// The bytes of the stream's live blocks and heaps, by class (`class_index`).
    block_bytes: [u64; 2],
    /// Each compute pipeline state made, in order, and the kernels made so far.
    creations: Vec<Creation>,
    kernels: HashSet<Kernel>,
}

/// A made object with no address.
fn unaddressed(object: Object) -> Result<Option<(Object, Made)>, PlayError> {
    Ok(Some((object, Made::default())))
}

/// A made buffer, at its address.
fn addressed(
    buffer: rt::Buffer,
    place: Option<Place>,
) -> Result<Option<(Object, Made)>, PlayError> {
    let made = Made {
        address: Some(buffer.gpu_address()),
    };
    Ok(Some((Object::Buffer(buffer, place), made)))
}

fn failed<'c>(call: &'c Call) -> impl Fn(rt::Error) -> PlayError + 'c {
    move |error| PlayError::Runtime {
        selector: call.selector.to_string(),
        error: error.to_string(),
    }
}

fn translated<'c, E: std::fmt::Display>(call: &'c Call) -> impl Fn(E) -> PlayError + 'c {
    move |error| PlayError::Runtime {
        selector: call.selector.to_string(),
        error: error.to_string(),
    }
}

/// The storage mode of resource options `raw`.
fn storage(call: &Call, raw: u64) -> Result<rt::StorageMode, PlayError> {
    match (raw >> 4) & 0xf {
        0 => Ok(rt::StorageMode::Shared),
        1 => Ok(rt::StorageMode::Managed),
        2 => Ok(rt::StorageMode::Private),
        3 => Ok(rt::StorageMode::Memoryless),
        other => Err(refuse(call, format!("storage mode {other}"))),
    }
}

/// The keys of a texture descriptor: `m2v.textureDescriptor`'s, and a drawable's.
const TEXTURE_KEYS: [&str; 19] = [
    "textureType",
    "pixelFormat",
    "width",
    "height",
    "depth",
    "mipmapLevelCount",
    "sampleCount",
    "arrayLength",
    "usage",
    "resourceOptions",
    "storageMode",
    "swizzle",
    "resourceIndex",
    "protectionOptions",
    "framebufferOnly",
    "isDrawable",
    "writeSwizzleEnabled",
    "allowGpuOptimizedContents",
    "rotation",
];

/// `f`'s properties set on `d`. The resource index is the guest's, which the player relocates
/// the IDs of; the GPU-optimized contents are a hint.
fn texture_fields(call: &Call, f: &Fields, d: &mut rt::TextureDescriptor) -> Result<(), PlayError> {
    if let Some(kind) = f.metal("textureType")? {
        d.texture_type = kind;
    }
    if let Some(format) = f.metal("pixelFormat")? {
        d.pixel_format = format;
    }
    let sizes: [(&str, &mut u32); 6] = [
        ("width", &mut d.width),
        ("height", &mut d.height),
        ("depth", &mut d.depth),
        ("mipmapLevelCount", &mut d.mipmap_level_count),
        ("sampleCount", &mut d.sample_count),
        ("arrayLength", &mut d.array_length),
    ];
    for (key, size) in sizes {
        if let Some(value) = f.count(key)? {
            *size = value;
        }
    }
    if let Some(usage) = f.metal("usage")? {
        d.usage = usage;
    }
    if let Some(raw) = f.uint("resourceOptions")? {
        d.storage_mode = storage(call, raw)?;
    }
    if let Some(mode) = f.uint("storageMode")? {
        d.storage_mode = storage(call, (mode & 0xf) << 4)?;
    }
    if let Some(s) = f.fields("swizzle", &["red", "green", "blue", "alpha"])? {
        d.swizzle = rt::TextureSwizzleChannels {
            red: s.required("red", Fields::metal)?,
            green: s.required("green", Fields::metal)?,
            blue: s.required("blue", Fields::metal)?,
            alpha: s.required("alpha", Fields::metal)?,
        };
    }
    f.only("protectionOptions", Fields::uint, 0)?;
    f.only("framebufferOnly", Fields::bool, false)?;
    f.only("isDrawable", Fields::bool, false)?;
    f.only("rotation", Fields::uint, 0)?;
    f.only("writeSwizzleEnabled", Fields::bool, false)?;
    Ok(())
}

/// The texture descriptor `call` makes: `m2v.textureDescriptor` or
/// `MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:`.
fn made_texture_descriptor(call: &Call) -> Result<rt::TextureDescriptor, PlayError> {
    match &*call.selector {
        "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:" => {
            let format: rt::PixelFormat = metal(call, uint(call, 0)?)?;
            let size = |at| {
                let raw = uint(call, at)?;
                u32::try_from(raw).map_err(|_| refuse(call, format!("argument {at} {raw}")))
            };
            let (width, height) = (size(1)?, size(2)?);
            let mipmapped = arg(call, 3)?.as_bool();
            let mipmapped = mipmapped.ok_or_else(|| refuse(call, "argument 3 is no bool"))?;
            let mut d = rt::TextureDescriptor::new(rt::TextureType::Type2D, format, width, height);
            if mipmapped {
                d.mipmap_level_count = 32 - width.max(height).leading_zeros();
            }
            Ok(d)
        }
        "m2v.textureDescriptor" => {
            let f = Fields::arg(call, 0, &TEXTURE_KEYS)?;
            let mut d =
                rt::TextureDescriptor::new(rt::TextureType::Type2D, rt::PixelFormat::Invalid, 1, 1);
            texture_fields(call, &f, &mut d)?;
            Ok(d)
        }
        _ => Err(refuse(call, "makes no texture descriptor")),
    }
}

/// Texture descriptor `d` changed by `call`, one of its setters.
fn set_texture_descriptor(call: &Call, d: &mut rt::TextureDescriptor) -> Result<(), PlayError> {
    let raw = uint(call, 0)?;
    match &*call.selector {
        "setUsage:" => d.usage = metal(call, raw)?,
        "setResourceOptions:" => d.storage_mode = storage(call, raw)?,
        _ => return Err(refuse(call, "not played on a texture descriptor")),
    }
    Ok(())
}

/// The class of a texture's index: none where no shader reaches it.
fn texture_class(usage: rt::TextureUsage) -> Option<rt::IndexClass> {
    if usage.is_empty() || usage.contains(rt::TextureUsage::SHADER_WRITE) {
        Some(rt::IndexClass::StorageTexture)
    } else if usage.contains(rt::TextureUsage::SHADER_READ) {
        Some(rt::IndexClass::Texture)
    } else {
        None
    }
}

impl NewRuntime {
    /// The backend on `device`: every texture and sampler index granted to it.
    pub fn new(device: rt::Device) -> Result<Self, rt::Error> {
        for class in [
            rt::IndexClass::Texture,
            rt::IndexClass::StorageTexture,
            rt::IndexClass::Sampler,
        ] {
            let range = class.range();
            device.grant_indices(class, range.start, range.end - range.start)?;
        }
        Ok(NewRuntime {
            places: Places::new(device.clone()),
            device,
            objects: HashMap::new(),
            last: [0; 3],
            constexprs: HashMap::new(),
            commits: 0,
            in_flight: VecDeque::new(),
            completed: 0,
            skips: Skips::default(),
            unanswered_pipeline_ids: 0,
            block_bytes: [0; 2],
            creations: Vec::new(),
            kernels: HashSet::new(),
        })
    }

    fn get(&self, call: &Call, id: u64) -> Result<&Object, PlayError> {
        self.objects.get(&id).ok_or_else(|| {
            PlayError::Invalid(format!("{}: #{id} is no live object", call.selector))
        })
    }

    /// The object argument `at` names.
    fn id(call: &Call, at: usize) -> Result<u64, PlayError> {
        let value = arg(call, at)?;
        value
            .as_object()
            .ok_or_else(|| refuse(call, format!("argument {at} is {value}, not an object")))
    }

    /// Whether object `id` is one the play skipped.
    fn skipped(&self, id: u64) -> bool {
        matches!(self.objects.get(&id), Some(Object::Skipped))
    }

    /// Whether any argument of `call` names an object the play skipped, in its lists and maps
    /// too.
    fn names_skipped(&self, call: &Call) -> bool {
        call.args.iter().any(|value| self.holds_skipped(value))
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

    /// Resource argument `at`: `{Buffer: object}` or `{Texture: object}`.
    fn resource(&self, call: &Call, at: usize) -> Result<Held, PlayError> {
        let (kind, object) = variant(call, arg(call, at)?)?;
        let id = object
            .as_object()
            .ok_or_else(|| refuse(call, format!("{kind} {object}")))?;
        if self.skipped(id) {
            return Ok(Held::Skipped);
        }
        match kind {
            "Buffer" => self.buffer_of(call, id).map(Held::Buffer),
            "Texture" => self.texture_of(call, id).map(Held::Texture),
            _ => Err(refuse(call, format!("a resource of kind {kind}"))),
        }
    }

    fn fence_of(&self, call: &Call, at: usize) -> Result<rt::Fence, PlayError> {
        let id = Self::id(call, at)?;
        match self.get(call, id)? {
            Object::Fence(fence) => Ok(fence.clone()),
            _ => Err(refuse(call, format!("#{id} is no fence"))),
        }
    }

    fn buffer_of(&self, call: &Call, id: u64) -> Result<rt::Buffer, PlayError> {
        match self.get(call, id)? {
            Object::Buffer(buffer, _) => Ok(buffer.clone()),
            _ => Err(refuse(call, format!("#{id} is no buffer"))),
        }
    }

    fn texture_of(&self, call: &Call, id: u64) -> Result<rt::Texture, PlayError> {
        match self.get(call, id)? {
            Object::Texture(texture) => Ok(texture.clone()),
            _ => Err(refuse(call, format!("#{id} is no texture"))),
        }
    }

    fn function_of(&self, call: &Call, id: u64) -> Result<Function, PlayError> {
        match self.get(call, id)? {
            Object::Function(function) => Ok((**function).clone()),
            _ => Err(refuse(call, format!("#{id} is no function"))),
        }
    }

    /// The commits that have completed, every earlier one with them.
    fn completed(&mut self) -> u64 {
        while let Some((at, front)) = self.in_flight.front() {
            match front.status() {
                rt::CommandBufferStatus::Completed | rt::CommandBufferStatus::Error => {
                    self.completed = *at;
                    self.in_flight.pop_front();
                }
                _ => break,
            }
        }
        self.completed
    }

    /// `make` at the next index of `class` that no live object holds.
    fn place<T>(
        &mut self,
        call: &Call,
        class: rt::IndexClass,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Result<T, PlayError> {
        let slot = match class {
            rt::IndexClass::Texture => 0,
            rt::IndexClass::StorageTexture => 1,
            _ => 2,
        };
        let range = class.range();
        for _ in range.clone() {
            let mut index = self.last[slot] + 1;
            if !range.contains(&index) {
                index = range.start;
            }
            self.last[slot] = index;
            if class == rt::IndexClass::Sampler && self.constexprs.values().any(|&i| i == index) {
                continue;
            }
            match make(index) {
                Err(rt::Error::Refused(rt::Refusal::IndexInUse { .. })) => continue,
                made => return made.map_err(failed(call)),
            }
        }
        Err(refuse(call, format!("every {class:?} index is in use")))
    }

    /// A buffer of `length` bytes and storage `mode` in a place of its own, zeroed, holding
    /// `bytes` from its start.
    fn new_buffer(
        &mut self,
        call: &Call,
        length: u64,
        mode: rt::StorageMode,
        bytes: &[u8],
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let class = match mode {
            rt::StorageMode::Private if bytes.is_empty() => rt::MemoryClass::Private,
            rt::StorageMode::Shared | rt::StorageMode::Managed => rt::MemoryClass::Shared,
            _ => return Err(refuse(call, format!("a {mode:?} buffer of this call"))),
        };
        let completed = self.completed();
        let place = (self.places.take(class, length, completed)).map_err(failed(call))?;
        let (start, block) = (place.range.start, place.block);
        let buffer = match mode {
            rt::StorageMode::Managed => {
                // The consumer's bytes first: the place may hold a released buffer's.
                let shared = self.device.new_buffer(block, start, length, true);
                let shared = shared.map_err(failed(call))?;
                write_into(&shared, 0, bytes)?;
                let copied = bytes.len() as u64;
                let managed = self.device.new_managed_buffer(block, start, length, copied);
                managed.map_err(failed(call))?
            }
            _ => {
                let buffer = self.device.new_buffer(block, start, length, true);
                let buffer = buffer.map_err(failed(call))?;
                write_into(&buffer, 0, bytes)?;
                buffer
            }
        };
        addressed(buffer, Some(place))
    }

    /// A texture of `descriptor`, at an index where shaders reach it.
    fn new_texture(
        &mut self,
        call: &Call,
        descriptor: &rt::TextureDescriptor,
        heap: Option<(rt::HeapRange, u64)>,
    ) -> Result<rt::Texture, PlayError> {
        let device = self.device.clone();
        let make = |index: Option<u32>| match heap {
            Some((range, offset)) => device.new_heap_texture(descriptor, index, range, offset),
            None => device.new_texture(descriptor, index),
        };
        match texture_class(descriptor.usage) {
            Some(class) => self.place(call, class, |index| make(Some(index))),
            None => make(None).map_err(failed(call)),
        }
    }

    /// The function `call`'s library makes.
    fn library_call(&self, call: &Call, library: &rt::Library) -> Result<Function, PlayError> {
        let name = arg(call, 0)?.as_str();
        let name = name.ok_or_else(|| refuse(call, "argument 0 is no name"))?;
        let kernel = (call.receiver, Arc::from(name));
        let function = library.function(name).map_err(translated(call))?;
        let mut descriptor = rt::FunctionDescriptor::new(function);
        let mut specialized_name = None;
        match &*call.selector {
            "newFunctionWithName:" => {}
            "newFunctionWithName:constantValues:error:" => {
                descriptor.constants = translator::constants(call, super::args::list(call, 1)?)?;
                descriptor.specialized = true;
            }
            "newFunctionWithDescriptor:error:" => {
                match arg(call, 3)? {
                    Value::Nil => {}
                    Value::Str(name) => specialized_name = Some(name.to_string()),
                    other => return Err(refuse(call, format!("specialized name {other}"))),
                }
                let constants = super::args::list(call, 1)?;
                if !constants.is_empty() {
                    descriptor.constants = translator::constants(call, constants)?;
                }
                descriptor.specialized = true;
                if uint(call, 2)? != 0 {
                    return Err(refuse(call, "function options"));
                }
            }
            _ => return Err(refuse(call, "not played on a library")),
        }
        Ok(Function {
            descriptor,
            specialized_name,
            kernel,
        })
    }

    /// The linked functions of a pipeline descriptor's `linked` map.
    fn linked(&self, call: &Call, f: Option<Fields>) -> Result<LinkedFunctions, PlayError> {
        let mut out = LinkedFunctions::new();
        let Some(f) = f else {
            return Ok(out);
        };
        for key in ["binaryFunctions", "groups"] {
            if !f.list(key)?.unwrap_or_default().is_empty() {
                return Err(refuse(call, format!("linked {key}")));
            }
        }
        let functions = |key: &str| -> Result<Vec<LinkedFunction>, PlayError> {
            let linked = |value: &Value| {
                let id = value
                    .as_object()
                    .ok_or_else(|| refuse(call, format!("{value} is no function")))?;
                let made = self.function_of(call, id)?;
                let mut function = LinkedFunction::new(made.descriptor.function.clone())
                    .with_constants(made.descriptor.constants.clone())
                    .with_specialized(made.descriptor.specialized);
                if let Some(name) = made.specialized_name {
                    function = function.with_specialized_name(name);
                }
                Ok(function)
            };
            f.list(key)?
                .unwrap_or_default()
                .iter()
                .map(linked)
                .collect()
        };
        for function in functions("functions")? {
            out = out.with_function(function);
        }
        for function in functions("privateFunctions")? {
            out = out.with_private_function(function);
        }
        Ok(out)
    }

    /// A compute pipeline state of `kernel`, its numbered constexpr samplers at indices of their
    /// own, the same words at one index.
    fn compute_pipeline_state(
        &mut self,
        call: &Call,
        descriptor: &rt::ComputePipelineDescriptor,
    ) -> Result<rt::ComputePipelineState, PlayError> {
        let mut taken = Vec::new();
        let made = {
            let (constexprs, last) = (&self.constexprs, self.last[2]);
            let sampler_ids = |samplers: &[rt::NumberedSampler]| {
                let mut next = last;
                let ids = samplers.iter().map(|sampler| match sampler {
                    rt::NumberedSampler::Constexpr(words) => match constexprs
                        .get(words)
                        .copied()
                        .or_else(|| taken.iter().find(|(w, _)| w == words).map(|&(_, i)| i))
                    {
                        Some(index) => index,
                        None => {
                            let range = rt::IndexClass::Sampler.range();
                            next = if range.contains(&(next + 1)) {
                                next + 1
                            } else {
                                range.start
                            };
                            taken.push((*words, next));
                            next
                        }
                    },
                    _ => 0,
                });
                Ok(ids.collect())
            };
            self.device
                .new_compute_pipeline_state(descriptor, sampler_ids)
        };
        for (words, index) in taken {
            self.constexprs.insert(words, index);
            self.last[2] = index;
        }
        made.map_err(failed(call))
    }

    /// A compute pipeline state of `descriptor`, of `kernel`'s, timed and counted (`creations`).
    fn timed_state(
        &mut self,
        call: &Call,
        descriptor: &rt::ComputePipelineDescriptor,
        kernel: Kernel,
    ) -> Result<rt::ComputePipelineState, PlayError> {
        let began = std::time::Instant::now();
        let state = self.compute_pipeline_state(call, descriptor)?;
        self.creations.push(Creation {
            nanos: u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX),
            new_kernel: self.kernels.insert(kernel),
        });
        Ok(state)
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`'s descriptor, and its
    /// kernel.
    fn compute_pipeline(
        &self,
        call: &Call,
    ) -> Result<(rt::ComputePipelineDescriptor, Kernel), PlayError> {
        let f = Fields::arg(
            call,
            0,
            &[
                "label",
                "function",
                "threadgroupSizeIsMultipleOfExecutionWidth",
                "maxTotalThreadsPerThreadgroup",
                "maxCallStackDepth",
                "supportIndirectCommandBuffers",
                "stageInput",
                "linked",
                "libraries",
                "openClMode",
            ],
        )?;
        let id = f
            .object("function")?
            .ok_or_else(|| refuse(call, "no function"))?;
        let linked = self.linked(
            call,
            f.fields(
                "linked",
                &["functions", "binaryFunctions", "groups", "privateFunctions"],
            )?,
        )?;
        let function = self.function_of(call, id)?;
        let kernel = function.descriptor.with_linked(linked);
        if f.get("stageInput").is_some() {
            return Err(refuse(call, "a stage input"));
        }
        if let Some(libraries) = f.fields("libraries", &["preloaded", "loaded"])? {
            for key in ["preloaded", "loaded"] {
                if !libraries.list(key)?.unwrap_or_default().is_empty() {
                    return Err(refuse(call, "dynamic libraries"));
                }
            }
        }
        f.only("openClMode", Fields::bool, false)?;
        let mut d = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(kernel));
        if let Some(multiple) = f.bool("threadgroupSizeIsMultipleOfExecutionWidth")? {
            d.threadgroup_size_is_multiple_of_execution_width = multiple;
        }
        if let Some(most) = f.count("maxTotalThreadsPerThreadgroup")? {
            d.max_total_threads_per_threadgroup = (most != 0).then_some(most);
        }
        if let Some(depth) = f.count("maxCallStackDepth")? {
            d.max_call_stack_depth = depth;
        }
        if let Some(support) = f.bool("supportIndirectCommandBuffers")? {
            d.support_indirect_command_buffers = support;
        }
        Ok((d, function.kernel))
    }

    /// A call of the platform (receiver 0) or the device, and what it made.
    fn device_call(&mut self, call: &Call) -> Result<Option<(Object, Made)>, PlayError> {
        match &*call.selector {
            "MTLCreateSystemDefaultDevice" => unaddressed(Object::Device),
            "newCommandQueue" => {
                let queue = self.device.new_command_queue(64).map_err(failed(call))?;
                unaddressed(Object::Queue(queue))
            }
            "newBufferWithLength:options:" => {
                let mode = storage(call, uint(call, 1)?)?;
                self.new_buffer(call, uint(call, 0)?, mode, &[])
            }
            "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                let mode = storage(call, uint(call, 2)?)?;
                self.new_buffer(call, uint(call, 1)?, mode, &[])
            }
            "newBufferWithBytes:length:options:" => {
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| data(call, 0).ok()?.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let mode = storage(call, uint(call, 2)?)?;
                self.new_buffer(call, length, mode, bytes)
            }
            "newBufferWithDescriptor:" => {
                let f = Fields::arg(
                    call,
                    0,
                    &[
                        "length",
                        "options",
                        "pinnedGpuAddress",
                        "contents",
                        "noCopy",
                        "deallocator",
                        "zeroFrom",
                    ],
                )?;
                if f.get("contents").is_some() {
                    return Err(refuse(call, "contents that are no bytes"));
                }
                let raw = option_sets(call, 0, f.list("options")?.unwrap_or_default())?;
                let length = f.required("length", Fields::uint)?;
                self.new_buffer(call, length, storage(call, raw)?, &[])
            }
            "m2v.newBlockWithLength:storageMode:" | "newHeapWithDescriptor:" => {
                let (length, class) = match &*call.selector {
                    "newHeapWithDescriptor:" => {
                        let f = Fields::arg(
                            call,
                            0,
                            &["size", "heapType", "options", "pinnedGpuAddress"],
                        )?;
                        f.only("heapType", Fields::uint, 1)?;
                        let raw =
                            option_sets(call, 2 << 4, f.list("options")?.unwrap_or_default())?;
                        (f.required("size", Fields::uint)?, (raw >> 4) & 0xf)
                    }
                    _ => (uint(call, 0)?, uint(call, 1)?),
                };
                let class = match class {
                    0 | 1 => rt::MemoryClass::Shared,
                    2 => rt::MemoryClass::Private,
                    other => return Err(refuse(call, format!("a block of storage mode {other}"))),
                };
                let info = self
                    .device
                    .take_block(class, length)
                    .map_err(failed(call))?;
                self.block_bytes[class_index(class)] += length;
                let made = Made {
                    address: Some(info.gpu_address),
                };
                let range = rt::HeapRange {
                    block: info.id,
                    offset: 0,
                    len: length,
                };
                Ok(Some((Object::Block(range, class), made)))
            }
            "newLibraryWithData:error:" => {
                let library = rt::Library::from_metallib(data(call, 0)?);
                unaddressed(Object::Library(library.map_err(translated(call))?))
            }
            "newComputePipelineStateWithFunction:error:" => {
                let function = self.function_of(call, Self::id(call, 0)?)?;
                let d = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(
                    function.descriptor,
                ));
                let state = self.timed_state(call, &d, function.kernel)?;
                unaddressed(Object::ComputePipeline(state))
            }
            "newComputePipelineStateWithDescriptor:options:reflection:error:" => {
                let (d, kernel) = self.compute_pipeline(call)?;
                let state = self.timed_state(call, &d, kernel)?;
                unaddressed(Object::ComputePipeline(state))
            }
            // Render pipeline and depth-stencil states, render pass descriptors and libraries from
            // source: the runtime builds none yet.
            "newRenderPipelineStateWithDescriptor:error:"
            | "newDepthStencilStateWithDescriptor:"
            | "MTLRenderPassDescriptor.new"
            | "MTLRenderPassDescriptor.renderPassDescriptor"
            | "newLibraryWithSource:options:error:" => self.skip_object(),
            "newFence" => unaddressed(Object::Fence(self.device.new_fence())),
            "newSamplerStateWithDescriptor:" => {
                let (mut descriptor, f) = translator::sampler(call)?;
                if let Some(color) = f.metal("borderColor")? {
                    descriptor.border_color = color;
                }
                let device = self.device.clone();
                let state = self.place(call, rt::IndexClass::Sampler, |index| {
                    device.new_sampler_state(&descriptor, index)
                })?;
                unaddressed(Object::Sampler(state))
            }
            "newTextureWithDescriptor:" => {
                let descriptor = match arg(call, 0)? {
                    // A drawable the stream never saw made: a texture of its descriptor.
                    map @ Value::Map(_) => {
                        let f = Fields::of(call, map, &TEXTURE_KEYS)?;
                        let mut d = rt::TextureDescriptor::new(
                            rt::TextureType::Type2D,
                            rt::PixelFormat::Invalid,
                            1,
                            1,
                        );
                        texture_fields(call, &f, &mut d)?;
                        d
                    }
                    _ => self.texture_descriptor(call, 0)?,
                };
                let texture = self.new_texture(call, &descriptor, None)?;
                unaddressed(Object::Texture(texture))
            }
            "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:"
            | "m2v.textureDescriptor" => {
                let descriptor = made_texture_descriptor(call)?;
                unaddressed(Object::TextureDescriptor(Box::new(descriptor)))
            }
            // Every index is the backend's already.
            "reserveResourceIndicesForResourceType:indices:indexCount:" => match uint(call, 0)? {
                0 | 1 => Ok(None),
                other => Err(refuse(call, format!("resource type {other}"))),
            },
            "newEvent" => {
                let event = self.device.new_event().map_err(failed(call))?;
                unaddressed(Object::Event(event))
            }
            "newSharedEvent" => {
                let event = self.device.new_shared_event().map_err(failed(call))?;
                unaddressed(Object::SharedEvent(event))
            }
            _ => Err(refuse(call, "not played on a device")),
        }
    }

    fn skip_object(&mut self) -> Result<Option<(Object, Made)>, PlayError> {
        self.skips.objects += 1;
        unaddressed(Object::Skipped)
    }

    /// The texture descriptor object argument `at` names, as it is now.
    fn texture_descriptor(
        &self,
        call: &Call,
        at: usize,
    ) -> Result<rt::TextureDescriptor, PlayError> {
        let id = Self::id(call, at)?;
        match self.get(call, id)? {
            Object::TextureDescriptor(descriptor) => Ok(**descriptor),
            _ => Err(refuse(call, format!("#{id} is no texture descriptor"))),
        }
    }

    /// A call of a block or a heap: the buffer or texture it places.
    fn block_call(
        &mut self,
        call: &Call,
        range: rt::HeapRange,
        class: rt::MemoryClass,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        // A buffer of `length` bytes and storage options `raw` at `offset` in the block.
        let device = self.device.clone();
        let placed = |length: u64, raw: u64, offset: u64| {
            let at = range.offset + offset;
            let buffer = match storage(call, raw)? {
                rt::StorageMode::Managed if class == rt::MemoryClass::Shared => {
                    device.new_managed_buffer(range.block, at, length, length)
                }
                _ => device.new_buffer(range.block, at, length, false),
            };
            addressed(buffer.map_err(failed(call))?, None)
        };
        match &*call.selector {
            "newBufferWithLength:options:offset:" => {
                placed(uint(call, 0)?, uint(call, 1)?, uint(call, 2)?)
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                if Self::id(call, 0)? != call.receiver {
                    return Err(refuse(call, "a block other than its receiver"));
                }
                placed(uint(call, 2)?, uint(call, 3)?, uint(call, 1)?)
            }
            "newTextureWithDescriptor:offset:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                let texture = self.new_texture(call, &descriptor, Some((range, uint(call, 1)?)))?;
                unaddressed(Object::Texture(texture))
            }
            _ => Err(refuse(call, "not played on a heap")),
        }
    }

    /// A call of buffer `buffer`.
    fn buffer_call(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        match &*call.selector {
            "didModifyRange:" => {
                let range = super::args::range(call, arg(call, 0)?)?;
                let did = buffer.did_modify_range(range.start, range.end - range.start);
                did.map_err(failed(call))?;
                Ok(None)
            }
            "newTextureWithDescriptor:offset:bytesPerRow:"
            | "newTiledTextureWithDescriptor:offset:bytesPerRow:"
            | "newTextureWithDescriptor:iosurface:plane:" => self.skip_object(),
            _ => Err(refuse(call, "not played on a buffer")),
        }
    }

    /// A call of texture `texture`, and the view it made.
    fn texture_call(
        &mut self,
        call: &Call,
        texture: &rt::Texture,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let region = |at| encoders::region(call, arg(call, at)?);
        let replaced = match &*call.selector {
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:" => texture
                .replace_region(
                    region(0)?,
                    super::args::index(call, 1)?,
                    super::args::index(call, 2)?,
                    data(call, 3)?,
                    uint(call, 4)?,
                    uint(call, 5)?,
                ),
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:" => texture.replace_region(
                region(0)?,
                super::args::index(call, 1)?,
                0,
                data(call, 2)?,
                uint(call, 3)?,
                0,
            ),
            // The runtime makes no texture views yet.
            "newTextureViewWithPixelFormat:"
            | "newTextureViewWithPixelFormat:resourceIndex:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:" => {
                return self.skip_object()
            }
            _ => return Err(refuse(call, "not played on a texture")),
        };
        replaced.map(|()| None).map_err(failed(call))
    }
}

/// `bytes` into `buffer` at `offset`, through its mapping.
fn write_into(buffer: &rt::Buffer, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let fits = offset
        .checked_add(bytes.len() as u64)
        .is_some_and(|end| end <= buffer.length());
    let contents = buffer.contents().filter(|_| fits).ok_or_else(|| {
        PlayError::Invalid("a write past a buffer's CPU-visible bytes".to_owned())
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

impl Backend for NewRuntime {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        if &*call.selector == "release" {
            match self.objects.remove(&call.receiver) {
                Some(Object::Buffer(_, Some(place))) => self.places.give(place, self.commits),
                // The runtime keeps the block while a buffer of it lives or work uses it, as a
                // heap's resources keep the heap.
                Some(Object::Block(range, class)) => {
                    self.device
                        .release_block(range.block)
                        .map_err(failed(call))?;
                    self.block_bytes[class_index(class)] -= range.len;
                }
                _ => {}
            }
            return Ok(Made::default());
        }
        if call.receiver != 0 && self.skipped(call.receiver) {
            // A call on a skipped object; what it makes is skipped too.
            self.skips.calls += 1;
            if call.result != 0 {
                self.objects.insert(call.result, Object::Skipped);
            }
            return Ok(Made::default());
        }
        let encodes = matches!(
            self.objects.get(&call.receiver),
            Some(Object::CommandBuffer(_) | Object::Compute(_) | Object::Blit(_))
        );
        if call.result != 0 && !encodes && self.names_skipped(call) {
            // What a call makes of a skipped object is skipped too.
            self.skips.objects += 1;
            self.objects.insert(call.result, Object::Skipped);
            return Ok(Made::default());
        }
        let made = match call.receiver {
            0 => self.device_call(call)?,
            id => match self.get(call, id)? {
                Object::Device => self.device_call(call)?,
                Object::Library(library) => {
                    let library = library.clone();
                    let function = self.library_call(call, &library)?;
                    unaddressed(Object::Function(Box::new(function)))?
                }
                Object::Queue(queue) => match &*call.selector {
                    "commandBuffer" => {
                        let command_buffer = queue.command_buffer().map_err(failed(call))?;
                        unaddressed(Object::CommandBuffer(command_buffer))?
                    }
                    _ => return Err(refuse(call, "not played on a command queue")),
                },
                Object::CommandBuffer(_) => self.command_buffer_call(call)?,
                Object::Compute(_) => {
                    self.compute_call(call)?;
                    None
                }
                Object::Blit(_) => {
                    self.blit_call(call)?;
                    None
                }
                Object::Buffer(buffer, _) => {
                    let buffer = buffer.clone();
                    self.buffer_call(call, &buffer)?
                }
                Object::Block(range, class) => {
                    let (range, class) = (*range, *class);
                    self.block_call(call, range, class)?
                }
                Object::Texture(texture) => {
                    let texture = texture.clone();
                    self.texture_call(call, &texture)?
                }
                Object::TextureDescriptor(_) => {
                    let Some(Object::TextureDescriptor(descriptor)) = self.objects.get_mut(&id)
                    else {
                        unreachable!("a texture descriptor's call");
                    };
                    set_texture_descriptor(call, descriptor)?;
                    None
                }
                Object::SharedEvent(event) => match &*call.selector {
                    "setSignaledValue:" => {
                        event.set_signaled_value(uint(call, 0)?);
                        None
                    }
                    _ => return Err(refuse(call, "not played on a shared event")),
                },
                _ => return Err(refuse(call, "not played on its receiver")),
            },
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
        match &*call.selector {
            "newTextureWithDescriptor:offset:" => {
                let (made, set) = descriptor
                    .split_first()
                    .ok_or_else(|| refuse(call, "a texture of no texture descriptor"))?;
                let mut descriptor = made_texture_descriptor(made)?;
                for setter in set {
                    set_texture_descriptor(setter, &mut descriptor)?;
                }
                let answer = self.device.heap_texture_size_and_align(&descriptor);
                let answer = answer.map_err(failed(call))?;
                Ok((answer.size, answer.align))
            }
            // A buffer lies at Metal's 256 bytes in a block, as the runtime places it anywhere.
            "newBufferWithLength:options:offset:" => {
                Ok((uint(call, 0)?.next_multiple_of(256), 256))
            }
            _ => Err(refuse(call, "places nothing in a heap")),
        }
    }

    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
        match self.objects.get(&target) {
            Some(Object::Buffer(buffer, _)) => write_into(buffer, offset, bytes),
            Some(Object::Block(range, rt::MemoryClass::Shared)) => {
                let whole = self
                    .device
                    .new_buffer(range.block, range.offset, range.len, false);
                let whole = whole.map_err(|error| PlayError::Runtime {
                    selector: "m2v.write".into(),
                    error: error.to_string(),
                })?;
                write_into(&whole, offset, bytes)
            }
            _ => Err(PlayError::Invalid(format!(
                "m2v.write to #{target}, which the CPU cannot write"
            ))),
        }
    }

    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        match self.objects.get(&command_buffer) {
            Some(Object::CommandBuffer(waited)) => {
                waited.wait_until_completed();
                match waited.error() {
                    None => Ok(()),
                    Some(error) => Err(PlayError::Runtime {
                        selector: "commit".into(),
                        error: format!("{error:?}"),
                    }),
                }
            }
            _ => Err(PlayError::Invalid(format!(
                "#{command_buffer} is no command buffer"
            ))),
        }
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        let unread = |why: &str| PlayError::Refused {
            selector: "m2v.check".into(),
            why: why.into(),
        };
        match self.objects.get(&object) {
            Some(Object::Buffer(buffer, _)) => {
                let contents = buffer
                    .contents()
                    .ok_or_else(|| unread("a Private buffer"))?;
                let length = usize::try_from(buffer.length()).map_err(|_| unread("its length"))?;
                // SAFETY: the buffer's mapping holds its length, and the player waited for the
                // work that writes it.
                Ok(unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec())
            }
            _ => Err(unread(
                "no buffer, or not yet one the new runtime reads back",
            )),
        }
    }

    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError> {
        let id = match self.objects.get(&object) {
            Some(Object::Texture(texture)) => texture.gpu_resource_id(),
            Some(Object::Sampler(sampler)) => sampler.gpu_resource_id(),
            // The runtime gives a compute pipeline state no resource ID: it builds no indirect
            // command buffers. The player reports the ID as differing, and the gap is counted.
            Some(Object::ComputePipeline(_)) => {
                self.unanswered_pipeline_ids += 1;
                return Ok(None);
            }
            Some(Object::Skipped) => return Ok(None),
            _ => {
                return Err(PlayError::Invalid(format!(
                    "gpuResourceID of #{object}, which has none"
                )))
            }
        };
        Ok(Some(id.to_bits()))
    }

    fn counters(&mut self) -> Counters {
        let metrics = self.device.counters().metrics();
        // Every metric that sums: a frame's is the difference of two reads.
        Counters::from([
            ("submissions", metrics.submissions),
            ("passes", metrics.passes),
            ("barriers", metrics.barriers),
            ("bytes_copied", metrics.bytes_copied),
            ("pipelines_created", metrics.pipelines_created),
            ("first_use_waits", metrics.first_use_waits),
            ("first_use_wait_nanos", metrics.first_use_wait_nanos),
            ("blocks_made", metrics.blocks_made),
            ("dedicated_blocks_reused", metrics.dedicated_blocks_reused),
            (
                "dispatches_without_effects",
                metrics.dispatches_without_effects,
            ),
            ("translations", metrics.translations),
            ("translations_loaded", metrics.translations_loaded),
            ("ring_writes", metrics.ring_writes),
            ("upload_waits", metrics.upload_waits),
            ("synchronizations", metrics.synchronizations),
            ("skipped_encoders", self.skips.encoders),
            ("skipped_objects", self.skips.objects),
            ("skipped_dispatches", self.skips.dispatches),
            ("skipped_blits", self.skips.blits),
            ("skipped_calls", self.skips.calls),
            ("gap_pipeline_ids", self.unanswered_pipeline_ids),
        ])
    }

    /// The runtime's gauges, and the bytes of the blocks the backend holds by class: the stream's
    /// blocks and heaps, and the blocks it places the stream's other buffers in.
    fn live(&mut self) -> Counters {
        let gauges = self.device.counters().gauges();
        let placed = self.places.bytes();
        let held = |class: rt::MemoryClass| {
            let at = class_index(class);
            self.block_bytes[at] + placed[at]
        };
        Counters::from([
            ("blocks", gauges.blocks),
            ("buffers", gauges.buffers),
            ("textures", gauges.textures),
            ("pipelines", gauges.pipelines),
            ("kernels", gauges.kernels),
            ("command_buffers", gauges.command_buffers),
            ("spilled_bytes", gauges.spilled_bytes),
            ("shared_idle_bytes", gauges.shared_idle_bytes),
            ("private_idle_bytes", gauges.private_idle_bytes),
            ("shared_block_bytes", held(rt::MemoryClass::Shared)),
            ("private_block_bytes", held(rt::MemoryClass::Private)),
        ])
    }

    /// A `pipeline I wall_nanos=N kernel=new|again` line for each compute pipeline state made, in
    /// order: its making's wall time, and whether its kernel was new to the play.
    fn lines(&self) -> Vec<String> {
        let lines = self.creations.iter().enumerate().map(|(at, made)| {
            let kernel = if made.new_kernel { "new" } else { "again" };
            format!("pipeline {at} wall_nanos={} kernel={kernel}", made.nanos)
        });
        lines.collect()
    }

    /// The runtime makes no blits between buffers yet.
    fn snapshots(&self) -> bool {
        false
    }
}
