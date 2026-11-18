//! The player's backend over the new runtime (`metal2vulkan-rt`), through its public API on one
//! device. The backend is the runtime's consumer: it takes the blocks the stream's own buffers lie
//! in (`places`) and every texture and sampler index, granted at the start. A block of the stream
//! is a block of the runtime, and a heap is a block of its own. What the runtime does not build
//! yet is skipped and counted, never stood in for (`Skips`): the render pipeline states, render
//! encoders, draws and other render commands it does not build (`render`), and the blit commands
//! it lacks; a dispatch, draw or blit that names a skipped object is skipped too. A dispatch is
//! never skipped otherwise. A play with any skip is no gate result past the first
//! (`Report::first_skip`). A selector, an argument or a descriptor key this module does not name
//! is refused.

mod aside;
mod encoders;
pub(super) mod indices;
pub(super) mod places;
mod render;
pub(super) mod second;
pub(super) mod unwritten;

use super::args::{arg, data, metal, option_sets, refuse, uint, variant, Fields};
use super::player::{Backend, Counters, Made, PlayError};
use super::stream::Call;
use super::translator;
use super::value::Value;
use metal2vulkan::pipeline::{LinkedFunction, LinkedFunctions};
use metal2vulkan_rt as rt;
use places::{class_index, Place, Places};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// What the play skipped because the runtime does not build it yet.
#[derive(Default)]
struct Skips {
    /// Encoders the runtime does not build: parallel render encoders, and render encoders of a
    /// pass it does not build (`render`).
    encoders: u64,
    /// Objects it does not build: libraries from source.
    objects: u64,
    /// Render pipeline states the runtime answers with a typed gap or the translator refuses.
    render_states: u64,
    /// The first such state's object id and the runtime's answer, for the report's lines.
    first_render_state: Option<(u64, String)>,
    /// Dispatches that bind a skipped object.
    dispatches: u64,
    /// Draws whose pipeline state, or a buffer or texture bound to either stage, is a skipped
    /// object.
    draws: u64,
    /// Render commands the runtime answers with a typed gap (a draw it does not build), by the
    /// gap: how many, and the frame and call of the first.
    render_commands: BTreeMap<String, (u64, u64, u64)>,
    /// Blit commands that name a skipped object.
    blits: u64,
    /// Pieces of the host's copies from the guest's RAM that the play cannot copy: a piece not of
    /// whole 4-byte words at a 4-byte offset into a buffer with no mapping to fill the rest from.
    guest_ram_writes: u64,
    /// The host's copies back to the guest's RAM: nothing in a play reads that RAM.
    guest_ram_reads: u64,
    /// `synchronizeTexture:slice:level:` and `generateMipmapsForTexture:`, which the runtime
    /// does not build yet.
    texture_syncs: u64,
    mipmaps: u64,
    /// Every other call on a skipped object or naming one, a skipped encoder's among them.
    calls: u64,
    /// The skipped objects whose `gpuResourceID` the player asked: nothing in the play reads
    /// their IDs, so none is compared.
    resource_ids: u64,
}

impl Skips {
    /// The render commands skipped, of every gap.
    fn render_command_count(&self) -> u64 {
        self.render_commands
            .values()
            .map(|&(count, ..)| count)
            .sum()
    }

    /// Everything skipped, in one count (`Backend::skips`).
    fn total(&self) -> u64 {
        let Skips {
            encoders,
            objects,
            render_states,
            first_render_state: _,
            dispatches,
            draws,
            render_commands: _,
            blits,
            guest_ram_writes,
            guest_ram_reads,
            texture_syncs,
            mipmaps,
            calls,
            resource_ids,
        } = self;
        let counted = [
            encoders,
            objects,
            render_states,
            dispatches,
            draws,
            blits,
            guest_ram_writes,
            guest_ram_reads,
            texture_syncs,
            mipmaps,
            calls,
            resource_ids,
        ];
        counted.into_iter().sum::<u64>() + self.render_command_count()
    }
}

/// Textures the play made in place of what the runtime cannot make: an ordinary texture of the
/// same descriptor, so that the IDs and addresses the program wrote name a live texture. The
/// recording's artefact, never the runtime's: a guest's surface texture is its host's import.
#[derive(Default)]
struct Substitutes {
    /// Textures over an IOSurface (`newTextureWithDescriptor:iosurface:plane:`).
    surfaces: u64,
    /// Tiled textures over a buffer (`newTiledTextureWithDescriptor:offset:bytesPerRow:`).
    tiled: u64,
    /// Textures over a buffer and views the runtime answers with a typed gap.
    gaps: u64,
}

/// One object of the stream, as the runtime holds it.
enum Object {
    Device,
    Queue(rt::CommandQueue),
    CommandBuffer(rt::CommandBuffer),
    Compute(Box<encoders::Compute>),
    Blit(Box<rt::BlitCommandEncoder>),
    Render(Box<render::Render>),
    /// A buffer, and its place where the backend placed it.
    Buffer(rt::Buffer, Option<Place>),
    /// A block of the stream (`m2v.newBlockWithLength:storageMode:`) or a heap: its range, and
    /// the handle that keeps the block while the object lives.
    Block(rt::HeapRange, rt::MemoryClass, rt::Block),
    Texture(rt::Texture),
    TextureDescriptor(Box<rt::TextureDescriptor>),
    Library(rt::Library),
    Function(Box<Function>),
    ComputePipeline(rt::ComputePipelineState),
    RenderPipeline(rt::RenderPipelineState),
    DepthStencil(rt::DepthStencilState),
    Pass(Box<render::Pass>),
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

/// How the device that recorded a stream kept a Managed buffer: Metal's `hasUnifiedMemory`
/// there. With unified memory a Managed buffer is one copy, so a CPU write needs no
/// `didModifyRange:`; without it the GPU has a copy of its own, made from the buffer's bytes at
/// its creation and changed after only by `didModifyRange:`. A stream does not record it: the
/// player is told it for each stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManagedCopies {
    /// Recorded with unified memory: every Managed buffer plays as a Shared one, on every device.
    One,
    /// Recorded without it: a Managed buffer plays as the runtime's Managed buffer. A no-copy
    /// one's GPU copy takes the bytes the stream records before its first use, its nearest record
    /// of its bytes at creation.
    Two,
}

/// The new runtime on one device.
pub struct NewRuntime {
    device: rt::Device,
    objects: HashMap<u64, Object>,
    places: Places,
    /// The indices textures and sampler states hold and the program reserved.
    indices: indices::Indices,
    /// The index the call being played names for the object it makes (a resource index).
    named: Option<u32>,
    /// The object the next call makes and the ID the program saw of it, where it saw one
    /// (`Backend::recorded_id`).
    recorded: Option<(u64, u64)>,
    /// Each live texture descriptor's resource index, where its record sets one.
    resource_indices: HashMap<u64, u32>,
    /// The texture descriptors the stream marks `outside` (section 5 of docs/REPLAY-STREAM.md):
    /// a texture of one takes an index outside those the program's objects take.
    outside: HashSet<u64>,
    /// Each constexpr sampler's index, by its state words.
    constexprs: HashMap<[u64; 2], u32>,
    /// The commits so far, and the committed command buffers not yet seen retired, in order, each
    /// with the flag its completed handler sets ([`commit`](Self::commit)).
    commits: u64,
    in_flight: VecDeque<(u64, rt::CommandBuffer, Arc<AtomicBool>)>,
    /// The commits that have completed, every earlier one with them.
    completed: u64,
    skips: Skips,
    /// The frame and the stream position of the call being played (`Backend::locate`).
    at: (u64, u64),
    substitutes: Substitutes,
    /// Each open encoder's command buffer.
    owners: HashMap<u64, u64>,
    /// How the recording's device kept Managed buffers; `None` where the player was not told,
    /// and a Managed buffer is refused.
    managed: Option<ManagedCopies>,
    /// The Managed buffers made.
    managed_buffers: u64,
    /// The no-copy Managed buffers of a two-copy stream whose first use has not been committed,
    /// with the command buffer that first uses one: the stream's writes into such a buffer go to
    /// its GPU copy too.
    fresh: HashMap<u64, Option<u64>>,
    /// The no-copy Managed buffers whose GPU copy took bytes written before their first use.
    from_first_use: HashSet<u64>,
    /// Whether the stream told the form itself (`m2v.deviceIdentity`): its no-copy Managed
    /// buffers then carry their bytes at creation.
    identity: bool,
    /// The Shared buffers a dispatch read before anything recorded reached their bytes.
    unwritten: unwritten::Unwritten,
    /// The staging places of each command buffer not yet committed, which hold the bytes of its
    /// guest-RAM copies: given back at its commit, free once it completes.
    staging: HashMap<u64, Vec<Place>>,
    /// The buffers in a place of the play's own, textures and sampler states that an encoder of
    /// each command buffer not yet committed named: a release of a buffer waits for those command
    /// buffers (`Places::give`), and a placement at a texture's or sampler state's index for
    /// their commits (`Indices::owed`).
    users: HashMap<u64, HashSet<u64>>,
    /// The CPU copies of a two-copy stream's Managed buffers on a one-copy device
    /// (`second`).
    seconds: second::Seconds,
    /// The read-backs of a stream that records its announced bytes, kept out of the CPU copies
    /// (`aside`).
    asides: aside::Asides,
    /// The places of the GPU copies of the Managed buffers in a place of the play's own, on a
    /// device that keeps a GPU copy: given back with the buffer's place.
    device_copies: HashMap<u64, Place>,
    /// The queue each command buffer was made on.
    queue_of: HashMap<u64, u64>,
    /// The bytes those copies staged.
    staged_bytes: u64,
    /// The compute pipeline states whose `gpuResourceID` the player asked, which the runtime does
    /// not give: a typed gap, counted.
    unanswered_pipeline_ids: u64,
    /// The bytes of the stream's live blocks and heaps, by class (`class_index`).
    block_bytes: [u64; 2],
    /// Each compute pipeline state made, in order, and the kernels made so far.
    creations: Vec<Creation>,
    kernels: HashSet<Kernel>,
}

/// The objects `values` name, lists included.
fn objects_named(values: &[Value], out: &mut Vec<u64>) {
    for value in values {
        match value {
            Value::Object(id) => out.push(*id),
            Value::List(items) => objects_named(items, out),
            _ => {}
        }
    }
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
const TEXTURE_KEYS: [&str; 20] = [
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
    "outside",
];

/// `f`'s properties set on `d`. The resource index places the texture (`named_index`); the
/// GPU-optimized contents are a hint.
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

/// The index of the resource index `call` sets on the texture descriptor it makes, where it sets
/// one.
fn resource_index(call: &Call) -> Result<Option<u32>, PlayError> {
    match &*call.selector {
        "m2v.textureDescriptor" => resource_index_of(call, &Fields::arg(call, 0, &TEXTURE_KEYS)?),
        _ => Ok(None),
    }
}

/// The index of a texture descriptor's `resourceIndex` (`[index, forced]`): Metal places the
/// texture there whether or not it is forced.
fn resource_index_of(call: &Call, f: &Fields) -> Result<Option<u32>, PlayError> {
    let Some(list) = f.list("resourceIndex")? else {
        return Ok(None);
    };
    match list {
        [index, Value::Bool(_)] => match index.as_uint().map(u32::try_from) {
            Some(Ok(index)) => Ok(Some(index)),
            _ => Err(refuse(call, format!("resource index {index}"))),
        },
        _ => Err(refuse(call, "a resource index that is no [index, forced]")),
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
/// Every slice of a texture of `d`, as a view counts them: a cube's six faces each one.
fn whole_slices(d: &rt::TextureDescriptor) -> Range<u32> {
    let faces = match d.texture_type {
        rt::TextureType::Cube | rt::TextureType::CubeArray => 6,
        _ => 1,
    };
    match d.texture_type {
        rt::TextureType::Type3D => 0..1,
        _ => 0..d.array_length.max(1) * faces,
    }
}

fn texture_class(usage: rt::TextureUsage) -> Option<rt::IndexClass> {
    let reached = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
    (usage.is_empty() || usage.intersects(reached)).then_some(rt::IndexClass::Texture)
}

impl NewRuntime {
    /// The backend on `device`: every texture and sampler index granted to it.
    pub fn new(device: rt::Device) -> Result<Self, rt::Error> {
        for class in [rt::IndexClass::Texture, rt::IndexClass::Sampler] {
            let range = class.range();
            device.grant_indices(class, range.start, range.end - range.start)?;
        }
        Ok(NewRuntime {
            places: Places::new(device.clone()),
            device,
            objects: HashMap::new(),
            indices: indices::Indices::default(),
            named: None,
            recorded: None,
            resource_indices: HashMap::new(),
            outside: HashSet::new(),
            constexprs: HashMap::new(),
            commits: 0,
            in_flight: VecDeque::new(),
            completed: 0,
            skips: Skips::default(),
            at: (0, 0),
            substitutes: Substitutes::default(),
            owners: HashMap::new(),
            managed: None,
            managed_buffers: 0,
            fresh: HashMap::new(),
            from_first_use: HashSet::new(),
            identity: false,
            unwritten: unwritten::Unwritten::default(),
            staging: HashMap::new(),
            users: HashMap::new(),
            seconds: second::Seconds::default(),
            asides: aside::Asides::default(),
            device_copies: HashMap::new(),
            queue_of: HashMap::new(),
            staged_bytes: 0,
            unanswered_pipeline_ids: 0,
            block_bytes: [0; 2],
            creations: Vec::new(),
            kernels: HashSet::new(),
        })
    }

    /// The backend told how the stream's recording device kept Managed buffers.
    pub fn with_managed(mut self, copies: ManagedCopies) -> Self {
        self.managed = Some(copies);
        self
    }

    /// How a Managed buffer of `call` plays: as a Shared buffer for a one-copy stream, the
    /// runtime's Managed buffer for a two-copy one; refused where the player was not told.
    fn managed_form(&mut self, call: &Call) -> Result<ManagedCopies, PlayError> {
        let copies = self.managed.ok_or_else(|| {
            refuse(
                call,
                "a Managed buffer, and the stream's Managed form (one copy or two: the \
                 recording device's hasUnifiedMemory) was not given",
            )
        })?;
        self.managed_buffers += 1;
        Ok(copies)
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

    /// The block of the heap argument `at` names; none for a skipped object.
    fn heap_of(&self, call: &Call, at: usize) -> Result<Option<rt::BlockId>, PlayError> {
        let id = Self::id(call, at)?;
        match self.get(call, id)? {
            Object::Block(range, _, _) => Ok(Some(range.block)),
            Object::Skipped => Ok(None),
            _ => Err(refuse(call, format!("#{id} is no heap"))),
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

    /// The command buffers not yet committed whose encoders named `object`.
    fn users_of(&self, object: u64) -> Vec<u64> {
        (self.users.iter())
            .filter(|(_, named)| named.contains(&object))
            .map(|(&command_buffer, _)| command_buffer)
            .collect()
    }

    /// Zeroes bytes `[from, len)` of `block`, a block or heap of the stream just made: the live
    /// system zeroes a heap's memory once at its making, so a buffer placed in it reads zero where
    /// nothing wrote it, while a block the runtime takes again holds a released one's bytes.
    /// `from` is the extent the runtime zeroed itself. Each piece is a buffer zeroed from its start
    /// and dropped at once: its zeroing stays (`rt::Device::new_buffer`).
    fn zero_block(&self, block: &rt::Block, from: u64, len: u64) -> Result<(), rt::Error> {
        let mut at = from;
        while at < len {
            let piece = (len - at).min(rt::MAX_BUFFER_LENGTH);
            drop(self.device.new_buffer(block.id, at, piece, Some(0))?);
            at += piece;
        }
        Ok(())
    }

    /// Commits `command_buffer` as the next commit, in flight until its completed handler runs:
    /// the runtime runs it once it has let go of what the command buffer held, the textures and
    /// samplers it named with their indices among them, which its status alone does not say
    /// (the status changes first).
    fn commit(&mut self, command_buffer: rt::CommandBuffer) -> Result<(), rt::Error> {
        let retired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&retired);
        command_buffer.add_completed_handler(move |_| flag.store(true, Ordering::Release))?;
        command_buffer.commit()?;
        self.commits += 1;
        self.in_flight
            .push_back((self.commits, command_buffer, retired));
        Ok(())
    }

    /// The commits that have retired ([`commit`](Self::commit)), every earlier one with them.
    fn completed(&mut self) -> u64 {
        while let Some((at, _, retired)) = self.in_flight.front() {
            if !retired.load(Ordering::Acquire) {
                break;
            }
            self.completed = *at;
            self.in_flight.pop_front();
        }
        self.completed
    }

    /// Waits until the commits up to `commit` have retired: `waitUntilCompleted` returns once the
    /// command buffer's handlers have run.
    fn wait_for(&mut self, commit: u64) {
        while self.completed() < commit {
            let Some((_, front, retired)) = self.in_flight.front() else {
                return;
            };
            front.wait_until_completed();
            if !retired.load(Ordering::Acquire) {
                return;
            }
        }
    }

    /// `make` at the index the stream names for the object `call` makes (`indices`): the call's
    /// resource index (`named`), else the ID the program saw of the object (`recorded`); else at
    /// the lowest free index of `class`.
    fn place<T>(
        &mut self,
        call: &Call,
        class: rt::IndexClass,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Result<T, PlayError> {
        let recorded = (self.recorded).filter(|&(object, _)| object == call.result);
        // The ID's low half is its index (docs/RESOURCE-ID.md).
        let recorded = recorded.map(|(_, id)| id as u32);
        let named = self.named.or(recorded);
        if let Some(index) = self.indices.target(class, named) {
            self.settle(class, index);
        }
        match self.indices.place(call.result, class, named, make) {
            Some(made) => made.map_err(failed(call)),
            None => Err(refuse(call, format!("every {class:?} index is in use"))),
        }
    }

    /// Before a placement at `index` of `class`: waits for the commits that named an object
    /// holding it to retire, where they have not (`Indices::owed`), and counts the wait.
    fn settle(&mut self, class: rt::IndexClass, index: u32) {
        let retired = self.completed();
        if let Some(commit) = self.indices.owed(class, index, retired) {
            let began = std::time::Instant::now();
            self.wait_for(commit);
            let nanos = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.indices.waited(class, index, nanos);
        }
    }

    /// `make` at the highest index of `class` that no object holds and the program did not
    /// reserve (`Indices::top`): outside the indices the program's own objects take.
    fn place_outside<T>(
        &mut self,
        call: &Call,
        class: rt::IndexClass,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Result<T, PlayError> {
        if let Some(index) = self.indices.highest(class, |_| false) {
            self.settle(class, index);
        }
        match self.indices.top(call.result, class, make) {
            Some(made) => made.map_err(failed(call)),
            None => Err(refuse(call, format!("every {class:?} index is in use"))),
        }
    }

    /// A buffer of `length` bytes and storage `mode` in a place of its own, zeroed, holding
    /// `bytes` from its start; `no_copy` where its bytes are the program's memory.
    fn new_buffer(
        &mut self,
        call: &Call,
        length: u64,
        mode: rt::StorageMode,
        bytes: &[u8],
        no_copy: bool,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let mode = match mode {
            rt::StorageMode::Managed => match self.managed_form(call)? {
                ManagedCopies::One => rt::StorageMode::Shared,
                ManagedCopies::Two => {
                    // A stream that tells its form gives a no-copy buffer's bytes at creation.
                    if no_copy && !self.identity && bytes.is_empty() {
                        self.fresh.insert(call.result, None);
                    }
                    rt::StorageMode::Managed
                }
            },
            mode => mode,
        };
        let class = match mode {
            rt::StorageMode::Private if bytes.is_empty() => rt::MemoryClass::Private,
            rt::StorageMode::Shared | rt::StorageMode::Managed => rt::MemoryClass::Shared,
            _ => return Err(refuse(call, format!("a {mode:?} buffer of this call"))),
        };
        if class == rt::MemoryClass::Shared && bytes.is_empty() {
            self.unwritten.made(call.result, None);
        }
        let completed = self.completed();
        let place = (self.places.take(class, length, completed)).map_err(failed(call))?;
        let (start, block) = (place.range.start, place.block);
        let buffer = match mode {
            // A two-copy stream's Managed buffer on a one-copy device: the GPU's copy here, the
            // CPU's in a place of its own (`second`), both holding the bytes it starts with.
            rt::StorageMode::Managed if self.device.has_unified_memory() => {
                let gpu = self.device.new_buffer(block, start, length, Some(0));
                let gpu = gpu.map_err(failed(call))?;
                write_into(&gpu, 0, bytes)?;
                let cpu_place =
                    (self.places.take(class, length, completed)).map_err(failed(call))?;
                let at = (cpu_place.block, cpu_place.range.start);
                let cpu = self.device.new_buffer(at.0, at.1, length, Some(0));
                let cpu = cpu.map_err(failed(call))?;
                write_into(&cpu, 0, bytes)?;
                self.seconds
                    .insert(call.result, cpu, cpu_place, gpu.clone());
                self.asides.two_copies(call.result);
                gpu
            }
            rt::StorageMode::Managed => {
                // The consumer's bytes first: the place may hold a released buffer's.
                let shared = self.device.new_buffer(block, start, length, Some(0));
                let shared = shared.map_err(failed(call))?;
                write_into(&shared, 0, bytes)?;
                let copied = bytes.len() as u64;
                // The GPU copy, placed as a Private buffer of its length is.
                let copy = (self
                    .places
                    .take(rt::MemoryClass::Private, length, completed))
                .map_err(failed(call))?;
                let at = (copy.block, copy.range.start);
                let managed =
                    (self.device).new_managed_buffer((block, start), Some(at), length, copied);
                self.asides.two_copies(call.result);
                let managed = managed.map_err(failed(call))?;
                self.device_copies.insert(call.result, copy);
                managed
            }
            _ => {
                let buffer = self.device.new_buffer(block, start, length, Some(0));
                let buffer = buffer.map_err(failed(call))?;
                write_into(&buffer, 0, bytes)?;
                buffer
            }
        };
        addressed(buffer, Some(place))
    }

    /// A texture of `descriptor`, at an index where shaders reach it: the next of its class's
    /// sequence, or for a texture `outside` the program's IDs one outside it.
    fn new_texture(
        &mut self,
        call: &Call,
        descriptor: &rt::TextureDescriptor,
        heap: Option<(rt::HeapRange, u64)>,
        outside: bool,
    ) -> Result<rt::Texture, PlayError> {
        let device = self.device.clone();
        let make = |index: Option<u32>| match heap {
            Some((range, offset)) => device.new_heap_texture(descriptor, index, range, offset),
            None => device.new_texture(descriptor, index),
        };
        match (texture_class(descriptor.usage), outside) {
            (Some(class), false) => self.place(call, class, |index| make(Some(index))),
            (Some(class), true) => self.place_outside(call, class, |index| make(Some(index))),
            (None, _) => make(None).map_err(failed(call)),
        }
    }

    /// The index `call` names for the texture it makes: its descriptor's resource index (an
    /// ordinary texture made in place of a surface's or a tiled one takes it too), or a view's
    /// `resourceIndex:`.
    fn named_index(&self, call: &Call) -> Result<Option<u32>, PlayError> {
        let at = match &*call.selector {
            "newTextureWithDescriptor:"
            | "newTextureWithDescriptor:offset:"
            | "newTextureWithDescriptor:offset:bytesPerRow:"
            | "newTextureWithDescriptor:iosurface:plane:"
            | "newTiledTextureWithDescriptor:offset:bytesPerRow:" => {
                return match arg(call, 0)? {
                    Value::Object(id) => Ok(self.resource_indices.get(id).copied()),
                    map @ Value::Map(_) => {
                        resource_index_of(call, &Fields::of(call, map, &TEXTURE_KEYS)?)
                    }
                    _ => Ok(None),
                };
            }
            "newTextureViewWithPixelFormat:resourceIndex:" => 1,
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:" => 4,
            _ => return Ok(None),
        };
        let index = u32::try_from(uint(call, at)?);
        index
            .map(Some)
            .map_err(|_| refuse(call, format!("resource index past u32 at {at}")))
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
        self.numbered(|device, ids| {
            device.new_compute_pipeline_state(descriptor, |samplers| ids(samplers))
        })
        .map_err(failed(call))
    }

    /// The pipeline state `make` makes on the device with `ids`, which gives each numbered
    /// constexpr sampler an index of its own, the highest free one, the same words one index: the
    /// stream names no index for it, and the program's sampler states take the lowest, which a
    /// constexpr sampler there would lose to them.
    fn numbered<T>(
        &mut self,
        make: impl FnOnce(
            &rt::Device,
            &mut dyn FnMut(&[rt::NumberedSampler]) -> Result<Vec<u32>, rt::Error>,
        ) -> Result<T, rt::Error>,
    ) -> Result<T, rt::Error> {
        let mut taken: Vec<([u64; 2], u32)> = Vec::new();
        let made = {
            let (constexprs, indices) = (&self.constexprs, &self.indices);
            let mut sampler_ids = |samplers: &[rt::NumberedSampler]| {
                let mut ids = Vec::with_capacity(samplers.len());
                for sampler in samplers {
                    let rt::NumberedSampler::Constexpr(words) = sampler else {
                        ids.push(0);
                        continue;
                    };
                    let known = (constexprs.get(words).copied())
                        .or_else(|| taken.iter().find(|(w, _)| w == words).map(|&(_, i)| i));
                    let index = match known {
                        Some(index) => index,
                        // The highest free sampler index this call has not taken yet.
                        None => {
                            let fresh = |index| taken.iter().any(|&(_, i)| i == index);
                            let index = indices.highest(rt::IndexClass::Sampler, fresh);
                            // None free: the next would be past the class's range.
                            let index = index.ok_or(rt::Error::Refused(
                                rt::Refusal::IndicesNotGranted {
                                    class: rt::IndexClass::Sampler,
                                    index: rt::IndexClass::Sampler.range().end,
                                },
                            ))?;
                            taken.push((*words, index));
                            index
                        }
                    };
                    ids.push(index);
                }
                Ok(ids)
            };
            make(&self.device, &mut sampler_ids)
        };
        for (words, index) in taken {
            self.constexprs.insert(words, index);
            self.indices.hold(None, rt::IndexClass::Sampler, index);
        }
        made
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
                self.new_buffer(call, uint(call, 0)?, mode, &[], false)
            }
            "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                let mode = storage(call, uint(call, 2)?)?;
                // The pages' bytes at creation, where the stream gives them.
                let bytes = arg(call, 0)?.as_bytes().unwrap_or_default();
                self.new_buffer(call, uint(call, 1)?, mode, bytes, true)
            }
            "newBufferWithBytes:length:options:" => {
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| data(call, 0).ok()?.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let mode = storage(call, uint(call, 2)?)?;
                self.new_buffer(call, length, mode, bytes, false)
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
                // A no-copy buffer's bytes at creation, where the stream gives them.
                let contents = match f.get("contents") {
                    None => &[][..],
                    Some(value) => value
                        .as_bytes()
                        .ok_or_else(|| refuse(call, "contents that are no bytes"))?,
                };
                let raw = option_sets(call, 0, f.list("options")?.unwrap_or_default())?;
                let length = f.required("length", Fields::uint)?;
                let no_copy = f.get("noCopy").and_then(Value::as_bool) == Some(true);
                self.new_buffer(call, length, storage(call, raw)?, contents, no_copy)
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
                let block = self
                    .device
                    .take_block(class, length)
                    .map_err(failed(call))?;
                // The bytes past the take's zero mark are the player's to zero.
                (self.zero_block(&block, block.zeroed, length)).map_err(failed(call))?;
                self.block_bytes[class_index(class)] += length;
                let made = Made {
                    address: Some(block.gpu_address),
                };
                let range = rt::HeapRange {
                    block: block.id,
                    offset: 0,
                    len: length,
                };
                Ok(Some((Object::Block(range, class, block), made)))
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
            "newRenderPipelineStateWithDescriptor:error:" => self.render_pipeline_state(call),
            "newDepthStencilStateWithDescriptor:" => {
                let descriptor = translator::depth_stencil(call)?;
                let state = self.device.new_depth_stencil_state(&descriptor);
                unaddressed(Object::DepthStencil(state))
            }
            "MTLRenderPassDescriptor.new" | "MTLRenderPassDescriptor.renderPassDescriptor" => {
                unaddressed(Object::Pass(Box::default()))
            }
            // Libraries from source: no device here compiles Metal source.
            "newLibraryWithSource:options:error:" => self.skip_object(),
            "newFence" => unaddressed(Object::Fence(self.device.new_fence())),
            "newSamplerStateWithDescriptor:" => {
                let (mut descriptor, f) = translator::sampler(call)?;
                if let Some(color) = f.metal("borderColor")? {
                    descriptor.border_color = color;
                }
                self.named = f.uint("resourceIndex")?.map(|index| index as u32);
                let device = self.device.clone();
                let state = self.place(call, rt::IndexClass::Sampler, |index| {
                    device.new_sampler_state(&descriptor, index)
                })?;
                unaddressed(Object::Sampler(state))
            }
            "newTextureWithDescriptor:" => {
                // A drawable the stream never saw made comes as a map: a texture of it.
                let outside =
                    matches!(arg(call, 0)?, Value::Object(id) if self.outside.contains(id));
                let descriptor = match outside {
                    true => self.described(call, 0)?,
                    false => self.descriptor_or_map(call, 0)?,
                };
                let texture = self.new_texture(call, &descriptor, None, outside)?;
                unaddressed(Object::Texture(texture))
            }
            "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:"
            | "m2v.textureDescriptor" => {
                let descriptor = made_texture_descriptor(call)?;
                if call.selector.starts_with("m2v.") {
                    let f = Fields::arg(call, 0, &TEXTURE_KEYS)?;
                    if f.bool("outside")? == Some(true) {
                        if f.list("resourceIndex")?.is_some() {
                            return Err(refuse(call, "a texture outside with a resource index"));
                        }
                        self.outside.insert(call.result);
                    }
                }
                if let Some(index) = resource_index(call)? {
                    self.resource_indices.insert(call.result, index);
                }
                unaddressed(Object::TextureDescriptor(Box::new(descriptor)))
            }
            // Every index is the backend's already.
            "reserveResourceIndicesForResourceType:indices:indexCount:" => {
                let class = match uint(call, 0)? {
                    0 => rt::IndexClass::Texture,
                    1 => rt::IndexClass::Sampler,
                    other => return Err(refuse(call, format!("resource type {other}"))),
                };
                for index in super::args::list(call, 1)? {
                    let index = index.as_uint().and_then(|i| u32::try_from(i).ok());
                    let index =
                        index.ok_or_else(|| refuse(call, "argument 1 is no list of indices"))?;
                    self.indices.reserve(class, index);
                }
                Ok(None)
            }
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
    /// The texture descriptor argument `at` names, or the descriptor it is where it is a map.
    fn descriptor_or_map(
        &self,
        call: &Call,
        at: usize,
    ) -> Result<rt::TextureDescriptor, PlayError> {
        match arg(call, at)? {
            map @ Value::Map(_) => {
                let f = Fields::of(call, map, &TEXTURE_KEYS)?;
                f.only("outside", Fields::bool, false)?;
                let mut d = rt::TextureDescriptor::new(
                    rt::TextureType::Type2D,
                    rt::PixelFormat::Invalid,
                    1,
                    1,
                );
                texture_fields(call, &f, &mut d)?;
                Ok(d)
            }
            _ => self.texture_descriptor(call, at),
        }
    }

    /// An ordinary texture of `descriptor` in place of one the runtime cannot make, counted in
    /// `count` (`Substitutes`).
    fn substitute(
        &mut self,
        call: &Call,
        descriptor: &rt::TextureDescriptor,
        count: fn(&mut Substitutes) -> &mut u64,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let texture = self.new_texture(call, descriptor, None, false)?;
        *count(&mut self.substitutes) += 1;
        unaddressed(Object::Texture(texture))
    }

    /// A texture `make` makes at an index of the class `usage` picks, where shaders reach it; an
    /// ordinary texture of `descriptor` in its place where the runtime answers a typed gap.
    fn texture_or_substitute(
        &mut self,
        call: &Call,
        usage: rt::TextureUsage,
        descriptor: &rt::TextureDescriptor,
        make: impl Fn(Option<u32>) -> Result<rt::Texture, rt::Error>,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let gap = std::cell::Cell::new(false);
        let noted = |index| {
            let made = make(index);
            gap.set(matches!(made, Err(rt::Error::Unsupported(_))));
            made
        };
        let made = match texture_class(usage) {
            Some(class) => self.place(call, class, |index| noted(Some(index))),
            None => noted(None).map_err(failed(call)),
        };
        match made {
            Ok(texture) => unaddressed(Object::Texture(texture)),
            Err(_) if gap.get() => self.substitute(call, descriptor, |s| &mut s.gaps),
            Err(error) => Err(error),
        }
    }

    /// A view of `texture`: `format` and `kind` over its `levels` and `slices`.
    fn view(
        &mut self,
        call: &Call,
        texture: &rt::Texture,
        format: rt::PixelFormat,
        kind: rt::TextureType,
        levels: Range<u32>,
        slices: Range<u32>,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let parent = *texture.descriptor();
        // A substitute's descriptor: the view's own, at its first level.
        let mut own = parent;
        own.pixel_format = format;
        own.texture_type = kind;
        own.width = (parent.width >> levels.start).max(1);
        own.height = (parent.height >> levels.start).max(1);
        own.mipmap_level_count = levels.len() as u32;
        own.array_length = match kind {
            rt::TextureType::Cube | rt::TextureType::CubeArray => (slices.len() as u32 / 6).max(1),
            _ => (slices.len() as u32).max(1),
        };
        let make =
            |index| texture.new_texture_view(format, kind, levels.clone(), slices.clone(), index);
        self.texture_or_substitute(call, parent.usage, &own, make)
    }

    /// The texture descriptor object argument `at` names, refused where the stream marks it
    /// `outside`: only a device's `newTextureWithDescriptor:` makes a texture outside the
    /// program's IDs.
    fn texture_descriptor(
        &self,
        call: &Call,
        at: usize,
    ) -> Result<rt::TextureDescriptor, PlayError> {
        if self.outside.contains(&Self::id(call, at)?) {
            return Err(refuse(call, format!("argument {at} is marked outside")));
        }
        self.described(call, at)
    }

    /// The texture descriptor object argument `at` names, marked or not.
    fn described(&self, call: &Call, at: usize) -> Result<rt::TextureDescriptor, PlayError> {
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
        // A buffer of `length` bytes and storage options `raw` at `offset` in the block. A
        // two-copy Managed one's GPU copy, on a device that keeps one, is the one place the play
        // holds for it, placed as a Private buffer of its length is; the stream's block holds its
        // bytes.
        let device = self.device.clone();
        let mut placed = |length: u64, raw: u64, offset: u64| {
            let at = range.offset + offset;
            let mut copy = None;
            let buffer = match storage(call, raw)? {
                rt::StorageMode::Managed if class == rt::MemoryClass::Shared => {
                    match self.managed_form(call)? {
                        ManagedCopies::One => device.new_buffer(range.block, at, length, None),
                        ManagedCopies::Two if device.has_unified_memory() => {
                            self.seconds.in_block += 1;
                            device.new_managed_buffer((range.block, at), None, length, length)
                        }
                        ManagedCopies::Two => {
                            let completed = self.completed();
                            let place =
                                (self
                                    .places
                                    .take(rt::MemoryClass::Private, length, completed))
                                .map_err(failed(call))?;
                            let copy_at = (place.block, place.range.start);
                            copy = Some(place);
                            device.new_managed_buffer(
                                (range.block, at),
                                Some(copy_at),
                                length,
                                length,
                            )
                        }
                    }
                }
                _ => device.new_buffer(range.block, at, length, None),
            };
            let buffer = buffer.map_err(failed(call))?;
            if class == rt::MemoryClass::Shared {
                let placed = (call.receiver, offset..offset + length);
                self.unwritten.made(call.result, Some(placed));
            }
            addressed(buffer, copy)
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
                let heap = Some((range, uint(call, 1)?));
                let texture = self.new_texture(call, &descriptor, heap, false)?;
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
                self.asides.announced(call.receiver);
                if self.seconds.name(call.receiver, range.start, range.end) {
                    return Ok(None);
                }
                let did = buffer.did_modify_range(range.start, range.end - range.start);
                did.map_err(failed(call))?;
                Ok(None)
            }
            "newTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                let (offset, bytes_per_row) = (uint(call, 1)?, uint(call, 2)?);
                let make = |index| buffer.new_texture(&descriptor, index, offset, bytes_per_row);
                self.texture_or_substitute(call, descriptor.usage, &descriptor, make)
            }
            "newTiledTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                self.substitute(call, &descriptor, |s| &mut s.tiled)
            }
            "newTextureWithDescriptor:iosurface:plane:" => {
                let descriptor = self.descriptor_or_map(call, 0)?;
                self.substitute(call, &descriptor, |s| &mut s.surfaces)
            }
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
            // The resource index places the view (`named_index`).
            "newTextureViewWithPixelFormat:" | "newTextureViewWithPixelFormat:resourceIndex:" => {
                let d = *texture.descriptor();
                let format = metal(call, uint(call, 0)?)?;
                let slices = whole_slices(&d);
                return self.view(
                    call,
                    texture,
                    format,
                    d.texture_type,
                    0..d.mipmap_level_count,
                    slices,
                );
            }
            "newTextureViewWithPixelFormat:textureType:levels:slices:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:" => {
                let format = metal(call, uint(call, 0)?)?;
                let kind = metal(call, uint(call, 1)?)?;
                let levels = super::args::range(call, arg(call, 2)?)?;
                let slices = super::args::range(call, arg(call, 3)?)?;
                let narrow = |r: std::ops::Range<u64>| -> Result<std::ops::Range<u32>, PlayError> {
                    let start =
                        u32::try_from(r.start).map_err(|_| refuse(call, "a range past u32"))?;
                    let end = u32::try_from(r.end).map_err(|_| refuse(call, "a range past u32"))?;
                    Ok(start..end)
                };
                return self.view(
                    call,
                    texture,
                    format,
                    kind,
                    narrow(levels)?,
                    narrow(slices)?,
                );
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
        self.named = self.named_index(call)?;
        if &*call.selector == "release" {
            self.owners.remove(&call.receiver);
            self.fresh.remove(&call.receiver);
            self.unwritten.released(call.receiver);
            self.outside.remove(&call.receiver);
            self.queue_of.remove(&call.receiver);
            // A second copy's CPU copy: the commits so far copied from it.
            if let Some(place) = self.seconds.remove(call.receiver) {
                self.places.give(place, self.commits, Vec::new());
            }
            // An aside: the commits so far and the command buffers that named its buffer copied
            // into it.
            if let Some(place) = self.asides.remove(call.receiver) {
                let users = self.users_of(call.receiver);
                self.places.give(place, self.commits, users);
            }
            let users = self.users_of(call.receiver).len();
            self.indices.give(call.receiver, users);
            self.resource_indices.remove(&call.receiver);
            match self.objects.remove(&call.receiver) {
                Some(Object::Buffer(_, Some(place))) => {
                    let users = self.users_of(call.receiver);
                    if let Some(copy) = self.device_copies.remove(&call.receiver) {
                        self.places.give(copy, self.commits, users.clone());
                    }
                    self.places.give(place, self.commits, users);
                }
                // The runtime keeps the block while a buffer of it lives or work uses it, as a
                // heap's resources keep the heap.
                Some(Object::Block(range, class, block)) => {
                    self.device.release_block(block).map_err(failed(call))?;
                    self.block_bytes[class_index(class)] -= range.len;
                }
                // A command buffer released uncommitted runs nothing: neither its staging nor its
                // uses of buffers released before it.
                Some(Object::CommandBuffer(_)) => {
                    for place in self.staging.remove(&call.receiver).unwrap_or_default() {
                        self.places.give(place, self.commits, Vec::new());
                    }
                    if let Some(named) = self.users.remove(&call.receiver) {
                        self.indices.ended(&named);
                        self.places.ended(call.receiver, self.commits);
                    }
                }
                _ => {}
            }
            return Ok(Made::default());
        }
        if let Some(&command_buffer) = self.owners.get(&call.receiver) {
            // Skipped encoders too: what the program's GPU passes reach is the stream's.
            let compute = matches!(self.objects.get(&call.receiver), Some(Object::Compute(_)));
            self.unwritten.encoded(call, command_buffer, compute);
            let mut named = Vec::new();
            objects_named(&call.args, &mut named);
            named.retain(|id| {
                matches!(
                    self.objects.get(id),
                    Some(Object::Buffer(_, Some(_)) | Object::Texture(_) | Object::Sampler(_))
                )
            });
            if !named.is_empty() {
                self.users.entry(command_buffer).or_default().extend(named);
            }
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
            Some(
                Object::CommandBuffer(_) | Object::Compute(_) | Object::Blit(_) | Object::Render(_)
            )
        );
        if call.result != 0 && !encodes && self.names_skipped(call) {
            // What a call makes of a skipped object is skipped too.
            self.skips.objects += 1;
            self.objects.insert(call.result, Object::Skipped);
            return Ok(Made::default());
        }
        if !self.fresh.is_empty() {
            // An encoder naming a no-copy Managed buffer not yet used: its command buffer is the
            // buffer's first use.
            if let Some(&user) = self.owners.get(&call.receiver) {
                let mut named = Vec::new();
                objects_named(&call.args, &mut named);
                for id in named {
                    if let Some(first @ None) = self.fresh.get_mut(&id) {
                        *first = Some(user);
                    }
                }
            }
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
                        self.queue_of.insert(call.result, id);
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
                Object::Render(_) => {
                    self.render_call(call)?;
                    None
                }
                Object::Pass(_) => {
                    self.pass_call(call)?;
                    None
                }
                Object::Buffer(buffer, _) => {
                    let buffer = buffer.clone();
                    self.buffer_call(call, &buffer)?
                }
                Object::Block(range, class, _) => {
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
        let end = offset + bytes.len() as u64;
        self.unwritten.written(target, offset..end);
        if let Some(cpu) = self.seconds.cpu(target) {
            // The CPU's copy of a buffer with a second copy; bytes recorded before a no-copy
            // one's first use reach its GPU copy at the next commit.
            write_into(cpu, offset, bytes)?;
            if self.fresh.contains_key(&target) && !bytes.is_empty() {
                self.seconds.name(target, offset, end);
                self.from_first_use.insert(target);
            }
            return Ok(());
        }
        match self.objects.get(&target) {
            Some(Object::Buffer(buffer, _)) => {
                write_into(buffer, offset, bytes)?;
                // Bytes recorded before a no-copy Managed buffer's first use are its contents
                // at creation, as near as the stream has them: its GPU copy takes them too.
                if self.fresh.contains_key(&target) && !bytes.is_empty() {
                    let length = bytes.len() as u64;
                    let did = buffer.did_modify_range(offset, length);
                    did.map_err(|error| PlayError::Runtime {
                        selector: "m2v.write".into(),
                        error: error.to_string(),
                    })?;
                    self.from_first_use.insert(target);
                }
                Ok(())
            }
            Some(Object::Block(range, rt::MemoryClass::Shared, _)) => {
                let whole = self
                    .device
                    .new_buffer(range.block, range.offset, range.len, None);
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

    fn gpu_interval(&mut self, command_buffer: u64) -> Option<(u64, u64)> {
        let Some(Object::CommandBuffer(done)) = self.objects.get(&command_buffer) else {
            return None;
        };
        // Both are 0 for one that ended with an error.
        let (start, end) = (done.gpu_start_time(), done.gpu_end_time());
        (start != 0 && end >= start).then_some((start, end))
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        let unread = |why: &str| PlayError::Refused {
            selector: "m2v.check".into(),
            why: why.into(),
        };
        // The CPU's bytes of a buffer with a second copy, as Metal's `contents` answers them: its
        // aside from a read-back of a stream that records its announced bytes to its next
        // announcement (`aside`).
        let cpu = self.asides.read(object).or(self.seconds.cpu(object));
        let buffer = match (cpu, self.objects.get(&object)) {
            (Some(cpu), _) => Some(cpu),
            (None, Some(Object::Buffer(buffer, _))) => Some(buffer),
            _ => None,
        };
        match buffer {
            Some(buffer) => {
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
            // The runtime gives a pipeline state no resource ID: it builds no indirect command
            // buffers. The player reports the ID as differing, and the gap is counted.
            Some(Object::ComputePipeline(_) | Object::RenderPipeline(_)) => {
                self.unanswered_pipeline_ids += 1;
                return Ok(None);
            }
            Some(Object::Skipped) => {
                self.skips.resource_ids += 1;
                return Ok(None);
            }
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
            ("pushes", metrics.pushes),
            ("pushes_held", metrics.pushes_held),
            ("passes", metrics.passes),
            ("barriers", metrics.barriers),
            ("bytes_copied", metrics.bytes_copied),
            ("bytes_zeroed", metrics.bytes_zeroed),
            ("allocations_zeroed", metrics.allocations_zeroed),
            ("allocation_zeroing_nanos", metrics.allocation_zeroing_nanos),
            ("kept_bytes_zeroed", metrics.kept_bytes_zeroed),
            ("kept_zeroing_nanos", metrics.kept_zeroing_nanos),
            ("zeroed_bytes_taken", metrics.zeroed_bytes_taken),
            ("zeroed_bytes_freed", metrics.zeroed_bytes_freed),
            ("block_zero_fills", metrics.block_zero_fills),
            ("block_zero_fill_bytes", metrics.block_zero_fill_bytes),
            ("pipelines_created", metrics.pipelines_created),
            ("first_use_waits", metrics.first_use_waits),
            ("first_use_wait_nanos", metrics.first_use_wait_nanos),
            ("blocks_made", metrics.blocks_made),
            ("block_bytes_made", metrics.block_bytes_made),
            ("blocks_reused", metrics.blocks_reused),
            ("consumer_allocations", metrics.consumer_allocations),
            (
                "consumer_allocation_nanos",
                metrics.consumer_allocation_nanos,
            ),
            ("maker_allocation_nanos", metrics.maker_allocation_nanos),
            ("allocations_freed", metrics.allocations_freed),
            ("submissions_during_frees", metrics.submissions_during_frees),
            ("idle_frees_under_250ms", metrics.idle_frees_by_age[0]),
            ("idle_frees_250_275ms", metrics.idle_frees_by_age[1]),
            ("idle_frees_275_350ms", metrics.idle_frees_by_age[2]),
            ("idle_frees_350_500ms", metrics.idle_frees_by_age[3]),
            ("idle_frees_500ms_1s", metrics.idle_frees_by_age[4]),
            ("idle_frees_1s_up", metrics.idle_frees_by_age[5]),
            ("idle_reuses_under_50ms", metrics.idle_reuses_by_age[0]),
            ("idle_reuses_50_100ms", metrics.idle_reuses_by_age[1]),
            ("idle_reuses_100_250ms", metrics.idle_reuses_by_age[2]),
            ("idle_reuses_250_500ms", metrics.idle_reuses_by_age[3]),
            ("idle_reuses_500ms_1s", metrics.idle_reuses_by_age[4]),
            ("idle_reuses_1_2s", metrics.idle_reuses_by_age[5]),
            ("idle_reuses_2s_up", metrics.idle_reuses_by_age[6]),
            (
                "dispatches_without_effects",
                metrics.dispatches_without_effects,
            ),
            ("translations", metrics.translations),
            ("translations_loaded", metrics.translations_loaded),
            ("ring_writes", metrics.ring_writes),
            ("upload_waits", metrics.upload_waits),
            ("synchronizations", metrics.synchronizations),
            ("managed_bytes_captured", metrics.managed_bytes_captured),
            ("capture_waits", metrics.capture_waits),
            ("capture_streams", metrics.capture_streams),
            ("capture_overflows", metrics.capture_overflows),
            ("capture_overflow_bytes", metrics.capture_overflow_bytes),
            ("capture_copy_nanos", metrics.capture_copy_nanos),
            ("capture_wait_nanos", metrics.capture_wait_nanos),
            ("skipped_encoders", self.skips.encoders),
            ("skipped_objects", self.skips.objects),
            ("skipped_render_states", self.skips.render_states),
            ("skipped_dispatches", self.skips.dispatches),
            ("skipped_draws", self.skips.draws),
            ("skipped_render_commands", self.skips.render_command_count()),
            ("skipped_blits", self.skips.blits),
            ("staged_guest_ram_bytes", self.staged_bytes),
            ("skipped_guest_ram_writes", self.skips.guest_ram_writes),
            ("skipped_guest_ram_reads", self.skips.guest_ram_reads),
            ("skipped_texture_syncs", self.skips.texture_syncs),
            ("skipped_mipmaps", self.skips.mipmaps),
            ("substituted_surface_textures", self.substitutes.surfaces),
            ("substituted_tiled_textures", self.substitutes.tiled),
            ("substituted_gap_textures", self.substitutes.gaps),
            ("skipped_calls", self.skips.calls),
            ("skipped_resource_ids", self.skips.resource_ids),
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
            ("managed_spilled_bytes", gauges.managed_spilled_bytes),
            ("shared_idle_bytes", gauges.shared_idle_bytes),
            ("private_idle_bytes", gauges.private_idle_bytes),
            ("shared_idle_age_nanos", gauges.shared_idle_age_nanos),
            ("private_idle_age_nanos", gauges.private_idle_age_nanos),
            ("capture_bytes", gauges.capture_bytes),
            ("staging_bytes", gauges.staging_bytes),
            ("stage_entries", gauges.stage_entries),
            ("stage_entry_bytes", gauges.stage_entry_bytes),
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
        let mut lines: Vec<String> = lines.collect();
        if let Some(copies) = self.managed {
            let from = match self.identity {
                true => "the stream's device identity",
                false => "the player's setting",
            };
            let mut line = match copies {
                ManagedCopies::One => format!(
                    "managed one copy ({from}): {} Managed buffers played as Shared",
                    self.managed_buffers
                ),
                ManagedCopies::Two => format!(
                    "managed two copies ({from}): {} Managed buffers; {} no-copy ones with \
                     contents from first use, not from create",
                    self.managed_buffers,
                    self.from_first_use.len()
                ),
            };
            if copies == ManagedCopies::Two && self.device.has_unified_memory() {
                line.push_str(
                    "; recorded with two copies, played as this device's one copy (unified memory)",
                );
            }
            lines.push(line);
        }
        if let Some((at, why)) = &self.skips.first_render_state {
            lines.push(format!(
                "skipped render states {}, the first #{at}: {why}",
                self.skips.render_states
            ));
        }
        for (gap, (count, frame, at)) in &self.skips.render_commands {
            lines.push(format!(
                "skipped render commands {gap} {count}, the first at frame {frame}, call {at}"
            ));
        }
        if self.managed == Some(ManagedCopies::Two) && self.device.has_unified_memory() {
            lines.push(self.seconds.line());
        }
        lines.push(self.asides.line());
        lines.push(self.unwritten.line());
        lines.push(self.indices.line());
        lines
    }

    fn skips(&self) -> u64 {
        self.skips.total()
    }

    fn locate(&mut self, frame: u64, call: u64) {
        self.at = (frame, call);
    }

    fn announced_bytes_recorded(&mut self) {
        self.asides.mark();
    }

    fn recorded_unified_memory(&mut self, unified: bool) {
        self.identity = true;
        self.managed = Some(match unified {
            true => ManagedCopies::One,
            false => ManagedCopies::Two,
        });
    }

    /// The runtime makes no blits between buffers yet.
    fn snapshots(&self) -> bool {
        false
    }

    fn recorded_id(&mut self, object: u64, id: u64) {
        self.recorded = Some((object, id));
    }

    fn managed_texture(&self, texture: u64) -> Option<bool> {
        match self.objects.get(&texture) {
            Some(Object::Texture(texture)) => {
                Some(texture.descriptor().storage_mode == rt::StorageMode::Managed)
            }
            _ => None,
        }
    }
}
