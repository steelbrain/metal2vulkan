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
mod guest_ram;
mod imports;
pub(super) mod indices;
mod judged;
pub(super) mod places;
mod render;
pub(super) mod second;
mod surfaces;
#[cfg(test)]
mod tests;
mod tiled;
pub(super) mod unwritten;
mod videos;

pub use judged::judged_records;

use super::args::{arg, data, metal, option_sets, refuse, uint, variant, Fields};
use super::covers::SlotStage;
use super::pixels::Shape;
use super::player::{Backend, Counters, Made, PlayError, ReadCopy};
use super::stream::Call;
use super::translator;
use super::value::Value;
use crate::gpu::wait::new_runtime::{wait_until_retired, Retired};
use crate::gpu::wait::SubmissionBackend;
use crate::gpu::DeviceType;
use metal2vulkan::metal::resource_word::TextureWord;
use metal2vulkan::metal::sampler::SamplerWord;
use metal2vulkan::pipeline::{LinkedFunction, LinkedFunctions};
use metal2vulkan_rt as rt;
use places::{class_index, granule, Place, Places};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::Arc;
pub use videos::VideoHandle;

/// What the play skipped because the runtime does not build it yet.
#[derive(Default)]
struct Skips {
    /// Encoders the runtime does not build: parallel render encoders, and render encoders of a
    /// pass it does not build (`render`).
    encoders: u64,
    /// Those encoders by kind: `ParallelRenderCommandEncoder`, `Unheld(…)` for a pass naming a
    /// property the runtime's descriptor does not hold, or the runtime's gap as Rust prints it
    /// for debugging; how many, and the frame and call of the first.
    encoder_kinds: BTreeMap<String, (u64, u64, u64)>,
    /// Objects it does not build: libraries from source.
    objects: u64,
    /// Render pipeline states the runtime answers with a typed gap or the translator refuses, by
    /// the gap (`Translation(KIND)` for the translator's refusals, `translation_kind`): how many,
    /// and the first one's object id and the runtime's answer.
    render_states: BTreeMap<String, (u64, u64, String)>,
    /// Compute pipeline states the same way, by the gap.
    compute_states: BTreeMap<String, (u64, u64, String)>,
    /// Dispatches that bind a skipped object, a skipped compute pipeline state among them.
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
    /// `synchronizeTexture:slice:level:`, which the runtime does not build yet.
    texture_syncs: u64,
    /// Sampler binds of LOD clamps the runtime answers with `Gap::LodClamps`: clamps whose samples
    /// read otherwise than the sampler's own.
    clamped_samplers: u64,
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

    /// The render pipeline states skipped, of every gap.
    fn render_state_count(&self) -> u64 {
        self.render_states.values().map(|&(count, ..)| count).sum()
    }

    /// The compute pipeline states skipped, of every gap.
    fn compute_state_count(&self) -> u64 {
        self.compute_states.values().map(|&(count, ..)| count).sum()
    }

    /// Whether a pipeline state the runtime answered `error` for is skipped: counted in `states`
    /// by its gap where the runtime answers a typed gap or the translator refuses it (by the
    /// refusal's kind, `translation_kind`); any other error is not.
    fn skipped_state(
        states: &mut BTreeMap<String, (u64, u64, String)>,
        call: &Call,
        error: &rt::Error,
    ) -> bool {
        let gap = match error {
            rt::Error::Unsupported(gap) => format!("{gap:?}"),
            rt::Error::Translation(refused) => translation_kind(refused),
            _ => return false,
        };
        let kind = states.entry(gap);
        kind.or_insert((0, call.result, error.to_string())).0 += 1;
        true
    }

    /// Everything skipped, in one count (`Backend::skips`).
    fn total(&self) -> u64 {
        let Skips {
            encoders,
            encoder_kinds: _,
            objects,
            render_states: _,
            compute_states: _,
            dispatches,
            draws,
            render_commands: _,
            blits,
            guest_ram_writes,
            texture_syncs,
            clamped_samplers,
            calls,
            resource_ids,
        } = self;
        let counted = [
            encoders,
            objects,
            dispatches,
            draws,
            blits,
            guest_ram_writes,
            texture_syncs,
            clamped_samplers,
            calls,
            resource_ids,
        ];
        counted.into_iter().sum::<u64>()
            + self.render_state_count()
            + self.compute_state_count()
            + self.render_command_count()
    }
}

/// A translator refusal's kind, as the report keys it: `Translation(Unsupported(FEATURE))` with
/// the refused feature as Rust prints it for debugging, `Translation(LimitExceeded(LIMIT))` and
/// `Translation(Internal(KIND))` the same, and the error's variant alone for the rest. Never a
/// symbol or a location: those name the program's functions.
fn translation_kind(refused: &metal2vulkan::error::Error) -> String {
    use metal2vulkan::error::Error as Refused;
    let kind = match refused {
        Refused::Unsupported(unsupported) => format!("Unsupported({:?})", unsupported.feature),
        Refused::LimitExceeded(limit) => format!("LimitExceeded({limit:?})"),
        Refused::Internal(internal) => format!("Internal({:?})", internal.kind),
        Refused::NeedsInput(_) => "NeedsInput".to_owned(),
        Refused::InvalidInput(_) => "InvalidInput".to_owned(),
        Refused::InvalidDescriptor(_) => "InvalidDescriptor".to_owned(),
        Refused::Budget(_) => "Budget".to_owned(),
        Refused::Cancelled => "Cancelled".to_owned(),
        _ => "Other".to_owned(),
    };
    format!("Translation({kind})")
}

/// Textures the play made in place of what the runtime cannot make: an ordinary texture of the
/// same descriptor, so that the IDs and addresses the program wrote name a live texture. The
/// recording's artefact, never the runtime's. Its texels are not the program's: a command that
/// names one reads them (`read`).
#[derive(Default)]
struct Substitutes {
    /// Textures over a buffer and views the runtime answers with a typed gap.
    gaps: u64,
    /// The substitutes by the runtime's gap as Rust prints it for debugging: how many, and the
    /// first one's object.
    kinds: BTreeMap<String, (u64, u64)>,
    /// Each live substitute, and the views of one, and whether a command named it yet.
    live: HashMap<u64, bool>,
    /// The substitutes a command named: how many, and the first few.
    read: u64,
    first_read: Vec<u64>,
}

impl Substitutes {
    /// Notes substitute `object`, made in place of what the runtime answers with `gap`.
    fn made(&mut self, gap: rt::Gap, object: u64) {
        self.gaps += 1;
        let kind = format!("{gap:?}");
        self.kinds.entry(kind).or_insert((0, object)).0 += 1;
        self.live.insert(object, false);
    }

    /// A `substituted textures KIND N, the first #ID` line for each kind made.
    fn lines(&self) -> impl Iterator<Item = String> + '_ {
        (self.kinds.iter()).map(|(kind, (count, first))| {
            format!("substituted textures {kind} {count}, the first #{first}")
        })
    }

    /// Notes the substitutes among `named`, which a command names.
    fn named(&mut self, named: &[u64]) {
        for object in named {
            if let Some(seen) = self.live.get_mut(object).filter(|seen| !**seen) {
                *seen = true;
                self.read += 1;
                if self.first_read.len() < 8 {
                    self.first_read.push(*object);
                }
            }
        }
    }
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

/// A committed command buffer not yet seen retired ([`NewRuntime::commit`]).
struct InFlight {
    /// Its place in the commit order, from 1.
    commit: u64,
    /// Its stream id (0 for one of the play's own).
    id: u64,
    /// Its queue's stream id (0 for the play's own).
    queue_id: u64,
    queue: rt::CommandQueue,
    command_buffer: rt::CommandBuffer,
    /// Set by its completed handler, which the runtime runs once it has let go of what the
    /// command buffer held, the textures and samplers it named with their indices among them,
    /// which its status alone does not say (the status changes first).
    retired: Arc<Retired>,
}

impl InFlight {
    /// Waits until it has retired, the wait bounded under the hang protocol (`gpu::wait`).
    fn wait(&self, backend: SubmissionBackend, operation: &str) {
        wait_until_retired(
            &self.queue,
            &self.command_buffer,
            &self.retired,
            backend,
            operation,
        );
    }
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
    /// The commits so far, and the committed command buffers not yet seen retired, in order
    /// ([`commit`](Self::commit)).
    commits: u64,
    in_flight: VecDeque<InFlight>,
    /// The device's kind, which names the backend of every bounded wait (`gpu::wait`).
    backend: SubmissionBackend,
    /// The committed command buffers no record played so far lets complete, by stream id, as the
    /// player last told them (`Backend::held`): a placement never waits for one.
    held: HashSet<u64>,
    /// The descriptor of the sampler state placed last at each sampler index, as recorded: a
    /// later one of the same descriptor there displaces nothing that work in flight reads.
    sampler_descriptors: HashMap<u32, Value>,
    /// The commits that have completed, every earlier one with them.
    completed: u64,
    skips: Skips,
    /// The frame and the stream position of the call being played (`Backend::locate`).
    at: (u64, u64),
    substitutes: Substitutes,
    unrecorded_tessellation: render::UnrecordedTessellation,
    /// Each open encoder's command buffer.
    owners: HashMap<u64, u64>,
    /// How the recording's device kept Managed buffers; `None` where the player was not told,
    /// and a Managed buffer is refused.
    managed: Option<ManagedCopies>,
    /// The host's display library (`with_display_library`), which a library from source of no
    /// arguments is made of.
    display: Option<Vec<u8>>,
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
    /// The host's copies between buffers and the guest's RAM, in the host's one-copy form
    /// (`guest_ram`).
    guest_ram: guest_ram::GuestRam,
    /// The tiled textures over a buffer of more than one level, and the copies they take at use
    /// (`tiled`).
    tiled: tiled::Tiled,
    /// The buffers in a place of the play's own that an encoder of each command buffer not yet
    /// committed named: a release of a buffer waits for those command buffers (`Places::give`).
    /// The textures and sampler states they name go to `Indices::named`.
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
    /// Whether a check or dump of a buffer with two copies reads its device copy back, in place
    /// of its CPU copy (`reading_device_copies`).
    device_reads: bool,
    /// The queue each command buffer was made on.
    queue_of: HashMap<u64, u64>,
    /// The queue of the play's own that reads a Private buffer back (`private_bytes`), made at
    /// the first such read.
    reads: Option<rt::CommandQueue>,
    /// The compute pipeline states whose `gpuResourceID` the player asked, which the runtime does
    /// not give: a typed gap, counted.
    unanswered_pipeline_ids: u64,
    /// The bytes of the stream's live blocks and heaps, by class (`class_index`).
    block_bytes: [u64; 2],
    /// Each compute pipeline state made, in order, and the kernels made so far.
    creations: Vec<Creation>,
    kernels: HashSet<Kernel>,
    /// The device's `minImportedHostPointerAlignment`, where it imports host memory: the
    /// alignment of the host pages a no-copy buffer's memory is imported from (`imports`).
    host_alignment: Option<u64>,
    /// The no-copy buffers by how they play, and the recorded ranges of the stream's blocks.
    no_copies: imports::NoCopies,
    /// The textures over buffers made (not substituted): in place, and held in an image of their
    /// own (`rt::Gauges::textures_copied_over` rose across the making).
    textures_over: [u64; 2],
    /// Of those in an image of their own, how many of each pixel format and usage.
    copied_over_kinds: BTreeMap<String, u64>,
    /// The textures over a surface's plane, and the entries under them.
    surfaces: surfaces::Surfaces,
    /// The video surfaces, made through the surface call.
    videos: videos::Videos,
}

/// The objects `values` name, lists included.
pub(super) fn objects_named(values: &[Value], out: &mut Vec<u64>) {
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

/// The keys of an `MTLTextureSwizzleChannels`.
const SWIZZLE_KEYS: [&str; 4] = ["red", "green", "blue", "alpha"];

/// The `MTLTextureSwizzleChannels` `s` holds, each channel set.
fn swizzle(s: &Fields) -> Result<rt::TextureSwizzleChannels, PlayError> {
    Ok(rt::TextureSwizzleChannels {
        red: s.required("red", Fields::metal)?,
        green: s.required("green", Fields::metal)?,
        blue: s.required("blue", Fields::metal)?,
        alpha: s.required("alpha", Fields::metal)?,
    })
}

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
    if let Some(s) = f.fields("swizzle", &SWIZZLE_KEYS)? {
        d.swizzle = swizzle(&s)?;
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

/// The class of the index of a texture of `pf` and `usage`, where shaders reach it, as the runtime
/// takes it (`rt::Refusal::TextureIndex`): unknown usage, shader-read or shader-write usage, or a
/// colour render target, which a fragment function reads through its attachment; none for a depth
/// or stencil target shaders neither read nor write.
fn texture_class(pf: rt::PixelFormat, usage: rt::TextureUsage) -> Option<rt::IndexClass> {
    let reached = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
    let color_target = usage.contains(rt::TextureUsage::RENDER_TARGET)
        && !metal2vulkan::metal::format::is_depth_stencil(pf);
    (usage.is_empty() || usage.intersects(reached) || color_target)
        .then_some(rt::IndexClass::Texture)
}

/// How long the device has, once the backend ends or drops, to finish its GPU work and stop its
/// threads.
const QUIESCE: std::time::Duration = std::time::Duration::from_secs(30);

/// What the play's device left once [`NewRuntime::end`] tore it down: the runtime's gauges, read
/// after the device dropped, and whether it quiesced in time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ended {
    /// Each of the runtime's gauges by name (`rt::Gauges`; `threads` the slots a thread still
    /// holds). Every one is zero once the device is torn down.
    pub gauges: Vec<(&'static str, u64)>,
    /// Whether every gauge read zero, whole (`rt::Gauges::default()`).
    pub zero: bool,
    /// Whether the device quiesced within 30 seconds.
    pub idle: bool,
}

impl Ended {
    /// The matrix's line of each gauge, `M2V-MATRIX gauge replay-rt/device/GAUGE@drop N`, in the
    /// form `scripts/matrix/results.py` reads: its rule fails a gauge not at zero at device drop.
    pub fn lines(&self) -> Vec<String> {
        let lines = self.gauges.iter().map(|(gauge, value)| {
            format!("M2V-MATRIX gauge replay-rt/device/{gauge}@drop {value}")
        });
        lines.collect()
    }

    /// Err names the gauges not at zero, a handle or a resource that outlived the play, or a
    /// quiesce past those 30 seconds whose threads run on into `exit()`.
    pub fn result(&self) -> Result<(), String> {
        if self.idle && self.zero {
            return Ok(());
        }
        let live = self.gauges.iter().filter(|(_, value)| *value != 0);
        let live: Vec<String> = live
            .map(|(gauge, value)| format!("{gauge}={value}"))
            .collect();
        let idle = match self.idle {
            true => "quiesced",
            false => "did not quiesce in time",
        };
        Err(format!(
            "the device {idle} and its gauges read [{}] after the backend dropped",
            live.join(" ")
        ))
    }
}

/// Every gauge of `gauges` by its name in `rt::Gauges`, `threads` counting the slots a thread
/// holds.
fn named(gauges: rt::Gauges) -> Vec<(&'static str, u64)> {
    let threads = gauges.threads.iter().filter(|&&id| id != 0).count() as u64;
    vec![
        ("threads", threads),
        ("buffers", gauges.buffers),
        ("textures", gauges.textures),
        ("samplers", gauges.samplers),
        ("pipelines", gauges.pipelines),
        ("kernels", gauges.kernels),
        ("kernel_entries", gauges.kernel_entries),
        ("render_contents", gauges.render_contents),
        ("attachment_views", gauges.attachment_views),
        ("command_buffers", gauges.command_buffers),
        ("descriptor_slots", gauges.descriptor_slots),
        ("blocks", gauges.blocks),
        ("imports", gauges.imports),
        ("exports", gauges.exports),
        ("surface_imports", gauges.surface_imports),
        ("events", gauges.events),
        ("device_samplers", gauges.device_samplers),
        ("spilled_bytes", gauges.spilled_bytes),
        ("managed_spilled_bytes", gauges.managed_spilled_bytes),
        ("shared_allocation_bytes", gauges.shared_allocation_bytes),
        ("private_allocation_bytes", gauges.private_allocation_bytes),
        ("shared_idle_bytes", gauges.shared_idle_bytes),
        ("private_idle_bytes", gauges.private_idle_bytes),
        ("shared_idle_age_nanos", gauges.shared_idle_age_nanos),
        ("private_idle_age_nanos", gauges.private_idle_age_nanos),
        ("shared_reserve_bytes", gauges.shared_reserve_bytes),
        ("private_reserve_bytes", gauges.private_reserve_bytes),
        ("shared_long_reused_bytes", gauges.shared_long_reused_bytes),
        (
            "private_long_reused_bytes",
            gauges.private_long_reused_bytes,
        ),
        ("staging_bytes", gauges.staging_bytes),
        ("stage_entries", gauges.stage_entries),
        ("stage_entry_bytes", gauges.stage_entry_bytes),
        ("pipeline_cache_bytes", gauges.pipeline_cache_bytes),
        ("shadows", gauges.shadows),
        ("shadow_bytes", gauges.shadow_bytes),
        ("textures_copied_over", gauges.textures_copied_over),
        ("timestamp_pools", gauges.timestamp_pools),
    ]
}

impl Drop for NewRuntime {
    /// The device's quiesce on every way out of a play, an error's and a panic's included: no
    /// runtime thread runs on into the process's `exit()`, where the driver tears itself down
    /// beside it. After [`NewRuntime::end`]'s quiesce, this one answers at once.
    fn drop(&mut self) {
        self.device.quiesce(QUIESCE);
    }
}

impl NewRuntime {
    /// The backend on `device`, of `device_type`: every texture and sampler index granted to it,
    /// and each wait for a command buffer bounded under the hang protocol (`gpu::wait`).
    pub fn new(device: rt::Device, device_type: DeviceType) -> Result<Self, rt::Error> {
        for class in [rt::IndexClass::Texture, rt::IndexClass::Sampler] {
            let range = class.range();
            device.grant_indices(class, range.start, range.end - range.start)?;
        }
        Ok(NewRuntime {
            places: Places::new(device.clone()),
            host_alignment: imports::host_alignment(&device),
            no_copies: imports::NoCopies::default(),
            textures_over: [0; 2],
            copied_over_kinds: BTreeMap::new(),
            surfaces: surfaces::Surfaces::default(),
            videos: videos::Videos::new(VideoHandle::DmaBuf),
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
            backend: SubmissionBackend::vulkan(device_type),
            held: HashSet::new(),
            sampler_descriptors: HashMap::new(),
            completed: 0,
            skips: Skips::default(),
            at: (0, 0),
            substitutes: Substitutes::default(),
            unrecorded_tessellation: render::UnrecordedTessellation::default(),
            owners: HashMap::new(),
            managed: None,
            display: None,
            managed_buffers: 0,
            fresh: HashMap::new(),
            from_first_use: HashSet::new(),
            identity: false,
            unwritten: unwritten::Unwritten::default(),
            guest_ram: guest_ram::GuestRam::default(),
            tiled: tiled::Tiled::default(),
            users: HashMap::new(),
            seconds: second::Seconds::default(),
            asides: aside::Asides::default(),
            device_copies: HashMap::new(),
            device_reads: false,
            queue_of: HashMap::new(),
            reads: None,
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

    /// The backend whose checks and dumps of a buffer with two copies read its device copy back,
    /// where it would read its CPU copy, as Metal's `contents` answers: through a blit into a
    /// Shared buffer of the play's own where the runtime keeps the device copy (that blit's
    /// commit takes the ranges named and not yet taken, as the program's next commit would), and
    /// through the buffer's mapping where the device's one copy is the GPU's and the player keeps
    /// the CPU's (`second`).
    pub fn reading_device_copies(mut self) -> Self {
        self.device_reads = true;
        self
    }

    /// The report header's line on a device that keeps a device copy of each Managed buffer (no
    /// unified memory): which copy a check or dump of one reads. None on a device with one copy,
    /// where a buffer the player keeps a second copy of says it in its own check line.
    pub fn managed_digests(&self) -> Option<String> {
        if self.device.has_unified_memory() {
            return None;
        }
        Some(match self.device_reads {
            false => "managed digests: this device keeps a device copy of each Managed buffer; a \
                      check or dump of one reads its CPU copy only, as Metal's contents answers \
                      (--read-device-copies yes reads the device copy back)"
                .to_owned(),
            true => "managed digests: this device keeps a device copy of each Managed buffer; a \
                     check or dump of one reads the device copy back"
                .to_owned(),
        })
    }

    /// The backend told the consumer's decoder route, the handle its video surfaces are made
    /// with (`videos`); a dma-buf where it is not told.
    pub fn with_video_handle(mut self, handle: VideoHandle) -> Self {
        self.videos = videos::Videos::new(handle);
        self
    }

    /// The backend given the host's display library, a metallib (the stream's
    /// `NAME.display.metallib`): the library from source the host makes of its own display
    /// shaders, a `newLibraryWithSource:options:error:` of no arguments, is made of it. Without
    /// it, that library is skipped and counted as any library from source.
    pub fn with_display_library(mut self, metallib: Vec<u8>) -> Self {
        self.display = Some(metallib);
        self
    }

    /// Ends the play's device before the process exits, as the runtime's contract asks of a
    /// consumer (`rt::Device`): quiesces it within 30 seconds (a committed command buffer waiting
    /// on an event no played record signals ends unrun), drops every handle the backend holds, the
    /// device's last among them, and reads the gauges once it is torn down ([`Ended`]).
    pub fn end(self) -> Ended {
        let counters = self.device.counters();
        let idle = self.device.quiesce(QUIESCE);
        drop(self);
        let gauges = counters.gauges();
        Ended {
            zero: gauges == rt::Gauges::default(),
            gauges: named(gauges),
            idle,
        }
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
    fn zero_block(&self, block: &rt::Block, len: u64) -> Result<(), rt::Error> {
        let mut at = 0;
        while at < len {
            let piece = (len - at).min(rt::MAX_BUFFER_LENGTH);
            drop(self.device.new_buffer(block.id, at, piece, Some(0))?);
            at += piece;
        }
        Ok(())
    }

    /// The bytes of `buffer`, a Private buffer, which has no `contents`, or the device copy of a
    /// Managed one, which `contents` does not answer (`reading_device_copies`): a blit copies them into
    /// a Shared buffer of the play's own, on a queue of its own (`reads`), and the read waits for
    /// it. The player reads only after the work that writes the buffer completed. A length that is
    /// not a multiple of 4 is the copy's own refusal (`rt::Refusal::BufferCopyUnaligned`).
    fn private_bytes(&mut self, buffer: &rt::Buffer) -> Result<Vec<u8>, rt::Error> {
        let length = buffer.length();
        if length == 0 {
            return Ok(Vec::new());
        }
        let queue = match &self.reads {
            Some(queue) => queue.clone(),
            None => (self.reads.insert(self.device.new_command_queue(1)?)).clone(),
        };
        let block = self.device.take_block(rt::MemoryClass::Shared, length)?;
        let copied = (|| {
            let into = self.device.new_buffer(block.id, 0, length, None)?;
            let command_buffer = queue.command_buffer()?;
            let mut blit = command_buffer.blit_command_encoder()?;
            blit.copy_from_buffer_to_buffer(buffer, 0, &into, 0, length)?;
            blit.end_encoding();
            command_buffer.commit()?;
            command_buffer.wait_until_completed()?;
            let contents = into.contents().expect("a Shared buffer is mapped");
            // SAFETY: the mapping holds `length` bytes, and the copy into them completed.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
            Ok(bytes.to_vec())
        })();
        self.device.release_block(block)?;
        copied
    }

    /// The texels of `texture`, a Private texture, which the CPU does not reach: a blit copies
    /// its first level's first slice into a Shared buffer of the play's own, laid out as
    /// `texture_bytes` reads a texture the CPU reaches, on the play's read queue (`reads`), and
    /// the read waits for it, as the old runtime's play reads every texture.
    fn private_texture_bytes(&mut self, texture: &rt::Texture) -> Result<Vec<u8>, rt::Error> {
        let d = texture.descriptor();
        let raw = u64::from(d.pixel_format.as_raw());
        let query = rt::TextureLevelQuery::plane_2d(raw, d.width.into(), d.height.into());
        let level = self.device.texture_level_info(&query);
        let length = level.bytes_per_image;
        if length == 0 {
            return Ok(Vec::new());
        }
        let queue = match &self.reads {
            Some(queue) => queue.clone(),
            None => (self.reads.insert(self.device.new_command_queue(1)?)).clone(),
        };
        let block = self.device.take_block(rt::MemoryClass::Shared, length)?;
        let copied = (|| {
            let into = self.device.new_buffer(block.id, 0, length, None)?;
            let command_buffer = queue.command_buffer()?;
            let mut blit = command_buffer.blit_command_encoder()?;
            let from = rt::TexturePlace {
                slice: 0,
                level: 0,
                origin: [0; 3],
            };
            let layout = rt::BufferLayout {
                offset: 0,
                bytes_per_row: level.bytes_per_row,
                bytes_per_image: level.bytes_per_image,
            };
            blit.copy_from_texture_to_buffer(texture, from, [d.width, d.height, 1], &into, layout)?;
            blit.end_encoding();
            command_buffer.commit()?;
            command_buffer.wait_until_completed()?;
            let contents = into.contents().expect("a Shared buffer is mapped");
            // SAFETY: the mapping holds `length` bytes, and the copy into them completed.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
            Ok(bytes.to_vec())
        })();
        self.device.release_block(block)?;
        copied
    }

    /// Commits `command_buffer`, of `queue` and the stream's `id` (0 for one of the play's own), as
    /// the next commit, in flight until its completed handler runs (`InFlight::retired`).
    fn commit(
        &mut self,
        queue: &rt::CommandQueue,
        command_buffer: rt::CommandBuffer,
        id: u64,
    ) -> Result<(), rt::Error> {
        let retired = Retired::watch(&command_buffer)?;
        command_buffer.commit()?;
        self.commits += 1;
        self.in_flight.push_back(InFlight {
            commit: self.commits,
            id,
            queue_id: self.queue_of.get(&id).copied().unwrap_or(0),
            queue: queue.clone(),
            command_buffer,
            retired,
        });
        Ok(())
    }

    /// The queue of `call`'s receiver, a command buffer of the stream.
    fn queue_for(&self, call: &Call) -> Result<rt::CommandQueue, PlayError> {
        let queue = self.queue_of.get(&call.receiver);
        match queue.and_then(|queue| self.objects.get(queue)) {
            Some(Object::Queue(queue)) => Ok(queue.clone()),
            _ => Err(refuse(call, "a command buffer of no queue")),
        }
    }

    /// The commits that have retired ([`commit`](Self::commit)), every earlier one with them.
    fn completed(&mut self) -> u64 {
        while let Some(front) = self.in_flight.front() {
            if !front.retired.is_set() {
                break;
            }
            self.completed = front.commit;
            self.in_flight.pop_front();
        }
        self.completed
    }

    /// Waits until the commits up to `commit` have retired, each wait bounded (`gpu::wait`): one
    /// past the bound ends the process as a hang. Where one of them is held (`held`), no wait can
    /// end: false, unwaited.
    fn wait_for(&mut self, commit: u64) -> bool {
        let mut flights = (self.in_flight.iter()).take_while(|flight| flight.commit <= commit);
        if flights.any(|flight| self.held.contains(&flight.id)) {
            return false;
        }
        while self.completed() < commit {
            let Some(front) = self.in_flight.front() else {
                return true;
            };
            let operation = "the replay's wait for an index's commits";
            front.wait(self.backend, operation);
        }
        true
    }

    /// `make` at the index the stream names for the object `call` makes (`indices`): the call's
    /// resource index (`named`), else the ID the program saw of the object (`recorded`); else at
    /// the lowest free index of `class`. `sampler` is a sampler state's recorded descriptor: where
    /// the sampler state placed last at the index has the same one, the placement displaces
    /// nothing work in flight reads (Metal hands an equal descriptor the same sampler and ID), and
    /// it takes the index without waiting.
    fn place<T>(
        &mut self,
        call: &Call,
        class: rt::IndexClass,
        sampler: Option<&Value>,
        make: impl Fn(u32) -> Result<T, rt::Error>,
    ) -> Result<T, PlayError> {
        let recorded = (self.recorded).filter(|&(object, _)| object == call.result);
        // The ID's low half is its index (docs/RESOURCE-ID.md).
        let recorded = recorded.map(|(_, id)| id as u32);
        let named = self.named.or(recorded);
        if let Some(index) = self.indices.target(class, named) {
            let equal = sampler.is_some() && self.sampler_descriptors.get(&index) == sampler;
            match equal {
                true => {
                    let retired = self.completed();
                    self.indices.equal(class, index, retired);
                }
                false => self.settle(class, index),
            }
        }
        let made = match self.indices.place(call.result, class, named, make) {
            Some(made) => made.map_err(failed(call))?,
            None => return Err(refuse(call, format!("every {class:?} index is in use"))),
        };
        if let (Some(descriptor), Some(index)) = (sampler, self.indices.index_of(call.result)) {
            self.sampler_descriptors.insert(index, descriptor.clone());
        }
        Ok(made)
    }

    /// Before a placement at `index` of `class`: waits for the commits that named an object
    /// holding it to retire, where they have not (`Indices::owed`), and counts the wait. Where one
    /// of them is held on records not yet played, no wait for it can end, and the recording
    /// device gave the index to the new object before that commit completed, as its index policy
    /// does: the placement takes the index without waiting, as the recording did, and is counted
    /// (`Indices::unwaited`).
    fn settle(&mut self, class: rt::IndexClass, index: u32) {
        let retired = self.completed();
        if let Some(commit) = self.indices.owed(class, index, retired) {
            let began = std::time::Instant::now();
            if !self.wait_for(commit) {
                self.indices.unwaited();
                return;
            }
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
    /// `bytes` from its start; `no_copy` (its creator) where its bytes are host memory, which is
    /// imported.
    fn new_buffer(
        &mut self,
        call: &Call,
        length: u64,
        mode: rt::StorageMode,
        bytes: &[u8],
        no_copy: Option<imports::NoCopy>,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let mode = match mode {
            rt::StorageMode::Managed => match self.managed_form(call)? {
                ManagedCopies::One => rt::StorageMode::Shared,
                ManagedCopies::Two => {
                    // A stream that tells its form gives a no-copy buffer's bytes at creation.
                    if no_copy.is_some() && !self.identity && bytes.is_empty() {
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
        // Each copy's place keeps the slack the buffer had in the recording.
        let recorded = granule(call.answer("gpuAddress").and_then(Value::as_address));
        // A bytes-form no-copy buffer's memory is host pages imported as a block of their own, as
        // the product imports the host's; any other buffer's, a descriptor-form no-copy one's
        // included (a pool buffer in the product, `imports::NoCopies`), a place of its own.
        let (block, start, import, place) = match no_copy {
            Some(imports::NoCopy::Bytes) => {
                let alignment = (self.host_alignment).ok_or_else(|| {
                    refuse(
                        call,
                        "a no-copy buffer on a device that imports no host memory",
                    )
                })?;
                let (import, len) =
                    imports::imported(&self.device, length, alignment).map_err(translated(call))?;
                self.no_copies.imported(len);
                (import.id, 0, Some(import), None)
            }
            descriptor => {
                let place =
                    (self.places.take(class, length, recorded, completed)).map_err(failed(call))?;
                if let Some(imports::NoCopy::Descriptor { pinned_unzeroed }) = descriptor {
                    self.no_copies.pooled(length, pinned_unzeroed);
                }
                (place.block, place.range.start, None, Some(place))
            }
        };
        // A place may hold a released buffer's bytes; an import's pages start zero, and the
        // runtime never zeroes a no-copy buffer's, as Metal does not.
        let zero = place.is_some().then_some(0);
        let buffer = match mode {
            // A two-copy stream's Managed buffer on a one-copy device: the GPU's copy here, the
            // CPU's in a place of its own (`second`), both holding the bytes it starts with. A
            // no-copy one's GPU copy is the import, which shaders and textures reach as the
            // product's one copy.
            rt::StorageMode::Managed if self.device.has_unified_memory() => {
                let gpu = self.device.new_buffer(block, start, length, zero);
                let gpu = gpu.map_err(failed(call))?;
                write_into(&gpu, 0, bytes)?;
                let cpu_place =
                    (self.places.take(class, length, recorded, completed)).map_err(failed(call))?;
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
                let shared = self.device.new_buffer(block, start, length, zero);
                let shared = shared.map_err(failed(call))?;
                write_into(&shared, 0, bytes)?;
                let copied = bytes.len() as u64;
                // The GPU copy, placed as a Private buffer of its length is, with the same slack.
                let copy =
                    (self
                        .places
                        .take(rt::MemoryClass::Private, length, recorded, completed))
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
                let buffer = self.device.new_buffer(block, start, length, zero);
                let buffer = buffer.map_err(failed(call))?;
                write_into(&buffer, 0, bytes)?;
                buffer
            }
        };
        if let Some(import) = import {
            // The buffer keeps the block, and the runtime the import, while it lives or work
            // uses it; then the import's release frees the pages.
            self.device.release_block(import).map_err(failed(call))?;
        }
        addressed(buffer, place)
    }

    /// The runtime's live textures over buffers in an image of their own. A copied texture freed
    /// on another thread between two reads moves it too, so a making counts by it nearly always.
    fn copied_over(&self) -> u64 {
        self.device.counters().gauges().textures_copied_over
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
        match (
            texture_class(descriptor.pixel_format, descriptor.usage),
            outside,
        ) {
            (Some(class), false) => self.place(call, class, None, |index| make(Some(index))),
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
            | "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:"
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
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:" => 5,
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
        descriptor: &rt::ComputePipelineDescriptor,
    ) -> Result<rt::ComputePipelineState, rt::Error> {
        self.numbered(|device, ids| {
            device.new_compute_pipeline_state(descriptor, |samplers| ids(samplers))
        })
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

    /// A compute pipeline state of `descriptor`, of `kernel`'s, timed and counted (`creations`);
    /// skipped and counted where the runtime answers a typed gap or the translator refuses it
    /// (`compute_states`), so that the play goes on and its dispatches are skipped.
    fn timed_state(
        &mut self,
        call: &Call,
        descriptor: &rt::ComputePipelineDescriptor,
        kernel: Kernel,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let began = std::time::Instant::now();
        let state = match self.compute_pipeline_state(descriptor) {
            Ok(state) => state,
            Err(error) if Skips::skipped_state(&mut self.skips.compute_states, call, &error) => {
                return unaddressed(Object::Skipped);
            }
            Err(error) => return Err(failed(call)(error)),
        };
        self.creations.push(Creation {
            nanos: u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX),
            new_kernel: self.kernels.insert(kernel),
        });
        unaddressed(Object::ComputePipeline(state))
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
                self.new_buffer(call, uint(call, 0)?, mode, &[], None)
            }
            "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                let mode = storage(call, uint(call, 2)?)?;
                // The pages' bytes at creation, where the stream gives them.
                let bytes = arg(call, 0)?.as_bytes().unwrap_or_default();
                let kind = Some(imports::NoCopy::Bytes);
                self.new_buffer(call, uint(call, 1)?, mode, bytes, kind)
            }
            "newBufferWithBytes:length:options:" => {
                let length = uint(call, 1)?;
                let bytes = usize::try_from(length)
                    .ok()
                    .and_then(|length| data(call, 0).ok()?.get(..length))
                    .ok_or_else(|| refuse(call, "fewer bytes than its length"))?;
                let mode = storage(call, uint(call, 2)?)?;
                self.new_buffer(call, length, mode, bytes, None)
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
                if no_copy {
                    // The program's buffer in a recorded block's storage: a buffer of the block.
                    let recorded = (call.answer("gpuAddress").and_then(Value::as_address))
                        .or_else(|| f.get("pinnedGpuAddress").and_then(Value::as_address));
                    let holding = recorded.and_then(|at| self.no_copies.holding(at, length));
                    if let Some((block, offset)) = holding {
                        self.no_copies.in_block();
                        return self.no_copy_in_block(call, block, offset, length, raw, contents);
                    }
                }
                let pinned_unzeroed =
                    f.get("pinnedGpuAddress").is_some() && f.get("zeroFrom").is_none();
                let kind = no_copy.then_some(imports::NoCopy::Descriptor { pinned_unzeroed });
                self.new_buffer(call, length, storage(call, raw)?, contents, kind)
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
                // A block's bytes are undefined: the player zeroes them.
                (self.zero_block(&block, length)).map_err(failed(call))?;
                self.block_bytes[class_index(class)] += length;
                let made = Made {
                    address: Some(block.gpu_address),
                };
                let range = rt::HeapRange {
                    block: block.id,
                    offset: 0,
                    len: length,
                };
                let recorded = call.answer("gpuAddress").and_then(Value::as_address);
                if let (Some(start), "m2v.newBlockWithLength:storageMode:") =
                    (recorded, &*call.selector)
                {
                    self.no_copies.block_made(call.result, start, length);
                }
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
                self.timed_state(call, &d, function.kernel)
            }
            "newComputePipelineStateWithDescriptor:options:reflection:error:" => {
                let (d, kernel) = self.compute_pipeline(call)?;
                self.timed_state(call, &d, kernel)
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
            "newLibraryWithSource:options:error:" => match &self.display {
                Some(metallib) if call.args.is_empty() => {
                    let library = rt::Library::from_metallib(metallib);
                    unaddressed(Object::Library(library.map_err(translated(call))?))
                }
                _ => self.skip_object(),
            },
            "newFence" => unaddressed(Object::Fence(self.device.new_fence())),
            "newSamplerStateWithDescriptor:" => {
                let (mut descriptor, f) = translator::sampler(call)?;
                if let Some(color) = f.metal("borderColor")? {
                    descriptor.border_color = color;
                }
                self.named = f.uint("resourceIndex")?.map(|index| index as u32);
                let device = self.device.clone();
                let recorded = arg(call, 0)?;
                let state = self.place(call, rt::IndexClass::Sampler, Some(recorded), |index| {
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

    /// An ordinary texture of `descriptor` in place of one the runtime answers with `gap`,
    /// counted (`Substitutes`).
    fn substitute(
        &mut self,
        call: &Call,
        descriptor: &rt::TextureDescriptor,
        gap: rt::Gap,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let texture = self.new_texture(call, descriptor, None, false)?;
        self.substitutes.made(gap, call.result);
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
        let gap = std::cell::Cell::new(None);
        let noted = |index| {
            let made = make(index);
            gap.set(match &made {
                Err(rt::Error::Unsupported(answered)) => Some(*answered),
                _ => None,
            });
            made
        };
        let made = match texture_class(descriptor.pixel_format, usage) {
            Some(class) => self.place(call, class, None, |index| noted(Some(index))),
            None => noted(None).map_err(failed(call)),
        };
        match (made, gap.get()) {
            (Ok(texture), _) => unaddressed(Object::Texture(texture)),
            (Err(_), Some(answered)) => self.substitute(call, descriptor, answered),
            (Err(error), None) => Err(error),
        }
    }

    /// A view of `texture`: `format` and `kind` over its levels and slices, through `swizzle`.
    /// A substitute holds the parent's swizzle, which its own descriptor takes.
    fn view(
        &mut self,
        call: &Call,
        texture: &rt::Texture,
        (format, kind): (rt::PixelFormat, rt::TextureType),
        [levels, slices]: [Range<u32>; 2],
        swizzle: rt::TextureSwizzleChannels,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        self.surface_view(call, format)?;
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
        let make = |index| {
            let (levels, slices) = (levels.clone(), slices.clone());
            texture.new_texture_view_with_swizzle(format, kind, levels, slices, swizzle, index)
        };
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
        let block = (call.receiver, range, class);
        match &*call.selector {
            "newBufferWithLength:options:offset:" => {
                self.in_block(call, block, uint(call, 0)?, uint(call, 1)?, uint(call, 2)?)
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                if Self::id(call, 0)? != call.receiver {
                    return Err(refuse(call, "a block other than its receiver"));
                }
                self.in_block(call, block, uint(call, 2)?, uint(call, 3)?, uint(call, 1)?)
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

    /// A buffer of `length` bytes and storage options `raw` at `offset` in `block` (the block's
    /// object, its range and its class). A two-copy Managed one's GPU copy, on a device that keeps
    /// one, is the one place the play holds for it, placed as a Private buffer of its length is;
    /// the stream's block holds its bytes.
    fn in_block(
        &mut self,
        call: &Call,
        (object, range, class): (u64, rt::HeapRange, rt::MemoryClass),
        length: u64,
        raw: u64,
        offset: u64,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let device = self.device.clone();
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
                        // The GPU copy keeps the slack the buffer had in the recording.
                        let recorded =
                            granule(call.answer("gpuAddress").and_then(Value::as_address));
                        let place = (self.places.take(
                            rt::MemoryClass::Private,
                            length,
                            recorded,
                            completed,
                        ))
                        .map_err(failed(call))?;
                        let copy_at = (place.block, place.range.start);
                        copy = Some(place);
                        device.new_managed_buffer((range.block, at), Some(copy_at), length, length)
                    }
                }
            }
            // A heap's buffer takes the heap's id in the hazard sets, as its textures do; a
            // stream block's buffers are no heap's.
            _ if &*call.selector == "newBufferWithLength:options:offset:" => {
                device.new_heap_buffer(range, offset, length)
            }
            _ => device.new_buffer(range.block, at, length, None),
        };
        let buffer = buffer.map_err(failed(call))?;
        if class == rt::MemoryClass::Shared {
            self.unwritten
                .made(call.result, Some((object, offset..offset + length)));
        }
        addressed(buffer, copy)
    }

    /// A descriptor-form no-copy buffer recorded inside block `object`'s recorded range, at
    /// `offset` there: the program's buffer in the block's storage, played as a buffer of the
    /// block (`in_block`), its bytes at its making (`contents`, a two-copy stream's) first put
    /// there.
    fn no_copy_in_block(
        &mut self,
        call: &Call,
        object: u64,
        offset: u64,
        length: u64,
        raw: u64,
        contents: &[u8],
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let (range, class) = match self.objects.get(&object) {
            Some(Object::Block(range, class, _)) => (*range, *class),
            _ => {
                let why = format!("#{object} holds a no-copy buffer and is no block");
                return Err(PlayError::Invalid(why));
            }
        };
        if !contents.is_empty() {
            let at = range.offset + offset;
            let bytes = self.device.new_buffer(range.block, at, length, None);
            write_into(&bytes.map_err(failed(call))?, 0, contents)?;
        }
        self.in_block(call, (object, range, class), length, raw, offset)
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
            "newTextureWithDescriptor:offset:bytesPerRow:"
            | "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:"
            | "newTiledTextureWithDescriptor:offset:bytesPerRow:" => {
                let descriptor = self.texture_descriptor(call, 0)?;
                // A tiled texture of more than one level takes the buffer's bytes at use
                // (`tiled`); one of one level is the texture over the buffer the first form
                // makes, as the old runtime makes it.
                if call.selector.starts_with("newTiled") && descriptor.mipmap_level_count > 1 {
                    return self.tiled_texture(call, buffer, &descriptor);
                }
                // The linear form of a 2D texture is the public one; Metal lays out the slices
                // of the array it also makes itself, which the runtime takes no form of.
                let linear = call.selector.starts_with("newLinear");
                if linear && descriptor.texture_type != rt::TextureType::Type2D {
                    return Err(refuse(call, "a linear texture of a type other than 2D"));
                }
                let (offset, bytes_per_row) = (uint(call, 1)?, uint(call, 2)?);
                let make = |index| buffer.new_texture(&descriptor, index, offset, bytes_per_row);
                let (gaps, copied) = (self.substitutes.gaps, self.copied_over());
                let made = self.texture_or_substitute(call, descriptor.usage, &descriptor, make)?;
                if self.substitutes.gaps == gaps {
                    let in_image = self.copied_over() > copied;
                    self.textures_over[usize::from(in_image)] += 1;
                    if in_image {
                        let kind = format!("{:?} {:?}", descriptor.pixel_format, descriptor.usage);
                        *self.copied_over_kinds.entry(kind).or_default() += 1;
                    }
                }
                Ok(made)
            }
            "newTextureWithDescriptor:iosurface:plane:" => self.surface_texture(call, buffer),
            "m2v.externalWriteDone" if self.videos.is_video(call.receiver) => {
                self.video_picture(call)
            }
            "m2v.externalWriteDone" => self.external_write_done(call),
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
                    data(call, 3)?.into(),
                    uint(call, 4)?,
                    uint(call, 5)?,
                ),
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:" => texture.replace_region(
                region(0)?,
                super::args::index(call, 1)?,
                0,
                data(call, 2)?.into(),
                uint(call, 3)?,
                0,
            ),
            // The resource index places the view (`named_index`).
            "newTextureViewWithPixelFormat:" | "newTextureViewWithPixelFormat:resourceIndex:" => {
                let d = *texture.descriptor();
                let format = metal(call, uint(call, 0)?)?;
                let ranges = [0..d.mipmap_level_count, whole_slices(&d)];
                let swizzle = rt::TextureSwizzleChannels::DEFAULT;
                return self.view(call, texture, (format, d.texture_type), ranges, swizzle);
            }
            "newTextureViewWithPixelFormat:textureType:levels:slices:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:"
            | "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:" => {
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
                let swizzle = match call.selector.contains("swizzle:") {
                    true => swizzle(&Fields::arg(call, 4, &SWIZZLE_KEYS)?)?,
                    false => rt::TextureSwizzleChannels::DEFAULT,
                };
                let ranges = [narrow(levels)?, narrow(slices)?];
                return self.view(call, texture, (format, kind), ranges, swizzle);
            }
            _ => return Err(refuse(call, "not played on a texture")),
        };
        replaced.map(|()| None).map_err(failed(call))
    }
}

/// A texture's bytes for a check, laid out as the old runtime's backend lays them out: its first
/// level and slice, the rows and the image as far apart as Metal's level info puts them, the bytes
/// between rows zero.
fn texture_bytes(device: &rt::Device, texture: &rt::Texture) -> Result<Vec<u8>, rt::Error> {
    let d = texture.descriptor();
    let raw = u64::from(d.pixel_format.as_raw());
    let query = rt::TextureLevelQuery::plane_2d(raw, d.width.into(), d.height.into());
    let level = device.texture_level_info(&query);
    let mut bytes = vec![0; level.bytes_per_image as usize];
    let region = rt::Region {
        origin: [0; 3],
        size: [d.width, d.height, 1],
    };
    let (row, image) = (level.bytes_per_row, level.bytes_per_image);
    texture.get_bytes(&mut bytes, row, image, region, 0, 0)?;
    Ok(bytes)
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
    fn held(&mut self, held: &[u64]) {
        self.held = held.iter().copied().collect();
    }

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
            self.indices.give(call.receiver);
            self.resource_indices.remove(&call.receiver);
            self.surfaces.released(call.receiver);
            self.videos.released(call.receiver);
            self.substitutes.live.remove(&call.receiver);
            self.tiled.released(call.receiver);
            self.unrecorded_tessellation.released(call.receiver);
            match self.objects.remove(&call.receiver) {
                // Its place and its device copy's, where it has them: a no-copy buffer's memory
                // is an import, and one in a block or heap the stream's.
                Some(Object::Buffer(_, place)) => {
                    let copy = self.device_copies.remove(&call.receiver);
                    let held: Vec<Place> = copy.into_iter().chain(place).collect();
                    if !held.is_empty() {
                        let users = self.users_of(call.receiver);
                        for place in held {
                            self.places.give(place, self.commits, users.clone());
                        }
                    }
                }
                // The runtime keeps the block while a buffer of it lives or work uses it, as a
                // heap's resources keep the heap.
                Some(Object::Block(range, class, block)) => {
                    self.no_copies.block_released(call.receiver);
                    self.device.release_block(block).map_err(failed(call))?;
                    self.block_bytes[class_index(class)] -= range.len;
                }
                // A command buffer released uncommitted runs nothing: neither its guest-RAM
                // copies nor its uses of buffers released before it.
                Some(Object::CommandBuffer(_)) => {
                    self.guest_ram_released(call.receiver);
                    self.indices.ended(call.receiver);
                    if self.users.remove(&call.receiver).is_some() {
                        self.places.ended(call.receiver, self.commits);
                    }
                }
                _ => {}
            }
            return Ok(Made::default());
        }
        if !self.surfaces.is_empty() || !self.substitutes.live.is_empty() {
            // Every object the call names, a descriptor's fields among them.
            let mut named = Vec::new();
            super::covers::named(&call.args, &mut named);
            if !self.surfaces.is_empty() {
                self.surface_names(call, &named)?;
            }
            // A call on a substitute reads it, but for one that makes a view of it, which is a
            // substitute in turn.
            let view = call.selector.starts_with("newTextureView");
            if !view {
                named.push(call.receiver);
            }
            self.substitutes.named(&named);
            if view && self.substitutes.live.contains_key(&call.receiver) {
                self.substitutes.live.insert(call.result, false);
            }
        }
        if !self.tiled.is_empty() {
            // Skipped encoders too: the old runtime takes the copy at their use.
            self.tiled_named(call)?;
        }
        if let Some(&command_buffer) = self.owners.get(&call.receiver) {
            // Skipped encoders too: what the program's GPU passes reach is the stream's.
            let compute = matches!(self.objects.get(&call.receiver), Some(Object::Compute(_)));
            self.unwritten.encoded(call, command_buffer, compute);
            let mut named = Vec::new();
            objects_named(&call.args, &mut named);
            self.indices.named(command_buffer, &named);
            // The buffers with a place, or a device copy's or an aside's (two copies).
            named.retain(|id| match self.objects.get(id) {
                Some(Object::Buffer(_, place)) => place.is_some() || self.asides.has_two(*id),
                _ => false,
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
        if self.surfaces.notes(call.receiver) {
            self.surface_encoded(call);
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
        if &*call.selector == "endEncoding" && self.surfaces.notes(call.receiver) {
            self.surface_ended(call)?;
        }
        let Some((object, made)) = made else {
            return Ok(Made::default());
        };
        if call.result == 0 {
            return Err(PlayError::Invalid(format!(
                "{} makes no object",
                call.selector
            )));
        }
        self.surface_made(call, &object);
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
                // A surface's texels under the bytes (`surfaces`).
                if self.surfaces.is_receiver(target) {
                    let buffer = buffer.clone();
                    return self.surface_written((target, &buffer), offset..end);
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
        let Some(Object::CommandBuffer(waited)) = self.objects.get(&command_buffer) else {
            return Err(PlayError::Invalid(format!(
                "#{command_buffer} is no command buffer"
            )));
        };
        // One not in flight has retired already.
        let flight = (self.in_flight.iter()).find(|flight| flight.id == command_buffer);
        // The hang record names the command buffer and its queue by their stream IDs.
        if let Some(flight) = flight {
            let operation = format!(
                "the replay's wait for command buffer #{command_buffer} of queue #{}",
                flight.queue_id
            );
            flight.wait(self.backend, &operation);
        }
        // Its copies back to the guest's RAM, which the guest sees done when it completes.
        for commit in self.guest_ram.downloads_of(command_buffer) {
            let flight = (self.in_flight.iter()).find(|flight| flight.commit == commit);
            if let Some(flight) = flight {
                let operation = format!(
                    "the replay's wait for a copy back to the guest's RAM after command buffer \
                     #{command_buffer}"
                );
                flight.wait(self.backend, &operation);
            }
        }
        let error = waited.error().map(|error| format!("{error:?}"));
        self.guest_ram_downloaded(command_buffer);
        match error {
            None => Ok(()),
            Some(error) => Err(PlayError::Runtime {
                selector: "commit".into(),
                error,
            }),
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

    fn texture_shape(&mut self, object: u64) -> Option<Shape> {
        match self.objects.get(&object) {
            Some(Object::Texture(texture)) => {
                let d = texture.descriptor();
                Some(Shape {
                    pixel_format: d.pixel_format.as_raw().into(),
                    width: d.width.into(),
                    height: d.height.into(),
                })
            }
            _ => None,
        }
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        let unread = |why: &str| PlayError::Refused {
            selector: "m2v.check".into(),
            why: why.into(),
        };
        if let Some(Object::Texture(texture)) = self.objects.get(&object) {
            let texture = texture.clone();
            let bytes = match texture.descriptor().storage_mode {
                rt::StorageMode::Private => self.private_texture_bytes(&texture),
                _ => texture_bytes(&self.device, &texture),
            };
            return bytes.map_err(|error| PlayError::Runtime {
                selector: "m2v.check".into(),
                error: error.to_string(),
            });
        }
        // The CPU's bytes of a buffer with a second copy, as Metal's `contents` answers them: its
        // aside from a read-back of a stream that records its announced bytes to its next
        // announcement (`aside`). Where the play reads device copies, the buffer's own: the
        // device's copy, which a blit reads where the runtime keeps one.
        let device = self.read_copy(object) == Some(ReadCopy::Device);
        let cpu = match device {
            true => None,
            false => self.asides.read(object).or(self.seconds.cpu(object)),
        };
        let buffer = match (cpu, self.objects.get(&object)) {
            (Some(cpu), _) => Some(cpu),
            (None, Some(Object::Buffer(buffer, _))) => Some(buffer),
            _ => None,
        };
        let Some(buffer) = buffer else {
            return Err(unread("no buffer or texture"));
        };
        if device && self.device_copies.contains_key(&object) {
            let buffer = buffer.clone();
            return (self.private_bytes(&buffer)).map_err(|error| PlayError::Runtime {
                selector: "m2v.check".into(),
                error: error.to_string(),
            });
        }
        let length = usize::try_from(buffer.length()).map_err(|_| unread("its length"))?;
        let Some(contents) = buffer.contents() else {
            let buffer = buffer.clone();
            return self
                .private_bytes(&buffer)
                .map_err(|error| PlayError::Runtime {
                    selector: "m2v.check".into(),
                    error: error.to_string(),
                });
        };
        // SAFETY: the buffer's mapping holds its length, and the player waited for the work that
        // writes it.
        Ok(unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec())
    }

    fn read_copy(&self, object: u64) -> Option<ReadCopy> {
        // A Managed buffer of a two-copy stream: the runtime's device copy and the consumer's
        // bytes, or the player's own CPU copy beside the device's one copy.
        let two = self.device_copies.contains_key(&object) || self.seconds.cpu(object).is_some();
        two.then_some(match self.device_reads {
            true => ReadCopy::Device,
            false => ReadCopy::Cpu,
        })
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

    fn resource_word(&mut self, object: u64) -> Option<u32> {
        match self.objects.get(&object)? {
            Object::Texture(texture) => {
                let descriptor = texture.descriptor();
                let word = TextureWord::of(descriptor.pixel_format, descriptor.swizzle);
                word.map(|word| word.0)
            }
            Object::Sampler(sampler) => Some(SamplerWord::of(sampler.descriptor()).0),
            _ => None,
        }
    }

    fn counters(&mut self) -> Counters {
        let metrics = self.device.counters().metrics();
        // Every metric that sums: a frame's is the difference of two reads.
        let mut counters = Counters::from([
            ("submissions", metrics.submissions),
            ("pushes", metrics.pushes),
            ("pushes_held", metrics.pushes_held),
            ("passes", metrics.passes),
            ("barriers", metrics.barriers),
            ("bytes_copied", metrics.bytes_copied),
            ("over_copies", metrics.over_copies),
            ("over_copy_bytes", metrics.over_copy_bytes),
            ("over_copies_imported", metrics.over_copies_imported),
            ("over_copy_bytes_imported", metrics.over_copy_bytes_imported),
            ("attachment_read_copies", metrics.attachment_read_copies),
            (
                "attachment_read_copy_bytes",
                metrics.attachment_read_copy_bytes,
            ),
            ("attachment_loop_reads", metrics.attachment_loop_reads),
            ("bytes_zeroed", metrics.bytes_zeroed),
            ("takes_from_free_ranges", metrics.takes_from_free_ranges),
            ("block_zero_fills", metrics.block_zero_fills),
            ("block_zero_fill_bytes", metrics.block_zero_fill_bytes),
            ("texture_clears", metrics.texture_clears),
            ("texture_clear_bytes", metrics.texture_clear_bytes),
            ("texture_clear_barriers", metrics.texture_clear_barriers),
            ("pipelines_created", metrics.pipelines_created),
            ("first_use_waits", metrics.first_use_waits),
            ("first_use_wait_nanos", metrics.first_use_wait_nanos),
            ("blocks_made", metrics.blocks_made),
            ("block_bytes_made", metrics.block_bytes_made),
            ("shared_long_reuses", metrics.shared_long_reuses),
            ("private_long_reuses", metrics.private_long_reuses),
            ("allocations_mapped", metrics.allocations_mapped),
            (
                "allocations_mapped_on_takes",
                metrics.allocations_mapped_on_takes,
            ),
            ("allocation_map_refusals", metrics.allocation_map_refusals),
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
            ("staging_sleeps", metrics.staging_sleeps),
            ("synchronizations", metrics.synchronizations),
            ("commit_lock_wait_nanos", metrics.commit_lock_wait_nanos),
            ("managed_bytes_captured", metrics.managed_bytes_captured),
            ("stamps_not_available", metrics.stamps_not_available),
            ("waits_behind_counter", metrics.waits_behind_counter),
            ("external_acquires", metrics.external_acquires),
            ("external_releases", metrics.external_releases),
            ("sparse_binds", metrics.sparse_binds),
            ("surfaces_in_place", metrics.surfaces_in_place),
            ("surfaces_copied", metrics.surfaces_copied),
            ("surface_waits", metrics.surface_waits),
            ("skipped_encoders", self.skips.encoders),
            ("skipped_objects", self.skips.objects),
            ("skipped_render_states", self.skips.render_state_count()),
            ("skipped_compute_states", self.skips.compute_state_count()),
            ("skipped_dispatches", self.skips.dispatches),
            ("skipped_draws", self.skips.draws),
            ("skipped_render_commands", self.skips.render_command_count()),
            ("skipped_blits", self.skips.blits),
            ("skipped_guest_ram_writes", self.skips.guest_ram_writes),
            ("skipped_texture_syncs", self.skips.texture_syncs),
            ("skipped_clamped_samplers", self.skips.clamped_samplers),
            ("substituted_gap_textures", self.substitutes.gaps),
            ("substitutes_read", self.substitutes.read),
            (
                "patch_draws_unrecorded_tessellation",
                self.unrecorded_tessellation.patch_draws,
            ),
            ("skipped_calls", self.skips.calls),
            ("skipped_resource_ids", self.skips.resource_ids),
            ("gap_pipeline_ids", self.unanswered_pipeline_ids),
        ]);
        self.surfaces.timed();
        counters.extend(self.surfaces.counted());
        counters.extend(self.videos.counted());
        counters.extend(self.guest_ram.counted());
        counters.extend(self.tiled.counted());
        counters
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
            ("imports", gauges.imports),
            ("exports", gauges.exports),
            ("surface_imports", gauges.surface_imports),
            ("buffers", gauges.buffers),
            ("textures", gauges.textures),
            ("textures_copied_over", gauges.textures_copied_over),
            ("pipelines", gauges.pipelines),
            ("kernels", gauges.kernels),
            ("command_buffers", gauges.command_buffers),
            ("events", gauges.events),
            ("timestamp_pools", gauges.timestamp_pools),
            ("spilled_bytes", gauges.spilled_bytes),
            ("managed_spilled_bytes", gauges.managed_spilled_bytes),
            ("shared_allocation_bytes", gauges.shared_allocation_bytes),
            ("private_allocation_bytes", gauges.private_allocation_bytes),
            ("shared_idle_bytes", gauges.shared_idle_bytes),
            ("private_idle_bytes", gauges.private_idle_bytes),
            ("shared_idle_age_nanos", gauges.shared_idle_age_nanos),
            ("private_idle_age_nanos", gauges.private_idle_age_nanos),
            ("shared_reserve_bytes", gauges.shared_reserve_bytes),
            ("private_reserve_bytes", gauges.private_reserve_bytes),
            ("shared_long_reused_bytes", gauges.shared_long_reused_bytes),
            (
                "private_long_reused_bytes",
                gauges.private_long_reused_bytes,
            ),
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
        // A maximum since the device was made: it does not sum, so it is not a frame's counter.
        let longest = self.device.counters().metrics().commit_lock_wait_max_nanos;
        lines.push(format!("commit lock wait longest {longest} ns"));
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
        for (gap, (count, at, why)) in &self.skips.render_states {
            lines.push(format!(
                "skipped render states {gap} {count}, the first #{at}: {why}"
            ));
        }
        for (gap, (count, at, why)) in &self.skips.compute_states {
            lines.push(format!(
                "skipped compute states {gap} {count}, the first #{at}: {why}"
            ));
        }
        for (gap, (count, frame, at)) in &self.skips.render_commands {
            lines.push(format!(
                "skipped render commands {gap} {count}, the first at frame {frame}, call {at}"
            ));
        }
        for (kind, (count, frame, at)) in &self.skips.encoder_kinds {
            lines.push(format!(
                "skipped render encoders {kind} {count}, the first at frame {frame}, call {at}"
            ));
        }
        if self.managed == Some(ManagedCopies::Two) && self.device.has_unified_memory() {
            lines.push(self.seconds.line());
        }
        lines.extend(self.no_copies.lines());
        lines.push(self.guest_ram.line());
        lines.push(self.tiled.line());
        let [in_place, copied] = self.textures_over;
        lines.push(format!(
            "textures over buffers {}: {in_place} in place, {copied} in an image of their own, \
             copied at use (the runtime's gauge rose across the making)",
            in_place + copied
        ));
        if !self.copied_over_kinds.is_empty() {
            let mut kinds: Vec<(&String, &u64)> = self.copied_over_kinds.iter().collect();
            kinds.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let kinds: Vec<String> = (kinds.iter())
                .map(|(kind, n)| format!("{kind} {n}"))
                .collect();
            lines.push(format!(
                "textures over buffers in an image of their own, by pixel format and usage: {}",
                kinds.join(", ")
            ));
        }
        lines.push(self.surfaces.line());
        lines.push(self.videos.line());
        lines.extend(self.substitutes.lines());
        let first: String = (self.substitutes.first_read.iter())
            .map(|object| format!(" #{object}"))
            .collect();
        lines.push(format!(
            "substitutes read {}: of {} gap textures made, and their views; the first:{}",
            self.substitutes.read,
            self.substitutes.gaps,
            if first.is_empty() { " -" } else { &first },
        ));
        lines.push(self.unrecorded_tessellation.line());
        lines.push(self.asides.line());
        lines.push(self.places.line());
        lines.push(self.unwritten.line());
        lines.push(self.indices.line());
        lines.push(self.indices.unwaited_line());
        lines
    }

    fn skips(&self) -> u64 {
        self.skips.total() + self.surfaces.across_planes()
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

    fn video_pages(&mut self, buffer: u64) {
        self.videos.pages(buffer);
    }

    fn texture_writes(&self, pipeline: u64, stage: SlotStage) -> Option<Vec<u64>> {
        let accesses = match (self.objects.get(&pipeline)?, stage) {
            (Object::ComputePipeline(state), SlotStage::Kernel) => state.texture_accesses(),
            (Object::RenderPipeline(state), SlotStage::Vertex) => {
                state.texture_accesses(rt::Stage::Vertex)
            }
            (Object::RenderPipeline(state), SlotStage::Fragment) => {
                state.texture_accesses(rt::Stage::Fragment)
            }
            _ => return None,
        };
        let written = accesses.into_iter();
        let written = written.filter(|(_, access)| !matches!(access, rt::BindingAccess::ReadOnly));
        Some(written.map(|(slot, _)| u64::from(slot)).collect())
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
