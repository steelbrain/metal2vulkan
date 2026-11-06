//! Resource IDs: the `gpuResourceID` of each texture, of each sampler state and of each
//! visible-function table, where a program places textures and sampler states, and what an ID a
//! function reads reaches.
//!
//! - **What they are for.** A program writes a resource's ID into an argument buffer, as on
//!   Metal. A translated function resolves a texture's through the device's texture table, which
//!   maps the ID of every live texture to the elements of set 1's heap arrays at that ID
//!   (`heap_sets.rs`), and a sampler state's through the device's sampler table, which maps the
//!   ID of every live state to its element of set 1's sampler arrays. An ID no table holds misses:
//!   it reads the neutral resource.
//! - **Spaces.** Textures, sampler states, visible-function tables and pipeline states made with
//!   `supportIndirectCommandBuffers` have a space of IDs each, as Metal's resource types do (an
//!   M3 Max and an M4: three tables made among textures and a sampler state took 1, 2 and 3, and
//!   a new table took 1 once the first was released; the probe 2 for pipeline states, whose other
//!   states answer 0). Indirect command buffers have a space of their own too, whose IDs the
//!   address space hands out with their records (`address.rs`), from a [`Pool`] as the others'.
//!   Every sampler state has an ID, with or without argument-buffer support, as on Metal (probes 5
//!   and 7), and a texture view has one of its own, as each `MTLTexture` does. The runtime's own
//!   textures (scratch and checks no program sees, the in-place check's on its thread among them)
//!   take none: they answer 0. A program's texture takes its ID as its making's last step, once
//!   nothing but the runtime's own fault can fail it, so a refused making takes none and the IDs
//!   follow the program's order of makings. A function table's ID is always plain.
//! - **Numbering is Metal's** (the coordinator's ruling of 2026-10-07: a program that writes an ID
//!   into memory and reads it back sees the number, and the consumer measured Metal's, so Metal's
//!   numbers are the contract, where only "plain allocation skips reserved IDs" was kept before).
//!   Plain allocation ([`IdPlacement::Any`]) hands out IDs as Metal's pool does ([`Pool`]; the M4
//!   probes, plain and under API validation alike, matched in every one of 1,915 orders after
//!   reservations and 20 orders of IDs given back, 8 of them predicted
//!   before they were measured). A space's pool holds the IDs below its size, 16 for textures and
//!   sampler states and 32 for visible-function tables, pipeline states and indirect command
//!   buffers (the probe's tables: 2 to 6 of 1 to 13 given back came back before 14, 2 to 10 of 1
//!   to 29 after 30 and 31), as free ranges from 1 (0 is the null handle). Each plain ID is the
//!   lowest of the smallest free range, the lower range of two alike; where none is free the pool
//!   doubles, its new IDs one range. An ID given back joins the free ranges either side of it, so
//!   without reservations the next ID is the lowest free one but where given-back neighbours
//!   joined: with 2, 3 and 5 of 1 to 8 given back, the next are 5, 2, 3, then 9. A reservation
//!   doubles the pool until it holds each reserved ID and splits the free range holding it: one
//!   ID inside the pool's second half or later starts plain IDs past it where the range above it
//!   is the smaller (10 gives 11 to 15, then 1 to 9, then 16), and one at a power of two, or
//!   below 16, keeps the order from 1.
//! - **IDs handed out twice** (the consumer's G331). After the device's first command queue or
//!   indirect command buffer, a texture reservation keeps the pool's first range beside the ranges
//!   it leaves, and Metal hands out that range's IDs again while textures hold them (20 reserved
//!   gives 21 to 31, 1 to 15, 1 to 15 again, 16 to 19, 32 on). The runtime hands them out as Metal
//!   does: both textures answer the ID, the slot is the later one's (Slots), and each gives it back
//!   as it goes, as Metal's do ([`Pool`]).
//! - **The device's sampler ID** (the consumer's M3 Max report at 701572bbf, the M4 probes, plain
//!   and under API validation alike). Metal's device keeps a sampler ID for itself for good as it
//!   makes its first command queue, indirect command buffer, or texture whose memory is its own (of
//!   any type, usage or storage mode, memoryless, shared, placed at an ID, a surface's, or a
//!   texture buffer the device makes); a library, a function, a compute or tile pipeline state, a
//!   visible function table, a depth-stencil state, an event, a fence, a buffer, a heap (automatic,
//!   placement or sparse), a heap's or a buffer's texture, a view of a heap's texture, a
//!   rasterization rate map and a binary archive take none. A texture
//!   opened from a shared handle needs none: the handle's texture took it on the same device. It is
//!   a plain sampler ID, handed out as a state's would be and never given back, so states made
//!   after it start past it, and a sampler reservation after it is ignored; Metal has one device
//!   per GPU, so it is taken once per process. The runtime takes it the same way, once per device
//!   ([`ResourceIds::take_device_sampler`]): a program that writes a state's ID into an argument
//!   buffer and reads it back sees Metal's number. A render or mesh render pipeline state with a
//!   fragment function, made before, numbers it with its samplers instead (`numbered.rs`): held
//!   while the state, its fragment function's objects or their library live, and back after
//!   them, unless the device kept it meanwhile.
//! - **Numbered samplers** (`numbered.rs`). A pipeline state, as it is made, takes a plain sampler
//!   ID for each of its functions' constexpr samplers, texture reads and texture writes that no
//!   live holder's equal one holds, as Metal's does; its function objects and their library hold
//!   them with it.
//! - **The live count** (the consumer's ask of 2026-10-09 at 8c4df1a5b:
//!   [`Device::sampler_ids_live`](super::Device::sampler_ids_live)). Metal keeps a descriptor in
//!   its sampler heap for each sampler ID it hands out, so a caller that models where that heap
//!   lies counts the IDs. Each holder of a sampler ID counts once while it holds it: a live
//!   sampler state, placed or plain; the device's own sampler once taken; and each key the live
//!   pipeline states, function objects and libraries numbered, device-wide. The count changes
//!   under the IDs' lock at the step that takes or gives back the ID, and a caller reads it with
//!   no lock. So does the sampler pool's size
//!   ([`Device::sampler_id_pool_size`](super::Device::sampler_id_pool_size)): the consumer
//!   measured Metal's descriptor array grow at the 15th, 31st, 63rd and 127th sampler states
//!   after the device's own, where the pool ([`Pool`]) doubles from 16 to 32, 64, 128 and 256.
//! - **Placement** (`resourceIndex` with `forceResourceIndex`, Apple's private properties, a
//!   runtime extension; on an M3 Max and an M4). A resource made [`At`](IdPlacement::At) an ID has
//!   it, whether a live resource has it too or not, and plain allocation does not count it: it may
//!   hand the ID out again. `At(ResourceId(0))` is plain allocation, as Metal's forced index 0 is.
//!   A pipeline state made with `supportIndirectCommandBuffers` is placed by
//!   `PipelineOptions::placement` (the probes on an M4): Metal places it at its descriptor's
//!   `resourceIndex`, forced or not, compute and render states in their one space, keeping the
//!   index's low 32 bits; a state without that support, and a tile state, answer 0 whatever the
//!   index. Metal ended the process for a state placed before any pipeline ID was handed out or
//!   reserved, a defect the runtime does not copy: it places that state too. An indirect command
//!   buffer is placed by its descriptor's `placement` (Metal's `resourceIndex`, unforced), keeping
//!   the index's low 32 bits, as the same rules say (`address.rs`). A texture
//!   [`Device::new_texture`](super::Device::new_texture) or
//!   [`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image) makes
//!   [`Outside`](IdPlacement::Outside) takes no ID, as the runtime's own textures: a runtime
//!   extension for a caller's helper texture or its swapchain image, refused for every other
//!   resource.
//! - **Reservations** ([`Device::reserve_resource_ids`](super::Device::reserve_resource_ids),
//!   textures, sampler states, pipeline states and indirect command buffers, Metal's types 0, 1,
//!   4 and 5) keep IDs from plain allocation for good, whether a resource placed at one lives or
//!   not. Only a space's first call with an ID counts: one once the space handed out an ID,
//!   placed or plain, or took a reservation is ignored whatever it holds, with a diagnostic
//!   line, as Metal's is (its pool logs "reserve() must be called before any allocation takes
//!   place" for both; an M3 Max, macOS 27.0.1, so ignored IDs of 2^31, 2^32 - 1, 2^32 + 5 and
//!   0 alone of indirect command buffers, and 2^28 + 5 of textures, each refused or dropped in
//!   a first call). A
//!   call of no ID changes nothing, and 0 beside other IDs is no ID (0 and 5 reserve 5). A call
//!   of 0 alone reserves nothing, but leaves the pool made, as a texture pool is by the device's
//!   first queue ([`Pool::make`]): a later reservation keeps the pool's first range beside the
//!   ranges it leaves, and Metal hands out that range's IDs twice, as the runtime does (an M3
//!   Max, macOS 27.0.1, indirect command buffers: 0 then 10 gives 1 to 9, 11 to 31, then 1 to
//!   15; 0 then 20, 21 to 31, 1 to 19, then 1 to 10; 0 then 40, 41 to 63, 1 to 31, then 1 to 16;
//!   0 then 31, 1 to 30, then 1 to 50; 0 then 32, 1 to 31, then 33 on, the first range equal to
//!   the one left; textures, 0 then 10, 11 to 15, 1 to 9, then 1 to 31).
//!   A call holding an ID of 2^32 or more is dropped whole, silently, and a later call is the
//!   reservation (2^32 to 2^63, alone or beside 10, for textures, sampler states and indirect
//!   command buffers).
//! - **Limits.** A placement ([`At`](IdPlacement::At)) of [`ID_LIMIT`] or more ended Metal's
//!   process for every kind (probes 3 and 4), and of [`PIPELINE_ID_LIMIT`] or more for pipeline
//!   states (a probe): it is refused `Invalid`. Metal itself was unreliable above about 9000 forced
//!   texture IDs (some runs ended there). A reservation past them is Metal's (16382, 16383, 20000,
//!   65536 and 1,000,000 of each type were taken, and plain IDs past them handed out: 20000 gives
//!   20001 next), and the runtime's. A texture or sampler state ID of [`RESERVED_POOL_LIMIT`]
//!   (2^28) or more, whose pool would hold 2^29 IDs, ended Metal's process (2^27 + 5 was taken,
//!   2^28 + 5 to 2^32 - 1 ended it, at the reservation or the first allocation after it): refused
//!   `Invalid`. An indirect command buffer ID is reserved to 2^31 - 1, and plain IDs are handed
//!   out past it, each buffer made and run (an M3 Max, macOS 27.0.1: 2^23 + 5, 2^30 + 5 and
//!   2^31 - 2 gave 2^23 + 6, 2^30 + 6 and 2^31 - 1 next); one of [`RECORDS_RESERVED_LIMIT`] (2^31)
//!   to 2^32 - 1 never returned there (Metal's pool spins), and is refused `Invalid`
//!   (`Problem::ReservationHangs`, which says so). An M4's
//!   Metal ended the process at the first allocation after a reservation of 2^30 + 5 to 2^31 - 1,
//!   which the M3 Max, of the same GPU family (Apple9), made and ran: the runtime follows the M3
//!   Max. A placement of one at [`RECORDS_PLACEMENT_LIMIT`] (16384) or more, past Metal's table
//!   of them, is refused `Invalid`. A buffer's records start at its ID's records address, at any
//!   length (Metal made and ran 2^22 commands at 2^26 + 6 and 2^23 + 6); a buffer is refused
//!   `OutOfMemory` only once every ID below 2^31 is held. Pipeline state IDs, measured to 1,000,000, are refused nowhere. The texture table
//!   reaches every ID a heap array can (`heap_sets.rs`), so a texture Metal numbers past its
//!   placement limit is reached as any other. A texture's ID indexes every set-1 array its usage
//!   can bind it through (sampled where it reads, storage where it writes), so an ID past the least
//!   of those arrays' update-after-bind limits is refused `Unsupported`; a reservation is refused
//!   only past the larger, since no usage is known then. A sampler state is never refused for the
//!   sampler arrays: its element there is not its ID but the lowest no live state holds
//!   (`ElementHold`), which the sampler table maps its ID to (its two windows reach IDs 0 to 32767
//!   and 32768 more from the first ID past them, a reservation's, `heap_sets.rs`); only where more
//!   states live than an array holds, or at an ID in neither window, is one left out,
//!   [`Degradation::UnlistedArgumentSampler`](super::Degradation::UnlistedArgumentSampler), and a
//!   function reading it through an argument buffer then reads the neutral sampler.
//! - **Slots.** Each ID has one slot, held by the resource last made with it, placed or plain:
//!   that is what a function reading the ID reaches, and every resource with the ID still
//!   answers it as its own. Once the holder is gone the slot reads the neutral resource, even
//!   while an earlier resource with the ID lives (Metal's kernel read the gone holder's texels
//!   there, probe 2). A pipeline state ID keeps the kind of its gone holder until a state takes
//!   it again (`ResourceIds::released_pipeline`), and the device its last holders, released or
//!   not (`ResourceIds::pipeline_holder_by`): a record the GPU wrote naming the ID runs the state
//!   that held it as the GPU wrote it on Metal, released since or not (`released_programs.rs`).
//! - **When an ID is resolved.** A command reaches texture ID `X` whatever its encoder named,
//!   as Metal's did (the M4 probe: a texture no encoder named, or one a later encoder
//!   made resident, was read through its ID in an argument buffer): the device writes each
//!   slot's holder into its texture table and heap sets as the holder takes the slot, and the
//!   neutral as it goes (`heap_sets.rs`), so a command reads the holder as it runs. Each slot's
//!   generation changes with its holder, so the device writes a slot again only where it
//!   changed. A sampler state is reached likewise: the device writes it into its heap sets' sampler
//!   arrays and its sampler tables as it takes its slot, and the neutral as it goes.

use super::device::{lock, DebugMessage, Device, MessageKind, MessageSeverity};
use super::error::{invalid, Error, Problem};
use super::pipeline::{RenderShared, StateShared};
use super::sampler::SamplerShared;
use super::texture::TextureShared;
use metal2vulkan::device::VkLimit;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::{TextureType, TextureUsage};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

mod numbered;
pub(super) use numbered::{
    function_name, number, FunctionHolds, LibraryHolds, Numbered, NumberedStage,
};

/// A Metal resource ID: what `gpuResourceID` answers and what an argument buffer holds for the
/// resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u64);

/// Where a texture or sampler state goes in its kind's space of resource IDs: an ID the runtime
/// picks, or the caller's (`resourceIndex` with `forceResourceIndex`, a runtime extension).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IdPlacement {
    /// The lowest free ID (the module docs).
    #[default]
    Any,
    /// This ID, which the resource's `gpu_resource_id` then answers.
    At(ResourceId),
    /// No ID, for a texture [`Device::new_texture`](super::Device::new_texture) or
    /// [`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image) makes: its
    /// `gpu_resource_id` answers 0, the null handle, it holds no slot (no ID a function reads
    /// reaches it), and it counts in no plain allocation or reservation and starts no space, as
    /// the runtime's own textures. It is otherwise made as with [`Any`](IdPlacement::Any) (by
    /// `new_texture`, the device's sampler ID included), so no Metal resource's ID depends on it:
    /// for a caller's helper texture that stands behind a Metal one, or a texture over the
    /// caller's swapchain image, so a program's textures take the IDs they take with no window.
    /// Its views take their own IDs at their own placements. A runtime extension, the counterpart
    /// of `AddressPlacement::Outside`: Metal has no such placement. Refused (`Invalid`,
    /// `Problem::OutsidePlacement`) for every other resource that takes its placement.
    Outside,
}

/// The kinds of resource whose IDs a program can reserve
/// (`reserveResourceIndicesForResourceType:indices:indexCount:`, Metal's types 0, 1, 4 and 5).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceIdKind {
    /// Textures: the call's resource type 0.
    Texture,
    /// Sampler states: the call's resource type 1.
    SamplerState,
    /// Pipeline states made with `supportIndirectCommandBuffers`, compute and render: the call's
    /// resource type 4.
    PipelineState,
    /// Indirect command buffers: the call's resource type 5.
    IndirectCommandBuffer,
}

/// The first ID Metal ended the process for at a placement, for every kind but pipeline states
/// (the module docs).
pub(super) const ID_LIMIT: u64 = 16381;

/// The first texture or sampler state ID whose reservation ended Metal's process: its pool would
/// hold 2^29 IDs or more (on an M4: 2^27 + 5 was taken, and 2^28 + 5 to 2^32 - 1 ended the
/// process).
const RESERVED_POOL_LIMIT: u64 = 1 << 28;

/// The first ID Metal ended the process for at a pipeline state's placement (probe 6c on an M4:
/// 16383 placed, 16384 to 65536 ended it).
pub(super) const PIPELINE_ID_LIMIT: u64 = 16384;

/// The first indirect command buffer ID whose reservation Metal never returns from: its pool
/// would hold 2^32 IDs (on an M3 Max, macOS 27.0.1: 2^23 + 5, 2^29 + 5, 2^30 + 5, 3 * 2^29 and
/// 2^31 - 2 were taken and plain IDs handed out past them, each buffer made and run; 2^31, 2^31 +
/// 5 and 2^32 - 1 spun, and 2^32 was dropped). An M4 ended the process at the first allocation
/// after 2^30 + 5 to 2^31 - 1, and at the reservation from 2^31.
const RECORDS_RESERVED_LIMIT: u64 = 1 << 31;

/// The first ID an indirect command buffer's placement is refused at: past Metal's table of
/// them, which Metal writes past (on an M4: 16384 to 50367 made in each of six
/// runs, 100000 ended the process; the consumer's M3 Max runs made or ended it at random from
/// 16384 to 50367).
pub(super) const RECORDS_PLACEMENT_LIMIT: u64 = 16384;

/// The kinds of resource with a space of IDs of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum IdSpace {
    Texture,
    Sampler,
    /// Visible-function tables (measured: a space of their own, beside textures' and sampler
    /// states').
    FunctionTable,
    /// Pipeline states made with `supportIndirectCommandBuffers` (measured: one space of their
    /// own, which compute and render states share).
    Pipeline,
}

impl IdSpace {
    /// The size of the space's first pool ([`Pool`]): 16 for textures and sampler states, 32 for
    /// visible-function tables and pipeline states (and indirect command buffers, `address.rs`),
    /// as Metal's (the module docs).
    fn first_pool(self) -> u64 {
        match self {
            IdSpace::Texture | IdSpace::Sampler => 16,
            IdSpace::FunctionTable | IdSpace::Pipeline => 32,
        }
    }
}

/// The resource holding a slot.
#[derive(Debug)]
pub(super) enum Holder {
    Texture(Weak<TextureShared>),
    Sampler(Weak<SamplerShared>),
    /// A compute pipeline state made with `supportIndirectCommandBuffers`.
    Pipeline(Weak<StateShared>),
    /// A render pipeline state made with `supportIndirectCommandBuffers`, in the space compute
    /// states take theirs from.
    RenderPipeline(Weak<RenderShared>),
}

/// The kind of a pipeline state that held a pipeline state ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StateKind {
    Compute,
    Render,
}

/// One device's resource IDs.
#[derive(Debug)]
pub(super) struct ResourceIds {
    textures: Space,
    samplers: Space,
    function_tables: Space,
    pipelines: Space,
    /// The sampler states' elements of the set-1 sampler arrays, apart from their IDs
    /// ([`ElementHold`]).
    sampler_elements: Elements,
    /// The last generation and hold serial handed out: one counter for both, so a value names
    /// one holder change.
    counter: u64,
    /// The sampler ID the device's first object of a kind that takes one holds for the device's
    /// life ([`ResourceIds::take_device_sampler`]), once taken.
    device_sampler: Option<u64>,
    /// The pipeline state IDs whose holder the program released, which no state took since,
    /// each with its last holder's kind and hold serial (`IdHold::serial`).
    released_pipelines: BTreeMap<u64, (StateKind, u64)>,
    /// The last [`PIPELINE_HOLDERS`] holders of each pipeline state ID, oldest first, released
    /// or not ([`ResourceIds::pipeline_holder_by`]).
    pipeline_holders: BTreeMap<u64, VecDeque<PipelineHolder>>,
    /// The keys pipeline states numbered that a holder may hold, each with its ID
    /// (`numbered.rs`).
    numbered: BTreeMap<numbered::SamplerKey, (u64, Weak<Numbered>)>,
    /// The holds of the live libraries, by content.
    libraries: BTreeMap<metal2vulkan::cache::Hash256, Weak<LibraryHolds>>,
    /// The holds of the functions of the live libraries' function objects, by their library's
    /// bytes and index.
    functions: BTreeMap<(usize, u32), Weak<FunctionHolds>>,
    /// How many holders hold a sampler ID now (the module docs' live count), shared with the
    /// device, which reads it without this lock ([`ResourceIds::sampler_id_reads`]).
    sampler_ids_live: Arc<AtomicU64>,
    /// How many times a sampler ID's slot took a holder or let one go, or a listing found its
    /// holder going ([`samplers`](Self::samplers)), shared with the device, which reads it without
    /// this lock ([`ResourceIds::sampler_id_reads`]): while it holds, the live states
    /// [`samplers`](Self::samplers) lists are the ones it listed.
    sampler_slots_changed: Arc<AtomicU64>,
}

// SAFETY: a `ResourceIds` is `Send` and `Sync` exactly when each of its fields is; the constant
// below makes the compiler prove that of each, its pattern naming every field (one added fails it
// until it is proved too), so these impls claim nothing it does not check. Written out, they end
// the compiler's walk of an object's fields at the IDs where a program's crate proves its own
// objects' `Send` and `Sync`: every object that holds an ID or a numbered sampler reaches this
// table, and from it, through its holders' weak references, nearly every other object, a walk
// that ran within four levels of rustc's default recursion limit
// (`recursion_depth_exceeding_limit`), which a program's crate keeps
// (`tests/consumer_auto_traits.rs`).
unsafe impl Send for ResourceIds {}
// SAFETY: as for `Send` above.
unsafe impl Sync for ResourceIds {}

// The compiler's proof of what `ResourceIds`'s `Send` and `Sync` impls rest on: each field's.
const _: fn(&ResourceIds) = |ids| {
    fn shared<T: Send + Sync>(_: &T) {}
    let ResourceIds {
        textures,
        samplers,
        function_tables,
        pipelines,
        sampler_elements,
        counter,
        device_sampler,
        released_pipelines,
        pipeline_holders,
        numbered,
        libraries,
        functions,
        sampler_ids_live,
        sampler_slots_changed,
    } = ids;
    shared(textures);
    shared(samplers);
    shared(function_tables);
    shared(pipelines);
    shared(sampler_elements);
    shared(counter);
    shared(device_sampler);
    shared(released_pipelines);
    shared(pipeline_holders);
    shared(numbered);
    shared(libraries);
    shared(functions);
    shared(sampler_ids_live);
    shared(sampler_slots_changed);
};

/// What a device reads of its sampler IDs without their lock: the live count and the sampler
/// pool's size (the module docs), each changed under the lock by the step that changes it.
#[derive(Clone, Debug)]
pub(super) struct SamplerIdReads {
    pub(super) live: Arc<AtomicU64>,
    pub(super) pool_size: Arc<AtomicU64>,
    /// The count of changes of the sampler IDs' slots (`ResourceIds::sampler_slots_changed`).
    pub(super) slots_changed: Arc<AtomicU64>,
}

impl Default for ResourceIds {
    /// Every space with nothing handed out, reserved or held.
    fn default() -> ResourceIds {
        ResourceIds {
            textures: Space::new(IdSpace::Texture),
            samplers: Space::new(IdSpace::Sampler),
            function_tables: Space::new(IdSpace::FunctionTable),
            pipelines: Space::new(IdSpace::Pipeline),
            sampler_elements: Elements::default(),
            counter: 0,
            device_sampler: None,
            released_pipelines: BTreeMap::new(),
            pipeline_holders: BTreeMap::new(),
            numbered: BTreeMap::new(),
            libraries: BTreeMap::new(),
            functions: BTreeMap::new(),
            sampler_ids_live: Arc::default(),
            sampler_slots_changed: Arc::default(),
        }
    }
}

impl ResourceIds {
    fn space(&mut self, space: IdSpace) -> &mut Space {
        match space {
            IdSpace::Texture => &mut self.textures,
            IdSpace::Sampler => &mut self.samplers,
            IdSpace::FunctionTable => &mut self.function_tables,
            IdSpace::Pipeline => &mut self.pipelines,
        }
    }

    fn next(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }

    /// A value of the counter past every hold serial handed out so far: a holder whose serial
    /// is below it took its ID before now ([`ResourceIds::pipeline_holder_by`]).
    pub(super) fn stamp(&mut self) -> u64 {
        self.next()
    }

    /// The state that held pipeline state ID `id` at `stamp` ([`ResourceIds::stamp`]): the last
    /// to take it before then, released since or not; none where none of the ID's last
    /// [`PIPELINE_HOLDERS`] holders took it before then.
    pub(super) fn pipeline_holder_by(&self, id: u64, stamp: u64) -> Option<HolderThen> {
        let holder = self
            .pipeline_holders
            .get(&id)?
            .iter()
            .rev()
            .find(|holder| holder.serial < stamp)?;
        let holds = self
            .pipelines
            .slots
            .get(&id)
            .is_some_and(|slot| slot.serial == holder.serial);
        Some(HolderThen {
            kind: holder.kind,
            serial: holder.serial,
            identifier: holder.identifier,
            holds,
        })
    }

    /// The count of held sampler IDs and the sampler pool's size (the module docs), which the
    /// device reads without this lock.
    pub(super) fn sampler_id_reads(&self) -> SamplerIdReads {
        SamplerIdReads {
            live: Arc::clone(&self.sampler_ids_live),
            pool_size: Arc::clone(&self.samplers.pool_size),
            slots_changed: Arc::clone(&self.sampler_slots_changed),
        }
    }

    /// A sampler ID's slot took a holder or let one go, or a listing found its holder going: the
    /// device's live states may differ.
    fn sampler_slot_changed(&self) {
        self.sampler_slots_changed.fetch_add(1, Ordering::AcqRel);
    }

    /// A holder took a sampler ID (`taken`) or gave one back: the live count follows.
    fn count_sampler_id(&self, taken: bool) {
        if taken {
            self.sampler_ids_live.fetch_add(1, Ordering::AcqRel);
        } else {
            self.sampler_ids_live.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Reserve `ids` of `space` from plain allocation ([`Pool::reserve`]): false, reserving
    /// nothing, once the space handed out an ID or took a reservation (Metal ignores the call
    /// then). A call of no ID reserves nothing and leaves the space open, as Metal's does.
    pub(super) fn reserve(&mut self, space: IdSpace, ids: &[ResourceId]) -> bool {
        self.space(space).reserve(ids)
    }

    /// Whether `space`'s pool handed out an ID or took a reservation ([`Pool::started`]).
    pub(super) fn started(&mut self, space: IdSpace) -> bool {
        self.space(space).plain.started()
    }

    /// Take the sampler ID Metal's device keeps for itself as it makes its first object of a
    /// kind that keeps one (the module docs): the ID a live pipeline state's numbered device
    /// sampler has, kept now for good (a queue or a texture made after a render
    /// pipeline state took no new ID, and its ID stayed out once the state went), else a plain
    /// sampler ID, handed out like a sampler state's; never given back, so a sampler reservation
    /// after it is ignored. Once per device.
    pub(super) fn take_device_sampler(&mut self) {
        if self.device_sampler.is_none() {
            let numbered = self
                .numbered
                .get(&numbered::SamplerKey::Device)
                .filter(|(_, held)| held.strong_count() > 0)
                .map(|&(id, _)| id);
            // A numbered ID the device keeps is counted already, as its key's.
            let id = match numbered {
                Some(id) => id,
                None => {
                    self.count_sampler_id(true);
                    self.samplers.take_plain()
                }
            };
            self.device_sampler = Some(id);
        }
    }

    /// Make the texture pool, as Metal's device makes it with its first command queue or
    /// indirect command buffer ([`Pool::make`]): a texture reservation before any texture then
    /// keeps the pool's first range beside its own.
    pub(super) fn make_texture_pool(&mut self) {
        self.textures.plain.make();
    }

    /// Change every texture slot's generation, its holder kept: a change of storage form swapped
    /// in images a slot's elements were written with, so the device writes them again
    /// (`heap_sets::renew`).
    pub(super) fn renew_textures(&mut self) {
        let generation = self.next();
        for slot in self.textures.slots.values_mut() {
            slot.generation = generation;
        }
    }

    /// The generation of texture slot `id`, and its holder with its hold serial
    /// ([`IdHold::serial`]): what a command reading the ID reaches now. A slot never held, or
    /// whose holder is gone, has none. The caller upgrades the holder once it no longer holds
    /// the lock (dropping the last handle of a texture takes it).
    pub(super) fn texture(&self, id: u64) -> (u64, Option<(u64, Weak<TextureShared>)>) {
        match self.textures.slots.get(&id) {
            Some(slot) => {
                let holder = match &slot.holder {
                    Holder::Texture(texture) => Some((slot.serial, Weak::clone(texture))),
                    _ => None,
                };
                (slot.generation, holder)
            }
            None => (0, None),
        }
    }

    /// The holder of sampler ID `id`'s slot, where a sampler state holds it, with its hold serial
    /// ([`IdHold::serial`]): what a function reading the ID reaches now. The caller upgrades the
    /// holder once it no longer holds the lock (dropping the last handle of a state takes it).
    pub(super) fn sampler(&self, id: u64) -> Option<(u64, Weak<SamplerShared>)> {
        let slot = self.samplers.slots.get(&id)?;
        match &slot.holder {
            Holder::Sampler(state) => Some((slot.serial, Weak::clone(state))),
            _ => None,
        }
    }

    /// The holders of pipeline state IDs, in ID order, live or not: the states an indirect
    /// command's record can name. The caller upgrades them once it no longer holds the lock.
    pub(super) fn pipelines(&self) -> Vec<Weak<StateShared>> {
        self.pipelines
            .slots
            .values()
            .filter_map(|slot| match &slot.holder {
                Holder::Pipeline(state) => Some(Weak::clone(state)),
                _ => None,
            })
            .collect()
    }

    /// The holders of render pipeline state IDs, in ID order, live or not: the render states an
    /// indirect render command's record can name. The caller upgrades them once it no longer
    /// holds the lock.
    pub(super) fn render_pipelines(&self) -> Vec<Weak<RenderShared>> {
        self.pipelines
            .slots
            .values()
            .filter_map(|slot| match &slot.holder {
                Holder::RenderPipeline(state) => Some(Weak::clone(state)),
                _ => None,
            })
            .collect()
    }

    /// The kind and hold serial of the state that last held pipeline state ID `id`, where the
    /// program released that state and no state took the ID since: a record the GPU wrote naming
    /// it runs the released state's program on Metal.
    pub(super) fn released_pipeline(&self, id: u64) -> Option<(StateKind, u64)> {
        self.released_pipelines.get(&id).copied()
    }

    /// The live holders of sampler IDs, in ID order: the states a sampler table lists, with the
    /// count of the slots' changes they are live at (`sampler_slots_changed`). A state going on
    /// another thread, its last handle gone but its ID's slot not yet given back (whose step
    /// counts a change, `IdHold`'s drop), is left out, and counted a change here first: a listing
    /// taken while it lived, which a bind state may keep by its count, is then at another count
    /// than this one, so every listing at one count lists the same states.
    pub(super) fn samplers(&self) -> (u64, Vec<Arc<SamplerShared>>) {
        let mut going = false;
        let states = self
            .samplers
            .slots
            .values()
            .filter_map(|slot| match &slot.holder {
                Holder::Sampler(state) => {
                    let live = state.upgrade();
                    going |= live.is_none();
                    live
                }
                _ => None,
            })
            .collect();
        if going {
            self.sampler_slot_changed();
        }
        let changed = self.sampler_slots_changed.load(Ordering::Acquire);
        (changed, states)
    }
}

/// One space.
#[derive(Debug)]
struct Space {
    plain: Pool,
    /// By ID, in order: a sampler table lists its states so.
    slots: BTreeMap<u64, Slot>,
    /// The pool's size, as the step that grew it left it, for a read without the IDs' lock
    /// ([`ResourceIds::sampler_id_reads`]). Only a plain ID or a reservation grows a pool.
    pool_size: Arc<AtomicU64>,
}

impl Space {
    /// `space` with nothing handed out, reserved or held.
    fn new(space: IdSpace) -> Space {
        Space {
            plain: Pool::new(space.first_pool()),
            slots: BTreeMap::new(),
            pool_size: Arc::new(AtomicU64::new(space.first_pool())),
        }
    }

    /// Reserve `ids` ([`ResourceIds::reserve`]).
    fn reserve(&mut self, ids: &[ResourceId]) -> bool {
        let reserved = self.plain.reserve(ids.iter().map(|id| id.0));
        self.pool_size.store(self.plain.size, Ordering::Release);
        reserved
    }

    /// The ID plain allocation hands out next ([`Pool::take`]), now handed out.
    fn take_plain(&mut self) -> u64 {
        let id = self.plain.take();
        self.pool_size.store(self.plain.size, Ordering::Release);
        id
    }
}

/// Plain allocation of one space as Metal's pool hands IDs out (the module docs): free ranges
/// inside a pool whose size doubles. The ranges may overlap where a texture pool the device made
/// before its reservation keeps its first range ([`Pool::make`]). An ID two ranges hold is handed
/// out from each, as Metal's pool hands it out: while the resource it went to first lives, a
/// second one then has it too (the module docs' slots).
///
/// A new range equal to a free one is that one ([`Pool::add`]): the first range a reservation
/// leaves (16 reserved after a queue gives 1 to 15 once), and an ID two textures held, given back
/// by both with nothing free beside it (5 twice gives 5 once). A range that grows to equal one is
/// a second ([`Pool::put`]): 5, 6, 5 and 6 given back, each held twice, gives 5, 6, 5, 6, where
/// 5, 5, 6, 6 gives 6, 5, 6 (on an M4). Metal's pool behaves as a set of ranges
/// whose joins change a range in place.
#[derive(Debug)]
pub(super) struct Pool {
    /// The pool's size: every ID below it is free, handed out or reserved.
    size: u64,
    /// The pool's first size, whose range from 1 a made pool keeps beside its reservation's.
    first: u64,
    /// The free ranges, (start, end) with the end exclusive, each with how many copies the pool
    /// holds: none touching another unless they overlap.
    ranges: BTreeMap<(u64, u64), u32>,
    /// The same ranges as (end, start), for an ID given back to join the range ending at it.
    ends: BTreeMap<(u64, u64), u32>,
    /// The same ranges as (length, start): the first is the one plain allocation takes from.
    fits: BTreeMap<(u64, u64), u32>,
    /// Whether the pool handed out an ID or took a reservation ([`Pool::start`]): Metal's pool
    /// ignores a reservation then ("reserve() must be called before any allocation takes
    /// place").
    started: bool,
    /// Whether the device made the pool before anything started it ([`Pool::make`]).
    made: bool,
}

impl Pool {
    /// A pool of `size` IDs, each above 0 free.
    pub(super) fn new(size: u64) -> Pool {
        let mut pool = Pool {
            size,
            first: size,
            ranges: BTreeMap::new(),
            ends: BTreeMap::new(),
            fits: BTreeMap::new(),
            started: false,
            made: false,
        };
        pool.add(1, size);
        pool
    }

    /// Whether the pool handed out an ID or took a reservation: a reservation is ignored then.
    pub(super) fn started(&self) -> bool {
        self.started
    }

    /// Count a resource of the space as an allocation, as Metal counts one placed at an ID:
    /// a reservation after it is ignored.
    pub(super) fn start(&mut self) {
        self.started = true;
    }

    /// The device made the pool before anything started it, as Metal's device makes its texture
    /// pool with its first command queue or indirect command buffer: a
    /// reservation then keeps the pool's first range beside the ranges it leaves.
    pub(super) fn make(&mut self) {
        if !self.started {
            self.made = true;
        }
    }

    /// Reserve `ids` ([`Pool::keep`]): false, reserving nothing, once the pool started. 0, the
    /// null handle, is no ID: a call of no ID leaves the pool as it was, and one of 0 alone leaves
    /// it made ([`Pool::make`]) before it started, and is ignored after (Metal logs it then, as
    /// any late call), as Metal's does. A made pool keeps its first range, from 1 to its first
    /// size, beside the ranges the reservation leaves, as Metal's does.
    pub(super) fn reserve(&mut self, ids: impl IntoIterator<Item = u64>) -> bool {
        let all: Vec<u64> = ids.into_iter().collect();
        let ids: Vec<u64> = all.iter().copied().filter(|&id| id != 0).collect();
        if ids.is_empty() {
            if all.is_empty() {
                return true;
            }
            self.make();
            return !self.started;
        }
        if self.started {
            return false;
        }
        self.started = true;
        for id in ids {
            self.keep(id);
        }
        if self.made {
            self.add(1, self.first);
        }
        true
    }

    /// A new free range `[start, end)`, unless empty: none where an equal one is free, as
    /// Metal's pool adds a new range (the type's docs).
    fn add(&mut self, start: u64, end: u64) {
        if !self.ranges.contains_key(&(start, end)) {
            self.put(start, end);
        }
    }

    /// Free range `[start, end)`, unless empty, a copy beside an equal one: what a range a join
    /// or a take changed in place becomes (the type's docs).
    fn put(&mut self, start: u64, end: u64) {
        if start < end {
            *self.ranges.entry((start, end)).or_default() += 1;
            *self.ends.entry((end, start)).or_default() += 1;
            *self.fits.entry((end - start, start)).or_default() += 1;
        }
    }

    /// One copy of free range `[start, end)`, taken out.
    fn remove(&mut self, start: u64, end: u64) {
        for (map, key) in [
            (&mut self.ranges, (start, end)),
            (&mut self.ends, (end, start)),
            (&mut self.fits, (end - start, start)),
        ] {
            if let Some(copies) = map.get_mut(&key) {
                *copies -= 1;
                if *copies == 0 {
                    map.remove(&key);
                }
            }
        }
    }

    /// The first free range ending at `end`, as (start, end).
    fn ending_at(&self, end: u64) -> Option<(u64, u64)> {
        let (&(_, start), _) = self.ends.range((end, 0)..=(end, u64::MAX)).next()?;
        Some((start, end))
    }

    /// Double the pool: its new IDs join the free range that ends where it ended.
    fn grow(&mut self) {
        let old = self.size;
        self.size = old.saturating_mul(2);
        match self.ending_at(old) {
            Some((start, end)) => {
                self.remove(start, end);
                self.put(start, self.size);
            }
            None => self.add(old, self.size),
        }
    }

    /// Keep `id` from plain allocation for good: the pool grows past it, and the free range
    /// holding it splits around it. The caller bounds `id` (below 2^32). The ranges do not
    /// overlap yet (a made pool's first range joins them after).
    fn keep(&mut self, id: u64) {
        while id >= self.size {
            self.grow();
        }
        let Some((&(start, end), _)) = self.ranges.range(..=(id, u64::MAX)).next_back() else {
            return;
        };
        if id < end {
            self.remove(start, end);
            self.add(start, id);
            self.add(id + 1, end);
        }
    }

    /// The ID plain allocation hands out next, now handed out: the lowest of the smallest free
    /// range, the lower range of two alike; the pool doubles where none is free. An ID that is
    /// out where overlapping ranges hold it twice is handed out again, as Metal's pool hands it
    /// out.
    pub(super) fn take(&mut self) -> u64 {
        self.started = true;
        loop {
            if let Some((&(length, start), _)) = self.fits.first_key_value() {
                self.remove(start, start + length);
                self.put(start + 1, start + length);
                return start;
            }
            self.grow();
        }
    }

    /// The ID [`Pool::take`] hands out next, not handed out.
    pub(super) fn next(&self) -> u64 {
        match self.fits.first_key_value() {
            Some((&(_, start), _)) => start,
            None => self.size,
        }
    }

    /// `id`, handed out, free again: it joins the free ranges either side of it, the one ending
    /// at it and the one starting past it, which change in place; with neither, it is a new
    /// range (the type's docs).
    pub(super) fn give(&mut self, id: u64) {
        let below = self.ending_at(id);
        let above = self
            .ranges
            .range((id + 1, 0)..=(id + 1, u64::MAX))
            .next()
            .map(|(&range, _)| range);
        for &(start, end) in below.iter().chain(&above) {
            self.remove(start, end);
        }
        let start = below.map_or(id, |(start, _)| start);
        let end = above.map_or(id + 1, |(_, end)| end);
        if below.is_none() && above.is_none() {
            self.add(start, end);
        } else {
            self.put(start, end);
        }
    }

    /// Whether no ID is handed out or reserved.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.ranges.len() == 1 && self.ranges.get(&(1, self.size)) == Some(&1)
    }
}

/// The sampler states' set-1 elements ([`ElementHold`]): every element up to `highest` that is
/// not free is held.
#[derive(Debug, Default)]
struct Elements {
    highest: u64,
    free: BTreeSet<u64>,
}

impl Elements {
    /// The lowest element above 0 not held, now held.
    fn take(&mut self) -> u64 {
        if let Some(n) = self.free.pop_first() {
            return n;
        }
        self.highest += 1;
        self.highest
    }

    /// `n`, held, free again; the top shrinks past every free element.
    fn give(&mut self, n: u64) {
        self.free.insert(n);
        while self.highest > 0 && self.free.remove(&self.highest) {
            self.highest -= 1;
        }
    }
}

/// How many of each pipeline state ID's holders the device keeps
/// ([`ResourceIds::pipeline_holder_by`]), and a buffer the programs of
/// (`released_programs.rs`).
pub(super) const PIPELINE_HOLDERS: usize = 4;

/// A pipeline state that took an ID: its kind, its hold serial and its `uniqueIdentifier`.
#[derive(Clone, Copy, Debug)]
struct PipelineHolder {
    kind: StateKind,
    serial: u64,
    identifier: u64,
}

/// The state that held a pipeline state ID at a stamp ([`ResourceIds::pipeline_holder_by`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HolderThen {
    pub(super) kind: StateKind,
    /// Its hold serial (`IdHold::serial`).
    pub(super) serial: u64,
    /// Its `uniqueIdentifier`.
    pub(super) identifier: u64,
    /// It holds the ID still: the program has not released it.
    pub(super) holds: bool,
}

/// One ID's slot, while its holder lives: the holder and its hold serial.
#[derive(Debug)]
struct Slot {
    holder: Holder,
    serial: u64,
    /// Changes with the holder.
    generation: u64,
}

/// A resource's ID, held until the resource is gone.
#[derive(Debug)]
pub(super) struct IdHold {
    ids: Arc<Mutex<ResourceIds>>,
    space: IdSpace,
    id: u64,
    /// Whether plain allocation handed the ID out, so it is plain allocation's again once the
    /// resource is gone.
    plain: bool,
    serial: u64,
}

impl IdHold {
    /// An ID of `space` among `ids` at `placement` (checked against the limits by the caller),
    /// not yet holding its slot ([`IdHold::hold`]); for [`IdPlacement::Outside`], none
    /// ([`IdHold::internal`]'s).
    pub(super) fn take(
        ids: &Arc<Mutex<ResourceIds>>,
        space: IdSpace,
        placement: IdPlacement,
    ) -> IdHold {
        if placement == IdPlacement::Outside {
            return IdHold::internal(ids, space);
        }
        let mut all = lock(ids);
        let serial = all.next();
        let of = all.space(space);
        of.plain.start();
        let (id, plain) = match placement {
            IdPlacement::At(ResourceId(id)) if id > 0 => (id, false),
            _ => (of.take_plain(), true),
        };
        if space == IdSpace::Sampler {
            all.count_sampler_id(true);
        }
        IdHold {
            ids: Arc::clone(ids),
            space,
            id,
            plain,
            serial,
        }
    }

    /// The ID of a resource of the runtime's own (a scratch or check image): 0, the null
    /// handle, which no slot has and no plain allocation or reservation counts, so no ID a
    /// program reads or reserves ever names it.
    pub(super) fn internal(ids: &Arc<Mutex<ResourceIds>>, space: IdSpace) -> IdHold {
        let serial = lock(ids).next();
        IdHold {
            ids: Arc::clone(ids),
            space,
            id: 0,
            plain: false,
            serial,
        }
    }

    /// Make `holder`, the resource of this ID, its slot's holder.
    pub(super) fn hold(&self, holder: Holder) {
        let mut all = lock(&self.ids);
        let generation = all.next();
        if self.space == IdSpace::Pipeline {
            all.released_pipelines.remove(&self.id);
        }
        all.space(self.space).slots.insert(
            self.id,
            Slot {
                holder,
                serial: self.serial,
                generation,
            },
        );
        if self.space == IdSpace::Sampler {
            all.sampler_slot_changed();
        }
    }

    /// Make `holder`, a pipeline state of `kind` whose `uniqueIdentifier` is `identifier`, its
    /// slot's holder ([`hold`](Self::hold)), and the ID's last holder
    /// ([`ResourceIds::pipeline_holder_by`]).
    pub(super) fn hold_state(&self, holder: Holder, kind: StateKind, identifier: u64) {
        self.hold(holder);
        let mut all = lock(&self.ids);
        let holders = all.pipeline_holders.entry(self.id).or_default();
        if holders.len() == PIPELINE_HOLDERS {
            holders.pop_front();
        }
        holders.push_back(PipelineHolder {
            kind,
            serial: self.serial,
            identifier,
        });
    }

    /// The ID.
    pub(super) fn id(&self) -> ResourceId {
        ResourceId(self.id)
    }

    /// The hold's serial: what its slot names its holder by while it holds it
    /// ([`ResourceIds::texture`]), one value per hold.
    pub(super) fn serial(&self) -> u64 {
        self.serial
    }
}

impl Drop for IdHold {
    fn drop(&mut self) {
        let mut all = lock(&self.ids);
        if self.space == IdSpace::Sampler && self.id != 0 {
            all.count_sampler_id(false);
        }
        let space = all.space(self.space);
        if self.plain {
            space.plain.give(self.id);
        }
        // The slot goes with its holder: one never held reads as it does.
        if space
            .slots
            .get(&self.id)
            .is_some_and(|slot| slot.serial == self.serial)
        {
            let kind = match space.slots.remove(&self.id).map(|slot| slot.holder) {
                Some(Holder::Pipeline(_)) => Some(StateKind::Compute),
                Some(Holder::RenderPipeline(_)) => Some(StateKind::Render),
                _ => None,
            };
            if let Some(kind) = kind {
                all.released_pipelines.insert(self.id, (kind, self.serial));
            }
            if self.space == IdSpace::Sampler {
                all.sampler_slot_changed();
            }
        }
    }
}

/// A sampler state's element of the set-1 sampler arrays, held until the state is gone: the
/// lowest above 0 no live state holds, whatever its resource ID (the module docs). The tables
/// map the state's ID to it (`bind::ArgumentSampler::element`).
#[derive(Debug)]
pub(super) struct ElementHold {
    ids: Arc<Mutex<ResourceIds>>,
    element: u64,
}

impl ElementHold {
    /// The lowest element no live sampler state of `ids`' device holds, now held.
    pub(super) fn take(ids: &Arc<Mutex<ResourceIds>>) -> ElementHold {
        let element = lock(ids).sampler_elements.take();
        ElementHold {
            ids: Arc::clone(ids),
            element,
        }
    }

    /// The element.
    pub(super) fn element(&self) -> u64 {
        self.element
    }
}

impl Drop for ElementHold {
    fn drop(&mut self) {
        lock(&self.ids).sampler_elements.give(self.element);
    }
}

impl Device {
    /// `reserveResourceIndicesForResourceType:indices:indexCount:` (Apple's private call, a
    /// runtime extension): keep `ids` of `kind` from plain allocation for the device's life,
    /// plain IDs then numbered as Metal numbers them (the module docs). Only the device's first
    /// call of `kind` with an ID counts: a later one, or one once the device handed out an ID of
    /// `kind`, is ignored whatever it holds (an ID of 2^32 or more, 0 alone, or one refused
    /// below), as Metal's is, and the device says so in one diagnostic line (through its message
    /// sink, else on stderr), as Metal logs it. A call of no ID changes nothing.
    ///
    /// Before that, a call holding an ID of 2^32 or more is dropped whole, silently, and leaves
    /// the kind's reservation to a later call, as Metal's is; 0, the null handle, reserves nothing. A call
    /// of 0 alone, before any ID of `kind`, leaves `kind`'s pool to keep its first range beside a
    /// later reservation's ranges, so plain allocation hands that range's IDs out twice, as
    /// Metal's does (the module docs).
    ///
    /// Refused, reserving nothing, where the call is not ignored: a texture or sampler state ID
    /// of 2^28 or more, which ends
    /// Metal's process (`Invalid`, `Problem::ResourceId`); an indirect command buffer ID of 2^31
    /// or more, at which Metal's reservation never returns (`Invalid`,
    /// `Problem::ReservationHangs`); and a texture ID no set-1 array of the device can hold
    /// (`Unsupported`, naming the limit). Plain indirect command buffer IDs past a reservation
    /// are Metal's, to 2^31 - 1, each buffer made at any length.
    pub fn reserve_resource_ids(
        &self,
        kind: ResourceIdKind,
        ids: &[ResourceId],
    ) -> Result<(), Error> {
        const CALL: &str = "reserve_resource_ids";
        // Metal ignores a late call before it looks at what the call holds (an M3 Max, macOS
        // 27.0.1: after a buffer or a reservation, 2^31, 2^31 + 5, 2^32 - 1, 2^32 + 5 and 0 alone
        // of indirect command buffers, and 2^28 + 5 and 2^31 + 5 of textures, each logged and
        // ignored, plain IDs going on as before).
        if !ids.is_empty() && self.reservation_closed(kind) {
            self.ignore_late_reservation(kind, ids.len());
            return Ok(());
        }
        if ids.iter().any(|id| id.0 > u64::from(u32::MAX)) {
            return Ok(());
        }
        for &ResourceId(id) in ids {
            match kind {
                // Whatever texture is placed at it: an ID no texture array can hold.
                ResourceIdKind::Texture => {
                    check_limit(CALL, id, RESERVED_POOL_LIMIT)?;
                    let [sampled, storage] = self.texture_arrays();
                    self.check_capacity(id, sampled.max(storage))?;
                }
                ResourceIdKind::SamplerState => check_limit(CALL, id, RESERVED_POOL_LIMIT)?,
                ResourceIdKind::IndirectCommandBuffer if id >= RECORDS_RESERVED_LIMIT => {
                    return Err(invalid(
                        CALL,
                        Problem::ReservationHangs {
                            id,
                            limit: RECORDS_RESERVED_LIMIT,
                        },
                    ));
                }
                ResourceIdKind::IndirectCommandBuffer => {}
                ResourceIdKind::PipelineState => {}
            }
        }
        let space = |space| lock(&self.shared().resource_ids).reserve(space, ids);
        let taken = match kind {
            ResourceIdKind::Texture => space(IdSpace::Texture),
            ResourceIdKind::SamplerState => space(IdSpace::Sampler),
            ResourceIdKind::PipelineState => space(IdSpace::Pipeline),
            ResourceIdKind::IndirectCommandBuffer => {
                let ids: Vec<u64> = ids.iter().map(|id| id.0).collect();
                lock(&self.shared().addresses).reserve_records(&ids)
            }
        };
        // Another thread's allocation may have come between: the call is late then too.
        if !taken {
            self.ignore_late_reservation(kind, ids.len());
        }
        Ok(())
    }

    /// Whether `kind`'s pool handed out an ID or took a reservation, so a reservation call is
    /// ignored ([`Pool::started`]).
    fn reservation_closed(&self, kind: ResourceIdKind) -> bool {
        let space = |space| lock(&self.shared().resource_ids).started(space);
        match kind {
            ResourceIdKind::Texture => space(IdSpace::Texture),
            ResourceIdKind::SamplerState => space(IdSpace::Sampler),
            ResourceIdKind::PipelineState => space(IdSpace::Pipeline),
            ResourceIdKind::IndirectCommandBuffer => {
                lock(&self.shared().addresses).records_started()
            }
        }
    }

    /// Say that a late reservation call of `count` IDs of `kind` is ignored, as Metal logs it.
    fn ignore_late_reservation(&self, kind: ResourceIdKind, count: usize) {
        const CALL: &str = "reserve_resource_ids";
        self.diagnose(
            "metal2vulkan-late-reservation",
            format!(
                "metal2vulkan: {CALL}: reserve() must be called before any allocation takes \
                 place; the reservation of {count} {kind:?} IDs is ignored, as Metal ignores it"
            ),
        );
    }

    /// How many sampler IDs the device holds now: one for each holder of one (the module docs'
    /// live count). That is each live sampler state (equal requests share one state, as on
    /// Metal; a state placed at an ID counts as one of its own); the device's own sampler once
    /// its first command queue, indirect command buffer or texture of its own memory took it,
    /// for the device's life; and each key that live pipeline states, their function objects or
    /// their libraries numbered (each constexpr sampler by state, a texture read, a texture
    /// write, and the device's sampler for a render pipeline with a fragment function made
    /// before the device took it), once device-wide however many hold it. Reserved IDs
    /// ([`Device::reserve_resource_ids`]) are not counted.
    ///
    /// The count changes inside the call that takes or gives back an ID: once
    /// `new_sampler_state*` or a pipeline state's creation returns, it holds what that call
    /// took, and once the last holder of an ID is dropped, the ID is no longer counted. A plain
    /// atomic read: it takes no lock, so it can be called from any thread, inside any callback.
    /// A runtime extension, with no Metal counterpart: Metal keeps a descriptor in its sampler
    /// heap for each of these IDs, so a caller that models that heap's placement counts them.
    pub fn sampler_ids_live(&self) -> u64 {
        self.shared().sampler_ids.live.load(Ordering::Acquire)
    }

    /// The size of the device's sampler ID pool, the runtime's model of Metal's: 16 on a new
    /// device, doubled (32, 64, 128, ...) by the plain sampler ID that finds no ID below it free
    /// and by a sampler reservation of an ID at or past it (until the pool holds the ID), as
    /// Metal's pool grows (the module docs; [`Device::reserve_resource_ids`]). Plain sampler IDs
    /// are the sampler states', the device's own and those pipeline states number
    /// ([`Device::sampler_ids_live`]). A sampler state placed at an ID
    /// ([`IdPlacement::At`]) does not grow it, as a placed ID is outside plain allocation in the
    /// model, and Metal's sampler heap does not grow for one either (measured by the consumer on
    /// an M3 Max under macOS 26.7: an argument-buffer sampler state placed at 15, 16, 20, 32, 600
    /// or 2000 after bring-up, one per process, moved neither the next range nor the next shared
    /// slot). It never shrinks, as the model's pool never does: IDs given back stay in it. The
    /// consumer measured Metal's sampler-descriptor array grow at the IDs that double the pool (the
    /// 15th, 31st, 63rd and 127th states after the device's own), so this is that array's
    /// element count.
    ///
    /// Read as [`Device::sampler_ids_live`] is: changed inside the call that grows the pool, a
    /// plain atomic read with no lock. A runtime extension, with no Metal counterpart.
    pub fn sampler_id_pool_size(&self) -> u64 {
        self.shared().sampler_ids.pool_size.load(Ordering::Acquire)
    }

    /// Take the sampler ID Metal's device takes as it makes its first object of a kind that
    /// takes one ([`ResourceIds::take_device_sampler`], the module docs): the caller made one.
    pub(super) fn take_device_sampler(&self) {
        lock(&self.shared().resource_ids).take_device_sampler();
    }

    /// Make the texture pool as Metal's device makes it with its first command queue or
    /// indirect command buffer ([`ResourceIds::make_texture_pool`]): the caller made one.
    pub(super) fn make_texture_pool(&self) {
        lock(&self.shared().resource_ids).make_texture_pool();
    }

    /// Refuse `placement` of a texture of `usage` and `texture_type` as the module docs say, for
    /// `call`: past the limit, or past a set-1 array the texture can be bound through (sampled
    /// where its usage reads, storage where it writes; a texture buffer through none); and
    /// [`IdPlacement::Outside`], which only `new_texture` and `new_texture_with_vk_image` take,
    /// neither checking it here.
    pub(super) fn check_texture_placement(
        &self,
        call: &'static str,
        placement: IdPlacement,
        usage: TextureUsage,
        texture_type: TextureType,
    ) -> Result<(), Error> {
        let id = match placement {
            IdPlacement::At(ResourceId(id)) => id,
            IdPlacement::Any => return Ok(()),
            IdPlacement::Outside => return Err(invalid(call, Problem::OutsidePlacement)),
        };
        check_limit(call, id, ID_LIMIT)?;
        if texture_type == TextureType::TextureBuffer {
            return Ok(());
        }
        let [sampled, storage] = self.texture_arrays();
        let arrays = [
            usage.contains(TextureUsage::SHADER_READ).then_some(sampled),
            usage
                .contains(TextureUsage::SHADER_WRITE)
                .then_some(storage),
        ];
        match arrays.into_iter().flatten().min() {
            Some(least) => self.check_capacity(id, least),
            None => Ok(()),
        }
    }

    /// Refuse `placement` of a sampler state past the limit, and [`IdPlacement::Outside`], for
    /// `call`. One no set-1 sampler array can hold is made, and a degradation (`sampler.rs`).
    pub(super) fn check_sampler_placement(
        &self,
        call: &'static str,
        placement: IdPlacement,
    ) -> Result<(), Error> {
        match placement {
            IdPlacement::At(ResourceId(id)) => check_limit(call, id, ID_LIMIT),
            IdPlacement::Any => Ok(()),
            IdPlacement::Outside => Err(invalid(call, Problem::OutsidePlacement)),
        }
    }

    /// Refuse ID `id` past `array`, a set-1 array's capacity and the limit that bounds it.
    fn check_capacity(&self, id: u64, (available, limit): (u64, VkLimit)) -> Result<(), Error> {
        if id < available {
            return Ok(());
        }
        Err(Error::Unsupported(Unsupported::needs(
            Feature::ArgumentBufferTextures,
            Capability::Limit {
                limit,
                required: id + 1,
                available,
            },
        )))
    }

    /// The most elements a set-1 sampled-image array and a set-1 storage-image array can have
    /// on the device (an ID is its element), each with the limit that bounds it.
    fn texture_arrays(&self) -> [(u64, VkLimit); 2] {
        let l = self.compile_device().profile().limits();
        let resources = (
            l.max_per_stage_update_after_bind_resources,
            VkLimit::MaxPerStageUpdateAfterBindResources,
        );
        let sampled = least(&[
            (
                l.max_descriptor_set_update_after_bind_sampled_images,
                VkLimit::MaxDescriptorSetUpdateAfterBindSampledImages,
            ),
            (
                l.max_per_stage_descriptor_update_after_bind_sampled_images,
                VkLimit::MaxPerStageDescriptorUpdateAfterBindSampledImages,
            ),
            resources,
        ]);
        let storage = least(&[
            (
                l.max_descriptor_set_update_after_bind_storage_images,
                VkLimit::MaxDescriptorSetUpdateAfterBindStorageImages,
            ),
            (
                l.max_per_stage_descriptor_update_after_bind_storage_images,
                VkLimit::MaxPerStageDescriptorUpdateAfterBindStorageImages,
            ),
            resources,
        ]);
        [sampled, storage]
    }

    /// The most elements a set-1 sampler array can have on the device: link's count for a
    /// pipeline's (`metal2vulkan::pipeline::heap_sampler_capacity`), 0 where no function reads
    /// one.
    pub(super) fn sampler_array(&self) -> u32 {
        metal2vulkan::pipeline::heap_sampler_capacity(self.compile_device().profile())
    }

    /// Say `text`, a general warning `code` names: through the device's message sink, else on
    /// stderr. Not a degradation: the answer is Metal's, and Metal says it too.
    fn diagnose(&self, code: &str, text: String) {
        let message = DebugMessage {
            severity: MessageSeverity::Warning,
            kind: MessageKind {
                general: true,
                ..MessageKind::default()
            },
            id_name: code.to_owned(),
            id: 0,
            text,
            degradation: None,
        };
        match self.shared().instance.sink() {
            Some(sink) => sink.deliver(&message),
            None => eprintln!("{}", message.text),
        }
    }
}

/// Refuse ID `id` of `limit` or more for `call`: Metal ended the process for one.
fn check_limit(call: &'static str, id: u64, limit: u64) -> Result<(), Error> {
    if id < limit {
        return Ok(());
    }
    Err(invalid(call, Problem::ResourceId { id, limit }))
}

/// `placement` of a pipeline state made with `supportIndirectCommandBuffers`, as Metal keeps it,
/// for `call`: the low 32 bits of the index (probe 6d on an M4: `2^32 + 5` placed the state at
/// 5, and `2^32` and `2^32 + 1` as 0 and 1 are placed), so an index whose low bits are 0 is plain
/// allocation; refused `Invalid(ResourceId)` at [`PIPELINE_ID_LIMIT`] or more, which ended
/// Metal's process (`2^31` and `2^32 - 1` too), and [`IdPlacement::Outside`].
pub(super) fn pipeline_placement(
    call: &'static str,
    placement: IdPlacement,
) -> Result<IdPlacement, Error> {
    let index = match placement {
        IdPlacement::At(ResourceId(index)) => index,
        IdPlacement::Any => return Ok(IdPlacement::Any),
        IdPlacement::Outside => return Err(invalid(call, Problem::OutsidePlacement)),
    };
    let kept = index & u64::from(u32::MAX);
    check_limit(call, kept, PIPELINE_ID_LIMIT)?;
    Ok(IdPlacement::At(ResourceId(kept)))
}

/// The records ID an indirect command buffer's `placement` places it at, as Metal keeps it, for
/// `call`: the low 32 bits of the index (on an M4: `2^32 + 5` placed the buffer at 5: `2^32` is
/// plain allocation), `None` for plain allocation; refused `Invalid(ResourceId)` at
/// [`RECORDS_PLACEMENT_LIMIT`] or more, and `Invalid(OutsidePlacement)`
/// for [`IdPlacement::Outside`].
pub(super) fn records_placement(
    call: &'static str,
    placement: IdPlacement,
) -> Result<Option<u64>, Error> {
    let index = match placement {
        IdPlacement::At(ResourceId(index)) => index,
        IdPlacement::Any => return Ok(None),
        IdPlacement::Outside => return Err(invalid(call, Problem::OutsidePlacement)),
    };
    let kept = index & u64::from(u32::MAX);
    if kept == 0 {
        return Ok(None);
    }
    check_limit(call, kept, RECORDS_PLACEMENT_LIMIT)?;
    Ok(Some(kept))
}

/// The least of `limits` (each a capacity and the limit it is), as a capacity.
fn least(limits: &[(u32, VkLimit)]) -> (u64, VkLimit) {
    limits
        .iter()
        .map(|&(value, limit)| (u64::from(value), limit))
        .min()
        .unwrap_or((0, VkLimit::MaxPerStageUpdateAfterBindResources))
}
