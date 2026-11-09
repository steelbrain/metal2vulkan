//! The resources a command buffer's commands use ([`CommandBuffer::used_resources`], a runtime
//! extension): the buffers, textures and heaps a command buffer holds until it completes, for a
//! program that must know whether committed work still uses a resource before it takes the
//! resource's memory away (the consumer's `detach_backing`).
//!
//! - **What counts.** Every resource a command of an ended encoder uses: one bound at a slot of
//!   a dispatch or draw (whether its function reads it or not), one the command buffer's
//!   encoders made resident (`useResource:`, `useHeap:`, the command buffer's residency,
//!   `residency.rs`), a blit's operands, a render pass's attachments and visibility buffer, an
//!   index, indirect-argument or tessellation-factor buffer, a texture a resource state encoder
//!   maps, and what the commit and its host steps record. The record is the hazard-tracking key
//!   of each such use (`tracker.rs`): an encoder keeps every key its commands' uses name, and its
//!   bindings' (`EncoderCore::touch_keys`), and hands them to its command buffer as it ends.
//! - **Stated by its memory.** Each is listed as what owns the memory it lies in, as the rebind
//!   rule counts uses (`rebind.rs`, `RebindProblem::EncodedUses`): a heap for a buffer or texture
//!   placed in it (and for a sparse texture, mapped from it); the buffer a texture is over; the
//!   texture a view was made from; else the buffer or texture itself. The device keeps each
//!   program resource's key with that owner (`Owners`): a buffer of its own as it is made
//!   (`Device::visible`), a texture with an image as it is made (`Texture::made`), and a heap
//!   with bytes in `HeapKeys`. The runtime's own resources (scratch, uploads, stand-ins) have no
//!   entry, and are never listed.
//! - **When.** From each encoder's end until the command buffer completes; a completed command
//!   buffer holds nothing and lists nothing. A resource every hold on which is gone is not
//!   listed.
//! - **By encoder** ([`CommandBuffer::ended_encoders`]). The command buffer keeps each ended
//!   encoder's keys and what it made resident apart (`EncoderRecord`), with its kind and the event
//!   waits and signals encoded before it, so a program can tell which of its encoders use a
//!   resource, and where Metal splits the command buffer into submissions. The command buffer's
//!   list is their union (each encoder's listed at the same granularity). A segment's keys (what an
//!   indirect command buffer's commands reach) are its encoder's from the moment it is recorded, at
//!   the commit or by a host step: read from the recording until it is placed, then from the record
//!   (`State::read_ended`).

use super::buffer::{Buffer, CallerMemory, WeakBuffer};
use super::command_buffer::CommandBuffer;
use super::device::lock;
use super::heap::{Heap, WeakHeap};
use super::key_hash::KeyMap;
use super::rebind::Taken;
use super::replay::Replay;
use super::residency::{Residency, Used};
use super::resource::{ObjectId, ResourceKey};
use super::texture::{Texture, TextureMemory, TextureShared};
use super::Device;
use std::collections::{BTreeSet, HashSet};
use std::sync::{Mutex, Weak};

/// The buffers, textures and heaps a command buffer's commands use, each listed as what owns
/// the memory it lies in, each once ([`CommandBuffer::used_resources`]).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UsedResources {
    /// The buffers not placed in a heap, whether a command uses them or a texture over them.
    pub buffers: Vec<Buffer>,
    /// The textures with an image of their own that is not over a buffer or placed in a heap,
    /// whether a command uses them or a view of them.
    pub textures: Vec<Texture>,
    /// The heaps, whether a command uses them or a resource placed in them.
    pub heaps: Vec<Heap>,
}

/// The kind of a command encoder ([`EndedEncoder::kind`]), by the Metal protocol it is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncoderKind {
    /// A blit encoder (`MTLBlitCommandEncoder`).
    Blit,
    /// A compute encoder (`MTLComputeCommandEncoder`).
    Compute,
    /// A render encoder, one render pass (`MTLRenderCommandEncoder`).
    Render,
    /// A resource state encoder (`MTLResourceStateCommandEncoder`).
    ResourceState,
}

/// One encoder a command buffer ended, as [`CommandBuffer::ended_encoders`] states it: a runtime
/// extension, no Metal type. Its kind, its place among the command buffer's event waits and
/// signals, and the resources its commands use, as [`CommandBuffer::used_resources`] lists them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndedEncoder {
    /// Its kind.
    pub kind: EncoderKind,
    /// The event waits (`encode_wait_for_event`) the command buffer encoded before it, each
    /// call counted: one encoder has more than the encoder before it where a wait lies between
    /// them.
    pub event_waits_before: usize,
    /// The event signals (`encode_signal_event`) the command buffer encoded before it, each call
    /// counted, as `event_waits_before` counts waits.
    pub event_signals_before: usize,
    /// What its own commands use, each listed as what owns the memory it lies in, as
    /// [`CommandBuffer::used_resources`] lists the command buffer's: what it bound, made
    /// resident (`use_resource`, `use_heap`), copied, attached or mapped, and what the commands
    /// of the indirect command buffers it executes reach, from the commit (or the host step)
    /// that records them on. Empty once the command buffer completed.
    pub used_resources: UsedResources,
}

/// An ended encoder as its command buffer keeps it, until it completes
/// ([`CommandBuffer::ended_encoders`]).
pub(super) struct EncoderRecord {
    pub(super) kind: EncoderKind,
    pub(super) event_waits_before: usize,
    pub(super) event_signals_before: usize,
    /// The keys its commands' uses and bindings named, and its placed segments'.
    pub(super) keys: BTreeSet<ResourceKey>,
    /// What it made resident (a compute or render encoder's); `None` once the command buffer
    /// completed.
    pub(super) residency: Option<Residency>,
    /// What its commands took of buffers over host pages, and which they named (`rebind.rs`),
    /// and its placed segments': held weakly from the commit, and handed to what the command
    /// buffer retains at its first submission.
    pub(super) taken: Taken,
    /// Its calls, logged (`replay.rs`): a rebind before the command buffer's first submission
    /// encodes it again from them. Kept, with what they hold, until the command buffer's release
    /// (`CommandBufferShared::complete`); `None` where it cannot be encoded again, or need not.
    pub(super) replay: Option<std::sync::Arc<Replay>>,
    /// Its executions' place among the command buffer's (`State::executions`).
    pub(super) executions: std::ops::Range<usize>,
    /// The images it holds in their attachment layout (`texture/held_depth.rs`): the scheduler
    /// moves each there before it.
    pub(super) held: Vec<ResourceKey>,
    /// Why it kept no log (`replay` `None`), where it named a buffer over host pages, for the
    /// rebound notice (`command_buffer/rebound_use.rs`).
    pub(super) unlogged: Option<crate::command_buffer::rebound_use::Unlogged>,
    /// Why a rebind's encoding of it again was refused (`command_buffer/reencode.rs`).
    pub(super) refused: Option<String>,
}

/// What owns the memory under a key, held without keeping it.
#[derive(Clone)]
enum Owner {
    /// A buffer not placed in a heap, and whether it lies over the caller's host pages (a no-copy
    /// buffer, which a commit lists among the live buffers once used: [`Reached`]).
    Buffer(WeakBuffer, bool),
    Texture(Weak<TextureShared>),
    Heap(WeakHeap),
}

impl Owner {
    /// The owner of `buffer`'s memory: its heap, or the buffer.
    fn of_buffer(buffer: &Buffer) -> Owner {
        match buffer.heap() {
            Some(heap) => Owner::Heap(heap.downgrade()),
            None => Owner::Buffer(
                buffer.downgrade(),
                buffer.caller_memory() == Some(CallerMemory::HostPages),
            ),
        }
    }

    /// Whether a commit lists it among the live buffers once used ([`Reached`]): a buffer over
    /// the caller's host pages, or a heap.
    fn reached(&self) -> bool {
        match self {
            Owner::Buffer(_, host_pages) => *host_pages,
            Owner::Texture(_) => false,
            Owner::Heap(_) => true,
        }
    }

    /// The owner of `texture`'s memory (`Texture::memory`).
    fn of_texture(texture: &Texture) -> Owner {
        match texture.memory() {
            TextureMemory::Heap(heap) => Owner::Heap(heap.downgrade()),
            TextureMemory::Buffer(buffer) => Owner::of_buffer(buffer),
            TextureMemory::Own(texture) => Owner::Texture(texture.downgrade()),
        }
    }

    /// The identity of what [`of_buffer`](Self::of_buffer) answers: its heap's, or the buffer's.
    fn id_of_buffer(buffer: &Buffer) -> ObjectId {
        buffer.heap().map_or(buffer.object_id(), Heap::object_id)
    }

    /// The identity of what [`of_texture`](Self::of_texture) answers.
    fn id_of_texture(texture: &Texture) -> ObjectId {
        match texture.memory() {
            TextureMemory::Heap(heap) => heap.object_id(),
            TextureMemory::Buffer(buffer) => Owner::id_of_buffer(buffer),
            TextureMemory::Own(texture) => texture.object_id(),
        }
    }

    /// The identity of what owns the memory of `used`, a resource made resident, as
    /// [`Listing::resident`] lists it; `None` for one it lists nothing of.
    fn id_of_resident(used: &Used) -> Option<ObjectId> {
        match used {
            Used::Buffer(buffer) => Some(Owner::id_of_buffer(buffer)),
            Used::Texture(texture) => Some(Owner::id_of_texture(texture)),
            Used::Heap { heap, .. } => Some(heap.object_id()),
            Used::IndirectCommandBuffer(icb) => icb.heap().map(Heap::object_id),
        }
    }

    /// Whether the owner lives.
    fn lives(&self) -> bool {
        match self {
            Owner::Buffer(buffer, _) => buffer.lives(),
            Owner::Texture(texture) => texture.strong_count() > 0,
            Owner::Heap(heap) => heap.lives(),
        }
    }

    /// Add the owner to `used`, once, while it lives.
    fn list(&self, used: &mut Listing) {
        match self {
            Owner::Buffer(buffer, _) => {
                if let Some(buffer) = buffer.upgrade() {
                    used.buffers.insert(buffer);
                }
            }
            Owner::Texture(texture) => {
                if let Some(texture) = Texture::upgrade(texture) {
                    used.textures.insert(texture);
                }
            }
            Owner::Heap(heap) => {
                if let Some(heap) = heap.upgrade() {
                    used.heaps.insert(heap);
                }
            }
        }
    }
}

/// The device's program resources by hazard-tracking key, each with what owns the memory under
/// it (the module docs). Gone owners' entries are dropped as the registry grows.
#[derive(Default)]
pub(super) struct Owners {
    inner: Mutex<Registry>,
}

#[derive(Default)]
struct Registry {
    /// Each key's owner, with the owner's identity.
    owners: KeyMap<ResourceKey, (Owner, ObjectId)>,
    /// The size at which the next insertion drops the gone owners' entries.
    prune_at: usize,
}

/// The least size at which the registry drops gone owners' entries.
const PRUNE_FLOOR: usize = 256;

impl Owners {
    /// Keep `key` with `owner`, whose identity is `id`, unless the key has one (a texture over a
    /// buffer of the program's may share the buffer's key: the buffer owns it).
    fn insert(&self, key: ResourceKey, owner: Owner, id: ObjectId) {
        let mut registry = lock(&self.inner);
        if registry.owners.len() >= registry.prune_at {
            registry.owners.retain(|_, (owner, _)| owner.lives());
            registry.prune_at = (2 * registry.owners.len()).max(PRUNE_FLOOR);
        }
        registry.owners.entry(key).or_insert((owner, id));
    }

    /// Keep a buffer the program made, by its key: a heap's buffer has its heap's, which
    /// `HeapKeys` holds.
    pub(super) fn buffer(&self, buffer: &Buffer) {
        if buffer.heap().is_none() {
            let id = Owner::id_of_buffer(buffer);
            self.insert(buffer.identity().key, Owner::of_buffer(buffer), id);
        }
    }

    /// Keep a texture the program made, not a view, by the key its uses name, with what owns
    /// the memory its texels lie in: where that key is a buffer's (a texture buffer, an image
    /// aliasing a buffer's bytes) the buffer or its heap has it already.
    pub(super) fn texture(&self, texture: &Texture) {
        let id = Owner::id_of_texture(texture);
        self.insert(texture.use_key(), Owner::of_texture(texture), id);
    }

    /// Add the owners of `keys` that a commit lists among the live buffers once used to
    /// `reached` ([`Reached`]), held weakly: no owner is let go under this lock.
    fn reached<'a>(&self, keys: impl IntoIterator<Item = &'a ResourceKey>, reached: &mut Reached) {
        let registry = lock(&self.inner);
        let owners = keys.into_iter().filter_map(|key| registry.owners.get(key));
        reached.0.extend(
            owners
                .filter(|(owner, _)| owner.reached())
                .map(|(owner, _)| owner.clone()),
        );
    }

    /// The live buffers the program made (`Device::visible`), for the device's recovery to carry
    /// (`device/carry.rs`).
    pub(super) fn buffers(&self) -> Vec<Buffer> {
        let registry = lock(&self.inner);
        (registry.owners.values())
            .filter_map(|(owner, _)| match owner {
                Owner::Buffer(buffer, _) => buffer.upgrade(),
                Owner::Texture(_) | Owner::Heap(_) => None,
            })
            .collect()
    }

    /// Add the owners of `keys` to `used`.
    fn list(&self, keys: &BTreeSet<ResourceKey>, used: &mut Listing) {
        let owners: Vec<Owner> = {
            let registry = lock(&self.inner);
            keys.iter()
                .filter_map(|key| registry.owners.get(key).map(|(owner, _)| owner.clone()))
                .collect()
        };
        for owner in &owners {
            owner.list(used);
        }
    }

    /// Whether one of `keys` has a live owner among `objects`, by its identity: no owner is held,
    /// so none is let go under this lock.
    fn names_any<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a ResourceKey>,
        objects: &[ObjectId],
    ) -> bool {
        let registry = lock(&self.inner);
        keys.into_iter()
            .filter_map(|key| registry.owners.get(key))
            .any(|(owner, id)| objects.contains(id) && owner.lives())
    }
}

/// Whether what `keys` name, or `residency` made resident, has an owner among `objects`, as
/// [`CommandBuffer::used_resources`] would list it: the device's owners of the keys (`shared`), the
/// heaps among them, and the residency's resources and heaps, each by its identity, with no
/// handle cloned.
pub(super) fn names_any<'a>(
    shared: &super::device::DeviceShared,
    keys: impl IntoIterator<Item = &'a ResourceKey> + Clone,
    residency: Option<&Residency>,
    objects: &[ObjectId],
) -> bool {
    if objects.is_empty() {
        return false;
    }
    if shared.owners.names_any(keys.clone(), objects) || shared.heap_keys.names_any(keys, objects) {
        return true;
    }
    let Some(residency) = residency else {
        return false;
    };
    let among = |id: ObjectId| objects.contains(&id);
    residency
        .resources()
        .filter_map(Owner::id_of_resident)
        .any(among)
        || residency.used_heaps().map(Heap::object_id).any(among)
}

/// What the commands of a committing command buffer's encoders that run use and a command reaches
/// unnamed only once used (`LiveBuffers::reach`): of what [`CommandBuffer::used_resources`] would
/// list, the buffers over the caller's host pages and the heaps, each as what owns its memory and
/// held weakly. Gathered under the command buffer's lock ([`Reached::gather`]), and listed without
/// it ([`Reached::list`]): its commit and its first submission list what this holds among the
/// live buffers, and nothing else of the listing. A command buffer that uses neither gathers
/// nothing.
#[derive(Default)]
pub(super) struct Reached(Vec<Owner>);

impl Reached {
    /// Add what the commands use that `keys` name, and what `residency` made resident.
    pub(super) fn gather<'a>(
        &mut self,
        shared: &super::device::DeviceShared,
        keys: impl IntoIterator<Item = &'a ResourceKey> + Clone,
        residency: Option<&Residency>,
    ) {
        shared.owners.reached(keys.clone(), self);
        shared
            .heap_keys
            .each_among(keys, |heap| self.0.push(Owner::Heap(heap)));
        let Some(residency) = residency else {
            return;
        };
        for used in residency.resources() {
            let owner = match used {
                Used::Buffer(buffer) => Owner::of_buffer(buffer),
                Used::Texture(texture) => Owner::of_texture(texture),
                Used::Heap { heap, .. } => Owner::Heap(heap.downgrade()),
                Used::IndirectCommandBuffer(icb) => match icb.heap() {
                    Some(heap) => Owner::Heap(heap.downgrade()),
                    None => continue,
                },
            };
            if owner.reached() {
                self.0.push(owner);
            }
        }
        let heaps = residency
            .used_heaps()
            .map(|heap| Owner::Heap(heap.downgrade()));
        self.0.extend(heaps);
    }

    /// What it holds that lives, listed as [`CommandBuffer::used_resources`] lists it, each once:
    /// the buffers over host pages and the heaps, no texture. Without the command buffer's lock:
    /// what it upgrades may be a resource's last hold once let go.
    pub(super) fn list(self) -> UsedResources {
        let mut used = UsedResources::default();
        for owner in self.0 {
            match owner {
                Owner::Buffer(buffer, _) => used.buffers.extend(buffer.upgrade()),
                Owner::Heap(heap) => used.heaps.extend(heap.upgrade()),
                Owner::Texture(_) => {}
            }
        }
        used.buffers.sort_unstable_by_key(Buffer::object_id);
        used.buffers.dedup_by_key(|buffer| buffer.object_id());
        used.heaps.sort_unstable_by_key(Heap::object_id);
        used.heaps.dedup_by_key(|heap| heap.object_id());
        used
    }
}

/// [`UsedResources`] as it is gathered: each owner once.
#[derive(Default)]
struct Listing {
    buffers: HashSet<Buffer>,
    textures: HashSet<Texture>,
    heaps: HashSet<Heap>,
}

impl Listing {
    /// Add what owns the memory of `used`, a resource made resident.
    fn resident(&mut self, used: &Used) {
        match used {
            Used::Buffer(buffer) => Owner::of_buffer(buffer).list(self),
            Used::Texture(texture) => Owner::of_texture(texture).list(self),
            Used::Heap { heap, .. } => {
                self.heaps.insert(heap.clone());
            }
            Used::IndirectCommandBuffer(icb) => {
                if let Some(heap) = icb.heap() {
                    self.heaps.insert(heap.clone());
                }
            }
        }
    }
}

impl Listing {
    /// Add what owns the memory of each key of `keys` and of what `residency` holds: the device's
    /// owners of the keys (`shared`), the heaps among them, and the residency's resources and
    /// heaps.
    fn add(
        &mut self,
        shared: &super::device::DeviceShared,
        keys: &BTreeSet<ResourceKey>,
        residency: Option<&Residency>,
    ) {
        shared.owners.list(keys, self);
        self.heaps
            .extend(shared.heap_keys.among(keys.iter().copied()));
        if let Some(residency) = residency {
            for used in residency.resources() {
                self.resident(used);
            }
            self.heaps.extend(residency.used_heaps().cloned());
        }
    }

    /// The listing, each list in no particular order.
    fn finish(self) -> UsedResources {
        let Listing {
            buffers,
            textures,
            heaps,
        } = self;
        UsedResources {
            buffers: buffers.into_iter().collect(),
            textures: textures.into_iter().collect(),
            heaps: heaps.into_iter().collect(),
        }
    }
}

/// An ended encoder's record as `CommandBuffer` reads it, to list outside its state's lock
/// (`State::read_ended`).
pub(super) struct ReadRecord {
    pub(super) kind: EncoderKind,
    pub(super) event_waits_before: usize,
    pub(super) event_signals_before: usize,
    /// Every key its commands use and reach.
    pub(super) keys: BTreeSet<ResourceKey>,
    pub(super) residency: Option<Residency>,
}

impl ReadRecord {
    /// What it uses, listed as `used_resources` lists it.
    fn listing(&self, shared: &super::device::DeviceShared) -> UsedResources {
        let mut listing = Listing::default();
        listing.add(shared, &self.keys, self.residency.as_ref());
        listing.finish()
    }
}

/// The union of what `records` use, listed as `used_resources` lists it.
pub(super) fn union_of(
    shared: &super::device::DeviceShared,
    records: &[ReadRecord],
) -> UsedResources {
    let mut listing = Listing::default();
    for record in records {
        listing.add(shared, &record.keys, record.residency.as_ref());
    }
    listing.finish()
}

impl CommandBuffer {
    /// The buffers, textures and heaps the commands of its ended encoders use, each listed as
    /// what owns the memory it lies in, until it completes (a runtime extension; the module docs
    /// of `used_resources.rs`): one bound at a slot of a dispatch or draw, made resident by an
    /// encoder (`use_resource`, `use_heap`), a blit's operand, an attachment, an index or
    /// indirect-argument buffer, a texture a resource state encoder maps; a heap for what is
    /// placed in it, the buffer a texture is over, the texture a view was made from. Empty once
    /// it completed. Each list is in no particular order. It is the union of
    /// [`ended_encoders`](Self::ended_encoders)'s lists.
    ///
    /// What the commands of an indirect command buffer an encoder executes reach is listed from
    /// the commit on, once the commit recorded them; read before the commit, the list holds what
    /// the encoders bound and made resident, but not what such commands reach without either.
    /// For an indirect command buffer the GPU may write, what its commands reach is listed once
    /// the host step before its batch recorded them.
    pub fn used_resources(&self) -> UsedResources {
        self.settle_end();
        let ended = self.state().read_ended(None);
        let used = union_of(self.device().shared(), &ended);
        // Released without the lock: what it holds may be a resource's last handle.
        drop(ended);
        used
    }

    /// Its ended encoders, in the order they ended (the numbering of
    /// [`CommitFailure`](super::CommitFailure)'s encoders), each with its kind, the event waits
    /// and signals encoded before it, and what its own commands use, listed as
    /// [`used_resources`](Self::used_resources) lists the command buffer's: a runtime extension,
    /// no Metal call (docs/API.md, "What a command buffer uses"). Metal splits a command buffer
    /// into submissions by its encoders' kinds and the event commands between them, and fails
    /// every encoder of a submission that uses a resource whose memory was taken away (measured
    /// by a consumer on an M3 Max, macOS 26.7); this is what a program needs to model that, and
    /// [`commit_failed`](Self::commit_failed) with a [`FailedEncoders::Set`](super::FailedEncoders::Set)
    /// fails such a set. An encoder split into parts (a host step's) is one encoder. Once the
    /// command buffer completed, each lists nothing. As for `used_resources`, what the commands of
    /// an indirect command buffer an encoder executes reach is listed from the commit on (for one
    /// the GPU may write, from its host step on), not before.
    pub fn ended_encoders(&self) -> Vec<EndedEncoder> {
        self.settle_end();
        let ended = self.state().read_ended(None);
        let shared = self.device().shared();
        let listed = ended
            .iter()
            .map(|record| EndedEncoder {
                kind: record.kind,
                event_waits_before: record.event_waits_before,
                event_signals_before: record.event_signals_before,
                used_resources: record.listing(shared),
            })
            .collect();
        // Released without the lock: what it holds may be a resource's last handle.
        drop(ended);
        listed
    }
}

impl CommandBuffer {
    /// Whether the commands of its ended encoders use any of `objects`, the identities
    /// ([`Buffer::object_id`], [`Texture::object_id`], [`Heap::object_id`]) of what
    /// [`used_resources`](Self::used_resources) would list: a runtime extension, no Metal call.
    /// Answered without the listing: no handle is cloned or held, and the list's resources are
    /// those of [`used_resources`](Self::used_resources), each as what owns the memory it lies in
    /// (a heap for a resource placed in it, the buffer a texture is over, the texture a view was
    /// made from), so a buffer placed in a heap is never among them, its heap is. False once the
    /// command buffer completed, and for an empty `objects`.
    pub fn uses(&self, objects: &[ObjectId]) -> bool {
        self.settle_end();
        self.state().uses(self.device().shared(), None, objects)
    }

    /// Whether the commands of its ended encoders at `encoders` (their places, the index of
    /// [`ended_encoders`](Self::ended_encoders)) use any of `objects`, as their
    /// [`EndedEncoder::used_resources`] would list them, answered as [`uses`](Self::uses) answers:
    /// a runtime extension, no Metal call. False for a range past its ended encoders.
    pub fn encoders_use(&self, encoders: std::ops::Range<usize>, objects: &[ObjectId]) -> bool {
        self.settle_end();
        (self.state()).uses(self.device().shared(), Some(encoders), objects)
    }
}

impl Device {
    /// `buffer`, one the program made of its own, kept among the device's program resources
    /// (`Owners::buffer`), and among its live buffers, which a command reaches unnamed, unless a
    /// command buffer's use must list it first (a no-copy buffer; `live_buffers.rs`). The
    /// device's recovery carries it from there (`device/carry.rs`), or its maker here, where that
    /// recovery took this generation's buffers before it was kept: a load of a flag on a
    /// generation never recovered.
    pub(super) fn visible(&self, buffer: Buffer) -> Buffer {
        let shared = self.shared();
        shared.owners.buffer(&buffer);
        shared.live_buffers.made(&buffer);
        if shared.carry.taken() && shared.carry.late_buffer(&buffer) {
            self.carry_late(|serving| buffer.carry(serving));
        }
        buffer
    }
}
