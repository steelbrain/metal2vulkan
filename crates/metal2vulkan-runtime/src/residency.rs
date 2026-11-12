//! The resources an encoder makes resident: the compute and render encoders' `use_resource(s)` and
//! `use_heap(s)`.
//!
//! - **Metal's contract.** A function reaches a resource the encoder does not bind (a buffer
//!   whose address an argument buffer or another buffer holds, a texture whose resource ID one
//!   holds) only once the encoder made it resident. The runtime lists each such resource among
//!   the bind state's used resources, so every later dispatch of the encoder reaches it:
//!   - a buffer's range is in the resident set, so a pointer loaded from memory into it is
//!     translated, and one into no resident range misses (it reads zero and drops writes, where
//!     Metal leaves the access undefined). The live buffers Metal reaches unnamed are in the set
//!     its command buffer's root tables search too (`live_buffers.rs`): every buffer of its own,
//!     and a no-copy buffer or a heap's buffer once a command buffer's commands used it; what
//!     misses is such a buffer no command buffer's commands used yet;
//!   - a texture is a member of the command's plan, under its resource ID (`resource_ids.rs`), so
//!     its uses are ordered as a bound texture's are; a handle loaded from memory finds every live
//!     texture, named or not, through the device's texture table (`heap_sets.rs`), as Metal's does;
//!   - a heap's whole range is in the resident set as one used resource (the `Heap` entry), so a
//!     pointer loaded from memory into any buffer placed in it is translated, and each texture made
//!     in it that lives when the encoder uses the heap is made resident as a texture is, and so is
//!     each texture over a buffer placed in it, which Metal's heap holds as it holds the buffer. A
//!     sparse heap holds no buffer and no range of its own: its live sparse textures alone are made
//!     resident. Metal's `useHeap:` makes a heap's resources resident for reading only, so its
//!     usage is `READ`;
//!   - a texture made in a heap after the encoder used it is made resident too, as the encoder
//!     encodes its next command (`sync_heaps`): Metal's `useHeap:` reaches every texture the heap
//!     holds when the command buffer is committed, those made after the call included (SPH heap
//!     probe 1, an M3 Max and an M4, compute and render, every heap type, with shader validation
//!     reporting nothing). One made after a command was encoded is reached by that command too,
//!     through the device's texture table, which holds every live texture as it is made
//!     (`heap_sets.rs`), though its uses are not ordered against the command's;
//!   - an indirect command buffer's records are in the resident set as its buffer's range, so a
//!     function's command encoders reach them through its resource ID, which is their address,
//!     while the buffer holds its ID's slot (a buffer made later with the ID takes it, and then a
//!     function reaches that one's records, or none once it is gone). Made resident for writing
//!     in a command buffer that also runs a function that may call an indirect-command operation,
//!     the buffer is written by the GPU from then on (`write_routes.rs`): its executions are
//!     replayed on the host.
//! - **The command buffer's** (the coordinator's ruling of 2026-10-07). Metal's residency belongs
//!   to the command buffer: on an M3 Max, an indirect command buffer's compute command executed by
//!   one compute encoder wrote a buffer that only a later encoder's `useResource:` named, and one a
//!   later blit filled. So what each encoder made resident joins its command buffer's residency
//!   ([`Residency::join_encoder`]): its heaps as it ends, and its resources where the command
//!   buffer's own work first reads that residency after it ended (`State::joined_residency` in
//!   `command_buffer/mod.rs`), each ended encoder's in the order they ended, so the residency
//!   holds what it would had each joined as it ended, in the same order, and a command buffer whose
//!   work never reads it joins none (`WorkCounts::resident_resources_joined`). The commit makes
//!   resident the textures made since in the heaps it holds (`sync_heaps`). The work recorded at the commit or by a host
//!   step (an indirect command buffer's executions, compute and render, and an indirect dispatch
//!   the host reads) resolves and binds against that residency, every encoder's, made resident
//!   before the call or after it. A compute encoder's resources reach every render stage there, as
//!   it names none. A later compute or render encoder starts with the shadowed textures of that
//!   residency: every live texture is reached through its ID anyway (`heap_sets.rs`), but a shadow
//!   holds a texture's texels only as an encoder that makes it resident copies them in
//!   (`shadow.rs`), so each such encoder copies them in for its commands and orders them against
//!   its other uses, and a function that reads the texture through its ID reads its bytes as they
//!   stand. Its usage is the joined one, and so are its render stages in a render encoder.
//! - **Ordering.** A plan lists a used resource's use where its function reads loaded pointers
//!   (a buffer or a heap) or resolves handles (a texture or a heap), over the whole resource,
//!   reading or writing as the usage says (no usage does both), but reading alone in a stage
//!   whose function stores to no buffer or texture and updates none atomically (the pipeline's
//!   `vertex_writes_memory` and `fragment_writes_memory`): a draw of such functions writes no
//!   resource another encoder made resident for writing, such as an indirect command buffer a
//!   compute encoder writes. The tracker orders it as it orders a bound one
//!   (`binding::command_uses`): a heap as the one resource its tracked buffers are (`heap.rs`). A
//!   shadowed texture is copied in and back as at a slot.
//! - **Tokens.** A resource has one token in the encoder, its position in the order the encoder
//!   first used it, and the union of every usage it was used with; so a plan's shape changes
//!   only when the encoder uses another resource or widens a usage. A resource is found among
//!   those made resident by its identity, so `useResources:count:usage:` of `n` resources costs
//!   `n` lookups, whatever the encoder made resident before.
//! - **Generations.** Each change of what a residency holds (a resource made resident, a usage or
//!   stage widened, a heap used or widened, a live table given or taken) gives it a generation no
//!   other residency of the process had ([`Residency::generation`]). An encoder lists its
//!   residency's share of the bind state (the used resources) again only when the generation, or
//!   what a used resource states besides (a buffer's addresses, an indirect command buffer's hold
//!   on its ID's slot), changed since its last draw or dispatch listed it (`live_bind.rs`,
//!   `compute/live.rs`): so that share is listed once per change, whatever else a command
//!   changes. A compute dispatch that finds the generation, and every other input, its last
//!   dispatch settled with takes what that one settled (`compute.rs`, `DispatchMemo`).
//! - **Copies.** A residency's copies share what it made resident until one of them changes it,
//!   which then takes a copy of its own: so each execution of an indirect command buffer, its
//!   settle at the commit and its host step, which each take the command buffer's residency
//!   (`CommandBuffer::residency`), costs a count, not the resources it holds. So does the kept
//!   key of their recordings ([`kept_key`](Residency::kept_key)), listed once for what a
//!   residency holds while no used resource's statement changes (the **Generations** rule).
//! - **Stages.** A render encoder makes a resource resident for the render stages it names
//!   (`useResource:usage:stages:`, `useHeap:stages:`), joined as the usages are: a draw's
//!   functions reach it in those of their stages alone (the binder's `UsedResource::stages`), and
//!   a tile dispatch only where they hold the tile stage. A compute encoder names none, and its
//!   dispatches reach every resource it made resident.

use super::buffer::{host_cells_moved, Buffer, ReadApart};
use super::device::{lock, Device};
use super::heap::Heap;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::key_hash::KeyMap;
use super::live_buffers::LiveTable;
use super::resource::{ResourceKey, ResourceRef};
use super::texture::Texture;
use super::tracker::{Access, Use, UseRange};
use ash::vk;
use metal2vulkan::bind::{ResidentRange, ResourceClass, UsedResource};
use metal2vulkan::metal::{RenderStages, ResourceUsage};
use metal2vulkan::pipeline::{BindingSource, PipelineLayout, Stage, SyntheticBinding};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// The generation last given to a residency of the process ([`Residency::generation`]).
static GENERATIONS: AtomicU64 = AtomicU64::new(0);

/// A generation no residency of the process had.
fn next_generation() -> u64 {
    GENERATIONS.fetch_add(1, Ordering::Relaxed) + 1
}

/// A resource an encoder made resident.
#[derive(Clone)]
pub(super) enum Used {
    Buffer(Buffer),
    Texture(Texture),
    /// A heap, with its own buffer (a sparse heap, which has none, is not listed: only its
    /// textures are).
    Heap {
        heap: Heap,
        bytes: Buffer,
    },
    /// Its records' buffer.
    IndirectCommandBuffer(IndirectCommandBuffer),
}

/// The resources an encoder made resident, by token, each with its usage and render stages.
#[derive(Clone, Default)]
pub(super) struct Residency {
    /// The resources (the module docs' **Copies**): shared by its copies until one of them
    /// changes them. `None` until it is given one, so a new residency, which every compute and
    /// render encoder's open makes (`CommandBufferShared::inherited`), allocates nothing; read as
    /// [`NOTHING`] ([`held`](Self::held)).
    held: Option<Arc<Held>>,
    /// Whether one of `textures` is sampled through an address-mode map other than the identity
    /// (`Texture::address_mode_map`, fixed as the texture is made; a view's is its texture's):
    /// set as such a texture is made resident, which none is ever made no longer, so a command's
    /// look ([`maps_addresses`](Self::maps_addresses)) reads no texture.
    maps_addresses: bool,
    /// The tokens of the shadowed textures (`shadow.rs`) among the first `textures_looked_at` of
    /// `textures`, in the order they were made resident: what a command buffer's next encoder
    /// starts with ([`shadows`](Self::shadows)) costs their count. Found where that is first
    /// asked ([`find_shadowed`](Self::find_shadowed)), so making a texture resident reads
    /// nothing of it, and each texture is looked at once.
    shadowed: Vec<u32>,
    /// How many of `textures` [`find_shadowed`](Self::find_shadowed) has looked at.
    textures_looked_at: usize,
    /// The heaps the encoder used, each with the render stages it used it in and the number of
    /// the first of its textures not made resident yet (`Heap::textures_from`).
    heaps: Vec<HeapUse>,
    /// Each heap of `heaps` by its place there.
    heap_places: KeyMap<Heap, usize>,
    /// A command buffer's: the device's live buffers at its commit, which an address in no
    /// resource made resident resolves against (`live_buffers.rs`).
    live: Option<Arc<LiveTable>>,
    /// The module docs' generation: 0 while it holds nothing it was given, else one no other
    /// residency of the process had, drawn at its last change.
    generation: u64,
    /// The buffers and heaps' own buffers of `used` by Metal address, made at the first
    /// [`resolve`](Self::resolve) after one was made resident.
    addresses: OnceLock<Arc<AddressIndex>>,
    /// [`heaps`](Self::heaps), found at the generation it holds
    /// ([`resident_heaps`](Self::resident_heaps)): shared by its copies, behind one pointer.
    heaps_found: Option<Arc<(u64, Vec<Heap>)>>,
}

/// What a [`Residency`] made resident, which its copies share.
#[derive(Default)]
struct Held {
    used: Vec<(Used, ResourceUsage, RenderStages)>,
    /// Each resource of `used` by its identity, with its token.
    tokens: KeyMap<Used, u32>,
    /// The tokens of the indirect command buffers among `used`.
    indirect: Vec<u32>,
    /// The tokens of the textures among `used`: a command's look at the textures made resident
    /// costs their count, not every resource's.
    textures: Vec<u32>,
    /// Its kept key ([`Residency::kept_key`]) as last listed, with the moves of no-copy buffers
    /// and the device's views epoch it was listed at: the listing while both hold.
    kept: Mutex<Option<(u64, u64, Arc<KeptListing>)>>,
}

/// A copy, to be changed: its kept key is listed again.
impl Clone for Held {
    fn clone(&self) -> Held {
        Held {
            used: self.used.clone(),
            tokens: self.tokens.clone(),
            indirect: self.indirect.clone(),
            textures: self.textures.clone(),
            kept: Mutex::new(None),
        }
    }
}

/// What a residency given nothing holds. Read only: its kept key is never listed here
/// ([`Residency::kept_key`]), so it holds nothing of any device's.
static NOTHING: Held = Held {
    used: Vec::new(),
    tokens: KeyMap::with_hasher(BuildHasherDefault::new()),
    indirect: Vec::new(),
    textures: Vec::new(),
    kept: Mutex::new(None),
};

impl Held {
    /// `held`, to change: its own (a copy where copies share it, a new one where it held
    /// nothing), its kept key to be listed again.
    fn changed(held: &mut Option<Arc<Held>>) -> &mut Held {
        let held = Arc::make_mut(held.get_or_insert_with(Arc::default));
        *held.kept.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        held
    }
}

/// What a kept recording of commands that resolved and bound against a residency was recorded
/// from (`kept_commands.rs`, [`Residency::kept_key`]): its resources as the bind state states
/// them and each one's key, in token order, and the keys sorted, for a look.
pub(super) struct KeptListing {
    used: Vec<UsedResource>,
    keys: Vec<ResourceKey>,
    sorted: Vec<ResourceKey>,
}

impl KeptListing {
    /// Whether `key` is one of the resources' keys: a look in the sorted keys.
    pub(super) fn holds_key(&self, key: &ResourceKey) -> bool {
        self.sorted.binary_search(key).is_ok()
    }
}

/// Equal where both list the same resources, stated alike, with the same keys.
impl PartialEq for KeptListing {
    fn eq(&self, other: &KeptListing) -> bool {
        self.used == other.used && self.keys == other.keys
    }
}

/// Where each buffer and heap's own buffer a residency made resident lies in Metal's address
/// space: the pieces of the space between their starts and ends in order, each with the first
/// made resident of those that hold it, and the ends, each with the first made resident of
/// those that end there. Made in the logarithm of their count times it; a look costs its
/// logarithm.
pub(super) struct AddressIndex {
    /// `(start, token)`: from `start` to the next piece's, the first resource made resident of
    /// those holding every address there (`None`: none holds them).
    pieces: Vec<(u128, Option<u32>)>,
    /// `(end, token)`, sorted: the first made resident of those that end at `end`.
    ends: Vec<(u128, u32)>,
}

impl AddressIndex {
    /// The index of the buffers and heaps' own buffers of `used`, by token.
    fn of_used(used: &[(Used, ResourceUsage, RenderStages)]) -> AddressIndex {
        let ranges = (used.iter().zip(0u32..)).filter_map(|((used, ..), token)| {
            let buffer = match used {
                Used::Buffer(buffer) | Used::Heap { bytes: buffer, .. } => buffer,
                Used::Texture(_) | Used::IndirectCommandBuffer(_) => return None,
            };
            Some((buffer.metal_address().0, buffer.length(), token))
        });
        AddressIndex::of(ranges)
    }

    /// The index of `ranges`, each `(start, length, token)`.
    pub(super) fn of(ranges: impl Iterator<Item = (u64, u64, u32)>) -> AddressIndex {
        // `(start, end, token)`, sorted by start.
        let mut ranges: Vec<(u128, u128, u32)> = ranges
            .map(|(start, length, token)| {
                let start = u128::from(start);
                (start, start + u128::from(length), token)
            })
            .collect();
        ranges.sort_unstable();
        let mut ends: Vec<(u128, u32)> =
            ranges.iter().map(|&(_, end, token)| (end, token)).collect();
        ends.sort_unstable();
        ends.dedup_by_key(|&mut (end, _)| end);
        let mut bounds: Vec<u128> = (ranges.iter())
            .flat_map(|&(start, end, _)| [start, end])
            .collect();
        bounds.sort_unstable();
        bounds.dedup();
        // A sweep over the bounds: the resources holding each piece, the first made resident
        // on top, those that ended dropped as they reach the top.
        let mut holding = std::collections::BinaryHeap::new();
        let mut next = ranges.iter().peekable();
        let mut pieces: Vec<(u128, Option<u32>)> = Vec::with_capacity(bounds.len());
        for at in bounds {
            while let Some(&&(start, end, token)) = next.peek() {
                if start > at {
                    break;
                }
                holding.push((std::cmp::Reverse(token), end));
                next.next();
            }
            while holding.peek().is_some_and(|&(_, end)| end <= at) {
                holding.pop();
            }
            let first = holding.peek().map(|&(std::cmp::Reverse(token), _)| token);
            if pieces.last().is_none_or(|&(_, last)| last != first) {
                pieces.push((at, first));
            }
        }
        AddressIndex { pieces, ends }
    }

    /// The first resource made resident of those holding `address`, else of those that end
    /// there.
    pub(super) fn first_holding(&self, address: u64) -> Option<u32> {
        let address = u128::from(address);
        let piece = self.pieces.partition_point(|&(start, _)| start <= address);
        let holding = piece.checked_sub(1).and_then(|piece| self.pieces[piece].1);
        holding.or_else(|| {
            let end = self
                .ends
                .binary_search_by_key(&address, |&(end, _)| end)
                .ok()?;
            Some(self.ends[end].1)
        })
    }
}

/// `read` and `write` of all of `buffer`'s bytes.
pub(super) fn whole_use(buffer: &Buffer, read: Access, write: Access) -> Use<ResourceKey> {
    let (identity, range) = buffer.tracked(0..buffer.length());
    Use {
        key: identity.key,
        tracked: identity.tracked,
        range: UseRange::Bytes(range),
        read,
        write,
    }
}

/// A heap an encoder used (`Residency::heaps`).
#[derive(Clone)]
struct HeapUse {
    heap: Heap,
    stages: RenderStages,
    next: u64,
}

impl Used {
    /// The key of the buffer whose bytes it is: a buffer's, a heap's own buffer's; `None` for a
    /// texture and an indirect command buffer.
    pub(super) fn bytes_key(&self) -> Option<ResourceKey> {
        match self {
            Used::Buffer(buffer) => Some(buffer.identity().key),
            Used::Heap { bytes, .. } => Some(bytes.identity().key),
            Used::Texture(_) | Used::IndirectCommandBuffer(_) => None,
        }
    }
}

impl Residency {
    /// What it made resident: [`NOTHING`] until it is given a resource.
    fn held(&self) -> &Held {
        self.held.as_deref().unwrap_or(&NOTHING)
    }

    /// Whether it holds nothing: no resource, heap or live table.
    pub(super) fn is_empty(&self) -> bool {
        self.held().used.is_empty() && self.heaps.is_empty() && self.live.is_none()
    }

    /// Its generation (the module docs): equal to another residency's only where neither changed
    /// since one was cloned from the other, or both hold nothing they were given.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether each indirect command buffer made resident holds its ID's slot as `used`, the bind
    /// state's used resources made of it ([`used_resources`](Self::used_resources)), states: a
    /// buffer made later with the ID takes the slot (`address.rs`) without changing the residency.
    pub(super) fn holds_slots_as(&self, used: &[UsedResource]) -> bool {
        self.held().indirect.iter().all(|&token| {
            let stated = used
                .get(token as usize)
                .map(|used| used.gpu_address.is_some());
            match self.held().used.get(token as usize) {
                Some((Used::IndirectCommandBuffer(icb), ..)) => stated == Some(icb.holds_id_slot()),
                _ => false,
            }
        })
    }

    /// The heap used at `heap`'s place in `heaps`, if the encoder used it.
    fn heap_use(&mut self, heap: &Heap) -> Option<&mut HeapUse> {
        let place = *self.heap_places.get(heap)?;
        self.heaps.get_mut(place)
    }

    /// Note `heap`'s first use, in render `stages`, its textures before number `next` made
    /// resident.
    fn push_heap(&mut self, heap: &Heap, stages: RenderStages, next: u64) {
        self.heap_places.insert(heap.clone(), self.heaps.len());
        self.heaps.push(HeapUse {
            heap: heap.clone(),
            stages,
            next,
        });
    }

    /// Make `resource` resident with `usage` in render `stages` (none for a compute encoder),
    /// joined to the usage and stages it had: the resource, when the encoder first uses it here.
    /// The residency holds what it made resident, and its encoder's record holds the residency
    /// until the command buffer completes (`command_buffer::State::encoder_ended`), so the caller
    /// holds none of it again.
    pub(super) fn use_resource(
        &mut self,
        resource: ResourceRef<'_>,
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Option<Used> {
        let used = match resource {
            ResourceRef::Buffer(buffer) => Used::Buffer(buffer.clone()),
            ResourceRef::Texture(texture) => Used::Texture(texture.clone()),
            ResourceRef::IndirectCommandBuffer(icb) => Used::IndirectCommandBuffer(icb.clone()),
        };
        self.join(used, usage, stages)
    }

    /// Make `heap` (with its own buffer; a sparse heap has none) and its live textures resident
    /// for reading in render `stages` (none for a compute encoder), as Metal's `useHeap:` and
    /// `useHeap:stages:` do, joined to the stages the encoder used the heap in before, and
    /// remember the heap for [`sync_heaps`](Self::sync_heaps): those the encoder first uses here
    /// (held as [`use_resource`](Self::use_resource) holds what it makes resident).
    pub(super) fn use_heap(&mut self, heap: &Heap, stages: RenderStages) -> Vec<Used> {
        let (textures, next) = heap.textures_from(0);
        let stages = match self.heap_use(heap) {
            Some(used) => {
                let widened = used.stages.union(stages);
                let changed = widened != used.stages || used.next != next;
                used.stages = widened;
                used.next = next;
                if changed {
                    self.generation = next_generation();
                }
                widened
            }
            None => {
                self.push_heap(heap, stages, next);
                self.generation = next_generation();
                stages
            }
        };
        let used = heap.bytes().map(|bytes| Used::Heap {
            heap: heap.clone(),
            bytes: bytes.clone(),
        });
        let textures = textures.into_iter().map(Used::Texture);
        used.into_iter()
            .chain(textures)
            .filter_map(|used| self.join(used, ResourceUsage::READ, stages))
            .collect()
    }

    /// [`heaps`](Self::heaps), found once a generation: a render encoder's draw, and the draws of
    /// an execution's chunks recorded apart that take it from one chunk to the next
    /// (`render/executions/kept.rs`), each note them once a change, which costs their count, not
    /// the resident resources'.
    pub(super) fn resident_heaps(&mut self) -> &[Heap] {
        let generation = self.generation;
        if self.heaps_found.as_deref().map(|(at, _)| *at) != Some(generation) {
            let heaps = self.heaps().collect();
            self.heaps_found = Some(Arc::new((generation, heaps)));
        }
        self.heaps_found.as_deref().map_or(&[], |(_, heaps)| heaps)
    }

    /// The heaps of what the encoder made resident: those it used, and those its resident
    /// buffers, textures and indirect command buffers lie in.
    pub(super) fn heaps(&self) -> impl Iterator<Item = Heap> + '_ {
        self.held().used.iter().filter_map(|(used, ..)| match used {
            Used::Buffer(buffer) => buffer.heap().cloned(),
            Used::Texture(texture) => texture.heap().cloned(),
            Used::Heap { heap, .. } => Some(heap.clone()),
            Used::IndirectCommandBuffer(icb) => icb.heap().cloned(),
        })
    }

    /// Make resident each texture made in a heap the encoder used since the encoder last looked,
    /// for reading in the stages it used the heap in, as `use_heap` made the heap's textures
    /// (the module docs): an encoder calls it as it encodes each command that reaches its
    /// resident resources. Those it first uses here (held as [`use_resource`](Self::use_resource)
    /// holds what it makes resident). A heap that made no texture since costs one atomic load
    /// (`Heap::textures_made`).
    pub(super) fn sync_heaps(&mut self) -> Vec<Used> {
        let made = self.heap_textures();
        self.join_heap_textures(made)
    }

    /// The textures made in the heaps it used since it last looked, each with the render stages
    /// it used its heap in: what [`sync_heaps`](Self::sync_heaps) makes resident. A heap that
    /// made no texture since costs one atomic load (`Heap::textures_made`); one that made one
    /// takes its list's lock to read them.
    pub(super) fn heap_textures(&mut self) -> Vec<(Texture, RenderStages)> {
        let mut made = Vec::new();
        for used in &mut self.heaps {
            if used.heap.textures_made() == used.next {
                continue;
            }
            let (textures, next) = used.heap.textures_from(used.next);
            used.next = next;
            made.extend(textures.into_iter().map(|texture| (texture, used.stages)));
            // The number it looks from on: what a later look reads.
            self.generation = next_generation();
        }
        made
    }

    /// Make resident for reading `made`, the textures [`heap_textures`](Self::heap_textures)
    /// found: those it first uses here.
    pub(super) fn join_heap_textures(&mut self, made: Vec<(Texture, RenderStages)>) -> Vec<Used> {
        made.into_iter()
            .filter_map(|(texture, stages)| {
                self.join(Used::Texture(texture), ResourceUsage::READ, stages)
            })
            .collect()
    }

    /// Join what an ended encoder made resident (`ended`) into its command buffer's residency
    /// (the module docs): each resource with its usage, and each heap it used, in the render
    /// stages it named, or every render stage for a compute encoder's (`compute`), which names
    /// none; a heap keeps the least number of a texture not made resident yet, so the commit's
    /// [`sync_heaps`](Self::sync_heaps) makes resident every texture made in it since.
    pub(super) fn join_encoder(&mut self, ended: &Residency, compute: bool) {
        self.join_encoder_resources(ended, compute);
        self.join_encoder_heaps(ended, compute);
    }

    /// The resources of [`join_encoder`](Self::join_encoder): each with its usage, in the render
    /// stages it named, or every render stage for a compute encoder's. How many `ended` holds.
    pub(super) fn join_encoder_resources(&mut self, ended: &Residency, compute: bool) -> usize {
        for (used, usage, stages) in &ended.held().used {
            self.join(used.clone(), *usage, reached(*stages, compute));
        }
        ended.held().used.len()
    }

    /// The heaps of [`join_encoder`](Self::join_encoder): each in the render stages it named, or
    /// every render stage for a compute encoder's, keeping the least number of a texture not made
    /// resident yet.
    pub(super) fn join_encoder_heaps(&mut self, ended: &Residency, compute: bool) {
        for heap in &ended.heaps {
            let stages = reached(heap.stages, compute);
            match self.heap_use(&heap.heap) {
                Some(had) => {
                    had.stages = had.stages.union(stages);
                    had.next = had.next.min(heap.next);
                }
                None => self.push_heap(&heap.heap, stages, heap.next),
            }
        }
        if !ended.heaps.is_empty() {
            self.generation = next_generation();
        }
    }

    /// Make resident what `from`, an earlier compute encoder's residency, made resident: each
    /// resource with its usage, and each heap with the textures it holds now (held as
    /// [`use_resource`](Self::use_resource) holds what it makes resident).
    pub(super) fn carry(&mut self, from: &Residency) {
        for (used, usage, _) in &from.held().used {
            self.join(used.clone(), *usage, RenderStages::empty());
        }
        for heap in &from.heaps {
            self.use_heap(&heap.heap, RenderStages::empty());
        }
    }

    /// The residency a compute command reaches, stated as a compute encoder's (no render
    /// stages), so its plan is a kernel's of the same shape as the encoder's own dispatch's
    /// (`tile_resources` alike).
    pub(super) fn kernel(mut self) -> Residency {
        let staged = (self.held().used.iter()).any(|(_, _, stages)| !stages.is_empty());
        if staged {
            for (_, _, stages) in &mut Held::changed(&mut self.held).used {
                *stages = RenderStages::empty();
            }
        }
        for heap in &mut self.heaps {
            heap.stages = RenderStages::empty();
        }
        if !self.is_empty() {
            self.generation = next_generation();
        }
        self
    }

    /// The indirect command buffers made resident for writing, each once.
    pub(super) fn writable_indirect_command_buffers(&self) -> Vec<IndirectCommandBuffer> {
        let mut writable: Vec<IndirectCommandBuffer> = Vec::new();
        for (used, usage, _) in &self.held().used {
            if let Used::IndirectCommandBuffer(icb) = used {
                if usage.contains(ResourceUsage::WRITE) && !writable.contains(icb) {
                    writable.push(icb.clone());
                }
            }
        }
        writable
    }

    /// The resources made resident.
    pub(super) fn resources(&self) -> impl Iterator<Item = &Used> {
        self.held().used.iter().map(|(used, ..)| used)
    }

    /// What a recording that may reach every resource it holds may use of them, for the
    /// scheduler's order across queues (`CommandBufferShared::footprint`): each buffer, heap's
    /// buffer and indirect command buffer's records over all their bytes, and each texture over
    /// its whole image, read where its usage reads or samples and written where it writes, in
    /// every stage. Metal orders a command's reach of a resource by the usage it was made
    /// resident with; what the commands reach unnamed (`live_buffers.rs`) it does not order.
    pub(super) fn bound_uses(&self) -> Vec<Use<ResourceKey>> {
        let stages = vk::PipelineStageFlags2::ALL_COMMANDS;
        (self.held().used.iter())
            .map(|(used, usage, _)| {
                let reads = usage.intersects(ResourceUsage::READ.union(ResourceUsage::SAMPLE));
                let read = match reads {
                    true => Access::new(stages, vk::AccessFlags2::MEMORY_READ),
                    false => Access::NONE,
                };
                let write = match usage.contains(ResourceUsage::WRITE) {
                    true => Access::new(stages, vk::AccessFlags2::MEMORY_WRITE),
                    false => Access::NONE,
                };
                let bytes = match used {
                    Used::Texture(texture) => return texture.bound_use(read, write),
                    Used::Buffer(buffer) | Used::Heap { bytes: buffer, .. } => buffer,
                    Used::IndirectCommandBuffer(icb) => icb.storage(),
                };
                whole_use(bytes, read, write)
            })
            .collect()
    }

    /// The heaps used (`use_heap`), a sparse heap among them.
    pub(super) fn used_heaps(&self) -> impl Iterator<Item = &Heap> {
        self.heaps.iter().map(|used| &used.heap)
    }

    /// The resources it holds that `other` does not.
    pub(super) fn past<'a>(&'a self, other: &'a Residency) -> impl Iterator<Item = &'a Used> {
        (self.held().used.iter())
            .map(|(used, ..)| used)
            .filter(|used| !other.held().tokens.contains_key(used))
    }

    /// `used` resident with `usage` in `stages`, joined to the usage and stages it had: `used`,
    /// when it is new here.
    fn join(&mut self, used: Used, usage: ResourceUsage, stages: RenderStages) -> Option<Used> {
        if let Some(&token) = self.held().tokens.get(&used) {
            let (_, had_usage, had_stages) = &self.held().used[token as usize];
            let (usage, stages) = (had_usage.union(usage), had_stages.union(stages));
            if (usage, stages) != (*had_usage, *had_stages) {
                let held = Held::changed(&mut self.held);
                let (_, had_usage, had_stages) = &mut held.used[token as usize];
                (*had_usage, *had_stages) = (usage, stages);
                self.generation = next_generation();
            }
            return None;
        }
        let held = Held::changed(&mut self.held);
        let token = u32::try_from(held.used.len()).expect("fewer than 2^32 resources resident");
        match &used {
            Used::IndirectCommandBuffer(_) => held.indirect.push(token),
            Used::Texture(texture) => {
                held.textures.push(token);
                self.maps_addresses |= !texture.address_mode_map().is_identity();
            }
            Used::Buffer(_) | Used::Heap { .. } => {}
        }
        held.tokens.insert(used.clone(), token);
        held.used.push((used.clone(), usage, stages));
        self.generation = next_generation();
        self.addresses = OnceLock::new();
        Some(used)
    }

    /// The bind state's used resources, each with its render stages, in token order (each token
    /// once). A buffer is stated in its
    /// `VkBuffer`'s space, as the argument table states a bound one (its length and Metal address
    /// widened by its base offset), so the resident range covers every byte the `VkBuffer` maps; a
    /// heap as its own buffer is (`Heap::bytes`), whose range holds every buffer placed in it; a
    /// texture by its facts and resource ID.
    pub(super) fn used_resources(&self) -> Vec<UsedResource> {
        self.held()
            .used
            .iter()
            .zip(0u32..)
            .map(|((resource, usage, stages), token)| {
                let mut used = match resource {
                    Used::Buffer(buffer) => stated(buffer, ResourceClass::Buffer, token),
                    Used::Heap { bytes, .. } => stated(bytes, ResourceClass::Heap, token),
                    // Reached through its ID only while it holds the ID's slot: a buffer made
                    // later with the ID takes it (`address.rs`).
                    Used::IndirectCommandBuffer(icb) => {
                        let mut used = stated(icb.storage(), ResourceClass::Buffer, token);
                        if !icb.holds_id_slot() {
                            used.gpu_address = None;
                        }
                        used
                    }
                    Used::Texture(texture) => {
                        let class = ResourceClass::Texture(texture.facts());
                        let mut used = UsedResource::new(class, token);
                        used.resource_id = Some(texture.gpu_resource_id().0);
                        used
                    }
                };
                used.usage = *usage;
                used.stages = *stages;
                used
            })
            .collect()
    }

    /// The used resources a tile dispatch reaches: those made resident in the tile stage, under
    /// their tokens, stated as a compute dispatch's are (no stages; its plan is a kernel's).
    pub(super) fn tile_resources(&self) -> Vec<UsedResource> {
        let mut used = self.used_resources();
        used.retain(|resource| resource.stages.contains(RenderStages::TILE));
        for resource in &mut used {
            resource.stages = RenderStages::empty();
        }
        used
    }

    /// What a kept recording of commands that resolved and bound against it was recorded from
    /// (`kept_commands.rs`, [`KeptListing`]), listed once for what it holds (the module docs'
    /// **Copies**), its copies' included: listed again only where a no-copy buffer moved since
    /// (`buffer::host_cells_moved`: a buffer's statement), an image of `device` let its views go
    /// (`DeviceShared::views_epoch`: a texture's key), or an indirect command buffer's hold on
    /// its ID's slot changed ([`holds_slots_as`](Self::holds_slots_as)). Each listing counted
    /// (`WorkCounts::resident_kept_keys_listed`). One given nothing answers the one empty listing
    /// every such residency shares, of no device.
    pub(super) fn kept_key(&self, device: &Device) -> Arc<KeptListing> {
        static EMPTY: OnceLock<Arc<KeptListing>> = OnceLock::new();
        let Some(held) = self.held.as_deref() else {
            return Arc::clone(EMPTY.get_or_init(|| Arc::new(self.listing())));
        };
        let epoch = device.shared().views_epoch.load(Ordering::Acquire);
        let at = (host_cells_moved(), epoch);
        if let Some((moved, epoch, listing)) = &*lock(&held.kept) {
            if (*moved, *epoch) == at && self.holds_slots_as(&listing.used) {
                debug_assert!(
                    **listing == self.listing(),
                    "a kept key listing that is not its residency's"
                );
                return Arc::clone(listing);
            }
        }
        let listing = Arc::new(self.listing());
        let counts = &device.shared().counts;
        counts.resident_kept_keys_listed(held.used.len() as u64);
        *lock(&held.kept) = Some((at.0, at.1, Arc::clone(&listing)));
        listing
    }

    /// Its [`kept_key`](Self::kept_key), listed now.
    fn listing(&self) -> KeptListing {
        let keys: Vec<ResourceKey> = (self.held().used.iter())
            .map(|(used, ..)| match used {
                Used::Buffer(buffer) => buffer.identity().key,
                Used::Heap { bytes, .. } => bytes.identity().key,
                Used::Texture(texture) => texture.use_key(),
                Used::IndirectCommandBuffer(icb) => icb.storage().identity().key,
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        KeptListing {
            used: self.used_resources(),
            keys,
            sorted,
        }
    }

    /// The resource of `token`.
    pub(super) fn get(&self, token: u32) -> Option<&Used> {
        self.held().used.get(token as usize).map(|(used, ..)| used)
    }

    /// The command buffer's residency with `live`, the device's live buffers at its commit.
    pub(super) fn with_live(self, live: Arc<LiveTable>) -> Residency {
        Residency {
            live: Some(live),
            generation: next_generation(),
            ..self
        }
    }

    /// It without the device's live buffers: what keeps it past its command buffer's commit
    /// holds no table of them, whose pages a release or a rebind lets go until the command
    /// buffer is first submitted.
    pub(super) fn without_live(self) -> Residency {
        let generation = match self.live {
            Some(_) => next_generation(),
            None => self.generation,
        };
        Residency {
            live: None,
            generation,
            ..self
        }
    }

    /// The ranges of its buffers, heaps (their own buffers) and indirect command buffers'
    /// records for a resident set, in their `VkBuffer`s' space, where they have device
    /// addresses: as the bind state's used resources state them.
    pub(super) fn resident_ranges(&self) -> Vec<ResidentRange> {
        self.used_resources()
            .into_iter()
            .filter_map(
                |used| match (used.class, used.gpu_address, used.device_address) {
                    (ResourceClass::Buffer | ResourceClass::Heap, Some(guest), Some(host)) => {
                        Some(ResidentRange::new(host, guest, used.length))
                    }
                    _ => None,
                },
            )
            .collect()
    }

    /// The resident buffer that Metal address `address` lies in, and its offset there: the first
    /// made resident of the buffers and heaps' own buffers (`Heap::bytes`) that hold it, else the
    /// first whose end it is (an empty binding), else, for a command buffer's, one of the
    /// device's live buffers of their own. `None`: in none of them. Looked up in the
    /// residency's [`AddressIndex`], made at the first look after a buffer or heap was made
    /// resident: a look costs the logarithm of their count, not their count.
    pub(super) fn resolve(&self, address: u64) -> Option<(Buffer, u64)> {
        let index = self.addresses.get_or_init(|| {
            let index = AddressIndex::of_used(&self.held().used);
            if let Some((buffer, _)) = self.buffers_written().next() {
                let counts = &buffer.device().shared().counts;
                counts.resident_address_index_made();
            }
            Arc::new(index)
        });
        let found = index.first_holding(address).and_then(|token| {
            let buffer = match self.held().used.get(token as usize)? {
                (Used::Buffer(buffer), ..) | (Used::Heap { bytes: buffer, .. }, ..) => buffer,
                (Used::Texture(_) | Used::IndirectCommandBuffer(_), ..) => return None,
            };
            Some((buffer.clone(), address - buffer.metal_address().0))
        });
        found.or_else(|| self.live.as_ref()?.resolve(address))
    }

    /// What a command buffer's next compute or render (`render`) encoder starts with, of the
    /// command buffer's residency (the module docs): the shadowed textures (`shadow.rs`) its
    /// ended encoders made resident, each with its usage and, for a render encoder, its render
    /// stages (a compute encoder names none).
    pub(super) fn shadows(&mut self, render: bool) -> Residency {
        let mut shadows = Residency::default();
        self.shadows_onto(&mut shadows, render, false);
        shadows
    }

    /// Look for the shadowed textures (`shadowed`) among those made resident since it last
    /// looked, in the order they were made resident.
    fn find_shadowed(&mut self) {
        // The field alone, so `shadowed` may change meanwhile.
        let held = self.held.as_deref().unwrap_or(&NOTHING);
        for &token in &held.textures[self.textures_looked_at..] {
            if let Some((Used::Texture(texture), ..)) = held.used.get(token as usize) {
                if texture.shadow().is_some() {
                    self.shadowed.push(token);
                }
            }
        }
        self.textures_looked_at = held.textures.len();
    }

    /// Join onto `shadows`, what a command buffer's next compute or render (`render`) encoder
    /// starts with ([`shadows`](Self::shadows)), the shadowed textures this residency holds, in the
    /// order they were made resident: this an ended encoder's not joined into its command
    /// buffer's yet, a compute encoder's (`compute`) reaching every render stage, as it would
    /// there.
    pub(super) fn shadows_onto(&mut self, shadows: &mut Residency, render: bool, compute: bool) {
        self.find_shadowed();
        for &token in &self.shadowed {
            let Some((Used::Texture(texture), usage, stages)) =
                self.held().used.get(token as usize)
            else {
                continue;
            };
            let stages = match render {
                true => reached(*stages, compute),
                false => RenderStages::empty(),
            };
            shadows.join(Used::Texture(texture.clone()), *usage, stages);
        }
    }

    /// The textures made resident, each with its usage and render stages.
    fn resident_textures(&self) -> impl Iterator<Item = (&Texture, ResourceUsage, RenderStages)> {
        self.held().textures.iter().filter_map(|&token| {
            match self.held().used.get(token as usize) {
                Some((Used::Texture(texture), usage, stages)) => Some((texture, *usage, *stages)),
                _ => None,
            }
        })
    }

    /// Whether a texture made resident is sampled through an address-mode map other than the
    /// identity (`address_map.rs`).
    pub(super) fn maps_addresses(&self) -> bool {
        self.maps_addresses
    }

    /// The textures made resident.
    pub(super) fn textures(&self) -> impl Iterator<Item = &Texture> {
        self.resident_textures().map(|(texture, ..)| texture)
    }

    /// The textures made resident, each with the render stages it was made resident in: those
    /// in one of a render command's stages whose functions read texture handles from memory are
    /// those it reaches (`pvrtc::ResidentUnreadable::reached`).
    pub(super) fn textures_made_in(&self) -> impl Iterator<Item = (&Texture, RenderStages)> {
        self.resident_textures()
            .map(|(texture, _, stages)| (texture, stages))
    }

    /// The textures made resident, each with whether a function's use of it may write: its
    /// usage writes, or neither reads nor writes, which a plan takes as both (the plan's used
    /// resources, `metal2vulkan::bind`).
    pub(super) fn textures_written(&self) -> impl Iterator<Item = (&Texture, bool)> {
        self.resident_textures().map(|(texture, usage, _)| {
            let reads = usage.intersects(ResourceUsage::READ | ResourceUsage::SAMPLE);
            (texture, usage.contains(ResourceUsage::WRITE) || !reads)
        })
    }

    /// The buffers made resident, a heap as its own buffer, each with whether a function's use
    /// of it may write, as [`textures_written`](Self::textures_written) says it of a texture.
    pub(super) fn buffers_written(&self) -> impl Iterator<Item = (&Buffer, bool)> {
        self.held().used.iter().filter_map(|(used, usage, _)| {
            let buffer = match used {
                Used::Buffer(buffer) => buffer,
                Used::Heap { bytes, .. } => bytes,
                Used::Texture(_) | Used::IndirectCommandBuffer(_) => return None,
            };
            let reads = usage.intersects(ResourceUsage::READ | ResourceUsage::SAMPLE);
            Some((buffer, usage.contains(ResourceUsage::WRITE) || !reads))
        })
    }
}

// Two are equal where they are one resource: a heap by the heap alone.
/// The render stages a resource made resident in `stages` reaches in its command buffer's
/// residency: every one for a compute encoder's (`compute`), which names none.
fn reached(stages: RenderStages, compute: bool) -> RenderStages {
    match compute {
        true => every_render_stage(),
        false => stages,
    }
}

impl PartialEq for Used {
    fn eq(&self, other: &Used) -> bool {
        match (self, other) {
            (Used::Buffer(a), Used::Buffer(b)) => a == b,
            (Used::Texture(a), Used::Texture(b)) => a == b,
            (Used::Heap { heap: a, .. }, Used::Heap { heap: b, .. }) => a == b,
            (Used::IndirectCommandBuffer(a), Used::IndirectCommandBuffer(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Used {}

impl std::hash::Hash for Used {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Used::Buffer(buffer) => buffer.hash(state),
            Used::Texture(texture) => texture.hash(state),
            Used::Heap { heap, .. } => heap.hash(state),
            Used::IndirectCommandBuffer(icb) => icb.hash(state),
        }
    }
}

/// `buffer` as a used resource of `class` with `token`, in its `VkBuffer`'s space.
fn stated(buffer: &Buffer, class: ResourceClass, token: u32) -> UsedResource {
    let base = buffer.base();
    let mut used = UsedResource::new(class, token);
    used.gpu_address = Some(buffer.exposed_address(ReadApart::Binding).0 - base);
    used.length = base + buffer.length();
    // A detached buffer's bytes are in no pages: an address reaching them misses and reads 0, as
    // Metal's unnamed reach of a detached buffer does (`detach.rs`), never the stand-in's.
    used.device_address = match buffer.detached() {
        true => None,
        false => buffer.device_address().map(|address| address - base),
    };
    used
}

/// The render stages whose functions read texture handles from memory in `layout`: those with a
/// handle table (`SyntheticBinding::HandleTable`), a kernel's as `kernel` (the tile stage for a
/// tile function). A stage the runtime does not know reaches every render stage.
pub(super) fn handle_stages(layout: &PipelineLayout, kernel: RenderStages) -> RenderStages {
    layout
        .sets
        .iter()
        .flat_map(|set| &set.bindings)
        .filter_map(|binding| match binding.source {
            BindingSource::Synthetic {
                stage,
                binding: SyntheticBinding::HandleTable,
            } => Some(stage),
            _ => None,
        })
        .fold(RenderStages::empty(), |stages, stage| {
            stages.union(match stage {
                Stage::Vertex => RenderStages::VERTEX,
                Stage::Fragment => RenderStages::FRAGMENT,
                Stage::Kernel => kernel,
                Stage::Object => RenderStages::OBJECT,
                Stage::Mesh => RenderStages::MESH,
                _ => every_render_stage(),
            })
        })
}

/// Every render stage Metal names.
pub(super) fn every_render_stage() -> RenderStages {
    RenderStages::VERTEX
        .union(RenderStages::FRAGMENT)
        .union(RenderStages::TILE)
        .union(RenderStages::OBJECT)
        .union(RenderStages::MESH)
}
