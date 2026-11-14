//! What a device's recovery carries to its next generation (`line.rs`): the program's events, its
//! uses of guest RAM, its no-copy buffers, the buffers of the runtime's memory whose bytes no
//! pointer of the program's reaches or whose pages the runtime owns, its `Private` heaps and its
//! heaps over the caller's memory with the buffers placed in them, its textures made by a call
//! the recovery makes again (those placed in the heaps it carries among them)
//! (`texture/carry.rs`), its pipeline states (`pipeline/carry.rs`) and its sampler states
//! (`sampler.rs`), each the same object on the new `VkDevice`, as Metal keeps its objects across its GPU's recovery. A fence, a depth/stencil state, a library, a function, an address
//! pool, a dynamic library, a binary archive and a rasterization rate map hold no Vulkan object
//! (`fence.rs`, `depth_stencil.rs`, `library.rs`, `address_pool.rs`, `dynamic_library.rs`,
//! `binary_archive`, `rate_map.rs`), and an argument encoder (`argument_encoder.rs`): the
//! recovery carries their handle of the device alone ([`LineDevice`]), and the device's note
//! that it released a map of layers. After the pipeline states, its indirect command buffers
//! (`indirect_command_buffer/carry.rs`) and its visible function tables
//! (`function_table/carry.rs`), each the same object there, their records and words buffers of
//! the runtime's own carried as such (`Buffer::carry_runtime`).
//!
//! - **Kept as made.** Each event (`EventShared::new`), heap (`Device::new_heap`) and texture of
//!   the program's (`Texture::made`) is kept here, held weakly, by the generation it was made on;
//!   each buffer the program made of its own (`Device::visible`) is
//!   kept among the generation's program resources already (`Owners::buffer`), which the
//!   recovery reads ([`Carry::take`]), so making one costs a load of a flag more and no lock.
//! - **Carried before the next generation serves.** The recovery takes the lost generation's live
//!   objects ([`Carry::take`]) and carries each to the new one before the line serves it
//!   ([`Device::carry_to`]), so no call meets the device half carried; a call that waits for the
//!   recovery meets every object carried.
//! - **None left behind.** A call that took its arguments before the loss completes on the lost
//!   generation (`Device::at_serving_taking`), and what it makes after the recovery took the
//!   generation's objects its maker carries ([`Device::carry_late`]). The take names the buffers
//!   it took under the registry's lock, which a maker that finds the generation taken asks
//!   ([`Carry::late_buffer`]), and an event or heap is kept under it, so each object is carried
//!   once, by one or the other; a texture's maker sets how it is made again before it asks
//!   ([`Carry::late_texture`]), and a texture carried already is carried no more. A buffer placed in a heap as the heap is carried finds its bytes
//!   carried under the lock the heap's carry takes its buffers under, and carries itself
//!   (`Heap::carry`).
//! - **What each kind becomes.** An event: a new timeline semaphore at the value the host saw
//!   (`EventShared::carry`). A `Private` buffer: new memory, its contents lost with the
//!   `VkDevice` (it reads zero); but one in exportable memory, which another API imports
//!   (`exported_memory.rs`): its block's payload, exported once from the lost `VkDevice` and kept,
//!   imported again once for every buffer carried from the block, the buffer at its offset there,
//!   the memory the other API's imports reach still, which it reads as it holds it (an
//!   exportable buffer re-imported over its own memory keeps whatever that memory holds). A buffer in fresh pages (`fresh_pages.rs`): the same pages
//!   imported again, its contents and its `contents` pointer as they were (`Buffer::carry`). A
//!   `Shared` or `Managed` buffer in a block of the runtime's own pages (`memory.rs`'s
//!   `Holds::PageBuffers`): the block's pages imported again, once for every buffer carried from
//!   it, the buffer at its offset there, its contents and `contents` pointer as they were. A
//!   no-copy buffer: made again over the caller's pages it lies in, of the memory they were given
//!   as (`HostPages::made_of`), as its making or last rebind made it there (the caller's gate and
//!   resolver of guest RAM runs asked again, on the recovering thread), its contents and
//!   `contents` pointer as they were; its deallocator and keep-alive wait for its release there
//!   and for its pages' import on each generation to be freed (`Release::carry_on`). A use of
//!   guest RAM (`HostRamUse`), before the buffers: made again on the new device, the lost one's
//!   let go of. Each buffer keeps its identity: its `ObjectId`, hazard key, Metal address (a
//!   share of its range) and range token, its label and purgeable state. A `Private` heap, after
//!   the buffers: its own buffer in new memory where its textures can lie too, its contents lost
//!   (it reads zero); a heap over the caller's memory, its own buffer made again over the same
//!   pages as a no-copy buffer is, its bytes as they are; either's identity kept as a buffer's,
//!   registered there by its key; each buffer
//!   placed in it that lives, the same buffer there at its offset in the heap's carried bytes,
//!   its identity and aliasable state kept, its range in an automatic heap given back once, by its
//!   parts on the generation it was carried to last (`Heap::carry`, `Buffer::carry_placed`). A
//!   texture, after the heaps: its call made again there as a texture of the runtime's own, which
//!   holds its parts there, a view's after its parent's, a heap's at its range of the heap
//!   carried; a texture of its own image's contents (a `Private` heap's too) lost whatever its
//!   storage mode, as its image in the lost `VkDevice`'s memory was their only copy, one placed in
//!   a heap over the caller's memory reading the pages, one over a carried buffer (a texture buffer of `Device::new_texture` over its texels'
//!   buffer too) reading the buffer's bytes; its identity, resource ID, label and purgeable state
//!   kept (`Texture::carry`). A pipeline state, after the textures, the compute and tile states
//!   first, then the drawing ones: its base variant's objects (and a drawing state's
//!   large-buffer form's) made there from the pipeline its creation compiled, so nothing is
//!   translated again, the `VkPipeline` its creation predicted queued on the making pool with no
//!   wait (a drawing state's neutrals and depth stand-in copies made as at its creation); the
//!   lost generation's memo of compiled pipelines taken first where both generations compile
//!   alike (`PipelineMemo::carry_from`), where alone a state is made again; its identity, ID,
//!   `uniqueIdentifier` and label kept (`ComputePipelineState::carry`, `GraphicsState::carry`).
//!   One whose making there failed stays the lost generation's, refused where bound
//!   (`RuntimeGap::RemadeAfterRecoveryFailed`). A sampler state, first after the handles: its
//!   handle of the device linked to the new generation's, the `VkSampler`s it kept forgotten (its
//!   binds make them there as they need them), and the state placed there as at its making, in
//!   every heap set and sampler table and in the map of live states an equal request shares; its
//!   descriptor, resource ID and element kept (`SamplerState::carry`), its placement on the lost
//!   generation taken out at that generation's rest. Where the generations compile otherwise, it
//!   stays the lost generation's.
//! - **The bytes kept are the pages'.** A buffer or texture carried with its contents reads the
//!   pages it lies in, imported again. Vulkan makes the content of external memory imported to a
//!   lost device undefined (vkspec 5.2.3, "Lost Device"): read here as the hung work's writes,
//!   which may have landed in part, as a hung Metal command buffer's may. Once the lost
//!   generation's import is let go of, the pages are the runtime's or the caller's memory alone,
//!   and the new generation reads what they hold; no promise is made about the bytes the hung
//!   work was writing.
//! - **What stays the lost generation's**, refused where it meets the next
//!   (`RuntimeGap::MadeBeforeRecovery`): a buffer of pooled host-visible memory (`Shared`, and
//!   `Managed`) the device allocated, whose `contents` pointer is its block's mapping, which
//!   Vulkan gives no way to keep over new memory (where the device takes no blocks of the
//!   runtime's own pages for it: it imports no host pointer, its check found that a held import
//!   costs its submissions, its check of fresh pages refused what a texture over such a buffer
//!   takes, one of them has not answered, or no memory type of the buffer's class takes such
//!   pages; and a `Managed` buffer's host copy without unified memory); a detached no-copy
//!   buffer, or one whose last detach's deallocator waits for its next rebind; a `Shared` heap,
//!   whose bytes are such pooled memory (a heap's own buffer lies where its textures can, never
//!   in the runtime's own pages), a detached heap over the caller's memory and a sparse heap,
//!   each with the buffers and textures placed in it; a texture over the
//!   caller's `VkImage`, a sparse or shareable texture, one over a buffer left, and a view of
//!   each (`texture/carry.rs`); a buffer over the caller's
//!   `VkBuffer`, an object of the lost `VkDevice` that only the caller can make again on the
//!   new one; a `Shared` or `Managed` buffer in exported memory, whose `contents` pointer is that
//!   memory's mapping, and a buffer in an opaque descriptor's dedicated allocation, whose import
//!   is dedicated to a buffer "identical to the buffer associated with the imported memory"
//!   (`VUID-VkMemoryDedicatedAllocateInfo-buffer-01879`), which Vulkan defines no buffer of another
//!   device to be (`exported_memory.rs`); an event whose semaphore was made exportable,
//!   whose importers hold the lost semaphore; and a visible function table whose pipeline state
//!   stays the lost generation's (`function_table/carry.rs`). The recovery's notice counts the
//!   buffers, heaps, textures and events it carried and left by kind
//!   (`Degradation::DeviceRecovered`): a heap's buffers with their heap.
//! - **Each generation's work reaches its own parts** (`NoticesFor::generation`). A command buffer of the
//!   lost generation may still record once the recovery carried its objects: an encoder a
//!   program's thread opened before the loss, a render encoder's end on the lowering thread.
//!   Inside a command buffer's work, a carried buffer or texture answers its parts of that
//!   command buffer's generation while it has them there; elsewhere, those of the generation it
//!   was carried to last. Such a command buffer fails as its device's lost one does. A carried event answers
//!   the generation it was carried to last everywhere: a command buffer of the lost generation
//!   that waits for or signals it after the carry refuses it, as its device is the next
//!   generation (`Device::foreign`). A call outside command buffers' work that reaches a buffer's
//!   Vulkan objects (a texture over its bytes, an export of its memory, an upload, a handle
//!   handed out) holds the parts it read (`Buffer::held_parts`), and a call on a texture or a
//!   pipeline state outside command buffers' work holds the object's parts (`Texture::hold`,
//!   `ComputePipelineState::hold`; [`PartsCalls`]): their release at the generation's rest waits
//!   for the call to end, and the call completes on the generation it began on. A handle handed
//!   out (`Buffer::vk_buffer`, `Texture::vk_image`, `Event::vk_semaphore`), which the caller may
//!   use while the object lives, keeps its parts, and the lost `VkDevice`, until the object goes.
//! - **The lost generation goes once nothing records on it.** A command buffer holds its queue's
//!   slot on the generation from its making to its completion, or to its drop where it never
//!   completed (`queue.rs`'s `Slot`); the registry knows the generation's queues' slots
//!   ([`Carry::queue`]) and looks at them as the last of a queue's slots comes back once the
//!   recovery took the generation's objects ([`RestWatch`]). So a generation never recovered pays
//!   a load of a flag as a command buffer is made and as a queue's last slot comes back, and
//!   nothing more. Once the line serves past it and no slot is held, the carried buffers' `VkBuffer`s there
//!   are destroyed and their memory freed, a carried event's semaphore once no host call on it is
//!   under way, and each carried object lets go of its handle of the generation ([`MadeOn`]):
//!   "When a device is lost, its child objects are not implicitly destroyed and their handles are
//!   still valid. Those objects must still be destroyed before their parents or the device can be
//!   destroyed" (vkspec 5.2.3, "Lost Device"). Its device-level objects of work, what it keeps
//!   for reuse and its kept empty blocks go then too, and no command buffer is made on it again
//!   (`DeviceShared::keep_nothing`). A carried texture's parts there and a carried pipeline
//!   state's program there go then too (`texture/carry.rs`, `pipeline/carry.rs`;
//!   `WorkCounts::lost_textures_released`, `WorkCounts::lost_programs_released`). So a carried
//!   object holds nothing of the lost generation but a handle handed out: the lost `VkDevice`
//!   goes with the last object left on it (a pooled buffer, a texture or pipeline state left
//!   there, a handle handed out), through the teardown every generation's goes through, and holds
//!   its own objects alone meanwhile (`WorkCounts::vk_devices_live`).

use super::{lock, Device, DeviceShared};
use crate::buffer::Buffer;
use crate::event::EventShared;
use crate::function_table::carry::WeakFunctionTable;
use crate::function_table::VisibleFunctionTable;
use crate::heap::{Heap, WeakHeap};
use crate::host_ram::RamUseCell;
use crate::indirect_command_buffer::carry::WeakIndirectCommandBuffer;
use crate::indirect_command_buffer::IndirectCommandBuffer;
use crate::pipeline::carry::{compiles_alike, StateCarried};
use crate::pipeline::{
    ComputePipelineState, GraphicsState, WeakComputePipelineState, WeakRenderPipelineState,
};
use crate::queue::Slots;
use crate::resource::ObjectId;
use crate::sampler::{SamplerShared, SamplerState};
use crate::texture::carry::{Remade, TextureLeft, WeakTexture};
use crate::texture::Texture;
use crate::Error;
use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::convert::Infallible;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// The least size at which the registry drops gone objects' entries.
const PRUNE_FLOOR: usize = 256;

/// One carried object's parts on a lost generation, which it lets go of once nothing records on it
/// (the module docs): their Vulkan objects, their memory, and the object's handle of the
/// generation ([`MadeOn`]).
pub(crate) trait LostParts: Send {
    /// Under the registry's lock: mark the parts let go of, so that the work of a command buffer
    /// made on the generation from now on meets them nowhere.
    fn mark(&self) {}

    /// Then, with no lock held: destroy their Vulkan objects, free their memory and let go of
    /// the object's handle of the generation.
    fn release(self: Box<Self>);
}

/// A generation's registry of what its recovery carries, and of the carried objects' parts it lets
/// go of once none of its command buffers may record (the module docs).
#[derive(Default)]
pub(crate) struct Carry {
    state: Mutex<State>,
    /// `State::taken`, read without the lock as a buffer or a command buffer is made
    /// ([`Carry::taken`]), and as a queue's last slot comes back ([`RestWatch`]).
    taken: Arc<AtomicBool>,
}

#[derive(Default)]
struct State {
    events: Vec<Weak<EventShared>>,
    /// The program's uses of guest RAM (`host_ram.rs`'s `HostRamUse`), held weakly: few, so their
    /// gone entries are dropped at each keeping.
    ram_uses: Vec<Weak<RamUseCell>>,
    /// The program's heaps, held weakly: as `ram_uses`.
    heaps: Vec<WeakHeap>,
    /// The program's textures and views (`texture/carry.rs`), held weakly, in the order they were
    /// kept: a view after the texture it was made from.
    textures: Vec<WeakTexture>,
    /// The handles of the device that the objects holding no Vulkan object answer
    /// ([`LineDevice`]), held weakly.
    links: Vec<Weak<GenerationLink>>,
    /// The program's compute states, tile states' kernels among them (`pipeline/carry.rs`), held
    /// weakly.
    compute_states: Vec<WeakComputePipelineState>,
    /// The program's drawing states, mesh ones and twins among them (`pipeline/render/carry.rs`),
    /// held weakly.
    render_states: Vec<WeakRenderPipelineState>,
    /// The program's sampler states (`sampler.rs`), held weakly.
    samplers: Vec<Weak<SamplerShared>>,
    /// The program's indirect command buffers (`indirect_command_buffer/carry.rs`), held weakly:
    /// few, so their gone entries are dropped at each keeping.
    indirect_command_buffers: Vec<WeakIndirectCommandBuffer>,
    /// The program's visible function tables (`function_table/carry.rs`), held weakly: as
    /// `indirect_command_buffers`.
    function_tables: Vec<WeakFunctionTable>,
    /// The size at which the next keeping drops the gone events' entries.
    prune_at: usize,
    /// The size at which the next keeping drops the gone textures' entries.
    textures_prune_at: usize,
    /// The size at which the next keeping drops the gone handles' entries.
    links_prune_at: usize,
    /// The size at which the next keeping drops the gone compute states' entries.
    states_prune_at: usize,
    /// The size at which the next keeping drops the gone drawing states' entries.
    render_states_prune_at: usize,
    /// The size at which the next keeping drops the gone sampler states' entries.
    samplers_prune_at: usize,
    /// The recovery took the generation's objects ([`Carry::take`]): what is kept from now on,
    /// its maker carries.
    taken: bool,
    /// The buffers the recovery took ([`Carry::take`]), which their makers leave to it
    /// ([`Carry::late_buffer`]).
    took: HashSet<ObjectId>,
    /// The slots of the generation's queues, which its command buffers hold while they may
    /// record ([`Carry::queue`]).
    queues: Vec<Weak<Slots>>,
    /// The carried objects' parts on this generation, let go of once none of its command buffers
    /// may record.
    lost: Vec<Box<dyn LostParts>>,
    /// The line serves a later generation ([`Carry::served_past`]): no call's command buffer is
    /// made here any more but one that passed its loss check before the loss.
    past: bool,
    /// None of its command buffers recorded once the line served past it: what it keeps for reuse
    /// and its device-level objects went (`DeviceShared::keep_nothing`), and no command buffer is
    /// made here again ([`Carry::refuses`]).
    at_rest: bool,
}

impl State {
    /// Drop the gone events' entries once the registry reached its size for that.
    fn prune(&mut self) {
        if self.events.len() < self.prune_at {
            return;
        }
        self.events.retain(|event| event.strong_count() > 0);
        self.prune_at = (2 * self.events.len()).max(PRUNE_FLOOR);
    }

    /// Whether a command buffer of the generation holds a slot of one of its queues.
    fn slot_held(&self) -> bool {
        (self.queues.iter())
            .filter_map(Weak::upgrade)
            .any(|slots| slots.held_any())
    }
}

/// The live objects of a lost generation, which its recovery carries ([`Carry::take`]).
pub(crate) struct Taken {
    buffers: Vec<Buffer>,
    events: Vec<Arc<EventShared>>,
    ram_uses: Vec<Arc<RamUseCell>>,
    heaps: Vec<Heap>,
    textures: Vec<Texture>,
    links: Vec<Arc<GenerationLink>>,
    compute_states: Vec<ComputePipelineState>,
    render_states: Vec<GraphicsState>,
    samplers: Vec<SamplerState>,
    indirect_command_buffers: Vec<IndirectCommandBuffer>,
    function_tables: Vec<VisibleFunctionTable>,
}

impl Carry {
    /// Whether the recovery took the generation's objects ([`Carry::take`]): one load, which a
    /// maker of a buffer reads once the buffer is among the generation's program resources, and a
    /// maker of a command buffer once it holds its slot.
    pub(crate) fn taken(&self) -> bool {
        self.taken.load(Ordering::Acquire)
    }

    /// Whether the maker of `buffer`, which the program made on this generation and which is
    /// among its program resources (`Owners::buffer`), carries it ([`Device::carry_late`]): the
    /// recovery took the generation's objects ([`taken`](Self::taken)) and not this one. Its
    /// take reads the program resources under this lock: a buffer it missed was kept there after
    /// it read them, so after it set the flag, which its maker then finds set.
    pub(crate) fn late_buffer(&self, buffer: &Buffer) -> bool {
        let state = lock(&self.state);
        state.taken && !state.took.contains(&buffer.object_id())
    }

    /// Keep `event`, made on this generation: as [`late_buffer`](Self::late_buffer).
    pub(crate) fn event(&self, event: &Arc<EventShared>) -> bool {
        let mut state = lock(&self.state);
        state.prune();
        state.events.push(Arc::downgrade(event));
        state.taken
    }

    /// Keep `cell`, a use of guest RAM made on this generation: as [`event`](Self::event).
    pub(crate) fn ram_use(&self, cell: &Arc<RamUseCell>) -> bool {
        let mut state = lock(&self.state);
        state.ram_uses.retain(|cell| cell.strong_count() > 0);
        state.ram_uses.push(Arc::downgrade(cell));
        state.taken
    }

    /// Keep `heap`, made on this generation: as [`ram_use`](Self::ram_use).
    pub(crate) fn heap(&self, heap: &Heap) -> bool {
        let mut state = lock(&self.state);
        state.heaps.retain(WeakHeap::lives);
        state.heaps.push(heap.downgrade());
        state.taken
    }

    /// Keep `texture`, a program's texture or view on this generation, as it is made or carried
    /// here (`texture/carry.rs`): as [`event`](Self::event), its gone entries dropped as the
    /// registry grows.
    pub(crate) fn texture(&self, texture: &Texture) -> bool {
        let mut state = lock(&self.state);
        if state.textures.len() >= state.textures_prune_at {
            state.textures.retain(WeakTexture::lives);
            state.textures_prune_at = (2 * state.textures.len()).max(PRUNE_FLOOR);
        }
        state.textures.push(WeakTexture::of(texture));
        state.taken
    }

    /// Whether the recovery took the generation's objects, read under the lock the take sets it
    /// under: a texture whose way to be made again was set before this read was in the take, its
    /// way read there; one set after, its maker carries (`Device::kept_texture`).
    pub(crate) fn late_texture(&self) -> bool {
        lock(&self.state).taken
    }

    /// Keep `state`, a compute state made or carried here (`pipeline/carry.rs`): as
    /// [`event`](Self::event), its gone entries dropped as the registry grows.
    pub(crate) fn compute_state(&self, state: &ComputePipelineState) -> bool {
        let mut state_list = lock(&self.state);
        if state_list.compute_states.len() >= state_list.states_prune_at {
            state_list
                .compute_states
                .retain(WeakComputePipelineState::lives);
            state_list.states_prune_at = (2 * state_list.compute_states.len()).max(PRUNE_FLOOR);
        }
        state_list.compute_states.push(state.downgrade());
        state_list.taken
    }

    /// Keep `state`, a drawing state made or carried here (`pipeline/render/carry.rs`): as
    /// [`compute_state`](Self::compute_state).
    pub(crate) fn render_state(&self, state: &GraphicsState) -> bool {
        let mut state_list = lock(&self.state);
        if state_list.render_states.len() >= state_list.render_states_prune_at {
            state_list
                .render_states
                .retain(WeakRenderPipelineState::lives);
            state_list.render_states_prune_at =
                (2 * state_list.render_states.len()).max(PRUNE_FLOOR);
        }
        state_list.render_states.push(state.downgrade());
        state_list.taken
    }

    /// Keep `state`, a sampler state made or carried here (`sampler.rs`): as
    /// [`compute_state`](Self::compute_state).
    pub(crate) fn sampler_state(&self, state: &SamplerState) -> bool {
        let mut state_list = lock(&self.state);
        if state_list.samplers.len() >= state_list.samplers_prune_at {
            state_list.samplers.retain(|state| state.strong_count() > 0);
            state_list.samplers_prune_at = (2 * state_list.samplers.len()).max(PRUNE_FLOOR);
        }
        state_list.samplers.push(state.downgrade());
        state_list.taken
    }

    /// Keep `icb`, an indirect command buffer made or carried here
    /// (`indirect_command_buffer/carry.rs`): as [`ram_use`](Self::ram_use).
    pub(crate) fn indirect_command_buffer(&self, icb: &IndirectCommandBuffer) -> bool {
        let mut state = lock(&self.state);
        (state.indirect_command_buffers).retain(WeakIndirectCommandBuffer::lives);
        state.indirect_command_buffers.push(icb.downgrade());
        state.taken
    }

    /// Keep `table`, a visible function table made or carried here (`function_table/carry.rs`):
    /// as [`ram_use`](Self::ram_use).
    pub(crate) fn function_table(&self, table: &VisibleFunctionTable) -> bool {
        let mut state = lock(&self.state);
        state.function_tables.retain(WeakFunctionTable::lives);
        state.function_tables.push(table.downgrade());
        state.taken
    }

    /// Keep `link`, this generation's handle of the device an object holding no Vulkan object
    /// answers ([`LineDevice`]): as [`event`](Self::event), its gone entries dropped as the
    /// registry grows.
    fn link(&self, link: &Arc<GenerationLink>) -> bool {
        let mut state = lock(&self.state);
        if state.links.len() >= state.links_prune_at {
            state.links.retain(|link| link.strong_count() > 0);
            state.links_prune_at = (2 * state.links.len()).max(PRUNE_FLOOR);
        }
        state.links.push(Arc::downgrade(link));
        state.taken
    }

    /// The live objects of `shared`, this generation, for its recovery to carry: its events, its
    /// uses of guest RAM, its heaps and its program resources' buffers (`Owners::buffers`). From
    /// now on, each one made is its maker's to carry.
    fn take(&self, shared: &DeviceShared) -> Taken {
        let mut state = lock(&self.state);
        state.taken = true;
        self.taken.store(true, Ordering::SeqCst);
        let buffers = shared.owners.buffers();
        state.took = buffers.iter().map(Buffer::object_id).collect();
        Taken {
            buffers,
            events: state.events.iter().filter_map(Weak::upgrade).collect(),
            ram_uses: state.ram_uses.iter().filter_map(Weak::upgrade).collect(),
            heaps: state.heaps.iter().filter_map(WeakHeap::upgrade).collect(),
            textures: (state.textures.iter())
                .filter_map(WeakTexture::upgrade)
                .collect(),
            links: state.links.iter().filter_map(Weak::upgrade).collect(),
            compute_states: (state.compute_states.iter())
                .filter_map(WeakComputePipelineState::upgrade)
                .collect(),
            render_states: (state.render_states.iter())
                .filter_map(WeakRenderPipelineState::graphics)
                .collect(),
            samplers: (state.samplers.iter())
                .filter_map(SamplerState::upgrade)
                .collect(),
            indirect_command_buffers: (state.indirect_command_buffers.iter())
                .filter_map(WeakIndirectCommandBuffer::upgrade)
                .collect(),
            function_tables: (state.function_tables.iter())
                .filter_map(WeakFunctionTable::upgrade)
                .collect(),
        }
    }

    /// Know `slots`, the slots of a queue of `shared`, this generation, which its command
    /// buffers hold while they may record: what watches them for the generation's rest, which a
    /// queue's last slot to come back reads.
    pub(crate) fn queue(&self, shared: &Arc<DeviceShared>, slots: &Arc<Slots>) -> RestWatch {
        let mut state = lock(&self.state);
        state.queues.retain(|queue| queue.strong_count() > 0);
        state.queues.push(Arc::downgrade(slots));
        RestWatch {
            taken: Arc::clone(&self.taken),
            generation: Arc::downgrade(shared),
        }
    }

    /// Whether a command buffer made on this generation, holding its slot, is refused: the
    /// generation is at rest, its device-level objects of work gone ([`let_go_at_rest`]), for a
    /// call that passed its loss check before the loss to refuse as on the lost device. A load of
    /// a flag on a generation never recovered. The rest looks at the slots held under this lock,
    /// so a command buffer that holds its slot as the generation comes to rest is either seen
    /// there or sees the rest here.
    ///
    /// [`let_go_at_rest`]: Self::let_go_at_rest
    pub(crate) fn refuses(&self) -> bool {
        self.taken() && lock(&self.state).at_rest
    }

    /// Whether the generation came to rest ([`let_go_at_rest`](Self::let_go_at_rest)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn at_rest(&self) -> bool {
        lock(&self.state).at_rest
    }

    /// A carried object's `parts` on `shared`, this generation, let go of once none of its
    /// command buffers may record: here where none may.
    pub(crate) fn lost(&self, shared: &DeviceShared, parts: Box<dyn LostParts>) {
        lock(&self.state).lost.push(parts);
        self.let_go_at_rest(shared);
    }

    /// The line serves the generation `shared`'s recovery made, past `shared`, this one: let go
    /// of what it may ([`let_go_at_rest`](Self::let_go_at_rest)).
    pub(super) fn served_past(&self, shared: &DeviceShared) {
        lock(&self.state).past = true;
        self.let_go_at_rest(shared);
    }

    /// On `shared`, this generation, once the line serves past it and none of its command
    /// buffers may record: let go of the carried objects' parts, and the first time, of what it
    /// keeps for reuse and its device-level objects (`DeviceShared::keep_nothing`). The parts are
    /// marked under the registry's lock, which a command buffer made after that takes
    /// ([`Carry::refuses`]), so its work meets none of them (`Buffer::s`), and released after
    /// it. Called with none of a command buffer's locks held, by a holder of the generation.
    fn let_go_at_rest(&self, shared: &DeviceShared) {
        let (lost, first) = {
            let mut state = lock(&self.state);
            if !state.taken || !state.past || state.slot_held() {
                return;
            }
            let mut lost = Vec::with_capacity(state.lost.len());
            for parts in state.lost.drain(..) {
                parts.mark();
                lost.push(parts);
            }
            let first = !state.at_rest;
            state.at_rest = true;
            // A maker that finds the generation taken now finds it at rest: the recovery carried
            // what it took before the line served past it.
            state.took.clear();
            state.took.shrink_to_fit();
            (lost, first)
        };
        if first {
            shared.keep_nothing();
        }
        for parts in lost {
            parts.release();
        }
    }
}

impl DeviceShared {
    /// A lost generation whose recovery carried its objects on, the line serving past it and none
    /// of its command buffers left to record, on which no work is made again: its device-level
    /// objects of work (the internal submissions' and the staging queue's pools, the runtime's own
    /// kernels, the GPU pools' blocks, the mipmap scratch, the live buffers' table,
    /// `LiveBuffers::close`), its parked buffers and images and the
    /// empty blocks it keeps go, and it parks and keeps none from now on. So the objects left on
    /// it hold the bare `VkDevice` and their own memory alone, and a carried object's memory there
    /// goes back to the device (`Recycler::close`, `memory::keep_none`). Its teardown destroys the
    /// rest, and the `VkDevice`, as ever (`DeviceShared`'s drop). Left to the process's end while
    /// it exits.
    fn keep_nothing(&self) {
        let Some(_busy) = crate::exit_guard::enter() else {
            return;
        };
        // Its queues idle, each under its lock: its submission checks' thread may submit there
        // still (`DeviceShared::queues_idle`).
        self.queues_idle();
        // Before the recycler closes: the table's buffers park there, and go with it.
        self.live_buffers.close();
        if let Some(internal) = lock(&self.internal).take() {
            internal.destroy(self.vk());
        }
        if let Some(staging) = &self.staging {
            staging.destroy(self.vk());
        }
        lock(&self.owned_kernels).destroy(self.vk());
        lock(&self.mipmap_scratch).destroy(self.vk(), &mut lock(&self.memory));
        self.gpu_pools.destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.recycled_buffers).close(self.vk(), &mut lock(&self.memory));
        lock(&self.textures.recycled).close(self.vk(), &mut lock(&self.memory));
        crate::memory::keep_none(&self.memory, self.vk());
    }
}

/// What a queue's slots on a generation read as the last of them comes back (`queue.rs`'s
/// `Slot`): whether the recovery took the generation's objects, and if so, the generation, to let
/// go of the carried objects' parts there where no slot of its queues is held now
/// ([`Carry::let_go_at_rest`]).
pub(crate) struct RestWatch {
    taken: Arc<AtomicBool>,
    generation: Weak<DeviceShared>,
}

impl RestWatch {
    /// The last of the queue's slots came back, with no lock of the queue's held: a load of a
    /// flag on a generation never recovered.
    pub(crate) fn last_slot_back(&self) {
        if !self.taken.load(Ordering::Acquire) {
            return;
        }
        if let Some(shared) = self.generation.upgrade() {
            shared.carry.let_go_at_rest(&shared);
        }
    }
}

/// Why a buffer stays the lost generation's ([`Buffer::carry`]; the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Left {
    /// Its bytes are pooled host-visible memory (`Shared`, `Managed`), its `contents` pointer its
    /// block's mapping.
    Pooled,
    /// A no-copy buffer detached from the caller's pages, or whose last detach's deallocator
    /// waits for its next rebind.
    NoCopy,
    /// A heap's buffer, which its heap's carry carries or leaves (`Heap::carry`).
    Heap,
    /// Over the caller's `VkBuffer` (`wrapped_buffer.rs`), which is an object of the lost
    /// `VkDevice`, made by the caller.
    Caller,
    /// A `Shared` or `Managed` buffer in memory another API imports (`exported_memory.rs`),
    /// allocated exportable on the lost `VkDevice`, its `contents` pointer that memory's mapping.
    Exported,
    /// Its making on the new generation failed (out of memory, the driver's refusal).
    Failed,
    /// A buffer whose memory another API imports as an opaque descriptor, which is the buffer's
    /// dedicated allocation (`exported_memory.rs`): an import of it is dedicated to a buffer
    /// "identical to the buffer associated with the imported memory"
    /// (`VUID-VkMemoryDedicatedAllocateInfo-buffer-01879`), and Vulkan defines no buffer of
    /// another device so.
    DedicatedOpaqueFd,
}

/// Why a heap stays the lost generation's, with the buffers placed in it (`Heap::carry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeapLeft {
    /// A `Shared` heap: its bytes are pooled host-visible memory, whose mapping its buffers'
    /// `contents` pointers are, as a pooled buffer's ([`Left::Pooled`]).
    Pooled,
    /// A heap over the caller's host memory (`Device::new_heap_with_bytes_no_copy`).
    NoCopy,
    /// A sparse heap, whose textures' tiles the lost `VkDevice` bound.
    Sparse,
    /// Its making on the new generation failed.
    Failed,
}

impl HeapLeft {
    /// Why a heap stays, its own buffer's carry having answered `left`.
    pub(crate) fn of(left: Left) -> HeapLeft {
        match left {
            Left::Pooled => HeapLeft::Pooled,
            Left::NoCopy => HeapLeft::NoCopy,
            Left::Heap | Left::Caller | Left::Exported | Left::Failed | Left::DedicatedOpaqueFd => {
                HeapLeft::Failed
            }
        }
    }
}

/// What the carry of a buffer did with its contents ([`Buffer::carry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Contents {
    /// They are where they were: the same pages, imported again.
    Kept,
    /// Lost with the `VkDevice`: these bytes of new memory, which read zero.
    Lost(u64),
}

/// What one recovery carried, for its notice (`line.rs`).
#[derive(Default)]
pub(crate) struct Carried {
    events: u64,
    /// The buffers whose contents were kept.
    kept: u64,
    /// The buffers whose contents were lost, and their bytes.
    lost: (u64, u64),
    /// The events left the lost generation's: their semaphores were made exportable.
    events_left: u64,
    /// The uses of guest RAM carried, and those the new generation refused.
    ram_uses: (u64, u64),
    /// The buffers left the lost generation's, by why.
    left: [u64; 7],
    /// The `Private` heaps carried, their bytes, which read zero, and the buffers placed in them
    /// carried with them.
    heaps: (u64, u64, u64),
    /// The heaps over the caller's memory carried over the same pages, and the buffers placed in
    /// them carried with them.
    host_heaps: (u64, u64),
    /// The heaps left the lost generation's, by why.
    heaps_left: [u64; 4],
    /// The textures carried, by what was made again ([`Remade`]): those whose contents were lost
    /// by storage mode, then those over carried buffers, views, and those placed in heaps over
    /// the caller's memory.
    textures: [u64; 6],
    /// The textures left the lost generation's, by why.
    textures_left: [u64; 7],
    /// The pipeline states made again there, and those whose making there failed.
    states: (u64, u64),
    /// The sampler states carried, and those left the lost generation's: it compiles otherwise.
    samplers: (u64, u64),
    /// The indirect command buffers carried with their records, those whose records were lost,
    /// and those left the lost generation's.
    indirect_command_buffers: (u64, u64, u64),
    /// The visible function tables carried, and those left the lost generation's.
    function_tables: (u64, u64),
}

impl Carried {
    /// The objects carried and left, as the recovery's notice says them: the pooled buffers
    /// left always, every other kind left where some was, and the uses of guest RAM where any
    /// lived.
    pub(crate) fn say(&self) -> String {
        let (lost, lost_bytes) = self.lost;
        let (ram_uses, ram_uses_left) = self.ram_uses;
        let kinds = [
            "pooled `Shared` or `Managed` buffers",
            "detached no-copy buffers",
            "heaps' buffers",
            "buffers over the caller's `VkBuffer`",
            "`Shared` or `Managed` buffers in exportable memory",
            "buffers whose making there failed",
            "buffers in a dedicated opaque-fd allocation (Vulkan defines no identical buffer on \
             another device)",
        ];
        let heap_kinds = [
            "`Shared` heaps",
            "no-copy heaps",
            "sparse heaps",
            "heaps whose making there failed",
        ];
        let texture_kinds = [
            "heaps' textures",
            "textures over the caller's `VkImage`",
            "sparse textures",
            "shareable textures",
            "textures over the buffers left",
            "textures whose making there failed",
            "textures of calls no recovery makes again",
        ];
        let left = (kinds.iter().zip(self.left).enumerate())
            .filter(|&(at, (_, count))| at == Left::Pooled as usize || count > 0)
            .map(|(_, (kind, count))| format!("{count} {kind}"))
            .chain(
                (heap_kinds.iter().zip(self.heaps_left))
                    .filter(|&(_, count)| count > 0)
                    .map(|(kind, count)| format!("{count} {kind}")),
            )
            .chain(
                (texture_kinds.iter().zip(self.textures_left))
                    .filter(|&(_, count)| count > 0)
                    .map(|(kind, count)| format!("{count} {kind}")),
            )
            .chain(
                (self.events_left > 0)
                    .then(|| format!("{} events made exportable", self.events_left)),
            )
            .chain(
                (ram_uses_left > 0)
                    .then(|| format!("{ram_uses_left} guest RAM uses the new device refused")),
            )
            .chain((self.samplers.1 > 0).then(|| {
                format!(
                    "{} sampler states of a device that compiles otherwise",
                    self.samplers.1
                )
            }))
            .chain((self.indirect_command_buffers.2 > 0).then(|| {
                format!(
                    "{} indirect command buffers whose making there failed",
                    self.indirect_command_buffers.2
                )
            }))
            .chain((self.function_tables.1 > 0).then(|| {
                format!(
                    "{} visible function tables of states left or whose making there failed",
                    self.function_tables.1
                )
            }))
            .collect::<Vec<_>>()
            .join(", ");
        let ram = match ram_uses {
            0 => String::new(),
            _ => format!("{ram_uses} guest RAM uses, "),
        };
        let heaps = match self.heaps {
            (0, ..) => String::new(),
            (heaps, bytes, buffers) => format!(
                ", and {heaps} `Private` heaps ({bytes} bytes) with the {buffers} buffers placed \
                 in them, whose contents were lost too"
            ),
        };
        let host_heaps = match self.host_heaps {
            (0, _) => String::new(),
            (heaps, buffers) => format!(
                ", and {heaps} no-copy heaps with the {buffers} buffers placed in them, over the \
                 same pages (their contents kept)"
            ),
        };
        let textures = match self.textures {
            [0, 0, 0, 0, 0, 0] => String::new(),
            [shared, managed, private, over, views, in_host_heaps] => format!(
                ", and {} textures whose contents were lost too (they read as new ones: {shared} \
                 `Shared`, {managed} `Managed`, {private} `Private` or memoryless), {over} \
                 textures over the buffers carried, {in_host_heaps} textures placed in the \
                 no-copy heaps carried and {views} texture views",
                shared + managed + private
            ),
        };
        let states = match self.states {
            (0, 0) => String::new(),
            (made, failed) => format!(
                ", and {made} pipeline states made again (their predicted pipelines queued), \
                 {failed} more whose making there failed (refused where bound: \
                 `RemadeAfterRecoveryFailed`)"
            ),
        };
        let samplers = match self.samplers.0 {
            0 => String::new(),
            carried => format!(", and {carried} sampler states"),
        };
        let icbs = match self.indirect_command_buffers {
            (0, 0, _) => String::new(),
            (kept, lost, _) => format!(
                ", and {} indirect command buffers, {lost} of them whose records were lost (every \
                 command reset)",
                kept + lost
            ),
        };
        let tables = match self.function_tables.0 {
            0 => String::new(),
            carried => format!(", and {carried} visible function tables with their entries"),
        };
        format!(
            "it carried {} events, {ram}{} buffers with their contents and {lost} `Private` \
             buffers ({lost_bytes} bytes) whose contents were lost (they read zero){heaps}\
             {host_heaps}{textures}{states}{samplers}{icbs}{tables}; left refused \
             (`MadeBeforeRecovery`): \
             {left}",
            self.events, self.kept
        )
    }

    /// One buffer's carry.
    fn buffer(&mut self, carried: Result<Contents, Left>) {
        match carried {
            Ok(Contents::Kept) => self.kept += 1,
            Ok(Contents::Lost(bytes)) => {
                self.lost.0 += 1;
                self.lost.1 += bytes;
            }
            Err(left) => self.left[left as usize] += 1,
        }
    }

    /// One heap's carry: its bytes and the buffers placed in it carried, or why it was left.
    fn heap(&mut self, carried: Result<Option<(Contents, u64)>, HeapLeft>) {
        match carried {
            Ok(Some((Contents::Lost(bytes), buffers))) => {
                self.heaps.0 += 1;
                self.heaps.1 += bytes;
                self.heaps.2 += buffers;
            }
            Ok(Some((Contents::Kept, buffers))) => {
                self.host_heaps.0 += 1;
                self.host_heaps.1 += buffers;
            }
            Ok(None) => {}
            Err(left) => self.heaps_left[left as usize] += 1,
        }
    }

    /// One texture's carry: what was made again, or why it was left.
    fn texture(&mut self, carried: Result<Option<Remade>, TextureLeft>) {
        match carried {
            Ok(Some(remade)) => self.textures[remade as usize] += 1,
            Ok(None) => {}
            Err(left) => self.textures_left[left as usize] += 1,
        }
    }

    /// One pipeline state's carry.
    fn state(&mut self, carried: StateCarried) {
        match carried {
            Some(Ok(())) => self.states.0 += 1,
            Some(Err(_)) => self.states.1 += 1,
            None => {}
        }
    }

    /// One indirect command buffer's carry.
    fn indirect_command_buffer(&mut self, carried: Result<Contents, Left>) {
        match carried {
            Ok(Contents::Kept) => self.indirect_command_buffers.0 += 1,
            Ok(Contents::Lost(_)) => self.indirect_command_buffers.1 += 1,
            Err(_) => self.indirect_command_buffers.2 += 1,
        }
    }

    /// One visible function table's carry.
    fn function_table(&mut self, carried: Result<Contents, Left>) {
        match carried {
            Ok(_) => self.function_tables.0 += 1,
            Err(_) => self.function_tables.1 += 1,
        }
    }

    /// One sampler state's carry.
    fn sampler(&mut self, carried: Option<bool>) {
        match carried {
            Some(true) => self.samplers.0 += 1,
            Some(false) => self.samplers.1 += 1,
            None => {}
        }
    }

    /// One event's carry.
    fn event(&mut self, carried: Result<(), Error>) {
        match carried {
            Ok(()) => self.events += 1,
            Err(_) => self.events_left += 1,
        }
    }

    /// One use of guest RAM's carry.
    fn ram_use(&mut self, carried: Result<(), Error>) {
        match carried {
            Ok(()) => self.ram_uses.0 += 1,
            Err(_) => self.ram_uses.1 += 1,
        }
    }
}

impl Device {
    /// Carry this generation's live events, buffers and heaps, lost, to `next`, the generation
    /// its recovery made, before `next` serves (the module docs): what was carried and left, for
    /// the recovery's notice.
    pub(super) fn carry_to(&self, next: &Device) -> Carried {
        let shared = self.shared();
        let Taken {
            buffers,
            events,
            ram_uses,
            heaps,
            textures,
            links,
            compute_states,
            render_states,
            samplers,
            indirect_command_buffers,
            function_tables,
        } = shared.carry.take(shared);
        for link in &links {
            link.carry(next);
        }
        // A map of no layers made after a map of layers was released is one whose passes Metal
        // draws from the memory it left, before the recovery or after (`rate_map.rs`).
        if shared.layered_rate_map_released.load(Ordering::Relaxed) {
            (next.shared().layered_rate_map_released).store(true, Ordering::Relaxed);
        }
        let mut carried = Carried::default();
        // The sampler states: each placed there as at its making (`sampler.rs`).
        for state in &samplers {
            carried.sampler(state.carry(next));
        }
        // The lost generation's compiled functions and pipelines, the next's too where both
        // compile alike: a creation there, or a state's remaking, translates nothing again.
        if compiles_alike(self, next) {
            next.shared()
                .shader_cache
                .memo
                .carry_from(&shared.shader_cache.memo);
        }
        for event in &events {
            carried.event(event.carry(next));
        }
        // Before the buffers: a no-copy buffer over a RAM's pages lies in its chunks there.
        for cell in &ram_uses {
            carried.ram_use(cell.carry(next));
        }
        for buffer in &buffers {
            carried.buffer(buffer.carry(next));
        }
        for heap in &heaps {
            carried.heap(heap.carry(next));
        }
        // After the buffers, which textures lie over; the textures before the views of them, as
        // they were kept (a view's carry carries its parent where it was not).
        for texture in &textures {
            carried.texture(texture.carry(next));
        }
        // The compute states, tile states' kernels among them: each its base variant's objects
        // made there from its compiled pipeline (`pipeline/carry.rs`).
        for state in &compute_states {
            carried.state(state.carry(next));
        }
        // Then the drawing states, mesh ones and twins among them, alike
        // (`pipeline/render/carry.rs`).
        for state in &render_states {
            carried.state(state.carry(next));
        }
        // After the states: the programs their commands run, and the tables' states, are there.
        for icb in &indirect_command_buffers {
            carried.indirect_command_buffer(icb.carry(next));
        }
        for table in &function_tables {
            carried.function_table(table.carry(next));
        }
        carried
    }

    /// Carry an object made on this generation after its recovery took the generation's objects
    /// (`Carry::late_buffer`, `Carry::event`) with `carry`, to the generation that serves once that
    /// recovery made it (the module docs). Nothing where no later generation serves: a line that
    /// cannot recover, or a thread command buffers' completion depends on, whose calls reach the
    /// lost generation (`Line::current`).
    pub(crate) fn carry_late<T, E>(&self, carry: impl FnOnce(&Device) -> Result<T, E>) {
        let serving = self.line().current(self);
        if serving.same(self) {
            return;
        }
        // A refusal leaves the object the lost generation's, refused where it meets the next.
        let _carried = carry(&serving);
    }
}

/// The device of an object a recovery may carry (a buffer, its pages, an event): the generation it
/// was made on, as every object's device is, until the object's parts there were let go of
/// (`Carry::lost`); then let go of itself ([`LetGo::let_go`]), so a carried object holds the lost
/// generation no longer and that generation goes with the objects left on it (the module docs).
/// Read as a [`Device`] at one indirection, as the field it stands for. No borrow of it reaches
/// the program, which may hold one past the generation's rest: an object's `device` answers a
/// clone.
pub(crate) type MadeOn = LetGo<Device>;

/// A carried object's part on one generation (its device handle, [`MadeOn`]; a pipeline state's
/// program there, `pipeline/carry.rs`; a texture's parts there, `texture/carry.rs`): the object's
/// own, read in place at one indirection as the field it stands for, until its parts on that
/// generation were let go of at the generation's rest (`Carry::lost`); then dropped in place
/// ([`let_go`](Self::let_go)), the object living on, and read no more: every reach of the object
/// from then on goes to where it was carried, and the work of a command buffer of the generation
/// meets it nowhere ([`gone`](Self::gone)).
pub(crate) struct LetGo<T> {
    value: UnsafeCell<ManuallyDrop<T>>,
    /// It was let go of.
    gone: AtomicBool,
}

// SAFETY: it is a `T`, which is `Send` and `Sync`, read through shared references, and let go of
// once, when nothing reads it any more (`LetGo::let_go`'s contract).
unsafe impl<T: Send + Sync> Sync for LetGo<T> {}

impl<T> LetGo<T> {
    /// `value`, held until it is let go of.
    pub(crate) fn new(value: T) -> LetGo<T> {
        LetGo {
            value: UnsafeCell::new(ManuallyDrop::new(value)),
            gone: AtomicBool::new(false),
        }
    }

    /// Whether it was let go of: the object's parts on the generation went, the object carried.
    pub(crate) fn gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }

    /// Let go of it, once.
    ///
    /// # Safety
    ///
    /// Nothing reads it from now on but [`gone`](Self::gone): its object's parts on the
    /// generation were let go of and no work of the generation's reaches them
    /// (`Carry::let_go_at_rest`), no call under way holds them (`PartsCalls`), and every other
    /// reach of the object goes to where it was carried.
    pub(crate) unsafe fn let_go(&self) {
        if self.gone.swap(true, Ordering::AcqRel) {
            return;
        }
        // SAFETY: the caller's: no reference to the value is left, and none is taken again.
        unsafe { ManuallyDrop::drop(&mut *self.value.get()) };
    }
}

impl<T> std::ops::Deref for LetGo<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: it lives until `let_go`, after which nothing reads it (its contract).
        unsafe { &*self.value.get() }
    }
}

impl<T> Drop for LetGo<T> {
    fn drop(&mut self) {
        if !*self.gone.get_mut() {
            // SAFETY: not let go of, and the last reference.
            unsafe { ManuallyDrop::drop(self.value.get_mut()) };
        }
    }
}

/// The calls outside command buffers' work under way on a carried object's parts (a buffer's on
/// one generation, `Buffer::held_parts`; a pipeline state's programs, `pipeline/carry.rs`; a
/// texture's, `texture/carry.rs`), and the releases of its parts on lost generations that wait for
/// them to end: the parts a call reached stay while it is under way, and the last call to end lets
/// go of those whose generation came to rest meanwhile ([`CallHold`]). A command buffer's work
/// holds no call: its queue slot holds its generation, whose rest waits for it (the module docs).
#[derive(Default)]
pub(crate) struct PartsCalls(Mutex<Calls>);

#[derive(Default)]
struct Calls {
    under_way: usize,
    waiting: Vec<Box<dyn LostParts>>,
}

impl PartsCalls {
    /// A call on the parts from now until the hold goes: none where `released`, read under the
    /// lock their release takes, answers that the parts were marked let go of (`LostParts::mark`)
    /// and no call under way holds their release back (one this thread is inside, say); else
    /// their release waits for the call to end.
    pub(crate) fn hold_unless(&self, released: impl FnOnce() -> bool) -> Option<CallHold<'_>> {
        let mut calls = lock(&self.0);
        if calls.under_way == 0 && released() {
            return None;
        }
        calls.under_way += 1;
        Some(CallHold(self))
    }

    /// A call on whichever of the parts it reaches from now until the hold goes: those of the
    /// generation the object was carried to last, never let go of, or those it reached before a
    /// carry, whose release waits for the call to end.
    pub(crate) fn hold(&self) -> CallHold<'_> {
        lock(&self.0).under_way += 1;
        CallHold(self)
    }

    /// `parts`, back to their release to let go of now, where no call is under way on the
    /// object's parts; else none, the parts kept for the last call under way to let go of as it
    /// ends ([`CallHold`]).
    pub(crate) fn unless_held<P: LostParts + 'static>(&self, parts: Box<P>) -> Option<Box<P>> {
        let mut calls = lock(&self.0);
        if calls.under_way > 0 {
            calls.waiting.push(parts);
            return None;
        }
        Some(parts)
    }
}

/// A call under way on a carried object's parts ([`PartsCalls`]), until it goes: the last to go
/// lets go of the parts whose release waited for it, unless one began since.
pub(crate) struct CallHold<'a>(&'a PartsCalls);

impl Drop for CallHold<'_> {
    fn drop(&mut self) {
        let waiting = {
            let mut calls = lock(&self.0 .0);
            calls.under_way -= 1;
            match calls.under_way {
                0 => std::mem::take(&mut calls.waiting),
                _ => Vec::new(),
            }
        };
        for lost in waiting {
            lost.release();
        }
    }
}

/// The device of an object that holds no Vulkan object (a fence, a depth/stencil state, a library
/// and its functions, an address pool, a dynamic library, a binary archive, a rasterization rate
/// map, an argument encoder), which a recovery carries as its handle of the device
/// alone: a root handle, which the object's own calls and checks go through
/// ([`root`](Self::root): of whatever generation serves, matched by its line), and the generation
/// handle its `device` answers ([`generation`](Self::generation)): the generation it was made on,
/// then each one the device's recoveries carried it to. A handle taken of it keeps meaning that
/// generation, as any object's device does: its raw handles are that generation's `VkDevice`'s,
/// kept alive while the caller holds it. The object lets go of a lost generation's handle at that
/// generation's rest ([`MadeOn`]), so it holds nothing of it after.
#[derive(Clone)]
pub(crate) struct LineDevice {
    root: Device,
    first: Arc<GenerationLink>,
}

/// One generation's handle of a [`LineDevice`], linked to the next generation's once a recovery
/// carried the object there.
pub(crate) struct GenerationLink {
    device: MadeOn,
    next: OnceLock<Arc<GenerationLink>>,
}

impl LineDevice {
    /// The device of an object made on `made_on`, a generation handle: its handle kept by the
    /// generation's registry for its recovery to carry ([`Carry::take`]), or carried by its maker
    /// where the recovery took the generation's objects before then ([`Device::carry_late`]).
    pub(crate) fn new(made_on: &Device) -> LineDevice {
        LineDevice {
            root: made_on.root(),
            first: GenerationLink::kept(made_on),
        }
    }

    /// The device of an object made on `made_on` whose own carry carries its handle, as it does
    /// more there than carry the handle (a sampler state, `sampler.rs`): as [`new`](Self::new),
    /// kept by no registry ([`link_unkept`](Self::link_unkept)).
    pub(crate) fn unkept(made_on: &Device) -> LineDevice {
        LineDevice {
            root: made_on.root(),
            first: GenerationLink::unkept(made_on),
        }
    }

    /// The root handle: of whatever generation serves, as the object is no generation's.
    pub(crate) fn root(&self) -> &Device {
        &self.root
    }

    /// The generation handle of the generation the object was carried to last, else of the one
    /// it was made on.
    pub(crate) fn generation(&self) -> &Device {
        &self.last().device
    }

    /// The link of the generation the object was carried to last, else of the one it was made on.
    fn last(&self) -> &Arc<GenerationLink> {
        let mut link = &self.first;
        while let Some(next) = link.next.get() {
            link = next;
        }
        link
    }

    /// Link an [`unkept`](Self::unkept) handle to `to`, the generation a recovery made, after the
    /// last: the lost generation's handle, to hand that generation's registry (`Carry::lost`) with
    /// no lock of the object's held, which lets go of it at the generation's rest. `None` where it
    /// was linked there already, or another carry linked one first.
    pub(crate) fn link_unkept(&self, to: &Device) -> Option<Box<dyn LostParts>> {
        let last = self.last();
        if last.device.same(to) || last.next.set(GenerationLink::unkept(to)).is_err() {
            return None;
        }
        Some(Box::new(LostHandle(Arc::clone(last))))
    }
}

impl std::fmt::Debug for LineDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.generation().fmt(f)
    }
}

impl GenerationLink {
    /// `device`'s handle, a generation handle, kept by its generation's registry; carried by
    /// this maker where that generation's recovery took its objects before.
    fn kept(device: &Device) -> Arc<GenerationLink> {
        let link = GenerationLink::unkept(device);
        if device.shared().carry.link(&link) {
            device.carry_late(|serving| {
                link.carry(serving);
                Ok::<(), Infallible>(())
            });
        }
        link
    }

    /// `device`'s handle, a generation handle, kept by no registry.
    fn unkept(device: &Device) -> Arc<GenerationLink> {
        Arc::new(GenerationLink {
            device: MadeOn::new(device.clone()),
            next: OnceLock::new(),
        })
    }

    /// Carry the handle to `to`, the generation this one's recovery made: `to`'s handle linked
    /// after it, and this one let go of at the lost generation's rest. Nothing where it was
    /// carried already.
    fn carry(self: &Arc<Self>, to: &Device) {
        if self.next.get().is_some() {
            return;
        }
        if self.next.set(GenerationLink::kept(to)).is_err() {
            // Another carry of it came first.
            return;
        }
        let lost = Device::clone(&self.device);
        let parts = Box::new(LostHandle(Arc::clone(self)));
        lost.shared().carry.lost(lost.shared(), parts);
    }
}

/// A [`LineDevice`]'s handle of a lost generation, let go of once none of the generation's command
/// buffers may record.
struct LostHandle(Arc<GenerationLink>);

impl LostParts for LostHandle {
    fn release(self: Box<Self>) {
        // SAFETY: its next link is set, so the object's device answers a later generation's handle
        // (`LineDevice::generation`), and its own calls and checks go through its root handle or
        // that later one: no reach of the object reads this handle any more.
        unsafe { self.0.device.let_go() };
    }
}
