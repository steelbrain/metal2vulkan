//! Set 1, the heap, and the texture and sampler tables, device-wide (residency part C): every live
//! texture and sampler state is at its resource ID for every command, as Metal's are (the M4
//! probe: a texture read through its ID in an argument buffer is reached with no encoder naming
//! it, and through a later encoder's `use_resource`).
//!
//! - **The texture table.** One buffer per device of [`TABLE_ENTRIES`] handle-table entries
//!   (`bind::handle_entry`), entry `i` for resource ID `i`: `{i, i, flags}` while a live texture
//!   holds the ID's slot (`resource_ids.rs`), its flags the texture's (`bind::handle_entry_flags`),
//!   else `{i, 0, 0}`, the neutral element. So the table is sorted, starts with the null entry, and
//!   maps every ID a heap array can reach, the handle search's reach (`HANDLE_TABLE_CAPACITY`,
//!   the most any pipeline's heap arrays have; 1 MiB): a texture Metal numbers past its
//!   placement limit after a reservation (`resource_ids.rs`) is reached too. Each command binds
//!   it at every texture handle table of its layout (`Synthetic(HandleTable)`) over the entries
//!   its heap arrays' count reaches ([`texture_table`]): the shader's search reads its length
//!   from the range, and an ID past the arrays misses, announced
//!   ([`Degradation::UnnamedTexturesPastCapacity`]).
//! - **Heap sets.** One per set-1 layout ([`HeapSet`], which each variant of that layout holds),
//!   shared by every command buffer. Element 0 of each array is its class's neutral (an image's,
//!   the neutral sampler, the null function table: zeros in the table's buffer). Element `i` of
//!   each image array holds the texture at ID `i` as the array's class takes it
//!   (`bind::heap_element`: its view where it fits the class, else the neutral), written as the
//!   texture takes the slot ([`place`]), for each set that lives then, and by a set made later
//!   for each texture that lives then. A batch holds those writes from its first work's take
//!   until its submission was made ([`hold`]): one asked meanwhile is made then, before the next
//!   submission takes its first work, so no texture registered after the take, whose move is
//!   not the batch's, is in its sets or table in `UNDEFINED` (`texture/first_move.rs`), and its
//!   making waits for no submission. A rebind or a detach of host pages writes again the
//!   elements of the textures whose images or texel views lie in those bytes ([`renew_over`]),
//!   and of no other texture (`WorkCounts::heap_slots_rewritten`). A texture the class fits that it cannot view (a format
//!   the device cannot sample or store formatless, or that needs its function's decode) is the
//!   neutral, announced ([`Degradation::UnnamedTextureUnviewable`]); one whose usage the class
//!   lacks is the neutral, as Metal's contract leaves the read undefined. A shadowed texture's
//!   element is its shadow image's view, which holds the texels an encoder that named it last
//!   copied in ([`Degradation::UnnamedShadowedTexture`], announced at the first command that
//!   binds a set holding it while neither naming it nor making it resident, [`bind`]: one that
//!   does has its shadow copied in). A carried texture's element is its decoded twin's view
//!   (`texture/decoded_twin.rs`), and placing it is no read: where the twin's texels are not
//!   Metal's decode exactly, `decoded-texel-precision` is announced at the first command that
//!   binds a set holding it, named or not, or as it is placed in a set a command buffer that
//!   bound the set and has not completed may read through. A texture a heap holds by its raw
//!   texels alone (`bind::heap_raw_form`: its raw view in the unsigned integer class) is read
//!   through the function's own decode of them, within the bounds `device-pack-conversion`
//!   states, and announced as a twin is, with that kind ([`RawRead`]). An element whose ID no
//!   texture holds is never read: the table maps the ID to element 0. The function tables'
//!   elements are the commands' writes ([`bind`]), each written where the set held another there.
//! - **Who reads an element while it changes.** The arrays are `UPDATE_AFTER_BIND` and, where
//!   the device has `descriptorBindingUpdateUnusedWhilePending`, `UPDATE_UNUSED_WHILE_PENDING`
//!   (`pipeline::link`): every command binds the set's one version ([`SetVersion`]), and a
//!   texture made while command buffers run writes its elements while they are pending, before
//!   the table names them, so a command reads them only once they hold it. A kept recording of
//!   commands (`kept_commands.rs`) binds that version again in each command buffer that runs it,
//!   which holds it until it completes as a command's binding does ([`bound_again`],
//!   [`HeapUse::bind_again`]): it reads every element written in place as the commands would.
//!   Where the device lacks the feature, nothing writes a set a committed command buffer binds:
//!   each command buffer binds a version of its own, made as it first binds the set, every element
//!   a texture's ID does not name the neutral and every other the set's ([`SetEntry::written`]),
//!   announced ([`Degradation::HeapSetCopied`]). Writes go to every version whose command buffer
//!   has not committed, and a committed one keeps its elements (a texture made after is the
//!   neutral there). A gone texture's elements turn neutral in every version no submission of its
//!   command buffer took yet, committed or not: its set is `UPDATE_AFTER_BIND`, and nothing
//!   pending binds it.
//! - **Retirement.** A texture going clears its table entry. A command buffer submitted before
//!   then may still read the element, so the texture's image (which holds the view) is kept
//!   until every such command buffer completed: each command buffer that binds a heap set, or
//!   whose host steps may, takes a [`Ticket`] as its first batch is submitted, and an image
//!   retired while tickets are out waits for those taken before it. (The ticket was taken at the
//!   commit.) One committed but not yet submitted (held at its gate) never reads the gone texture:
//!   the table its submission's commands read maps the ID to the neutral, and its own versions'
//!   elements turned neutral. So the image goes as Metal's does where no submitted command buffer
//!   may read it, with the buffer or heap it lies in, whose no-copy pages' deallocator then runs
//!   with their import freed (`buffer.rs`). What a rebind or a detach replaced ([`retire`]) waits
//!   instead for the command buffers that committed before it, each holding a second ticket from
//!   its commit: one committed before it keeps the elements that name it until it completes (a
//!   placement's or a rebind's writes go only to versions not committed).
//! - **Samplers.** Every live sampler state is at its element of every set's sampler arrays
//!   (`resource_ids.rs`: the lowest no live state holds), each holding the Vulkan sampler made for
//!   the array's use, as a bound argument sampler's is (`bind::sampler_desc`), written as the
//!   state takes its ID's slot ([`place_sampler`]), for each set that lives then (its one version,
//!   or each command buffer's own that no submission took yet, committed or not: a sampler holds
//!   no layout a batch's first moves order), and by a set made later for each state that lives
//!   then. One the array's use refuses (a device feature it lacks) or whose sampler the device
//!   cannot make is the neutral there, announced ([`Degradation::UnlistedArgumentSampler`]). As the state goes ([`sampler_gone`]) its
//!   samplers wait for the command buffers submitted before (the **Retirement** above), and its
//!   element turns neutral in every version no submission took yet, as a gone texture's does.
//! - **Sampler tables.** One buffer per kind of [`TABLE_ENTRIES`] handle-table entries, which each
//!   command binds at its layout's sampler tables (`Synthetic(SamplerTable)`, [`sampler_table`]),
//!   whole: so a function reaches every live state by its ID as it runs, as Metal's does (the M3
//!   Max under macOS 27.0.1: a state made after the commit, its ID written into the argument
//!   buffer before the event the command buffer waits for was signalled, was the one sampled),
//!   whatever states lived as the command was encoded. Entry `i` of the low window, below 32768,
//!   is sampler ID `i`; entry `32768 + i`, the high window's, is ID `base + i`, its base taken as
//!   the first state past the low window takes its slot ([`high_window_base`]: a reservation's
//!   plain IDs, which Metal reads through an argument buffer as any other, 100001 to 100003 on the
//!   M3 Max), the window's entries rewritten from the highest down so the table stays sorted for
//!   a command searching it meanwhile; before that it is IDs 32768 to 65535. Each entry is
//!   `{id, element, word}` while a placed state holds the ID's slot, its word its
//!   [`SamplerWord`], else `{id, 0, 0}`, the neutral. A kind ([`SamplerTableKind`]) maps only the
//!   states whose elements its layouts' sampler arrays hold, and cubic states only where each of
//!   the command's stages with a sampler table tests the states it reaches for a cubic filter
//!   (`RuntimeFact::CubicArgumentSampler`): another is the neutral there, announced. So is a
//!   state at an ID in neither window. An entry is written after the state's elements, and
//!   cleared before they go, so a command reading it meanwhile reads the state or the neutral.
//! - **Release.** A heap set goes with the last variant of its layout and the last command buffer
//!   that bound it, what its elements name (the sampler states' samplers, which no command holds)
//!   retired as a gone texture's image is; the texture and sampler tables with the device.

use super::binding::{write_sets, Info, Write};
use super::degradation::Degradation;
use super::descriptor_arena::SetNeeds;
use super::device::{lock, Device};
use super::error::{internal, vulkan, Error};
use super::exit_guard;
use super::memory::{Memory, MemoryClass, Placement};
use super::pipeline::{set_needs, VariantObjects};
use super::release;
use super::resource::ResourceKey;
use super::sampler::{SamplerShared, SamplerState};
use super::texture::{Texture, TextureShared, TwinRead};
use super::tracker::{Access, Use};
use ash::vk;
use metal2vulkan::bind::{
    handle_entry, handle_entry_flags, heap_element, heap_raw_form, ArgumentSampler, BindState,
    BindingPlan, FunctionTableWords, HeapElement, HeapElements, NeutralResource,
    HANDLE_ENTRY_BYTES, HANDLE_TABLE_CAPACITY,
};
use metal2vulkan::error::BindError;
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse, SamplerWord};
use metal2vulkan::pipeline::{
    BindingSource, DescriptorSetLayout, HandleClass, ImageAccess, ImageHandleClass, PipelineLayout,
    RuntimeDependency, RuntimeFact, SampledType, Stage, SyntheticBinding,
};
use std::any::Any;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

/// The heap's set number.
pub(super) const HEAP_SET: u32 = 1;

/// The entries of a device's texture table: one per texture resource ID a heap array can reach,
/// the handle search's reach (the module docs).
pub(super) const TABLE_ENTRIES: u32 = HANDLE_TABLE_CAPACITY;

/// The bytes of the texture table: past them, the null function table's words.
const TABLE_BYTES: u64 = TABLE_ENTRIES as u64 * HANDLE_ENTRY_BYTES as u64;

/// The sampler IDs of each window of a sampler table (the module docs): the low window holds IDs
/// 0 to 32767, each at its own entry, and the high window as many IDs from its base.
const SAMPLER_WINDOW: u64 = TABLE_ENTRIES as u64 / 2;

/// How far below the first sampler ID past the low window the high window starts (the module
/// docs), rounded down to a multiple of it.
const HIGH_WINDOW_SLACK: u64 = 1024;

/// What keeps a resource apart from it: a texture's image, a sampler, a table's words.
pub(super) type Keeper = Box<dyn Any + Send + Sync>;

/// A device's heap sets and texture table (the module docs), under its `heaps` lock: taken
/// before its resource IDs' lock, never under it.
#[derive(Default)]
pub(super) struct DeviceHeaps {
    /// By resource ID, the textures placed: each live slot's holder.
    textures: BTreeMap<u64, Placed>,
    /// Made at the first command that binds a texture handle table or a heap set.
    table: Option<Table>,
    /// The live heap sets, by key ([`HeapSet`]).
    sets: BTreeMap<u64, SetEntry>,
    /// The live heap sets by layout, so the variants of one layout share one.
    by_layout: HashMap<DescriptorSetLayout, (u64, Weak<HeapSet>)>,
    next_set: u64,
    /// Gone textures' images, which wait for the command buffers submitted before they went.
    retirement: Retirement<Keeper>,
    /// What a rebind or a detach replaced ([`retire`]), which waits for the command buffers
    /// committed before it.
    replaced: Retirement<Keeper>,
    /// The batches holding the texture slots' writes ([`hold`]): while one is, a write waits in
    /// `deferred`.
    holds: usize,
    /// The slot writes asked while a batch held them, in order, made as the last hold ends.
    deferred: VecDeque<Slots>,
    /// By element of the sampler arrays, the sampler states placed: each live state, which every
    /// heap set holds at its element.
    sampler_states: BTreeMap<u64, PlacedSampler>,
    /// By resource ID, what the sampler tables map each sampler ID to: its slot's holder.
    sampler_slots: BTreeMap<u64, SamplerSlot>,
    /// The base of the sampler tables' high window, once a state took an ID past the low one.
    high_window: Option<u64>,
    /// The sampler tables, by what they map ([`SamplerTableKind`]), each made at the first
    /// command that binds one of its kind.
    sampler_tables: BTreeMap<SamplerTableKind, Table>,
}

/// A sampler state every heap set holds at its element: its hold serial and the state (not
/// held: upgraded only where the caller drops it once the lock is free).
struct PlacedSampler {
    serial: u64,
    state: Weak<SamplerShared>,
}

/// What the sampler tables map a sampler ID to: its slot's holder, by its hold serial, its
/// element, its word ([`SamplerWord`]) and whether it filters bicubically.
#[derive(Clone, Copy)]
struct SamplerSlot {
    serial: u64,
    element: u64,
    word: u32,
    cubic: bool,
}

/// What a sampler table maps (the module docs): the IDs of the states whose elements are below
/// `capacity`, the element count of the sampler arrays of the layouts that bind it, each to its
/// state; those of cubic states only where `cubic`, its commands' stages testing the states they
/// reach for a cubic filter (`RuntimeFact::CubicArgumentSampler`). Every other ID it maps to
/// the neutral.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SamplerTableKind {
    capacity: u32,
    cubic: bool,
}

impl SamplerTableKind {
    /// Where the table maps a sampler ID whose slot `slot` holds: its holder's element and word,
    /// or the neutral's.
    fn entry(self, slot: Option<&SamplerSlot>) -> (u32, u32) {
        match slot {
            Some(slot) if self.holds(slot) => (slot.element as u32, slot.word),
            _ => (0, 0),
        }
    }

    /// Whether the table maps a sampler ID to `slot`'s holder.
    fn holds(self, slot: &SamplerSlot) -> bool {
        slot.element < u64::from(self.capacity) && (self.cubic || !slot.cubic)
    }
}

/// A texture the table and the heap sets hold: its slot's generation and hold serial when it
/// was written, its table entry's flags, and the texture (not held: its ID stays the program's
/// to release; upgraded only where the caller drops it once the lock is free).
struct Placed {
    generation: u64,
    serial: u64,
    flags: u32,
    /// The key of the buffer whose bytes its images and texel views lie in
    /// (`Texture::bytes_key`), whose rebind or detach writes its elements again ([`renew_over`]).
    bytes: Option<ResourceKey>,
    texture: Weak<TextureShared>,
}

/// Images retired while command buffers that may read them run (the module docs): each waits
/// for the tickets taken before it.
#[derive(Debug)]
pub(super) struct Retirement<T> {
    /// The epoch a ticket taken now has; each retirement while tickets are out ends one.
    epoch: u64,
    /// The tickets out, by epoch.
    out: BTreeMap<u64, usize>,
    /// In epoch order.
    retired: VecDeque<(u64, T)>,
}

impl<T> Default for Retirement<T> {
    fn default() -> Retirement<T> {
        Retirement {
            epoch: 0,
            out: BTreeMap::new(),
            retired: VecDeque::new(),
        }
    }
}

impl<T> Retirement<T> {
    /// Retire `kept`: returned, to drop now, where no ticket is out.
    pub(super) fn retire(&mut self, kept: T) -> Option<T> {
        if self.out.is_empty() {
            return Some(kept);
        }
        self.retired.push_back((self.epoch, kept));
        self.epoch += 1;
        None
    }

    /// Take a ticket: its epoch.
    pub(super) fn take(&mut self) -> u64 {
        *self.out.entry(self.epoch).or_default() += 1;
        self.epoch
    }

    /// Give back a ticket of `epoch`: what no ticket out waits for any more, to drop.
    pub(super) fn give(&mut self, epoch: u64) -> Vec<T> {
        if let Some(count) = self.out.get_mut(&epoch) {
            *count -= 1;
            if *count == 0 {
                self.out.remove(&epoch);
            }
        }
        let least = self.out.keys().next().copied().unwrap_or(u64::MAX);
        let mut freed = Vec::new();
        while self
            .retired
            .front()
            .is_some_and(|(retired, _)| *retired < least)
        {
            freed.extend(self.retired.pop_front().map(|(_, kept)| kept));
        }
        freed
    }
}

/// The texture table's buffer, with the null function table's words past its entries.
struct Table {
    buffer: vk::Buffer,
    placement: Placement,
}

impl Table {
    /// A new table of `device`, entry `i` the neutral's at ID `id_at(i)`, and the null table's
    /// words zero.
    fn make(device: &Device, id_at: impl Fn(u64) -> u64) -> Result<Table, Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let null = FunctionTableWords::byte_len(0).next_multiple_of(16);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(TABLE_BYTES + null)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
        );
        // SAFETY: the device is live and `info` valid.
        let buffer = unsafe { crate::creation::buffer(raw, &info) }.map_err(vulkan)?;
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw,
            &requirements,
            MemoryClass::WriteCombined,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw, buffer) };
                return Err(error);
            }
        };
        let table = Table { buffer, placement };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        let bound = unsafe {
            crate::creation::bind_buffer(
                raw,
                buffer,
                table.placement.memory,
                table.placement.offset,
            )
        };
        if let Err(error) = bound.map_err(vulkan).and_then(|()| table.start()) {
            table.destroy(raw, &mut lock(&shared.memory));
            return Err(error);
        }
        for at in 0..u64::from(TABLE_ENTRIES) {
            table.put(at, id_at(at), 0, 0);
        }
        let start = table.start()?;
        // SAFETY: the null table's words lie inside the placement, which nothing reads yet.
        unsafe { std::ptr::write_bytes(start.add(TABLE_BYTES as usize), 0, null as usize) };
        Ok(table)
    }

    /// The host address of the table's first byte.
    fn start(&self) -> Result<*mut u8, Error> {
        self.placement
            .mapped
            .map(|mapped| mapped.as_ptr())
            .ok_or_else(|| internal("a handle table in memory the host does not see"))
    }

    /// Write entry `at`: ID `id`, element `element`, flags `flags`.
    fn put(&self, at: u64, id: u64, element: u32, flags: u32) {
        if at >= u64::from(TABLE_ENTRIES) {
            return;
        }
        let Ok(start) = self.start() else {
            return;
        };
        let entry = handle_entry(id, element, flags);
        let at = at as usize * entry.len();
        // SAFETY: `at` is below `TABLE_ENTRIES`, so the entry lies inside the mapped placement;
        // a command reading it meanwhile reads the old entry or the new one (the module docs).
        unsafe { std::ptr::copy_nonoverlapping(entry.as_ptr(), start.add(at), entry.len()) };
    }

    /// The descriptor of the table's first `entries` entries.
    fn entries(&self, entries: u32) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(0)
            .range(u64::from(entries) * u64::from(HANDLE_ENTRY_BYTES))
    }

    /// The descriptor of the null function table.
    fn null_words(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(TABLE_BYTES)
            .range(FunctionTableWords::byte_len(0))
    }

    /// Destroy the buffer, freeing its memory into `memory`. Nothing uses it.
    fn destroy(&self, raw: &ash::Device, memory: &mut Memory) {
        // SAFETY: the caller's.
        unsafe { release::object(raw, self.buffer) };
        memory.free(raw, &self.placement);
    }
}

impl DeviceHeaps {
    /// Destroy the texture and sampler tables, freeing their memory into `memory`: the device is
    /// idle.
    pub(super) fn destroy(&self, raw: &ash::Device, memory: &mut Memory) {
        for table in self.table.iter().chain(self.sampler_tables.values()) {
            table.destroy(raw, memory);
        }
    }

    /// The texture table of `device`, made at its first use.
    fn table(&mut self, device: &Device) -> Result<&Table, Error> {
        if self.table.is_none() {
            let table = Table::make(device, |at| at)?;
            for (&id, placed) in &self.textures {
                table.put(id, id, id as u32, placed.flags);
            }
            self.table = Some(table);
        }
        self.table
            .as_ref()
            .ok_or_else(|| internal("a texture table not made"))
    }

    /// Write `slots` ([`sync`]): the slots that failed, each with its error, their entries the
    /// neutral's. Their holders go to `held`, to drop once the lock is free.
    fn write_slots(
        &mut self,
        device: &Device,
        slots: Slots,
        held: &mut Vec<Texture>,
        notes: &mut Notes,
    ) -> Vec<(u64, Error)> {
        let (ids, force): (Vec<u64>, bool) = match slots {
            Slots::Changed(id) => (vec![id], false),
            Slots::All => (self.textures.keys().copied().collect(), true),
            Slots::Over(key) => {
                let over = self.textures.iter();
                let over = over.filter(|(_, placed)| placed.bytes == Some(key));
                (over.map(|(&id, _)| id).collect(), true)
            }
        };
        if force {
            device.shared().counts.heap_slots_rewritten(ids.len());
        }
        let mut failed = Vec::new();
        for id in ids {
            if let Err(error) = self.sync(device, id, force, held, notes) {
                let cleared = self.clear(device, id);
                failed.push((id, cleared.err().unwrap_or(error)));
            }
        }
        failed
    }

    /// Write texture slot `id` into the table and every heap set where its holder changed since
    /// (whatever changed where `force`, a rebind's): its holder's elements, or the neutral's entry where
    /// it has none. The holder goes to `held`, to drop once the lock is free.
    fn sync(
        &mut self,
        device: &Device,
        id: u64,
        force: bool,
        held: &mut Vec<Texture>,
        notes: &mut Notes,
    ) -> Result<(), Error> {
        if id == 0 || id >= u64::from(TABLE_ENTRIES) {
            return Ok(());
        }
        let (generation, holder) = lock(&device.shared().resource_ids).texture(id);
        let unchanged = self
            .textures
            .get(&id)
            .is_some_and(|placed| placed.generation == generation);
        if unchanged && !force {
            return Ok(());
        }
        let Some((serial, weak, texture)) = holder.and_then(|(serial, weak)| {
            let texture = Texture::upgrade(&weak)?;
            Some((serial, weak, texture))
        }) else {
            return self.clear(device, id);
        };
        // The holder may be a texture's last handle: it goes once the lock is free.
        held.push(texture.clone());
        let facts = texture.facts();
        let flags = handle_entry_flags(&facts);
        let bytes = texture.bytes_key();
        // The neutral's entry until every set holds the texture.
        if let Some(table) = &self.table {
            table.put(id, id, 0, 0);
        }
        self.unplace(device, id);
        for entry in self.sets.values_mut() {
            let placed = texture_writes(device, &entry.images, id, &texture, notes)?;
            entry.write(device, placed.writes)?;
            entry.shadow(id, &texture, placed.shadow);
            if let Some(read) = entry.twin(id, placed.twin) {
                notes.twin_placed = Some(read);
            }
            if let Some(read) = entry.raw(id, placed.raw) {
                notes.raw_placed = Some(read);
            }
        }
        if let Some(table) = &self.table {
            table.put(id, id, id as u32, flags);
        }
        self.textures.insert(
            id,
            Placed {
                generation,
                serial,
                flags,
                bytes,
                texture: weak,
            },
        );
        self.note_textures_end(device);
        notes.placed(&texture, !self.sets.is_empty());
        Ok(())
    }

    /// Forget the texture placed at ID `id`.
    fn unplace(&mut self, device: &Device, id: u64) {
        self.textures.remove(&id);
        self.note_textures_end(device);
    }

    /// One past the highest ID a texture is placed at, 0 for none, where a command reads it
    /// without the lock (`DeviceShared::heap_textures_end`, [`TextureTableAt::table`]).
    fn note_textures_end(&self, device: &Device) {
        let end = self.textures.keys().next_back().map_or(0, |&id| id + 1);
        (device.shared().heap_textures_end).store(end, Ordering::Release);
    }

    /// Clear ID `id`: the neutral's entry, no element of it kept for a version made later, and
    /// the neutral at it in each version a command buffer has yet to commit (the module docs).
    fn clear(&mut self, device: &Device, id: u64) -> Result<(), Error> {
        self.unplace(device, id);
        for entry in self.sets.values_mut() {
            entry.shadows.remove(&id);
            entry.twins.remove(&id);
            entry.raws.remove(&id);
        }
        if let Some(table) = &self.table {
            table.put(id, id, 0, 0);
        }
        let Ok(element) = u32::try_from(id) else {
            return Ok(());
        };
        for entry in self.sets.values_mut() {
            for array in &entry.images {
                entry.written.remove(&(array.binding, element));
            }
            let open = entry.unsubmitted();
            if open.is_empty() {
                continue;
            }
            let neutrals: Vec<Write> = entry
                .images
                .iter()
                .filter(|array| element < array.count)
                .map(|array| raw_write(array.binding, element, array.ty, image_info(array.neutral)))
                .collect();
            for version in &open {
                // SAFETY: no submission of the version's command buffer took it (its flag is set
                // under the `heaps` lock held here, before the submission), so nothing pending
                // binds it, and its set is `UPDATE_AFTER_BIND`; the neutral views live with the
                // device.
                unsafe { write_sets(device, &[version.set], &neutrals) }?;
            }
        }
        Ok(())
    }

    /// A new heap set of `layout`, holding every live texture and sampler state. The states it
    /// upgrades go to `held_samplers`, to drop once the lock is free.
    fn made_set(
        &mut self,
        device: &Device,
        layout: &DescriptorSetLayout,
        (held, held_samplers): (&mut Vec<Texture>, &mut Vec<SamplerState>),
        notes: &mut Notes,
    ) -> Result<Arc<HeapSet>, Error> {
        let needs = set_needs(layout)?;
        let raw_layout = Arc::new(RawLayout::create(device, layout)?);
        let key = self.next_set;
        // The one version every command binds, where the device lets a pending one's set be
        // written; else each command buffer makes its own ([`bind`]).
        let shared = match unused_while_pending(device) {
            true => Some(Arc::new(SetVersion::allocate(
                device,
                &raw_layout,
                &needs,
                key,
            )?)),
            false => None,
        };
        let null_words = self.table(device)?.null_words();
        let mut images = Vec::new();
        let mut samplers = Vec::new();
        let mut neutrals = Vec::new();
        for binding in &layout.bindings {
            let ty: vk::DescriptorType = binding.ty.into();
            let info = match binding.source {
                BindingSource::Heap(HandleClass::Image(class)) => {
                    let neutral = NeutralResource::for_image(&class)
                        .ok_or_else(|| internal("a heap array of a class without a neutral"))?;
                    let neutral = device.neutral_view(neutral)?;
                    images.push(ImageArray {
                        binding: binding.binding,
                        class,
                        ty,
                        count: binding.count,
                        neutral,
                    });
                    image_info(neutral)
                }
                BindingSource::Heap(class @ HandleClass::Sampler { compare, .. }) => {
                    let neutral = device.neutral_sampler(compare)?;
                    let use_ = class
                        .sampler_use()
                        .ok_or_else(|| internal("a sampler class without a use"))?;
                    samplers.push(SamplerArray {
                        binding: binding.binding,
                        use_,
                        count: binding.count,
                        neutral,
                    });
                    sampler_info(neutral)
                }
                BindingSource::Heap(HandleClass::FunctionTable) => Info::Buffer(null_words),
                _ => return Err(internal("a heap set binding of no heap array")),
            };
            neutrals.push(HeapWrite {
                binding: binding.binding,
                element: 0,
                ty,
                info,
                keeper: None,
            });
        }
        let mut entry = SetEntry {
            key,
            layout: layout.clone(),
            raw_layout,
            needs,
            images,
            samplers,
            shared,
            open: Vec::new(),
            written: BTreeMap::new(),
            shadows: BTreeMap::new(),
            twins: BTreeMap::new(),
            raws: BTreeMap::new(),
        };
        entry.write(device, neutrals)?;
        let mut writes = Vec::new();
        for (&id, placed) in &self.textures {
            if let Some(texture) = Texture::upgrade(&placed.texture) {
                held.push(texture.clone());
                let placed = texture_writes(device, &entry.images, id, &texture, notes)?;
                writes.extend(placed.writes);
                entry.shadow(id, &texture, placed.shadow);
                // A new set no command bound yet: its first binding announces.
                entry.twin(id, placed.twin);
                entry.raw(id, placed.raw);
                notes.placed(&texture, true);
            }
        }
        for placed in self.sampler_states.values() {
            if let Some(state) = SamplerState::upgrade(&placed.state) {
                writes.extend(sampler_writes(device, &entry.samplers, &state, notes));
                held_samplers.push(state);
            }
        }
        entry.write(device, writes)?;
        self.next_set += 1;
        let set = Arc::new(HeapSet {
            device: device.clone(),
            key,
        });
        self.sets.insert(key, entry);
        self.by_layout
            .insert(layout.clone(), (key, Arc::downgrade(&set)));
        Ok(set)
    }
}

impl DeviceHeaps {
    /// Place sampler state `state`, which just took its ID's slot: its elements into every heap
    /// set, then its ID's entry into every sampler table where it still holds the slot (the
    /// module docs). The states upgraded go to `held`, to drop once the lock is free.
    fn place_sampler(
        &mut self,
        device: &Device,
        state: &SamplerState,
        held: &mut Vec<SamplerState>,
        notes: &mut Notes,
    ) -> Result<(), Error> {
        for entry in self.sets.values_mut() {
            let writes = sampler_writes(device, &entry.samplers, state, notes);
            entry.write_sampler(device, writes)?;
        }
        self.sampler_states.insert(
            state.heap_element(),
            PlacedSampler {
                serial: state.id_serial(),
                state: state.downgrade(),
            },
        );
        self.sync_sampler_slot(device, state.gpu_resource_id().0, held, notes);
        Ok(())
    }

    /// Write sampler ID `id`'s entry into every sampler table: its slot's holder where that is a
    /// placed state, else the neutral.
    fn sync_sampler_slot(
        &mut self,
        device: &Device,
        id: u64,
        held: &mut Vec<SamplerState>,
        notes: &mut Notes,
    ) {
        let holder = lock(&device.shared().resource_ids).sampler(id);
        let slot = holder
            .and_then(|(serial, weak)| {
                let state = SamplerState::upgrade(&weak)?;
                held.push(state.clone());
                Some((serial, state))
            })
            .filter(|(serial, state)| {
                self.sampler_states
                    .get(&state.heap_element())
                    .is_some_and(|placed| placed.serial == *serial)
            })
            .map(|(serial, state)| SamplerSlot {
                serial,
                element: state.heap_element(),
                word: SamplerWord::of(state.descriptor()).0,
                cubic: SamplerWord::of(state.descriptor()).0 & SamplerWord::CUBIC != 0,
            });
        match slot {
            Some(slot) => {
                self.put_sampler(id, Some(&slot), notes);
                self.sampler_slots.insert(id, slot);
            }
            None => {
                if self.sampler_slots.remove(&id).is_some() {
                    self.put_sampler(id, None, notes);
                }
            }
        }
    }

    /// Write sampler ID `id`'s entry into every sampler table: `slot`'s holder where the table
    /// holds it, else the neutral. An ID past the low window takes the high window's base where
    /// it has none yet (`slot` given).
    fn put_sampler(&mut self, id: u64, slot: Option<&SamplerSlot>, notes: &mut Notes) {
        if id >= SAMPLER_WINDOW && self.high_window.is_none() && slot.is_some() {
            self.open_high_window(high_window_base(id));
        }
        let Some(at) = sampler_entry(id, self.high_window) else {
            if slot.is_some() {
                let base = self.high_window.unwrap_or(SAMPLER_WINDOW);
                notes.sampler_outside(id, base);
            }
            return;
        };
        for (kind, table) in &self.sampler_tables {
            let (element, word) = kind.entry(slot);
            table.put(at, id, element, word);
            if let Some(slot) = slot.filter(|slot| !kind.holds(slot)) {
                match slot.element < u64::from(kind.capacity) {
                    true => notes.sampler_cubic(id),
                    false => notes.sampler_past(id, u64::from(kind.capacity)),
                }
            }
        }
    }

    /// Start the sampler tables' high window at `base`: each table's entries of it are rewritten
    /// from the highest down, the neutral's at the window's IDs, so every table stays sorted
    /// while a command searches it. No state holds an ID of the window yet.
    fn open_high_window(&mut self, base: u64) {
        self.high_window = Some(base);
        for table in self.sampler_tables.values() {
            for i in (0..SAMPLER_WINDOW).rev() {
                table.put(SAMPLER_WINDOW + i, base + i, 0, 0);
            }
        }
    }

    /// Sampler state `state`'s placement and slot go, where they are its hold `serial`'s: its
    /// ID's entry the neutral's, and its elements' samplers into `kept`, to retire, the neutral
    /// at its element in each set's version no submission took yet (the module docs).
    fn sampler_gone(
        &mut self,
        device: &Device,
        (id, serial, element): (u64, u64, u64),
        kept: &mut Vec<Keeper>,
        notes: &mut Notes,
    ) -> Result<(), Error> {
        if self
            .sampler_slots
            .get(&id)
            .is_some_and(|slot| slot.serial == serial)
        {
            self.sampler_slots.remove(&id);
            self.put_sampler(id, None, notes);
        }
        let placed = self
            .sampler_states
            .get(&element)
            .is_some_and(|placed| placed.serial == serial);
        if !placed {
            return Ok(());
        }
        self.sampler_states.remove(&element);
        let Ok(element) = u32::try_from(element) else {
            return Ok(());
        };
        for entry in self.sets.values_mut() {
            let mut neutrals = Vec::new();
            for array in &entry.samplers {
                if let Some(written) = entry.written.remove(&(array.binding, element)) {
                    kept.extend(written.keeper);
                }
                if element < array.count {
                    let info = sampler_info(array.neutral);
                    neutrals.push(raw_write(
                        array.binding,
                        element,
                        vk::DescriptorType::SAMPLER,
                        info,
                    ));
                }
            }
            if neutrals.is_empty() {
                continue;
            }
            for version in &entry.unsubmitted() {
                // SAFETY: as `DeviceHeaps::clear`'s: nothing pending binds the version, its set
                // is `UPDATE_AFTER_BIND`, and the neutral samplers live with the device.
                unsafe { write_sets(device, &[version.set], &neutrals) }?;
            }
        }
        Ok(())
    }

    /// The sampler table of `kind`, made at its first use with every placed state's entry.
    fn sampler_table(&mut self, device: &Device, kind: SamplerTableKind) -> Result<&Table, Error> {
        if !self.sampler_tables.contains_key(&kind) {
            let high = self.high_window.unwrap_or(SAMPLER_WINDOW);
            let table = Table::make(device, |at| match at < SAMPLER_WINDOW {
                true => at,
                false => high + (at - SAMPLER_WINDOW),
            })?;
            for (&id, slot) in &self.sampler_slots {
                if let Some(at) = sampler_entry(id, self.high_window) {
                    let (element, word) = kind.entry(Some(slot));
                    table.put(at, id, element, word);
                }
            }
            self.sampler_tables.insert(kind, table);
        }
        self.sampler_tables
            .get(&kind)
            .ok_or_else(|| internal("a sampler table not made"))
    }
}

/// The entry of sampler ID `id` in a sampler table whose high window starts at `high` (the
/// module docs): `None` where it is in neither window.
pub(super) fn sampler_entry(id: u64, high: Option<u64>) -> Option<u64> {
    if id < SAMPLER_WINDOW {
        return Some(id);
    }
    let base = high.unwrap_or(SAMPLER_WINDOW);
    let offset = id.checked_sub(base)?;
    (offset < SAMPLER_WINDOW).then_some(SAMPLER_WINDOW + offset)
}

/// The base of the high window the first sampler ID past the low window, `id`, opens: the low
/// window's end where `id` lies within one window of it, else [`HIGH_WINDOW_SLACK`] below `id`,
/// rounded down to a multiple of it, so states that take IDs just below it (a reservation's plain
/// IDs, which other holders of sampler IDs took first and gave back) lie in it too.
pub(super) fn high_window_base(id: u64) -> u64 {
    if id < 2 * SAMPLER_WINDOW {
        return SAMPLER_WINDOW;
    }
    let below = id.saturating_sub(HIGH_WINDOW_SLACK);
    (below - below % HIGH_WINDOW_SLACK).max(SAMPLER_WINDOW)
}

/// Place `state`, a sampler state that just took its ID's slot, into every heap set and sampler
/// table (the module docs). A failure to write its elements leaves its ID's entries the
/// neutral's, announced.
pub(super) fn place_sampler(device: &Device, state: &SamplerState) {
    let mut held = Vec::new();
    let mut notes = Notes::default();
    let placed = {
        let mut heaps = lock(&device.shared().heaps);
        let placed = heaps.place_sampler(device, state, &mut held, &mut notes);
        if placed.is_err() {
            heaps.sampler_slots.remove(&state.gpu_resource_id().0);
            heaps.put_sampler(state.gpu_resource_id().0, None, &mut notes);
        }
        placed
    };
    drop(held);
    notes.announce(device);
    if let Err(error) = placed {
        device.degrade(Degradation::UnlistedArgumentSampler, || {
            (
                format!("sampler state at resource ID {}", state.gpu_resource_id().0),
                format!(
                    "writing its elements of the set-1 sampler arrays failed ({error}), so a \
                     function reading it through an argument buffer reads the neutral sampler"
                ),
            )
        });
    }
}

/// A sampler state at ID `id`, of hold serial `serial` and element `element`, is going: the
/// sampler tables map its ID to the neutral where it held the slot, and the Vulkan samplers the
/// heap sets held at its element wait for the command buffers that may read them (the module
/// docs). Before its element is given back.
pub(super) fn sampler_gone(device: &Device, id: u64, serial: u64, element: u64) {
    let mut notes = Notes::default();
    let mut kept = Vec::new();
    let (freed, gone) = {
        let mut heaps = lock(&device.shared().heaps);
        let gone = heaps.sampler_gone(device, (id, serial, element), &mut kept, &mut notes);
        let freed = match kept.is_empty() {
            true => None,
            false => heaps.retirement.retire(Box::new(kept) as Keeper),
        };
        (freed, gone)
    };
    // The samplers go with no lock held.
    drop(freed);
    notes.announce(device);
    if let Err(error) = gone {
        device.degrade(Degradation::UnlistedArgumentSampler, || {
            (
                format!("sampler state at resource ID {id}"),
                format!("writing the neutral at its elements failed: {error}"),
            )
        });
    }
}

/// Whether the device's heap sets take writes while command buffers that bind them are pending:
/// one version of each set, `UPDATE_UNUSED_WHILE_PENDING` (the module docs).
pub(super) fn unused_while_pending(device: &Device) -> bool {
    metal2vulkan::pipeline::heap_unused_while_pending(device.compile_device().profile())
}

/// One image array of a heap set: its binding, class, descriptor type, element count, and
/// neutral's view (the device's).
struct ImageArray {
    binding: u32,
    class: ImageHandleClass,
    ty: vk::DescriptorType,
    count: u32,
    neutral: vk::ImageView,
}

/// One sampler array of a heap set: its binding, the use its class makes a state's Vulkan sampler
/// for, its element count, and its neutral sampler (the device's).
struct SamplerArray {
    binding: u32,
    use_: SamplerUse,
    count: u32,
    neutral: vk::Sampler,
}

/// One element a heap set holds: the descriptor and what keeps what it names (a sampler, a
/// table's words; a texture's image is retired as the texture goes instead).
struct HeapWrite {
    binding: u32,
    element: u32,
    ty: vk::DescriptorType,
    info: Info,
    keeper: Option<Keeper>,
}

/// One live heap set's state (the module docs).
struct SetEntry {
    /// Its key ([`HeapSet`]).
    key: u64,
    layout: DescriptorSetLayout,
    raw_layout: Arc<RawLayout>,
    needs: SetNeeds,
    images: Vec<ImageArray>,
    samplers: Vec<SamplerArray>,
    /// The one version every command binds, where the device has
    /// `descriptorBindingUpdateUnusedWhilePending`.
    shared: Option<Arc<SetVersion>>,
    /// Where it lacks it, the command buffers' own versions ([`bind`]), while they live and no
    /// submission took them.
    open: Vec<Weak<SetVersion>>,
    /// Every element written, by `(binding, element)`, but those of textures gone: what a
    /// command buffer's own version holds as it is made.
    written: BTreeMap<(u32, u32), Written>,
    /// By resource ID, the shadowed textures whose shadow image's view an array of the set
    /// holds, with the key a command's uses name each by (`Texture::bound_use`): a command that
    /// binds the set and neither names one nor makes it resident is announced
    /// (`Degradation::UnnamedShadowedTexture`).
    shadows: BTreeMap<u64, ResourceKey>,
    /// By resource ID, the textures whose decoded twin's view an array of the set holds where
    /// the twin's texels are not Metal's decode exactly, with what a read of it announces
    /// (`decoded-texel-precision`, `texture/decoded_twin.rs`): a command that binds the set may
    /// read it through its ID, named or not.
    twins: BTreeMap<u64, TwinRead>,
    /// By resource ID, the textures whose raw view an array of the set holds (a heap holds them
    /// by their raw texels alone, [`RawRead`]): a command that binds the set may read one through
    /// its ID, named or not, through its function's decode.
    raws: BTreeMap<u64, RawRead>,
}

/// An element a heap set holds ([`HeapWrite`]).
struct Written {
    ty: vk::DescriptorType,
    info: Info,
    /// Held while the set holds the element; dropped once another takes its place, the
    /// command buffers that bound it holding their own, or, a sampler state's, retired as the
    /// state goes ([`sampler_gone`]).
    keeper: Option<Keeper>,
}

impl SetEntry {
    /// Whether a command that binds it finds nothing to announce of what it holds: no shadowed
    /// texture, decoded twin or raw read ([`bind`]).
    fn notes_nothing(&self) -> bool {
        self.shadows.is_empty() && self.twins.is_empty() && self.raws.is_empty()
    }

    /// Note texture `texture` at ID `id` among the shadowed textures the set holds a view of
    /// where `shadow` (its arrays' writes hold its shadow image's view), else not.
    fn shadow(&mut self, id: u64, texture: &Texture, shadow: bool) {
        match shadow {
            true => {
                let key = texture.bound_use(Access::NONE, Access::NONE).key;
                self.shadows.insert(id, key)
            }
            false => self.shadows.remove(&id),
        };
    }

    /// Note texture `id` among the textures whose decoded twin's view the set holds, with what a
    /// read of it announces, where `read` (else not); answer `read` where a command that bound
    /// the set may yet read the view through its ID ([`pending_read`](Self::pending_read)).
    fn twin(&mut self, id: u64, read: Option<TwinRead>) -> Option<TwinRead> {
        let read = noted(&mut self.twins, id, read)?;
        self.pending_read().then_some(read)
    }

    /// Note texture `id` among the textures whose raw view the set holds, with what a read of it
    /// announces, where `read` (else not); answer `read` where a command that bound the set may
    /// yet read the view through its ID, as [`twin`](Self::twin) does.
    fn raw(&mut self, id: u64, read: Option<RawRead>) -> Option<RawRead> {
        let read = noted(&mut self.raws, id, read)?;
        self.pending_read().then_some(read)
    }

    /// Whether a command that bound the set may yet read an element placed now: the set's shared
    /// version is held by a command buffer that bound it and has not completed (each holds it
    /// until then, [`HeapUse`]), or a command buffer that bound its own version has not committed
    /// (which takes the element).
    fn pending_read(&mut self) -> bool {
        let shared = (self.shared.as_ref()).is_some_and(|shared| Arc::strong_count(shared) > 1);
        shared || !self.open().is_empty()
    }

    /// The command buffers' own versions whose command buffer has not committed.
    fn open(&mut self) -> Vec<Arc<SetVersion>> {
        self.unsubmitted_where(|version| !version.committed.load(Ordering::Acquire))
    }

    /// The command buffers' own versions no submission took yet, committed or not.
    fn unsubmitted(&mut self) -> Vec<Arc<SetVersion>> {
        self.unsubmitted_where(|_| true)
    }

    /// The own versions no submission took yet that `pick` picks (those gone or submitted leave
    /// `open`).
    fn unsubmitted_where(&mut self, pick: impl Fn(&SetVersion) -> bool) -> Vec<Arc<SetVersion>> {
        let mut picked = Vec::new();
        self.open.retain(|version| match version.upgrade() {
            Some(version) if !version.submitted.load(Ordering::Acquire) => {
                if pick(&version) {
                    picked.push(version);
                }
                true
            }
            _ => false,
        });
        picked
    }

    /// The versions writes go to: the shared one, and the [`open`](SetEntry::open) ones.
    fn targets(&mut self) -> Vec<Arc<SetVersion>> {
        let mut targets: Vec<Arc<SetVersion>> = self.shared.iter().cloned().collect();
        targets.extend(self.open());
        targets
    }

    /// Write `writes` into every version writes go to ([`SetEntry::targets`]), and keep them
    /// for a version made later.
    fn write(&mut self, device: &Device, writes: Vec<HeapWrite>) -> Result<(), Error> {
        let targets = self.targets();
        self.write_into(device, writes, targets)
    }

    /// Write a sampler state's `writes` into the shared version and every own version no
    /// submission took yet, committed or not (the module docs' **Samplers**), and keep them for a
    /// version made later.
    fn write_sampler(&mut self, device: &Device, writes: Vec<HeapWrite>) -> Result<(), Error> {
        let mut targets: Vec<Arc<SetVersion>> = self.shared.iter().cloned().collect();
        targets.extend(self.unsubmitted());
        self.write_into(device, writes, targets)
    }

    /// Write `writes` into `targets`, and keep them for a version made later.
    fn write_into(
        &mut self,
        device: &Device,
        writes: Vec<HeapWrite>,
        targets: Vec<Arc<SetVersion>>,
    ) -> Result<(), Error> {
        if writes.is_empty() {
            return Ok(());
        }
        let raw: Vec<Write> = writes
            .iter()
            .map(|write| raw_write(write.binding, write.element, write.ty, write.info))
            .collect();
        for version in targets {
            // SAFETY: the shared version's arrays are `UPDATE_UNUSED_WHILE_PENDING`, and no
            // pending command buffer reads the elements written (the module docs); a command
            // buffer's own version is written only until it commits, or, a sampler state's
            // elements, until a submission takes it (under this lock, before the submission), and
            // its set is `UPDATE_AFTER_BIND`. Every view, sampler and buffer written is held while
            // the set holds it.
            unsafe { write_sets(device, &[version.set], &raw) }?;
        }
        for write in writes {
            self.written.insert(
                (write.binding, write.element),
                Written {
                    ty: write.ty,
                    info: write.info,
                    keeper: write.keeper,
                },
            );
        }
        Ok(())
    }

    /// A command buffer's own version of the set (the module docs): every image array's
    /// elements a texture's ID may name, and every sampler array's elements, the neutral, then
    /// every element `written`. `notes` takes the count written.
    fn own_version(
        &mut self,
        device: &Device,
        notes: &mut Notes,
    ) -> Result<Arc<SetVersion>, Error> {
        let version = SetVersion::allocate(device, &self.raw_layout, &self.needs, self.key)?;
        let mut all: Vec<Write> = Vec::new();
        for array in &self.images {
            let reach = array.count.min(TABLE_ENTRIES);
            all.extend((0..reach).map(|element| {
                raw_write(array.binding, element, array.ty, image_info(array.neutral))
            }));
        }
        for array in &self.samplers {
            all.extend((0..array.count).map(|element| {
                let info = sampler_info(array.neutral);
                raw_write(array.binding, element, vk::DescriptorType::SAMPLER, info)
            }));
        }
        all.extend(self.written.iter().map(|(&(binding, element), written)| {
            raw_write(binding, element, written.ty, written.info)
        }));
        // SAFETY: the set was just allocated; every view, sampler and buffer `written` names is
        // held (a texture's until its image retires), and the neutral views live with the device.
        unsafe { write_sets(device, &[version.set], &all) }?;
        notes.copied = Some(all.len());
        let version = Arc::new(version);
        self.open.push(Arc::downgrade(&version));
        Ok(version)
    }
}

/// The write of `info` at element `element` of binding `binding` of a heap set.
fn raw_write(binding: u32, element: u32, ty: vk::DescriptorType, info: Info) -> Write {
    Write {
        set: 0,
        binding,
        element,
        ty,
        info,
        held: None,
    }
}

/// An image view's descriptor info, in `GENERAL` (`binding.rs`).
fn image_info(view: vk::ImageView) -> Info {
    Info::Image(
        vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL),
    )
}

/// A sampler's descriptor info.
fn sampler_info(sampler: vk::Sampler) -> Info {
    Info::Image(vk::DescriptorImageInfo::default().sampler(sampler))
}

/// The writes of sampler state `state` into `arrays`: its element of each, which holds the Vulkan
/// sampler made for the array's use (`bind::sampler_desc`, as a command's argument sampler is
/// made). An element past an array's count is left out, noted (no table of its arrays maps the
/// state's ID there, [`SamplerTableKind`]); where its use refuses the state or the sampler is
/// not made (the device's samplers at their limit), the element holds the neutral sampler,
/// noted.
fn sampler_writes(
    device: &Device,
    arrays: &[SamplerArray],
    state: &SamplerState,
    notes: &mut Notes,
) -> Vec<HeapWrite> {
    let id = state.gpu_resource_id().0;
    let element = state.heap_element();
    let mut writes = Vec::with_capacity(arrays.len());
    for array in arrays {
        let element = match u32::try_from(element) {
            Ok(element) if element < array.count => element,
            _ => {
                notes.sampler_past(id, u64::from(array.count));
                continue;
            }
        };
        let made = metal2vulkan::bind::sampler_desc(
            device.compile_device(),
            state.descriptor(),
            array.use_,
            OpaqueBlackBorder::Stated,
            None,
        )
        .map_err(Error::Bind)
        .and_then(|desc| state.vk_sampler(&desc));
        let (info, keeper) = match made {
            Ok(sampler) => (
                sampler_info(sampler.raw()),
                Some(Box::new(sampler) as Keeper),
            ),
            Err(error) => {
                notes.sampler_unmade(id, &error);
                (sampler_info(array.neutral), None)
            }
        };
        writes.push(HeapWrite {
            binding: array.binding,
            element,
            ty: vk::DescriptorType::SAMPLER,
            info,
            keeper,
        });
    }
    writes
}

/// Note `read` at ID `id` of `reads`, or none there where it is `None`; answer `read`.
fn noted<R: Clone>(reads: &mut BTreeMap<u64, R>, id: u64, read: Option<R>) -> Option<R> {
    let Some(read) = read else {
        reads.remove(&id);
        return None;
    };
    reads.insert(id, read.clone());
    Some(read)
}

/// A texture a heap holds by its raw texels alone (`bind::heap_raw_form`: an extended-range,
/// padded 10-bit or bfloat form of `ExtendedRange::HEAP_READ`), whose raw view a set's unsigned
/// integer array holds: a function's `float` read, sample or gather of it through its ID decodes
/// the raw texels itself, by the device's arithmetic, so a command that may read it announces
/// `device-pack-conversion` with the texture's name, once per device (at each occurrence where
/// the options ask for that), as a decoded twin's view announces its own ([`TwinRead`]). Which
/// textures a function reads through an argument buffer the runtime does not see: a command that
/// binds a set holding such a view announces it whether or not its function reads the texture,
/// and one whose sets hold none announces nothing for it.
#[derive(Clone)]
struct RawRead {
    object: Arc<str>,
}

impl RawRead {
    /// Why a read of one differs from Metal's.
    const WHY: &'static str = "a heap holds the texture by its raw texels alone, which the \
         function's float read, sample or gather decodes itself (no Vulkan format holds them) by \
         the device's arithmetic, where Metal divides correctly rounded and rounds a half result \
         with ties away from zero: a linear value within one f32 ulp of Metal's, a half result \
         within one half ulp, a bfloat NaN's bits as they are";

    /// What a read of texture `texture`, at ID `id`, announces.
    fn of(texture: &Texture, id: u64) -> RawRead {
        RawRead {
            object: format!("{:?} texture at resource ID {id}", texture.pixel_format()).into(),
        }
    }

    /// Announce it on `device`, where a command that binds a set holding it may read it.
    fn announce(&self, device: &Device) {
        device.degrade(Degradation::DevicePackConversion, || {
            (self.object.to_string(), Self::WHY.to_owned())
        });
    }

    /// Announce it on `device` once per device, whatever its options ask: as it is placed in a
    /// set a pending command buffer may read through, which a submission may do on a runtime
    /// thread.
    fn announce_once(&self, device: &Device) {
        device.degrade_once(Degradation::DevicePackConversion, || {
            (self.object.to_string(), Self::WHY.to_owned())
        });
    }
}

/// What [`texture_writes`] places of a texture into a set's arrays.
struct TexturePlaced {
    writes: Vec<HeapWrite>,
    /// An element is its shadow image's view.
    shadow: bool,
    /// An element is its decoded twin's view whose texels are not Metal's decode exactly: what a
    /// read of it announces.
    twin: Option<TwinRead>,
    /// An element is its raw view, which a function's read decodes: what a read of it announces.
    raw: Option<RawRead>,
}

/// The writes of texture `texture`, at ID `id`, into `arrays`: element `id` of each, which it
/// holds as the array's class takes it (the module docs). An ID past an array's count is left
/// out, noted. With whether an element is its shadow image's view, and what a read of its
/// decoded twin's view, or of its raw view, announces where an element is that.
fn texture_writes(
    device: &Device,
    arrays: &[ImageArray],
    id: u64,
    texture: &Texture,
    notes: &mut Notes,
) -> Result<TexturePlaced, Error> {
    let mut placed = TexturePlaced {
        writes: Vec::with_capacity(arrays.len()),
        shadow: false,
        twin: None,
        raw: None,
    };
    for array in arrays {
        let element = match u32::try_from(id) {
            Ok(element) if element < array.count => element,
            _ => {
                notes.past = Some((id, array.count));
                continue;
            }
        };
        let (info, shadowed, twin, raw) = texture_info(device, array, texture, notes)?;
        placed.shadow |= shadowed;
        if twin.is_some() {
            placed.twin = twin;
        }
        if raw && placed.raw.is_none() {
            placed.raw = Some(RawRead::of(texture, id));
        }
        placed.writes.push(HeapWrite {
            binding: array.binding,
            element,
            ty: array.ty,
            info,
            keeper: None,
        });
    }
    Ok(placed)
}

/// What an element of `array` holds for `texture` (the module docs), whether it is the
/// texture's shadow image's view, what a read of it announces where it is a decoded twin's
/// view whose texels are not Metal's decode exactly (`Texture::held_view`), and whether it is
/// the raw view of a texture a heap holds by its raw texels alone ([`RawRead`]).
fn texture_info(
    device: &Device,
    array: &ImageArray,
    texture: &Texture,
    notes: &mut Notes,
) -> Result<(Info, bool, Option<TwinRead>, bool), Error> {
    let neutral = || Ok((image_info(array.neutral), false, None, false));
    let profile = device.compile_device().profile();
    let class = HandleClass::Image(array.class);
    let facts = texture.facts();
    let desc = match heap_element(
        profile,
        &class,
        &facts,
        metal2vulkan::vk::ImageLayout::GENERAL,
    ) {
        Ok(HeapElement::View(desc)) => desc,
        Ok(_) => return neutral(),
        Err(BindError::Unsupported(_)) => {
            notes.unviewable(texture);
            return neutral();
        }
        Err(error) => return Err(Error::Bind(error)),
    };
    let storage = array.ty == vk::DescriptorType::STORAGE_IMAGE;
    // The unsigned integer class holds such a texture's raw view (`bind::heap::fit`).
    let raw = array.class.access == ImageAccess::Sampled
        && array.class.sampled_type == SampledType::Uint
        && heap_raw_form(&facts).is_some();
    match texture.held_view("new_texture", &desc, storage) {
        Ok((view, twin)) => Ok((image_info(view), texture.shadow().is_some(), twin, raw)),
        // The class needs what the texture's usage or levels lack: Metal's read is undefined.
        Err(Error::Invalid(_)) => neutral(),
        Err(Error::Unsupported(_)) => {
            notes.unviewable(texture);
            neutral()
        }
        Err(error) => Err(error),
    }
}

/// What placing textures and writing sets announces, once no lock is held.
#[derive(Default)]
struct Notes {
    /// A texture a class fits that it cannot view.
    unviewable: Option<Texture>,
    /// The resource ID of a shadowed texture a set holds that the command binding the set
    /// neither names nor makes resident.
    shadowed: Option<u64>,
    /// An ID past an array's count, and the count.
    past: Option<(u64, u32)>,
    /// The elements a command buffer's own version of a set was made with.
    copied: Option<usize>,
    /// A texture a set holds whose blocks a carrier holds for copies (`pvrtc.rs`).
    unreadable: Option<Texture>,
    /// A sampler state a sampler table or a set's sampler array cannot hold: the first one's
    /// notice (`Degradation::UnlistedArgumentSampler`).
    unlisted: Option<(String, String)>,
    /// A decoded twin whose texels are not Metal's decode exactly in a set the command binds,
    /// whose function may read it (`Degradation::DecodedTexelPrecision`).
    twin_read: Option<TwinRead>,
    /// One placed in a set a command buffer that bound it and has not completed may read through
    /// ([`SetEntry::twin`]): announced once, as a submission may place it on a runtime thread.
    twin_placed: Option<TwinRead>,
    /// A texture a heap holds by its raw texels alone in a set the command binds, whose function
    /// may read it through its decode ([`RawRead`]).
    raw_read: Option<RawRead>,
    /// One placed in a set a command buffer that bound it and has not completed may read through
    /// ([`SetEntry::raw`]): announced once, as `twin_placed` is.
    raw_placed: Option<RawRead>,
}

impl Notes {
    fn unviewable(&mut self, texture: &Texture) {
        if self.unviewable.is_none() {
            self.unviewable = Some(texture.clone());
        }
    }

    /// The sampler state at ID `id` is past a set's sampler arrays of `count` elements.
    fn sampler_past(&mut self, id: u64, count: u64) {
        self.unlisted
            .get_or_insert_with(|| unlisted_notice((id, count), "a layout's commands"));
    }

    /// The Vulkan sampler of the sampler state at ID `id` for a set's sampler array was not
    /// made: `error`.
    fn sampler_unmade(&mut self, id: u64, error: &Error) {
        self.unlisted.get_or_insert_with(|| {
            (
                format!("sampler state at resource ID {id}"),
                format!(
                    "its sampler for a set-1 sampler array was not made ({error}), so the array \
                     holds the neutral sampler at its element, which a function reading the \
                     state through an argument buffer samples through"
                ),
            )
        });
    }

    /// The sampler state at ID `id` lies in neither window of the sampler tables, whose high one
    /// starts at `base`.
    fn sampler_outside(&mut self, id: u64, base: u64) {
        self.unlisted.get_or_insert_with(|| {
            (
                format!("sampler state at resource ID {id}"),
                format!(
                    "the device's sampler tables map sampler IDs 0 to {} and {base} to {}, \
                     which the ID is outside, so a function reading it through an argument \
                     buffer reads the neutral sampler",
                    SAMPLER_WINDOW - 1,
                    base + SAMPLER_WINDOW - 1
                ),
            )
        });
    }

    /// The cubic sampler state at ID `id` is live where a command's stage does not test the
    /// states it reaches for a cubic filter.
    fn sampler_cubic(&mut self, id: u64) {
        self.unlisted.get_or_insert_with(|| {
            (
                format!("sampler state at resource ID {id}"),
                "a cubic state, whose bicubic filter a command's stage that does not test for \
                 it (a fragment stage never does) cannot sample through, so a function of that \
                 stage reading it through an argument buffer reads the neutral sampler"
                    .to_owned(),
            )
        });
    }

    /// `texture` placed, into a set where `in_set`.
    fn placed(&mut self, texture: &Texture, in_set: bool) {
        let held = || super::pvrtc::held_for_copies(texture).is_some();
        if in_set && self.unreadable.is_none() && held() {
            self.unreadable = Some(texture.clone());
        }
    }

    /// Announce what was noted.
    fn announce(self, device: &Device) {
        if let Some(texture) = &self.unviewable {
            device.degrade(Degradation::UnnamedTextureUnviewable, || {
                (
                    format!(
                        "{:?} texture at resource ID {}",
                        texture.pixel_format(),
                        texture.gpu_resource_id().0
                    ),
                    "a heap array whose class the texture fits cannot view its format (one the \
                     device cannot sample or store without a format, or that needs its \
                     function's decode)"
                        .to_owned(),
                )
            });
        }
        if let Some(id) = self.shadowed {
            device.degrade(Degradation::UnnamedShadowedTexture, || {
                (
                    format!("texture at resource ID {id}"),
                    "the texture's texels are a buffer's, which a shadow image holds only as \
                     the encoders that name it copy them in"
                        .to_owned(),
                )
            });
        }
        if let Some((id, count)) = self.past {
            device.degrade(Degradation::UnnamedTexturesPastCapacity, || {
                (
                    format!("texture at resource ID {id}"),
                    format!(
                        "a pipeline's set-1 heap arrays hold {count} elements, which the ID is past"
                    ),
                )
            });
        }
        if let Some(elements) = self.copied {
            device.degrade(Degradation::HeapSetCopied, || {
                (
                    "a set-1 heap set".to_owned(),
                    format!(
                        "the device lacks descriptorBindingUpdateUnusedWhilePending, so each \
                         command buffer binds a copy of its own, written as it first binds the \
                         set ({elements} elements here) and not after it commits"
                    ),
                )
            });
        }
        if let Some(texture) = &self.unreadable {
            super::pvrtc::degrade_unreadable(device, [texture], "a heap set");
        }
        if let Some(unlisted) = self.unlisted {
            device.degrade(Degradation::UnlistedArgumentSampler, || unlisted);
        }
        if let Some(read) = &self.twin_read {
            read.announce(device);
        }
        if let Some(read) = &self.twin_placed {
            read.announce_once(device);
        }
        if let Some(read) = &self.raw_read {
            read.announce(device);
        }
        if let Some(read) = &self.raw_placed {
            read.announce_once(device);
        }
    }
}

/// A heap set's `VkDescriptorSetLayout`, which its versions hold.
struct RawLayout {
    device: Device,
    raw: vk::DescriptorSetLayout,
}

impl RawLayout {
    /// The layout of `layout`, a set-1 layout (no immutable sampler).
    fn create(device: &Device, layout: &DescriptorSetLayout) -> Result<RawLayout, Error> {
        if layout
            .bindings
            .iter()
            .any(|b| b.immutable_sampler.is_some())
        {
            return Err(internal("a heap set with an immutable sampler"));
        }
        let raw = super::pipeline::create_set_layout(device, layout, (&[], &[]), false)?;
        Ok(RawLayout {
            device: device.clone(),
            raw,
        })
    }
}

impl Drop for RawLayout {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: every set of the layout went (each version holds it).
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// One version of a heap set: a pool and the one set allocated from it (the module docs).
pub(crate) struct SetVersion {
    device: Device,
    _layout: Arc<RawLayout>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Its heap set's key.
    key: u64,
    /// A command buffer that binds it committed (set under the device's `heaps` lock): nothing
    /// writes a command buffer's own version any more (the shared one's writers ignore it), but
    /// a gone texture's neutrals until it is submitted.
    committed: AtomicBool,
    /// A batch of a command buffer that binds it was submitted (set under the device's `heaps`
    /// lock, before the `vkQueueSubmit`): nothing writes it any more.
    submitted: AtomicBool,
}

impl SetVersion {
    /// A new pool of one set of `layout`, which takes `needs`, and the set, of heap set `key`.
    fn allocate(
        device: &Device,
        layout: &Arc<RawLayout>,
        needs: &SetNeeds,
        key: u64,
    ) -> Result<SetVersion, Error> {
        let raw = device.shared().vk();
        let sizes = needs.sizes(1);
        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(needs.flags)
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `info` is live for the call.
        let pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let layouts = [layout.raw];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool holds one set of the layout.
        let allocated = unsafe { raw.allocate_descriptor_sets(&allocate) }.map_err(vulkan);
        let set = match allocated.and_then(|sets| {
            sets.first()
                .copied()
                .ok_or_else(|| internal("the driver allocated no heap set"))
        }) {
            Ok(set) => set,
            Err(error) => {
                // SAFETY: nothing uses the pool.
                unsafe { release::object(raw, pool) };
                return Err(error);
            }
        };
        Ok(SetVersion {
            device: device.clone(),
            _layout: Arc::clone(layout),
            pool,
            set,
            key,
            committed: AtomicBool::new(false),
            submitted: AtomicBool::new(false),
        })
    }
}

impl Drop for SetVersion {
    fn drop(&mut self) {
        // As for `RawLayout`.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command buffer binds the set (each holds the version until it
        // completes); the set goes with its pool.
        unsafe { release::object(self.device.shared().vk(), self.pool) };
    }
}

/// A variant's hold on the heap set of its set-1 layout (`VariantObjects::heap_set`): the set
/// lives while a variant of the layout does, and its versions while a command buffer binds them.
pub(crate) struct HeapSet {
    device: Device,
    key: u64,
}

impl Drop for HeapSet {
    /// The last variant of its layout went: what its elements name (the sampler states' Vulkan
    /// samplers among them, which no command holds) waits for the command buffers that may read
    /// it (the module docs' **Retirement**).
    fn drop(&mut self) {
        let freed = {
            let mut heaps = lock(&self.device.shared().heaps);
            match heaps.sets.remove(&self.key) {
                Some(entry) => {
                    let ours = heaps
                        .by_layout
                        .get(&entry.layout)
                        .is_some_and(|(key, _)| *key == self.key);
                    if ours {
                        heaps.by_layout.remove(&entry.layout);
                    }
                    heaps.retirement.retire(Box::new(entry) as Keeper)
                }
                None => None,
            }
        };
        // What the set held goes with no lock held.
        drop(freed);
    }
}

/// The heap set of set-1 layout `layout` on `device`: the live one, else a new one holding every
/// live texture (the module docs).
pub(crate) fn heap_set(
    device: &Device,
    layout: &DescriptorSetLayout,
) -> Result<Arc<HeapSet>, Error> {
    let mut held = Vec::new();
    let mut held_samplers = Vec::new();
    let mut notes = Notes::default();
    let set = {
        let mut heaps = lock(&device.shared().heaps);
        match heaps
            .by_layout
            .get(layout)
            .and_then(|(_, set)| set.upgrade())
        {
            Some(set) => Ok(set),
            None => heaps.made_set(device, layout, (&mut held, &mut held_samplers), &mut notes),
        }
    };
    drop((held, held_samplers));
    notes.announce(device);
    set
}

/// Place `texture`, a program's texture that just took its ID's slot, into the table and every
/// heap set (the module docs). A failure to write it leaves the ID's entry the neutral's,
/// announced.
pub(super) fn place(device: &Device, texture: &Texture) {
    sync(device, Slots::Changed(texture.gpu_resource_id().0));
}

/// The texture slots a [`sync`] writes.
enum Slots {
    /// One, where its holder changed since it was written.
    Changed(u64),
    /// Every live texture's.
    All,
    /// Those of the live textures whose images and texel views lie in the bytes of the buffer of
    /// a key (`Texture::bytes_key`).
    Over(ResourceKey),
}

/// Write `slots`: a changed one where its holder changed, the others whatever changed. While a
/// batch holds the slots' writes ([`hold`]), they wait for it.
fn sync(device: &Device, slots: Slots) {
    let mut held = Vec::new();
    let mut notes = Notes::default();
    let failed = {
        let mut heaps = lock(&device.shared().heaps);
        match heaps.holds {
            0 => heaps.write_slots(device, slots, &mut held, &mut notes),
            _ => {
                heaps.deferred.push_back(slots);
                device.shared().counts.heap_slot_writes_deferred();
                Vec::new()
            }
        }
    };
    drop(held);
    notes.announce(device);
    for (id, error) in failed {
        failed_write(device, id, &error);
    }
}

/// A batch's hold of the device's texture slots' writes ([`hold`]), from before its first work's
/// take until its submission was made or the work given back (`texture/first_move.rs`'s
/// `Taken`). The writes asked meanwhile ([`place`], [`renew_over`], [`renew`]) are made as the
/// last hold ends, before the batch lets go of its place in the first work's order: so no
/// texture's elements reach the table or a heap set between the take and the `vkQueueSubmit`
/// (a texture registered after the take, whose move the batch did not take, would be in them in
/// `UNDEFINED`), and every later batch, which takes its place after, finds them written.
pub(super) struct HeldSlots {
    device: Device,
}

/// Hold `device`'s texture slots' writes for a batch taking its first work (the module docs): only
/// on a device whose heap sets take writes while a command buffer that binds them is pending, as
/// the writes made as it ends reach the sets the batch's commands bound, which it is by then
/// (`texture/first_move.rs` takes no hold elsewhere).
pub(super) fn hold(device: &Device) -> HeldSlots {
    lock(&device.shared().heaps).holds += 1;
    HeldSlots {
        device: device.clone(),
    }
}

impl Drop for HeldSlots {
    fn drop(&mut self) {
        let device = &self.device;
        let mut held = Vec::new();
        let mut notes = Notes::default();
        let failed = {
            let mut heaps = lock(&device.shared().heaps);
            heaps.holds -= 1;
            let mut failed = Vec::new();
            while heaps.holds == 0 {
                let Some(slots) = heaps.deferred.pop_front() else {
                    break;
                };
                failed.extend(heaps.write_slots(device, slots, &mut held, &mut notes));
            }
            failed
        };
        drop(held);
        notes.announce(device);
        for (id, error) in failed {
            failed_write(device, id, &error);
        }
    }
}

/// Announce that writing the heap elements of texture slot `id` failed with `error`: its entry
/// is the neutral's.
fn failed_write(device: &Device, id: u64, error: &Error) {
    device.degrade(Degradation::UnnamedTextureUnviewable, || {
        (
            format!("texture at resource ID {id}"),
            format!("writing its heap elements failed: {error}"),
        )
    });
}

/// Whether a command buffer that may read the heap sets is not yet complete: a ticket is out.
pub(super) fn tickets_out(device: &Device) -> bool {
    let heaps = lock(&device.shared().heaps);
    !heaps.retirement.out.is_empty() || !heaps.replaced.out.is_empty()
}

/// Write every live texture's elements again: a change of storage form swapped images in
/// (`texture/storage_form.rs`) while no command buffer that may read the elements is pending.
pub(super) fn renew(device: &Device) {
    sync(device, Slots::All);
}

/// Write again the elements of the live textures whose images and texel views lie in the bytes
/// of the buffer of `key` (`Texture::bytes_key`; a heap's own buffer for its textures and those
/// over its buffers), and no other's: a rebind or a detach of those bytes remade their images or
/// views, or made them present stand-ins or their own images again (`buffer_descriptor.rs`,
/// `detach.rs`); its caller commits no command that uses the bytes meanwhile, so no pending
/// command buffer reads the elements rewritten. Every other texture's images and views stay
/// what its elements were written with.
pub(super) fn renew_over(device: &Device, key: ResourceKey) {
    sync(device, Slots::Over(key));
}

/// A program's texture at `id`, of hold serial `serial`, went: its entry is the neutral's where
/// it held the slot, and `keeper` (what keeps its image) waits for the command buffers that may
/// read it (the module docs).
pub(super) fn gone(device: &Device, id: u64, serial: u64, keeper: Option<Keeper>) {
    let (freed, cleared) = {
        let mut heaps = lock(&device.shared().heaps);
        let held = heaps
            .textures
            .get(&id)
            .is_some_and(|placed| placed.serial == serial);
        let cleared = match held {
            true => heaps.clear(device, id),
            false => Ok(()),
        };
        (
            keeper.and_then(|keeper| heaps.retirement.retire(keeper)),
            cleared,
        )
    };
    drop(freed);
    if let Err(error) = cleared {
        failed_write(device, id, &error);
    }
}

/// Keep `kept`, what the sets' elements named until they were written again without it
/// (`renew_over`): the views and images of textures a detach or a reattach of their bytes replaced
/// (`detach.rs`, `buffer_descriptor.rs`). It waits for the command buffers committed before it,
/// which may read it (the module docs).
pub(super) fn retire(device: &Device, kept: Keeper) {
    let freed = lock(&device.shared().heaps).replaced.retire(kept);
    // It goes with no lock held.
    drop(freed);
}

/// A command buffer's heap: the versions its commands bound (the module docs), until it
/// completes.
#[derive(Default)]
pub(super) struct HeapUse {
    versions: Vec<Arc<SetVersion>>,
    /// Its first submission takes a ticket: its commands, or its host steps', may read the heap.
    wants_ticket: bool,
    /// Its ticket was taken.
    ticketed: bool,
}

impl HeapUse {
    /// Whether a command bound a heap set.
    pub(super) fn bound(&self) -> bool {
        !self.versions.is_empty()
    }

    /// A kept recording's run bound `versions` again ([`bound_again`]), as its commands' [`bind`]
    /// did: each held until the command buffer completes.
    pub(super) fn bind_again(&mut self, versions: Vec<Arc<SetVersion>>) {
        for version in versions {
            if !self.versions.iter().any(|v| Arc::ptr_eq(v, &version)) {
                self.versions.push(version);
            }
        }
    }

    /// As the command buffer commits, before it is submitted, where its commands or its host
    /// steps may read the heap: its ticket for what a rebind or a detach replaces, which it holds
    /// until it completes; its first submission takes its ticket for gone textures and sampler
    /// states ([`submit`]). Nothing writes its own versions from now on but a gone texture's or
    /// sampler state's neutrals (the module docs).
    ///
    /// [`submit`]: HeapUse::submit
    pub(super) fn commit(&mut self, device: &Device) -> Ticket {
        self.wants_ticket = true;
        let mut heaps = lock(&device.shared().heaps);
        for version in &self.versions {
            version.committed.store(true, Ordering::Release);
        }
        let epoch = heaps.replaced.take();
        drop(heaps);
        Ticket {
            device: device.clone(),
            epoch,
            of: Retired::Replaced,
        }
    }

    /// As a batch of the command buffer is submitted, before its `vkQueueSubmit`: its versions
    /// are submitted (nothing writes them any more), and the first such batch of one whose
    /// commit wanted it takes its ticket for gone textures, which it holds until it completes.
    pub(super) fn submit(&mut self, device: &Device) -> Option<Ticket> {
        let ticket = self.wants_ticket && !self.ticketed;
        if self.versions.is_empty() && !ticket {
            return None;
        }
        let mut heaps = lock(&device.shared().heaps);
        for version in &self.versions {
            version.submitted.store(true, Ordering::Release);
        }
        if !ticket {
            return None;
        }
        self.ticketed = true;
        let epoch = heaps.retirement.take();
        drop(heaps);
        Some(Ticket {
            device: device.clone(),
            epoch,
            of: Retired::Gone,
        })
    }

    /// The command buffer completed: the versions its commands bound come out, for the caller to
    /// let go of once its completed handlers ran (each version's last hold releases its
    /// descriptor pool).
    pub(super) fn clear(&mut self) -> Vec<Arc<SetVersion>> {
        self.wants_ticket = false;
        self.ticketed = false;
        std::mem::take(&mut self.versions)
    }
}

/// A command buffer's hold on what it may read through the heap (the module docs'
/// **Retirement**), given back as it completes.
pub(super) struct Ticket {
    device: Device,
    epoch: u64,
    of: Retired,
}

/// Which retirement a [`Ticket`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Retired {
    /// Gone textures' images: taken as the command buffer's first batch is submitted.
    Gone,
    /// What a rebind or a detach replaced: taken as it commits.
    Replaced,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let freed = {
            let mut heaps = lock(&self.device.shared().heaps);
            match self.of {
                Retired::Gone => heaps.retirement.give(self.epoch),
                Retired::Replaced => heaps.replaced.give(self.epoch),
            }
        };
        // The images go with no lock held.
        drop(freed);
    }
}

/// Where a command binds the device's texture table: its layout's texture handle tables, and
/// the table's entries its heap arrays reach ([`texture_table`]).
pub(super) struct TextureTable<'a> {
    pub(super) info: vk::DescriptorBufferInfo,
    /// The `(set, binding)` of each texture handle table.
    pub(super) at: Cow<'a, [(u32, u32)]>,
}

/// The texture table a command of `layout` binds at its texture handle tables, over the entries
/// its heap arrays' count reaches: `None` for a layout without one. A variant's commands take it
/// from what their variant worked out once ([`HeapTables`]).
pub(super) fn texture_table(
    device: &Device,
    layout: &PipelineLayout,
) -> Result<Option<TextureTable<'static>>, Error> {
    device.shared().counts.heap_tables_worked_out();
    let Some(at) = TextureTableAt::of(layout) else {
        return Ok(None);
    };
    let table = at.table(device)?;
    Ok(Some(TextureTable {
        info: table.info,
        at: Cow::Owned(at.at),
    }))
}

/// Where a layout binds the device's texture table (its texture handle tables) and over how many
/// entries, worked out from the layout alone; and the table's descriptor once read, which stays:
/// the device makes its table once.
pub(super) struct TextureTableAt {
    at: Vec<(u32, u32)>,
    entries: u32,
    info: OnceLock<vk::DescriptorBufferInfo>,
}

impl TextureTableAt {
    /// `layout`'s, `None` for a layout without a texture handle table.
    fn of(layout: &PipelineLayout) -> Option<TextureTableAt> {
        let at: Vec<(u32, u32)> = layout
            .sets
            .iter()
            .flat_map(|set| set.bindings.iter().map(move |b| (set.set, b)))
            .filter(|(_, b)| {
                matches!(
                    b.source,
                    BindingSource::Synthetic {
                        binding: SyntheticBinding::HandleTable,
                        ..
                    }
                )
            })
            .map(|(set, b)| (set, b.binding))
            .collect();
        if at.is_empty() {
            return None;
        }
        // A table without heap arrays (a function that only tests handles for null) reaches
        // every entry.
        let capacity = layout
            .sets
            .get(HEAP_SET as usize)
            .and_then(|set| set.bindings.iter().map(|b| b.count).min())
            .unwrap_or(TABLE_ENTRIES);
        Some(TextureTableAt {
            at,
            entries: capacity.min(TABLE_ENTRIES),
            info: OnceLock::new(),
        })
    }

    /// The table a command binds at them, the device's, a texture placed at an ID past the
    /// entries announced (`Degradation::UnnamedTexturesPastCapacity`): the highest ID placed read
    /// without the `heaps` lock (`DeviceShared::heap_textures_end`), and the table's descriptor
    /// under it only until read once.
    fn table(&self, device: &Device) -> Result<TextureTable<'_>, Error> {
        let entries = self.entries;
        let end = (device.shared().heap_textures_end).load(Ordering::Acquire);
        let past = end.checked_sub(1).filter(|&id| id >= u64::from(entries));
        let info = match self.info.get() {
            Some(info) => *info,
            None => {
                let info = lock(&device.shared().heaps).table(device)?.entries(entries);
                *self.info.get_or_init(|| info)
            }
        };
        if let Some(id) = past {
            Notes {
                past: Some((id, entries)),
                ..Notes::default()
            }
            .announce(device);
        }
        Ok(TextureTable {
            info,
            at: Cow::Borrowed(&self.at),
        })
    }
}

/// Where a command binds a device sampler table: its layout's sampler tables, and the entries of
/// the table of the kind they read ([`sampler_table`]).
pub(super) struct SamplerTable<'a> {
    pub(super) info: vk::DescriptorBufferInfo,
    /// The `(set, binding)` of each sampler table.
    pub(super) at: Cow<'a, [(u32, u32)]>,
}

/// The sampler table a command of `layout`, of a pipeline stating `dependencies`, binds at its
/// sampler tables, every entry of it: the table of the element count of its set-1 sampler arrays
/// (every element where it has none), mapping cubic states where each stage with a sampler table
/// tests the states it reaches for a cubic filter (`RuntimeFact::CubicArgumentSampler`). `None`
/// for a layout without one. A variant's commands take it from what their variant worked out once
/// ([`HeapTables`]).
pub(super) fn sampler_table(
    device: &Device,
    layout: &PipelineLayout,
    dependencies: &[RuntimeDependency],
) -> Result<Option<SamplerTable<'static>>, Error> {
    device.shared().counts.heap_tables_worked_out();
    let Some(at) = SamplerTableAt::of(layout, dependencies) else {
        return Ok(None);
    };
    let table = at.table(device)?;
    Ok(Some(SamplerTable {
        info: table.info,
        at: Cow::Owned(at.at),
    }))
}

/// Where a layout binds the device's sampler table (its sampler tables) and of which kind, worked
/// out from the layout and its pipeline's dependencies alone; and the table's descriptor once
/// read, which stays: the device makes its table of a kind once.
pub(super) struct SamplerTableAt {
    at: Vec<(u32, u32)>,
    kind: SamplerTableKind,
    info: OnceLock<vk::DescriptorBufferInfo>,
}

impl SamplerTableAt {
    /// `layout`'s, of a pipeline stating `dependencies`; `None` for a layout without a sampler
    /// table.
    fn of(layout: &PipelineLayout, dependencies: &[RuntimeDependency]) -> Option<SamplerTableAt> {
        let tables: Vec<(Stage, u32, u32)> = layout
            .sets
            .iter()
            .flat_map(|set| set.bindings.iter().map(move |b| (set.set, b)))
            .filter_map(|(set, b)| match b.source {
                BindingSource::Synthetic {
                    stage,
                    binding: SyntheticBinding::SamplerTable,
                } => Some((stage, set, b.binding)),
                _ => None,
            })
            .collect();
        if tables.is_empty() {
            return None;
        }
        let capacity = layout
            .sets
            .get(HEAP_SET as usize)
            .and_then(|set| {
                set.bindings
                    .iter()
                    .filter(|b| {
                        matches!(b.source, BindingSource::Heap(HandleClass::Sampler { .. }))
                    })
                    .map(|b| b.count)
                    .min()
            })
            .unwrap_or(TABLE_ENTRIES);
        let cubic = tables.iter().all(|&(stage, ..)| {
            dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::CubicArgumentSampler { stage })
        });
        let at = tables
            .into_iter()
            .map(|(_, set, binding)| (set, binding))
            .collect();
        Some(SamplerTableAt {
            at,
            kind: SamplerTableKind { capacity, cubic },
            info: OnceLock::new(),
        })
    }

    /// The table a command binds at them, the device's of their kind: its descriptor under the
    /// `heaps` lock only until read once.
    fn table(&self, device: &Device) -> Result<SamplerTable<'_>, Error> {
        let info = match self.info.get() {
            Some(info) => *info,
            None => {
                let info = (lock(&device.shared().heaps).sampler_table(device, self.kind)?)
                    .entries(TABLE_ENTRIES);
                *self.info.get_or_init(|| info)
            }
        };
        Ok(SamplerTable {
            info,
            at: Cow::Borrowed(&self.at),
        })
    }
}

/// What a variant's layout binds of the device's tables ([`TextureTableAt`],
/// [`SamplerTableAt`]), worked out at its first command (`VariantObjects::heap_tables`): its
/// commands then read the layout no more, and take the `heaps` lock in [`bind`] alone.
pub(crate) struct HeapTables {
    texture: Option<TextureTableAt>,
    sampler: Option<SamplerTableAt>,
}

impl HeapTables {
    /// `layout`'s, of a pipeline stating `dependencies`, of `device`.
    pub(super) fn of(
        device: &Device,
        layout: &PipelineLayout,
        dependencies: &[RuntimeDependency],
    ) -> HeapTables {
        device.shared().counts.heap_tables_worked_out();
        HeapTables {
            texture: TextureTableAt::of(layout),
            sampler: SamplerTableAt::of(layout, dependencies),
        }
    }

    /// The texture table a command binds ([`texture_table`]).
    pub(super) fn texture_table(&self, device: &Device) -> Result<Option<TextureTable<'_>>, Error> {
        self.texture.as_ref().map(|at| at.table(device)).transpose()
    }

    /// The sampler table a command binds ([`sampler_table`]).
    pub(super) fn sampler_table(&self, device: &Device) -> Result<Option<SamplerTable<'_>>, Error> {
        self.sampler.as_ref().map(|at| at.table(device)).transpose()
    }
}

/// The heap set a command of `objects` (a layout of two sets) binds, with `writes` (its set-1
/// writes) at their elements: the device's set of its layout, where every live texture is at its
/// ID (the module docs). Of `writes`, the function tables' are written where the set held
/// another there; the textures', the sampler states' and the neutrals' the set holds already. The
/// command buffer's `heap` holds the set's version until it completes. `uses` are the command's
/// (`binding::command_uses`): a shadowed texture the set holds a view of that they neither name nor
/// make resident is announced (`Degradation::UnnamedShadowedTexture`): the command's function may
/// read its shadow through its ID, which the command's copies do not bring up to date, where a
/// texture they name or make resident has its shadow copied in. With the set, the version a kept
/// recording of the command binds again, where one may ([`Bound`]).
pub(super) fn bind(
    device: &Device,
    objects: &VariantObjects,
    heap: &mut HeapUse,
    writes: Vec<Write>,
    uses: &[Use<ResourceKey>],
) -> Result<Bound, Error> {
    if objects.set_count() != HEAP_SET as usize + 1 {
        return Err(internal("a heap in a layout of other than two sets"));
    }
    if writes.iter().any(|write| write.set != HEAP_SET) {
        return Err(internal("a heap write to another set"));
    }
    let set = objects.heap_set()?;
    let mut notes = Notes::default();
    let (version, again) = {
        let mut heaps = lock(&device.shared().heaps);
        let entry = heaps
            .sets
            .get_mut(&set.key)
            .ok_or_else(|| internal("a variant's heap set gone"))?;
        // A shadowed texture the command neither names nor makes resident (`useResource:`, a
        // texture of a heap it uses, the command buffer's residency): its function may read the
        // shadow through the ID, which the command's copies leave as the last encoder that named
        // it copied it (Metal's own non-resident read is undefined). One its uses name has its
        // shadow copied in (`shadow::shadow_uses`, compute's and render's alike).
        notes.shadowed = entry
            .shadows
            .iter()
            .find(|(_, key)| !uses.iter().any(|used| used.key == **key))
            .map(|(&id, _)| id);
        // A decoded twin whose texels are not Metal's decode exactly, which the command's
        // function may read through its ID whether or not the command names it.
        notes.twin_read = entry.twins.values().next().cloned();
        // A texture held by its raw texels alone, which the function may read through its ID
        // and its own decode.
        notes.raw_read = entry.raws.values().next().cloned();
        // The shared version, or the command buffer's own, made as it first binds the set.
        let own = heap.versions.iter().find(|v| v.key == set.key).cloned();
        let version = match (&entry.shared, own) {
            (Some(shared), _) => Arc::clone(shared),
            (None, Some(own)) => own,
            (None, None) => entry.own_version(device, &mut notes)?,
        };
        let held = (writes.iter()).all(|write| holds_element(write.ty, write.element));
        let again = match &entry.shared {
            None => Err(Unkept::Unshared),
            Some(_) if !held => Err(Unkept::Writes),
            Some(_) if !entry.notes_nothing() => Err(Unkept::Noted),
            Some(shared) => Ok(Arc::downgrade(shared)),
        };
        let staged: Vec<HeapWrite> = writes
            .into_iter()
            .filter(|write| {
                !holds_element(write.ty, write.element)
                    && !entry
                        .written
                        .get(&(write.binding, write.element))
                        .is_some_and(|written| same(&written.info, &write.info))
            })
            .map(|write| HeapWrite {
                binding: write.binding,
                element: write.element,
                ty: write.ty,
                info: write.info,
                keeper: write.held,
            })
            .collect();
        entry.write(device, staged)?;
        (version, again)
    };
    notes.announce(device);
    let raw = version.set;
    if !heap.versions.iter().any(|v| Arc::ptr_eq(v, &version)) {
        heap.versions.push(version);
    }
    Ok(Bound { set: raw, again })
}

/// What a command's [`bind`] bound: the heap set, and the version a kept recording of the command
/// binds again in a later command buffer (`kept_commands::KeptEffects`, [`bound_again`]), else why
/// it may not.
pub(super) struct Bound {
    pub(super) set: vk::DescriptorSet,
    pub(super) again: Result<Weak<SetVersion>, Unkept>,
}

/// Why a command that binds a heap set may not run again from a kept recording in a later command
/// buffer (`render/executions/kept.rs`): what a kept recording's run would not do as the
/// command's [`bind`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unkept {
    /// The device lacks `descriptorBindingUpdateUnusedWhilePending`: each command buffer binds a
    /// version of its own (the module docs), which no other command buffer may bind.
    Unshared,
    /// It writes elements of its own into the set (a function table's), which a later command of
    /// another table may have written over.
    Writes,
    /// The set holds a shadowed texture, a decoded twin or a raw read, which a command's binding
    /// announces (`Degradation::UnnamedShadowedTexture`, `decoded-texel-precision`).
    Noted,
    /// The command reads the device's sampler tables ([`sampler_table`]): no draw over them is
    /// kept (`docs/FAST-PATHS.md`, known gaps).
    SamplerTable,
    /// Its tables bind a texture at a slot its heap part reads (`render/heap_part.rs`), which no
    /// kept key holds: a host step's tables bind none.
    TableTextures,
}

/// The versions `kept` that a kept recording's commands bound ([`Bound`]), where a later command
/// buffer's run of the recording binds them again as its commands' [`bind`] would: each lives, is
/// still its set's one version, and its set announces nothing a binding would ([`Unkept::Noted`]).
/// `None` where one is not. Taken under the `heaps` lock once a run, and dropped with none held.
pub(super) fn bound_again(
    device: &Device,
    kept: &[Weak<SetVersion>],
) -> Option<Vec<Arc<SetVersion>>> {
    if kept.is_empty() {
        return Some(Vec::new());
    }
    let versions: Vec<Arc<SetVersion>> = kept.iter().map(Weak::upgrade).collect::<Option<_>>()?;
    let binds = {
        let heaps = lock(&device.shared().heaps);
        versions.iter().all(|version| {
            heaps.sets.get(&version.key).is_some_and(|entry| {
                let shared = entry.shared.as_ref();
                shared.is_some_and(|shared| Arc::ptr_eq(shared, version)) && entry.notes_nothing()
            })
        })
    };
    binds.then_some(versions)
}

/// Whether every heap set holds what a command's write of type `ty` at `element` of one of its
/// arrays would write, so the command writes nothing there ([`bind`]; `binding::bind_into` makes
/// no such write): the neutral element, and every image element, which holds the texture at its
/// ID as the array's class takes it, written as the texture takes the slot (the module docs).
pub(super) fn holds_element(ty: vk::DescriptorType, element: u32) -> bool {
    element == metal2vulkan::bind::NEUTRAL_HEAP_ELEMENT
        || matches!(
            ty,
            vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE
        )
}

/// Whether two descriptors name the same objects (each held while a set holds it, so a handle
/// is never another object's meanwhile).
fn same(a: &Info, b: &Info) -> bool {
    match (a, b) {
        (Info::Buffer(a), Info::Buffer(b)) => {
            (a.buffer, a.offset, a.range) == (b.buffer, b.offset, b.range)
        }
        (Info::Image(a), Info::Image(b)) => {
            (a.sampler, a.image_view, a.image_layout) == (b.sampler, b.image_view, b.image_layout)
        }
        (Info::Texel(a), Info::Texel(b)) => a == b,
        _ => false,
    }
}

/// `samplers` as the bind state's argument samplers, each at its element of the sampler arrays
/// (`resource_ids.rs`; one past every array's capacity at `u32::MAX`, which [`fit_samplers`]
/// leaves out).
pub(super) fn listed(samplers: &[SamplerState]) -> Vec<ArgumentSampler> {
    samplers.iter().map(argument_sampler).collect()
}

/// `sampler` as one of the bind state's argument samplers ([`listed`]).
fn argument_sampler(sampler: &SamplerState) -> ArgumentSampler {
    let element = u32::try_from(sampler.heap_element()).unwrap_or(u32::MAX);
    ArgumentSampler::new(*sampler.descriptor(), sampler.gpu_resource_id().0).at_element(element)
}

/// Whether `kept` is `samplers` [`listed`], bit for bit (a float of the other sign, which
/// compares equal, is another listing): what a kept bind state need not take again.
pub(super) fn lists(kept: &[ArgumentSampler], samplers: &[SamplerState]) -> bool {
    kept.len() == samplers.len()
        && (kept.iter().zip(samplers))
            .all(|(kept, sampler)| same_sampler(kept, &argument_sampler(sampler)))
}

/// Whether two listings of argument samplers are one, bit for bit ([`lists`]).
pub(super) fn same_listing(a: &[ArgumentSampler], b: &[ArgumentSampler]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_sampler(a, b))
}

/// Whether two argument samplers are one, bit for bit: a float of the other sign, which compares
/// equal, is another.
fn same_sampler(a: &ArgumentSampler, b: &ArgumentSampler) -> bool {
    let bits = |s: &ArgumentSampler| [s.desc.lod_min_clamp, s.desc.lod_max_clamp].map(f32::to_bits);
    a == b && bits(a) == bits(b)
}

/// Where `plan`'s heap writes go for `state`, each heap array holding the capacity of set 1 of
/// `layout`: `None` for a layout without a heap.
pub(super) fn heap_elements(
    layout: &PipelineLayout,
    plan: &BindingPlan,
    state: &BindState,
) -> Result<Option<HeapElements>, Error> {
    let Some(capacity) = capacity(layout) else {
        return Ok(None);
    };
    plan.heap_elements(state, capacity)
        .map(Some)
        .map_err(Error::Bind)
}

/// The capacity of each heap array of set 1 of `layout`, `None` for a layout without a heap.
pub(super) fn capacity(layout: &PipelineLayout) -> Option<u32> {
    let set = layout.sets.get(1)?;
    Some(set.bindings.iter().map(|b| b.count).min().unwrap_or(0))
}

/// Whether every one of `samplers` (the argument samplers a command lists) has an element in
/// the heap set `set` of the command's variant (`None`: a layout without one, which reads none):
/// each heap array has the variant's heap capacity of elements, and a state's element is the
/// lowest no other live state held as it was made (`resource_ids.rs`), so one is past them only
/// where more states live than they hold. Where some have none, `samplers` keeps those that do
/// and the caller plans again with them; `unlisted` then takes the first left out's resource ID
/// and the capacity, where the set has sampler arrays (nothing reads them where it has none),
/// for [`degrade_unlisted`].
pub(super) fn fit_samplers(
    set: Option<&DescriptorSetLayout>,
    samplers: &mut Vec<SamplerState>,
    unlisted: &mut Option<(u64, u64)>,
) -> bool {
    let Some(set) = set else {
        return true;
    };
    let capacity = u64::from(set.bindings.iter().map(|b| b.count).min().unwrap_or(0));
    let fits = |sampler: &SamplerState| sampler.heap_element() < capacity;
    if samplers.iter().all(fits) {
        return true;
    }
    if set
        .bindings
        .iter()
        .any(|b| b.ty == metal2vulkan::vk::DescriptorType::SAMPLER)
    {
        let past = samplers.iter().find(|sampler| !fits(sampler));
        *unlisted = past.map(|sampler| (sampler.gpu_resource_id().0, capacity));
    }
    samplers.retain(fits);
    false
}

/// Announce that `commands` (the commands of a function, named) leave the sampler state at
/// resource ID `id` out of their argument samplers, their variant's set-1 sampler arrays holding
/// `capacity` elements ([`fit_samplers`]): a function of theirs reading it reads the neutral
/// sampler (their sampler table maps its ID there, [`SamplerTableKind`]).
pub(super) fn degrade_unlisted(device: &Device, unlisted: (u64, u64), commands: &str) {
    device.degrade(Degradation::UnlistedArgumentSampler, || {
        unlisted_notice(unlisted, commands)
    });
}

/// The object and the reason of [`degrade_unlisted`]'s notice.
pub(super) fn unlisted_notice((id, capacity): (u64, u64), commands: &str) -> (String, String) {
    (
        format!("sampler state at resource ID {id}"),
        format!(
            "{commands}' set-1 sampler arrays hold {capacity} elements, fewer than the live \
             sampler states, so they leave the state out of their argument samplers, and a \
             function of theirs reading it reads the neutral sampler"
        ),
    )
}
