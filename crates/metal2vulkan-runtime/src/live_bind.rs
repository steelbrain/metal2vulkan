//! A render encoder's bind state, kept between its draws: each draw brings the state the encoder's
//! last draw bound to its own in place, and makes its bind shape again only where a change may have
//! changed it.
//!
//! - **Brought in place.** Each slot's binding is derived again from the encoder's tables and
//!   compared with the kept one (`ArgumentTable::refresh_reading`), at each draw whose table may
//!   bring in another: a stage's table read last through the same interface and samplers' maps,
//!   that no call changed since (its stamp, `argument_table::Stamp`), whose slots that tell the
//!   device's view base alignment apart its drawing state still holds as they were read
//!   (`argument_table::AtBindViews`, which another table's offsets may shrink), and that no rebind
//!   or detach of a no-copy buffer may have changed (which changes a bound buffer's binding with
//!   no call: `buffer::host_cells_moved`), is not read again ([`ReadFrom`]; checked whole in debug
//!   builds). Where only calls changed it since it was so read, and the table keeps the slots
//!   they changed (`ArgumentTable::refresh_changed`), only those slots are brought in: every
//!   other slot's binding is a function of what the table bound under the stamp read, the
//!   interface, the maps and the moves, none of which changed (checked whole in debug builds).
//!   Every other field is computed as before and compared ([`Reshape::field`]), as much of it as
//!   the shape reads ([`Reshape::shaped`]; a value no shape reads, such as the depth/stencil state,
//!   the stencil references or the blend color, is stored uncompared). The new value is always
//!   stored, so the state is the draw's exactly, a float of the other sign (which compares equal)
//!   included. While the same slots stay bound, nothing is allocated.
//! - **The shape found again only where it may have changed.** A bind shape is each bound slot's
//!   shape beside the shapes of the state's other fields (`BindState::shape`: a stage's tables
//!   are its slots' shapes, and a stage with none is left out). Where every changed slot keeps its
//!   shape, no slot was bound or unbound, and no other field's shape changed, the shape is the
//!   kept one: a function of those alone, so a draw that changes only values the shape does not
//!   read (a depth/stencil state, as draws between which the program sets another) finds it kept.
//!   After any other change, it is one of the shapes the state took lately while it listed the
//!   used resources it lists now, where it is among them ([`RecentShapes`]): their used
//!   resources' shape is the state's, so the rest of each is compared alone
//!   (`BindState::has_shape_beside_used`), and draws whose calls alternate a field the shape
//!   reads (a draw's primitive type picks its viewport compensation and its band) find theirs
//!   without listing the resources made resident. Else, and at an encoder's first draw, it is
//!   one the drawing state settled lately where the state has it (`BindState::position_of_shape`,
//!   compared in place: `GraphicsState::kept_shape`), so a pass that draws as the one before it
//!   makes none; else it is made again, whole.
//! - **One per pipeline state lately drawn.** Beside the state its last draw brought in place, the
//!   encoder sets aside the states of up to [`SET_ASIDE`] pipeline states it drew before that one
//!   ([`LiveBind::select`]), each under the pipeline state whose draw left it, and a draw starts
//!   from its pipeline state's, where one is set aside, though another state's draws share its
//!   interface (states of one function pair, blending apart): draws that switch between pipeline
//!   states each find their own state's shape kept, their stage tables read through their own
//!   interface, where no slot of theirs changed its shape, and their own state's last draw kept
//!   (`render/draw_again.rs`). Where none is, it starts from one no live state's draw left, else
//!   from the latest read through its interface, else from the one set aside longest.
//!   Exact for the reason a draw's is: whichever state a draw starts from, it brings every field
//!   in place, and the starting state decides only whether the shape is kept.
//! - **The settle kept with the state.** Beside each state, the variant and plan its last recorded
//!   draw settled ([`SettledHere`]), with its program and buffer forms: a draw of the same program
//!   in the same forms whose shape is the kept one (the very `Arc`) settles as it did, taking no
//!   lock of its program's (`GraphicsState::settle_shape`, whose answer is a function of the
//!   shape and forms alone). Only a draw that recorded keeps it, so its encoder holds its pipeline
//!   state (and the command buffer the variant's objects); it goes as the encoder ends
//!   ([`SpareLiveBinds::give`]), so a spare state holds nothing alive, but where the encoder
//!   carries its kept draws into its command buffer's next render encoder, which takes the state
//!   with its settles and the kept draws set aside that carry ([`LiveBind::carry_aside`],
//!   `render/carried.rs`, [`LiveBind::end_carried`]).
//! - **Kept from one encoder to the next.** An encoder that ends gives its state, with its shape,
//!   to its device ([`SpareLiveBinds`]), and the next render encoder opened starts from it, as a
//!   draw starts from the draw before it: its first draw brings it in place, so a pass that binds
//!   as the one before it finds its shape kept and adds no slot. The shape it keeps may be one
//!   another pipeline state's draws made (an encoder of another thread's, say): a pipeline
//!   state's settle of an equal shape answers the encoder's very one
//!   (`GraphicsState::settle_shape`), so its draws find their settle kept from its first draw
//!   on. Exact for the reason a draw's is:
//!   every field of the state a render encoder's draw binds is derived again at each draw (its
//!   slots, by `ArgumentTable::refresh_reading`, which unbinds every slot the encoder's tables do
//!   not bind, and every slot the pipeline state does not read (`argument_table::ReadSlots`);
//!   every other field, by `RenderCommandEncoder::settle_draw` and
//!   `refresh_bind_state`), and none other is ever set: a render encoder's state holds no compute
//!   or tile field (`stage_in_region`, `kept_imageblock_bytes`, `resident_ranges`, a stage's
//!   acceleration structures), which a tile dispatch binds in a state of its own. The state names
//!   no object (its bindings are values, and its tables' drawing state is held weakly), so a kept
//!   one holds nothing alive; the next encoder's tables are new, with stamps of their own.
//! - **The draws' storage kept with it.** The vectors a draw fills and drops (its commands, push
//!   writes, uses, descriptor writes, pushed set 0 and vertex buffers, and a draw rebound's slot
//!   rebinds, which outlast the settle they were made for) are kept here too, each draw's written
//!   over the last one's storage: of each state set aside, its own last draw's, with the draw
//!   its encoder kept of it, so a draw of that state after draws of others is rebound from it
//!   (`render/draw_again.rs`). Those hold values and raw handles, but for the
//!   descriptor writes, whose keepers hold samplers and function tables: those are emptied as
//!   the draw is done with them ([`Emptied`]), so they hold nothing past it, as a new vector's did.
//! - **The resident textures' shadows listed once per change.** The shadows' copies of the
//!   textures the encoder made resident (`shadow::ResidentShadows`) are kept here between draws
//!   and listed again only where the residency's generation, or the count of usage shadows made,
//!   moved, as a compute encoder keeps them (`compute/live.rs`): a draw's look at them costs
//!   their count, not the resident textures'. They hold the copies, so they go as the encoder
//!   ends ([`SpareLiveBinds::give`]).
//! - **The heap parts kept while what they are made of holds.** A draw's heap tables, where its
//!   heap writes go and its writes past the heap's entries (`render/heap_part.rs`) are kept here
//!   for the plans the encoder's draws drew through lately, each while the residency's
//!   generation, the argument samplers, the function tables and what its plan's heap slots bind
//!   hold: a draw that keeps its part costs what it binds, not the textures made resident. They
//!   go as the encoder ends ([`SpareLiveBinds::give`]).

use ash::vk;
use metal2vulkan::bind::{BindShape, BindState, DrawCommands, SlotRebinds};
use metal2vulkan::pipeline::{PipelineInterface, SlotRef};
use std::any::Any;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, Weak};

use super::address_map::SamplerMaps;
use super::argument_table::Stamp;
use super::binding::{CommandUses, PushWrite, SetWrite, Write};
use super::device::lock;
use super::noted_apart::NotedApart;
use super::pipeline::{BufferForms, GraphicsState, ProgramName, Settled, WeakRenderPipelineState};
use super::render::{DrawKept, KeptHeap, KeptStorage};
use super::sampler::OwnedSampler;
use super::shadow::ResidentShadows;
use super::work_counts::Counting;

/// The most bind states a device keeps for its render encoders to start from: one per encoder
/// open at once beyond it starts empty, as every encoder did before.
const SPARE_LIVE_BINDS: usize = 4;

/// The most states a [`LiveBind`] sets aside, of the pipeline states its draws drew through before
/// the last draw's ([`LiveBind::select`]): a draw of one of more pipeline states in turn than
/// this starts from one another state's draw left, read through its interface where one is.
const SET_ASIDE: usize = 7;

/// The most shapes a state keeps of those it took lately ([`RecentShapes`]): one per primitive
/// type a draw names (Metal's five, `MTLPrimitiveType`), the call's one input that varies from
/// draw to draw with nothing bound in between and that a shape reads (its viewport compensation,
/// its band, its provoking forms), so draws of one pipeline state that cycle through every type,
/// as an indirect command buffer's draws may, each find theirs. Draws that alternate more shapes
/// than this (a type in two index types, for a pipeline that writes the primitive ID) find the
/// one kept longest replaced, and take the drawing state's settled shapes as before.
const RECENT_SHAPES: usize = 5;

/// Whether the changes brought into a kept bind state may have changed its shape.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Reshape(bool);

impl Reshape {
    /// Store `value` at `index` of `map`, a slot table of a kept stage table: its shape may change
    /// where the index was unbound, or where the binding's `shape` changes.
    pub(super) fn slot<V: PartialEq, S: PartialEq>(
        &mut self,
        map: &mut BTreeMap<u32, V>,
        index: u32,
        value: V,
        shape: impl Fn(&V) -> S,
    ) {
        match map.entry(index) {
            Entry::Vacant(entry) => {
                entry.insert(value);
                self.0 = true;
            }
            Entry::Occupied(mut entry) => self.replace(entry.get_mut(), value, shape),
        }
    }

    /// Store `value` in `slot`, a bound slot of a kept stage table: its shape may change where the
    /// binding's `shape` changes.
    pub(super) fn replace<V: PartialEq, S: PartialEq>(
        &mut self,
        slot: &mut V,
        value: V,
        shape: impl Fn(&V) -> S,
    ) {
        self.replace_by(slot, value, |was, now| shape(was) == shape(now));
    }

    /// Store `value` in `slot` as [`replace`](Self::replace) does, where `same_shape` tells
    /// whether two bindings have one shape without making either's.
    pub(super) fn replace_by<V: PartialEq>(
        &mut self,
        slot: &mut V,
        value: V,
        same_shape: impl FnOnce(&V, &V) -> bool,
    ) {
        if !self.0 && *slot != value && !same_shape(&*slot, &value) {
            self.0 = true;
        }
        *slot = value;
    }

    /// A slot of a kept stage table unbound.
    pub(super) fn unbound(&mut self) {
        self.0 = true;
    }

    /// Whether a change it was told of may have changed the shape: a compute encoder's kept state
    /// then finds its shape again (`compute/live.rs`), and a draw rebound takes every step
    /// (`render/draw_again/rebound.rs`).
    pub(super) fn may_reshape(self) -> bool {
        self.0
    }

    /// Store `value` in `field`, a field of the kept state beside its stage tables: any change
    /// may change the shape.
    pub(super) fn field<T: PartialEq>(&mut self, field: &mut T, value: T) {
        if *field != value {
            self.0 = true;
        }
        *field = value;
    }

    /// [`field`](Self::field) of a value the caller holds on to: compared in place, and copied
    /// into `field` only where it differs, so a field held as it was copies nothing. For a type
    /// whose equality is its value's (`Eq`), so a field kept as equal holds what a copy would.
    pub(super) fn field_from<T: Eq + Clone>(&mut self, field: &mut T, value: &T) {
        if *field != *value {
            self.0 = true;
            field.clone_from(value);
        }
    }

    /// Store `value` in `field`, a field of the kept state of which the shape reads `shape` alone
    /// (`BindState::shape`): its shape may change only where that does.
    pub(super) fn shaped<T, S: PartialEq>(
        &mut self,
        field: &mut T,
        value: T,
        shape: impl Fn(&T) -> S,
    ) {
        if !self.0 && shape(field) != shape(&value) {
            self.0 = true;
        }
        *field = value;
    }
}

/// A render encoder's bind state as its last draw bound it, and its shape while no change may
/// have changed it (the module docs).
#[derive(Default)]
pub(super) struct LiveBind {
    /// The state. Each draw brings it to its own through [`Reshape`] (its values past its settle,
    /// which no shape reads, aside), and then hands the reshape to [`reshaped`](Self::reshaped).
    /// Boxed, so a draw of another pipeline state changes it for one set aside by its address
    /// ([`select`](Self::select)).
    pub(super) state: Box<BindState>,
    /// The state's shape, `None` where a change may have changed it.
    shape: Option<Arc<BindShape>>,
    /// The shapes the state took since it last listed other used resources ([`RecentShapes`]).
    pub(super) recent: RecentShapes,
    /// The count of the device's sampler IDs' slots' changes (`DeviceShared::sampler_slots_changed`)
    /// at which the state's argument samplers are every live sampler state: while the count
    /// holds, a draw lists them as they are, without taking the states again
    /// (`sampler::ArgumentSamplers`). `None` where they are not (none listed yet, or those a
    /// variant's set-1 arrays hold alone).
    pub(super) listed_at: Option<u64>,
    /// What the state's stage tables were read from ([`ReadFrom`]).
    pub(super) read_from: ReadFrom,
    /// The pipeline state whose draw last brought the state in place ([`select`](Self::select)),
    /// named without holding it; `None` before any did.
    drawn: Option<WeakRenderPipelineState>,
    /// The states the encoder's draws of other pipeline states brought in place, each as its
    /// pipeline state's last draw left it ([`select`](Self::select)).
    set_aside: [SetAside; SET_ASIDE],
    /// How many times a state went aside: each set aside holds the count as it went, so the one
    /// set aside longest holds the least.
    asides: u64,
    /// What the state's last recorded draw settled, while its encoder lasts ([`SettledHere`]):
    /// set by a draw that recorded, in place of what the state kept.
    pub(super) settled: Option<SettledHere>,
    /// The lists the state's settles make their plans' slot rebinds in
    /// ([`SettledHere::rebinds`]), kept with the state, so their room goes from one settle and
    /// one encoder to the next.
    pub(super) rebinds: SlotRebinds,
    /// The last draw its encoder noted apart (`noted_apart.rs`), kept here so its storage goes
    /// from one encoder to the next.
    pub(super) noted: NotedApart,
    /// The commands the last draw's plan instantiated, which the next draw's are written over
    /// (`BindingPlan::instantiate_draw_into`), kept here so their storage goes from one draw and
    /// one encoder to the next.
    pub(super) commands: DrawCommands,
    /// The last draw's push writes with its holes filled (`binding::push_writes`), kept likewise.
    pub(super) pushes: Vec<PushWrite>,
    /// The last draw's uses (`binding::command_uses_into`), kept likewise.
    pub(super) uses: CommandUses,
    /// The storage of the last draw's descriptor writes (`binding::bind`), kept likewise and
    /// empty between draws ([`Emptied`]).
    pub(super) writes: Vec<Write>,
    /// The last draw's pushed set 0 (`binding::SetWrite::push_into`), kept likewise.
    pub(super) pushed: Vec<SetWrite>,
    /// The dynamic offsets the last draw bound its set 0 with (`binding::dynamic_writes`), kept
    /// likewise.
    pub(super) offsets: Vec<u32>,
    /// The last draw's vertex buffers, `(binding, buffer, offset, stride)`, kept likewise.
    pub(super) vertex_buffers: Vec<(u32, vk::Buffer, u64, u64)>,
    /// Where the space each of those buffers is bound in starts in its `VkBuffer`
    /// (`Buffer::space`), in their order, kept likewise: a draw rebound binds a buffer whose
    /// offset alone moved since from there, without looking at where the buffer lies
    /// (`render/draw_again/rebound.rs`).
    pub(super) vertex_starts: Vec<u64>,
    /// The storage of the samplers the last draw's writes named (`binding::bind_into`), kept
    /// likewise and empty between draws: its encoder retains them as the draw records.
    pub(super) samplers: Vec<Arc<OwnedSampler>>,
    /// What a draw rebound takes of its encoder's kept draw beyond its objects, kept likewise
    /// (`render/draw_again/rebound.rs`).
    pub(super) kept: KeptStorage,
    /// The kept draw of the state's pipeline state, set aside here while its encoder draws
    /// through another (`render/draw_again.rs`, `RenderLowering::select_kept`): in place of the
    /// state, the encoder holds it.
    draw_kept: Option<DrawKept>,
    /// The shadows' copies of the textures its encoder made resident, listed once per change (the
    /// module docs).
    pub(super) shadows: ResidentShadows,
    /// The heap parts its draws worked out lately, each kept while what it is made of holds (the
    /// module docs).
    pub(super) heap: KeptHeap,
}

/// The shapes a state took since it last listed other used resources (`BindState::used_resources`),
/// the latest last, at most [`RECENT_SHAPES`]: each was the state's shape while it listed the used
/// resources it lists now, so the used resources' shape of each is the state's, and a state has
/// one of them as its shape where the rest of it matches (`BindState::has_shape_beside_used`).
/// Forgotten where the state lists other used resources (`RenderCommandEncoder::
/// refresh_bind_state`).
#[derive(Default)]
pub(super) struct RecentShapes(Vec<Arc<BindShape>>);

impl RecentShapes {
    /// The used resources the state lists changed: none of the shapes may be its shape.
    pub(super) fn forget(&mut self) {
        self.0.clear();
    }

    /// The one of them that is `state`'s shape, where one is.
    fn find(&self, state: &BindState) -> Option<Arc<BindShape>> {
        let mut latest_first = self.0.iter().rev();
        latest_first
            .find(|shape| state.has_shape_beside_used(shape))
            .cloned()
    }

    /// Keep `shape`, the state's now, in place of the one kept longest where they are full.
    fn keep(&mut self, shape: &Arc<BindShape>) {
        if let Some(at) = self.0.iter().position(|kept| Arc::ptr_eq(kept, shape)) {
            self.0.remove(at);
        } else if self.0.len() == RECENT_SHAPES {
            self.0.remove(0);
        }
        self.0.push(Arc::clone(shape));
    }
}

/// A bind state a [`LiveBind`] set aside, with its shape, its listing's count and what its stage
/// tables were read from, as [`LiveBind`] holds them for its last draw's state.
#[derive(Default)]
struct SetAside {
    state: Box<BindState>,
    shape: Option<Arc<BindShape>>,
    recent: RecentShapes,
    listed_at: Option<u64>,
    read_from: ReadFrom,
    drawn: Option<WeakRenderPipelineState>,
    settled: Option<SettledHere>,
    rebinds: SlotRebinds,
    /// Its last draw's storage and kept draw, as [`LiveBind`] holds them for its last draw's
    /// state: a draw rebound after draws of other states rebinds its own state's last draw.
    commands: DrawCommands,
    pushes: Vec<PushWrite>,
    uses: CommandUses,
    pushed: Vec<SetWrite>,
    offsets: Vec<u32>,
    vertex_buffers: Vec<(u32, vk::Buffer, u64, u64)>,
    vertex_starts: Vec<u64>,
    kept: KeptStorage,
    draw_kept: Option<DrawKept>,
    /// [`LiveBind::asides`] as it went aside.
    at: u64,
}

/// What a recorded draw settled of its state's shape (`GraphicsState::settle_shape`): the variant
/// and plan, the program it ran (named without being held) and the buffer forms it took; the shape
/// is the state's kept one, compared by identity. With them, whether the plan rebinds the slots
/// of the slot rebinds made in the state's lists ([`LiveBind::rebinds`]) for this settle; `None`
/// before any were made for it.
pub(super) struct SettledHere {
    program: ProgramName,
    forms: BufferForms,
    settled: Settled,
    made: Option<bool>,
}

impl SettledHere {
    /// What the draw settled.
    pub(super) fn settled(&self) -> &Settled {
        &self.settled
    }

    /// What a draw of `pipeline` in `forms` settled.
    pub(super) fn new(pipeline: &GraphicsState, forms: BufferForms, settled: Settled) -> Self {
        SettledHere {
            program: pipeline.program_name(),
            forms,
            settled,
            made: None,
        }
    }

    /// What the draw settled, and its plan's slot rebinds of `slots` (`None` where the plan reads
    /// one of them some other way), in `lists`, the state's ([`LiveBind::rebinds`]): made there
    /// at the first draw rebound that changed those slots (`BindingPlan::slot_rebinds_into`), in
    /// place of those made before for other slots or settles, and kept for each next one that
    /// changes the same slots, as the plan is the same.
    pub(super) fn rebinds<'a>(
        &'a mut self,
        slots: &[SlotRef],
        lists: &'a mut SlotRebinds,
    ) -> (&'a Settled, Option<&'a SlotRebinds>) {
        let made = match self.made {
            Some(made) if lists.slots() == slots => made,
            _ => {
                let made = (self.settled.plan).slot_rebinds_into(slots, lists);
                self.made = Some(made);
                made
            }
        };
        (&self.settled, made.then_some(lists))
    }
}

/// What a kept state's stage tables were read from at the draw that read them last
/// (`RenderCommandEncoder::refresh_bind_state`; the module docs): the encoder's table of each stage
/// by its stamp (`argument_table::Stamp`), the interface they were read through, the samplers'
/// address-mode maps, and the count of no-copy buffers' moves (`buffer::host_cells_moved`); and the
/// generation of the residency the state's used resources were made of (`Residency::generation`). A
/// stage's table that keeps its stamp, read through the same interface with the same maps while no
/// no-copy buffer moved, brings in what it brought in then where its drawing state's slots that
/// tell the view base alignment apart are still those it read under (`AtBindViews::holds`, checked
/// at each draw): a slot's binding is a function of the object bound there (its length, its base
/// and addresses, which only a move changes; a texture's facts, made with it; a sampler's
/// descriptor), of the table's other entries, of the function's reading of the slot (the
/// interface), of the maps, and of those slots.
#[derive(Default)]
pub(super) struct ReadFrom {
    /// Each stage's table's stamp, by the stage's place in the draw's reading order.
    tables: [Option<Stamp>; 4],
    /// The interface they were read through, by address, with the drawing state that holds it,
    /// held without keeping it alive: while that state lives, no other interface lies there.
    interface: Option<(usize, Weak<dyn Any + Send + Sync>)>,
    /// The maps they were read with.
    maps: SamplerMaps,
    /// The count of no-copy buffers' moves as they were read.
    moved: u64,
    /// The generation of the residency the state's used resources were made of, at `moved`
    /// moves; `None` before any were.
    resident: Option<u64>,
}

impl ReadFrom {
    /// Whether the tables were read through `interface` with `maps` at `moved` moves: then a
    /// table that keeps its stamp is [`unchanged`](Self::unchanged).
    pub(super) fn reads(
        &self,
        interface: &PipelineInterface,
        maps: &SamplerMaps,
        moved: u64,
    ) -> bool {
        self.moved == moved && self.through(interface) && self.maps == *maps
    }

    /// Whether the tables were read through `interface` at `moved` moves, with the maps they were
    /// read with: [`reads`](Self::reads) of its own maps, which it compares with none.
    pub(super) fn reads_through(&self, interface: &PipelineInterface, moved: u64) -> bool {
        self.moved == moved && self.through(interface)
    }

    /// The maps the tables were read with.
    pub(super) fn maps(&self) -> &SamplerMaps {
        &self.maps
    }

    /// Whether the tables were read through `interface`: the one at its address, while the
    /// drawing state that held it then lives.
    fn through(&self, interface: &PipelineInterface) -> bool {
        let at = std::ptr::from_ref(interface).addr();
        (self.interface.as_ref())
            .is_some_and(|(read, state)| *read == at && state.strong_count() > 0)
    }

    /// Whether the state's used resources were made of a residency of `generation` with no no-copy
    /// buffer moved since (`moved` moves): they are its used resources still, but for an indirect
    /// command buffer's hold on its ID's slot, which a buffer made later takes without changing
    /// the residency (`Residency::holds_slots_as`, the caller's to check).
    pub(super) fn resident(&self, generation: u64, moved: u64) -> bool {
        self.moved == moved && self.resident == Some(generation)
    }

    /// The state's used resources are those of a residency of `generation`, at the moves the
    /// tables were last [`read`](Self::read) at.
    pub(super) fn made_resident(&mut self, generation: u64) {
        self.resident = Some(generation);
    }

    /// Whether the table at `at` in the reading order, stamped `stamp`, is the one read.
    pub(super) fn unchanged(&self, at: usize, stamp: Stamp) -> bool {
        self.tables[at] == Some(stamp)
    }

    /// The stamp the table at `at` in the reading order was read under, if any: one whose calls
    /// since it keeps brings in their slots alone (`ArgumentTable::refresh_changed`).
    pub(super) fn read_under(&self, at: usize) -> Option<Stamp> {
        self.tables[at]
    }

    /// The tables of `stamps` read now through `interface` (of the drawing state `state` gives)
    /// with `maps` at `moved` moves.
    pub(super) fn read(
        &mut self,
        stamps: [Stamp; 4],
        (interface, state): (
            &PipelineInterface,
            impl FnOnce() -> Arc<dyn Any + Send + Sync>,
        ),
        maps: &SamplerMaps,
        moved: u64,
    ) {
        if !self.through(interface) {
            let at = std::ptr::from_ref(interface).addr();
            self.interface = Some((at, Arc::downgrade(&state())));
        }
        if self.maps != *maps {
            self.maps = maps.clone();
        }
        self.tables = stamps.map(Some);
        self.moved = moved;
    }

    /// [`read`](Self::read) with the maps the tables were read with, which it neither compares
    /// nor copies: the tables of `stamps` read now through `interface` (of the drawing state
    /// `state` gives) at `moved` moves.
    pub(super) fn read_again(
        &mut self,
        stamps: [Stamp; 4],
        (interface, state): (
            &PipelineInterface,
            impl FnOnce() -> Arc<dyn Any + Send + Sync>,
        ),
        moved: u64,
    ) {
        if !self.through(interface) {
            let at = std::ptr::from_ref(interface).addr();
            self.interface = Some((at, Arc::downgrade(&state())));
        }
        self.tables = stamps.map(Some);
        self.moved = moved;
    }
}

impl LiveBind {
    /// Its encoder ended: what its draws settled, the shadows' copies listed and the heap parts
    /// kept go with the encoder (the module docs), so the next encoder may start from it.
    pub(super) fn end_encoder(&mut self) {
        self.end_chunk();
        self.end_carried();
    }

    /// Its encoder ended, carrying its kept draws into its command buffer's next render encoder
    /// (`render/carried.rs`): as [`end_encoder`](Self::end_encoder), but what its draws settled
    /// stays, which the command buffer holds the variants' objects of, with the kept draws set
    /// aside that carry ([`carry_aside`](Self::carry_aside)). The shadows' copies listed and the
    /// heap parts kept go: they are its residency's, which is the encoder's.
    pub(super) fn end_carried(&mut self) {
        self.shadows.let_go();
        self.heap.let_go();
    }

    /// Its encoder, a chunk's of an execution's commands recorded apart, ended, and the next
    /// chunk's starts from it in the same recording (`render/executions/kept.rs`): as
    /// [`end_encoder`](Self::end_encoder), but the shadows' copies listed and the heap parts
    /// stay, so the chunks' draws take them as the draws of one encoder do. Each is a function of
    /// what a draw compares before it takes it: the residency's generation and the device's count
    /// of usage shadows made (`shadow::ResidentShadows`), and what a part is made of
    /// (`render/heap_part.rs`).
    pub(super) fn end_chunk(&mut self) {
        self.forget_settles();
    }

    /// Start a draw of `pipeline` from the state that pipeline state's last draw left, where one
    /// is set aside: it and the state in place change places. Where none is, the draw starts from
    /// a state no live pipeline state's draw left, the one in place first; else from the one in
    /// place where its tables were read through `pipeline`'s interface (so the draw reads only
    /// what changed since), else from the one last set aside of those so read, else from the one
    /// set aside longest, which it takes as `pipeline`'s. Each is a state as draws left it (or
    /// empty, as a new encoder's): exact whichever a draw starts from, as it brings every field of
    /// the state in place (the module docs), so this chooses only whether the shape and the draw
    /// may be kept, and which reads the draw skips.
    pub(super) fn select(&mut self, pipeline: &GraphicsState) {
        if self.draws(pipeline) {
            return;
        }
        let interface = &pipeline.base().interface;
        let lives = |drawn: &Option<WeakRenderPipelineState>| {
            (drawn.as_ref()).is_some_and(WeakRenderPipelineState::lives)
        };
        let set_aside = self.set_aside.iter().enumerate();
        let own = (set_aside.clone())
            .find(|(_, aside)| (aside.drawn.as_ref()).is_some_and(|drawn| drawn.is(pipeline)));
        // `None` starts from the state in place.
        let at = if let Some((at, _)) = own {
            Some(at)
        } else if !lives(&self.drawn) {
            None
        } else if let Some((at, _)) = set_aside.clone().find(|(_, aside)| !lives(&aside.drawn)) {
            Some(at)
        } else if self.read_from.through(interface) {
            None
        } else {
            let through = (set_aside.clone())
                .filter(|(_, aside)| aside.read_from.through(interface))
                .max_by_key(|(_, aside)| aside.at);
            let longest = || set_aside.clone().min_by_key(|(_, aside)| aside.at);
            through.or_else(longest).map(|(at, _)| at)
        };
        if let Some(at) = at {
            self.swap_aside(at);
        }
        if !self.draws(pipeline) {
            self.drawn = Some(pipeline.downgrade());
        }
    }

    /// Change the state in place for the one set aside at `at`.
    fn swap_aside(&mut self, at: usize) {
        self.asides += 1;
        let aside = &mut self.set_aside[at];
        std::mem::swap(&mut self.state, &mut aside.state);
        std::mem::swap(&mut self.shape, &mut aside.shape);
        std::mem::swap(&mut self.recent, &mut aside.recent);
        std::mem::swap(&mut self.listed_at, &mut aside.listed_at);
        std::mem::swap(&mut self.read_from, &mut aside.read_from);
        std::mem::swap(&mut self.drawn, &mut aside.drawn);
        std::mem::swap(&mut self.settled, &mut aside.settled);
        std::mem::swap(&mut self.rebinds, &mut aside.rebinds);
        std::mem::swap(&mut self.commands, &mut aside.commands);
        std::mem::swap(&mut self.pushes, &mut aside.pushes);
        std::mem::swap(&mut self.uses, &mut aside.uses);
        std::mem::swap(&mut self.pushed, &mut aside.pushed);
        std::mem::swap(&mut self.offsets, &mut aside.offsets);
        std::mem::swap(&mut self.vertex_buffers, &mut aside.vertex_buffers);
        std::mem::swap(&mut self.vertex_starts, &mut aside.vertex_starts);
        std::mem::swap(&mut self.kept, &mut aside.kept);
        std::mem::swap(&mut self.draw_kept, &mut aside.draw_kept);
        aside.at = self.asides;
    }

    /// Whether the state in place is the one `pipeline`'s last draw left, which a draw of it
    /// starts from ([`select`](Self::select) keeps it).
    pub(super) fn draws(&self, pipeline: &GraphicsState) -> bool {
        (self.drawn.as_ref()).is_some_and(|drawn| drawn.is(pipeline))
    }

    /// Set `kept` aside with the state in place, its pipeline state's (the field's docs).
    pub(super) fn set_aside_kept(&mut self, kept: Option<DrawKept>) {
        self.draw_kept = kept;
    }

    /// The kept draw set aside with the state in place, taken.
    pub(super) fn take_kept(&mut self) -> Option<DrawKept> {
        self.draw_kept.take()
    }

    /// Of the kept draws set aside with the states of other pipeline states, keep for the next
    /// render encoder of its encoder's command buffer (`render/carried.rs`) those that `carries`
    /// tells carry (given each and whether its state holds its settle), of the `most` states set
    /// aside most lately, and let go of every other: how many that carry were let go past `most`.
    pub(super) fn carry_aside(
        &mut self,
        most: usize,
        carries: impl Fn(&DrawKept, bool) -> bool,
    ) -> u64 {
        let mut latest_first: [usize; SET_ASIDE] = std::array::from_fn(|at| at);
        latest_first.sort_unstable_by_key(|&at| std::cmp::Reverse(self.set_aside[at].at));
        let (mut kept, mut past) = (0, 0);
        for at in latest_first {
            let aside = &mut self.set_aside[at];
            let settled = aside.settled.is_some();
            let carry = (aside.draw_kept.as_ref()).is_some_and(|draw| carries(draw, settled));
            if carry && kept < most {
                kept += 1;
                continue;
            }
            past += u64::from(carry);
            aside.draw_kept = None;
        }
        past
    }

    /// Carry each kept draw set aside into the render encoder that starts from this state
    /// (`render/carried.rs`) by `carry`: how many.
    pub(super) fn carried_aside(&mut self, mut carry: impl FnMut(&mut DrawKept)) -> u64 {
        let mut carried = 0;
        for draw in (self.set_aside.iter_mut()).filter_map(|aside| aside.draw_kept.as_mut()) {
            carry(draw);
            carried += 1;
        }
        carried
    }

    /// What the state's last recorded draw settled, where it drew `pipeline` in `forms` and its
    /// shape was `shape` (the very one): what `GraphicsState::settle_shape` answers for them.
    pub(super) fn settled(
        &self,
        pipeline: &GraphicsState,
        shape: &Arc<BindShape>,
        forms: BufferForms,
    ) -> Option<&Settled> {
        let here = self.settled.as_ref()?;
        let same = here.forms == forms
            && Arc::ptr_eq(&here.settled.shape, shape)
            && pipeline.runs(&here.program);
        same.then_some(&here.settled)
    }

    /// Whether a draw of `pipeline` in `forms` settles as the state's last recorded draw did
    /// ([`settled`](Self::settled)), its shape kept: no change since may have changed it.
    pub(super) fn settles_kept(&self, pipeline: &GraphicsState, forms: BufferForms) -> bool {
        let shape = self.shape.as_ref();
        shape.is_some_and(|shape| self.settled(pipeline, shape, forms).is_some())
    }

    /// Let go of every state's kept settle, as their encoder ends.
    fn forget_settles(&mut self) {
        self.settled = None;
        self.draw_kept = None;
        for aside in &mut self.set_aside {
            aside.settled = None;
            aside.draw_kept = None;
        }
    }

    /// Take the changes brought into the state since its shape was made.
    pub(super) fn reshaped(&mut self, reshape: Reshape) {
        if reshape.0 {
            self.shape = None;
        }
    }

    /// The state's shape: the kept one, else the one of those it took lately that is its shape
    /// ([`RecentShapes`]), else the one `found` finds of the state, else made again (counted in
    /// `counts`, found or made).
    pub(super) fn shape(
        &mut self,
        counts: Counting<'_>,
        found: impl FnOnce(&BindState) -> Option<Arc<BindShape>>,
    ) -> Arc<BindShape> {
        if let Some(shape) = &self.shape {
            debug_assert!(
                **shape == self.state.shape(),
                "a kept bind shape that is not its state's"
            );
            return Arc::clone(shape);
        }
        if let Some(shape) = self.recent.find(&self.state) {
            debug_assert!(
                *shape == self.state.shape(),
                "a recent bind shape that is not its state's"
            );
            self.shape = Some(Arc::clone(&shape));
            return shape;
        }
        let shape = match found(&self.state) {
            Some(shape) => {
                debug_assert!(
                    *shape == self.state.shape(),
                    "a bind shape found that is not its state's"
                );
                counts.draw_shape_found();
                shape
            }
            None => {
                counts.draw_shape_made();
                Arc::new(self.state.shape())
            }
        };
        self.recent.keep(&shape);
        self.shape = Some(Arc::clone(&shape));
        shape
    }
}

/// A vector in storage kept beyond its use (a [`LiveBind`]'s), emptied as it is made and as it
/// drops: what it holds lives as long as it would in a vector of its own, on every path out of its
/// use, while the storage goes on to the next one.
pub(super) struct Emptied<'a, T>(&'a mut Vec<T>);

impl<'a, T> Emptied<'a, T> {
    /// `storage`, emptied.
    pub(super) fn new(storage: &'a mut Vec<T>) -> Emptied<'a, T> {
        storage.clear();
        Emptied(storage)
    }
}

impl<T> Deref for Emptied<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        self.0
    }
}

impl<T> DerefMut for Emptied<'_, T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        self.0
    }
}

impl<T> Drop for Emptied<'_, T> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

/// A device's bind states that ended render encoders left, which the render encoders it opens
/// next start from (the module docs): at most [`SPARE_LIVE_BINDS`], the latest given taken first.
/// Each is kept in the box its encoder held it in, so an encoder's open and end move and allocate
/// none.
#[derive(Default)]
pub(super) struct SpareLiveBinds(Mutex<Vec<Box<LiveBind>>>);

impl SpareLiveBinds {
    /// The state a render encoder opening starts from: the latest an encoder gave back, else an
    /// empty one.
    pub(super) fn take(&self) -> Box<LiveBind> {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep `live`, an ended render encoder's state, for the next encoder to start from, where
    /// fewer than [`SPARE_LIVE_BINDS`] are kept; else it goes. A draw that panicked left it as far
    /// as it got, which the next draw brings in place as any.
    pub(super) fn give(&self, mut live: Box<LiveBind>) {
        live.end_encoder();
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_LIVE_BINDS {
            spare.push(live);
        }
    }
}
