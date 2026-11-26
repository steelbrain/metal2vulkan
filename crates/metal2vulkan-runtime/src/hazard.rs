//! The hazard sets of a command buffer's compute encoders: the resources read and the resources
//! written since the last barrier, by resource as Metal tracks them (a heap's resources as one, the
//! heap's: [`HeapId`]), not by byte range. They are a
//! table indexed by resource id with a generation counter, so starting the sets again costs one
//! increment. They carry across the encoders of a command buffer until a barrier, so a write after
//! an encoder that only read still finds the read.

use crate::alias::Places;
use crate::device::Core;
use crate::memory::BlockId;
use metal2vulkan::metal::BindingAccess;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};

/// How a dispatch uses a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub(crate) fn writes(self) -> bool {
        matches!(self, Access::Write | Access::ReadWrite)
    }

    pub(crate) fn reads(self) -> bool {
        matches!(self, Access::Read | Access::ReadWrite)
    }

    /// Metal's name of it: `MTLBindingAccess`.
    pub(crate) fn binding(self) -> BindingAccess {
        match self {
            Access::Read => BindingAccess::ReadOnly,
            Access::Write => BindingAccess::WriteOnly,
            Access::ReadWrite => BindingAccess::ReadWrite,
        }
    }
}

/// What an access depends on in the sets ([`Hazards::dependency`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dependency {
    /// A write of the resource: the access reads or writes what was written.
    Write,
    /// Only a read of it: the access writes what was read.
    Read,
}

/// The sets.
pub(crate) struct Hazards {
    /// Per resource id, the generations it was last read and last written in; a mark of an older
    /// generation is in neither set.
    marks: Vec<[u32; 2]>,
    /// The same marks by a resource's own id, of each resource that takes a shared id in
    /// `marks` (a heap's, [`HeapId`]): they place no barrier, and only tell a dependency on the
    /// resource itself from a conflict through the shared id alone, for the barrier's count.
    own: Vec<[u32; 2]>,
    /// The current generation, from 1.
    generation: u32,
    /// Since the last barrier: something was written, through a resource or through an address
    /// outside every resource.
    wrote: bool,
    /// Since the last barrier: something was read or written.
    touched: bool,
}

impl Default for Hazards {
    fn default() -> Hazards {
        Hazards {
            marks: Vec::new(),
            own: Vec::new(),
            generation: 1,
            wrote: false,
            touched: false,
        }
    }
}

impl Hazards {
    /// What `access` of resource `id` depends on in the sets, where it conflicts with them: it
    /// reads or writes a resource in the written set (`Write`), or writes one in the read set
    /// alone (`Read`).
    pub(crate) fn dependency(&self, id: u32, access: Access) -> Option<Dependency> {
        depends(&self.marks, self.generation, id, access)
    }

    /// [`dependency`](Self::dependency) by the own id of a resource that takes a shared id, from
    /// its [`mark_own`](Self::mark_own) marks.
    pub(crate) fn own_dependency(&self, id: u32, access: Access) -> Option<Dependency> {
        depends(&self.own, self.generation, id, access)
    }

    /// Adds `access` of resource `id` to the sets.
    pub(crate) fn mark(&mut self, id: u32, access: Access) {
        set(&mut self.marks, self.generation, id, access);
        if access.writes() {
            self.wrote = true;
        }
        self.touched = true;
    }

    /// Adds `access` by the own id of a resource that takes a shared id, which [`mark`](Self::mark)
    /// adds by the shared one.
    pub(crate) fn mark_own(&mut self, id: u32, access: Access) {
        set(&mut self.own, self.generation, id, access);
    }

    /// A write through an address outside every resource.
    pub(crate) fn mark_wild_write(&mut self) {
        self.wrote = true;
        self.touched = true;
    }

    /// A read through an address outside every resource: nothing to mark, but it was touched.
    pub(crate) fn mark_wild_read(&mut self) {
        self.touched = true;
    }

    /// Whether anything was written since the last barrier.
    pub(crate) fn wrote(&self) -> bool {
        self.wrote
    }

    /// Whether anything was read or written since the last barrier.
    pub(crate) fn touched(&self) -> bool {
        self.touched
    }

    /// Starts both sets again, at a barrier.
    pub(crate) fn clear(&mut self) {
        if self.generation == u32::MAX {
            self.marks.fill([0, 0]);
            self.own.fill([0, 0]);
            self.generation = 0;
        }
        self.generation += 1;
        self.wrote = false;
        self.touched = false;
    }
}

/// What `access` of resource `id` depends on in `marks` at `generation`: a write of it, or a read
/// of it alone where the access writes.
fn depends(marks: &[[u32; 2]], generation: u32, id: u32, access: Access) -> Option<Dependency> {
    let &[read, written] = marks.get(id as usize)?;
    if written == generation {
        Some(Dependency::Write)
    } else if access.writes() && read == generation {
        Some(Dependency::Read)
    } else {
        None
    }
}

/// Marks `access` of resource `id` in `marks` at `generation`.
fn set(marks: &mut Vec<[u32; 2]>, generation: u32, id: u32, access: Access) {
    let index = id as usize;
    if index >= marks.len() {
        marks.resize(index + 1, [0, 0]);
    }
    let mark = &mut marks[index];
    if access.reads() {
        mark[0] = generation;
    }
    if access.writes() {
        mark[1] = generation;
    }
}

/// Resource ids: small and dense, so the sets' table stays short; an id is reused once its
/// resource is gone. A reused id can only add a barrier, never lose one.
pub(crate) struct Ids {
    state: std::sync::Mutex<IdState>,
}

struct IdState {
    next: u32,
    free: Vec<u32>,
}

impl Ids {
    pub(crate) fn new() -> Ids {
        Ids {
            state: std::sync::Mutex::new(IdState {
                next: 0,
                free: Vec::new(),
            }),
        }
    }

    pub(crate) fn take(&self) -> u32 {
        let mut state = crate::device::lock(&self.state);
        match state.free.pop() {
            Some(id) => id,
            None => {
                state.next += 1;
                state.next - 1
            }
        }
    }

    pub(crate) fn give_back(&self, id: u32) {
        crate::device::lock(&self.state).free.push(id);
    }
}

/// A heap's id in the hazard sets, which every resource placed in the heap takes there in place
/// of its own: Metal tracks a heap's resources as one ("tracking is done at the granularity of the
/// heap", `MTLHeap.h`'s `hazardTrackingMode`), so a write through any of them orders every later
/// use of any other, whether or not their bytes meet. An exported surface's planes in place and
/// its buffer take one alike ([`HeapId::unit`]). Its id goes back once no resource holds it.
pub(crate) struct HeapId {
    core: Arc<Core>,
    id: u32,
    /// Its resources' live placements, whose bytes meeting link them (`crate::alias`).
    places: Mutex<Places>,
}

impl HeapId {
    /// A new id that resources over one memory take as one: an exported surface's planes in
    /// place and its buffer ([`crate::Surface::buffer`]).
    pub(crate) fn unit(core: &Arc<Core>) -> Arc<HeapId> {
        Arc::new(HeapId {
            core: Arc::clone(core),
            id: core.ids.take(),
            places: Mutex::new(Places::default()),
        })
    }

    /// The id.
    pub(crate) fn get(&self) -> u32 {
        self.id
    }

    /// Its resources' placements.
    pub(crate) fn places(&self) -> &Mutex<Places> {
        &self.places
    }

    /// A placement of the heap began to take part (`gained`) or ceased to (`crate::alias`).
    pub(crate) fn linked(&self, gained: bool) {
        let linked = &self.core.heap_ids.linked;
        match gained {
            true => linked.fetch_add(1, Relaxed),
            false => linked.fetch_sub(1, Relaxed),
        };
    }
}

impl Drop for HeapId {
    fn drop(&mut self) {
        self.core.ids.give_back(self.id);
    }
}

/// The heaps that hold a resource, by their block and their offset in it, each held weakly by
/// the device and strongly by its resources: the entries of heaps gone are dropped once there are
/// as many as there were entries at the last such pass, so the map stays within twice the live
/// heaps (and [`HEAP_IDS_PRUNE`]).
#[derive(Default)]
pub(crate) struct HeapIds {
    state: Mutex<HeapIdState>,
    /// The device's heap placements that take part (`crate::alias`): while none does, a command's
    /// uses look at no placement.
    linked: AtomicUsize,
}

#[derive(Default)]
struct HeapIdState {
    heaps: HashMap<(BlockId, u64), Weak<HeapId>>,
    prune_at: usize,
}

/// The entries at which [`HeapIds`] first drops the gone heaps'.
const HEAP_IDS_PRUNE: usize = 64;

impl HeapIds {
    /// Whether a heap placement of the device has links.
    pub(crate) fn any_linked(&self) -> bool {
        self.linked.load(Relaxed) != 0
    }

    /// The id of the heap at `offset` in `block`, a new one where no resource of it lives.
    pub(crate) fn of(&self, core: &Arc<Core>, block: BlockId, offset: u64) -> Arc<HeapId> {
        let mut state = crate::device::lock(&self.state);
        if let Some(id) = (state.heaps.get(&(block, offset))).and_then(Weak::upgrade) {
            return id;
        }
        if state.heaps.len() >= state.prune_at.max(HEAP_IDS_PRUNE) {
            state.heaps.retain(|_, id| id.strong_count() > 0);
            state.prune_at = state.heaps.len() * 2;
        }
        let id = Arc::new(HeapId {
            core: Arc::clone(core),
            id: core.ids.take(),
            places: Mutex::new(Places::default()),
        });
        state.heaps.insert((block, offset), Arc::downgrade(&id));
        id
    }

    /// For the tests: the entries the map holds, gone heaps' among them.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn entries(&self) -> usize {
        crate::device::lock(&self.state).heaps.len()
    }
}
