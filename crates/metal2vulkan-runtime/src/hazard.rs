//! The hazard sets of a command buffer's compute encoders: the resources read and the resources
//! written since the last barrier, by resource as Metal tracks them (a heap's resources as one, the
//! heap's: [`HeapId`]), not by byte range. They are a
//! table indexed by resource id with a generation counter, so starting the sets again costs one
//! increment. They carry across the encoders of a command buffer until a barrier, so a write after
//! an encoder that only read still finds the read.

use crate::alias::Placement;
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

/// The sets.
pub(crate) struct Hazards {
    /// Per resource id, the generations it was last read and last written in; a mark of an older
    /// generation is in neither set.
    marks: Vec<[u32; 2]>,
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
            generation: 1,
            wrote: false,
            touched: false,
        }
    }
}

impl Hazards {
    /// Whether `access` of resource `id` conflicts with the sets: it reads or writes a resource
    /// in the written set, or writes one in the read set.
    pub(crate) fn conflicts(&self, id: u32, access: Access) -> bool {
        match self.marks.get(id as usize) {
            Some(&[read, written]) => {
                written == self.generation || (access.writes() && read == self.generation)
            }
            None => false,
        }
    }

    /// Adds `access` of resource `id` to the sets.
    pub(crate) fn mark(&mut self, id: u32, access: Access) {
        let index = id as usize;
        if index >= self.marks.len() {
            self.marks.resize(index + 1, [0, 0]);
        }
        let mark = &mut self.marks[index];
        if access.reads() {
            mark[0] = self.generation;
        }
        if access.writes() {
            mark[1] = self.generation;
            self.wrote = true;
        }
        self.touched = true;
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
            self.generation = 0;
        }
        self.generation += 1;
        self.wrote = false;
        self.touched = false;
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
    /// Its resources' placements, held weakly, whose bytes meeting link them (`crate::alias`).
    places: Mutex<Vec<Weak<Placement>>>,
}

impl HeapId {
    /// A new id that resources over one memory take as one: an exported surface's planes in
    /// place and its buffer ([`crate::Surface::buffer`]).
    pub(crate) fn unit(core: &Arc<Core>) -> Arc<HeapId> {
        Arc::new(HeapId {
            core: Arc::clone(core),
            id: core.ids.take(),
            places: Mutex::new(Vec::new()),
        })
    }

    /// The id.
    pub(crate) fn get(&self) -> u32 {
        self.id
    }

    /// Its resources' placements.
    pub(crate) fn places(&self) -> &Mutex<Vec<Weak<Placement>>> {
        &self.places
    }

    /// A placement of the heap gained its first link (`gained`) or lost its last.
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
    /// The device's heap placements with links (`crate::alias`): while none has, a command's
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
            places: Mutex::new(Vec::new()),
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
