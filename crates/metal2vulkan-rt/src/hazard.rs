//! The hazard sets of a command buffer's compute encoders: the resources read and the resources
//! written since the last barrier, by resource as Metal tracks them, not by byte range. They are a
//! table indexed by resource id with a generation counter, so starting the sets again costs one
//! increment. They carry across the encoders of a command buffer until a barrier, so a write after
//! an encoder that only read still finds the read.

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
