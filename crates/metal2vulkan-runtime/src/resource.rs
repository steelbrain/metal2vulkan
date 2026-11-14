//! The identity every resource carries: a key no other resource of the process ever has, and
//! whether Metal tracks the resource's hazards. Hazard tracking keys on it; each resource kind
//! answers it from its own file (`Buffer::identity` in buffer.rs).

use super::values::{HazardTrackingMode, StorageMode};
use super::{Buffer, Device, IndirectCommandBuffer, Texture};
use std::sync::atomic::{AtomicU64, Ordering};

/// A resource's key: unique in the process, never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ResourceKey(u64);

/// The bit of a read twin's key ([`ResourceKey::read_twin`]): no count of keys reaches it.
const READ_TWIN: u64 = 1 << 63;

impl ResourceKey {
    /// A key no resource had before.
    pub(super) fn new() -> ResourceKey {
        ResourceKey(next_key())
    }

    /// A key no resource had before, of a read twin (`read_twin.rs`): a runtime buffer over a
    /// no-copy buffer's read-only pages that no command may write, which every use states
    /// ([`is_read_twin`](Self::is_read_twin)) so that an encoder refuses a write of it.
    pub(super) fn read_twin() -> ResourceKey {
        ResourceKey(next_key() | READ_TWIN)
    }

    /// Whether it is a read twin's key ([`read_twin`](Self::read_twin)).
    pub(super) fn is_read_twin(self) -> bool {
        self.0 & READ_TWIN != 0
    }

    /// Its place among the process's keys, made in order: the first is 1.
    pub(super) fn ordinal(self) -> u64 {
        self.0 & !READ_TWIN
    }
}

/// The next key's number, counted from 1 for every key of the process, a read twin's too.
fn next_key() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A buffer's, texture's, heap's or command buffer's identity (`Buffer::object_id`,
/// `Texture::object_id`, `Heap::object_id`, `CommandBuffer::object_id`): a runtime extension, no
/// Metal type. Metal names an object by its pointer (`id`), which another object may take once it
/// is freed, and by nothing else where it has no address (a sparse heap's `gpuAddress` is 0).
/// This number is unique among the process's buffers, textures, heaps and command buffers, is the
/// object's for its whole life (a clone is the same object), and is never given to another, so a
/// program can key what it knows of an object by it without holding the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(u64);

impl ObjectId {
    /// An identity no object had before.
    pub(super) fn new() -> ObjectId {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        ObjectId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The identity whose number ([`get`](Self::get)) is `number`; `None` for 0, no object's.
    pub(super) fn of_number(number: u64) -> Option<ObjectId> {
        (number != 0).then_some(ObjectId(number))
    }

    /// The number, never 0.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// What hazard tracking knows of a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Identity {
    pub(super) key: ResourceKey,
    /// Metal orders the resource's accesses between encoders and between dispatches of a serial
    /// encoder; an untracked resource is ordered by the program.
    pub(super) tracked: bool,
}

impl Identity {
    /// A new resource's identity for its hazard tracking `mode`: `Default` is tracked for a
    /// resource made outside a heap and untracked for one made in a heap, as in Metal.
    pub(super) fn new(mode: HazardTrackingMode, in_heap: bool) -> Identity {
        let tracked = match mode {
            HazardTrackingMode::Tracked => true,
            HazardTrackingMode::Untracked => false,
            HazardTrackingMode::Default => !in_heap,
        };
        Identity {
            key: ResourceKey::new(),
            tracked,
        }
    }

    /// A read twin's identity (`read_twin.rs`): tracked, so its reads are ordered after the
    /// writes of a writable buffer over the same pages whose memory it joined
    /// (`host_aliases.rs`).
    pub(super) fn read_twin() -> Identity {
        Identity {
            key: ResourceKey::read_twin(),
            tracked: true,
        }
    }

    /// The resolved hazard tracking mode: `Tracked` or `Untracked`.
    pub(super) fn mode(self) -> HazardTrackingMode {
        if self.tracked {
            HazardTrackingMode::Tracked
        } else {
            HazardTrackingMode::Untracked
        }
    }
}

/// Any Metal resource, for the calls that take a list of them (`memoryBarrierWithResources:`).
/// Each resource kind adds its variant and its arm in the helpers below.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ResourceRef<'a> {
    /// A buffer.
    Buffer(&'a Buffer),
    /// A texture, or a texture view (its texture's resource).
    Texture(&'a Texture),
    /// An indirect command buffer: a function encodes commands into it once the encoder made it
    /// resident for writing.
    IndirectCommandBuffer(&'a IndirectCommandBuffer),
}

impl<'a> From<&'a Buffer> for ResourceRef<'a> {
    fn from(buffer: &'a Buffer) -> ResourceRef<'a> {
        ResourceRef::Buffer(buffer)
    }
}

impl<'a> From<&'a Texture> for ResourceRef<'a> {
    fn from(texture: &'a Texture) -> ResourceRef<'a> {
        ResourceRef::Texture(texture)
    }
}

impl<'a> From<&'a IndirectCommandBuffer> for ResourceRef<'a> {
    fn from(icb: &'a IndirectCommandBuffer) -> ResourceRef<'a> {
        ResourceRef::IndirectCommandBuffer(icb)
    }
}

impl Buffer {
    /// The buffer as a resource of a resource list.
    pub fn as_resource(&self) -> ResourceRef<'_> {
        ResourceRef::Buffer(self)
    }
}

impl Texture {
    /// The texture as a resource of a resource list.
    pub fn as_resource(&self) -> ResourceRef<'_> {
        ResourceRef::Texture(self)
    }
}

impl IndirectCommandBuffer {
    /// The indirect command buffer as a resource of a resource list.
    pub fn as_resource(&self) -> ResourceRef<'_> {
        ResourceRef::IndirectCommandBuffer(self)
    }
}

impl ResourceRef<'_> {
    /// The device that made the resource.
    pub(super) fn device(&self) -> &Device {
        match self {
            ResourceRef::Buffer(buffer) => buffer.generation(),
            ResourceRef::Texture(texture) => texture.generation(),
            ResourceRef::IndirectCommandBuffer(icb) => icb.generation(),
        }
    }

    /// The resource's storage mode.
    pub(super) fn storage_mode(&self) -> StorageMode {
        match self {
            ResourceRef::Buffer(buffer) => buffer.storage_mode(),
            ResourceRef::Texture(texture) => texture.storage_mode(),
            ResourceRef::IndirectCommandBuffer(icb) => icb.storage_mode(),
        }
    }

    /// Whether the resource is memoryless: a memoryless texture or texture buffer, whose use
    /// fails Metal's command buffer (`CommandBufferError::InvalidResource`).
    pub(super) fn is_memoryless(&self) -> bool {
        match self {
            ResourceRef::Texture(texture) => texture.is_memoryless(),
            _ => self.storage_mode() == StorageMode::Memoryless,
        }
    }
}
