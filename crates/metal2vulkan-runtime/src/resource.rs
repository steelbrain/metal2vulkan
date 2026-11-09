//! The identity every resource carries: a key no other resource of the process ever has, and
//! whether Metal tracks the resource's hazards. Hazard tracking keys on it; each resource kind
//! answers it from its own file (`Buffer::identity` in buffer.rs).

use super::values::{HazardTrackingMode, StorageMode};
use super::{Buffer, Device, IndirectCommandBuffer, Texture};
use std::sync::atomic::{AtomicU64, Ordering};

/// A resource's key: unique in the process, never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ResourceKey(u64);

impl ResourceKey {
    /// A key no resource had before.
    pub(super) fn new() -> ResourceKey {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        ResourceKey(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Its place among the process's keys, made in order: the first is 1.
    pub(super) fn ordinal(self) -> u64 {
        self.0
    }
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
            ResourceRef::Buffer(buffer) => buffer.device(),
            ResourceRef::Texture(texture) => texture.device(),
            ResourceRef::IndirectCommandBuffer(icb) => icb.device(),
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
        self.storage_mode() == StorageMode::Memoryless
    }
}
