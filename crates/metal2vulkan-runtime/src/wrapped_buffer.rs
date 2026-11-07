//! Buffers over buffers the caller made: `Device::new_buffer_with_vk_buffer`, the buffer twin of
//! `Device::new_texture_with_vk_image` (`texture/wrapped.rs`). Memory a consumer brings in with
//! plain Vulkan (another process's memory it imported as a dma-buf, for example) becomes a Metal
//! buffer the runtime binds, copies and addresses as its own.
//!
//! - **Ownership.** The caller made the `VkBuffer` and the `VkDeviceMemory` it is bound to, and
//!   keeps both: the runtime never destroys or frees them (`buffer::Owner::Vulkan`). It holds a
//!   range of its Metal address space for the buffer (`gpu_address`), released with it.
//! - **What the buffer is.** Vulkan has no query for a buffer's creation, so the caller states it:
//!   the `VkBufferCreateInfo` it made the buffer with, and where it bound it
//!   ([`VkBufferMemory`]). The buffer must be one the runtime could have made: at least the
//!   length padded to 16 bytes (the binding layer reads whole 16-byte words), every usage the
//!   runtime's own buffers have ([`Device::vk_buffer_usage`]: a Metal buffer may be bound or used
//!   as anything), not sparse or protected, and shared with the device's queue family if
//!   concurrent. A contradiction is refused, naming it ([`WrappedBufferProblem`]).
//! - **Storage.** `Shared`: the CPU reaches the bytes at the host address the caller gives, a
//!   mapping of the memory (its own `vkMapMemory`) or another mapping of the same pages (the pages
//!   an imported dma-buf or host allocation came from), in host-visible coherent memory, as the
//!   runtime's own `Shared` memory is. `Private`: no CPU access, in memory of any type. `Managed`
//!   and `Memoryless` storage are refused (`Invalid`, `StorageMode`).
//! - **Contents** are the caller's: the runtime neither clears them when it wraps the buffer nor
//!   empties them (`set_purgeable_state` answers `NonVolatile`, as for a no-copy buffer).
//! - **Textures over it** copy their texels between the buffer and an image of the runtime's
//!   (`shadowed-texture`): the memory may be an import whose handle type no image of the
//!   runtime's was made for, and an image bound to imported memory must be. (A no-copy buffer's
//!   handle type is the runtime's own, host allocations, so a texture over one aliases its bytes
//!   where the device lays them out as Metal does.)
//! - **Hazards** are tracked as for a buffer the runtime made; the caller orders its own use of
//!   the memory against the runtime's work through semaphores or completion.

use super::address::AddressPlacement;
use super::buffer::{usage, Buffer, BufferParts, Modes, Owner};
use super::device::lock;
use super::error::{invalid, nil, Error, Problem};
use super::memory::Placement;
use super::sharing::Sharing;
use super::values::{ResourceOptions, StorageMode};
use super::Device;
use ash::vk;
use metal2vulkan::vk::BufferUsageFlags;
use std::ptr::NonNull;

/// Where the caller bound a `VkBuffer` it hands to [`Device::new_buffer_with_vk_buffer`]. No Metal
/// counterpart: an interop hook.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkBufferMemory {
    /// The `VkDeviceMemory` the buffer is bound to.
    pub memory: vk::DeviceMemory,
    /// Where the buffer is bound in it (`vkBindBufferMemory`'s `memoryOffset`).
    pub offset: u64,
    /// The memory's type (`VkMemoryAllocateInfo::memoryTypeIndex`).
    pub type_index: u32,
    /// For `Shared` storage, the host address of the buffer's byte 0: a mapping of the memory or
    /// another mapping of the same pages. `None` for `Private` storage.
    pub contents: Option<NonNull<u8>>,
}

impl VkBufferMemory {
    /// The buffer is bound at `offset` in `memory`, of type `type_index`, its bytes at
    /// `contents` on the host where it has them.
    pub fn new(
        memory: vk::DeviceMemory,
        offset: u64,
        type_index: u32,
        contents: Option<NonNull<u8>>,
    ) -> VkBufferMemory {
        VkBufferMemory {
            memory,
            offset,
            type_index,
            contents,
        }
    }
}

/// Why a buffer the caller made cannot be wrapped as a Metal buffer
/// ([`Device::new_buffer_with_vk_buffer`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrappedBufferProblem {
    /// A null buffer handle.
    NullBuffer,
    /// A null memory handle.
    NullMemory,
    /// The buffer is smaller than the length padded to 16 bytes, which a binding of the buffer
    /// may reach.
    Size {
        /// The buffer's size (`VkBufferCreateInfo::size`).
        buffer: u64,
        /// The length padded to 16 bytes.
        needed: u64,
    },
    /// The buffer lacks usages the runtime's own buffers have ([`Device::vk_buffer_usage`]).
    Usage {
        /// The usages the buffer lacks.
        missing: BufferUsageFlags,
    },
    /// A sparse buffer, which has no one memory binding.
    Sparse,
    /// A protected buffer: the runtime makes no protected submissions.
    Protected,
    /// A buffer not shared with every queue family of the device: a concurrent one whose
    /// families leave out one of the device's, or, on a device of several families, an exclusive
    /// one, as the runtime transfers no ownership between them.
    QueueFamily,
    /// A memory type the device does not have, or, for `Shared` storage, one that is not
    /// host-visible and coherent.
    MemoryType {
        /// The memory type's index.
        index: u32,
    },
    /// `Shared` storage without the host address of the bytes, or `Private` storage with one.
    Contents,
}

/// What the runtime checks of a caller's buffer: its handle, and its creation's size, usage,
/// flags and sharing.
#[derive(Clone, Copy)]
struct Created<'a> {
    buffer: vk::Buffer,
    size: u64,
    usage: vk::BufferUsageFlags,
    flags: vk::BufferCreateFlags,
    /// A concurrent buffer's queue families; `None` for an exclusive one.
    families: Option<&'a [u32]>,
}

/// Why the buffer `created` and bound at `memory` cannot be a buffer of `length` bytes and
/// `storage` on a device whose buffers take `needed` usage, whose memory type `type_index` has
/// `type_flags` (`None` where it has no such type), and whose buffers are shared as `sharing`
/// says.
fn problem(
    created: &Created<'_>,
    memory: &VkBufferMemory,
    type_flags: Option<vk::MemoryPropertyFlags>,
    storage: StorageMode,
    length: u64,
    needed: vk::BufferUsageFlags,
    sharing: Sharing<'_>,
) -> Option<WrappedBufferProblem> {
    use vk::MemoryPropertyFlags as P;
    let padded = length.checked_next_multiple_of(16).unwrap_or(u64::MAX);
    let missing = needed & !created.usage;
    let shared = storage == StorageMode::Shared;
    let host = P::HOST_VISIBLE | P::HOST_COHERENT;
    let admitted = match created.families {
        Some(families) => sharing.admits(vk::SharingMode::CONCURRENT, families),
        None => sharing.admits(vk::SharingMode::EXCLUSIVE, &[]),
    };
    if created.buffer == vk::Buffer::null() {
        Some(WrappedBufferProblem::NullBuffer)
    } else if memory.memory == vk::DeviceMemory::null() {
        Some(WrappedBufferProblem::NullMemory)
    } else if created.size < padded {
        Some(WrappedBufferProblem::Size {
            buffer: created.size,
            needed: padded,
        })
    } else if !missing.is_empty() {
        Some(WrappedBufferProblem::Usage {
            missing: missing.into(),
        })
    } else if created.flags.intersects(
        vk::BufferCreateFlags::SPARSE_BINDING
            | vk::BufferCreateFlags::SPARSE_RESIDENCY
            | vk::BufferCreateFlags::SPARSE_ALIASED,
    ) {
        Some(WrappedBufferProblem::Sparse)
    } else if created.flags.contains(vk::BufferCreateFlags::PROTECTED) {
        Some(WrappedBufferProblem::Protected)
    } else if !admitted {
        Some(WrappedBufferProblem::QueueFamily)
    } else if type_flags.is_none_or(|flags| shared && !flags.contains(host)) {
        Some(WrappedBufferProblem::MemoryType {
            index: memory.type_index,
        })
    } else if memory.contents.is_some() != shared {
        Some(WrappedBufferProblem::Contents)
    } else {
        None
    }
}

impl Device {
    /// The usage a `VkBuffer` the caller makes for [`Device::new_buffer_with_vk_buffer`] needs:
    /// what a Metal buffer may be bound or used as (transfers, uniform, storage, texel, index,
    /// vertex and indirect buffers), and device addresses where the device enables
    /// `bufferDeviceAddress`, since the runtime asks such a buffer for its own address (the
    /// buffers it places in its own memory take theirs from their memory blocks instead).
    pub fn vk_buffer_usage(&self) -> BufferUsageFlags {
        usage(self.shared().device_address).into()
    }

    /// A buffer of `length` bytes and `options` over `buffer`, which the caller made with `info`,
    /// bound at `memory`, and keeps (the module docs): the runtime never destroys the buffer or
    /// frees its memory, and does not clear its bytes. Hazard tracking orders its uses as it
    /// orders a buffer the runtime made, and it holds a range of the device's Metal address space
    /// (`gpu_address`).
    ///
    /// Refused: a length of 0 or past the device's `max_buffer_length` (`Nil`), as
    /// [`Device::new_buffer`] refuses it; storage other than `Shared` and `Private` (`Invalid`,
    /// `StorageMode`); a buffer or memory that contradicts the length or storage, or that the
    /// runtime could not have made (`Invalid` with a [`WrappedBufferProblem`]).
    ///
    /// # Safety
    ///
    /// - `buffer` is a live buffer of this device, made with exactly `info` (its `pNext` chain is
    ///   not read), and bound whole, from `memory.offset`, to `memory.memory`, a live allocation
    ///   of memory type `memory.type_index`. Where the device enables `bufferDeviceAddress`
    ///   ([`Device::vk_buffer_usage`] then names `SHADER_DEVICE_ADDRESS`), the memory was
    ///   allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`.
    /// - For `Shared` storage, `memory.contents` is the host address of the buffer's byte 0, mapped
    ///   readable and writable for `length` bytes, coherent with the device's view of the memory,
    ///   for as long as the buffer lives.
    /// - The buffer and its memory outlive the Metal buffer and every command buffer that uses
    ///   it; the caller does not destroy, free, rebind or unmap them meanwhile.
    /// - The caller orders its own use of the memory, on the host or the device, against the
    ///   runtime's work on the buffer (by the binary semaphores of
    ///   [`CommandBuffer::encode_wait_for_vk_semaphore`](super::CommandBuffer::encode_wait_for_vk_semaphore)
    ///   and
    ///   [`encode_signal_vk_semaphore`](super::CommandBuffer::encode_signal_vk_semaphore), or by
    ///   waiting for their completion).
    pub unsafe fn new_buffer_with_vk_buffer(
        &self,
        buffer: vk::Buffer,
        info: &vk::BufferCreateInfo<'_>,
        memory: VkBufferMemory,
        length: u64,
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer_with_vk_buffer";
        let options = self.own_memory_options(CALL, options)?;
        let modes = Modes::of(options)?;
        if !matches!(modes.storage, StorageMode::Shared | StorageMode::Private) {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(CALL, Problem::BufferLength { length }));
        }
        let shared = self.shared();
        let families = (info.sharing_mode == vk::SharingMode::CONCURRENT).then(|| {
            if info.p_queue_family_indices.is_null() {
                &[][..]
            } else {
                // SAFETY: the caller's contract: `info` is the buffer's creation, whose
                // concurrent queue families are `queue_family_index_count` at
                // `p_queue_family_indices`.
                unsafe {
                    std::slice::from_raw_parts(
                        info.p_queue_family_indices,
                        info.queue_family_index_count as usize,
                    )
                }
            }
        });
        let created = Created {
            buffer,
            size: info.size,
            usage: info.usage,
            flags: info.flags,
            families,
        };
        let type_flags = lock(&shared.memory).type_flags(memory.type_index);
        if let Some(problem) = problem(
            &created,
            &memory,
            type_flags,
            modes.storage,
            length,
            usage(shared.device_address),
            shared.sharing(),
        ) {
            return Err(invalid(CALL, Problem::WrappedBuffer(problem)));
        }
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let placement = Placement::caller(
            memory.memory,
            memory.offset,
            info.size,
            memory.type_index,
            memory.contents,
        );
        let (address, reserved) = lock(&shared.addresses).place(AddressPlacement::Any, length)?;
        let buffer = self.buffer_of(BufferParts {
            raw: buffer,
            memory: placement,
            offset: 0,
            base: 0,
            length,
            reach: length,
            modes,
            address,
            reserved,
            owner: Owner::Vulkan,
            release: None,
            of_heap: false,
            drops_writes: false,
            mirror: None,
            #[cfg(unix)]
            again: None,
        });
        Ok(self.visible(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    const FAMILY: u32 = 2;

    fn needed() -> vk::BufferUsageFlags {
        usage(true)
    }

    fn created(families: Option<&[u32]>) -> Created<'_> {
        Created {
            buffer: vk::Buffer::from_raw(9),
            size: 256,
            usage: needed(),
            flags: vk::BufferCreateFlags::empty(),
            families,
        }
    }

    fn memory(contents: bool) -> VkBufferMemory {
        VkBufferMemory::new(
            vk::DeviceMemory::from_raw(7),
            64,
            1,
            contents.then(NonNull::dangling),
        )
    }

    fn host() -> Option<vk::MemoryPropertyFlags> {
        Some(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
    }

    fn check(
        created: &Created<'_>,
        memory: &VkBufferMemory,
        flags: Option<vk::MemoryPropertyFlags>,
        storage: StorageMode,
        length: u64,
    ) -> Option<WrappedBufferProblem> {
        problem(
            created,
            memory,
            flags,
            storage,
            length,
            needed(),
            Sharing::new(&[FAMILY]),
        )
    }

    /// A buffer the runtime could have made wraps as `Shared` in host-visible coherent memory
    /// with a host address, and as `Private` in any memory without one.
    #[test]
    fn a_buffer_the_runtime_could_have_made_wraps() {
        let shared = StorageMode::Shared;
        let private = StorageMode::Private;
        let device_local = Some(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        assert_eq!(
            check(&created(None), &memory(true), host(), shared, 256),
            None
        );
        assert_eq!(
            check(&created(None), &memory(true), host(), shared, 241),
            None
        );
        assert_eq!(
            check(&created(None), &memory(false), device_local, private, 1),
            None
        );
        let families = [0, FAMILY];
        assert_eq!(
            check(
                &created(Some(&families)),
                &memory(false),
                host(),
                private,
                1
            ),
            None
        );
    }

    /// Each contradiction is refused, naming it.
    #[test]
    fn a_buffer_the_runtime_could_not_have_made_is_refused_naming_why() {
        use WrappedBufferProblem as W;
        let shared = StorageMode::Shared;
        let at = |created: &Created<'_>, memory: &VkBufferMemory, flags, storage, length| {
            check(created, memory, flags, storage, length)
        };
        let mut null = created(None);
        null.buffer = vk::Buffer::null();
        assert_eq!(
            at(&null, &memory(true), host(), shared, 16),
            Some(W::NullBuffer)
        );
        let mut no_memory = memory(true);
        no_memory.memory = vk::DeviceMemory::null();
        assert_eq!(
            at(&created(None), &no_memory, host(), shared, 16),
            Some(W::NullMemory)
        );
        assert_eq!(
            at(&created(None), &memory(true), host(), shared, 257),
            Some(W::Size {
                buffer: 256,
                needed: 272
            })
        );
        let mut narrow = created(None);
        narrow.usage = needed() & !vk::BufferUsageFlags::INDIRECT_BUFFER;
        assert_eq!(
            at(&narrow, &memory(true), host(), shared, 16),
            Some(W::Usage {
                missing: vk::BufferUsageFlags::INDIRECT_BUFFER.into()
            })
        );
        let mut sparse = created(None);
        sparse.flags = vk::BufferCreateFlags::SPARSE_BINDING;
        assert_eq!(
            at(&sparse, &memory(true), host(), shared, 16),
            Some(W::Sparse)
        );
        let mut protected = created(None);
        protected.flags = vk::BufferCreateFlags::PROTECTED;
        assert_eq!(
            at(&protected, &memory(true), host(), shared, 16),
            Some(W::Protected)
        );
        let others = [0, 1];
        assert_eq!(
            at(&created(Some(&others)), &memory(true), host(), shared, 16),
            Some(W::QueueFamily)
        );
        // On a device of several families, an exclusive buffer is refused, and a concurrent one
        // must name every family.
        let several = [0, 1, FAMILY];
        let on_several = |created: &Created<'_>| {
            problem(
                created,
                &memory(true),
                host(),
                shared,
                16,
                needed(),
                Sharing::new(&several),
            )
        };
        assert_eq!(on_several(&created(None)), Some(W::QueueFamily));
        assert_eq!(
            on_several(&created(Some(&[0, FAMILY]))),
            Some(W::QueueFamily)
        );
        assert_eq!(on_several(&created(Some(&[FAMILY, 1, 0]))), None);
        let device_local = Some(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        assert_eq!(
            at(&created(None), &memory(true), device_local, shared, 16),
            Some(W::MemoryType { index: 1 })
        );
        assert_eq!(
            at(&created(None), &memory(true), None, shared, 16),
            Some(W::MemoryType { index: 1 })
        );
        assert_eq!(
            at(&created(None), &memory(false), host(), shared, 16),
            Some(W::Contents)
        );
        assert_eq!(
            at(
                &created(None),
                &memory(true),
                host(),
                StorageMode::Private,
                16
            ),
            Some(W::Contents)
        );
    }
}
