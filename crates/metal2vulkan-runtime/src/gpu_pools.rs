//! A command buffer's GPU pools: the index pool its index bounds and index rewrite passes reserve
//! indices in (`index_bounds.rs`, `index_rewrite.rs`), and the records pool its scanned emulated
//! mesh draws reserve records in, and their indirect draws with an object function their object
//! slots (`render/mesh.rs`). The GPU decides how much of each a command
//! buffer takes (an indirect draw's count, an object stage's grids), after the draws that read
//! them were recorded, so a pool is one block bound for the whole command buffer.
//!
//! - **Sized from the device's memory** ([`pool_budget`]): a sixty-fourth of the device's largest
//!   device-local heap, within one storage binding (`maxStorageBufferRange`, less the index
//!   pool's header) and one allocation (`maxMemoryAllocationSize`), in whole words; the index
//!   pool's within a `u32` of bytes, its counter's. Not a constant of the runtime's: a device of
//!   twice the memory has pools of twice the bytes. Before, the index pool was 1 MiB on every
//!   device.
//! - **One block of each kind per queue, shared by its command buffers** ([`QueueBlocks`]). The
//!   first pass of a command buffer that needs a pool leases its queue's block of that kind,
//!   which the first lease of the queue makes; the command buffer holds the [`PoolLease`] until
//!   it completes (its completed handlers ran, `command_buffer/mod.rs`), or until it is dropped
//!   unsubmitted. Every command buffer of the queue reserves from the block's start (each one's
//!   counter, or cursor, is its own), so they overwrite one another's reservations, in order:
//!   - The queue's batches go to its `VkQueue` in commit order, a command buffer's after every
//!     batch of the command buffers committed before it (`scheduler/mod.rs`, the order), so no
//!     other command buffer of the queue lies between a command buffer's batches.
//!   - Every command that writes a pool's bytes (an index bounds plan and fill, an index
//!     rewrite's dispatches, an emulated mesh draw's and a tessellation draw's scan) runs after a
//!     `Barrier::everything` its own command buffer records before it, from `ALL_COMMANDS` and
//!     `MEMORY_WRITE`. Vulkan applies that barrier's first scope to every command submitted to
//!     the `VkQueue` before it, earlier command buffers' included ("The first synchronization
//!     scope and access scope of each memory dependency defined by pDependencyInfo are applied
//!     to operations that occurred earlier in submission order", `vkCmdPipelineBarrier2`): so a
//!     command buffer's writes wait for the earlier ones' draws, which read the block (a write
//!     after a read needs only that execution dependency), and for their writes (the memory
//!     dependency). The barrier was there before; sharing adds no wait.
//!   - Queues sharing a `VkQueue` interleave their batches there (`scheduler/mod.rs`, many
//!     logical queues), so a block is the queue's, not the `VkQueue`'s.
//!
//!   A queue's block goes once no lease held it for [`QUIET`] or longer, as a later lease of its
//!   kind ends on any queue, or as the queue goes. Before, each command buffer in flight had a
//!   block of its own: two in flight at once made and released one a round, and keeping them made
//!   none but held as many as were in flight. A block of 1 GiB (a sixty-fourth of a 64 GiB heap)
//!   took 24 to 29 ms to make on the M3 Max (MoltenVK 1.4.1, macOS 27.0.1), 35 to 39 ms while the
//!   GPU ran, all of it MoltenVK's commit of the block to its device's residency set inside
//!   `vkBindBufferMemory`, under the lock its `vkDestroyBuffer` and `vkFreeMemory` take too, and
//!   wired its 1 GiB of host memory whatever the GPU wrote of it.
//! - **Raw.** A block is a `VkBuffer` bound to private memory, outside the Metal address space,
//!   not a `Buffer`, which would hold the device from the device's own blocks (as the mipmap
//!   scratch's, `mipmap_scratch.rs`). The device releases the blocks left as it is destroyed. A
//!   block released as a lease ends goes on the thread that ends it, a queue's waiter as its
//!   command buffer completes, so the release reaches the device inside a section of the exit
//!   guard (`Block::release_in_section`, `exit_guard.rs`); once the process is exiting the block is
//!   left to its end.
//! - **Not cleared.** Nothing reads a pool's bytes before a pass of the same command buffer wrote
//!   them:
//!   the index pool's counter lies in the command buffer's upload arena (zero as it is uploaded),
//!   and its block holds only the indices the fills and writes place (the block's first
//!   `IndexBoundsPass::POOL_HEADER_BYTES` are never read); the records pool's cursor is a status
//!   word.
//! - **What still fails.** A draw whose indices or records the rest of its command buffer's pool
//!   cannot hold draws nothing and fails the command buffer (`IndexPoolOverflow`,
//!   `MeshRecordsOverflow`): the GPU learns the demand after the draw was recorded against the
//!   block, and no block can be made between. `Exactness::ExactWithinPool` says so.

use super::device::lock;
use super::error::Error;
use super::memory::Memory;
use super::mipmap_scratch::Block;
use super::upload::Upload;
use super::Device;
use ash::vk;
use metal2vulkan::bind::IndexBoundsPass;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The share of the largest device-local heap a pool takes: one part in this many.
const POOL_SHARE: u64 = 64;

/// How long a queue's block stays without a lease before the end of another lease of its kind
/// releases it (the module docs).
pub(super) const QUIET: Duration = Duration::from_secs(2);

/// The bytes of a pool of a device with `properties`' heaps, whose storage bindings reach
/// `max_range` bytes and whose allocations `max_allocation`, behind `header` bytes the block
/// holds before them: a sixty-fourth of the largest device-local heap, within one binding and
/// one allocation of the block, in whole words (the module docs).
pub(crate) fn pool_budget(
    properties: &vk::PhysicalDeviceMemoryProperties,
    max_range: u32,
    max_allocation: u64,
    header: u64,
) -> u64 {
    device_share(properties)
        .min(u64::from(max_range).saturating_sub(header))
        .min(max_allocation.saturating_sub(header))
        / 4
        * 4
}

/// A sixty-fourth of the largest device-local heap of a device with `properties`' heaps: a pool's
/// bytes before its bindings' and allocation's bounds ([`pool_budget`]), and the bytes of images a
/// command queue keeps for its heaps' multisample textures' moves (`texture/copy_images.rs`).
pub(crate) fn device_share(properties: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    let count = (properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    let heap = properties.memory_heaps[..count]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    heap / POOL_SHARE
}

/// Which of a command buffer's pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PoolKind {
    /// The index pool: indices behind `IndexBoundsPass::POOL_HEADER_BYTES` the block does not
    /// use, a storage buffer and an index buffer.
    Index,
    /// The mesh records pool: records from the block's start, a storage buffer.
    MeshRecords,
}

impl PoolKind {
    /// The usage of the kind's blocks.
    fn usage(self) -> vk::BufferUsageFlags {
        match self {
            PoolKind::Index => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER
            }
            PoolKind::MeshRecords => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }

    /// Whether the kind's blocks have a device address where the device has device addresses:
    /// the index pool's, whose lists the private list form reads at their address
    /// (`index_rewrite.rs`). It costs the block's memory a buffer of addresses, made once per
    /// memory block (`memory::allocate_addressed`), and nothing per lease.
    fn addressed(self) -> bool {
        matches!(self, PoolKind::Index)
    }
}

/// One queue's block of a kind and the leases holding it.
#[derive(Debug)]
struct Slot<T> {
    block: Option<T>,
    /// The leases holding the block now.
    leases: usize,
    /// When the last lease ended, where none holds the block.
    idle_since: Instant,
}

/// The blocks of one kind, one for each queue that leased one, the decisions pure so that the
/// CPU tests cover them (the module docs).
#[derive(Debug)]
pub(super) struct QueueBlocks<T> {
    /// By the queue's identity (`CommandQueue::id`).
    slots: BTreeMap<u64, Slot<T>>,
}

impl<T> Default for QueueBlocks<T> {
    fn default() -> QueueBlocks<T> {
        QueueBlocks {
            slots: BTreeMap::new(),
        }
    }
}

impl<T> QueueBlocks<T> {
    /// A lease of `queue`'s block: what `raw` reads of it, and the leases holding it now, this
    /// one's included; `None` where the queue holds none, and the lease makes one ([`made`]).
    ///
    /// [`made`]: Self::made
    pub(super) fn take<R>(&mut self, queue: u64, raw: impl FnOnce(&T) -> R) -> Option<(R, usize)> {
        let slot = self.slots.get_mut(&queue)?;
        let block = slot.block.as_ref()?;
        slot.leases += 1;
        Some((raw(block), slot.leases))
    }

    /// `block`, made for a lease of `queue`, which held none: what `raw` reads of the queue's
    /// block, the leases holding it now, and the block to release, `block` itself where another
    /// lease of the queue made one meanwhile.
    pub(super) fn made<R>(
        &mut self,
        queue: u64,
        block: T,
        now: Instant,
        raw: impl FnOnce(&T) -> R,
    ) -> (R, usize, Option<T>) {
        let slot = self.slots.entry(queue).or_insert(Slot {
            block: None,
            leases: 0,
            idle_since: now,
        });
        let (read, extra) = match &slot.block {
            Some(held) => (raw(held), Some(block)),
            None => (raw(slot.block.insert(block)), None),
        };
        slot.leases += 1;
        (read, slot.leases, extra)
    }

    /// A lease of `queue`'s block ended at `now`: the blocks to release, each queue's that no
    /// lease held for [`QUIET`] or longer.
    pub(super) fn give(&mut self, queue: u64, now: Instant) -> Vec<T> {
        if let Some(slot) = self.slots.get_mut(&queue) {
            slot.leases = slot.leases.saturating_sub(1);
            if slot.leases == 0 {
                slot.idle_since = now;
            }
        }
        let quiet: Vec<u64> = (self.slots.iter())
            .filter(|(_, slot)| {
                slot.leases == 0 && now.saturating_duration_since(slot.idle_since) >= QUIET
            })
            .map(|(&queue, _)| queue)
            .collect();
        quiet
            .into_iter()
            .filter_map(|queue| self.slots.remove(&queue).and_then(|slot| slot.block))
            .collect()
    }

    /// `queue` went, so no lease holds its block: the block, to release.
    pub(super) fn forget(&mut self, queue: u64) -> Option<T> {
        self.slots.remove(&queue).and_then(|slot| slot.block)
    }

    /// Every block, as the device is destroyed.
    pub(super) fn drain(&mut self) -> Vec<T> {
        let slots = std::mem::take(&mut self.slots);
        slots.into_values().filter_map(|slot| slot.block).collect()
    }

    /// The blocks held.
    #[cfg(test)]
    pub(super) fn blocks(&self) -> usize {
        self.slots
            .values()
            .filter(|slot| slot.block.is_some())
            .count()
    }
}

/// A device's pools: each kind's bytes and its queues' blocks (the module docs).
#[derive(Default)]
pub(super) struct GpuPools {
    /// The index pool's bytes of indices, behind its header.
    pub(super) index_bytes: u64,
    /// The mesh records pool's bytes.
    pub(super) mesh_bytes: u64,
    index: std::sync::Mutex<QueueBlocks<Block>>,
    mesh: std::sync::Mutex<QueueBlocks<Block>>,
}

impl GpuPools {
    /// The pools of a device whose kinds hold `index_bytes` of indices and `mesh_bytes` of
    /// records.
    pub(super) fn new(index_bytes: u64, mesh_bytes: u64) -> GpuPools {
        GpuPools {
            index_bytes,
            mesh_bytes,
            ..GpuPools::default()
        }
    }

    fn blocks(&self, kind: PoolKind) -> &std::sync::Mutex<QueueBlocks<Block>> {
        match kind {
            PoolKind::Index => &self.index,
            PoolKind::MeshRecords => &self.mesh,
        }
    }

    /// The bytes of a block of `kind`: the index pool's header and indices, or the records.
    pub(super) fn block_bytes(&self, kind: PoolKind) -> u64 {
        match kind {
            PoolKind::Index => IndexBoundsPass::POOL_HEADER_BYTES + self.index_bytes,
            PoolKind::MeshRecords => self.mesh_bytes,
        }
    }

    /// Release the blocks as the device is destroyed.
    pub(super) fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        for kind in [PoolKind::Index, PoolKind::MeshRecords] {
            for block in lock(self.blocks(kind)).drain() {
                // SAFETY: the device is idle, so no command uses the block.
                unsafe { block.release(device, memory) };
            }
        }
    }
}

/// A command buffer's lease of its queue's block of a pool, which ends as it drops (the module
/// docs).
pub(super) struct PoolLease {
    device: Device,
    kind: PoolKind,
    queue: u64,
    raw: vk::Buffer,
    address: Option<u64>,
    bytes: u64,
}

impl PoolLease {
    /// The leased block's buffer.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// The device address of the leased block's byte 0, where its kind's blocks have one
    /// ([`PoolKind::addressed`]) on a device with device addresses.
    pub(super) fn address(&self) -> Option<u64> {
        self.address
    }

    /// The block's bytes.
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl std::fmt::Debug for PoolLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolLease")
            .field("kind", &self.kind)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        let shared = self.device.shared();
        let released = lock(shared.gpu_pools.blocks(self.kind)).give(self.queue, Instant::now());
        for block in released {
            // SAFETY: a lease ends when its command buffer completes or is dropped unsubmitted,
            // and no lease holds a block released, so no command uses it. It ends on the queue's
            // waiter that finished the command buffer, a runtime thread: the release reaches the
            // device inside a section of the exit guard.
            if unsafe { block.release_in_section(shared) } {
                shared.counts.gpu_pool_block_released();
            }
        }
    }
}

/// A command buffer's index pool as its passes bind it (`State::index_pool`).
#[derive(Clone, Debug)]
pub(super) struct IndexPool {
    /// The counter: [`IndexBoundsPass::POOL_HEADER_BYTES`] of the command buffer's arena, zero
    /// before its first pass; the binding of the pool an index bounds plan and an index rewrite's
    /// place dispatch read and write.
    pub(super) counter: Upload,
    /// The block's `VkBuffer`: from its byte 0, the binding of the pool an index bounds fill and
    /// an index rewrite's write dispatch write, whose indices lie from
    /// [`IndexBoundsPass::POOL_HEADER_BYTES`] on, where the draws bind it as their index buffer.
    pub(super) block: vk::Buffer,
    /// The block's bytes, its unused header's included.
    pub(super) bytes: u64,
    /// The device address of the block's byte 0, where the device has device addresses: a list
    /// in the private list form is read there (`index_rewrite.rs`).
    pub(super) address: Option<u64>,
}

impl IndexPool {
    /// The pool of `counter` and the leased `block`.
    pub(super) fn of(counter: &Upload, block: &PoolLease) -> IndexPool {
        IndexPool {
            counter: counter.clone(),
            block: block.raw(),
            bytes: block.bytes(),
            address: block.address(),
        }
    }

    /// The bytes of its indices: the block's past its header.
    pub(super) fn index_bytes(&self) -> u64 {
        self.bytes - IndexBoundsPass::POOL_HEADER_BYTES
    }
}

impl Device {
    /// A lease of `queue`'s block of `kind` for a command buffer of that queue (its
    /// `CommandQueue::id`), which the queue's first lease makes (the module docs). Refused as
    /// memory is.
    pub(super) fn lease_pool(&self, kind: PoolKind, queue: u64) -> Result<PoolLease, Error> {
        let shared = self.shared();
        let bytes = shared.gpu_pools.block_bytes(kind);
        let read = |block: &Block| (block.raw(), block.address());
        let taken = lock(shared.gpu_pools.blocks(kind)).take(queue, read);
        let ((raw, address), leases) = match taken {
            Some(taken) => taken,
            None => {
                // Made without the lock: a lease of another queue, or one ending, waits for no
                // making.
                let block = self.make_block(bytes, kind.usage(), kind.addressed())?;
                shared.counts.gpu_pool_block_made();
                let (raw, leases, extra) =
                    lock(shared.gpu_pools.blocks(kind)).made(queue, block, Instant::now(), read);
                if let Some(extra) = extra {
                    // SAFETY: made just now, the block was never bound to a command.
                    if unsafe { extra.release_in_section(shared) } {
                        shared.counts.gpu_pool_block_released();
                    }
                }
                (raw, leases)
            }
        };
        shared.counts.gpu_pool_leased(leases as u64);
        Ok(PoolLease {
            device: self.clone(),
            kind,
            queue,
            raw,
            address,
            bytes,
        })
    }

    /// `queue` went: no lease holds its blocks, which are released (the module docs).
    pub(super) fn forget_queue_pools(&self, queue: u64) {
        let shared = self.shared();
        for kind in [PoolKind::Index, PoolKind::MeshRecords] {
            let block = lock(shared.gpu_pools.blocks(kind)).forget(queue);
            if let Some(block) = block {
                // SAFETY: every command buffer of the queue holds it, so none is left, and no
                // command uses the block.
                if unsafe { block.release_in_section(shared) } {
                    shared.counts.gpu_pool_block_released();
                }
            }
        }
    }
}
