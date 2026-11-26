//! A kernel's buffer arguments as storage buffers of set 0, on a device whose translator profile
//! binds them (`DeviceProfile::binds_kernel_buffers`): the twelfth form a query chooses.
//!
//! There set 0's layout is [`KERNEL_BUFFER_BINDINGS`] storage buffers at the compute stage
//! (`crate::arrays::buffer_bindings`), and every kernel's pipeline is made with robust buffer
//! access 2 for its storage buffers (`crate::pipeline::kernel_robustness`). A kernel the
//! translator gave set-0 bindings (`Interface::views`: its views in binding order, each of a
//! buffer slot) reads each argument through its binding, and each dispatch gives each binding its
//! slot's descriptor ([`Descriptor`]):
//!
//! - a buffer: the `VkBuffer` shaders reach it through, from its first byte there rounded down to
//!   the buffer base alignment (`DeviceProfile::buffer_base_alignment`,
//!   `max(minStorageBufferOffsetAlignment, 16)`) through its last byte; the bind offset's distance
//!   from that start is the slot's pushed word (`FieldValue::BindOffset`). A read past the
//!   buffer's last byte reads zero and a write there is dropped, as the pipeline's robustness
//!   gives; one before its first byte, by fewer bytes than the alignment, reaches the bytes there
//!   in the same `VkBuffer` (`docs/API.md`'s accepted differences);
//! - a `set_bytes` copy: its bytes in the ring, from its own first byte, which the ring keeps at
//!   that alignment (`crate::ring`);
//! - nothing bound: a null descriptor (the floor's `nullDescriptor`), which reads zero and drops
//!   every write.
//!
//! The descriptors of a buffer depend on the buffer alone, so a changed bind offset changes only
//! the pushed word. Where a dispatch's descriptors begin the set the command buffer bound last in
//! its current Vulkan command buffer, nothing is written or bound
//! ([`Metrics::buffer_sets_held`](crate::Metrics::buffer_sets_held)); where they begin the set it
//! wrote last, which a new Vulkan command buffer of the same command buffer has not bound, that set
//! is bound again, also held; else the command buffer takes the next set of its own, writes it and
//! binds it ([`Metrics::buffer_sets_written`](crate::Metrics::buffer_sets_written)). Each bound
//! dispatch counts in [`Metrics::bound_dispatches`](crate::Metrics::bound_dispatches).
//!
//! A command buffer takes its sets in order from pools of its own ([`Sets`]), [`POOL_SETS`] a
//! pool, which the device lends it ([`SetPools`]); it gives them back as it retires, or as it
//! drops uncommitted. A pool's sets are allocated as it is made and live with it, so a take is no
//! allocation, and a set is written only while no pending work names it: its pool is a command
//! buffer's until that command buffer retired.

use crate::counters::{Counters, Gauge};
use crate::device::{lock, Core};
use crate::error::Error;
use ash::vk;
use metal2vulkan::device::KERNEL_BUFFER_BINDINGS;
use std::sync::{Mutex, PoisonError};

/// The sets of a pool.
pub(crate) const POOL_SETS: u32 = 256;

/// The most pools the device keeps free: a pool given back past them is destroyed.
pub(crate) const FREE_POOLS: usize = 16;

/// One binding's descriptor: the `VkBuffer`, the offset of the descriptor's first byte in it and
/// its range; a null descriptor is the null handle, 0 and `VK_WHOLE_SIZE`.
pub(crate) type Descriptor = (vk::Buffer, u64, u64);

/// The null descriptor of a binding nothing is bound to.
pub(crate) const NULL: Descriptor = (vk::Buffer::null(), 0, vk::WHOLE_SIZE);

/// A pool and its sets, each of set 0's layout.
struct Pool {
    pool: vk::DescriptorPool,
    sets: Box<[vk::DescriptorSet]>,
}

/// The device's set pools that no command buffer holds: at most [`FREE_POOLS`]. Every pool alive
/// is counted ([`Gauge::BufferSetPools`]).
pub(crate) struct SetPools {
    free: Mutex<Vec<Pool>>,
}

impl SetPools {
    pub(crate) fn new() -> SetPools {
        SetPools {
            free: Mutex::new(Vec::new()),
        }
    }

    /// A pool: a free one, else a new one, its [`POOL_SETS`] sets allocated.
    ///
    /// Err: [`Error::Vulkan`] where the device makes none.
    fn take(&self, core: &Core) -> Result<Pool, Error> {
        if let Some(pool) = lock(&self.free).pop() {
            return Ok(pool);
        }
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: POOL_SETS * KERNEL_BUFFER_BINDINGS,
        }];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(POOL_SETS)
            .pool_sizes(&sizes);
        // SAFETY: the info is complete.
        let pool = unsafe { core.device.create_descriptor_pool(&info, None) }
            .map_err(Error::vulkan("vkCreateDescriptorPool"))?;
        let layouts = [core.arrays.buffer_layout(); POOL_SETS as usize];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool is the device's, sized for its sets of set 0's layout.
        match unsafe { core.device.allocate_descriptor_sets(&info) } {
            Ok(sets) => {
                core.counters.rise(Gauge::BufferSetPools);
                Ok(Pool {
                    pool,
                    sets: sets.into_boxed_slice(),
                })
            }
            Err(result) => {
                // SAFETY: nothing uses the pool.
                unsafe { core.device.destroy_descriptor_pool(pool, None) };
                Err(Error::Vulkan {
                    call: "vkAllocateDescriptorSets",
                    result,
                })
            }
        }
    }

    /// Takes back `pool` once no work that names it is pending: kept free, or destroyed where
    /// [`FREE_POOLS`] are.
    fn give_back(&self, core: &Core, pool: Pool) {
        let mut free = lock(&self.free);
        if free.len() < FREE_POOLS {
            free.push(pool);
            return;
        }
        drop(free);
        // SAFETY: no pending work names the pool or its sets; destroying it frees them.
        unsafe { core.device.destroy_descriptor_pool(pool.pool, None) };
        core.counters.fall(Gauge::BufferSetPools);
    }

    /// At the device's teardown, once it is idle: destroys every free pool.
    pub(crate) fn destroy(&mut self, device: &ash::Device, counters: &Counters) {
        let free = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for pool in free.drain(..) {
            // SAFETY: the device is idle, and no call reads the pool.
            unsafe { device.destroy_descriptor_pool(pool.pool, None) };
            counters.fall(Gauge::BufferSetPools);
        }
    }
}

/// A command buffer's sets of set 0: the pools it took, in order, and how many of the last one's
/// sets it took.
#[derive(Default)]
pub(crate) struct Sets {
    pools: Vec<Pool>,
    next: u32,
}

impl Sets {
    /// The next set: of the last pool taken, or the first of a pool taken now where every one of
    /// that pool's is taken.
    ///
    /// Err: [`Error::Vulkan`] where the device makes no pool.
    pub(crate) fn take(&mut self, core: &Core) -> Result<vk::DescriptorSet, Error> {
        if self.pools.is_empty() || self.next == POOL_SETS {
            self.pools.push(core.buffer_sets.take(core)?);
            self.next = 0;
        }
        let pool = self.pools.last().expect("a pool was taken");
        let set = pool.sets[self.next as usize];
        self.next += 1;
        Ok(set)
    }

    /// As its command buffer retires, or drops uncommitted: gives every pool back to the device.
    pub(crate) fn give_back(self, core: &Core) {
        for pool in self.pools {
            core.buffer_sets.give_back(core, pool);
        }
    }
}

/// What a command buffer bound of set 0 in its current Vulkan command buffer: the set it wrote
/// last and its descriptors, and whether that Vulkan command buffer has it bound.
#[derive(Default)]
pub(crate) struct Held {
    set: Option<vk::DescriptorSet>,
    descriptors: Vec<Descriptor>,
    bound: bool,
}

impl Held {
    /// A new Vulkan command buffer of the same command buffer binds nothing yet.
    pub(crate) fn unbound(&mut self) {
        self.bound = false;
    }

    /// Binds a set whose first bindings hold `descriptors` into `command` at set 0 of the
    /// device's layout: the held set where its descriptors begin with them (bound again where
    /// `command` does not hold it), else the next of `sets`, written. Whether a set was written.
    ///
    /// # Safety
    ///
    /// `command` is the current Vulkan command buffer of the command buffer whose sets `sets` are,
    /// recording; every buffer a descriptor names lives until that command buffer retires.
    pub(crate) unsafe fn bind(
        &mut self,
        core: &Core,
        (command, sets): (vk::CommandBuffer, &mut Sets),
        descriptors: &[Descriptor],
    ) -> Result<bool, Error> {
        let held = self
            .set
            .filter(|_| self.descriptors.starts_with(descriptors));
        let (set, written) = match held {
            Some(_) if self.bound => return Ok(false),
            Some(set) => (set, false),
            None => {
                let set = sets.take(core)?;
                let mut infos =
                    [vk::DescriptorBufferInfo::default(); KERNEL_BUFFER_BINDINGS as usize];
                for (info, &(buffer, offset, range)) in infos.iter_mut().zip(descriptors) {
                    *info = vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range);
                }
                // Its bindings are alike, so one write holds consecutive bindings from 0; a
                // kernel has at most one binding a slot.
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[..descriptors.len()]);
                // SAFETY: the set is the command buffer's, which no pending work names: its pool
                // came back from a retired command buffer or is new.
                unsafe { core.device.update_descriptor_sets(&[write], &[]) };
                self.set = Some(set);
                self.descriptors.clear();
                self.descriptors.extend_from_slice(descriptors);
                (set, true)
            }
        };
        // SAFETY: by the caller; the set is of set 0's layout, the device layout's.
        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                core.layout,
                0,
                &[set],
                &[],
            );
        }
        self.bound = true;
        Ok(written)
    }
}
