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
//! The device's sets live in pools of [`POOL_SETS`], each allocated as its pool is made and kept
//! with it ([`SetPools`]). Command buffers share the pool the device hands sets out from: each
//! takes its sets in runs, the first of one set and each next as long as all it took before, at
//! most [`RUN_SETS`], so a command buffer holds at most as many sets unused as it wrote, and at
//! most `RUN_SETS - 1` ([`Sets`]); it gives every run back as it retires, or as it drops
//! uncommitted. A pool whose every set was handed out gives way to a free one, or to a new one; it
//! is free again once every set taken from it is given back, and the device keeps at most
//! [`FREE_POOLS`] free besides the one it hands out from, destroying the rest. So a device with no
//! pending command buffer holds at most `FREE_POOLS + 1` pools, `(FREE_POOLS + 1) * POOL_SETS`
//! sets of [`KERNEL_BUFFER_BINDINGS`] storage-buffer descriptors each, and one with pending ones
//! also the pools their sets are of. A set is written only while no pending work names it, by the
//! one command buffer that holds it.
//!
//! The device's one lock over its pools ([`SetPools`]) orders what Vulkan synchronizes on a pool:
//! a pool's making and its sets' allocation, a run's take and give-back, and a pool's destruction,
//! once it has left the lock's lists. A set's write (`vkUpdateDescriptorSets`, synchronized on its
//! set alone) is its command buffer's, recording on one thread.

use crate::counters::{Counters, Gauge};
use crate::device::{lock, Core};
use crate::error::Error;
use ash::vk;
use metal2vulkan::device::KERNEL_BUFFER_BINDINGS;
use std::sync::{Arc, Mutex, PoisonError};

/// The sets of a pool.
pub(crate) const POOL_SETS: u32 = 256;

/// The most pools the device keeps free besides the one it hands sets out from: a pool given
/// back past them is destroyed.
pub(crate) const FREE_POOLS: usize = 16;

/// The most sets a command buffer takes at once.
pub(crate) const RUN_SETS: u32 = 32;

/// One binding's descriptor: the `VkBuffer`, the offset of the descriptor's first byte in it and
/// its range; a null descriptor is the null handle, 0 and `VK_WHOLE_SIZE`.
pub(crate) type Descriptor = (vk::Buffer, u64, u64);

/// The null descriptor of a binding nothing is bound to.
pub(crate) const NULL: Descriptor = (vk::Buffer::null(), 0, vk::WHOLE_SIZE);

/// A pool, its sets, each of set 0's layout, and how many of them are handed out.
struct Pool {
    pool: vk::DescriptorPool,
    sets: Arc<[vk::DescriptorSet]>,
    /// The next set handed out: every one before it was.
    next: u32,
    /// Its sets that command buffers hold: handed out and not given back.
    held: u32,
}

/// The device's set pools, behind one lock. Every pool alive is counted
/// ([`Gauge::BufferSetPools`]).
pub(crate) struct SetPools {
    pools: Mutex<Pools>,
}

#[derive(Default)]
struct Pools {
    /// The pool sets are handed out from.
    current: Option<Pool>,
    /// The pools whose every set was handed out, of which command buffers hold some.
    out: Vec<Pool>,
    /// The pools no command buffer holds a set of: at most [`FREE_POOLS`].
    free: Vec<Pool>,
}

/// Sets of one pool a command buffer took: the pool's sets, and how many of them.
struct Taken {
    sets: Arc<[vk::DescriptorSet]>,
    count: u32,
}

impl SetPools {
    pub(crate) fn new() -> SetPools {
        SetPools {
            pools: Mutex::new(Pools::default()),
        }
    }

    /// A run of at most `want` sets, from the pool the device hands out from: its sets from
    /// `.1` on, `.2` of them, at least one. Where that pool's every set is handed out, the next
    /// is a free one, else a new one.
    ///
    /// Err: [`Error::Vulkan`] where the device makes no pool.
    fn take(&self, core: &Core, want: u32) -> Result<(Arc<[vk::DescriptorSet]>, u32, u32), Error> {
        let mut pools = lock(&self.pools);
        let pools = &mut *pools;
        if pools
            .current
            .as_ref()
            .is_none_or(|pool| pool.next == POOL_SETS)
        {
            let next = match pools.free.pop() {
                Some(pool) => pool,
                None => make(core)?,
            };
            // A pool whose every set is handed out has some held: one whose last set came back
            // started again at its first.
            if let Some(full) = pools.current.replace(next) {
                pools.out.push(full);
            }
        }
        let pool = pools.current.as_mut().expect("a pool to hand out from");
        let count = want.min(POOL_SETS - pool.next);
        let first = pool.next;
        pool.next += count;
        pool.held += count;
        Ok((Arc::clone(&pool.sets), first, count))
    }

    /// Takes back what a command buffer took, once no work that names it is pending. A pool
    /// whose sets all came back starts again at its first: the one handed out from stays so,
    /// another is kept free, or destroyed where [`FREE_POOLS`] are.
    fn give_back(&self, core: &Core, taken: Vec<Taken>) {
        if taken.is_empty() {
            return;
        }
        let mut destroyed = Vec::new();
        {
            let mut pools = lock(&self.pools);
            let pools = &mut *pools;
            for Taken { sets, count } in taken {
                if let Some(pool) =
                    (pools.current.as_mut()).filter(|pool| Arc::ptr_eq(&pool.sets, &sets))
                {
                    pool.held -= count;
                    if pool.held == 0 {
                        pool.next = 0;
                    }
                    continue;
                }
                let at = (pools.out.iter())
                    .position(|pool| Arc::ptr_eq(&pool.sets, &sets))
                    .expect("a pool sets are held of is handed out from or out");
                let pool = &mut pools.out[at];
                pool.held -= count;
                if pool.held > 0 {
                    continue;
                }
                let mut pool = pools.out.swap_remove(at);
                pool.next = 0;
                match pools.free.len() < FREE_POOLS {
                    true => pools.free.push(pool),
                    false => destroyed.push(pool.pool),
                }
            }
        }
        for pool in destroyed {
            // SAFETY: no pending work names the pool or its sets, and no list holds it; destroying
            // it frees them.
            unsafe { core.device.destroy_descriptor_pool(pool, None) };
            core.counters.fall(Gauge::BufferSetPools);
        }
    }

    /// At the device's teardown, once it is idle: destroys every pool.
    pub(crate) fn destroy(&mut self, device: &ash::Device, counters: &Counters) {
        let pools = self.pools.get_mut().unwrap_or_else(PoisonError::into_inner);
        let all = (pools.current.take().into_iter())
            .chain(pools.out.drain(..))
            .chain(pools.free.drain(..));
        for pool in all {
            // SAFETY: the device is idle, and no call reads the pool.
            unsafe { device.destroy_descriptor_pool(pool.pool, None) };
            counters.fall(Gauge::BufferSetPools);
        }
    }
}

/// A new pool, its [`POOL_SETS`] sets allocated, none handed out.
///
/// Err: [`Error::Vulkan`] where the device makes none.
fn make(core: &Core) -> Result<Pool, Error> {
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
    // SAFETY: the pool is new, under the pools' lock, sized for its sets of set 0's layout.
    match unsafe { core.device.allocate_descriptor_sets(&info) } {
        Ok(sets) => {
            core.counters.rise(Gauge::BufferSetPools);
            Ok(Pool {
                pool,
                sets: sets.into(),
                next: 0,
                held: 0,
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

/// A command buffer's sets of set 0: what it took of each pool, in order, and the sets of its
/// last run it has not used, `next..end` of the last pool's.
#[derive(Default)]
pub(crate) struct Sets {
    taken: Vec<Taken>,
    next: u32,
    end: u32,
    /// The sets it took.
    count: u32,
}

impl Sets {
    /// The next set: of its last run, or the first of a run taken now where it used that run's
    /// every set; the run as long as all it took before, at least 1 and at most [`RUN_SETS`].
    ///
    /// Err: [`Error::Vulkan`] where the device makes no pool.
    pub(crate) fn take(&mut self, core: &Core) -> Result<vk::DescriptorSet, Error> {
        if self.next == self.end {
            let want = self.count.clamp(1, RUN_SETS);
            let (sets, first, count) = core.buffer_sets.take(core, want)?;
            (self.next, self.end) = (first, first + count);
            self.count += count;
            match self.taken.last_mut() {
                Some(last) if Arc::ptr_eq(&last.sets, &sets) => last.count += count,
                _ => self.taken.push(Taken { sets, count }),
            }
        }
        let last = self.taken.last().expect("a run was taken");
        let set = last.sets[self.next as usize];
        self.next += 1;
        Ok(set)
    }

    /// As its command buffer retires, or drops uncommitted: gives every set back to the device.
    pub(crate) fn give_back(self, core: &Core) {
        core.buffer_sets.give_back(core, self.taken);
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
