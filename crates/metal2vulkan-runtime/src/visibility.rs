//! Visibility results (`setVisibilityResultMode:offset:`): the occlusion queries a render encoder
//! counts its draws in, the copies of their results into the pass's visibility result buffer,
//! and the pools the queries come from.
//!
//! Each `Boolean` or `Counting` mode set starts a segment, which the next mode set or the
//! encoder's end ends. A segment counts its draws in one precise occlusion query
//! (`VK_QUERY_CONTROL_PRECISE_BIT`, the floor's `occlusionQueryPrecise`), begun in the pass's
//! rendering before the segment's first draw and ended there as the segment ends; a segment with
//! no draw begins and ends its query as it ends, which counts no sample, so its offset is
//! written 0, as Metal writes it. After the pass's rendering ends, `vkCmdCopyQueryPoolResults`
//! writes each segment's 64-bit count, once the query is available
//! (`VK_QUERY_RESULT_WAIT_BIT`), into the visibility result buffer at its offset: one copy for
//! each run of segments whose offsets follow each other by 8 bytes and whose queries follow each
//! other in one pool. The pass's second barrier, recorded or owed, reaches every later command,
//! and the segment's end barrier the host where the buffer is Shared. An offset no mode of the
//! encoder took keeps its bytes.
//!
//! A command buffer takes its queries in order from pools of its own ([`Queries`]),
//! [`POOL_QUERIES`] a pool, which the device lends it ([`OcclusionPools`]); it gives them back as
//! it retires, or as it drops uncommitted, each query it took reset on the host first
//! (`hostQueryReset`, the floor's), so no command the GPU runs resets a query, and no work that
//! names a pool is pending while it is reset.

use crate::counters::{Counters, Gauge};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::times;
use ash::vk;
use std::collections::HashSet;
use std::sync::{Mutex, PoisonError};

/// The queries of an occlusion query pool.
pub(crate) const POOL_QUERIES: u32 = 256;

/// The most pools the device keeps free: a pool given back past them is destroyed.
pub(crate) const FREE_POOLS: usize = 16;

/// An occlusion query: its pool and its index there.
#[derive(Clone, Copy)]
pub(crate) struct Query {
    pool: vk::QueryPool,
    index: u32,
}

/// The device's occlusion query pools that no command buffer holds: at most [`FREE_POOLS`], every
/// query of each reset. Every pool alive is counted ([`Gauge::OcclusionPools`]).
pub(crate) struct OcclusionPools {
    free: Mutex<Vec<vk::QueryPool>>,
}

impl OcclusionPools {
    pub(crate) fn new() -> OcclusionPools {
        OcclusionPools {
            free: Mutex::new(Vec::new()),
        }
    }

    /// A pool, every query reset: a free one, else a new one.
    ///
    /// Err: [`Error::Vulkan`] where the device makes none.
    fn take(&self, core: &Core) -> Result<vk::QueryPool, Error> {
        if let Some(pool) = lock(&self.free).pop() {
            return Ok(pool);
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(POOL_QUERIES);
        // SAFETY: the info is complete.
        let pool = unsafe { core.device.create_query_pool(&info, None) }
            .map_err(Error::vulkan("vkCreateQueryPool"))?;
        times::reset(&core.device, pool, 0, POOL_QUERIES);
        core.counters.rise(Gauge::OcclusionPools);
        Ok(pool)
    }

    /// Takes back `pool`, whose first `used` queries a command buffer took, once no work that
    /// names it is pending: those queries reset on the host, and the pool kept free, or destroyed
    /// where [`FREE_POOLS`] are.
    fn give_back(&self, core: &Core, pool: vk::QueryPool, used: u32) {
        times::reset(&core.device, pool, 0, used);
        let mut free = lock(&self.free);
        if free.len() < FREE_POOLS {
            free.push(pool);
            return;
        }
        drop(free);
        // SAFETY: no pending work names the pool, and no call reads it.
        unsafe { core.device.destroy_query_pool(pool, None) };
        core.counters.fall(Gauge::OcclusionPools);
    }

    /// At the device's teardown, once it is idle: destroys every free pool.
    pub(crate) fn destroy(&mut self, device: &ash::Device, counters: &Counters) {
        let free = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for pool in free.drain(..) {
            // SAFETY: the device is idle, and no call reads the pool.
            unsafe { device.destroy_query_pool(pool, None) };
            counters.fall(Gauge::OcclusionPools);
        }
    }
}

/// A command buffer's occlusion queries: the pools it took, in order, and how many of the last
/// one's queries it took.
#[derive(Default)]
pub(crate) struct Queries {
    pools: Vec<vk::QueryPool>,
    next: u32,
}

impl Queries {
    /// The next query: of the last pool taken, or the first of a pool taken now where every one
    /// of that pool's is taken.
    ///
    /// Err: [`Error::Vulkan`] where the device makes no pool.
    pub(crate) fn take(&mut self, core: &Core) -> Result<Query, Error> {
        let pool = match self.pools.last() {
            Some(&pool) if self.next < POOL_QUERIES => pool,
            _ => {
                let pool = core.occlusion.take(core)?;
                self.pools.push(pool);
                self.next = 0;
                pool
            }
        };
        self.next += 1;
        Ok(Query {
            pool,
            index: self.next - 1,
        })
    }

    /// As its command buffer retires, or drops uncommitted: gives every pool back to the device,
    /// each query it took reset.
    pub(crate) fn give_back(self, core: &Core) {
        let last = self.pools.len().saturating_sub(1);
        for (at, pool) in self.pools.into_iter().enumerate() {
            let used = match at == last {
                true => self.next,
                false => POOL_QUERIES,
            };
            core.occlusion.give_back(core, pool, used);
        }
    }
}

/// A render encoder's visibility results: the segment of the mode set last, and the segments
/// ended, each an offset and its query.
#[derive(Default)]
pub(crate) struct Results {
    /// The `Boolean` or `Counting` mode set last, where no mode set or end of the encoder ended
    /// it: its offset, its query, and whether a draw began the query.
    segment: Option<(u64, Query, bool)>,
    /// The segments ended, in order.
    ended: Vec<(u64, Query)>,
    /// The offsets the encoder's modes took.
    taken: HashSet<u64>,
}

impl Results {
    /// Whether a mode of the encoder took `offset`.
    pub(crate) fn taken(&self, offset: u64) -> bool {
        self.taken.contains(&offset)
    }

    /// Whether no mode of the encoder took an offset.
    pub(crate) fn is_empty(&self) -> bool {
        self.taken.is_empty()
    }

    /// Whether a query counts the draws: begun in the rendering, it ends there, so nothing may
    /// end the rendering before the segment does.
    pub(crate) fn counting(&self) -> bool {
        matches!(self.segment, Some((_, _, true)))
    }

    /// A segment of `query` at `offset` starts, the one before it ended ([`end`](Self::end)).
    pub(crate) fn start(&mut self, offset: u64, query: Query) {
        self.taken.insert(offset);
        self.segment = Some((offset, query, false));
    }

    /// Before a draw: begins the segment's query where no draw of it began it.
    ///
    /// # Safety
    ///
    /// `command` is inside the pass's rendering, and keeps the query's pool until it retires.
    pub(crate) unsafe fn draw(&mut self, device: &ash::Device, command: vk::CommandBuffer) {
        if let Some((_, query, begun @ false)) = &mut self.segment {
            // SAFETY: by the caller; the query was reset on the host before any submission of it,
            // and no other query is active.
            unsafe { begin(device, command, *query) };
            *begun = true;
        }
    }

    /// Ends the segment, where one is: its query, begun first where no draw began it, so it
    /// counts no sample.
    ///
    /// # Safety
    ///
    /// As [`draw`](Self::draw).
    pub(crate) unsafe fn end(&mut self, device: &ash::Device, command: vk::CommandBuffer) {
        let Some((offset, query, begun)) = self.segment.take() else {
            return;
        };
        // SAFETY: by the caller; a query no draw began is reset, and no other is active.
        unsafe {
            if !begun {
                begin(device, command, query);
            }
            device.cmd_end_query(command, query.pool, query.index);
        }
        self.ended.push((offset, query));
    }

    /// After the pass's rendering: copies each segment's result into the visibility result
    /// buffer, `buffer` from byte `at` of it, one `vkCmdCopyQueryPoolResults` for each run of
    /// segments whose offsets and queries follow each other.
    ///
    /// # Safety
    ///
    /// Every segment ended ([`end`](Self::end)); `command` records outside any rendering and
    /// keeps the buffer until it retires; `at` plus each offset is a multiple of 8 whose 8 bytes
    /// lie in `buffer`.
    pub(crate) unsafe fn copy(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        (buffer, at): (vk::Buffer, u64),
    ) {
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        let mut results = self.ended.iter().peekable();
        while let Some(&(offset, first)) = results.next() {
            let mut count = 1;
            while let Some(&&(next, query)) = results.peek() {
                let follows = query.pool == first.pool
                    && query.index == first.index + count
                    && next == offset + 8 * u64::from(count);
                if !follows {
                    break;
                }
                count += 1;
                results.next();
            }
            // SAFETY: by the caller; the queries are ended, and their run lies in their pool.
            unsafe {
                device.cmd_copy_query_pool_results(
                    command,
                    first.pool,
                    first.index,
                    count,
                    buffer,
                    at + offset,
                    8,
                    flags,
                )
            };
        }
    }
}

/// Begins `query`, precise.
///
/// # Safety
///
/// `command` is inside a rendering, and keeps the query's pool until it retires; the query is
/// reset, and no other occlusion query is active.
unsafe fn begin(device: &ash::Device, command: vk::CommandBuffer, query: Query) {
    // SAFETY: by the caller; the floor enables `occlusionQueryPrecise`.
    unsafe {
        device.cmd_begin_query(
            command,
            query.pool,
            query.index,
            vk::QueryControlFlags::PRECISE,
        )
    };
}
